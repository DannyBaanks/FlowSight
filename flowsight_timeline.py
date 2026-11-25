#!/usr/bin/env python3
"""FlowSight M5-B — Temporal UI Perception over Screenshot Sequences & Videos (`flowsight_timeline.py`).

Applies FFmpeg-style scene-change skipping + FlowSight M4 identity tracking (`K0..Kn`)
and real local OCR across an ordered sequence of screenshots OR a video file
(`.mp4`, `.mkv`, `.webm`, `.avi` via `cv2.VideoCapture`):

  frames / video -> sample & skip static frames -> layout + controls + OCR
                 -> temporal identity ui_diff (K0..Kn) -> verifiable timestamped event log

Event types emitted:
  - CONTROL_APPEARED      (DEMONSTRATED: NEW_CONTROL with bbox, color, text)
  - CONTROL_DISAPPEARED   (DEMONSTRATED: REMOVED_CONTROL)
  - CONTROL_COLOR_CHANGED (DEMONSTRATED: color transition + INFERRED STATE_CHANGED_CANDIDATE)
  - CONTROL_MOVED         (DEMONSTRATED: delta_center_px)
  - CONTROL_RESIZED       (DEMONSTRATED: bbox size change)
  - CONTROL_TEXT_CHANGED  (DEMONSTRATED: OCR text transition on same track)
  - TEXT_APPEARED         (DEMONSTRATED: new OCR text fragment with global bbox & conf)
  - TEXT_DISAPPEARED      (DEMONSTRATED: previously observed OCR text absent in new keyframe)
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Callable, Sequence

import cv2
import numpy as np

import flowsight_diff as fdiff
import flowsight_ocr as focr
import flowsight_screen as fscr

SCHEMA_TIMELINE = 'flowsight-timeline-v1'
VIDEO_EXTS = frozenset({'.mp4', '.mkv', '.webm', '.avi', '.mov'})


def _frame_delta_score(rgb_a: np.ndarray, rgb_b: np.ndarray) -> float:
    """Fast localized peak-tile RGB delta in [0..255] for FFmpeg-style static frame skipping.

    A single button changing color or a small error toast alters only ~1% of a 1080p frame,
    which global mean delta misses. Taking the max mean-abs-delta across an 8x6 tile grid
    (20x15 px each on a 160x90 downsample) stays near 0 on compressed static frames while
    spiking (>15) on any localized UI change.
    """
    sa = cv2.resize(rgb_a, (160, 90), interpolation=cv2.INTER_AREA).astype(np.int16)
    sb = cv2.resize(rgb_b, (160, 90), interpolation=cv2.INTER_AREA).astype(np.int16)
    diff = np.mean(np.abs(sa - sb), axis=2)  # (90, 160)
    tiles = diff.reshape(6, 15, 8, 20).mean(axis=(1, 3))
    return float(np.max(tiles))


def _analyze_rgb_frame(
    rgb: np.ndarray,
    *,
    source_name: str,
    use_ocr: bool,
    ocr_provider: Callable[..., dict[str, Any]] | None = None,
) -> fscr.FilterContext:
    h, w = int(rgb.shape[0]), int(rgb.shape[1])
    sha = hashlib.sha256(rgb.tobytes()).hexdigest()
    ctx = fscr.FilterContext(
        source_path=source_name,
        source_name=source_name,
        source_sha256=sha,
        full_w=w,
        full_h=h,
        rgb=rgb,
        ocr_provider=ocr_provider,
    )
    fscr.FILTERS['layout'].fn(ctx, {})
    fscr.FILTERS['controls'].fn(ctx, {})
    if use_ocr:
        fscr.FILTERS['ocr'].fn(ctx, {})
    return ctx


def _norm_ocr_map(ctx: fscr.FilterContext) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for item in ctx.artifacts.get('ocr', {}).get('items', []):
        norm = focr.normalize_text(str(item.get('text', '')))
        if not norm:
            continue
        prev = out.get(norm)
        if prev is None or float(item.get('confidence', 0.0)) > float(prev.get('confidence', 0.0)):
            out[norm] = item
    return out


def _diff_keyframes(
    prev_ctx: fscr.FilterContext,
    cur_ctx: fscr.FilterContext,
    *,
    prev_track_map: dict[str, str] | None,
    t_sec: float,
    frame_idx: int,
    source_name: str,
) -> tuple[list[dict[str, Any]], dict[str, str], dict[str, int]]:
    panels_a = prev_ctx.artifacts['layout']['panels']
    panels_b = cur_ctx.artifacts['layout']['panels']
    ctls_a = prev_ctx.artifacts['controls']['controls']
    ctls_b = cur_ctx.artifacts['controls']['controls']

    tracking = fdiff.match_controls_temporal(
        controls_a=ctls_a,
        controls_b=ctls_b,
        panels_a=panels_a,
        panels_b=panels_b,
        screen_size=(cur_ctx.full_w, cur_ctx.full_h),
        prev_track_map=prev_track_map,
    )

    events: list[dict[str, Any]] = []
    for e in tracking['tracks']:
        cls = e['classification']
        flags = e['flags']
        if cls == fdiff.CLS_UNCHANGED:
            continue
        if cls == fdiff.CLS_NEW:
            fb_c = e['frame_b']
            events.append({
                't_sec': round(t_sec, 3),
                'frame_idx': frame_idx,
                'source': source_name,
                'event_type': 'CONTROL_APPEARED',
                'provenance': 'DEMONSTRATED',
                'track': e['track'],
                'bbox': fb_c['bbox'],
                'color': f'{fb_c["hex"]}({fb_c["family"]})',
                'text': fb_c.get('text'),
            })
        elif cls == fdiff.CLS_REMOVED:
            fa_c = e['frame_a']
            events.append({
                't_sec': round(t_sec, 3),
                'frame_idx': frame_idx,
                'source': source_name,
                'event_type': 'CONTROL_DISAPPEARED',
                'provenance': 'DEMONSTRATED',
                'track': e['track'],
                'bbox': fa_c['bbox'],
                'color': f'{fa_c["hex"]}({fa_c["family"]})',
                'text': fa_c.get('text'),
            })
        elif cls == fdiff.CLS_AMBIGUOUS:
            events.append({
                't_sec': round(t_sec, 3),
                'frame_idx': frame_idx,
                'source': source_name,
                'event_type': 'CONTROL_AMBIGUOUS_MATCH',
                'provenance': 'UNCERTAIN',
                'track': e['track'],
                'bbox': e['frame_b']['bbox'],
                'detail': f'score={e["affinity"]["score"]}',
            })
        else:
            if fdiff.CLS_COLOR_CHANGED in flags:
                events.append({
                    't_sec': round(t_sec, 3),
                    'frame_idx': frame_idx,
                    'source': source_name,
                    'event_type': 'CONTROL_COLOR_CHANGED',
                    'provenance': 'DEMONSTRATED',
                    'inferred_flags': e.get('inferred_flags', []),
                    'track': e['track'],
                    'bbox': e['frame_b']['bbox'],
                    'transition': e.get('color_transition'),
                    'text': e['frame_b'].get('text'),
                })
            if fdiff.CLS_TEXT_CHANGED in flags:
                events.append({
                    't_sec': round(t_sec, 3),
                    'frame_idx': frame_idx,
                    'source': source_name,
                    'event_type': 'CONTROL_TEXT_CHANGED',
                    'provenance': 'DEMONSTRATED',
                    'track': e['track'],
                    'bbox': e['frame_b']['bbox'],
                    'transition': e.get('text_transition'),
                })
            if fdiff.CLS_MOVED in flags:
                events.append({
                    't_sec': round(t_sec, 3),
                    'frame_idx': frame_idx,
                    'source': source_name,
                    'event_type': 'CONTROL_MOVED',
                    'provenance': 'DEMONSTRATED',
                    'track': e['track'],
                    'bbox': e['frame_b']['bbox'],
                    'delta_center_px': e.get('delta_center_px'),
                    'text': e['frame_b'].get('text'),
                })
            if fdiff.CLS_RESIZED in flags:
                events.append({
                    't_sec': round(t_sec, 3),
                    'frame_idx': frame_idx,
                    'source': source_name,
                    'event_type': 'CONTROL_RESIZED',
                    'provenance': 'DEMONSTRATED',
                    'track': e['track'],
                    'bbox': e['frame_b']['bbox'],
                })

    # OCR text emergence/disappearance across keyframes
    prev_ocr = _norm_ocr_map(prev_ctx)
    cur_ocr = _norm_ocr_map(cur_ctx)
    for norm, item in cur_ocr.items():
        if norm not in prev_ocr:
            events.append({
                't_sec': round(t_sec, 3),
                'frame_idx': frame_idx,
                'source': source_name,
                'event_type': 'TEXT_APPEARED',
                'provenance': 'DEMONSTRATED',
                'bbox': item['bbox'],
                'text': item['text'],
                'conf': round(float(item.get('confidence', 0.0)), 1),
            })
    for norm, item in prev_ocr.items():
        if norm not in cur_ocr:
            events.append({
                't_sec': round(t_sec, 3),
                'frame_idx': frame_idx,
                'source': source_name,
                'event_type': 'TEXT_DISAPPEARED',
                'provenance': 'DEMONSTRATED',
                'bbox': item['bbox'],
                'text': item['text'],
                'conf': round(float(item.get('confidence', 0.0)), 1),
            })

    return events, tracking['frame_b_tracks'], tracking['summary_counts']


def _scan_frame_stream(
    frames: Sequence[tuple[float, int, str, np.ndarray]],
    *,
    source_desc: str,
    scene_thresh: float = 6.0,
    use_ocr: bool = True,
    ocr_provider: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    if not frames:
        raise ValueError('no frames provided to timeline scanner')

    keyframes: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    skipped = 0

    prev_rgb: np.ndarray | None = None
    prev_ctx: fscr.FilterContext | None = None
    prev_tracks: dict[str, str] | None = None

    for t_sec, fidx, name, rgb in frames:
        if prev_rgb is not None:
            delta = _frame_delta_score(prev_rgb, rgb)
            if delta < scene_thresh:
                skipped += 1
                continue
        else:
            delta = 0.0

        ctx = _analyze_rgb_frame(rgb, source_name=name, use_ocr=use_ocr, ocr_provider=ocr_provider)
        kf_entry = {
            't_sec': round(t_sec, 3),
            'frame_idx': fidx,
            'source': name,
            'delta_from_prev': round(delta, 2),
            'controls': len(ctx.artifacts['controls']['controls']),
            'ocr_items': len(ctx.artifacts.get('ocr', {}).get('items', [])),
            'ocr_status': ctx.artifacts.get('ocr', {}).get('status', 'OCR_SKIPPED'),
        }

        if prev_ctx is not None:
            step_events, next_tracks, counts = _diff_keyframes(
                prev_ctx, ctx,
                prev_track_map=prev_tracks,
                t_sec=t_sec,
                frame_idx=fidx,
                source_name=name,
            )
            events.extend(step_events)
            prev_tracks = next_tracks
            kf_entry['diff_counts'] = counts
            kf_entry['events_emitted'] = len(step_events)
        else:
            prev_tracks = {c['id']: f'K{i}' for i, c in enumerate(ctx.artifacts['controls']['controls'])}
            kf_entry['events_emitted'] = 0

        keyframes.append(kf_entry)
        prev_rgb = rgb
        prev_ctx = ctx

    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    return {
        'schema': SCHEMA_TIMELINE,
        'source': source_desc,
        'total_sampled_frames': len(frames),
        'keyframes_analyzed': len(keyframes),
        'frames_skipped_static': skipped,
        'scene_thresh': scene_thresh,
        'use_ocr': use_ocr,
        'elapsed_ms': elapsed_ms,
        'keyframes': keyframes,
        'events': events,
    }


def scan_sequence(
    image_paths: Sequence[str | Path],
    *,
    scene_thresh: float = 6.0,
    use_ocr: bool = True,
    ocr_provider: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Scan an ordered sequence of screenshot paths for temporal UI events."""
    frames: list[tuple[float, int, str, np.ndarray]] = []
    for idx, raw in enumerate(image_paths):
        p = Path(raw).expanduser().resolve()
        rgb, _, name = fscr._load_screen_rgb(p)
        frames.append((float(idx), idx, name, rgb))
    return _scan_frame_stream(
        frames,
        source_desc=f'sequence({len(frames)} frames)',
        scene_thresh=scene_thresh,
        use_ocr=use_ocr,
        ocr_provider=ocr_provider,
    )


def scan_video(
    video_path: str | Path,
    *,
    sample_fps: float = 2.0,
    max_frames: int = 60,
    scene_thresh: float = 6.0,
    use_ocr: bool = True,
    ocr_provider: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Sample a video file with OpenCV, skip static frames, and emit verifiable temporal UI events."""
    p = Path(video_path).expanduser().resolve()
    if not p.is_file():
        raise FileNotFoundError(f'video not found: {p}')
    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        raise ValueError(f'OpenCV failed to open video: {p}')

    try:
        native_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        if native_fps <= 0.1 or native_fps > 240.0:
            native_fps = 10.0
        target_fps = max(0.2, min(native_fps, float(sample_fps)))
        step = max(1, int(round(native_fps / target_fps)))
        cap_max = max(2, min(200, int(max_frames)))

        sampled: list[tuple[float, int, str, np.ndarray]] = []
        frame_idx = 0
        while len(sampled) < cap_max:
            ok, bgr = cap.read()
            if not ok:
                break
            if frame_idx % step == 0:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                t_sec = frame_idx / native_fps
                sampled.append((t_sec, frame_idx, f'{p.name}@f{frame_idx}', rgb))
            frame_idx += 1
    finally:
        cap.release()

    res = _scan_frame_stream(
        sampled,
        source_desc=str(p.name),
        scene_thresh=scene_thresh,
        use_ocr=use_ocr,
        ocr_provider=ocr_provider,
    )
    res['video_native_fps'] = round(native_fps, 2)
    res['sample_fps'] = round(target_fps, 2)
    return res


def filter_timeline_events(timeline: dict[str, Any], query: str) -> list[dict[str, Any]]:
    """Filter timeline events by text substring or event_type token."""
    q = (query or '').strip().lower()
    if not q:
        return list(timeline.get('events', []))
    tokens = [tok for tok in q.split() if tok]
    out: list[dict[str, Any]] = []
    for ev in timeline.get('events', []):
        hay = ' '.join(
            str(ev.get(k, ''))
            for k in ('event_type', 'track', 'text', 'transition', 'color', 'source', 'inferred_flags')
        ).lower()
        if all(tok in hay for tok in tokens):
            out.append(ev)
    return out


def format_timeline_report(timeline: dict[str, Any], *, query: str = '', max_chars: int = 5500) -> str:
    ev_list = filter_timeline_events(timeline, query) if query else timeline['events']
    lines = [
        f'FLOWSIGHT TIMELINE v1 | source={timeline["source"]} | '
        f'sampled={timeline["total_sampled_frames"]} keyframes={timeline["keyframes_analyzed"]} '
        f'skipped_static={timeline["frames_skipped_static"]} | events={len(ev_list)} ({timeline["elapsed_ms"]}ms)',
        'Provenance: DEMONSTRATED frame geometry/color/OCR transitions; STATE_CHANGED_CANDIDATE is INFERRED.',
    ]
    if query:
        lines.append(f'EVENT FILTER: {query!r}')
    if not ev_list:
        lines.append('  (no temporal events matched)')
    for ev in ev_list[:40]:
        parts = [f't={ev["t_sec"]:.2f}s (f{ev["frame_idx"]}) [{ev["event_type"]}]']
        if ev.get('track'):
            parts.append(f'track={ev["track"]}')
        if ev.get('bbox'):
            parts.append(f'bbox={ev["bbox"]}')
        if ev.get('transition'):
            parts.append(f'transition={ev["transition"]}')
        if ev.get('color'):
            parts.append(f'color={ev["color"]}')
        if ev.get('text'):
            parts.append(f'text={ev["text"]!r}')
        if ev.get('inferred_flags'):
            parts.append(f'INFERRED={ev["inferred_flags"]}')
        lines.append('  ' + ' '.join(parts))
    txt = '\n'.join(lines)
    if len(txt) > max_chars:
        txt = txt[:max(0, max_chars - 36)] + '\n[TRUNCATED: filter with --query]'
    return txt


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description='FlowSight M5-B: Temporal UI scanner for videos & screenshot sequences')
    p.add_argument('inputs', nargs='+', help='One video file (.mp4/.mkv/.webm) OR 2+ ordered screenshot paths')
    p.add_argument('--fps', type=float, default=2.0, help='Video sampling FPS (default 2.0)')
    p.add_argument('--max-frames', type=int, default=60)
    p.add_argument('--scene-thresh', type=float, default=6.0, help='Static frame skip threshold (0..255)')
    p.add_argument('--no-ocr', action='store_true')
    p.add_argument('--query', default='', help='Filter output events by keyword/event_type')
    p.add_argument('--json', action='store_true')
    args = p.parse_args(argv)

    first = Path(args.inputs[0])
    if len(args.inputs) == 1 and first.suffix.lower() in VIDEO_EXTS:
        res = scan_video(
            first,
            sample_fps=args.fps,
            max_frames=args.max_frames,
            scene_thresh=args.scene_thresh,
            use_ocr=not args.no_ocr,
        )
    else:
        res = scan_sequence(
            args.inputs,
            scene_thresh=args.scene_thresh,
            use_ocr=not args.no_ocr,
        )

    if args.json:
        if args.query:
            res = dict(res, filtered_events=filter_timeline_events(res, args.query))
        print(json.dumps(res, indent=2, ensure_ascii=False))
    else:
        print(format_timeline_report(res, query=args.query))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
