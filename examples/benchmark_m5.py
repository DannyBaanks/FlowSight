#!/usr/bin/env python3
"""FlowSight M5 Benchmark & Evidence Builder (7-screenshot progressive search + real MP4 timeline).

Generates deterministic synthetic desktop screenshots + a real MP4 video with static frames
and temporal transitions, runs REAL local OCR (zero mocks), and measures:
  1. Full 7-report character cost vs Progressive Search + single winning crop zoom cost.
  2. First-pass indexing latency vs SHA-256 deduplicated cached re-index latency.
  3. Real MP4 video scanning with static-frame skipping, persistent K-tracks, color transition,
     modal appearance, and error banner emergence timestamping.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tempfile

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'examples'))

import corpus_m4
import flowsight_index as fidx
import flowsight_screen as fscr
import flowsight_timeline as ftl

OUT_DIR = REPO / 'evidence' / 'm5'


def _font(size: int = 20):
    for p in (
        '/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf',
        '/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',
    ):
        if Path(p).is_file():
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def _make_screen(bg: tuple[int, int, int], title: str, lines: list[str], buttons: list[tuple[str, tuple[int, int, int], tuple[int, int, int, int]]]) -> Image.Image:
    im = Image.new('RGB', (1280, 720), bg)
    d = ImageDraw.Draw(im)
    dark = sum(bg) / 3 < 128
    bar = (28, 32, 44) if dark else (228, 232, 240)
    fg = (235, 240, 248) if dark else (20, 24, 34)
    d.rectangle([0, 0, 1279, 54], fill=bar)
    d.line([0, 54, 1279, 54], fill=(70, 80, 100), width=2)
    d.text((28, 15), title, fill=fg, font=_font(22))
    y = 95
    f = _font(21)
    for ln in lines:
        d.text((48, y), ln, fill=fg, font=f)
        y += 38
    fb = _font(20)
    for label, col, box in buttons:
        d.rectangle(box, fill=col)
        tx = box[0] + 18
        ty = box[1] + max(6, (box[3] - box[1] - 24) // 2)
        d.text((tx, ty), label, fill=(255, 255, 255), font=fb)
    return im


def build_seven_screenshots(out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    specs = [
        (
            'shot_01_browser_docs.png',
            (244, 246, 250),
            'Browser - SQLite FTS5 Documentation',
            [
                'SQLite Full Text Search extension supports BM25 ranking.',
                'Use unicode61 tokenizer for fast phrase lookups.',
                'No firewall configuration on this page.',
            ],
            [('Read Docs', (37, 99, 235), (80, 540, 250, 588))],
        ),
        (
            'shot_02_code_editor.png',
            (18, 20, 28),
            'Editor - flowsight_index.py',
            [
                'def search_index(query: str, limit: int = 5):',
                '    expire_stale(db_path)',
                '    return ranked_candidates',
            ],
            [('Run Tests', (22, 163, 74), (80, 540, 250, 588))],
        ),
        (
            'shot_03_ufw_terminal.png',
            (14, 16, 22),
            'Terminal - UFW Firewall Status',
            [
                '$ sudo ufw status numbered',
                'Regla anadida (v6)',
                '3128/tcp on br-575398bb8361 ALLOW # artex-lab nim proxy (E4N)',
                'Status: active on port 3128',
            ],
            [('Apply Rule', (22, 163, 74), (80, 540, 270, 588))],
        ),
        (
            'shot_04_docker_logs.png',
            (16, 18, 26),
            'Terminal - Container Egress Probe',
            [
                '$ docker exec artex-1 curl https://example.org',
                'HTTP_CODE=000 TIME_TOTAL=0.002s',
                'ERROR: exit=56 connection reset by peer',
            ],
            [('Retry Probe', (220, 38, 38), (80, 540, 280, 588))],
        ),
        (
            'shot_05_settings_light.png',
            (246, 247, 251),
            'System Settings - Display and Audio',
            [
                'Resolution: 1920 x 1080 (60 Hz)',
                'Scale: 100%  Night Light: Off',
                'Output Device: Built-in Audio Analog Stereo',
            ],
            [('Save Settings', (37, 99, 235), (80, 540, 290, 588))],
        ),
        (
            'shot_06_git_pr.png',
            (18, 20, 28),
            'GitHub - Pull Request #123',
            [
                'feat(tui): integrate FlowSight blind-vision fallback',
                'Checks passed: 35/35 CI coverage green',
                'Ready for review by maintainer',
            ],
            [('Merge pull request', (22, 163, 74), (80, 540, 340, 588))],
        ),
        (
            'shot_07_chat_summary.png',
            (20, 22, 30),
            'Chat - Release Checklist',
            [
                'All unit tests passed on main branch.',
                'Remember to purge temporary indexes after audit.',
                'Next milestone: temporal video timeline.',
            ],
            [('Approve', (234, 148, 24), (80, 540, 240, 588))],
        ),
    ]
    paths: list[Path] = []
    for fname, bg, title, lines, btns in specs:
        p = out_dir / fname
        _make_screen(bg, title, lines, btns).save(p)
        paths.append(p)
    return paths


def build_synthetic_timeline_video(out_video: Path, work_dir: Path) -> Path:
    """Create a 4-second 10-FPS MP4 video with 4 phases (repeated static frames + 3 real UI events)."""
    manifest = corpus_m4.build_corpus(work_dir / 'm4_seed')
    f_base = Image.open(manifest['cases']['modal_and_color_change']['frame_a']).convert('RGB')

    # Phase 2: green Merge button turns orange (same position, same text!)
    f_orange = f_base.copy()
    d2 = ImageDraw.Draw(f_orange)
    d2.rectangle([260, 520, 509, 567], fill=(234, 148, 24))
    d2.text((285, 533), 'Merge pull request', fill=(255, 255, 255), font=_font(20))

    # Phase 3: modal dialog with 'Dismiss' button appears (from corpus_m4 frame_b)
    f_modal = Image.open(manifest['cases']['modal_and_color_change']['frame_b']).convert('RGB')

    # Phase 4: red error banner 'ERROR: exit=56' appears at top of main pane
    f_err = f_modal.copy()
    d4 = ImageDraw.Draw(f_err)
    d4.rectangle([260, 320, 680, 385], fill=(185, 28, 28))
    d4.text((285, 340), 'ERROR: exit=56', fill=(255, 255, 255), font=_font(24))

    phases = [f_base, f_orange, f_modal, f_err]
    out_video.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(str(out_video), fourcc, 10.0, (1280, 720))
    if not writer.isOpened():
        raise RuntimeError(f'failed to open VideoWriter for {out_video}')
    try:
        for img in phases:
            bgr = cv2.cvtColor(np.array(img, dtype=np.uint8), cv2.COLOR_RGB2BGR)
            for _ in range(10):  # 1.0s per phase (10 frames each -> 40 frames total)
                writer.write(bgr)
    finally:
        writer.release()
    return out_video


def run_benchmark() -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    shots = build_seven_screenshots(OUT_DIR / 'seven_shots')

    # 1. Naive baseline: 7 full FlowSight reports dumped to LLM context
    full_reports = []
    for p in shots:
        ctx = fscr.run_pipeline(p, vf='theme,layout,controls,text_bands,ocr,ui_map=44:16')
        full_reports.append(fscr.format_pipeline_report(ctx, max_chars=6500))
    naive_total_chars = sum(len(r) for r in full_reports)

    # 2. Progressive search M5-A on ephemeral private DB
    with tempfile.TemporaryDirectory(prefix='flowsight_m5_bench_') as td:
        db_path = Path(td) / 'bench_index.sqlite'
        idx_cold = fidx.index_paths(shots, use_ocr=True, ttl_hours=24.0, db_path=db_path)
        idx_warm = fidx.index_paths(shots, use_ocr=True, ttl_hours=24.0, db_path=db_path)

        search_res = fidx.search_index('UFW 3128', theme='dark', limit=3, db_path=db_path)
        search_report = fidx.format_search_report(search_res)
        (OUT_DIR / 'example_progressive_search.txt').write_text(search_report + '\n', encoding='utf-8')

        top = search_res['candidates'][0]
        zoom_vf = f'{top["suggested_crop_vf"]},controls,ocr'
        ctx_zoom = fscr.run_pipeline(top['path'], vf=zoom_vf)
        zoom_report = fscr.format_pipeline_report(ctx_zoom, max_chars=4000)
        (OUT_DIR / 'example_zoom_crop.txt').write_text(zoom_report + '\n', encoding='utf-8')

        progressive_total_chars = len(search_report) + len(zoom_report)
        char_savings_pct = round((1.0 - progressive_total_chars / max(1, naive_total_chars)) * 100.0, 2)

        # 3. Real MP4 video timeline M5-B
        vid_path = build_synthetic_timeline_video(OUT_DIR / 'demo_timeline.mp4', Path(td))
        tl_res = ftl.scan_video(vid_path, sample_fps=2.0, max_frames=40, scene_thresh=6.0, use_ocr=True)
        tl_report = ftl.format_timeline_report(tl_res)
        (OUT_DIR / 'example_video_timeline.txt').write_text(tl_report + '\n', encoding='utf-8')

    summary = {
        'seven_screenshots_progressive_search': {
            'screenshots': len(shots),
            'naive_7_full_reports_chars': naive_total_chars,
            'progressive_search_chars': len(search_report),
            'winning_crop_zoom_chars': len(zoom_report),
            'progressive_total_chars': progressive_total_chars,
            'char_savings_pct': char_savings_pct,
            'index_cold_ms': idx_cold['elapsed_ms'],
            'index_cached_dedup_ms': idx_warm['elapsed_ms'],
            'db_bytes': idx_cold['db_bytes'],
            'top1_source': top['source_name'],
            'top1_score': top['score'],
            'top1_crop_vf': top['suggested_crop_vf'],
            'top1_matched_terms': top['matched_terms'],
        },
        'video_timeline_mp4': {
            'video': vid_path.name,
            'total_sampled_frames': tl_res['total_sampled_frames'],
            'keyframes_analyzed': tl_res['keyframes_analyzed'],
            'frames_skipped_static': tl_res['frames_skipped_static'],
            'events_total': len(tl_res['events']),
            'elapsed_ms': tl_res['elapsed_ms'],
            'event_types': sorted({e['event_type'] for e in tl_res['events']}),
        },
    }
    (OUT_DIR / 'benchmark_m5.json').write_text(json.dumps(summary, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return summary


if __name__ == '__main__':
    run_benchmark()
