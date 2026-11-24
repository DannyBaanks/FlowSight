"""FlowSight M4-B — Identity-based temporal UI diff & control tracking.

Fixes the M3 index-shift bug where inserting a new control renumbered local
IDs (`W1 -> W2`) and caused misattribution of changes.

Design:
  - Position in a list is NEVER identity.
  - Global optimal assignment via `scipy.optimize.linear_sum_assignment` over a
    composite affinity combining:
      * Spatial overlap (IoU) & center proximity & size similarity
      * Normalized OCR text similarity (when OCR text is available)
      * Containing panel compatibility
      * Control geometric kind compatibility
      * Controlled RGB color weight (NEVER an absolute identity key, so a button
        changing green->orange keeps its persistent track ID!)
  - Explicit ambiguity detection (`AMBIGUOUS_MATCH`) when two visually/spatially
    competing candidates have near-tied affinity without a differentiating cue.
  - Emits persistent track IDs (`K0, K1, ...`) separate from per-frame local IDs
    (`W0, W1, ...`) and preserves both Frame A and Frame B evidence.

Required classifications:
  NEW_CONTROL, REMOVED_CONTROL, MOVED_CONTROL, RESIZED_CONTROL, TEXT_CHANGED,
  COLOR_CHANGED, STATE_CHANGED_CANDIDATE (explicitly INFERRED),
  UNCHANGED_CONTROL, AMBIGUOUS_MATCH.
"""
from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment

import flowsight_ocr as focr

CLS_NEW = 'NEW_CONTROL'
CLS_REMOVED = 'REMOVED_CONTROL'
CLS_MOVED = 'MOVED_CONTROL'
CLS_RESIZED = 'RESIZED_CONTROL'
CLS_TEXT_CHANGED = 'TEXT_CHANGED'
CLS_COLOR_CHANGED = 'COLOR_CHANGED'
CLS_STATE_CANDIDATE = 'STATE_CHANGED_CANDIDATE'
CLS_UNCHANGED = 'UNCHANGED_CONTROL'
CLS_AMBIGUOUS = 'AMBIGUOUS_MATCH'

MATCH_MIN_AFFINITY = 0.44
AMBIGUITY_MARGIN = 0.06
MOVE_THRESH_PX = 8.0
RESIZE_THRESH_PX = 8.0
RESIZE_RATIO_THRESH = 0.15
COLOR_DELTA_THRESH = 24.0


def _hex_to_rgb(hx: str) -> tuple[int, int, int]:
    h = hx.lstrip('#')
    if len(h) >= 6:
        return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return (0, 0, 0)


def _iou(a: list[int], b: list[int]) -> float:
    ix0 = max(a[0], b[0])
    iy0 = max(a[1], b[1])
    ix1 = min(a[2], b[2])
    iy1 = min(a[3], b[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0.0
    inter = (ix1 - ix0) * (iy1 - iy0)
    aa = max(1, (a[2] - a[0]) * (a[3] - a[1]))
    ab = max(1, (b[2] - b[0]) * (b[3] - b[1]))
    return inter / max(1, aa + ab - inter)


def _text_similarity(ta: str | None, tb: str | None) -> float | None:
    na = focr.normalize_text(ta or '')
    nb = focr.normalize_text(tb or '')
    if not na or not nb:
        return None
    if na == nb:
        return 1.0
    dist = focr.levenshtein(na, nb)
    return max(0.0, 1.0 - dist / max(len(na), len(nb)))


def _panel_hint_map(panels: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {p['id']: p for p in panels}


def _pair_affinity(
    ca: dict[str, Any],
    cb: dict[str, Any],
    panels_a: dict[str, dict[str, Any]],
    panels_b: dict[str, dict[str, Any]],
    diag: float,
) -> tuple[float, dict[str, float]]:
    """Compute bounded composite affinity S(a, b) in [0, 1] + component breakdown."""
    ba = ca['global_bbox']
    bb = cb['global_bbox']
    iou = _iou(ba, bb)
    ax, ay = ca['click_center']
    bx, by = cb['click_center']
    dist = math.hypot(ax - bx, ay - by)
    # Proximity decays smoothly over ~22% of screen diagonal (allows real moves)
    prox = math.exp(-dist / max(40.0, 0.22 * diag))

    wa, ha = max(1, ba[2] - ba[0]), max(1, ba[3] - ba[1])
    wb, hb = max(1, bb[2] - bb[0]), max(1, bb[3] - bb[1])
    size_sim = 0.5 * (min(wa, wb) / max(wa, wb) + min(ha, hb) / max(ha, hb))

    # Panel compatibility via position_hint or panel bbox IoU
    pa = panels_a.get(ca.get('inside_panel') or '')
    pb = panels_b.get(cb.get('inside_panel') or '')
    if pa and pb:
        if pa['position_hint'] == pb['position_hint'] or _iou(pa['global_bbox'], pb['global_bbox']) >= 0.50:
            panel_sim = 1.0
        else:
            panel_sim = 0.15
    elif pa is None and pb is None:
        panel_sim = 0.7
    else:
        panel_sim = 0.4

    # Kind compatibility
    kind_sim = 1.0 if ca.get('kind') == cb.get('kind') else 0.55

    # Controlled color similarity (weight capped so green->orange stays matched!)
    ra, ga, bca = _hex_to_rgb(ca.get('hex', '#000000'))
    rb, gb, bcb = _hex_to_rgb(cb.get('hex', '#000000'))
    color_delta = (abs(ra - rb) + abs(ga - gb) + abs(bca - bcb)) / 3.0
    color_sim = max(0.0, 1.0 - color_delta / 180.0)

    # OCR text similarity when both frames have observed/uncertain OCR text
    tsim = _text_similarity(ca.get('text'), cb.get('text'))

    # Geometry combined: strong IoU dominates when control stayed in place
    geom_sim = max(iou, 0.55 * prox + 0.45 * size_sim if dist < 0.35 * diag else 0.35 * prox)
    if iou >= 0.65:
        geom_sim = 0.70 * iou + 0.20 * size_sim + 0.10 * prox

    if tsim is not None:
        # Both controls carry OCR text: text + geometry share primary weight
        score = (
            0.38 * geom_sim
            + 0.32 * tsim
            + 0.12 * panel_sim
            + 0.10 * kind_sim
            + 0.08 * color_sim
        )
        # Hard guard: completely different text at distant non-overlapping locations shouldn't match
        if tsim < 0.25 and iou < 0.35:
            score *= 0.55
    else:
        # Geometry + size + panel + kind carry primary weight; color stays <= 0.12
        score = (
            0.46 * geom_sim
            + 0.16 * size_sim
            + 0.16 * panel_sim
            + 0.12 * kind_sim
            + 0.10 * color_sim
        )
        # Without OCR text, very far unrelated controls must not match purely by color/size
        if iou == 0.0 and dist > 0.30 * diag:
            score *= 0.45

    breakdown = {
        'score': round(score, 4),
        'iou': round(iou, 3),
        'center_dist_px': round(dist, 1),
        'size_sim': round(size_sim, 3),
        'panel_sim': round(panel_sim, 2),
        'color_delta': round(color_delta, 1),
        'text_sim': round(tsim, 3) if tsim is not None else None,
    }
    return score, breakdown


def match_controls_temporal(
    controls_a: list[dict[str, Any]],
    controls_b: list[dict[str, Any]],
    panels_a: list[dict[str, Any]],
    panels_b: list[dict[str, Any]],
    screen_size: tuple[int, int],
    prev_track_map: dict[str, str] | None = None,
    next_track_start: int = 0,
) -> dict[str, Any]:
    """Match controls from Frame A -> Frame B globally with persistent track IDs.

    `prev_track_map`: optional `{local_id_in_A: "K7"}` from prior frame for
    multi-frame sequence consistency.
    """
    t0 = time.perf_counter()
    w, h = screen_size
    diag = math.hypot(w, h)
    pa_map = _panel_hint_map(panels_a)
    pb_map = _panel_hint_map(panels_b)
    prev_tracks = dict(prev_track_map or {})

    # Assign deterministic track IDs to Frame A controls (reusing prev_tracks when present)
    used_nums = []
    for tid in prev_tracks.values():
        if tid.startswith('K') and tid[1:].isdigit():
            used_nums.append(int(tid[1:]))
    next_k = max([next_track_start] + [n + 1 for n in used_nums])

    track_of_a: dict[str, str] = {}
    for ca in controls_a:
        lid = ca['id']
        if lid in prev_tracks:
            track_of_a[lid] = prev_tracks[lid]
        else:
            track_of_a[lid] = f'K{next_k}'
            next_k += 1

    na, nb = len(controls_a), len(controls_b)
    aff = np.zeros((na, nb), dtype=np.float64)
    details: dict[tuple[int, int], dict[str, float]] = {}

    for i, ca in enumerate(controls_a):
        for j, cb in enumerate(controls_b):
            s, bd = _pair_affinity(ca, cb, pa_map, pb_map, diag)
            aff[i, j] = s
            details[(i, j)] = bd

    matched_a: set[int] = set()
    matched_b: set[int] = set()
    pairs: list[tuple[int, int, float]] = []

    if na > 0 and nb > 0:
        # Global optimal assignment maximizing total affinity
        cost = 1.0 - aff
        row_ind, col_ind = linear_sum_assignment(cost)
        for i, j in zip(row_ind, col_ind):
            s = float(aff[i, j])
            if s >= MATCH_MIN_AFFINITY:
                pairs.append((int(i), int(j), s))
                matched_a.add(int(i))
                matched_b.add(int(j))

    entries: list[dict[str, Any]] = []
    track_of_b: dict[str, str] = {}

    # Evaluate matched pairs for ambiguity & change classifications
    for i, j, best_s in pairs:
        ca = controls_a[i]
        cb = controls_b[j]
        bd = details[(i, j)]
        tid = track_of_a[ca['id']]

        # Check ambiguity: is there another competing candidate in row i or col j within AMBIGUITY_MARGIN
        # WITHOUT a strong anchor (high IoU >= 0.72 or unique exact text match)?
        strong_anchor = (bd['iou'] >= 0.72) or (bd['text_sim'] == 1.0 and sum(
            1 for jj in range(nb) if details[(i, jj)]['text_sim'] == 1.0
        ) == 1)

        alt_cands = []
        if not strong_anchor:
            for jj in range(nb):
                if jj != j and (best_s - float(aff[i, jj])) <= AMBIGUITY_MARGIN and float(aff[i, jj]) >= MATCH_MIN_AFFINITY:
                    alt_cands.append({
                        'frame_a_local': ca['id'],
                        'frame_b_local': controls_b[jj]['id'],
                        'score': round(float(aff[i, jj]), 4),
                    })
            for ii in range(na):
                if ii != i and (best_s - float(aff[ii, j])) <= AMBIGUITY_MARGIN and float(aff[ii, j]) >= MATCH_MIN_AFFINITY:
                    alt_cands.append({
                        'frame_a_local': controls_a[ii]['id'],
                        'frame_b_local': cb['id'],
                        'score': round(float(aff[ii, j]), 4),
                    })

        if alt_cands:
            entries.append({
                'track': tid,
                'classification': CLS_AMBIGUOUS,
                'flags': [CLS_AMBIGUOUS],
                'inferred_flags': [],
                'frame_a_local': ca['id'],
                'frame_b_local': cb['id'],
                'frame_a': _control_snapshot(ca, tid),
                'frame_b': _control_snapshot(cb, tid),
                'affinity': bd,
                'competing_candidates': alt_cands,
                'note': 'Multiple competing matches within margin without unique spatial/text anchor; equivalence not forced.',
            })
            continue

        track_of_b[cb['id']] = tid
        ba, bb = ca['global_bbox'], cb['global_bbox']
        wa, ha = ba[2] - ba[0], ba[3] - ba[1]
        wb, hb = bb[2] - bb[0], bb[3] - bb[1]
        dw, dh = abs(wb - wa), abs(hb - ha)

        flags: list[str] = []
        inferred: list[str] = []

        moved = bd['center_dist_px'] >= MOVE_THRESH_PX
        resized = (
            (dw >= RESIZE_THRESH_PX and dw / max(1, wa) >= RESIZE_RATIO_THRESH)
            or (dh >= RESIZE_THRESH_PX and dh / max(1, ha) >= RESIZE_RATIO_THRESH)
        )
        color_changed = bd['color_delta'] >= COLOR_DELTA_THRESH
        text_changed = (
            ca.get('text_source') == focr.OCR_OBSERVED
            and cb.get('text_source') == focr.OCR_OBSERVED
            and focr.normalize_text(ca.get('text', '')) != focr.normalize_text(cb.get('text', ''))
        )

        if moved:
            flags.append(CLS_MOVED)
        if resized:
            flags.append(CLS_RESIZED)
        if text_changed:
            flags.append(CLS_TEXT_CHANGED)
        if color_changed:
            flags.append(CLS_COLOR_CHANGED)
        if (color_changed or text_changed) and bd['iou'] >= 0.65:
            # Explicit epistemic separation: state change is INFERRED without accessibility signal
            inferred.append(CLS_STATE_CANDIDATE)

        if not flags:
            flags.append(CLS_UNCHANGED)

        primary = flags[0]
        if CLS_COLOR_CHANGED in flags and CLS_MOVED not in flags and CLS_RESIZED not in flags:
            primary = CLS_TEXT_CHANGED if CLS_TEXT_CHANGED in flags else CLS_COLOR_CHANGED

        entries.append({
            'track': tid,
            'classification': primary,
            'flags': flags,
            'inferred_flags': inferred,
            'frame_a_local': ca['id'],
            'frame_b_local': cb['id'],
            'frame_a': _control_snapshot(ca, tid),
            'frame_b': _control_snapshot(cb, tid),
            'affinity': bd,
            'color_transition': f'{ca.get("hex")}({ca.get("family")})->{cb.get("hex")}({cb.get("family")})' if color_changed else None,
            'text_transition': f'{ca.get("text")!r}->{cb.get("text")!r}' if text_changed else None,
            'delta_center_px': [cb['click_center'][0] - ca['click_center'][0], cb['click_center'][1] - ca['click_center'][1]],
        })

    # Unmatched in Frame A -> REMOVED_CONTROL
    for i, ca in enumerate(controls_a):
        if i in matched_a:
            continue
        tid = track_of_a[ca['id']]
        entries.append({
            'track': tid,
            'classification': CLS_REMOVED,
            'flags': [CLS_REMOVED],
            'inferred_flags': [],
            'frame_a_local': ca['id'],
            'frame_b_local': None,
            'frame_a': _control_snapshot(ca, tid),
            'frame_b': None,
            'affinity': None,
        })

    # Unmatched in Frame B -> NEW_CONTROL (fresh persistent track ID!)
    for j, cb in enumerate(controls_b):
        if j in matched_b:
            continue
        tid = f'K{next_k}'
        next_k += 1
        track_of_b[cb['id']] = tid
        entries.append({
            'track': tid,
            'classification': CLS_NEW,
            'flags': [CLS_NEW],
            'inferred_flags': [],
            'frame_a_local': None,
            'frame_b_local': cb['id'],
            'frame_a': None,
            'frame_b': _control_snapshot(cb, tid),
            'affinity': None,
        })

    # Deterministic order: changed/new/removed/ambiguous first, then unchanged, by track number
    def _sort_key(e: dict[str, Any]):
        prio = 1 if e['classification'] == CLS_UNCHANGED else 0
        tnum = int(e['track'][1:]) if e['track'].startswith('K') and e['track'][1:].isdigit() else 999999
        return (prio, tnum)

    entries.sort(key=_sort_key)
    counts: dict[str, int] = {}
    for e in entries:
        counts[e['classification']] = counts.get(e['classification'], 0) + 1

    return {
        'tracks': entries,
        'summary_counts': counts,
        'frame_a_tracks': track_of_a,
        'frame_b_tracks': track_of_b,
        'next_track_id': next_k,
        'elapsed_ms': round((time.perf_counter() - t0) * 1000.0, 2),
    }


def _control_snapshot(c: dict[str, Any], track_id: str) -> dict[str, Any]:
    return {
        'track': track_id,
        'local': c['id'],
        'kind': c['kind'],
        'bbox': c['global_bbox'],
        'center': c['click_center'],
        'hex': c['hex'],
        'family': c['family'],
        'panel': c.get('inside_panel'),
        'text': c.get('text'),
        'text_source': c.get('text_source'),
        'association': c.get('association'),
    }
