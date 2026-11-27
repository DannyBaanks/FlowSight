#!/usr/bin/env python3
"""FlowSight M3 — ffmpeg-style composable filter pipeline for screenshots.

Architecture inspired by `ffmpeg -vf "filter1=arg,filter2=...,filter3"`:
  Input PNG/JPG/WEBP (read-only) -> FilterContext -> composable deterministic filters
  -> structured bounded UI report (panels, clickable controls, text bands, wireframe
     ASCII map, optional OCR, UI diff, and block splitting/stitching).

Epistemic contract:
  - Purely deterministic computer vision (OpenCV + NumPy + Pillow); zero LLM inside filters.
  - Reports MEASURED geometry/color/contrast/edge facts (`DEMONSTRATED`): panel dividers,
    bounding boxes, clickable centers (cx, cy), accent/bordered widget candidates, and
    high-frequency horizontal text bands.
  - Never invents app names, button labels without OCR, or semantic identities.
  - Optional `ocr` filter fails closed (`OCR_BACKEND=ABSENT`) when tesseract is unavailable.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any, Callable

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

import flowsight_blocks as fb
import flowsight_diff as fdiff
import flowsight_ocr as focr
import flowsight_vlm as fvlm

SCHEMA_SCREEN = 'flowsight-screen-v1'
MAX_SCREEN_DIM = 8192


class FilterDeny(ValueError):
    pass


@dataclass
class FilterContext:
    """Mutable pipeline carrier (never mutates original image file on disk)."""

    source_path: str
    source_name: str
    source_sha256: str
    full_w: int
    full_h: int
    rgb: np.ndarray                 # current working RGB uint8 (H, W, 3)
    offset_x: int = 0               # global x offset of current crop
    offset_y: int = 0               # global y offset of current crop
    artifacts: dict[str, Any] = field(default_factory=dict)
    trace: list[dict[str, Any]] = field(default_factory=list)
    ocr_provider: Callable[..., dict[str, Any]] | None = None
    vlm_policy: fvlm.VLMProbePolicy = field(default_factory=lambda: fvlm.DEFAULT_POLICY)
    vlm_backend: Callable[..., str] | None = None
    prev_track_map: dict[str, str] | None = None

    @property
    def w(self) -> int:
        return int(self.rgb.shape[1])

    @property
    def h(self) -> int:
        return int(self.rgb.shape[0])

    @property
    def global_bbox(self) -> list[int]:
        return [self.offset_x, self.offset_y, self.offset_x + self.w, self.offset_y + self.h]

    @property
    def source_frame_id(self) -> str:
        return f'{self.source_name}:{self.source_sha256[:12]}'

    def to_global(self, x0: int, y0: int, x1: int, y1: int) -> list[int]:
        return [self.offset_x + x0, self.offset_y + y0, self.offset_x + x1, self.offset_y + y1]


@dataclass(frozen=True)
class FilterSpec:
    name: str
    summary: str
    syntax: str
    fn: Callable[[FilterContext, dict[str, str]], None]
    inputs: tuple[str, ...] = ('rgb',)
    outputs: tuple[str, ...] = ()
    optional_deps: tuple[str, ...] = ()
    requires_network: bool = False
    requires_gpu: bool = False
    deterministic: bool = True
    coord_space: str = 'global'
    provenance: str = 'DEMONSTRATED'
    degradations: str = 'fail-closed FilterDeny on invalid args'


FILTERS: dict[str, FilterSpec] = {}


def register_filter(
    name: str,
    summary: str,
    syntax: str,
    *,
    inputs: tuple[str, ...] = ('rgb',),
    outputs: tuple[str, ...] = (),
    optional_deps: tuple[str, ...] = (),
    requires_network: bool = False,
    requires_gpu: bool = False,
    deterministic: bool = True,
    coord_space: str = 'global',
    provenance: str = 'DEMONSTRATED',
    degradations: str = 'fail-closed FilterDeny on invalid args',
):
    def _dec(fn: Callable[[FilterContext, dict[str, str]], None]):
        FILTERS[name] = FilterSpec(
            name=name, summary=summary, syntax=syntax, fn=fn,
            inputs=inputs, outputs=outputs, optional_deps=optional_deps,
            requires_network=requires_network, requires_gpu=requires_gpu,
            deterministic=deterministic, coord_space=coord_space,
            provenance=provenance, degradations=degradations,
        )
        return fn
    return _dec


def _hex(rgb: tuple[int, int, int] | list[int] | np.ndarray) -> str:
    return f'#{int(rgb[0]):02x}{int(rgb[1]):02x}{int(rgb[2]):02x}'


def _box_contains(outer: list[int], inner: list[int], min_overlap: float = 0.65) -> bool:
    ix0 = max(outer[0], inner[0])
    iy0 = max(outer[1], inner[1])
    ix1 = min(outer[2], inner[2])
    iy1 = min(outer[3], inner[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return False
    inter = (ix1 - ix0) * (iy1 - iy0)
    area = max(1, (inner[2] - inner[0]) * (inner[3] - inner[1]))
    return (inter / area) >= min_overlap


def _load_screen_rgb(path: str | Path) -> tuple[np.ndarray, str, str]:
    p = Path(path)
    if not p.is_file():
        raise FilterDeny(f'image not found: {p}')
    raw = p.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    try:
        with Image.open(p) as im:
            w, h = im.size
            if not (8 <= w <= MAX_SCREEN_DIM and 8 <= h <= MAX_SCREEN_DIM):
                raise FilterDeny(f'screen {w}x{h} outside 8..{MAX_SCREEN_DIM}')
            arr = np.asarray(im.convert('RGB'), dtype=np.uint8)
            return arr, sha, p.name
    except (OSError, UnidentifiedImageError, Image.DecompressionBombError) as e:
        raise FilterDeny(f'cannot decode image: {e}') from e


# ============================================================================
# Filter graph parser (`-vf "crop=0:0:960:540,theme,layout,controls,ui_map=48:20"`)
# ============================================================================

def parse_vf(vf: str) -> list[tuple[str, dict[str, str]]]:
    """Parse ffmpeg-style comma-separated filter chain."""
    if not vf or not vf.strip():
        raise FilterDeny('empty filter chain (-vf)')
    steps: list[tuple[str, dict[str, str]]] = []
    # Split on commas outside quotes
    parts: list[str] = []
    cur: list[str] = []
    in_q = False
    for ch in vf:
        if ch in ("'", '"'):
            in_q = not in_q
            continue
        if ch == ',' and not in_q:
            tok = ''.join(cur).strip()
            if tok:
                parts.append(tok)
            cur = []
        else:
            cur.append(ch)
    tail = ''.join(cur).strip()
    if tail:
        parts.append(tail)

    for part in parts:
        if '=' in part:
            name, raw_args = part.split('=', 1)
            name = name.strip().lower()
            raw_args = raw_args.strip()
        else:
            name, raw_args = part.strip().lower(), ''
        if name not in FILTERS:
            raise FilterDeny(f'unknown filter {name!r}; available: {sorted(FILTERS)}')
        params: dict[str, str] = {}
        if raw_args:
            # Supports both key=val:key2=val2 and positional colons (e.g. crop=x:y:w:h or ui_map=48:20)
            tokens = raw_args.split(':')
            pos_idx = 0
            for tok in tokens:
                if '=' in tok:
                    k, v = tok.split('=', 1)
                    params[k.strip()] = v.strip()
                else:
                    params[f'_{pos_idx}'] = tok.strip()
                    pos_idx += 1
        steps.append((name, params))
    return steps


def _int_param(params: dict[str, str], key: str, pos: int | None, default: int, lo: int, hi: int) -> int:
    raw = params.get(key)
    if raw is None and pos is not None:
        raw = params.get(f'_{pos}')
    if raw is None or raw == '':
        return default
    try:
        v = int(raw, 10)
    except ValueError:
        raise FilterDeny(f'parameter {key} must be int, got {raw!r}') from None
    if not (lo <= v <= hi):
        raise FilterDeny(f'parameter {key}={v} outside [{lo},{hi}]')
    return v


# ============================================================================
# Built-in Screenshot Filters (ffmpeg-style composable blocks)
# ============================================================================

@register_filter(
    'crop',
    'Crop working frame preserving exact global screen coordinates',
    'crop=x:y:w:h (or crop=x=0:y=0:w=960:h=540)',
)
def _f_crop(ctx: FilterContext, params: dict[str, str]) -> None:
    x = _int_param(params, 'x', 0, 0, 0, ctx.w - 1)
    y = _int_param(params, 'y', 1, 0, 0, ctx.h - 1)
    w = _int_param(params, 'w', 2, ctx.w - x, 8, ctx.w - x)
    h = _int_param(params, 'h', 3, ctx.h - y, 8, ctx.h - y)
    ctx.rgb = ctx.rgb[y:y + h, x:x + w].copy()
    ctx.offset_x += x
    ctx.offset_y += y
    ctx.artifacts['crop'] = {'global_bbox': ctx.global_bbox, 'size': [ctx.w, ctx.h]}


@register_filter(
    'theme',
    'Classify UI polarity (dark/light/mixed), background surface, and accent colors',
    'theme (no args)',
)
def _f_theme(ctx: FilterContext, params: dict[str, str]) -> None:
    # Subsample for speed on 1920x1080 (<10ms)
    step = max(1, min(ctx.w, ctx.h) // 240)
    sub = ctx.rgb[::step, ::step, :]
    luma = 0.299 * sub[:, :, 0] + 0.587 * sub[:, :, 1] + 0.114 * sub[:, :, 2]
    med_luma = float(np.median(luma))
    mean_luma = float(np.mean(luma))
    dark_pct = float(np.mean(luma < 85.0) * 100.0)
    light_pct = float(np.mean(luma > 180.0) * 100.0)
    if dark_pct >= 55.0:
        polarity = 'dark'
    elif light_pct >= 55.0:
        polarity = 'light'
    else:
        polarity = 'mixed'

    # Dominant surface colors via 16-level per-channel histogram (fast, exact binning)
    q = (sub // 16) * 16 + 8
    flat = q.reshape(-1, 3)
    codes = (flat[:, 0].astype(np.int32) << 16) | (flat[:, 1].astype(np.int32) << 8) | flat[:, 2].astype(np.int32)
    uniq, cnts = np.unique(codes, return_counts=True)
    order = np.argsort(-cnts)
    total = int(flat.shape[0])
    surfaces = []
    accents = []
    for idx in order[:40]:
        c = int(uniq[idx])
        r, g, b = (c >> 16) & 255, (c >> 8) & 255, c & 255
        pct = round(int(cnts[idx]) * 100.0 / total, 1)
        fam = fb._hsv_family(r, g, b)
        mx, mn = max(r, g, b), min(r, g, b)
        sat = (mx - mn) / max(1, mx)
        entry = {'hex': _hex((r, g, b)), 'rgb': [r, g, b], 'family': fam, 'pct': pct}
        if len(surfaces) < 5:
            surfaces.append(entry)
        if sat >= 0.42 and mx >= 80 and pct >= 0.1 and len(accents) < 6:
            accents.append(entry)

    ctx.artifacts['theme'] = {
        'polarity': polarity,
        'median_luma': round(med_luma, 1),
        'mean_luma': round(mean_luma, 1),
        'dark_pct': round(dark_pct, 1),
        'light_pct': round(light_pct, 1),
        'dominant_surfaces': surfaces,
        'accent_colors': accents,
    }


def _merge_1d_lines(indices: np.ndarray, min_gap: int = 8) -> list[int]:
    if indices.size == 0:
        return []
    groups = [[int(indices[0])]]
    for v in indices[1:]:
        iv = int(v)
        if iv - groups[-1][-1] <= min_gap:
            groups[-1].append(iv)
        else:
            groups.append([iv])
    return [g[len(g) // 2] for g in groups]


def _merge_1d_ranked(indices: np.ndarray, cov: np.ndarray, min_gap: int = 8, max_keep: int = 8) -> list[int]:
    """Merge nearby 1D divider indices, rank groups by coverage strength, keep top max_keep, then sort by position."""
    if indices.size == 0 or cov.size == 0:
        return []
    groups = [[int(indices[0])]]
    for v in indices[1:]:
        iv = int(v)
        if iv - groups[-1][-1] <= min_gap:
            groups[-1].append(iv)
        else:
            groups.append([iv])
    scored = []
    n_cov = len(cov)
    for g in groups:
        best_i = max(g, key=lambda idx: float(cov[min(max(0, idx - 1), n_cov - 1)]))
        strength = max(float(cov[min(max(0, idx - 1), n_cov - 1)]) for idx in g)
        scored.append((strength, best_i))
    scored.sort(key=lambda t: (-t[0], t[1]))
    return sorted(idx for _, idx in scored[:max_keep])


def _detect_morph_boxes(gray: np.ndarray, dy: np.ndarray, dx: np.ndarray) -> tuple[list[list[int]], set[int]]:
    """Detect closed axis-aligned thin-bordered rectangles via morphological line extraction.
    Returns (list of [x0, y0, x1, y1], set of row y indices that are borders of closed input/control boxes).
    """
    h, w = gray.shape
    dy_bin = np.zeros((h, w), dtype=np.uint8)
    dx_bin = np.zeros((h, w), dtype=np.uint8)
    dy_bin[1:, :] = (dy >= 5.0).astype(np.uint8) * 255
    dx_bin[:, 1:] = (dx >= 5.0).astype(np.uint8) * 255
    hk = cv2.getStructuringElement(cv2.MORPH_RECT, (max(24, w // 36), 1))
    vk = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(12, h // 72)))
    h_lines = cv2.morphologyEx(dy_bin, cv2.MORPH_OPEN, hk)
    v_lines = cv2.morphologyEx(dx_bin, cv2.MORPH_OPEN, vk)
    h_d = cv2.dilate(h_lines, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 3)))
    v_d = cv2.dilate(v_lines, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 9)))
    grid = cv2.bitwise_or(h_d, v_d)
    cnts, _ = cv2.findContours(grid, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    boxes: list[list[int]] = []
    box_rows: set[int] = set()
    for cnt in cnts:
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw < 48 or bh < 18 or bh > h * 0.45:
            continue
        if bw > w * 0.98 and bh > h * 0.85:
            continue
        top_cov = float(np.mean(np.max(h_lines[max(0, y - 2):min(h, y + 4), x:x + bw], axis=0) > 0)) if bh > 4 else 0.0
        bot_cov = float(np.mean(np.max(h_lines[max(0, y + bh - 4):min(h, y + bh + 2), x:x + bw], axis=0) > 0)) if bh > 4 else 0.0
        left_cov = float(np.mean(np.max(v_lines[y:y + bh, max(0, x - 2):min(w, x + 4)], axis=1) > 0)) if bw > 4 else 0.0
        right_cov = float(np.mean(np.max(v_lines[y:y + bh, max(0, x + bw - 4):min(w, x + bw + 2)], axis=1) > 0)) if bw > 4 else 0.0
        if top_cov >= 0.65 and bot_cov >= 0.65 and left_cov >= 0.45 and right_cov >= 0.45:
            boxes.append([x, y, x + bw, y + bh])
            # Closed control boxes with left or right margin inside the screen are controls, not layout dividers
            if bh <= 110 and (x >= 6 or x + bw <= w - 6):
                for ry in range(max(0, y - 4), min(h, y + 5)):
                    box_rows.add(ry)
                for ry in range(max(0, y + bh - 5), min(h, y + bh + 4)):
                    box_rows.add(ry)
    return boxes, box_rows


def _xy_cut_panels(
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    dy: np.ndarray,
    dx: np.ndarray,
    box_rows: set[int],
    full_w: int,
    full_h: int,
    depth: int = 0,
) -> list[list[int]]:
    """Recursive XY-cut panel partitioning: splits a region only on dividers strong WITHIN that region."""
    rw, rh = x1 - x0, y1 - y0
    if depth >= 5 or rw < 48 or rh < 28:
        return [[x0, y0, x1, y1]]
    min_h_gap = max(12, full_h // 45)
    min_v_gap = max(12, full_w // 45)

    sub_dy = dy[y0:y1 - 1, x0:x1]
    h_cov = np.mean(sub_dy >= 10.0, axis=1) if sub_dy.size > 0 else np.zeros(0)
    for r in box_rows:
        if y0 < r < y1 and 0 <= (r - y0 - 1) < len(h_cov):
            h_cov[r - y0 - 1] = 0.0
    h_thresh = 0.60 if depth == 0 else 0.68
    h_idx = np.where(h_cov >= h_thresh)[0] + 1
    h_idx = h_idx[(h_idx >= 12) & (h_idx <= rh - 12)]
    h_splits = [y0 + int(y) for y in _merge_1d_ranked(h_idx, h_cov, min_gap=min_h_gap, max_keep=8)]

    sub_dx = dx[y0:y1, x0:x1 - 1]
    v_cov = np.mean(sub_dx >= 10.0, axis=0) if sub_dx.size > 0 else np.zeros(0)
    v_thresh = 0.60 if depth == 0 else 0.68
    v_idx = np.where(v_cov >= v_thresh)[0] + 1
    v_idx = v_idx[(v_idx >= 12) & (v_idx <= rw - 12)]
    v_splits = [x0 + int(x) for x in _merge_1d_ranked(v_idx, v_cov, min_gap=min_v_gap, max_keep=8)]

    best_h = float(np.max(h_cov[h_idx - 1])) if len(h_splits) > 0 else 0.0
    best_v = float(np.max(v_cov[v_idx - 1])) if len(v_splits) > 0 else 0.0

    if not h_splits and not v_splits:
        return [[x0, y0, x1, y1]]

    out: list[list[int]] = []
    if best_h >= best_v and h_splits:
        ys = [y0] + h_splits + [y1]
        for i in range(len(ys) - 1):
            if ys[i + 1] - ys[i] >= 10:
                out.extend(_xy_cut_panels(x0, ys[i], x1, ys[i + 1], dy, dx, box_rows, full_w, full_h, depth + 1))
    else:
        xs = [x0] + v_splits + [x1]
        for i in range(len(xs) - 1):
            if xs[i + 1] - xs[i] >= 16:
                out.extend(_xy_cut_panels(xs[i], y0, xs[i + 1], y1, dy, dx, box_rows, full_w, full_h, depth + 1))
    return out


@register_filter(
    'layout',
    'Detect structural UI dividers (H/V splits) and panel/pane containers with global bboxes',
    'layout[=max_panels=12]',
)
def _f_layout(ctx: FilterContext, params: dict[str, str]) -> None:
    max_panels = _int_param(params, 'max_panels', 0, 12, 2, 32)
    h, w = ctx.h, ctx.w
    gray = cv2.cvtColor(ctx.rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)

    dy = np.abs(np.diff(gray, axis=0))  # (H-1, W)
    dx = np.abs(np.diff(gray, axis=1))  # (H, W-1)

    _, box_rows = _detect_morph_boxes(gray, dy, dx)

    h_cov = np.mean(dy >= 10.0, axis=1)
    v_cov = np.mean(dx >= 10.0, axis=0)
    for r in box_rows:
        if 0 <= r - 1 < len(h_cov):
            h_cov[r - 1] = 0.0

    h_idx = np.where(h_cov >= 0.42)[0] + 1
    v_idx = np.where(v_cov >= 0.42)[0] + 1
    h_idx = h_idx[(h_idx >= 12) & (h_idx <= h - 12)]
    v_idx = v_idx[(v_idx >= 12) & (v_idx <= w - 12)]

    # Rank by coverage strength first, keep top 8, then sort by position
    h_divs = _merge_1d_ranked(h_idx, h_cov, min_gap=max(12, h // 45), max_keep=8)
    v_divs = _merge_1d_ranked(v_idx, v_cov, min_gap=max(12, w // 45), max_keep=8)

    # Recursive XY-cut so local dividers in main pane never slice across sidebars
    xy_rects = _xy_cut_panels(0, 0, w, h, dy, dx, box_rows, w, h, depth=0)

    merged_rows = []
    for x0, y0, x1, y1 in xy_rects:
        if y1 - y0 < 10 or x1 - x0 < 16:
            continue
        patch = ctx.rgb[y0:y1, x0:x1]
        med_rgb = [int(np.median(patch[:, :, c])) for c in range(3)]
        luma = round(0.299 * med_rgb[0] + 0.587 * med_rgb[1] + 0.114 * med_rgb[2], 1)
        gpatch = gray[y0:y1, x0:x1]
        std_l = round(float(np.std(gpatch)), 1)
        area_pct = round((x1 - x0) * (y1 - y0) * 100.0 / (w * h), 1)
        merged_rows.append({
            'local': [x0, y0, x1, y1],
            'bg_rgb': med_rgb,
            'luma': luma,
            'std_luma': std_l,
            'area_pct': area_pct,
        })

    # Positional geometric hint (pure geometry, NEVER an app/semantic claim)
    panels = []
    merged_rows.sort(key=lambda z: (-z['area_pct'], z['local'][1], z['local'][0]))
    for idx, p in enumerate(merged_rows[:max_panels]):
        x0, y0, x1, y1 = p['local']
        pw, ph = x1 - x0, y1 - y0
        if y0 <= 12 and ph <= h * 0.14 and pw >= w * 0.55:
            pos_hint = 'top-bar-band'
        elif y1 >= h - 12 and ph <= h * 0.12 and pw >= w * 0.55:
            pos_hint = 'bottom-bar-band'
        elif x0 <= 12 and pw <= w * 0.32 and ph >= h * 0.45:
            pos_hint = 'left-sidebar-band'
        elif x1 >= w - 12 and pw <= w * 0.32 and ph >= h * 0.45:
            pos_hint = 'right-sidebar-band'
        elif p['area_pct'] >= 25.0:
            pos_hint = 'main-pane'
        else:
            pos_hint = 'sub-pane'
        panels.append({
            'id': f'P{idx}',
            'global_bbox': ctx.to_global(x0, y0, x1, y1),
            'size': [pw, ph],
            'area_pct': p['area_pct'],
            'bg_hex': _hex(p['bg_rgb']),
            'bg_family': fb._hsv_family(*p['bg_rgb']),
            'luma': p['luma'],
            'std_luma': p['std_luma'],
            'position_hint': pos_hint,
        })

    ctx.artifacts['layout'] = {
        'h_dividers_global_y': [ctx.offset_y + y for y in h_divs],
        'v_dividers_global_x': [ctx.offset_x + x for x in v_divs],
        'panels': panels,
        'total_panels_detected': len(merged_rows),
        'truncated': max(0, len(merged_rows) - len(panels)),
    }


@register_filter(
    'controls',
    'Detect clickable UI control candidates (accent buttons, bordered inputs/cards, badges) with click centers',
    'controls[=max=24]',
)
def _f_controls(ctx: FilterContext, params: dict[str, str]) -> None:
    max_controls = _int_param(params, 'max', 0, 24, 1, 100)
    h, w = ctx.h, ctx.w
    hsv = cv2.cvtColor(ctx.rgb, cv2.COLOR_RGB2HSV)
    gray = cv2.cvtColor(ctx.rgb, cv2.COLOR_RGB2GRAY)
    gray_f = gray.astype(np.float32)
    panels = ctx.artifacts.get('layout', {}).get('panels', [])

    candidates: list[dict[str, Any]] = []

    # Detect high-color-variance graphic/image regions (e.g., pixel-art banners or photos)
    # where small saturated blobs are visual art rather than interactive UI controls.
    # True dark UI backgrounds have low HSV Value (<45) even if Saturation is high, so require V >= 70.
    image_regions: list[list[int]] = []
    sat_mask = ((hsv[:, :, 1] >= 85) & (hsv[:, :, 2] >= 70)).astype(np.uint8) * 255
    hue_q = (hsv[:, :, 0] // 20) + 1  # 1..9 hue buckets
    bright_hue = np.where((hsv[:, :, 1] >= 85) & (hsv[:, :, 2] >= 70), hue_q, 0)
    k_img = cv2.getStructuringElement(cv2.MORPH_RECT, (max(32, w // 24), max(16, h // 36)))
    img_Grouping = cv2.morphologyEx(sat_mask, cv2.MORPH_CLOSE, k_img)
    icnts, _ = cv2.findContours(img_Grouping, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in icnts:
        ix, iy, ibw, ibh = cv2.boundingRect(cnt)
        if ibw >= w * 0.25 and 40 <= ibh <= h * 0.45 and ibw * ibh >= w * h * 0.03:
            sub_bh = bright_hue[iy:iy + ibh:2, ix:ix + ibw:2]
            bright_frac = float(np.mean(sub_bh > 0))
            distinct_hues = len(np.unique(sub_bh[sub_bh > 0]))
            if bright_frac >= 0.15 and distinct_hues >= 4:
                image_regions.append([ix, iy, ix + ibw, iy + ibh])
                sub_rgb = ctx.rgb[iy:iy + ibh:2, ix:ix + ibw:2]
                med = [int(np.median(sub_rgb[:, :, c])) for c in range(3)]
                candidates.append({
                    'local': [ix, iy, ix + ibw, iy + ibh],
                    'kind': 'image-region',
                    'hex': _hex(med),
                    'family': fb._hsv_family(*med),
                    'fill_ratio': round(bright_frac, 2),
                    'has_inner_contrast': True,
                    'priority': 1,
                    'score': float(ibw * ibh),
                })

    def _in_image_region(box: list[int]) -> bool:
        for ir in image_regions:
            if _box_contains(ir, box, min_overlap=0.50):
                return True
        return False

    # 1. Accent-colored filled rectangles/pills (buttons, primary CTAs, alert banners, badges)
    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
    sat_closed = cv2.morphologyEx(sat_mask, cv2.MORPH_CLOSE, k_close)
    cnts, _ = cv2.findContours(sat_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in cnts:
        x, y, bw, bh = cv2.boundingRect(cnt)
        area = bw * bh
        if bw < 18 or bh < 14 or bw > w * 0.85 or bh > h * 0.45:
            continue
        if area < 300:
            continue
        fill_ratio = float(cv2.contourArea(cnt)) / max(1.0, float(area))
        if fill_ratio < 0.70:
            continue
        patch = ctx.rgb[y:y + bh, x:x + bw]
        med = [int(np.median(patch[:, :, c])) for c in range(3)]
        gpatch = gray_f[y:y + bh, x:x + bw]
        inner_dx = np.abs(np.diff(gpatch[2:-2, 2:-2], axis=1)) if (bh > 6 and bw > 6) else np.zeros((1, 1))
        has_label = bool(np.mean(inner_dx >= 28.0) >= 0.025)
        kind = 'badge' if (bw < 64 and bh < 32) else 'accent-control'
        in_img = _in_image_region([x, y, x + bw, y + bh])
        # Downweight saturated blobs inside high-color-variance image/banner regions
        prio = 2 if in_img else 0
        sc = area * (1.5 if has_label else 1.0) * (0.05 if in_img else 1.0)
        candidates.append({
            'local': [x, y, x + bw, y + bh],
            'kind': kind,
            'hex': _hex(med),
            'family': fb._hsv_family(*med),
            'fill_ratio': round(fill_ratio, 2),
            'has_inner_contrast': has_label,
            'priority': prio,
            'score': sc,
        })

    # 2. Sharp bordered rectangular boxes (Canny + Morphological line extraction for 1px/wide inputs)
    edges = cv2.Canny(gray, 35, 110)
    k_dil = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges_d = cv2.dilate(edges, k_dil, iterations=1)
    cnts2, _ = cv2.findContours(edges_d, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    raw_rect_boxes: list[tuple[int, int, int, int, float]] = []
    for cnt in cnts2:
        x, y, bw, bh = cv2.boundingRect(cnt)
        area = bw * bh
        if bw < 48 or bh < 18 or bw > w * 0.99 or bh > h * 0.50:
            continue
        if area < 1000 or area > w * h * 0.45:
            continue
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.025 * peri, True)
        if not (4 <= len(approx) <= 8):
            continue
        rect_ratio = float(cv2.contourArea(cnt)) / max(1.0, float(area))
        if rect_ratio < 0.84:
            continue
        raw_rect_boxes.append((x, y, bw, bh, rect_ratio))

    dy = np.abs(np.diff(gray_f, axis=0))
    dx = np.abs(np.diff(gray_f, axis=1))
    morph_boxes, _ = _detect_morph_boxes(gray_f, dy, dx)
    for x0, y0, x1, y1 in morph_boxes:
        raw_rect_boxes.append((x0, y0, x1 - x0, y1 - y0, 0.96))

    for x, y, bw, bh, rect_ratio in raw_rect_boxes:
        area = bw * bh
        # Never discard wide bordered inputs; only skip small/tall sub-rectangles inside an image-region
        if _in_image_region([x, y, x + bw, y + bh]) and not (bh <= 64 and bw >= w * 0.25):
            continue
        patch = ctx.rgb[y + 2:y + bh - 2, x + 2:x + bw - 2] if (bh > 6 and bw > 6) else ctx.rgb[y:y + bh, x:x + bw]
        med = [int(np.median(patch[:, :, c])) for c in range(3)]
        gpatch = gray_f[y:y + bh, x:x + bw]
        inner_dx = np.abs(np.diff(gpatch[3:-3, 3:-3], axis=1)) if (bh > 8 and bw > 8) else np.zeros((1, 1))
        has_label = bool(np.mean(inner_dx >= 20.0) >= 0.010)
        aspect = bw / max(1.0, float(bh))
        if bh <= 64 and aspect >= 2.6:
            kind = 'input-or-bar-box'
        elif bh <= 64 and bw <= 280:
            kind = 'bordered-control'
        else:
            kind = 'card-or-dialog'
        # Rank bordered inputs with inner text in the top priority tier so they are never starved
        is_primary_input = (kind == 'input-or-bar-box' and (has_label or bw >= w * 0.25))
        prio = 0 if is_primary_input else 1
        sc = area * (2.5 if is_primary_input else (1.2 if kind != 'card-or-dialog' else 0.7))
        candidates.append({
            'local': [x, y, x + bw, y + bh],
            'kind': kind,
            'hex': _hex(med),
            'family': fb._hsv_family(*med),
            'fill_ratio': round(rect_ratio, 2),
            'has_inner_contrast': has_label,
            'priority': prio,
            'score': sc,
        })

    # Non-maximum suppression (deduplicate overlapping detections IoU >= 0.55)
    candidates.sort(key=lambda c: (c['priority'], -c['score']))
    deduped: list[dict[str, Any]] = []
    for cand in candidates:
        lx0, ly0, lx1, ly1 = cand['local']
        la = (lx1 - lx0) * (ly1 - ly0)
        dup = False
        for prev in deduped:
            px0, py0, px1, py1 = prev['local']
            ix0, iy0 = max(lx0, px0), max(ly0, py0)
            ix1, iy1 = min(lx1, px1), min(ly1, py1)
            if ix1 > ix0 and iy1 > iy0:
                inter = (ix1 - ix0) * (iy1 - iy0)
                pa = (px1 - px0) * (py1 - py0)
                iou = inter / max(1, la + pa - inter)
                # Also suppress thin inner/outer double-border duplicates of the exact same box
                near_border = (abs(lx0 - px0) <= 6 and abs(ly0 - py0) <= 6 and abs(lx1 - px1) <= 6 and abs(ly1 - py1) <= 6)
                if iou >= 0.55 or near_border:
                    dup = True
                    break
        if not dup:
            deduped.append(cand)

    total_detected = len(deduped)
    # Sort top-to-bottom, left-to-right for human/agent readability
    kept = deduped[:max_controls]
    kept.sort(key=lambda c: (c['local'][1] // 16, c['local'][0]))

    controls = []
    for i, c in enumerate(kept):
        gb = ctx.to_global(*c['local'])
        cx = (gb[0] + gb[2]) // 2
        cy = (gb[1] + gb[3]) // 2
        inside = None
        for p in panels:
            if _box_contains(p['global_bbox'], gb, min_overlap=0.70):
                inside = p['id']
                break
        controls.append({
            'id': f'W{i}',
            'kind': c['kind'],
            'global_bbox': gb,
            'size': [gb[2] - gb[0], gb[3] - gb[1]],
            'click_center': [cx, cy],
            'hex': c['hex'],
            'family': c['family'],
            'has_inner_contrast': c['has_inner_contrast'],
            'inside_panel': inside,
        })

    ctx.artifacts['controls'] = {
        'total_detected': total_detected,
        'shown': len(controls),
        'truncated': max(0, total_detected - len(controls)),
        'controls': controls,
    }


@register_filter(
    'text_bands',
    'Detect geometric horizontal text line bands (no OCR required; pure contrast/stroke frequency)',
    'text_bands[=max=35]',
)
def _f_text_bands(ctx: FilterContext, params: dict[str, str]) -> None:
    max_bands = _int_param(params, 'max', 0, 35, 1, 150)
    h, w = ctx.h, ctx.w
    gray = cv2.cvtColor(ctx.rgb, cv2.COLOR_RGB2GRAY)

    # Text strokes create strong horizontal gradients (vertical glyph stems)
    gx = cv2.Sobel(gray, cv2.CV_16S, 1, 0, ksize=3)
    abs_gx = cv2.convertScaleAbs(gx)
    _, bin_gx = cv2.threshold(abs_gx, 45, 255, cv2.THRESH_BINARY)

    # Connect letters into word/line bands horizontally, keeping lines vertically separated
    k_line = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
    dil = cv2.dilate(bin_gx, k_line, iterations=1)
    k_open = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3))
    clean = cv2.morphologyEx(dil, cv2.MORPH_OPEN, k_open)

    cnts, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    panels = ctx.artifacts.get('layout', {}).get('panels', [])
    controls = ctx.artifacts.get('controls', {}).get('controls', [])

    raw_bands = []
    for cnt in cnts:
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bh < 7 or bh > 54 or bw < 20:
            continue
        if bw / max(1.0, float(bh)) < 1.3:
            continue
        # Measure horizontal transition density inside band to reject single vertical borders
        sub_bin = bin_gx[y:y + bh, x:x + bw]
        row_trans = np.sum(np.abs(np.diff((sub_bin > 0).astype(np.int8), axis=1)), axis=1)
        med_trans = float(np.median(row_trans))
        if med_trans < 4.0:
            continue
        sub_g = gray[y:y + bh, x:x + bw]
        contrast = int(np.percentile(sub_g, 95) - np.percentile(sub_g, 5))
        if contrast < 40:
            continue
        raw_bands.append({
            'local': [x, y, x + bw, y + bh],
            'transitions': round(med_trans, 1),
            'contrast': contrast,
        })

    raw_bands.sort(key=lambda b: (b['local'][1] // 10, b['local'][0]))
    bands = []
    for i, b in enumerate(raw_bands[:max_bands]):
        gb = ctx.to_global(*b['local'])
        inside_p = next((p['id'] for p in panels if _box_contains(p['global_bbox'], gb, 0.65)), None)
        inside_w = next((c['id'] for c in controls if _box_contains(c['global_bbox'], gb, 0.55)), None)
        bands.append({
            'id': f'T{i}',
            'global_bbox': gb,
            'size': [gb[2] - gb[0], gb[3] - gb[1]],
            'transitions': b['transitions'],
            'contrast': b['contrast'],
            'inside_panel': inside_p,
            'inside_control': inside_w,
        })

    ctx.artifacts['text_bands'] = {
        'total_detected': len(raw_bands),
        'shown': len(bands),
        'bands': bands,
    }


@register_filter(
    'ocr',
    'Local OCR (Tesseract or RapidOCR ONNX) + geometric text-control association (TEXT_INSIDE/LABEL_ASSOCIATED/AMBIGUOUS/NO_TEXT_EVIDENCE)',
    'ocr[=lang=eng:min_conf=60:backend=auto]',
    inputs=('rgb', 'optional:layout', 'optional:controls'),
    outputs=('ocr', 'text_control_association'),
    optional_deps=('tesseract', 'rapidocr-onnxruntime'),
    requires_network=False,
    requires_gpu=False,
    deterministic=True,
    coord_space='global',
    provenance='DEMONSTRATED (OCR_OBSERVED) / UNCERTAIN (OCR_UNCERTAIN) / ABSENT (OCR_ABSENT)',
    degradations='returns OCR_ABSENT when no local OCR backend is installed; never invents text',
)
def _f_ocr(ctx: FilterContext, params: dict[str, str]) -> None:
    lang = params.get('lang') or params.get('_0') or 'eng'
    min_conf = float(_int_param(params, 'min_conf', 1, 60, 0, 100))
    psm = _int_param(params, 'psm', None, 11, 3, 13)
    backend = params.get('backend') or params.get('_2') or 'auto'
    res = focr.run_ocr(
        ctx.rgb,
        offset_x=ctx.offset_x,
        offset_y=ctx.offset_y,
        source_frame=ctx.source_frame_id,
        lang=lang,
        min_observed_conf=min_conf,
        psm=psm,
        backend=backend,
        provider=ctx.ocr_provider,
    )
    # Backward-compatible alias `lines` pointing to `items`
    res['lines'] = res.get('items', [])
    ctx.artifacts['ocr'] = res

    # If controls were already extracted earlier in the pipeline, associate text immediately
    if 'controls' in ctx.artifacts:
        panels = ctx.artifacts.get('layout', {}).get('panels', [])
        assoc = focr.associate_text_to_controls(
            ctx.artifacts['controls']['controls'],
            res,
            panels=panels,
        )
        ctx.artifacts['controls']['controls'] = assoc['controls']
        ctx.artifacts['text_control_association'] = {
            'elapsed_ms': assoc['elapsed_ms'],
            'ambiguities': assoc['ambiguities'],
            'unmatched_ocr_ids': assoc['unmatched_ocr_ids'],
        }


@register_filter(
    'vlm_probe',
    'Optional crop-only VLM semantic escalation contract (DISABLED by default; fail-closed)',
    'vlm_probe[=question=...:model=...]',
    inputs=('rgb', 'crop'),
    outputs=('vlm_probe',),
    optional_deps=('authorized_vlm_backend',),
    requires_network=True,
    requires_gpu=False,
    deterministic=False,
    coord_space='global',
    provenance='VLM_INFERRED',
    degradations='returns VLM_DISABLED by default; refuses full-frame transfers',
)
def _f_vlm_probe(ctx: FilterContext, params: dict[str, str]) -> None:
    question = params.get('question') or params.get('_0') or 'Describe the cropped UI element.'
    model = params.get('model') or params.get('_1') or ctx.vlm_policy.default_model
    raw_bytes = ctx.rgb.tobytes()
    req = fvlm.VLMProbeRequest(
        source_sha256=ctx.source_sha256,
        full_size=(ctx.full_w, ctx.full_h),
        global_bbox=tuple(ctx.global_bbox),
        crop_sha256=hashlib.sha256(raw_bytes).hexdigest(),
        question=question,
        model=model,
    )
    res = fvlm.execute_vlm_probe(raw_bytes, req, policy=ctx.vlm_policy, backend=ctx.vlm_backend)
    ctx.artifacts['vlm_probe'] = res.to_dict()


@register_filter(
    'ui_map',
    'Render compact ASCII wireframe schematic of layout dividers, controls, and text bands',
    'ui_map[=cols:rows] (default 48:20)',
)
def _f_ui_map(ctx: FilterContext, params: dict[str, str]) -> None:
    cols = _int_param(params, 'cols', 0, 48, 16, 80)
    rows = _int_param(params, 'rows', 1, 20, 8, 40)
    h, w = ctx.h, ctx.w
    ox, oy = ctx.offset_x, ctx.offset_y

    # Base brightness map ('.' dark, ' ' light)
    small = cv2.resize(ctx.rgb, (cols, rows), interpolation=cv2.INTER_AREA)
    luma = 0.299 * small[:, :, 0] + 0.587 * small[:, :, 1] + 0.114 * small[:, :, 2]
    canvas = [['.' if luma[r, c] < 128 else ' ' for c in range(cols)] for r in range(rows)]

    def gx_to_c(gx: int) -> int:
        return max(0, min(cols - 1, int((gx - ox) * cols / max(1, w))))

    def gy_to_r(gy: int) -> int:
        return max(0, min(rows - 1, int((gy - oy) * rows / max(1, h))))

    # 1. Text bands ('=')
    for tb in ctx.artifacts.get('text_bands', {}).get('bands', []):
        x0, y0, x1, y1 = tb['global_bbox']
        r0, r1 = gy_to_r(y0), gy_to_r(max(y0, y1 - 1))
        c0, c1 = gx_to_c(x0), gx_to_c(max(x0, x1 - 1))
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                canvas[r][c] = '='

    # 2. Controls ('[###]' for accent controls, '[...]' for bordered boxes)
    for ctl in ctx.artifacts.get('controls', {}).get('controls', []):
        x0, y0, x1, y1 = ctl['global_bbox']
        r0, r1 = gy_to_r(y0), gy_to_r(max(y0, y1 - 1))
        c0, c1 = gx_to_c(x0), gx_to_c(max(x0, x1 - 1))
        fill = '#' if 'accent' in ctl['kind'] or ctl['kind'] == 'badge' else '+'
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                if c == c0 and c1 > c0:
                    canvas[r][c] = '['
                elif c == c1 and c1 > c0:
                    canvas[r][c] = ']'
                else:
                    canvas[r][c] = fill

    # 3. Structural dividers ('-' and '|' and '+')
    lay = ctx.artifacts.get('layout', {})
    for gy in lay.get('h_dividers_global_y', []):
        r = gy_to_r(gy)
        for c in range(cols):
            canvas[r][c] = '-'
    for gx in lay.get('v_dividers_global_x', []):
        c = gx_to_c(gx)
        for r in range(rows):
            canvas[r][c] = '+' if canvas[r][c] == '-' else '|'

    ctx.artifacts['ui_map'] = {
        'size': [cols, rows],
        'legend': 'bg: .=dark space=light | dividers: - | + | text_band: = | control: [#] accent / [+] box',
        'lines': [''.join(row) for row in canvas],
    }


@register_filter(
    'ui_diff',
    'Identity-based temporal UI diff (persistent K-tracks, global optimal matching, pixel regions, both frames preserved)',
    'ui_diff=path_to_before.png',
    inputs=('rgb', 'optional:layout', 'optional:controls', 'optional:ocr'),
    outputs=('ui_diff',),
    optional_deps=('tesseract', 'rapidocr-onnxruntime'),
    requires_network=False,
    requires_gpu=False,
    deterministic=True,
    coord_space='global',
    provenance='MIXED (DEMONSTRATED geometry/color/OCR + INFERRED STATE_CHANGED_CANDIDATE)',
    degradations='declares AMBIGUOUS_MATCH on tied candidates; rejects mismatched frame sizes',
)
def _f_ui_diff(ctx: FilterContext, params: dict[str, str]) -> None:
    ref_path = params.get('ref') or params.get('_0')
    if not ref_path:
        raise FilterDeny('ui_diff requires reference image path (ui_diff=/path/to/before.png)')
    ref_full, ref_sha, ref_name = _load_screen_rgb(ref_path)
    if (ref_full.shape[1], ref_full.shape[0]) != (ctx.full_w, ctx.full_h):
        raise FilterDeny(
            f'ui_diff requires same full size: ref {ref_full.shape[1]}x{ref_full.shape[0]} vs {ctx.full_w}x{ctx.full_h}'
        )
    ref_crop = ref_full[ctx.offset_y:ctx.offset_y + ctx.h, ctx.offset_x:ctx.offset_x + ctx.w]

    # Reuse current frame B layout/controls/ocr if already computed; run missing ones cleanly
    if 'layout' not in ctx.artifacts:
        _f_layout(ctx, {})
    if 'controls' not in ctx.artifacts:
        _f_controls(ctx, {})
    use_ocr = 'ocr' in ctx.artifacts

    # Extract Frame A (reference) with identical crop/offset and stages (never mutating Frame B!)
    ref_ctx = FilterContext(
        source_path=str(Path(ref_path).resolve()),
        source_name=ref_name,
        source_sha256=ref_sha,
        full_w=ctx.full_w,
        full_h=ctx.full_h,
        rgb=ref_crop,
        offset_x=ctx.offset_x,
        offset_y=ctx.offset_y,
        ocr_provider=ctx.ocr_provider,
    )
    _f_layout(ref_ctx, {})
    _f_controls(ref_ctx, {})
    if use_ocr:
        _f_ocr(ref_ctx, {'lang': ctx.artifacts['ocr'].get('lang', 'eng')})

    panels_a = ref_ctx.artifacts['layout']['panels']
    panels_b = ctx.artifacts['layout']['panels']
    controls_a = ref_ctx.artifacts['controls']['controls']
    controls_b = ctx.artifacts['controls']['controls']

    # Identity-based global optimal temporal matching (persistent track=K0..Kn)
    tracking = fdiff.match_controls_temporal(
        controls_a=controls_a,
        controls_b=controls_b,
        panels_a=panels_a,
        panels_b=panels_b,
        screen_size=(ctx.full_w, ctx.full_h),
        prev_track_map=ctx.prev_track_map,
    )

    # Pixel-level changed regions D0..Dn linked to persistent tracks & panels
    delta = np.mean(np.abs(ctx.rgb.astype(np.int16) - ref_crop.astype(np.int16)), axis=2)
    mask = (delta >= 12.0).astype(np.uint8) * 255
    changed_px = int(np.count_nonzero(mask))
    total_px = ctx.w * ctx.h

    k = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    merged = cv2.dilate(mask, k, iterations=1)
    cnts, _ = cv2.findContours(merged, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    track_by_b_local = tracking['frame_b_tracks']
    regions = []
    for cnt in cnts:
        x, y, bw, bh = cv2.boundingRect(cnt)
        if bw * bh < 64:
            continue
        sub_m = mask[y:y + bh, x:x + bw] > 0
        c_cnt = int(np.count_nonzero(sub_m))
        if c_cnt == 0:
            continue
        old_med = [int(np.median(ref_crop[y:y + bh, x:x + bw][sub_m, ch])) for ch in range(3)]
        new_med = [int(np.median(ctx.rgb[y:y + bh, x:x + bw][sub_m, ch])) for ch in range(3)]
        gb = ctx.to_global(x, y, x + bw, y + bh)
        in_p = next((p['id'] for p in panels_b if _box_contains(p['global_bbox'], gb, 0.55)), None)
        hit_w = [
            c['id'] for c in controls_b
            if _box_contains(gb, c['global_bbox'], 0.40) or _box_contains(c['global_bbox'], gb, 0.40)
        ]
        hit_tracks = [track_by_b_local[w_id] for w_id in hit_w if w_id in track_by_b_local]
        regions.append({
            'global_bbox': gb,
            'changed_pixels': c_cnt,
            'old_hex': _hex(old_med),
            'old_family': fb._hsv_family(*old_med),
            'new_hex': _hex(new_med),
            'new_family': fb._hsv_family(*new_med),
            'inside_panel': in_p,
            'overlapping_controls': hit_w[:4],
            'overlapping_tracks': hit_tracks[:4],
        })
    regions.sort(key=lambda r: -r['changed_pixels'])
    for i, r in enumerate(regions[:12]):
        r['id'] = f'D{i}'

    ctx.artifacts['ui_diff'] = {
        'frame_a': {
            'source_name': ref_name,
            'source_sha256': ref_sha,
            'control_count': len(controls_a),
            'controls': [fdiff._control_snapshot(c, tracking['frame_a_tracks'][c['id']]) for c in controls_a],
        },
        'frame_b': {
            'source_name': ctx.source_name,
            'source_sha256': ctx.source_sha256,
            'control_count': len(controls_b),
            'controls': [
                fdiff._control_snapshot(c, tracking['frame_b_tracks'].get(c['id'], 'AMBIGUOUS'))
                for c in controls_b
            ],
        },
        'reference_name': ref_name,
        'reference_sha256': ref_sha[:16],
        'changed_pixels': changed_px,
        'changed_pct': round(changed_px * 100.0 / total_px, 2),
        'tracking': tracking,
        'regions': regions[:12],
        'total_regions': len(regions),
    }


@register_filter(
    'blocks',
    'Attach M2 halves/quarters block partitioning + cross-seam stitch summary to the screen pipeline',
    'blocks[=quarters|halves_h|halves_v:colors=10]',
)
def _f_blocks(ctx: FilterContext, params: dict[str, str]) -> None:
    split = params.get('split') or params.get('_0') or 'quarters'
    k = _int_param(params, 'colors', 1, 10, 2, 24)
    sess = fb.map_image_blocks(
        ctx.source_path,
        split=split,
        palette_k=k,
        box=tuple(ctx.global_bbox),
    )
    st = fb.stitch_blocks(sess['session_id'])
    ctx.artifacts['blocks'] = {
        'session_id': sess['session_id'],
        'split': split,
        'quantization_error': sess['quantization_error'],
        'seams': st['seams'],
        'cross_seam_regions': [r for r in st['unified_regions'] if r['crosses_seam']][:8],
    }


# ============================================================================
# Pipeline execution & compact bounded formatting
# ============================================================================

def run_pipeline(
    image_path: str | Path,
    vf: str = 'theme,layout,controls,text_bands,ui_map=48:20',
    *,
    ocr_provider: Callable[..., dict[str, Any]] | None = None,
    vlm_policy: fvlm.VLMProbePolicy | None = None,
    vlm_backend: Callable[..., str] | None = None,
    prev_track_map: dict[str, str] | None = None,
) -> FilterContext:
    """Execute an ffmpeg-style filter graph over a screenshot (read-only)."""
    steps = parse_vf(vf)
    rgb, sha, name = _load_screen_rgb(image_path)
    ctx = FilterContext(
        source_path=str(Path(image_path).resolve()),
        source_name=name,
        source_sha256=sha,
        full_w=int(rgb.shape[1]),
        full_h=int(rgb.shape[0]),
        rgb=rgb,
        ocr_provider=ocr_provider,
        vlm_policy=vlm_policy or fvlm.DEFAULT_POLICY,
        vlm_backend=vlm_backend,
        prev_track_map=prev_track_map,
    )
    for fname, params in steps:
        t0 = time.perf_counter()
        FILTERS[fname].fn(ctx, params)
        ms = round((time.perf_counter() - t0) * 1000.0, 2)
        ctx.trace.append({'filter': fname, 'params': params, 'ms': ms})
    return ctx


def to_canonical_dict(ctx: FilterContext) -> dict[str, Any]:
    """Canonical structured screen model; compact text reports are views over this object."""
    return {
        'schema': SCHEMA_SCREEN,
        'source_name': ctx.source_name,
        'source_sha256': ctx.source_sha256,
        'source_frame': ctx.source_frame_id,
        'full_size': [ctx.full_w, ctx.full_h],
        'scope': ctx.global_bbox,
        'trace': ctx.trace,
        'artifacts': ctx.artifacts,
    }


def query_canonical(ctx: FilterContext, selector: str) -> str:
    """Bounded targeted query over canonical artifacts (e.g. 'P0', 'W1', 'K0', 'ambiguities', 'ocr', 'ui_diff')."""
    sel = selector.strip()
    if sel.upper().startswith('P') and sel[1:].isdigit():
        panels = ctx.artifacts.get('layout', {}).get('panels', [])
        p = next((x for x in panels if x['id'] == sel.upper()), None)
        if not p:
            raise FilterDeny(f'panel {sel!r} not found')
        ctls = [c for c in ctx.artifacts.get('controls', {}).get('controls', []) if c.get('inside_panel') == p['id']]
        lines = [
            f'PANEL {p["id"]} ({p["position_hint"]}) bbox={p["global_bbox"]} bg={p["bg_hex"]}({p["bg_family"]}) L={p["luma"]} area={p["area_pct"]}%',
            f'Controls inside {p["id"]} ({len(ctls)}):',
        ]
        for c in ctls:
            lines.append('  ' + _format_control_line(c))
        return '\n'.join(lines)

    if sel.upper().startswith('W') and sel[1:].isdigit():
        ctls = ctx.artifacts.get('controls', {}).get('controls', [])
        c = next((x for x in ctls if x['id'] == sel.upper()), None)
        if not c:
            raise FilterDeny(f'control {sel!r} not found')
        out = [_format_control_line(c)]
        if c.get('association_candidates'):
            out.append('AMBIGUITY CANDIDATES:')
            for a in c['association_candidates']:
                out.append(f'  ocr={a["ocr_id"]} text={a["text"]!r} reason={a["reason"]} cands={a["candidates"]}')
        return '\n'.join(out)

    if sel.upper().startswith('K') and sel[1:].isdigit():
        tr = ctx.artifacts.get('ui_diff', {}).get('tracking', {}).get('tracks', [])
        e = next((x for x in tr if x['track'] == sel.upper()), None)
        if not e:
            raise FilterDeny(f'track {sel!r} not found (run ui_diff first)')
        return json.dumps(e, indent=2, ensure_ascii=False)

    if sel.lower() == 'ambiguities':
        lines = []
        tca = ctx.artifacts.get('text_control_association', {})
        for a in tca.get('ambiguities', []):
            lines.append(f'OCR_ASSOC_AMBIGUOUS {a["ocr_id"]} text={a["text"]!r} reason={a["reason"]} cands={a["candidates"]}')
        for e in ctx.artifacts.get('ui_diff', {}).get('tracking', {}).get('tracks', []):
            if e['classification'] == fdiff.CLS_AMBIGUOUS:
                lines.append(
                    f'TRACK_AMBIGUOUS {e["track"]} a={e["frame_a_local"]} b={e["frame_b_local"]} '
                    f'competing={e.get("competing_candidates")}'
                )
        return '\n'.join(lines) if lines else 'NO AMBIGUITIES RECORDED'

    if sel.lower() in ctx.artifacts:
        return json.dumps(ctx.artifacts[sel.lower()], indent=2, ensure_ascii=False)[:6000]
    raise FilterDeny(f'unknown selector {selector!r}; use P<n>, W<n>, K<n>, ambiguities, or artifact key')


def _format_control_line(c: dict[str, Any]) -> str:
    txt_part = ''
    assoc = c.get('association')
    if assoc:
        if c.get('text') is not None:
            conf_s = f'(conf={c["text_confidence"]})' if c.get('text_confidence') is not None else ''
            txt_part = f' text={c["text"]!r} text_source={c["text_source"]}{conf_s} association={assoc}'
        else:
            txt_part = f' text=None text_source={c.get("text_source")} association={assoc}({c.get("association_reason")})'
    else:
        lbl = 'has_inner_text/icon' if c.get('has_inner_contrast') else 'flat'
        txt_part = f' [{lbl}]'
    return (
        f'{c["id"]:<3s} type={c["kind"]:<16s} bbox={c["global_bbox"]} click={tuple(c["click_center"])} '
        f'color={c["hex"]}({c["family"]}) panel={c["inside_panel"]}{txt_part}'
    )


def format_pipeline_report(ctx: FilterContext, max_chars: int = 6500, ocr_limit: int = 25) -> str:
    total_ms = round(sum(t['ms'] for t in ctx.trace), 2)
    chain = ' -> '.join(f'{t["filter"]}({t["ms"]}ms)' for t in ctx.trace)
    out = [
        f'FLOWSIGHT SCREEN v1 | {ctx.source_name} ({ctx.full_w}x{ctx.full_h}) | scope={ctx.global_bbox} | SHA256={ctx.source_sha256[:16]}',
        f'PIPELINE ({total_ms}ms): {chain}',
        'Coordinates: origin top-left (0,0); x right, y down; bboxes [x0,y0,x1,y1) EXCLUSIVE; click=(cx,cy). Local W-IDs are per-frame only.',
    ]

    if 'theme' in ctx.artifacts:
        th = ctx.artifacts['theme']
        surf = ', '.join(f'{s["hex"]}({s["family"]}):{s["pct"]}%' for s in th['dominant_surfaces'][:4])
        acc = ', '.join(f'{a["hex"]}({a["family"]}):{a["pct"]}%' for a in th['accent_colors'][:4]) or '(none)'
        out.append(
            f'THEME: polarity={th["polarity"]} median_luma={th["median_luma"]} '
            f'(dark={th["dark_pct"]}% light={th["light_pct"]}%)'
        )
        out.append(f'  surfaces: {surf}')
        out.append(f'  accents:  {acc}')

    if 'layout' in ctx.artifacts:
        lay = ctx.artifacts['layout']
        shown_p = len(lay['panels'])
        total_p = lay['total_panels_detected']
        out.append(
            f'LAYOUT (DEMONSTRATED dividers & panes): H_div_y={lay["h_dividers_global_y"]} '
            f'V_div_x={lay["v_dividers_global_x"]} | panels={shown_p}/{total_p}:'
        )
        for p in lay['panels']:
            out.append(
                f'  {p["id"]:<3s} {p["position_hint"]:<18s} bbox={p["global_bbox"]} '
                f'({p["size"][0]}x{p["size"][1]}, {p["area_pct"]}%) bg={p["bg_hex"]}({p["bg_family"]}) L={p["luma"]}'
            )
        if total_p > shown_p:
            out.append(f'  [TRUNCATED {total_p - shown_p} of {total_p} additional panels; raise max_panels or use crop=x:y:w:h]')

    if 'controls' in ctx.artifacts:
        ct = ctx.artifacts['controls']
        shown_c = len(ct['controls'])
        total_c = ct['total_detected']
        out.append(
            f'CONTROLS (DEMONSTRATED geometric candidates — appearance != verified interactivity, '
            f'showing={shown_c}/{total_c}, total={total_c}):'
        )
        for c in ct['controls']:
            out.append('  ' + _format_control_line(c))
        if total_c > shown_c:
            out.append(f'  [TRUNCATED {total_c - shown_c} of {total_c} additional controls; raise controls=max=N or use crop=x:y:w:h]')

    if 'text_control_association' in ctx.artifacts:
        tca = ctx.artifacts['text_control_association']
        if tca['ambiguities']:
            ambs = tca['ambiguities']
            out.append(f'AMBIGUOUS TEXT-CONTROL ASSOCIATIONS ({len(ambs)}, NO arbitrary winner chosen):')
            for a in ambs[:6]:
                c_str = ', '.join(f'{cand["control_id"]}({cand.get("relation","overlap")})' for cand in a['candidates'][:3])
                out.append(f'  {a["ocr_id"]} text={a["text"]!r} bbox={a["bbox"]} reason={a["reason"]} candidates=[{c_str}]')
            if len(ambs) > 6:
                out.append(f'  [TRUNCATED {len(ambs) - 6} of {len(ambs)} additional ambiguities]')

    if 'ocr' in ctx.artifacts:
        oc = ctx.artifacts['ocr']
        st = oc['status']
        if st in (focr.OCR_ABSENT, focr.OCR_ERROR):
            out.append(f'OCR: status={st} engine={oc.get("engine")} ({oc["detail"]})')
        else:
            items = oc.get('items', [])
            lim = max(1, min(400, int(ocr_limit)))
            out.append(
                f'OCR: status={st} engine={oc.get("engine")} lang={oc.get("lang")} '
                f'scale={oc.get("confidence_scale")} items={len(items)} ({oc.get("elapsed_ms")}ms):'
            )
            for ln in items[:lim]:
                out.append(f'  {ln["id"]:<3s} [{ln["status"]}] bbox={ln["bbox"]} conf={ln["confidence"]} text={ln["text"]!r}')
            if len(items) > lim:
                out.append(f'  [TRUNCATED {len(items) - lim} of {len(items)} additional OCR items; raise ocr_limit or use crop=x:y:w:h]')

    if 'ui_map' in ctx.artifacts:
        um = ctx.artifacts['ui_map']
        out.append(f'UI WIREFRAME MAP {um["size"][0]}x{um["size"][1]} ({um["legend"]}):')
        for ln in um['lines']:
            out.append('  ' + ln)

    if 'text_bands' in ctx.artifacts:
        tb = ctx.artifacts['text_bands']
        out.append(
            f'TEXT BANDS (GEOMETRIC ONLY, no OCR; showing {min(20, tb["shown"])}/{tb["total_detected"]}):'
        )
        for b in tb['bands'][:20]:
            loc = f'control={b["inside_control"]}' if b['inside_control'] else f'panel={b["inside_panel"]}'
            out.append(
                f'  {b["id"]:<3s} bbox={b["global_bbox"]} ({b["size"][0]}x{b["size"][1]}) '
                f'trans={b["transitions"]} contrast={b["contrast"]} {loc}'
            )
        if tb['total_detected'] > 20:
            out.append(f'  [TRUNCATED {tb["total_detected"] - 20} of {tb["total_detected"]} additional text bands; use crop=x:y:w:h to inspect]')

    if 'ui_diff' in ctx.artifacts:
        df = ctx.artifacts['ui_diff']
        tr = df['tracking']
        out.append(
            f'UI DIFF (IDENTITY-BASED) frame_a={df["frame_a"]["source_name"]}({df["frame_a"]["source_sha256"][:8]}) '
            f'-> frame_b={df["frame_b"]["source_name"]}({df["frame_b"]["source_sha256"][:8]}) | '
            f'changed_px={df["changed_pixels"]} ({df["changed_pct"]}%) | tracks={tr["summary_counts"]}'
        )
        out.append('TRACKED CONTROLS (persistent track=K<n> separate from per-frame local W<n>):')
        for e in tr['tracks']:
            inf = f' INFERRED={e["inferred_flags"]}' if e.get('inferred_flags') else ''
            flags_s = '+'.join(e['flags'])
            if e['classification'] == fdiff.CLS_NEW:
                fb_c = e['frame_b']
                txt = f' text={fb_c["text"]!r}' if fb_c.get('text') else ''
                out.append(
                    f'  track={e["track"]:<3s} local_b={e["frame_b_local"]:<3s} {flags_s} '
                    f'bbox={fb_c["bbox"]} color={fb_c["hex"]}({fb_c["family"]}) panel={fb_c["panel"]}{txt}'
                )
            elif e['classification'] == fdiff.CLS_REMOVED:
                fa_c = e['frame_a']
                txt = f' text={fa_c["text"]!r}' if fa_c.get('text') else ''
                out.append(
                    f'  track={e["track"]:<3s} local_a={e["frame_a_local"]:<3s} {flags_s} '
                    f'bbox={fa_c["bbox"]} color={fa_c["hex"]}({fa_c["family"]}) panel={fa_c["panel"]}{txt}'
                )
            elif e['classification'] == fdiff.CLS_AMBIGUOUS:
                out.append(
                    f'  track={e["track"]:<3s} local_a={e["frame_a_local"]} local_b={e["frame_b_local"]} '
                    f'AMBIGUOUS_MATCH score={e["affinity"]["score"]} competing={e["competing_candidates"]}'
                )
            else:
                extra = []
                if e.get('color_transition'):
                    extra.append(f'color={e["color_transition"]}')
                if e.get('text_transition'):
                    extra.append(f'text={e["text_transition"]}')
                if fdiff.CLS_MOVED in e['flags']:
                    extra.append(f'delta_center={tuple(e["delta_center_px"])}')
                if e['frame_b'].get('text'):
                    extra.append(f'text={e["frame_b"]["text"]!r}')
                ex_s = (' ' + ' '.join(extra)) if extra else ''
                out.append(
                    f'  track={e["track"]:<3s} local_a={e["frame_a_local"]:<3s}->local_b={e["frame_b_local"]:<3s} '
                    f'{flags_s} bbox={e["frame_b"]["bbox"]}{ex_s}{inf}'
                )
        out.append(f'PIXEL DIFF REGIONS ({len(df["regions"])}/{df["total_regions"]}):')
        for r in df['regions']:
            out.append(
                f'  {r["id"]:<3s} bbox={r["global_bbox"]} px={r["changed_pixels"]} '
                f'{r["old_hex"]}({r["old_family"]})->{r["new_hex"]}({r["new_family"]}) '
                f'panel={r["inside_panel"]} tracks={r["overlapping_tracks"]}'
            )

    if 'vlm_probe' in ctx.artifacts:
        vp = ctx.artifacts['vlm_probe']
        out.append(
            f'VLM_PROBE [{vp["status"]}] provenance={vp["provenance"]} bbox={vp["global_bbox"]}: {vp["detail"]}'
        )
        if vp.get('description'):
            out.append(f'  VLM_INFERRED_TEXT: {vp["description"]}')

    if 'blocks' in ctx.artifacts:
        bk = ctx.artifacts['blocks']
        seams = ', '.join(f'{s["pair"]}:{s["continuity_pct"]}%' for s in bk['seams'])
        out.append(
            f'BLOCKS session={bk["session_id"]} split={bk["split"]} '
            f'PSNR={bk["quantization_error"]["psnr_db"]}dB seams=[{seams}] '
            f'cross_seam={len(bk["cross_seam_regions"])}'
        )

    text = '\n'.join(out)
    if len(text) > max_chars:
        suf = '\n[TRUNCATED: narrow scope with crop=x:y:w:h or select fewer filters in -vf]'
        budget = max(0, max_chars - len(suf))
        kept_lines: list[str] = []
        used = 0
        for ln in out:
            add_len = len(ln) + (1 if kept_lines else 0)
            if used + add_len > budget:
                break
            kept_lines.append(ln)
            used += add_len
        if kept_lines:
            text = '\n'.join(kept_lines) + suf
        else:
            text = text[:budget] + suf
    return text


def format_filter_catalog() -> str:
    lines = ['FLOWSIGHT SCREEN FILTERS (ffmpeg-style -vf "f1=...,f2,..."):']
    for name in sorted(FILTERS):
        sp = FILTERS[name]
        net = 'net' if sp.requires_network else 'local'
        gpu = 'gpu' if sp.requires_gpu else 'cpu'
        det = 'det' if sp.deterministic else 'nondet'
        deps = ','.join(sp.optional_deps) if sp.optional_deps else 'none'
        lines.append(
            f'  {sp.name:<12s} | {sp.syntax:<44s} | [{net}/{gpu}/{det} deps={deps} {sp.coord_space}] {sp.summary}'
        )
    return '\n'.join(lines)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description='FlowSight Screen: ffmpeg-style filter pipeline for screenshots')
    p.add_argument('-i', '--input', help='Screenshot image path (PNG/JPG/WEBP)')
    p.add_argument('-vf', default='theme,layout,controls,text_bands,ui_map=48:20',
                   help='Comma-separated filter chain (default: theme,layout,controls,text_bands,ui_map=48:20)')
    p.add_argument('--json', action='store_true', help='Emit structured JSON instead of compact text')
    p.add_argument('--max-chars', type=int, default=6500)
    p.add_argument('--ocr-limit', type=int, default=25, help='Maximum OCR items printed in report (default: 25, max: 400)')
    p.add_argument('--filters', action='store_true', help='List available filters and exit')
    p.add_argument('--enable-vlm', action='store_true', help='Opt-in enable crop-only vlm_probe (or set FLOWSIGHT_VLM_ENABLED=1)')
    p.add_argument('--vlm-url', default=None, help='OpenAI-compatible VLM endpoint URL (or FLOWSIGHT_VLM_URL)')
    p.add_argument('--vlm-model', default=None, help='VLM model name (or FLOWSIGHT_VLM_MODEL)')
    p.add_argument('--vlm-allow-remote', action='store_true', help='Allow non-loopback VLM URL (or FLOWSIGHT_VLM_ALLOW_REMOTE=1)')
    p.add_argument('--vlm-timeout', type=float, default=None, help='VLM HTTP timeout seconds (or FLOWSIGHT_VLM_TIMEOUT_S)')
    args = p.parse_args(argv)

    if args.filters:
        print(format_filter_catalog())
        return 0
    if not args.input:
        p.error('missing -i/--input (or use --filters)')
    try:
        vlm_policy, vlm_backend = fvlm.resolve_runtime_vlm_config(
            enable=True if args.enable_vlm else None,
            url=args.vlm_url,
            model=args.vlm_model,
            allow_remote=True if args.vlm_allow_remote else None,
            timeout_s=args.vlm_timeout,
        )
        ctx = run_pipeline(args.input, vf=args.vf, vlm_policy=vlm_policy, vlm_backend=vlm_backend)
        if args.json:
            payload = {
                'schema': SCHEMA_SCREEN,
                'source_name': ctx.source_name,
                'source_sha256': ctx.source_sha256,
                'full_size': [ctx.full_w, ctx.full_h],
                'scope': ctx.global_bbox,
                'trace': ctx.trace,
                'artifacts': ctx.artifacts,
            }
            print(json.dumps(payload, indent=2, ensure_ascii=False))
        else:
            print(format_pipeline_report(ctx, max_chars=args.max_chars, ocr_limit=args.ocr_limit))
        return 0
    except (FilterDeny, fb.BlocksDeny, OSError) as e:
        print(f'DENY {e}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
