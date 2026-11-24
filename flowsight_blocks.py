#!/usr/bin/env python3
"""FlowSight M2 — High-resolution block partitioning (halves/quarters) & stitching.

Designed for large images / photos / screenshots (e.g. 1920x1080, >64 colors)
where lossless M0/M1 pixel-exact indexing is rejected by design.

Epistemic contract:
  - EXPLICITLY APPROXIMATE / QUANTIZED (`mode=photo-blocks-v1`): never claims
    pixel-exactness; always reports measured quantization error (MAE, RMSE,
    PSNR, % within delta<=24).
  - Separates MEASURED geometric/chromatic/seam evidence (`DEMONSTRATED`) from
    text-only agent notes aligned during stitching (`AGENT_NOTES / INFERRED`).
  - Read-only over source images; deterministic given identical input & params.
"""
from __future__ import annotations

import colorsys
import hashlib
import json
import math
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
from PIL import Image, UnidentifiedImageError

SCHEMA_BLOCKS = 'flowsight-blocks-v1'
MAX_PHOTO_DIM = 8192
MAX_PALETTE_K = 24
MIN_PALETTE_K = 2
SYMBOLS = '0123456789ABCDEFGHJKLMNPQRSTUVWXYZ'
VALID_SPLITS = ('quarters', 'halves_h', 'halves_v')

# Deterministic analysis resolution per block (keeps 1920x1080 fast <200ms)
ANALYSIS_W = 96
ANALYSIS_H = 56
SEAM_BANDS = 16


class BlocksDeny(ValueError):
    pass


def _hsv_family(r: int, g: int, b: int) -> str:
    """Objective HSV color-family tag (pure math, NEVER an object label)."""
    h, s, v = colorsys.rgb_to_hsv(r / 255.0, g / 255.0, b / 255.0)
    if v < 0.14:
        return 'very-dark'
    if s < 0.14:
        if v > 0.85:
            return 'near-white'
        if v > 0.55:
            return 'light-gray'
        return 'dark-gray'
    deg = h * 360.0
    if deg < 18 or deg >= 342:
        hue = 'red'
    elif deg < 42:
        hue = 'orange'
    elif deg < 68:
        hue = 'yellow'
    elif deg < 160:
        hue = 'green'
    elif deg < 200:
        hue = 'cyan'
    elif deg < 258:
        hue = 'blue'
    elif deg < 305:
        hue = 'violet'
    else:
        hue = 'magenta'
    if v < 0.35:
        return f'dark-{hue}'
    if v > 0.80 and s < 0.45:
        return f'pale-{hue}'
    return hue


def _hex_rgb(rgb: tuple[int, int, int]) -> str:
    return f'#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}'


def _default_session_dir() -> Path:
    d = Path(tempfile.gettempdir()) / 'flowsight_sessions'
    d.mkdir(parents=True, exist_ok=True)
    return d


def _load_image_rgb(path: str | Path) -> tuple[Image.Image, str]:
    p = Path(path)
    if not p.is_file():
        raise BlocksDeny(f'image not found: {p}')
    raw_bytes = p.read_bytes()
    sha = hashlib.sha256(raw_bytes).hexdigest()
    try:
        with Image.open(p) as im:
            w, h = im.size
            if not (2 <= w <= MAX_PHOTO_DIM and 2 <= h <= MAX_PHOTO_DIM):
                raise BlocksDeny(f'image {w}x{h} outside 2..{MAX_PHOTO_DIM}')
            rgb = im.convert('RGB')
            rgb.load()
            return rgb, sha
    except (OSError, UnidentifiedImageError, Image.DecompressionBombError) as e:
        raise BlocksDeny(f'cannot decode image: {e}') from e


def _build_global_palette(crop_img: Image.Image, palette_k: int) -> tuple[np.ndarray, dict[str, float]]:
    """Deterministic median-cut global palette + measured quantization error."""
    if not (MIN_PALETTE_K <= palette_k <= MAX_PALETTE_K):
        raise BlocksDeny(f'palette_k must be in {MIN_PALETTE_K}..{MAX_PALETTE_K}')

    w, h = crop_img.size
    # Build palette on deterministic downscaled sample (<= 480x270) for speed,
    # then measure exact nearest-color error on a deterministic evaluation grid.
    sample_w = min(w, 480)
    sample_h = min(h, 270)
    sample = crop_img.resize((sample_w, sample_h), Image.Resampling.BILINEAR)
    q = sample.quantize(colors=palette_k, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    raw_pal = q.getpalette() or []
    used_indices = sorted(int(v) for v in np.unique(np.asarray(q)))
    colors = []
    for idx in used_indices:
        colors.append((raw_pal[idx * 3], raw_pal[idx * 3 + 1], raw_pal[idx * 3 + 2]))
    if not colors:
        colors = [(0, 0, 0)]
    pal_f = np.array(colors, dtype=np.float32)  # (K, 3)

    # Deterministic Lloyd k-means refinement (4 passes) on subsampled pixels
    # so high-contrast minority colors (e.g. white bars, bright disks, text) are preserved.
    arr = np.asarray(sample, dtype=np.float32)  # (H, W, 3)
    flat = arr.reshape(-1, 3)
    sub = flat[::4] if len(flat) > 16000 else flat
    for _ in range(4):
        d2 = np.sum((sub[:, None, :] - pal_f[None, :, :]) ** 2, axis=2)
        lbl = np.argmin(d2, axis=1)
        new_pal = pal_f.copy()
        for ci in range(len(pal_f)):
            mask = lbl == ci
            if np.any(mask):
                new_pal[ci] = np.mean(sub[mask], axis=0)
        pal_f = new_pal
    # Deduplicate rounded uint8 centroids
    uniq = []
    seen_rgb = set()
    for row in np.clip(np.round(pal_f), 0, 255).astype(np.int32):
        t = (int(row[0]), int(row[1]), int(row[2]))
        if t not in seen_rgb:
            seen_rgb.add(t)
            uniq.append(t)
    pal_arr = np.array(uniq, dtype=np.int32)

    # Assign every pixel of sample to nearest palette color (deterministic, int32 to avoid sq overflow)
    flat_i = flat.astype(np.int32)
    dists = np.sum((flat_i[:, None, :] - pal_arr[None, :, :]) ** 2, axis=2)
    assign = np.argmin(dists, axis=1)

    # Sort palette deterministically by descending pixel count, then RGB tuple
    counts = np.bincount(assign, minlength=len(pal_arr))
    order = sorted(range(len(pal_arr)), key=lambda i: (-int(counts[i]), tuple(int(v) for v in pal_arr[i])))
    pal_sorted = pal_arr[order]

    # Recompute exact error metrics against sample
    inv_map = np.zeros(len(order), dtype=np.int32)
    for new_i, old_i in enumerate(order):
        inv_map[old_i] = new_i
    assign_sorted = inv_map[assign]
    recon = pal_sorted[assign_sorted]
    diff = np.abs(flat_i - recon).astype(np.float64)
    mae = float(np.mean(diff))
    mse = float(np.mean(diff ** 2))
    rmse = math.sqrt(mse)
    psnr = 99.9 if mse < 1e-9 else 20.0 * math.log10(255.0 / rmse)
    pix_l1 = np.mean(diff, axis=1)
    within_24 = float(np.mean(pix_l1 <= 24.0) * 100.0)

    metrics = {
        'sample_size': [sample_w, sample_h],
        'mae_rgb': round(mae, 2),
        'rmse_rgb': round(rmse, 2),
        'psnr_db': round(psnr, 2),
        'within_delta24_pct': round(within_24, 1),
    }
    return pal_sorted.astype(np.uint8), metrics


def _quantize_array(arr_u8: np.ndarray, pal_u8: np.ndarray) -> np.ndarray:
    """Map (H, W, 3) uint8 RGB image to (H, W) palette indices using exact L2."""
    h, w, _ = arr_u8.shape
    flat = arr_u8.reshape(-1, 3).astype(np.int32)
    pal = pal_u8.astype(np.int32)
    dists = np.sum((flat[:, None, :] - pal[None, :, :]) ** 2, axis=2)
    return np.argmin(dists, axis=1).reshape(h, w).astype(np.uint8)


def _sobel_edges(arr_u8: np.ndarray) -> tuple[np.ndarray, str, float]:
    """Compute Sobel gradient magnitude on luminance + dominant orientation."""
    luma = (0.299 * arr_u8[:, :, 0] + 0.587 * arr_u8[:, :, 1] + 0.114 * arr_u8[:, :, 2]).astype(np.float32)
    if luma.shape[0] < 3 or luma.shape[1] < 3:
        return np.zeros_like(luma), 'flat', 0.0
    p = np.pad(luma, 1, mode='edge')
    gx = (
        -p[:-2, :-2] + p[:-2, 2:]
        - 2.0 * p[1:-1, :-2] + 2.0 * p[1:-1, 2:]
        - p[2:, :-2] + p[2:, 2:]
    ) / 4.0
    gy = (
        -p[:-2, :-2] - 2.0 * p[:-2, 1:-1] - p[:-2, 2:]
        + p[2:, :-2] + 2.0 * p[2:, 1:-1] + p[2:, 2:]
    ) / 4.0
    mag = np.hypot(gx, gy)
    strong = mag >= 22.0
    edge_density = float(np.mean(strong) * 100.0)
    if edge_density < 1.5:
        return mag, 'flat', edge_density
    # Orientation of strong edges: gradient perpendicular to edge line
    # |gy| > |gx| -> horizontal edge; |gx| > |gy| -> vertical edge
    ax = float(np.sum(np.abs(gx[strong])))
    ay = float(np.sum(np.abs(gy[strong])))
    ratio = ay / max(ax, 1e-6)
    if ratio > 1.45:
        orient = 'horizontal'
    elif ratio < 1.0 / 1.45:
        orient = 'vertical'
    else:
        # Check diagonal bias
        d1 = float(np.sum(np.abs(gx[strong] + gy[strong])))
        d2 = float(np.sum(np.abs(gx[strong] - gy[strong])))
        orient = 'diagonal' if max(d1, d2) / max(min(d1, d2), 1e-6) > 1.25 else 'isotropic'
    return mag, orient, edge_density


def _split_boxes(box: tuple[int, int, int, int], split: str) -> list[tuple[str, tuple[int, int, int, int]]]:
    x0, y0, x1, y1 = box
    mx = (x0 + x1) // 2
    my = (y0 + y1) // 2
    if split == 'quarters':
        return [
            ('TL', (x0, y0, mx, my)),
            ('TR', (mx, y0, x1, my)),
            ('BL', (x0, my, mx, y1)),
            ('BR', (mx, my, x1, y1)),
        ]
    if split == 'halves_h':
        return [
            ('L', (x0, y0, mx, y1)),
            ('R', (mx, y0, x1, y1)),
        ]
    if split == 'halves_v':
        return [
            ('T', (x0, y0, x1, my)),
            ('B', (x0, my, x1, y1)),
        ]
    raise BlocksDeny(f'invalid split {split!r}; allowed: {VALID_SPLITS}')


def _seam_pairs(split: str) -> list[tuple[str, str, str]]:
    """Return list of (block_a, block_b, orientation) where orientation is 'V' (A left of B) or 'H' (A above B)."""
    if split == 'quarters':
        return [
            ('TL', 'TR', 'V'),
            ('BL', 'BR', 'V'),
            ('TL', 'BL', 'H'),
            ('TR', 'BR', 'H'),
        ]
    if split == 'halves_h':
        return [('L', 'R', 'V')]
    if split == 'halves_v':
        return [('T', 'B', 'H')]
    return []


def _extract_block_regions(
    q_grid: np.ndarray,
    edge_mag: np.ndarray,
    bbox: tuple[int, int, int, int],
    max_regions: int = 6,
) -> list[dict[str, Any]]:
    """4-connected quantized regions projected to exact global coordinates + seam touches."""
    ah, aw = q_grid.shape
    bx0, by0, bx1, by1 = bbox
    bw = bx1 - bx0
    bh = by1 - by0
    seen = np.zeros((ah, aw), dtype=np.uint8)
    comps = []
    flat_q = q_grid.reshape(-1)
    flat_e = edge_mag.reshape(-1)

    for start in range(ah * aw):
        sy, sx = divmod(start, aw)
        if seen[sy, sx]:
            continue
        cid = int(flat_q[start])
        seen[sy, sx] = 1
        stack = [(sy, sx)]
        n = 0
        xmin, ymin, xmax, ymax = aw, ah, 0, 0
        e_sum = 0.0
        touches = set()
        while stack:
            y, x = stack.pop()
            n += 1
            e_sum += float(flat_e[y * aw + x] >= 22.0)
            if x < xmin:
                xmin = x
            if y < ymin:
                ymin = y
            if x + 1 > xmax:
                xmax = x + 1
            if y + 1 > ymax:
                ymax = y + 1
            if x == 0:
                touches.add('LEFT')
            if x == aw - 1:
                touches.add('RIGHT')
            if y == 0:
                touches.add('TOP')
            if y == ah - 1:
                touches.add('BOTTOM')
            for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= ny < ah and 0 <= nx < aw and not seen[ny, nx] and int(q_grid[ny, nx]) == cid:
                    seen[ny, nx] = 1
                    stack.append((ny, nx))
        if n * 100 < ah * aw * 2:  # skip tiny <2% noise specks in approximate mode
            continue
        gx0 = bx0 + int(round(xmin * bw / aw))
        gy0 = by0 + int(round(ymin * bh / ah))
        gx1 = bx0 + int(round(xmax * bw / aw))
        gy1 = by0 + int(round(ymax * bh / ah))
        comps.append({
            'cid': cid,
            'global_bbox': [gx0, gy0, max(gx0 + 1, gx1), max(gy0 + 1, gy1)],
            'area_pct': round(n * 100.0 / (ah * aw), 1),
            'edge_pct': round(e_sum * 100.0 / n, 1),
            'touches': sorted(touches),
        })

    comps.sort(key=lambda r: (-r['area_pct'], r['global_bbox'][1], r['global_bbox'][0], r['cid']))
    return comps[:max_regions]


def _analyze_block(
    img: Image.Image,
    block_id: str,
    bbox: tuple[int, int, int, int],
    pal_u8: np.ndarray,
    grid_w: int,
    grid_h: int,
) -> dict[str, Any]:
    x0, y0, x1, y1 = bbox
    crop = img.crop((x0, y0, x1, y1))
    aw = min(x1 - x0, ANALYSIS_W)
    ah = min(y1 - y0, ANALYSIS_H)
    small = crop.resize((aw, ah), Image.Resampling.BILINEAR)
    arr = np.asarray(small, dtype=np.uint8)
    q = _quantize_array(arr, pal_u8)
    edge_mag, orient, edge_density = _sobel_edges(arr)

    luma = 0.299 * arr[:, :, 0] + 0.587 * arr[:, :, 1] + 0.114 * arr[:, :, 2]
    mean_luma = float(np.mean(luma))
    std_luma = float(np.std(luma))

    # Top local colors (using global palette IDs)
    counts = np.bincount(q.reshape(-1), minlength=len(pal_u8))
    total_px = aw * ah
    top_colors = []
    for cid in sorted(range(len(pal_u8)), key=lambda i: (-int(counts[i]), i)):
        if counts[cid] == 0:
            continue
        pct = round(int(counts[cid]) * 100.0 / total_px, 1)
        if pct < 1.0 and len(top_colors) >= 4:
            continue
        top_colors.append({'cid': cid, 'pct': pct})
        if len(top_colors) >= 6:
            break

    # Compact color & edge maps (grid_w x grid_h)
    gw = max(4, min(grid_w, aw, 40))
    gh = max(4, min(grid_h, ah, 24))
    color_rows = []
    edge_rows = []
    for ty in range(gh):
        ry0 = ty * ah // gh
        ry1 = max(ry0 + 1, (ty + 1) * ah // gh)
        c_line = []
        e_line = []
        for tx in range(gw):
            rx0 = tx * aw // gw
            rx1 = max(rx0 + 1, (tx + 1) * aw // gw)
            tile_q = q[ry0:ry1, rx0:rx1].reshape(-1)
            bc = np.bincount(tile_q, minlength=len(pal_u8))
            best = int(np.argmax(bc))
            if bc[best] * 100 >= tile_q.size * 55:
                c_line.append(SYMBOLS[best])
            else:
                c_line.append('?')
            em = float(np.mean(edge_mag[ry0:ry1, rx0:rx1]))
            if em < 8.0:
                e_line.append('.')
            elif em < 20.0:
                e_line.append(':')
            elif em < 42.0:
                e_line.append('+')
            else:
                e_line.append('#')
        color_rows.append(''.join(c_line))
        edge_rows.append(''.join(e_line))

    regions = _extract_block_regions(q, edge_mag, bbox, max_regions=6)

    # Exact pixel seam strips sampled from the real full-res crop border (2px strip)
    def _strip_profile(strip_img: Image.Image, vertical_strip: bool) -> list[dict[str, Any]]:
        # vertical_strip=True -> left/right border (height varies along y)
        sarr = np.asarray(strip_img, dtype=np.uint8)
        sq = _quantize_array(sarr, pal_u8)
        axis_len = sarr.shape[0] if vertical_strip else sarr.shape[1]
        out = []
        for i in range(SEAM_BANDS):
            a = i * axis_len // SEAM_BANDS
            b = max(a + 1, (i + 1) * axis_len // SEAM_BANDS)
            sub_rgb = sarr[a:b, :, :] if vertical_strip else sarr[:, a:b, :]
            sub_q = sq[a:b, :].reshape(-1) if vertical_strip else sq[:, a:b].reshape(-1)
            mrgb = [round(float(np.mean(sub_rgb[:, :, c])), 1) for c in range(3)]
            bc = np.bincount(sub_q, minlength=len(pal_u8))
            dom = int(np.argmax(bc))
            out.append({'cid': dom, 'rgb': mrgb})
        return out

    bw = x1 - x0
    bh = y1 - y0
    sw = min(2, bw)
    sh = min(2, bh)
    seams = {
        'LEFT': _strip_profile(crop.crop((0, 0, sw, bh)), True),
        'RIGHT': _strip_profile(crop.crop((bw - sw, 0, bw, bh)), True),
        'TOP': _strip_profile(crop.crop((0, 0, bw, sh)), False),
        'BOTTOM': _strip_profile(crop.crop((0, bh - sh, bw, bh)), False),
    }

    return {
        'id': block_id,
        'bbox': [x0, y0, x1, y1],
        'size': [bw, bh],
        'mean_luma': round(mean_luma, 1),
        'std_luma': round(std_luma, 1),
        'edge_density_pct': round(edge_density, 1),
        'dominant_orientation': orient,
        'top_colors': top_colors,
        'grid_size': [gw, gh],
        'color_map': color_rows,
        'edge_map': edge_rows,
        'regions': regions,
        'seams': seams,
    }


def map_image_blocks(
    image_path: str | Path,
    split: str = 'quarters',
    palette_k: int = 12,
    grid_w: int = 24,
    grid_h: int = 14,
    box: tuple[int, int, int, int] | None = None,
    session_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Partition an image (e.g. 1920x1080) into halves or quarters with shared global palette."""
    if split not in VALID_SPLITS:
        raise BlocksDeny(f'invalid split {split!r}; expected one of {VALID_SPLITS}')
    img, sha = _load_image_rgb(image_path)
    full_w, full_h = img.size

    if box is None:
        root_box = (0, 0, full_w, full_h)
    else:
        bx0, by0, bx1, by1 = box
        if not (0 <= bx0 < bx1 <= full_w and 0 <= by0 < by1 <= full_h):
            raise BlocksDeny(f'box {box} outside image {full_w}x{full_h}')
        if (bx1 - bx0) < 4 or (by1 - by0) < 4:
            raise BlocksDeny(f'box {box} too small to split (<4px)')
        root_box = (bx0, by0, bx1, by1)

    crop_root = img.crop(root_box)
    pal_u8, err_metrics = _build_global_palette(crop_root, palette_k)

    # Global palette coverage on downscaled root
    sw, sh = err_metrics['sample_size']
    sample_arr = np.asarray(crop_root.resize((sw, sh), Image.Resampling.BILINEAR), dtype=np.uint8)
    sample_q = _quantize_array(sample_arr, pal_u8)
    g_counts = np.bincount(sample_q.reshape(-1), minlength=len(pal_u8))
    palette_info = []
    for cid in range(len(pal_u8)):
        r, g, b = int(pal_u8[cid, 0]), int(pal_u8[cid, 1]), int(pal_u8[cid, 2])
        luma = round(0.299 * r + 0.587 * g + 0.114 * b, 1)
        palette_info.append({
            'cid': cid,
            'symbol': SYMBOLS[cid],
            'rgb': [r, g, b],
            'hex': _hex_rgb((r, g, b)),
            'luma': luma,
            'family': _hsv_family(r, g, b),
            'pct': round(int(g_counts[cid]) * 100.0 / sample_q.size, 1),
        })

    blocks = {}
    block_order = []
    for bid, bbox in _split_boxes(root_box, split):
        blocks[bid] = _analyze_block(img, bid, bbox, pal_u8, grid_w, grid_h)
        block_order.append(bid)

    key_str = f'{sha}:{split}:{palette_k}:{grid_w}x{grid_h}:{root_box}'
    session_id = hashlib.sha256(key_str.encode('utf-8')).hexdigest()[:16]

    session = {
        'schema': SCHEMA_BLOCKS,
        'mode': 'photo-blocks-v1 (APPROXIMATE QUANTIZED, NOT lossless)',
        'session_id': session_id,
        'source_path': str(Path(image_path).resolve()),
        'source_name': Path(image_path).name,
        'source_sha256': sha,
        'full_size': [full_w, full_h],
        'root_box': list(root_box),
        'split': split,
        'palette_k': len(palette_info),
        'quantization_error': err_metrics,
        'palette': palette_info,
        'block_order': block_order,
        'blocks': blocks,
    }

    sdir = Path(session_dir) if session_dir else _default_session_dir()
    sdir.mkdir(parents=True, exist_ok=True)
    spath = sdir / f'{session_id}.json'
    spath.write_text(json.dumps(session, indent=2, ensure_ascii=False), encoding='utf-8')
    return session


def load_session(session_id: str, session_dir: str | Path | None = None) -> dict[str, Any]:
    sdir = Path(session_dir) if session_dir else _default_session_dir()
    p = Path(session_id)
    if p.suffix == '.json' and p.is_file():
        data = json.loads(p.read_text(encoding='utf-8'))
    else:
        safe = ''.join(c for c in session_id if c.isalnum() or c in ('-', '_'))
        sp = sdir / f'{safe}.json'
        if not sp.is_file():
            raise BlocksDeny(f'session not found: {session_id!r}')
        data = json.loads(sp.read_text(encoding='utf-8'))
    if data.get('schema') != SCHEMA_BLOCKS:
        raise BlocksDeny('invalid session schema')
    return data


def format_map_summary(session: dict[str, Any]) -> str:
    """Compact L0 global summary after partitioning into blocks."""
    fw, fh = session['full_size']
    rb = session['root_box']
    err = session['quantization_error']
    out = [
        f'FLOWSIGHT BLOCKS v1 | session={session["session_id"]} | image={session["source_name"]} ({fw}x{fh})',
        f'Scope: global bbox={rb} (origin top-left (0,0), x right, y down, end EXCLUSIVE) | split={session["split"]}',
        (
            f'APPROXIMATE QUANTIZED (K={session["palette_k"]}): '
            f'MAE_RGB={err["mae_rgb"]} RMSE_RGB={err["rmse_rgb"]} PSNR={err["psnr_db"]}dB '
            f'within_d24={err["within_delta24_pct"]}% (measured on {err["sample_size"][0]}x{err["sample_size"][1]})'
        ),
        'Shared Global Palette (HSV family = math tag, NOT semantic object identity):',
    ]
    for p in session['palette']:
        out.append(
            f'  {p["symbol"]}=C{p["cid"]:<2d} {p["hex"]} {p["family"]:<14s} L={p["luma"]:5.1f} ({p["pct"]:4.1f}%)'
        )
    out.append(f'Blocks ({len(session["block_order"])}):')
    for bid in session['block_order']:
        b = session['blocks'][bid]
        top3 = ', '.join(f'C{c["cid"]}:{c["pct"]:.0f}%' for c in b['top_colors'][:3])
        out.append(
            f'  {bid:<3s} bbox={b["bbox"]} ({b["size"][0]}x{b["size"][1]}) '
            f'luma={b["mean_luma"]:5.1f}±{b["std_luma"]:4.1f} '
            f'edges={b["edge_density_pct"]:4.1f}% ({b["dominant_orientation"]}) '
            f'top=[{top3}] regions={len(b["regions"])}'
        )
    out.append('Next tools: flowsight_get_block(session_id, block_id), flowsight_subdivide_block(...), flowsight_stitch_blocks(session_id, block_notes)')
    return '\n'.join(out)


def format_block_detail(session: dict[str, Any], block_id: str, include_edge_map: bool = True) -> str:
    """Compact detailed report for a single block (for isolated / parallel agent inspection)."""
    bid = block_id.strip().upper()
    if bid not in session['blocks']:
        raise BlocksDeny(f'unknown block_id {block_id!r}; available: {session["block_order"]}')
    b = session['blocks'][bid]
    pal_by_id = {p['cid']: p for p in session['palette']}
    gw, gh = b['grid_size']

    out = [
        f'BLOCK {bid} | session={session["session_id"]} | global_bbox={b["bbox"]} ({b["size"][0]}x{b["size"][1]} of {session["full_size"][0]}x{session["full_size"][1]})',
        f'STATS: mean_luma={b["mean_luma"]} std_luma={b["std_luma"]} edge_density={b["edge_density_pct"]}% orient={b["dominant_orientation"]}',
        'TOP COLORS (shared global palette): ' + ', '.join(
            f'{pal_by_id[c["cid"]]["symbol"]}=C{c["cid"]}({pal_by_id[c["cid"]]["hex"]},{pal_by_id[c["cid"]]["family"]}):{c["pct"]}%'
            for c in b['top_colors']
        ),
        f'COLOR MAP {gw}x{gh} (approximate cell majority >=55%, else ?):',
    ]
    for row in b['color_map']:
        out.append('  ' + row)
    if include_edge_map:
        out.append(f'EDGE MAP {gw}x{gh} (Sobel gradient: .=flat :=soft +=medium #=strong):')
        for row in b['edge_map']:
            out.append('  ' + row)
    out.append(f'LOCAL REGIONS ({len(b["regions"])}, projected to global [x0,y0,x1,y1)):')
    for i, r in enumerate(b['regions']):
        p = pal_by_id[r['cid']]
        t = ','.join(r['touches']) if r['touches'] else 'none'
        out.append(
            f'  {bid}.R{i}: C{r["cid"]}({p["symbol"]},{p["family"]},{p["hex"]}) '
            f'bbox={r["global_bbox"]} area={r["area_pct"]}% edges={r["edge_pct"]}% touches=[{t}]'
        )
    return '\n'.join(out)


def subdivide_block(
    session_id: str,
    block_id: str,
    split: str = 'quarters',
    palette_k: int | None = None,
    grid_w: int = 24,
    grid_h: int = 14,
    session_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Recursively split an existing block (e.g. TL 960x540 -> 4x 480x270) preserving global coords."""
    parent = load_session(session_id, session_dir)
    bid = block_id.strip().upper()
    if bid not in parent['blocks']:
        raise BlocksDeny(f'unknown block_id {block_id!r}; available: {parent["block_order"]}')
    bbox = tuple(parent['blocks'][bid]['bbox'])
    k = palette_k if palette_k is not None else parent['palette_k']
    child = map_image_blocks(
        parent['source_path'],
        split=split,
        palette_k=k,
        grid_w=grid_w,
        grid_h=grid_h,
        box=bbox,
        session_dir=session_dir,
    )
    child['parent_session_id'] = parent['session_id']
    child['parent_block_id'] = bid
    sdir = Path(session_dir) if session_dir else _default_session_dir()
    (sdir / f'{child["session_id"]}.json').write_text(
        json.dumps(child, indent=2, ensure_ascii=False), encoding='utf-8'
    )
    return child


def stitch_blocks(
    session_id: str,
    block_notes: dict[str, str] | None = None,
    session_dir: str | Path | None = None,
    seam_delta_thresh: float = 32.0,
) -> dict[str, Any]:
    """Structurally & semantically stitch analyzed blocks back into one global model.

    Combines:
      1. Measured pixel-adjacent seam continuity across shared block borders.
      2. Cross-seam Union-Find merging of compatible regions into global G0..Gn.
      3. Re-assembled global color + edge maps with explicit seam markers.
      4. Aligned per-block agent observations (`block_notes`), explicitly
         separated as INFERRED notes vs DEMONSTRATED seam/geometry measurements.
    """
    session = load_session(session_id, session_dir)
    pal_by_id = {p['cid']: p for p in session['palette']}
    blocks = session['blocks']
    split = session['split']

    # 1. Measure real seam continuity between every adjacent block pair
    seam_reports = []
    for ba_id, bb_id, orient in _seam_pairs(split):
        ba = blocks[ba_id]
        bb = blocks[bb_id]
        if orient == 'V':
            prof_a = ba['seams']['RIGHT']
            prof_b = bb['seams']['LEFT']
            seam_coord = ba['bbox'][2]  # shared x
            span_start, span_end = ba['bbox'][1], ba['bbox'][3]  # y range
            axis_name = 'y'
            coord_name = 'x'
        else:
            prof_a = ba['seams']['BOTTOM']
            prof_b = bb['seams']['TOP']
            seam_coord = ba['bbox'][3]  # shared y
            span_start, span_end = ba['bbox'][0], ba['bbox'][2]  # x range
            axis_name = 'x'
            coord_name = 'y'

        bands = []
        cont_count = 0
        for i in range(SEAM_BANDS):
            ra, ga, bca = prof_a[i]['rgb']
            rb, gb, bcb = prof_b[i]['rgb']
            delta = (abs(ra - rb) + abs(ga - gb) + abs(bca - bcb)) / 3.0
            ca, cb = prof_a[i]['cid'], prof_b[i]['cid']
            is_cont = (ca == cb) or (delta <= seam_delta_thresh)
            if is_cont:
                cont_count += 1
            s0 = span_start + i * (span_end - span_start) // SEAM_BANDS
            s1 = span_start + (i + 1) * (span_end - span_start) // SEAM_BANDS
            bands.append({
                'span': [s0, s1],
                'continuous': is_cont,
                'delta_rgb': round(delta, 1),
                'cid_a': ca,
                'cid_b': cb,
            })

        # Collapse adjacent continuous bands with same cid_a into clean runs
        merged_runs = []
        for bd in bands:
            if not bd['continuous']:
                continue
            if merged_runs and merged_runs[-1]['span'][1] == bd['span'][0] and merged_runs[-1]['cid'] == bd['cid_a']:
                merged_runs[-1]['span'][1] = bd['span'][1]
                merged_runs[-1]['deltas'].append(bd['delta_rgb'])
            else:
                merged_runs.append({
                    'span': list(bd['span']),
                    'cid': bd['cid_a'],
                    'deltas': [bd['delta_rgb']],
                })
        clean_runs = [
            {
                'span': r['span'],
                'cid': r['cid'],
                'family': pal_by_id[r['cid']]['family'],
                'hex': pal_by_id[r['cid']]['hex'],
                'mean_delta_rgb': round(sum(r['deltas']) / len(r['deltas']), 1),
            }
            for r in merged_runs
        ]
        seam_reports.append({
            'pair': f'{ba_id}<->{bb_id}',
            'orientation': orient,
            'seam_at': f'{coord_name}={seam_coord}',
            'axis': axis_name,
            'continuity_pct': round(cont_count * 100.0 / SEAM_BANDS, 1),
            'continuous_bands': clean_runs,
        })

    # 2. Cross-seam region merging (Union-Find over local regions that touch shared borders)
    nodes = []
    for bid in session['block_order']:
        for ri, reg in enumerate(blocks[bid]['regions']):
            nodes.append({
                'key': f'{bid}.R{ri}',
                'block': bid,
                'cid': reg['cid'],
                'bbox': reg['global_bbox'],
                'area_pct_block': reg['area_pct'],
                'edge_pct': reg['edge_pct'],
                'touches': set(reg['touches']),
            })

    parent_uf = list(range(len(nodes)))

    def find(i: int) -> int:
        while parent_uf[i] != i:
            parent_uf[i] = parent_uf[parent_uf[i]]
            i = parent_uf[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent_uf[rj] = ri

    pair_map = {(a, b): orient for a, b, orient in _seam_pairs(split)}
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            ni, nj = nodes[i], nodes[j]
            if ni['block'] == nj['block']:
                continue
            # Check color compatibility (same global cid or RGB delta <= seam_delta_thresh)
            pi = pal_by_id[ni['cid']]['rgb']
            pj = pal_by_id[nj['cid']]['rgb']
            cdelta = (abs(pi[0] - pj[0]) + abs(pi[1] - pj[1]) + abs(pi[2] - pj[2])) / 3.0
            if ni['cid'] != nj['cid'] and cdelta > seam_delta_thresh:
                continue
            # Check adjacency across shared seam
            orient = pair_map.get((ni['block'], nj['block']))
            swapped = False
            if orient is None:
                orient = pair_map.get((nj['block'], ni['block']))
                swapped = orient is not None
            if orient is None:
                continue
            left_n, right_n = (nj, ni) if swapped else (ni, nj)
            if orient == 'V':
                if 'RIGHT' in left_n['touches'] and 'LEFT' in right_n['touches']:
                    # Overlap on Y axis
                    oy0 = max(left_n['bbox'][1], right_n['bbox'][1])
                    oy1 = min(left_n['bbox'][3], right_n['bbox'][3])
                    if oy1 > oy0:
                        union(i, j)
            else:
                if 'BOTTOM' in left_n['touches'] and 'TOP' in right_n['touches']:
                    # Overlap on X axis
                    ox0 = max(left_n['bbox'][0], right_n['bbox'][0])
                    ox1 = min(left_n['bbox'][2], right_n['bbox'][2])
                    if ox1 > ox0:
                        union(i, j)

    groups: dict[int, list[dict[str, Any]]] = {}
    for idx, nd in enumerate(nodes):
        groups.setdefault(find(idx), []).append(nd)

    n_blocks = len(session['block_order'])
    unified_regions = []
    for members in groups.values():
        gx0 = min(m['bbox'][0] for m in members)
        gy0 = min(m['bbox'][1] for m in members)
        gx1 = max(m['bbox'][2] for m in members)
        gy1 = max(m['bbox'][3] for m in members)
        spans = sorted({m['block'] for m in members}, key=lambda b: session['block_order'].index(b))
        # Dominant cid weighted by block area
        cid_w: dict[int, float] = {}
        for m in members:
            cid_w[m['cid']] = cid_w.get(m['cid'], 0.0) + m['area_pct_block']
        dom_cid = max(sorted(cid_w), key=lambda c: cid_w[c])
        total_global_pct = round(sum(m['area_pct_block'] for m in members) / n_blocks, 1)
        mean_edge = round(sum(m['edge_pct'] for m in members) / len(members), 1)
        unified_regions.append({
            'cid': dom_cid,
            'symbol': pal_by_id[dom_cid]['symbol'],
            'hex': pal_by_id[dom_cid]['hex'],
            'family': pal_by_id[dom_cid]['family'],
            'spans': spans,
            'crosses_seam': len(spans) > 1,
            'global_bbox': [gx0, gy0, gx1, gy1],
            'est_global_area_pct': total_global_pct,
            'mean_edge_pct': mean_edge,
            'sources': [m['key'] for m in members],
        })
    unified_regions.sort(key=lambda r: (-int(r['crosses_seam']), -r['est_global_area_pct'], r['global_bbox'][1], r['global_bbox'][0]))
    for idx, ur in enumerate(unified_regions):
        ur['gid'] = f'G{idx}'

    # 3. Re-assemble stitched global overview maps (downsampled 2:1 per block to stay compact)
    def _downsample_grid(rows: list[str], step_y: int = 2, step_x: int = 2) -> list[str]:
        return [''.join(row[x] for x in range(0, len(row), step_x)) for row in rows[::step_y]]

    def _assemble(map_key: str) -> list[str]:
        if split == 'quarters':
            tl = _downsample_grid(blocks['TL'][map_key])
            tr = _downsample_grid(blocks['TR'][map_key])
            bl = _downsample_grid(blocks['BL'][map_key])
            br = _downsample_grid(blocks['BR'][map_key])
            top = [f'{a}|{b}' for a, b in zip(tl, tr)]
            sep = '-' * len(tl[0]) + '+' + '-' * len(tr[0])
            bot = [f'{a}|{b}' for a, b in zip(bl, br)]
            return top + [sep] + bot
        if split == 'halves_h':
            left = _downsample_grid(blocks['L'][map_key])
            right = _downsample_grid(blocks['R'][map_key])
            return [f'{a}|{b}' for a, b in zip(left, right)]
        top = _downsample_grid(blocks['T'][map_key])
        bot = _downsample_grid(blocks['B'][map_key])
        return top + ['-' * len(top[0])] + bot

    stitched_color = _assemble('color_map')
    stitched_edge = _assemble('edge_map')

    # 4. Align optional agent per-block notes cleanly
    aligned_notes = {}
    if block_notes:
        for k, v in block_notes.items():
            bk = str(k).strip().upper()
            if bk in blocks:
                aligned_notes[bk] = str(v).strip()

    return {
        'session_id': session['session_id'],
        'source_name': session['source_name'],
        'full_size': session['full_size'],
        'root_box': session['root_box'],
        'split': split,
        'quantization_error': session['quantization_error'],
        'seams': seam_reports,
        'unified_regions': unified_regions[:12],
        'stitched_color_map': stitched_color,
        'stitched_edge_map': stitched_edge,
        'aligned_notes': aligned_notes,
        'palette': session['palette'],
        'blocks_bbox': {bid: blocks[bid]['bbox'] for bid in session['block_order']},
        'blocks_stats': {
            bid: {
                'mean_luma': blocks[bid]['mean_luma'],
                'std_luma': blocks[bid]['std_luma'],
                'edge_density_pct': blocks[bid]['edge_density_pct'],
                'dominant_orientation': blocks[bid]['dominant_orientation'],
            }
            for bid in session['block_order']
        },
    }


def format_stitch_report(stitched: dict[str, Any]) -> str:
    fw, fh = stitched['full_size']
    err = stitched['quantization_error']
    out = [
        f'FLOWSIGHT STITCH v1 | session={stitched["session_id"]} | {stitched["source_name"]} ({fw}x{fh}) | scope={stitched["root_box"]} split={stitched["split"]}',
        f'APPROXIMATE QUANTIZED: MAE_RGB={err["mae_rgb"]} PSNR={err["psnr_db"]}dB (seam continuity below is measured directly on full-res border pixels)',
        '=== 0. PER-BLOCK MEASURED STATS (DEMONSTRATED) ===',
    ]
    for bid, bs in stitched.get('blocks_stats', {}).items():
        out.append(
            f'  {bid:<3s} bbox={stitched["blocks_bbox"][bid]} luma={bs["mean_luma"]:5.1f}±{bs["std_luma"]:4.1f} '
            f'edges={bs["edge_density_pct"]:4.1f}% orient={bs["dominant_orientation"]}'
        )
    out.append('=== 1. SEAM CONTINUITY (DEMONSTRATED from real border pixels) ===')
    for sm in stitched['seams']:
        out.append(f'  {sm["pair"]} at {sm["seam_at"]} ({sm["orientation"]}): continuity={sm["continuity_pct"]}%')
        if sm['continuous_bands']:
            for cb in sm['continuous_bands'][:4]:
                out.append(
                    f'    {sm["axis"]}={cb["span"]} continuous C{cb["cid"]}({cb["family"]},{cb["hex"]}) mean_dRGB={cb["mean_delta_rgb"]}'
                )
        else:
            out.append('    (sharp boundary across entire seam: no continuous color band)')

    out.append('=== 2. UNIFIED GLOBAL REGIONS (DEMONSTRATED cross-seam merge) ===')
    for ur in stitched['unified_regions']:
        tag = 'CROSS-SEAM' if ur['crosses_seam'] else 'SINGLE-BLK'
        out.append(
            f'  {ur["gid"]} [{tag}] spans={ur["spans"]} C{ur["cid"]}({ur["symbol"]},{ur["family"]},{ur["hex"]}) '
            f'bbox={ur["global_bbox"]} area~{ur["est_global_area_pct"]}% edges={ur["mean_edge_pct"]}%'
        )

    out.append('=== 3. STITCHED OVERVIEW MAPS (seam lines marked with | / -) ===')
    out.append('Color map (global palette symbols):')
    for row in stitched['stitched_color_map']:
        out.append('  ' + row)
    out.append('Edge map (.=flat :=soft +=medium #=strong):')
    for row in stitched['stitched_edge_map']:
        out.append('  ' + row)

    if stitched['aligned_notes']:
        out.append('=== 4. ALIGNED BLOCK OBSERVATIONS (AGENT_NOTES / INFERRED, NOT raw pixel proof) ===')
        for bid, note in stitched['aligned_notes'].items():
            out.append(f'  [{bid} {stitched["blocks_bbox"][bid]}]: {note}')
    return '\n'.join(out)


def main(argv=None) -> int:
    import argparse
    import sys

    p = argparse.ArgumentParser(description='FlowSight M2: 1920x1080 block partitioning & stitching CLI')
    sub = p.add_subparsers(dest='cmd', required=True)

    pm = sub.add_parser('map')
    pm.add_argument('image')
    pm.add_argument('--split', choices=VALID_SPLITS, default='quarters')
    pm.add_argument('--colors', type=int, default=12)
    pm.add_argument('--grid-w', type=int, default=24)
    pm.add_argument('--grid-h', type=int, default=14)
    pm.add_argument('--session-dir', default=None)

    pg = sub.add_parser('block')
    pg.add_argument('session_id')
    pg.add_argument('block_id')
    pg.add_argument('--no-edges', action='store_true')
    pg.add_argument('--session-dir', default=None)

    ps = sub.add_parser('subdivide')
    ps.add_argument('session_id')
    ps.add_argument('block_id')
    ps.add_argument('--split', choices=VALID_SPLITS, default='quarters')
    ps.add_argument('--session-dir', default=None)

    pt = sub.add_parser('stitch')
    pt.add_argument('session_id')
    pt.add_argument('--notes-json', default=None)
    pt.add_argument('--session-dir', default=None)

    args = p.parse_args(argv)
    try:
        if args.cmd == 'map':
            sess = map_image_blocks(args.image, args.split, args.colors, args.grid_w, args.grid_h, session_dir=args.session_dir)
            print(format_map_summary(sess))
        elif args.cmd == 'block':
            sess = load_session(args.session_id, args.session_dir)
            print(format_block_detail(sess, args.block_id, include_edge_map=not args.no_edges))
        elif args.cmd == 'subdivide':
            child = subdivide_block(args.session_id, args.block_id, split=args.split, session_dir=args.session_dir)
            print(format_map_summary(child))
        elif args.cmd == 'stitch':
            notes = json.loads(args.notes_json) if args.notes_json else None
            st = stitch_blocks(args.session_id, block_notes=notes, session_dir=args.session_dir)
            print(format_stitch_report(st))
        return 0
    except (BlocksDeny, OSError, json.JSONDecodeError) as e:
        print(f'DENY {e}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
