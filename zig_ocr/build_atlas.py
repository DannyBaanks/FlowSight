#!/usr/bin/env python3
"""Build compact FSOA v1 glyph shape atlas for FlowSight Zig Shape OCR.

Renders ASCII + Spanish + UI punctuation glyphs across installed Linux UI/TUI
fonts at multiple pixel sizes, extracts normalized 12x16 shape vectors plus
aspect ratio, relative baseline/cap vertical coordinates, and topological hole
count (Euler/border flood-fill), deduplicates near-identical templates, and
writes a compact binary atlas (`atlas.fsoa`).
"""
from __future__ import annotations

import os
import struct
import sys
from collections import deque
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

GRID_W = 12
GRID_H = 16
GRID_LEN = GRID_W * GRID_H  # 192 (multiple of 16 for Zig SIMD @Vector(16, f32))

# ASCII printable (33..126) + Spanish accents/punctuation + common UI glyphs
CHARSET = (
    "".join(chr(c) for c in range(33, 127))
    + "áéíóúüñÁÉÍÓÚÜÑ¡¿·•—"
)

CANDIDATE_FONTS = [
    "/usr/share/fonts/truetype/ubuntu/UbuntuSans[wdth,wght].ttf",
    "/usr/share/fonts/truetype/ubuntu/UbuntuSansMono[wght].ttf",
    "/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf",
    "/usr/share/fonts/truetype/ubuntu/Ubuntu-B.ttf",
    "/usr/share/fonts/truetype/ubuntu/UbuntuMono-R.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansMono-Regular.ttf",
    "/usr/share/fonts/truetype/noto/NotoSansMono-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Bold.ttf",
]

FONT_SIZES = [11, 13, 15, 18]


def count_holes(mask: np.ndarray) -> int:
    """Count enclosed background holes inside a binary glyph mask (0, 1, or 2+)."""
    h, w = mask.shape
    padded = np.zeros((h + 2, w + 2), dtype=np.uint8)
    padded[1:-1, 1:-1] = mask
    visited = np.zeros_like(padded, dtype=np.uint8)
    # Flood-fill exterior background starting at (0, 0)
    q = deque([(0, 0)])
    visited[0, 0] = 1
    while q:
        y, x = q.popleft()
        for ny, nx in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
            if 0 <= ny < h + 2 and 0 <= nx < w + 2:
                if not visited[ny, nx] and padded[ny, nx] == 0:
                    visited[ny, nx] = 1
                    q.append((ny, nx))
    holes = 0
    for y in range(1, h + 1):
        for x in range(1, w + 1):
            if padded[y, x] == 0 and not visited[y, x]:
                # Measure hole size (ignore 1-pixel anti-alias pinholes if large glyph)
                area = 0
                q = deque([(y, x)])
                visited[y, x] = 1
                while q:
                    cy, cx = q.popleft()
                    area += 1
                    for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
                        if 0 <= ny < h + 2 and 0 <= nx < w + 2:
                            if not visited[ny, nx] and padded[ny, nx] == 0:
                                visited[ny, nx] = 1
                                q.append((ny, nx))
                if area >= 2:
                    holes += 1
    return min(holes, 3)


def resample_bilinear(crop: np.ndarray, out_h: int = GRID_H, out_w: int = GRID_W) -> np.ndarray:
    """Bilinear resample float32 2D crop to (out_h, out_w) matching Zig implementation."""
    src_h, src_w = crop.shape
    ys = (np.arange(out_h, dtype=np.float32) + 0.5) * (src_h / float(out_h)) - 0.5
    xs = (np.arange(out_w, dtype=np.float32) + 0.5) * (src_w / float(out_w)) - 0.5
    ys = np.clip(ys, 0.0, max(0.0, src_h - 1.0))
    xs = np.clip(xs, 0.0, max(0.0, src_w - 1.0))
    y0 = np.floor(ys).astype(np.int32)
    x0 = np.floor(xs).astype(np.int32)
    y1 = np.minimum(y0 + 1, src_h - 1)
    x1 = np.minimum(x0 + 1, src_w - 1)
    fy = (ys - y0)[:, None]
    fx = (xs - x0)[None, :]
    v00 = crop[y0[:, None], x0[None, :]]
    v01 = crop[y0[:, None], x1[None, :]]
    v10 = crop[y1[:, None], x0[None, :]]
    v11 = crop[y1[:, None], x1[None, :]]
    return (v00 * (1.0 - fx) + v01 * fx) * (1.0 - fy) + (v10 * (1.0 - fx) + v11 * fx) * fy


def normalize_vec(grid: np.ndarray) -> np.ndarray:
    """Zero-mean, unit-L2 normalized vector (length 192)."""
    v = grid.reshape(-1).astype(np.float32)
    v = v - float(np.mean(v))
    norm = float(np.linalg.norm(v))
    if norm > 1e-6:
        v = v / norm
    return v


def render_glyph(font: ImageFont.FreeTypeFont, ch: str, ref_top: float, ref_bot: float):
    """Render a single glyph and return feature dict or None."""
    canvas_w, canvas_h = 64, 64
    img = Image.new("L", (canvas_w, canvas_h), 0)
    draw = ImageDraw.Draw(img)
    draw.text((16, 16), ch, fill=255, font=font)
    arr = np.asarray(img, dtype=np.float32)
    ys, xs = np.where(arr >= 36.0)
    if len(ys) < 2:
        return None
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    crop = arr[y0:y1, x0:x1] / 255.0
    gw = x1 - x0
    gh = y1 - y0
    ref_h = max(1.0, ref_bot - ref_top)

    grid = resample_bilinear(crop, GRID_H, GRID_W)
    vec = normalize_vec(grid)
    bin_mask = (crop >= 0.35).astype(np.uint8)
    holes = count_holes(bin_mask)

    aspect_q8 = int(round(min(65535.0, (gw / float(gh)) * 256.0)))
    y_top_q8 = int(round(max(-1024.0, min(1024.0, ((y0 - ref_top) / ref_h) * 256.0))))
    y_bot_q8 = int(round(max(-1024.0, min(1024.0, ((y1 - ref_top) / ref_h) * 256.0))))

    return {
        "codepoint": ord(ch),
        "vec": vec,
        "aspect_q8": aspect_q8,
        "y_top_q8": y_top_q8,
        "y_bot_q8": y_bot_q8,
        "holes": holes,
    }


def measure_font_ref(font: ImageFont.FreeTypeFont) -> tuple[float, float]:
    """Measure reference cap-top and baseline-bottom using 'HXM0'."""
    img = Image.new("L", (128, 64), 0)
    draw = ImageDraw.Draw(img)
    draw.text((16, 16), "HXM0", fill=255, font=font)
    arr = np.asarray(img, dtype=np.float32)
    ys, _ = np.where(arr >= 48.0)
    if len(ys) == 0:
        return 16.0, 30.0
    return float(ys.min()), float(ys.max()) + 1.0


def build_atlas(out_path: Path) -> None:
    fonts = [p for p in CANDIDATE_FONTS if os.path.isfile(p)]
    if not fonts:
        raise RuntimeError("No candidate TrueType fonts found on disk")

    by_cp: dict[int, list[dict]] = {}
    raw_count = 0

    for fp in fonts:
        for sz in FONT_SIZES:
            try:
                font = ImageFont.truetype(fp, sz)
            except OSError:
                continue
            ref_top, ref_bot = measure_font_ref(font)
            for ch in CHARSET:
                feat = render_glyph(font, ch, ref_top, ref_bot)
                if feat is None:
                    continue
                raw_count += 1
                cp = feat["codepoint"]
                bucket = by_cp.setdefault(cp, [])
                # Deduplicate near-identical shapes for the same codepoint
                dup = False
                for ex in bucket:
                    cos = float(np.dot(feat["vec"], ex["vec"]))
                    if (
                        cos >= 0.965
                        and abs(feat["aspect_q8"] - ex["aspect_q8"]) <= 28
                        and feat["holes"] == ex["holes"]
                        and abs(feat["y_top_q8"] - ex["y_top_q8"]) <= 32
                        and abs(feat["y_bot_q8"] - ex["y_bot_q8"]) <= 32
                    ):
                        dup = True
                        break
                if not dup:
                    bucket.append(feat)

    templates = []
    for cp in sorted(by_cp.keys()):
        templates.extend(by_cp[cp])

    # Pre-sort by aspect_q8 ascending so Zig runtime needs zero sorting on load.
    templates.sort(key=lambda t: (t["aspect_q8"], t["codepoint"]))

    # Binary format FSOA v3 (little-endian, pre-sorted by aspect, SoA quantized i8 + 4x8 coarse):
    # Header (16 bytes): magic "FSOA", version u16=3, grid_w u16=12, grid_h u16=16,
    #                    reserved u16=0, count u32
    # Section 1 — Meta SoA (count * 12 bytes):
    #   codepoint: u32, aspect_q8: u16, y_top_q8: i16, y_bot_q8: i16, holes: u8, pad: u8
    # Section 2 — Coarse 4x8 SoA (count * 32 bytes):
    #   32 * i8 (3x2 block-averaged from 12x16, zero-mean unit-L2, quantized * 256)
    # Section 3 — Fine 12x16 SoA (count * 192 bytes):
    #   192 * i8 quantized as round(clamp(v * 256.0, -127.0, 127.0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(struct.pack("<4sHHHHI", b"FSOA", 3, GRID_W, GRID_H, 0, len(templates)))
        for t in templates:
            f.write(
                struct.pack(
                    "<IHhhBB",
                    t["codepoint"],
                    t["aspect_q8"],
                    t["y_top_q8"],
                    t["y_bot_q8"],
                    t["holes"],
                    0,
                )
            )
        for t in templates:
            g = t["vec"].reshape(GRID_H, GRID_W)
            coarse = g.reshape(8, 2, 4, 3).mean(axis=(1, 3)).reshape(-1)
            coarse = coarse - float(np.mean(coarse))
            cnorm = float(np.linalg.norm(coarse))
            if cnorm > 1e-6:
                coarse = coarse / cnorm
            qc = np.round(np.clip(coarse * 256.0, -127.0, 127.0)).astype(np.int8)
            f.write(qc.tobytes())
        for t in templates:
            q = np.round(np.clip(t["vec"] * 256.0, -127.0, 127.0)).astype(np.int8)
            f.write(q.tobytes())

    size_kb = out_path.stat().st_size / 1024.0
    print(
        f"Built {out_path}: {len(templates)} unique templates "
        f"(from {raw_count} rendered across {len(fonts)} fonts x {len(FONT_SIZES)} sizes, "
        f"{len(by_cp)} codepoints, {size_kb:.1f} KiB)"
    )


if __name__ == "__main__":
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "atlas.fsoa"
    build_atlas(out)
