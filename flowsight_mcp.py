#!/usr/bin/env python3
"""FlowSight MCP Server (mcp 2.x stdio) — Text-only visual peripheral.

Exposes:
  1. flowsight_map_blocks      — Split high-res images/photos (e.g. 1920x1080) into
                                 halves or quarters with shared global palette & error metrics.
  2. flowsight_get_block       — Inspect a single block (color map + Sobel edge map + regions).
  3. flowsight_subdivide_block — Recursively split any block into sub-quarters/halves (quadtree zoom).
  4. flowsight_stitch_blocks   — Structurally & semantically stitch blocks across real pixel seams,
                                 merging cross-seam regions and aligning per-block agent notes.
  5. flowsight_inspect_lossless — Query M1 lossless indices (--mode flow or --mode visual).

Read-only over user images. Never modifies FLOW or FlowGen.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
from typing import Literal

from mcp.server.mcpserver import MCPServer

import flowsight as fs
import flowsight_blocks as fb
import flowsight_screen as fscr

mcp = MCPServer(
    'flowsight',
    instructions=(
        'FlowSight gives text-only agents structured, bounded spatial perception over screenshots, '
        'images, and FLOW programs without giant per-pixel JSON. '
        'For desktop/mobile/web screenshots (ffmpeg-style composable filters): call '
        'flowsight_screen_filters() to discover filters, then flowsight_screen_run(image_path, vf=...) '
        'e.g. vf="theme,layout,controls,text_bands,ui_map=48:20" or crop=x:y:w:h to zoom into a panel. '
        'For high-res split/stitch analysis: flowsight_map_blocks -> flowsight_get_block -> flowsight_stitch_blocks. '
        'For FLOW programs: flowsight_inspect_lossless(mode="flow"). '
        'Always distinguish DEMONSTRATED measurements from INFERRED semantic hypotheses.'
    ),
)


@mcp.tool(
    description=(
        'Map any PNG/JPEG/WEBP image (including 1920x1080 photos or screenshots) into equal '
        'halves (halves_h, halves_v) or quarters (2x2: TL, TR, BL, BR) using a shared global '
        'quantized palette. Reports real quantization error (MAE, RMSE, PSNR) and returns a '
        'session_id for per-block inspection, recursive subdivision, and stitching.'
    ),
)
def flowsight_map_blocks(
    image_path: str,
    split: Literal['quarters', 'halves_h', 'halves_v'] = 'quarters',
    palette_k: int = 12,
    grid_w: int = 24,
    grid_h: int = 14,
) -> str:
    try:
        sess = fb.map_image_blocks(
            image_path=image_path,
            split=split,
            palette_k=palette_k,
            grid_w=grid_w,
            grid_h=grid_h,
        )
        return fb.format_map_summary(sess)
    except fb.BlocksDeny as e:
        return f'DENY {e}'


@mcp.tool(
    description=(
        'Inspect a single block (TL, TR, BL, BR, L, R, T, or B) from a mapped session. '
        'Returns global coordinates, luminance/edge statistics, a compact color map, '
        'a Sobel edge/structure map, and local dominant regions with seam touch markers.'
    ),
)
def flowsight_get_block(
    session_id: str,
    block_id: str,
    include_edge_map: bool = True,
) -> str:
    try:
        sess = fb.load_session(session_id)
        return fb.format_block_detail(sess, block_id, include_edge_map=include_edge_map)
    except fb.BlocksDeny as e:
        return f'DENY {e}'


@mcp.tool(
    description=(
        'Recursively subdivide an existing block (e.g. TL 960x540 -> 4 sub-blocks 480x270) '
        'preserving exact global image coordinates. Returns a new child session_id.'
    ),
)
def flowsight_subdivide_block(
    session_id: str,
    block_id: str,
    split: Literal['quarters', 'halves_h', 'halves_v'] = 'quarters',
    palette_k: int | None = None,
) -> str:
    try:
        child = fb.subdivide_block(
            session_id=session_id,
            block_id=block_id,
            split=split,
            palette_k=palette_k,
        )
        return fb.format_map_summary(child)
    except fb.BlocksDeny as e:
        return f'DENY {e}'


@mcp.tool(
    description=(
        'Structurally and semantically stitch the blocks of a session back into a unified '
        'global scene. Measures real full-resolution pixel continuity across shared seams, '
        'merges cross-seam regions into global G0..Gn objects, renders stitched overview maps, '
        'and aligns optional per-block text observations (block_notes: {"TL": "...", ...}).'
    ),
)
def flowsight_stitch_blocks(
    session_id: str,
    block_notes: dict[str, str] | None = None,
    seam_delta_thresh: float = 32.0,
) -> str:
    try:
        stitched = fb.stitch_blocks(
            session_id=session_id,
            block_notes=block_notes,
            seam_delta_thresh=seam_delta_thresh,
        )
        return fb.format_stitch_report(stitched)
    except fb.BlocksDeny as e:
        return f'DENY {e}'


@mcp.tool(
    description=(
        'Lossless M1 inspection for FLOW program PNGs (mode="flow", exact unquantized R/G '
        'vector field + discrete B opcodes) or <=64-color pixel art (mode="visual"). '
        'Queries: summary, point, scanline, region, field, instructions, find, components, edges.'
    ),
)
def flowsight_inspect_lossless(
    image_path: str,
    mode: Literal['flow', 'visual'] = 'flow',
    query: Literal['summary', 'point', 'scanline', 'region', 'field', 'instructions', 'find', 'components', 'edges'] = 'summary',
    x: int = 0,
    y: int = 0,
    w: int = 16,
    h: int = 16,
    grid: bool = False,
    token: str | None = None,
    limit: int = 40,
) -> str:
    try:
        p = Path(image_path).resolve()
        if not p.is_file():
            return f'DENY file not found: {image_path}'
        key = hashlib.sha256(f'{p}:{p.stat().st_mtime_ns}:{mode}'.encode()).hexdigest()[:16]
        cache_dir = Path(tempfile.gettempdir()) / 'flowsight_lossless_cache'
        cache_dir.mkdir(parents=True, exist_ok=True)
        db_path = cache_dir / f'{key}_{mode}.sqlite'
        if not db_path.exists():
            fs.index_image(str(p), str(db_path), mode=mode, force=True)
        with fs.open_index(db_path) as (db, meta):
            if query == 'summary':
                return fs.summary(db, meta)
            if query == 'point':
                return fs.point(db, meta, x, y)
            if query == 'scanline':
                return fs.scanline(db, meta, y, max_lines=limit)
            if query == 'region':
                return fs.region(db, meta, x, y, w, h, max_lines=limit, grid=grid)
            if query == 'field':
                box = (x, y, w, h) if (w > 0 and h > 0 and (x > 0 or y > 0 or w < int(meta['width']) or h < int(meta['height']))) else None
                return fs.field_query(db, meta, box=box)
            if query == 'instructions':
                return fs.instructions_query(db, meta, opcode=token, max_lines=limit)
            if query == 'find':
                if not token:
                    return 'DENY find requires token (#hex or opcode)'
                return fs.find_color(db, meta, token, max_lines=limit)
            if query == 'components':
                return fs.components(db, meta, max_lines=limit)
            if query == 'edges':
                box = (x, y, w, h) if (w > 0 and h > 0) else None
                return fs.edges(db, meta, max_lines=limit, box=box)
            return f'DENY unknown query: {query}'
    except fs.Deny as e:
        return f'DENY {e}'


@mcp.tool(
    description=(
        'List all available ffmpeg-style screenshot filters and their `-vf` syntax '
        '(crop, theme, layout, controls, text_bands, ocr, ui_map, ui_diff, blocks).'
    ),
)
def flowsight_screen_filters() -> str:
    return fscr.format_filter_catalog()


@mcp.tool(
    description=(
        'Run an ffmpeg-style composable filter pipeline (`-vf`) specialized for desktop/web/app '
        'screenshots (e.g. 1920x1080). Example vf: "theme,layout,controls,text_bands,ui_map=48:20" '
        'or "crop=0:0:400:1080,controls,text_bands" or "layout,controls,ui_diff=/path/before.png". '
        'Returns panels, clickable control centers (cx, cy), geometric text bands, wireframe map, '
        'and exact global coordinates.'
    ),
)
def flowsight_screen_run(
    image_path: str,
    vf: str = 'theme,layout,controls,text_bands,ui_map=48:20',
    max_chars: int = 6500,
) -> str:
    try:
        ctx = fscr.run_pipeline(image_path, vf=vf)
        return fscr.format_pipeline_report(ctx, max_chars=max(500, max_chars))
    except (fscr.FilterDeny, fb.BlocksDeny) as e:
        return f'DENY {e}'


@mcp.tool(
    description=(
        'Bounded targeted query over a screenshot pipeline result without dumping full JSON. '
        'Selectors: "P0" (panel), "W1" (local control), "K0" (persistent diff track), '
        '"ambiguities", "ocr", "controls", "layout", "ui_diff".'
    ),
)
def flowsight_screen_query(
    image_path: str,
    selector: str,
    vf: str = 'theme,layout,controls,ocr',
) -> str:
    try:
        ctx = fscr.run_pipeline(image_path, vf=vf)
        return fscr.query_canonical(ctx, selector)
    except (fscr.FilterDeny, fb.BlocksDeny) as e:
        return f'DENY {e}'


if __name__ == '__main__':
    mcp.run(transport='stdio')
