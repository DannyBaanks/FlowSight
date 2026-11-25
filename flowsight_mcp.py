#!/usr/bin/env python3
"""FlowSight MCP Server (mcp 2.x stdio) — Text-only visual peripheral.

Exposes 10 tools (read-only over user images; private local per-PC failure logging):
  1.  flowsight_map_blocks      — Split high-res images/photos (e.g. 1920x1080) into
                                  halves or quarters with shared global palette & error metrics.
  2.  flowsight_get_block       — Inspect a single block (color map + Sobel edge map + regions).
  3.  flowsight_subdivide_block — Recursively split any block into sub-quarters/halves (quadtree zoom).
  4.  flowsight_stitch_blocks   — Structurally & semantically stitch blocks across real pixel seams,
                                  merging cross-seam regions and aligning per-block agent notes.
  5.  flowsight_inspect_lossless — Query M1 lossless indices (--mode flow or --mode visual).
  6.  flowsight_screen_filters  — List ffmpeg-style screenshot filters & metadata.
  7.  flowsight_screen_run      — Run composable -vf screenshot pipeline (layout, controls, OCR, ui_diff, opt-in vlm_probe).
  8.  flowsight_screen_query    — Bounded selector query over canonical artifacts (P0, W1, K0, ambiguities, ocr, ui_diff).
  9.  flowsight_log_failure     — Record a user-confirmed perception error into this PC's private local corpus.
  10. flowsight_list_failures   — Inspect recent failure entries stored locally on this PC.

Read-only over user images. Never modifies FLOW or FlowGen.
Opt-in VLM escalation (`vlm_probe`) stays disabled unless FLOWSIGHT_VLM_ENABLED=1 is set in env.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
from typing import Literal

from mcp.server.mcpserver import MCPServer

import flowsight as fs
import flowsight_blocks as fb
import flowsight_corpus as fcorp
import flowsight_index as fidx
import flowsight_screen as fscr
import flowsight_timeline as ftl
import flowsight_vlm as fvlm

mcp = MCPServer(
    'flowsight',
    instructions=(
        'FlowSight gives text-only agents structured, bounded spatial perception over screenshots, '
        'images, videos, and FLOW programs without giant per-pixel JSON. '
        'When inspecting MULTIPLE screenshots (e.g. 5-50 files), prefer PROGRESSIVE SEARCH to save context: '
        'call flowsight_index_add(paths) -> flowsight_index_search(query, theme, color) -> then deepen ONLY '
        'the winning candidate crop via flowsight_screen_run(image_path, vf="crop=...,controls,ocr"). '
        'For videos (.mp4/.mkv/.webm) or ordered screenshot sequences, call flowsight_timeline_scan(inputs) '
        'to detect when modals, state/color changes, or error texts appeared. '
        'For single screenshots: flowsight_screen_filters() / flowsight_screen_run() / flowsight_screen_query(). '
        'For high-res split/stitch analysis: flowsight_map_blocks -> flowsight_get_block -> flowsight_stitch_blocks. '
        'For FLOW programs: flowsight_inspect_lossless(mode="flow"). '
        'Always distinguish DEMONSTRATED measurements from INFERRED hypotheses. '
        'When assisting a user on a text-only model: explain what FlowSight observed, verify accuracy with '
        'the user via ask_user, and if the user reports a FlowSight error (OCR, missed control, layout, diff), '
        'log it locally on this PC via flowsight_log_failure for private debugging.'
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
        '(crop, theme, layout, controls, text_bands, ocr, ui_map, ui_diff, vlm_probe, blocks).'
    ),
)
def flowsight_screen_filters() -> str:
    return fscr.format_filter_catalog()


@mcp.tool(
    description=(
        'Run an ffmpeg-style composable filter pipeline (`-vf`) specialized for desktop/web/app '
        'screenshots (e.g. 1920x1080). Example vf: "theme,layout,controls,text_bands,ocr,ui_map=48:20" '
        'or "crop=0:0:400:400,vlm_probe=question=Describe icon" (requires FLOWSIGHT_VLM_ENABLED=1) '
        'or "layout,controls,ocr,ui_diff=/path/before.png". '
        'Returns panels, clickable control centers (cx, cy), OCR text + associations, wireframe map, '
        'and exact global coordinates.'
    ),
)
def flowsight_screen_run(
    image_path: str,
    vf: str = 'theme,layout,controls,text_bands,ocr,ui_map=48:20',
    max_chars: int = 6500,
) -> str:
    try:
        vlm_policy, vlm_backend = fvlm.resolve_runtime_vlm_config()
        ctx = fscr.run_pipeline(image_path, vf=vf, vlm_policy=vlm_policy, vlm_backend=vlm_backend)
        return fscr.format_pipeline_report(ctx, max_chars=max(500, max_chars))
    except (fscr.FilterDeny, fb.BlocksDeny) as e:
        return f'DENY {e}'


@mcp.tool(
    description=(
        'Bounded targeted query over a screenshot pipeline result without dumping full JSON. '
        'Selectors: "P0" (panel), "W1" (local control), "K0" (persistent diff track), '
        '"ambiguities", "ocr", "controls", "layout", "ui_diff", "vlm_probe".'
    ),
)
def flowsight_screen_query(
    image_path: str,
    selector: str,
    vf: str = 'theme,layout,controls,ocr',
) -> str:
    try:
        vlm_policy, vlm_backend = fvlm.resolve_runtime_vlm_config()
        ctx = fscr.run_pipeline(image_path, vf=vf, vlm_policy=vlm_policy, vlm_backend=vlm_backend)
        return fscr.query_canonical(ctx, selector)
    except (fscr.FilterDeny, fb.BlocksDeny) as e:
        return f'DENY {e}'


@mcp.tool(
    description=(
        'Log a user-confirmed FlowSight perception failure into THIS PC ONLY private local corpus '
        '(~/.local/share/flowsight/private_failures/, mode 0700/0600, never synced or committed to git). '
        'Use after confirming via ask_user what FlowSight missed or misread so developers can debug locally.'
    ),
)
def flowsight_log_failure(
    category: Literal[
        'ocr_error',
        'control_missed',
        'control_false_positive',
        'layout_error',
        'diff_identity_error',
        'vlm_probe_error',
        'blind_misinterpretation',
        'other',
    ],
    user_correction: str,
    flowsight_observed: str = '',
    agent_explanation: str = '',
    expected_behavior: str = '',
    image_path: str = '',
    vf_pipeline: str = '',
    provider: str = '',
    model: str = '',
    notes: str = '',
    save_local_copy: bool = False,
) -> str:
    try:
        res = fcorp.log_failure(
            category=category,
            user_correction=user_correction,
            flowsight_observed=flowsight_observed,
            agent_explanation=agent_explanation,
            expected_behavior=expected_behavior,
            image_path=image_path or None,
            vf_pipeline=vf_pipeline,
            provider=provider,
            model=model,
            notes=notes,
            save_local_copy=save_local_copy,
        )
        return (
            f'LOGGED {res["entry_id"]} [category={res["category"]}] machine={res["machine_id"]}\n'
            f'corpus={res["corpus_path"]} img_sha={(res["source_image_sha256"] or "-")[:16]} '
            f'local_copy={res["local_image_copy"] or "none"}'
        )
    except (ValueError, OSError) as e:
        return f'DENY {e}'


@mcp.tool(
    description=(
        'List recent entries from THIS PC private local failure corpus '
        '(~/.local/share/flowsight/private_failures/) for local debugging.'
    ),
)
def flowsight_list_failures(
    limit: int = 20,
    category: str = '',
) -> str:
    rows = fcorp.list_failures(limit=limit, category=category or None)
    return fcorp.format_failure_list(rows)


@mcp.tool(
    description=(
        'Explicitly index one or more screenshot paths (or a single non-recursive directory) into '
        'the local ephemeral SQLite+FTS5 progressive visual index (~/.local/share/flowsight/, 0700/0600, '
        'NO image pixels stored, SHA-256 deduplicated, expires automatically after ttl_hours). '
        'Follow with flowsight_index_search to avoid dumping many full screenshot reports into context.'
    ),
)
def flowsight_index_add(
    paths: list[str],
    use_ocr: bool = True,
    ttl_hours: float = 24.0,
    max_files: int = 64,
) -> str:
    try:
        res = fidx.index_paths(paths, use_ocr=use_ocr, ttl_hours=ttl_hours, max_files=max_files)
        lines = [
            f'INDEXED new={res["indexed_new"]} cached={res["reused_cached"]} '
            f'total_in={res["total_input_files"]} errors={len(res["errors"])} '
            f'({res["elapsed_ms"]}ms, db={res["db_bytes"]}B, expires={res["expires_utc"]})',
            f'db={res["db_path"]}',
        ]
        for e in res['entries'][:20]:
            tag = 'CACHED' if e['cached'] else 'NEW'
            lines.append(
                f'  [{tag}] {e["source_name"]} sha={e["sha256"]} theme={e["theme"]} '
                f'ocr={e["ocr_status"]}({e["ocr_items"]}) controls={e["controls"]}'
            )
        for err in res['errors'][:5]:
            lines.append(f'  [ERR] {err["path"]}: {err["error"]}')
        return '\n'.join(lines)
    except (ValueError, OSError) as e:
        return f'DENY {e}'


@mcp.tool(
    description=(
        'Progressive visual search over explicitly indexed screenshots (BM25 FTS5 + exact token match + '
        'theme/color/time filters). Returns compact ranked candidates with matched OCR fragments, exact '
        'global bboxes [x0,y0,x1,y1), and ready-to-run padded `crop=x:y:w:h` hints for deepening.'
    ),
)
def flowsight_index_search(
    query: str = '',
    theme: str = '',
    color: str = '',
    since: str = '',
    until: str = '',
    limit: int = 5,
) -> str:
    try:
        res = fidx.search_index(
            query,
            theme=theme,
            color=color,
            since=since,
            until=until,
            limit=limit,
        )
        return fidx.format_search_report(res)
    except (ValueError, OSError) as e:
        return f'DENY {e}'


@mcp.tool(
    description=(
        'Explicit lifecycle management for the local progressive visual index: purge expired entries '
        '(mode="expired"), purge everything (mode="all"), or remove by sha256 prefix / path_prefix.'
    ),
)
def flowsight_index_purge(
    mode: Literal['expired', 'all', 'sha256', 'path_prefix'] = 'expired',
    sha256: str = '',
    path_prefix: str = '',
) -> str:
    try:
        res = fidx.purge_index(mode=mode, sha256=sha256, path_prefix=path_prefix)
        return (
            f'PURGED mode={res["mode"]} removed={res["removed"]} remaining={res["remaining"]} '
            f'db={res["db_path"]}'
        )
    except (ValueError, OSError) as e:
        return f'DENY {e}'


@mcp.tool(
    description=(
        'Temporal UI perception over a video file (.mp4/.mkv/.webm/.avi) OR an ordered list of '
        'screenshot paths. Skips near-identical static frames, tracks persistent controls (K0..Kn) '
        'and OCR text across keyframes, and reports timestamped events (CONTROL_APPEARED, '
        'CONTROL_COLOR_CHANGED + INFERRED STATE_CHANGED_CANDIDATE, TEXT_APPEARED, etc.).'
    ),
)
def flowsight_timeline_scan(
    inputs: list[str],
    sample_fps: float = 2.0,
    max_frames: int = 40,
    scene_thresh: float = 6.0,
    use_ocr: bool = True,
    query: str = '',
) -> str:
    try:
        if not inputs:
            return 'DENY inputs must contain a video path or >=2 image paths'
        first = Path(inputs[0]).expanduser()
        if len(inputs) == 1 and first.suffix.lower() in ftl.VIDEO_EXTS:
            res = ftl.scan_video(
                first,
                sample_fps=sample_fps,
                max_frames=max_frames,
                scene_thresh=scene_thresh,
                use_ocr=use_ocr,
            )
        else:
            res = ftl.scan_sequence(
                inputs,
                scene_thresh=scene_thresh,
                use_ocr=use_ocr,
            )
        return ftl.format_timeline_report(res, query=query)
    except (ValueError, OSError, fscr.FilterDeny) as e:
        return f'DENY {e}'


if __name__ == '__main__':
    mcp.run(transport='stdio')
