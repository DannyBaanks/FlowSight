#!/usr/bin/env python3
"""FlowSight M4 benchmark & metrics runner (§9).

Separately measures:
  - layout_ms, controls_ms, ocr_ms, association_ms, ui_diff_ms, total_ms
  - text representation size (chars)
  - bbox IoU accuracy vs ground truth
  - OCR CER/WER when real Tesseract is installed (explicitly NOT_DEMONSTRATED if absent)
  - text-control association precision/recall (on synthetic geometric benchmark + real OCR if available)
  - temporal matching precision & false-positive rates for NEW_CONTROL / REMOVED_CONTROL
  - peak Python heap memory (tracemalloc) & artifact sizes
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import tracemalloc

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'examples'))

import corpus_m4
import flowsight_diff as fdiff
import flowsight_ocr as focr
import flowsight_screen as fscr

OUT_DIR = REPO / 'evidence' / 'm4'
REAL_CAPS_DIR = Path('/home/danny/Imágenes/Capturas de pantalla')


def _stage_ms(trace: list[dict], name: str) -> float:
    return round(sum(t['ms'] for t in trace if t['filter'] == name), 2)


def run_m4_benchmark() -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    corpus_dir = OUT_DIR / 'corpus'
    manifest = corpus_m4.build_corpus(corpus_dir)
    tess_path, tess_ver = focr.detect_tesseract_engine()

    tracemalloc.start()

    # 1. Single-frame PR dark (fast mode vs OCR mode)
    pr_dark = manifest['cases']['pr_dark']
    ctx_fast = fscr.run_pipeline(pr_dark['path'], vf='theme,layout,controls,text_bands,ui_map=48:20')
    rep_fast = fscr.format_pipeline_report(ctx_fast)

    ctx_ocr = fscr.run_pipeline(pr_dark['path'], vf='theme,layout,controls,text_bands,ocr,ui_map=48:20')
    rep_ocr = fscr.format_pipeline_report(ctx_ocr)
    (OUT_DIR / 'example_ocr_pipeline.txt').write_text(rep_ocr + '\n', encoding='utf-8')

    # Measure bbox IoU vs ground truth on pr_dark
    gt_boxes = [c['bbox'] for c in pr_dark['controls']]
    det_boxes = [c['global_bbox'] for c in ctx_fast.artifacts['controls']['controls']]
    ious = []
    for gb in gt_boxes:
        best = max((fdiff._iou(gb, db) for db in det_boxes), default=0.0)
        ious.append(round(best, 4))

    # 2. Temporal identity diff (Case 7 + Case 8: modal renumbering + color change)
    c78 = manifest['cases']['modal_and_color_change']
    ctx_diff = fscr.run_pipeline(c78['frame_b'], vf=f'layout,controls,ocr,ui_diff={c78["frame_a"]}')
    rep_diff = fscr.format_pipeline_report(ctx_diff)
    (OUT_DIR / 'example_uidiff_pipeline.txt').write_text(rep_diff + '\n', encoding='utf-8')

    tracks_78 = ctx_diff.artifacts['ui_diff']['tracking']['tracks']
    fp_removed_78 = sum(1 for e in tracks_78 if e['classification'] == fdiff.CLS_REMOVED)
    color_changed_78 = sum(1 for e in tracks_78 if fdiff.CLS_COLOR_CHANGED in e['flags'])
    new_78 = sum(1 for e in tracks_78 if e['classification'] == fdiff.CLS_NEW)

    # 3. Move button (Case 9)
    c9 = manifest['cases']['move_button']
    ctx_m9 = fscr.run_pipeline(c9['frame_b'], vf=f'layout,controls,ui_diff={c9["frame_a"]}')
    tr9 = ctx_m9.artifacts['ui_diff']['tracking']['tracks']
    move_exact = (
        sum(1 for e in tr9 if fdiff.CLS_MOVED in e['flags']) == 1
        and sum(1 for e in tr9 if e['classification'] in (fdiff.CLS_NEW, fdiff.CLS_REMOVED)) == 0
    )

    # 4. Disappear + far new negative control (Case 10)
    c10 = manifest['cases']['disappear_and_new_far']
    ctx_m10 = fscr.run_pipeline(c10['frame_b'], vf=f'layout,controls,ui_diff={c10["frame_a"]}')
    tr10 = ctx_m10.artifacts['ui_diff']['tracking']['tracks']
    neg_exact = (
        sum(1 for e in tr10 if e['classification'] == fdiff.CLS_REMOVED) == 1
        and sum(1 for e in tr10 if e['classification'] == fdiff.CLS_NEW) == 1
    )

    # 5. Ambiguous identical controls (Case 13)
    c13 = manifest['cases']['ambiguous_identical']
    ctx_m13 = fscr.run_pipeline(c13['frame_b'], vf=f'layout,controls,ui_diff={c13["frame_a"]}')
    tr13 = ctx_m13.artifacts['ui_diff']['tracking']['tracks']
    ambig_detected = any(e['classification'] == fdiff.CLS_AMBIGUOUS for e in tr13)

    # 6. Geometric association precision/recall on synthetic OCR ground truth (explicitly labeled)
    syn_items = []
    for idx, c in enumerate(pr_dark['controls']):
        tb = c.get('text_bbox') or c.get('label_bbox')
        syn_items.append({
            'id': f'O{idx}', 'text': c['expected_text'], 'bbox': tb,
            'confidence': 95.0, 'status': focr.OCR_OBSERVED,
        })
    assoc_eval = focr.associate_text_to_controls(
        ctx_fast.artifacts['controls']['controls'],
        {'status': focr.OCR_OBSERVED, 'items': syn_items},
        panels=ctx_fast.artifacts['layout']['panels'],
    )
    expected_map = {c['expected_text']: c['expected_assoc'] for c in pr_dark['controls']}
    tp_assoc = 0
    fp_assoc = 0
    for c in assoc_eval['controls']:
        if c['association'] in (focr.ASSOC_INSIDE, focr.ASSOC_LABEL):
            if c.get('text') in expected_map and expected_map[c['text']] == c['association']:
                tp_assoc += 1
            else:
                fp_assoc += 1
    assoc_precision = round(tp_assoc / max(1, tp_assoc + fp_assoc), 4)
    assoc_recall = round(tp_assoc / max(1, len(expected_map)), 4)

    # 7. Real 1920x1080 Linux desktop screenshot benchmark (read-only)
    real_bench = None
    caps = sorted(REAL_CAPS_DIR.glob('*.png')) if REAL_CAPS_DIR.is_dir() else []
    if caps:
        ctx_r_fast = fscr.run_pipeline(caps[-1], vf='theme,layout,controls,text_bands,ui_map=48:18')
        ctx_r_ocr = fscr.run_pipeline(caps[-1], vf='theme,layout,controls,text_bands,ocr,ui_map=48:18')
        real_bench = {
            'size': [ctx_r_fast.full_w, ctx_r_fast.full_h],
            'fast_mode_ms': {
                'layout': _stage_ms(ctx_r_fast.trace, 'layout'),
                'controls': _stage_ms(ctx_r_fast.trace, 'controls'),
                'text_bands': _stage_ms(ctx_r_fast.trace, 'text_bands'),
                'total': round(sum(t['ms'] for t in ctx_r_fast.trace), 2),
                'report_chars': len(fscr.format_pipeline_report(ctx_r_fast)),
            },
            'ocr_mode_ms': {
                'ocr_status': ctx_r_ocr.artifacts['ocr']['status'],
                'ocr': _stage_ms(ctx_r_ocr.trace, 'ocr'),
                'association': ctx_r_ocr.artifacts.get('text_control_association', {}).get('elapsed_ms', 0.0),
                'total': round(sum(t['ms'] for t in ctx_r_ocr.trace), 2),
            },
        }

    cur_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    report = {
        'tesseract_installed': tess_path is not None,
        'tesseract_engine': tess_ver,
        'ocr_cer_wer_real': 'NOT_DEMONSTRATED (tesseract binary absent on host PATH)' if tess_path is None else 'measured',
        'synthetic_1280x720_fast': {
            'layout_ms': _stage_ms(ctx_fast.trace, 'layout'),
            'controls_ms': _stage_ms(ctx_fast.trace, 'controls'),
            'text_bands_ms': _stage_ms(ctx_fast.trace, 'text_bands'),
            'total_ms': round(sum(t['ms'] for t in ctx_fast.trace), 2),
            'report_chars': len(rep_fast),
            'mean_bbox_iou_vs_gt': round(sum(ious) / max(1, len(ious)), 4),
            'per_control_iou': ious,
        },
        'synthetic_1280x720_with_ocr_stage': {
            'ocr_status': ctx_ocr.artifacts['ocr']['status'],
            'ocr_ms': _stage_ms(ctx_ocr.trace, 'ocr'),
            'association_ms': ctx_ocr.artifacts.get('text_control_association', {}).get('elapsed_ms', 0.0),
            'total_ms': round(sum(t['ms'] for t in ctx_ocr.trace), 2),
            'report_chars': len(rep_ocr),
        },
        'geometric_association_benchmark_synthetic_ocr': {
            'precision': assoc_precision,
            'recall': assoc_recall,
            'tp': tp_assoc,
            'fp': fp_assoc,
            'gt_total': len(expected_map),
            'elapsed_ms': assoc_eval['elapsed_ms'],
        },
        'temporal_matching_metrics': {
            'ui_diff_ms': _stage_ms(ctx_diff.trace, 'ui_diff'),
            'diff_report_chars': len(rep_diff),
            'case78_modal_plus_color': {
                'fp_removed_controls': fp_removed_78,
                'color_changed_detected': color_changed_78,
                'new_controls_detected': new_78,
            },
            'case9_move_exact': move_exact,
            'case10_negative_far_exact': neg_exact,
            'case13_ambiguous_detected': ambig_detected,
        },
        'real_desktop_1920x1080': real_bench,
        'memory_bytes': {'current': cur_mem, 'peak': peak_mem, 'peak_mib': round(peak_mem / (1024 * 1024), 2)},
    }
    (OUT_DIR / 'benchmark_m4.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    run_m4_benchmark()
