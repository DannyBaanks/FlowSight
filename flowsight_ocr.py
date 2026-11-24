"""FlowSight M4-A — Real OCR extraction (Tesseract) & geometric text-control association.

Epistemic guarantees:
  - Uses local `tesseract` binary when available; NEVER invents labels or fakes OCR.
  - When `tesseract` is missing, returns explicit `status="OCR_ABSENT"` with cause.
  - Distinguishes `OCR_OBSERVED`, `OCR_UNCERTAIN`, `OCR_ABSENT`, and `OCR_ERROR`.
  - Explicitly documents confidence scale (`tesseract_0_100_uncalibrated`) and states
    that OCR confidence is NOT a calibrated probability of correctness.
  - An empty OCR output is NOT proof that no text exists on screen.
  - Geometric text-control association distinguishes `TEXT_INSIDE`, `LABEL_ASSOCIATED`,
    `AMBIGUOUS_ASSOCIATION`, and `NO_TEXT_EVIDENCE` without picking arbitrarily on ties.
"""
from __future__ import annotations

import math
import re
import shutil
import subprocess
import tempfile
import time
from typing import Any, Callable

import numpy as np
from PIL import Image

OCR_OBSERVED = 'OCR_OBSERVED'
OCR_UNCERTAIN = 'OCR_UNCERTAIN'
OCR_ABSENT = 'OCR_ABSENT'
OCR_ERROR = 'OCR_ERROR'

ASSOC_INSIDE = 'TEXT_INSIDE'
ASSOC_LABEL = 'LABEL_ASSOCIATED'
ASSOC_AMBIGUOUS = 'AMBIGUOUS_ASSOCIATION'
ASSOC_NONE = 'NO_TEXT_EVIDENCE'

DEFAULT_MIN_OBSERVED_CONF = 60.0
TESSERACT_CONF_SCALE = 'tesseract_0_100_uncalibrated (NOT calibrated probability)'
RAPIDOCR_CONF_SCALE = 'rapidocr_0_100_uncalibrated (raw 0..1 x100; NOT calibrated probability)'

_RAPIDOCR_SINGLETON: Any = None


def detect_rapidocr_engine() -> tuple[bool, str]:
    """Return (available, version_string) for local rapidocr-onnxruntime."""
    try:
        import rapidocr_onnxruntime as rmod
        ver = getattr(rmod, '__version__', 'installed')
        return True, f'rapidocr-onnxruntime {ver} (PP-OCRv4 ONNX)'
    except ImportError:
        return False, 'absent'


def _get_rapidocr_instance():
    global _RAPIDOCR_SINGLETON
    if _RAPIDOCR_SINGLETON is None:
        from rapidocr_onnxruntime import RapidOCR
        _RAPIDOCR_SINGLETON = RapidOCR()
    return _RAPIDOCR_SINGLETON


def _run_rapidocr(
    rgb: np.ndarray,
    offset_x: int,
    offset_y: int,
    source_frame: str,
    lang: str,
    min_observed_conf: float,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    avail, engine_ver = detect_rapidocr_engine()
    if not avail:
        return {
            'status': OCR_ABSENT,
            'engine': 'none',
            'confidence_scale': RAPIDOCR_CONF_SCALE,
            'source_frame': source_frame,
            'lang': lang,
            'preprocessing': 'none',
            'detail': 'rapidocr-onnxruntime not installed',
            'note': 'OCR_ABSENT means text could not be read; it does NOT prove the screen has no text.',
            'items': [],
            'elapsed_ms': round((time.perf_counter() - t0) * 1000.0, 2),
        }
    try:
        engine = _get_rapidocr_instance()
        # RapidOCR expects BGR or RGB numpy array (PIL/RGB works cleanly; pass BGR for cv2 native consistency)
        bgr = rgb[:, :, ::-1].copy()
        result, _elapse = engine(bgr)
    except Exception as e:
        return {
            'status': OCR_ERROR,
            'engine': engine_ver,
            'confidence_scale': RAPIDOCR_CONF_SCALE,
            'source_frame': source_frame,
            'lang': lang,
            'preprocessing': 'native_bgr',
            'detail': f'rapidocr execution failed: {e}',
            'items': [],
            'elapsed_ms': round((time.perf_counter() - t0) * 1000.0, 2),
        }

    items: list[dict[str, Any]] = []
    for entry in (result or []):
        # entry = [quad_points, text, score_0_to_1]
        if not entry or len(entry) < 3:
            continue
        pts, raw_txt, raw_score = entry[0], str(entry[1]).strip(), float(entry[2])
        if not raw_txt:
            continue
        xs = [int(round(p[0])) for p in pts]
        ys = [int(round(p[1])) for p in pts]
        x0, x1 = max(0, min(xs)), max(xs)
        y0, y1 = max(0, min(ys)), max(ys)
        if x1 <= x0 or y1 <= y0:
            continue
        conf = round(max(0.0, min(100.0, raw_score * 100.0)), 1)
        st = OCR_OBSERVED if conf >= min_observed_conf else OCR_UNCERTAIN
        items.append({
            'text': raw_txt,
            'bbox': [offset_x + x0, offset_y + y0, offset_x + x1, offset_y + y1],
            'confidence': conf,
            'min_word_confidence': conf,
            'confidence_scale': RAPIDOCR_CONF_SCALE,
            'engine': engine_ver,
            'source_frame': source_frame,
            'status': st,
        })

    items.sort(key=lambda r: (r['bbox'][1] // 10, r['bbox'][0]))
    for idx, it in enumerate(items):
        it['id'] = f'O{idx}'

    overall = OCR_OBSERVED if any(it['status'] == OCR_OBSERVED for it in items) else (
        OCR_UNCERTAIN if items else OCR_OBSERVED
    )
    return {
        'status': overall,
        'engine': engine_ver,
        'confidence_scale': RAPIDOCR_CONF_SCALE,
        'source_frame': source_frame,
        'lang': lang,
        'preprocessing': 'native_bgr',
        'detail': f'{len(items)} line(s) extracted via RapidOCR; empty result does NOT prove absence of text',
        'items': items,
        'elapsed_ms': round((time.perf_counter() - t0) * 1000.0, 2),
    }


def normalize_text(text: str) -> str:
    """Normalize OCR/reference text for comparison (lowercase, collapsed whitespace)."""
    return re.sub(r'\s+', ' ', (text or '').strip()).lower()


def levenshtein(seq_a, seq_b) -> int:
    """Exact deterministic Levenshtein edit distance over sequences (str or list[str])."""
    n, m = len(seq_a), len(seq_b)
    if n == 0:
        return m
    if m == 0:
        return n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        ai = seq_a[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ai == seq_b[j - 1] else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
        prev = cur
    return prev[m]


def compute_cer_wer(reference: str, hypothesis: str) -> dict[str, float]:
    """Compute Character Error Rate (CER) and Word Error Rate (WER) on normalized strings."""
    ref_c = normalize_text(reference)
    hyp_c = normalize_text(hypothesis)
    cer = levenshtein(ref_c, hyp_c) / max(1, len(ref_c)) if ref_c else (0.0 if not hyp_c else 1.0)
    ref_w = ref_c.split()
    hyp_w = hyp_c.split()
    wer = levenshtein(ref_w, hyp_w) / max(1, len(ref_w)) if ref_w else (0.0 if not hyp_w else 1.0)
    return {'cer': round(cer, 4), 'wer': round(wer, 4)}


def detect_tesseract_engine() -> tuple[str | None, str]:
    """Return (binary_path, version_string) or (None, 'absent')."""
    tess = shutil.which('tesseract')
    if not tess:
        return None, 'absent'
    try:
        proc = subprocess.run([tess, '--version'], capture_output=True, text=True, timeout=5)
        first = (proc.stdout or proc.stderr or '').strip().splitlines()
        ver = first[0].strip() if first else 'tesseract (unknown version)'
        return tess, ver
    except (OSError, subprocess.SubprocessError):
        return tess, 'tesseract (version query failed)'


def run_ocr(
    rgb: np.ndarray,
    offset_x: int,
    offset_y: int,
    source_frame: str,
    lang: str = 'eng',
    min_observed_conf: float = DEFAULT_MIN_OBSERVED_CONF,
    psm: int = 11,
    timeout_s: float = 30.0,
    backend: str = 'auto',
    provider: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run local OCR (`tesseract` or `rapidocr-onnxruntime`) over an RGB array.

    `backend`:
      - 'auto' (default): uses `tesseract` if on PATH, else `rapidocr-onnxruntime` if installed, else `OCR_ABSENT`.
      - 'tesseract': require local tesseract binary (returns `OCR_ABSENT` if missing).
      - 'rapidocr': require local rapidocr-onnxruntime (returns `OCR_ABSENT` if missing).
    Never fabricates text when no local OCR engine is available.
    """
    t0 = time.perf_counter()
    if provider is not None:
        res = provider(rgb, offset_x=offset_x, offset_y=offset_y, source_frame=source_frame, lang=lang)
        res.setdefault('elapsed_ms', round((time.perf_counter() - t0) * 1000.0, 2))
        return res

    bmode = (backend or 'auto').strip().lower()
    if bmode == 'rapidocr':
        return _run_rapidocr(rgb, offset_x, offset_y, source_frame, lang, min_observed_conf)

    tess, engine_ver = detect_tesseract_engine()
    if not tess:
        if bmode == 'auto':
            rap_ok, _ = detect_rapidocr_engine()
            if rap_ok:
                return _run_rapidocr(rgb, offset_x, offset_y, source_frame, lang, min_observed_conf)
        return {
            'status': OCR_ABSENT,
            'engine': 'none',
            'confidence_scale': TESSERACT_CONF_SCALE,
            'source_frame': source_frame,
            'lang': lang,
            'preprocessing': 'none',
            'detail': 'no local OCR engine available (tesseract absent on PATH' + (
                '; rapidocr-onnxruntime absent)' if bmode == 'auto' else ')'
            ),
            'note': 'OCR_ABSENT means text could not be read; it does NOT prove the screen has no text.',
            'items': [],
            'elapsed_ms': round((time.perf_counter() - t0) * 1000.0, 2),
        }

    # Dark-mode screenshots often read markedly better when inverted to dark-on-light for Tesseract.
    luma = 0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1] + 0.114 * rgb[:, :, 2]
    med_luma = float(np.median(luma))
    if med_luma < 110.0:
        ocr_img = 255 - rgb
        preproc = 'inverted_for_dark_theme'
    else:
        ocr_img = rgb
        preproc = 'native_rgb'

    try:
        with tempfile.NamedTemporaryFile(suffix='.png') as tf:
            Image.fromarray(ocr_img, mode='RGB').save(tf.name, format='PNG')
            proc = subprocess.run(
                [tess, tf.name, 'stdout', '-l', lang, '--psm', str(psm), 'tsv'],
                capture_output=True, text=True, timeout=timeout_s,
            )
    except (OSError, subprocess.SubprocessError) as e:
        return {
            'status': OCR_ERROR,
            'engine': engine_ver,
            'confidence_scale': TESSERACT_CONF_SCALE,
            'source_frame': source_frame,
            'lang': lang,
            'preprocessing': preproc,
            'detail': f'tesseract execution failed: {e}',
            'items': [],
            'elapsed_ms': round((time.perf_counter() - t0) * 1000.0, 2),
        }

    if proc.returncode != 0:
        return {
            'status': OCR_ERROR,
            'engine': engine_ver,
            'confidence_scale': TESSERACT_CONF_SCALE,
            'source_frame': source_frame,
            'lang': lang,
            'preprocessing': preproc,
            'detail': (proc.stderr or proc.stdout or 'non-zero exit').strip()[:240],
            'items': [],
            'elapsed_ms': round((time.perf_counter() - t0) * 1000.0, 2),
        }

    # Group TSV word entries by (page, block, par, line)
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for ln in proc.stdout.splitlines()[1:]:
        cols = ln.split('\t')
        if len(cols) < 12:
            continue
        txt = cols[11].strip()
        if not txt:
            continue
        try:
            conf = float(cols[10])
            x, y, bw, bh = int(cols[6]), int(cols[7]), int(cols[8]), int(cols[9])
        except ValueError:
            continue
        if conf < 0 or bw <= 0 or bh <= 0:
            continue
        key = (cols[1], cols[2], cols[3], cols[4])
        grouped.setdefault(key, []).append({
            'x0': x, 'y0': y, 'x1': x + bw, 'y1': y + bh, 'text': txt, 'conf': conf,
        })

    items: list[dict[str, Any]] = []
    for words in grouped.values():
        words.sort(key=lambda w: w['x0'])
        x0 = min(w['x0'] for w in words)
        y0 = min(w['y0'] for w in words)
        x1 = max(w['x1'] for w in words)
        y1 = max(w['y1'] for w in words)
        text = ' '.join(w['text'] for w in words)
        mean_conf = round(sum(w['conf'] for w in words) / len(words), 1)
        min_conf = round(min(w['conf'] for w in words), 1)
        st = OCR_OBSERVED if mean_conf >= min_observed_conf else OCR_UNCERTAIN
        items.append({
            'text': text,
            'bbox': [offset_x + x0, offset_y + y0, offset_x + x1, offset_y + y1],
            'confidence': mean_conf,
            'min_word_confidence': min_conf,
            'confidence_scale': TESSERACT_CONF_SCALE,
            'engine': engine_ver,
            'source_frame': source_frame,
            'status': st,
        })

    items.sort(key=lambda r: (r['bbox'][1] // 10, r['bbox'][0]))
    for idx, it in enumerate(items):
        it['id'] = f'O{idx}'

    overall = OCR_OBSERVED if any(it['status'] == OCR_OBSERVED for it in items) else (
        OCR_UNCERTAIN if items else OCR_OBSERVED
    )
    return {
        'status': overall,
        'engine': engine_ver,
        'confidence_scale': TESSERACT_CONF_SCALE,
        'source_frame': source_frame,
        'lang': lang,
        'preprocessing': preproc,
        'detail': f'{len(items)} line(s) extracted; empty result does NOT prove absence of text',
        'items': items,
        'elapsed_ms': round((time.perf_counter() - t0) * 1000.0, 2),
    }


# ============================================================================
# Geometric text <-> control association (explicit geometry, no blind proximity)
# ============================================================================

def _intersection_area(a: list[int], b: list[int]) -> int:
    ix0 = max(a[0], b[0])
    iy0 = max(a[1], b[1])
    ix1 = min(a[2], b[2])
    iy1 = min(a[3], b[3])
    if ix1 <= ix0 or iy1 <= iy0:
        return 0
    return (ix1 - ix0) * (iy1 - iy0)


def _area(box: list[int]) -> int:
    return max(1, (box[2] - box[0]) * (box[3] - box[1]))


def _center(box: list[int]) -> tuple[float, float]:
    return (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0


def _panel_for_box(box: list[int], panels: list[dict[str, Any]]) -> str | None:
    best_id = None
    best_ratio = 0.0
    a = _area(box)
    for p in panels:
        r = _intersection_area(p['global_bbox'], box) / a
        if r >= 0.55 and r > best_ratio:
            best_ratio = r
            best_id = p['id']
    return best_id


def associate_text_to_controls(
    controls: list[dict[str, Any]],
    ocr_result: dict[str, Any],
    panels: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Associate OCR fragments to UI controls using explicit geometry & conflict detection.

    Rules:
      1. Containment (`inter / text_area >= 0.55`) -> candidate for `TEXT_INSIDE`.
         - If a text fragment has containment `>= 0.30` in >=2 distinct controls ->
           `AMBIGUOUS_ASSOCIATION` for both controls (never silently choose one).
         - Multiple disjoint fragments contained in the same control are merged in
           reading order `(y_band, x0)`.
      2. External label association (`LABEL_ASSOCIATED`):
         - Only for fragments NOT contained in any control.
         - Must be panel-compatible (same panel when both have panel assignments).
         - Must satisfy explicit UI label geometry:
           * LEFT-OF control: horizontal gap in `[-6, max(160, 2.5*ctrl_h)]` and vertical overlap >= 0.35.
           * ABOVE control: vertical gap in `[-6, max(44, 1.4*ctrl_h)]` and horizontal overlap >= 0.25 or left-edge alignment `<= 40px`.
         - If >=2 controls compete within score margin `0.15` (relative closeness) ->
           `AMBIGUOUS_ASSOCIATION` with explicit candidate list + geometry evidence!
      3. Controls with no matched OCR get `NO_TEXT_EVIDENCE` with explicit cause
         (`OCR_ABSENT`, `OCR_ERROR`, or `no_ocr_fragment_matched`).
    """
    t0 = time.perf_counter()
    panels = panels or []
    ocr_status = ocr_result.get('status', OCR_ABSENT)
    items = ocr_result.get('items', []) or []

    # Annotate OCR items with panel membership
    item_panel = {it['id']: _panel_for_box(it['bbox'], panels) for it in items}

    # Per-control accumulators
    inside_frags: dict[str, list[dict[str, Any]]] = {c['id']: [] for c in controls}
    label_frags: dict[str, list[dict[str, Any]]] = {c['id']: [] for c in controls}
    ambig_records: dict[str, list[dict[str, Any]]] = {c['id']: [] for c in controls}
    global_ambiguities: list[dict[str, Any]] = []

    if ocr_status in (OCR_ABSENT, OCR_ERROR) or not items:
        reason = ocr_status if ocr_status in (OCR_ABSENT, OCR_ERROR) else 'no_ocr_fragment_matched'
        enriched = []
        for c in controls:
            nc = dict(c)
            nc['text'] = None
            nc['text_source'] = ocr_status if ocr_status in (OCR_ABSENT, OCR_ERROR) else 'NONE'
            nc['text_confidence'] = None
            nc['association'] = ASSOC_NONE
            nc['association_reason'] = reason
            nc['ocr_ids'] = []
            nc['association_candidates'] = []
            enriched.append(nc)
        return {
            'controls': enriched,
            'unmatched_ocr_ids': [it['id'] for it in items],
            'ambiguities': [],
            'elapsed_ms': round((time.perf_counter() - t0) * 1000.0, 2),
        }

    assigned_ocr_ids: set[str] = set()

    # Phase 1: Containment / intersection (`TEXT_INSIDE`)
    external_items: list[dict[str, Any]] = []
    for it in items:
        tb = it['bbox']
        ta = _area(tb)
        tcx, tcy = _center(tb)
        hits = []
        for c in controls:
            cb = c['global_bbox']
            inter = _intersection_area(cb, tb)
            if inter <= 0:
                continue
            cov = inter / ta
            iou = inter / max(1, ta + _area(cb) - inter)
            ccx, ccy = _center(cb)
            cdist = math.hypot(tcx - ccx, tcy - ccy)
            if cov >= 0.30:
                hits.append({
                    'control_id': c['id'],
                    'coverage': round(cov, 3),
                    'iou': round(iou, 3),
                    'center_dist': round(cdist, 1),
                    'ctrl_area': _area(cb),
                })
        if not hits:
            external_items.append(it)
            continue

        # Prefer innermost (smallest) containing control when nested (e.g. button inside card),
        # UNLESS two non-nested controls both claim substantial overlap (>= 0.30).
        hits.sort(key=lambda h: (-h['coverage'], h['ctrl_area'], h['center_dist']))
        strong = [h for h in hits if h['coverage'] >= 0.55]
        if len(strong) >= 2:
            # Check if one control strictly contains the other (nested card vs inner button)
            c_map = {c['id']: c['global_bbox'] for c in controls}
            top0, top1 = strong[0], strong[1]
            b0, b1 = c_map[top0['control_id']], c_map[top1['control_id']]
            nested = (
                _intersection_area(b0, b1) / min(_area(b0), _area(b1)) >= 0.85
            )
            if nested:
                # Assign to the smaller (more specific) control
                winner = top0 if _area(b0) <= _area(b1) else top1
                inside_frags[winner['control_id']].append({'item': it, 'evidence': winner})
                assigned_ocr_ids.add(it['id'])
            else:
                rec = {
                    'ocr_id': it['id'],
                    'text': it['text'],
                    'bbox': tb,
                    'reason': 'text_fragment_overlaps_multiple_controls',
                    'candidates': strong,
                }
                global_ambiguities.append(rec)
                for h in strong:
                    ambig_records[h['control_id']].append(rec)
                assigned_ocr_ids.add(it['id'])
        elif len(strong) == 1:
            # Check if another non-nested control also has significant partial overlap (>= 0.35)
            other_conflict = [h for h in hits if h['control_id'] != strong[0]['control_id'] and h['coverage'] >= 0.35]
            c_map = {c['id']: c['global_bbox'] for c in controls}
            b0 = c_map[strong[0]['control_id']]
            real_conflicts = [
                h for h in other_conflict
                if _intersection_area(b0, c_map[h['control_id']]) / min(_area(b0), _area(c_map[h['control_id']])) < 0.85
            ]
            if real_conflicts:
                rec = {
                    'ocr_id': it['id'],
                    'text': it['text'],
                    'bbox': tb,
                    'reason': 'partial_overlap_conflict_between_controls',
                    'candidates': [strong[0]] + real_conflicts,
                }
                global_ambiguities.append(rec)
                for h in rec['candidates']:
                    ambig_records[h['control_id']].append(rec)
                assigned_ocr_ids.add(it['id'])
            else:
                inside_frags[strong[0]['control_id']].append({'item': it, 'evidence': strong[0]})
                assigned_ocr_ids.add(it['id'])
        else:
            # All overlaps in [0.30, 0.55): ambiguous straddling border
            rec = {
                'ocr_id': it['id'],
                'text': it['text'],
                'bbox': tb,
                'reason': 'partial_border_straddle',
                'candidates': hits,
            }
            global_ambiguities.append(rec)
            for h in hits:
                ambig_records[h['control_id']].append(rec)
            assigned_ocr_ids.add(it['id'])

    # Phase 2: External label association (`LABEL_ASSOCIATED` vs `AMBIGUOUS_ASSOCIATION`)
    for it in external_items:
        tb = it['bbox']
        tx0, ty0, tx1, ty1 = tb
        tw, th = max(1, tx1 - tx0), max(1, ty1 - ty0)
        tcx, tcy = _center(tb)
        tp = item_panel.get(it['id'])

        cands = []
        for c in controls:
            # Prefer associating external labels with controls that don't already have strong
            # full-text inside, OR inputs/bordered boxes (but still evaluate all for conflict detection!)
            cb = c['global_bbox']
            cx0, cy0, cx1, cy1 = cb
            cw, ch = max(1, cx1 - cx0), max(1, cy1 - cy0)
            ccx, ccy = _center(cb)
            cp = c.get('inside_panel')
            if tp is not None and cp is not None and tp != cp:
                continue  # Never associate a label across different UI panels

            # Relation A: Label immediately LEFT of control
            v_ov = max(0, min(ty1, cy1) - max(ty0, cy0)) / min(th, ch)
            h_gap_left = cx0 - tx1
            left_limit = max(160.0, 2.5 * ch)
            if -6 <= h_gap_left <= left_limit and v_ov >= 0.35:
                gap = max(0.0, float(h_gap_left))
                score = max(0.0, 1.0 - gap / (left_limit + 1.0)) * (0.65 + 0.35 * v_ov)
                cands.append({
                    'control_id': c['id'],
                    'relation': 'left_of',
                    'edge_gap_px': round(gap, 1),
                    'center_dist_px': round(math.hypot(tcx - ccx, tcy - ccy), 1),
                    'alignment': round(v_ov, 2),
                    'score': round(score, 3),
                })
                continue

            # Relation B: Label immediately ABOVE control
            h_ov = max(0, min(tx1, cx1) - max(tx0, cx0)) / min(tw, cw)
            left_align = abs(tx0 - cx0) <= 44
            v_gap_above = cy0 - ty1
            above_limit = max(44.0, 1.4 * ch)
            if -6 <= v_gap_above <= above_limit and (h_ov >= 0.25 or left_align):
                gap = max(0.0, float(v_gap_above))
                align_bonus = max(h_ov, 1.0 - abs(tx0 - cx0) / 50.0 if left_align else 0.0)
                score = max(0.0, 1.0 - gap / (above_limit + 1.0)) * (0.60 + 0.40 * align_bonus)
                cands.append({
                    'control_id': c['id'],
                    'relation': 'above',
                    'edge_gap_px': round(gap, 1),
                    'center_dist_px': round(math.hypot(tcx - ccx, tcy - ccy), 1),
                    'alignment': round(align_bonus, 2),
                    'score': round(score, 3),
                })

        if not cands:
            continue
        cands.sort(key=lambda z: (-z['score'], z['edge_gap_px'], z['center_dist_px']))
        if len(cands) >= 2 and (cands[0]['score'] - cands[1]['score']) <= 0.15:
            # Ambiguous external label between >=2 nearby controls: NEVER pick arbitrarily
            rec = {
                'ocr_id': it['id'],
                'text': it['text'],
                'bbox': tb,
                'reason': 'external_label_ambiguous_between_controls',
                'candidates': cands[:3],
            }
            global_ambiguities.append(rec)
            for h in cands[:2]:
                ambig_records[h['control_id']].append(rec)
            assigned_ocr_ids.add(it['id'])
        elif cands[0]['score'] >= 0.35:
            label_frags[cands[0]['control_id']].append({'item': it, 'evidence': cands[0]})
            assigned_ocr_ids.add(it['id'])

    # Phase 3: Assemble per-control text & association status
    enriched_controls: list[dict[str, Any]] = []
    for c in controls:
        cid = c['id']
        nc = dict(c)
        ins = inside_frags[cid]
        lbl = label_frags[cid]
        amb = ambig_records[cid]

        if amb and not ins:
            nc['text'] = None
            nc['text_source'] = OCR_UNCERTAIN
            nc['text_confidence'] = None
            nc['association'] = ASSOC_AMBIGUOUS
            nc['association_reason'] = amb[0]['reason']
            nc['ocr_ids'] = [a['ocr_id'] for a in amb]
            nc['association_candidates'] = amb
        elif ins:
            ins.sort(key=lambda x: (x['item']['bbox'][1] // 10, x['item']['bbox'][0]))
            merged_txt = ' '.join(x['item']['text'] for x in ins)
            min_conf = min(x['item']['confidence'] for x in ins)
            worst_st = OCR_UNCERTAIN if any(x['item']['status'] == OCR_UNCERTAIN for x in ins) else OCR_OBSERVED
            nc['text'] = merged_txt
            nc['text_source'] = worst_st
            nc['text_confidence'] = min_conf
            nc['association'] = ASSOC_INSIDE
            nc['association_reason'] = f'contained_fragments={len(ins)}'
            nc['ocr_ids'] = [x['item']['id'] for x in ins]
            nc['association_candidates'] = amb
            if lbl:
                lbl.sort(key=lambda x: x['evidence']['edge_gap_px'])
                nc['external_label'] = ' '.join(x['item']['text'] for x in lbl)
        elif lbl:
            lbl.sort(key=lambda x: (x['evidence']['edge_gap_px'], x['item']['bbox'][0]))
            merged_txt = ' '.join(x['item']['text'] for x in lbl)
            min_conf = min(x['item']['confidence'] for x in lbl)
            worst_st = OCR_UNCERTAIN if any(x['item']['status'] == OCR_UNCERTAIN for x in lbl) else OCR_OBSERVED
            nc['text'] = merged_txt
            nc['text_source'] = worst_st
            nc['text_confidence'] = min_conf
            nc['association'] = ASSOC_LABEL
            nc['association_reason'] = f'{lbl[0]["evidence"]["relation"]} gap={lbl[0]["evidence"]["edge_gap_px"]}px'
            nc['ocr_ids'] = [x['item']['id'] for x in lbl]
            nc['association_candidates'] = []
        else:
            nc['text'] = None
            nc['text_source'] = 'NONE'
            nc['text_confidence'] = None
            nc['association'] = ASSOC_NONE
            nc['association_reason'] = 'no_ocr_fragment_matched'
            nc['ocr_ids'] = []
            nc['association_candidates'] = []

        enriched_controls.append(nc)

    unmatched = [it['id'] for it in items if it['id'] not in assigned_ocr_ids]
    return {
        'controls': enriched_controls,
        'unmatched_ocr_ids': unmatched,
        'ambiguities': global_ambiguities,
        'elapsed_ms': round((time.perf_counter() - t0) * 1000.0, 2),
    }
