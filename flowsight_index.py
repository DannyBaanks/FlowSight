#!/usr/bin/env python3
"""FlowSight M5-A — Progressive Visual Index with Verifiable Geometry (`flowsight_index.py`).

Builds an explicit, opt-in, lifecycle-bounded SQLite + FTS5 index over
screenshots so text-only agents do NOT dump 7+ full screen reports into context:

  1. Index   (explicit paths only; never background-scans private folders;
              stores metadata + layout/controls + OCR bboxes, NEVER image bytes)
  2. Filter  (FTS5 BM25 + exact technical-token matching + theme/color/time filters)
  3. Select  (compact ranked candidates with matched OCR evidence + exact global bboxes
              + suggested padded `crop=x:y:w:h`)
  4. Deepen  (zoom ONLY into the winning candidate crop preserving global coordinates)

Privacy & lifecycle guarantees:
  - Default DB lives outside any git repo:
    ${FLOWSIGHT_INDEX_DB:-${XDG_DATA_HOME:-~/.local/share}/flowsight/progressive_index.sqlite}
  - Directory mode 0700, DB file mode 0600 (POSIX).
  - Ephemeral by default (`ttl_hours=24.0`); expired entries are pruned automatically.
  - Explicit `expire_stale()` and `purge()` operations.
  - Deduplicates by image SHA-256 + OCR mode so already-indexed images skip redundant OCR.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import sys
import time
from typing import Any, Iterable

import flowsight_screen as fscr

SCHEMA_INDEX = 'flowsight-progressive-index-v1'
IMAGE_EXTS = frozenset({'.png', '.jpg', '.jpeg', '.webp'})


def default_index_db() -> Path:
    override = os.environ.get('FLOWSIGHT_INDEX_DB', '').strip()
    if override:
        return Path(override).expanduser().resolve()
    xdg = os.environ.get('XDG_DATA_HOME', '').strip()
    base = Path(xdg).expanduser() if xdg else (Path.home() / '.local' / 'share')
    return (base / 'flowsight' / 'progressive_index.sqlite').resolve()


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _iso_utc(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def _ensure_private_parent(db_path: Path) -> None:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if os.name == 'posix':
        try:
            os.chmod(db_path.parent, 0o700)
        except OSError:
            pass


def _chmod_private_file(path: Path) -> None:
    if os.name == 'posix' and path.exists():
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass


def _connect(db_path: Path) -> sqlite3.Connection:
    _ensure_private_parent(db_path)
    con = sqlite3.connect(str(db_path))
    con.row_factory = sqlite3.Row
    con.execute('PRAGMA foreign_keys = ON')
    con.execute('PRAGMA journal_mode = WAL')
    _init_schema(con)
    _chmod_private_file(db_path)
    return con


def _init_schema(con: sqlite3.Connection) -> None:
    con.executescript(
        """
        CREATE TABLE IF NOT EXISTS meta (
            k TEXT PRIMARY KEY,
            v TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS frames (
            frame_id INTEGER PRIMARY KEY AUTOINCREMENT,
            sha256 TEXT NOT NULL,
            ocr_enabled INTEGER NOT NULL,
            path TEXT NOT NULL,
            source_name TEXT NOT NULL,
            width INTEGER NOT NULL,
            height INTEGER NOT NULL,
            mtime_utc TEXT NOT NULL,
            indexed_utc TEXT NOT NULL,
            expires_utc TEXT NOT NULL,
            theme_polarity TEXT NOT NULL,
            median_luma REAL NOT NULL,
            ocr_status TEXT NOT NULL,
            ocr_engine TEXT NOT NULL,
            panel_count INTEGER NOT NULL,
            control_count INTEGER NOT NULL,
            ocr_item_count INTEGER NOT NULL,
            accent_families TEXT NOT NULL,
            UNIQUE(sha256, ocr_enabled)
        );

        CREATE TABLE IF NOT EXISTS controls (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            frame_id INTEGER NOT NULL REFERENCES frames(frame_id) ON DELETE CASCADE,
            local_id TEXT NOT NULL,
            kind TEXT NOT NULL,
            x0 INTEGER NOT NULL,
            y0 INTEGER NOT NULL,
            x1 INTEGER NOT NULL,
            y1 INTEGER NOT NULL,
            click_x INTEGER NOT NULL,
            click_y INTEGER NOT NULL,
            hex_color TEXT NOT NULL,
            color_family TEXT NOT NULL,
            panel_id TEXT,
            text TEXT,
            text_source TEXT,
            association TEXT
        );

        CREATE TABLE IF NOT EXISTS ocr_items (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            frame_id INTEGER NOT NULL REFERENCES frames(frame_id) ON DELETE CASCADE,
            ocr_id TEXT NOT NULL,
            status TEXT NOT NULL,
            x0 INTEGER NOT NULL,
            y0 INTEGER NOT NULL,
            x1 INTEGER NOT NULL,
            y1 INTEGER NOT NULL,
            confidence REAL NOT NULL,
            text TEXT NOT NULL,
            control_id TEXT
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS ocr_fts USING fts5(
            text,
            frame_id UNINDEXED,
            ocr_row_id UNINDEXED,
            tokenize="unicode61 tokenchars '-_./:'"
        );

        CREATE INDEX IF NOT EXISTS idx_frames_expires ON frames(expires_utc);
        CREATE INDEX IF NOT EXISTS idx_frames_theme ON frames(theme_polarity);
        CREATE INDEX IF NOT EXISTS idx_controls_frame ON controls(frame_id);
        CREATE INDEX IF NOT EXISTS idx_controls_color ON controls(color_family);
        CREATE INDEX IF NOT EXISTS idx_ocr_frame ON ocr_items(frame_id);
        """
    )
    con.execute('INSERT OR IGNORE INTO meta(k, v) VALUES (?, ?)', ('schema', SCHEMA_INDEX))
    con.commit()


def _delete_frame_fts(con: sqlite3.Connection, frame_ids: list[int]) -> None:
    if not frame_ids:
        return
    placeholders = ','.join('?' for _ in frame_ids)
    con.execute(f'DELETE FROM ocr_fts WHERE frame_id IN ({placeholders})', frame_ids)


def expire_stale(db_path: Path | None = None, *, now: datetime | None = None) -> int:
    """Remove all expired entries from the index. Returns count of removed frames."""
    path = db_path or default_index_db()
    if not path.is_file():
        return 0
    cutoff = _iso_utc(now or _utc_now())
    with _connect(path) as con:
        rows = con.execute('SELECT frame_id FROM frames WHERE expires_utc <= ?', (cutoff,)).fetchall()
        fids = [int(r['frame_id']) for r in rows]
        if not fids:
            return 0
        _delete_frame_fts(con, fids)
        placeholders = ','.join('?' for _ in fids)
        con.execute(f'DELETE FROM frames WHERE frame_id IN ({placeholders})', fids)
        con.commit()
        return len(fids)


def purge_index(
    db_path: Path | None = None,
    *,
    mode: str = 'expired',
    sha256: str = '',
    path_prefix: str = '',
) -> dict[str, Any]:
    """Explicitly purge index entries ('expired', 'all', 'sha256', or 'path_prefix')."""
    path = db_path or default_index_db()
    if not path.is_file():
        return {'db_path': str(path), 'mode': mode, 'removed': 0, 'remaining': 0}

    if mode == 'expired':
        removed = expire_stale(path)
        with _connect(path) as con:
            rem = int(con.execute('SELECT COUNT(*) FROM frames').fetchone()[0])
        return {'db_path': str(path), 'mode': mode, 'removed': removed, 'remaining': rem}

    with _connect(path) as con:
        if mode == 'all':
            rows = con.execute('SELECT frame_id FROM frames').fetchall()
        elif mode == 'sha256':
            pref = sha256.strip().lower()
            if not pref:
                raise ValueError('purge mode="sha256" requires a non-empty sha256 prefix')
            rows = con.execute('SELECT frame_id FROM frames WHERE LOWER(sha256) LIKE ?', (pref + '%',)).fetchall()
        elif mode == 'path_prefix':
            pref = str(Path(path_prefix).expanduser().resolve()) if path_prefix.strip() else ''
            if not pref:
                raise ValueError('purge mode="path_prefix" requires a non-empty path_prefix')
            rows = con.execute('SELECT frame_id FROM frames WHERE path LIKE ?', (pref + '%',)).fetchall()
        else:
            raise ValueError(f'unknown purge mode {mode!r}; use expired, all, sha256, or path_prefix')

        fids = [int(r['frame_id']) for r in rows]
        if fids:
            _delete_frame_fts(con, fids)
            placeholders = ','.join('?' for _ in fids)
            con.execute(f'DELETE FROM frames WHERE frame_id IN ({placeholders})', fids)
            con.commit()
        rem = int(con.execute('SELECT COUNT(*) FROM frames').fetchone()[0])
    return {'db_path': str(path), 'mode': mode, 'removed': len(fids), 'remaining': rem}


def _expand_explicit_paths(paths: Iterable[str | Path], max_files: int) -> list[Path]:
    """Resolve explicit files or immediate children of explicit directories (never recursive)."""
    out: list[Path] = []
    seen: set[Path] = set()
    cap = max(1, min(500, int(max_files)))
    for raw in paths:
        p = Path(raw).expanduser().resolve()
        if p.is_file():
            if p.suffix.lower() in IMAGE_EXTS and p not in seen:
                seen.add(p)
                out.append(p)
        elif p.is_dir():
            # Non-recursive, newest mtime first so recent screenshots are prioritized
            candidates = [c for c in p.iterdir() if c.is_file() and c.suffix.lower() in IMAGE_EXTS]
            candidates.sort(key=lambda c: (-c.stat().st_mtime, c.name))
            for c in candidates:
                rc = c.resolve()
                if rc not in seen:
                    seen.add(rc)
                    out.append(rc)
                if len(out) >= cap:
                    break
        if len(out) >= cap:
            break
    return out[:cap]


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def index_paths(
    paths: Iterable[str | Path],
    *,
    use_ocr: bool = True,
    ttl_hours: float = 24.0,
    max_files: int = 64,
    db_path: Path | None = None,
    ocr_provider: Any = None,
) -> dict[str, Any]:
    """Explicitly index one or more screenshot files/directories into the local progressive index."""
    t0 = time.perf_counter()
    target_db = db_path or default_index_db()
    ttl = max(0.05, min(24.0 * 30.0, float(ttl_hours)))
    files = _expand_explicit_paths(paths, max_files=max_files)
    now = _utc_now()
    expires = _iso_utc(now + timedelta(hours=ttl))
    indexed_iso = _iso_utc(now)
    ocr_flag = 1 if use_ocr else 0

    expire_stale(target_db, now=now)

    indexed_count = 0
    reused_count = 0
    errors: list[dict[str, str]] = []
    entries: list[dict[str, Any]] = []

    vf = 'theme,layout,controls,ocr' if use_ocr else 'theme,layout,controls'

    with _connect(target_db) as con:
        for img_path in files:
            try:
                sha = _sha256_file(img_path)
                st = img_path.stat()
                mtime_iso = _iso_utc(datetime.fromtimestamp(st.st_mtime, tz=timezone.utc))

                existing = con.execute(
                    'SELECT frame_id, width, height, theme_polarity, ocr_status, control_count, ocr_item_count '
                    'FROM frames WHERE sha256 = ? AND ocr_enabled = ?',
                    (sha, ocr_flag),
                ).fetchone()
                if existing is not None:
                    con.execute(
                        'UPDATE frames SET path = ?, source_name = ?, mtime_utc = ?, indexed_utc = ?, expires_utc = ? '
                        'WHERE frame_id = ?',
                        (str(img_path), img_path.name, mtime_iso, indexed_iso, expires, int(existing['frame_id'])),
                    )
                    reused_count += 1
                    entries.append({
                        'frame_id': int(existing['frame_id']),
                        'source_name': img_path.name,
                        'sha256': sha[:16],
                        'cached': True,
                        'theme': existing['theme_polarity'],
                        'ocr_status': existing['ocr_status'],
                        'controls': int(existing['control_count']),
                        'ocr_items': int(existing['ocr_item_count']),
                    })
                    continue

                ctx = fscr.run_pipeline(img_path, vf=vf, ocr_provider=ocr_provider)
                th = ctx.artifacts.get('theme', {})
                lay = ctx.artifacts.get('layout', {})
                ct = ctx.artifacts.get('controls', {})
                oc = ctx.artifacts.get('ocr', {})

                accents = sorted({a.get('family', '') for a in th.get('accent_colors', []) if a.get('family')})
                controls = ct.get('controls', [])
                ocr_items = oc.get('items', []) if use_ocr else []

                # Map OCR item id -> associated control local_id
                ocr_to_ctl: dict[str, str] = {}
                for ctl in controls:
                    for oid in ctl.get('matched_ocr_ids', []) or []:
                        ocr_to_ctl[str(oid)] = str(ctl['id'])

                cur = con.execute(
                    """
                    INSERT INTO frames (
                        sha256, ocr_enabled, path, source_name, width, height,
                        mtime_utc, indexed_utc, expires_utc,
                        theme_polarity, median_luma, ocr_status, ocr_engine,
                        panel_count, control_count, ocr_item_count, accent_families
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        sha, ocr_flag, str(img_path), img_path.name, ctx.full_w, ctx.full_h,
                        mtime_iso, indexed_iso, expires,
                        str(th.get('polarity', 'unknown')), float(th.get('median_luma', 0.0)),
                        str(oc.get('status', 'OCR_SKIPPED' if not use_ocr else 'OCR_ABSENT')),
                        str(oc.get('engine', 'none')),
                        len(lay.get('panels', [])), len(controls), len(ocr_items),
                        ','.join(accents),
                    ),
                )
                fid = int(cur.lastrowid)

                for ctl in controls:
                    gb = ctl['global_bbox']
                    cc = ctl['click_center']
                    con.execute(
                        """
                        INSERT INTO controls (
                            frame_id, local_id, kind, x0, y0, x1, y1,
                            click_x, click_y, hex_color, color_family,
                            panel_id, text, text_source, association
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            fid, ctl['id'], ctl['kind'],
                            int(gb[0]), int(gb[1]), int(gb[2]), int(gb[3]),
                            int(cc[0]), int(cc[1]), ctl['hex'], ctl['family'],
                            ctl.get('inside_panel'), ctl.get('text'),
                            ctl.get('text_source'), ctl.get('association'),
                        ),
                    )

                for item in ocr_items:
                    bb = item['bbox']
                    cid = ocr_to_ctl.get(str(item['id']))
                    c_ocr = con.execute(
                        """
                        INSERT INTO ocr_items (
                            frame_id, ocr_id, status, x0, y0, x1, y1,
                            confidence, text, control_id
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            fid, str(item['id']), str(item['status']),
                            int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3]),
                            float(item.get('confidence', 0.0)),
                            str(item['text']), cid,
                        ),
                    )
                    ocr_row_id = int(c_ocr.lastrowid)
                    con.execute(
                        'INSERT INTO ocr_fts(text, frame_id, ocr_row_id) VALUES (?, ?, ?)',
                        (str(item['text']), fid, ocr_row_id),
                    )

                indexed_count += 1
                entries.append({
                    'frame_id': fid,
                    'source_name': img_path.name,
                    'sha256': sha[:16],
                    'cached': False,
                    'theme': str(th.get('polarity', 'unknown')),
                    'ocr_status': str(oc.get('status', 'OCR_SKIPPED' if not use_ocr else 'OCR_ABSENT')),
                    'controls': len(controls),
                    'ocr_items': len(ocr_items),
                })
            except Exception as exc:
                errors.append({'path': str(img_path), 'error': f'{type(exc).__name__}: {exc}'})
        con.commit()

    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    db_bytes = target_db.stat().st_size if target_db.is_file() else 0
    return {
        'schema': SCHEMA_INDEX,
        'db_path': str(target_db),
        'ttl_hours': ttl,
        'expires_utc': expires,
        'total_input_files': len(files),
        'indexed_new': indexed_count,
        'reused_cached': reused_count,
        'errors': errors,
        'elapsed_ms': elapsed_ms,
        'db_bytes': db_bytes,
        'entries': entries,
    }


_STOPWORDS = frozenset({
    'a', 'al', 'and', 'con', 'de', 'del', 'el', 'en', 'for', 'in', 'la', 'las',
    'los', 'of', 'on', 'or', 'para', 'por', 'que', 'the', 'to', 'un', 'una', 'with', 'y',
})


def _extract_query_terms(query: str) -> list[str]:
    raw = re.findall(r'[A-Za-z0-9_./:-]+', query or '')
    terms: list[str] = []
    seen: set[str] = set()
    for tok in raw:
        clean = tok.strip('.,;:"\'()[]{}')
        low = clean.lower()
        if not low or low in _STOPWORDS:
            continue
        if len(low) == 1 and not low.isdigit():
            continue
        if low not in seen:
            seen.add(low)
            terms.append(low)
    return terms


def _pad_union_bbox(boxes: list[list[int]], width: int, height: int, pad: int = 36) -> list[int]:
    x0 = max(0, min(b[0] for b in boxes) - pad)
    y0 = max(0, min(b[1] for b in boxes) - pad)
    x1 = min(width, max(b[2] for b in boxes) + pad)
    y1 = min(height, max(b[3] for b in boxes) + pad)
    return [x0, y0, max(x0 + 1, x1), max(y0 + 1, y1)]


def search_index(
    query: str = '',
    *,
    theme: str = '',
    color: str = '',
    since: str = '',
    until: str = '',
    limit: int = 5,
    db_path: Path | None = None,
) -> dict[str, Any]:
    """Progressive search over the local index returning ranked candidates with exact bboxes & crop hints."""
    t0 = time.perf_counter()
    target_db = db_path or default_index_db()
    if not target_db.is_file():
        return {
            'schema': SCHEMA_INDEX,
            'db_path': str(target_db),
            'query': query,
            'terms': _extract_query_terms(query),
            'total_indexed_frames': 0,
            'candidates': [],
            'elapsed_ms': 0.0,
        }

    expire_stale(target_db)
    terms = _extract_query_terms(query)
    theme_f = (theme or '').strip().lower()
    color_f = (color or '').strip().lower()
    since_f = (since or '').strip()
    until_f = (until or '').strip()
    top_k = max(1, min(30, int(limit)))

    with _connect(target_db) as con:
        total_frames = int(con.execute('SELECT COUNT(*) FROM frames').fetchone()[0])
        where = ['1=1']
        params: list[Any] = []
        if theme_f in ('dark', 'light', 'mixed'):
            where.append('LOWER(theme_polarity) = ?')
            params.append(theme_f)
        if since_f:
            where.append('mtime_utc >= ?')
            params.append(since_f)
        if until_f:
            where.append('mtime_utc <= ?')
            params.append(until_f)

        frame_rows = con.execute(
            f'SELECT * FROM frames WHERE {" AND ".join(where)} ORDER BY mtime_utc DESC',
            params,
        ).fetchall()

        # Precompute FTS5 BM25 boost per frame when terms exist
        fts_boost: dict[int, float] = {}
        if terms:
            # Quote each term safely for FTS5 OR query
            fts_expr = ' OR '.join('"' + t.replace('"', '""') + '"' for t in terms)
            try:
                fts_rows = con.execute(
                    'SELECT frame_id, bm25(ocr_fts) AS rank FROM ocr_fts WHERE ocr_fts MATCH ?',
                    (fts_expr,),
                ).fetchall()
                for r in fts_rows:
                    fid = int(r['frame_id'])
                    # SQLite bm25 is negative (more negative = stronger match)
                    score_add = max(0.1, min(5.0, -float(r['rank']) + 1.0))
                    fts_boost[fid] = fts_boost.get(fid, 0.0) + score_add
            except sqlite3.OperationalError:
                pass

        candidates: list[dict[str, Any]] = []
        for fr in frame_rows:
            fid = int(fr['frame_id'])
            ctls = con.execute('SELECT * FROM controls WHERE frame_id = ?', (fid,)).fetchall()
            ocrs = con.execute('SELECT * FROM ocr_items WHERE frame_id = ?', (fid,)).fetchall()

            # Optional color filter against controls OR dominant accent families
            matched_color_controls: list[dict[str, Any]] = []
            if color_f:
                for c in ctls:
                    if color_f in str(c['color_family']).lower():
                        matched_color_controls.append(dict(c))
                acc_str = str(fr['accent_families']).lower()
                if not matched_color_controls and color_f not in acc_str:
                    continue

            matched_ocr: list[dict[str, Any]] = []
            matched_terms_set: set[str] = set()
            if terms:
                for item in ocrs:
                    low_txt = str(item['text']).lower()
                    hit_terms = [t for t in terms if t in low_txt]
                    if hit_terms:
                        matched_terms_set.update(hit_terms)
                        matched_ocr.append({
                            'ocr_id': item['ocr_id'],
                            'status': item['status'],
                            'text': item['text'],
                            'conf': round(float(item['confidence']), 1),
                            'bbox': [int(item['x0']), int(item['y0']), int(item['x1']), int(item['y1'])],
                            'control_id': item['control_id'],
                            'hit_terms': hit_terms,
                        })
                if not matched_ocr:
                    continue

            # Deterministic interpretable match score
            term_cov = (len(matched_terms_set) / len(terms)) if terms else 1.0
            ctl_hits = sum(1 for m in matched_ocr if m['control_id'])
            bm25_component = min(1.0, fts_boost.get(fid, 0.0) / 10.0)
            score = round(
                term_cov * 0.70
                + bm25_component * 0.20
                + min(0.10, ctl_hits * 0.05)
                + (0.05 if matched_color_controls else 0.0),
                4,
            )

            # Compute suggested padded crop around matched evidence
            ev_boxes = [m['bbox'] for m in matched_ocr]
            if not ev_boxes and matched_color_controls:
                ev_boxes = [[int(c['x0']), int(c['y0']), int(c['x1']), int(c['y1'])] for c in matched_color_controls]
            suggested_crop_box = (
                _pad_union_bbox(ev_boxes, int(fr['width']), int(fr['height']), pad=36)
                if ev_boxes else [0, 0, int(fr['width']), int(fr['height'])]
            )
            cw = suggested_crop_box[2] - suggested_crop_box[0]
            ch = suggested_crop_box[3] - suggested_crop_box[1]
            crop_vf = f'crop={suggested_crop_box[0]}:{suggested_crop_box[1]}:{cw}:{ch}'

            candidates.append({
                'frame_id': fid,
                'source_name': fr['source_name'],
                'path': fr['path'],
                'sha256': str(fr['sha256'])[:16],
                'size': [int(fr['width']), int(fr['height'])],
                'mtime_utc': fr['mtime_utc'],
                'theme': fr['theme_polarity'],
                'ocr_status': fr['ocr_status'],
                'score': score,
                'matched_terms': sorted(matched_terms_set),
                'term_coverage': round(term_cov, 3),
                'matched_ocr': matched_ocr[:8],
                'matched_color_controls': [
                    {
                        'id': c['local_id'],
                        'color': f'{c["hex_color"]}({c["color_family"]})',
                        'bbox': [int(c['x0']), int(c['y0']), int(c['x1']), int(c['y1'])],
                        'text': c['text'],
                    }
                    for c in matched_color_controls[:4]
                ],
                'suggested_crop_bbox': suggested_crop_box,
                'suggested_crop_vf': crop_vf,
            })

        candidates.sort(key=lambda c: (-c['score'], -len(c['matched_terms']), c['mtime_utc']), reverse=False)
        # Tie-break newest mtime first
        candidates.sort(key=lambda c: (-c['score'], -len(c['matched_terms'])))
        shown = candidates[:top_k]

    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 2)
    return {
        'schema': SCHEMA_INDEX,
        'db_path': str(target_db),
        'query': query,
        'terms': terms,
        'filters': {'theme': theme_f or None, 'color': color_f or None, 'since': since_f or None, 'until': until_f or None},
        'total_indexed_frames': total_frames,
        'total_matches': len(candidates),
        'candidates': shown,
        'elapsed_ms': elapsed_ms,
    }


def index_stats(db_path: Path | None = None) -> dict[str, Any]:
    path = db_path or default_index_db()
    if not path.is_file():
        return {'db_path': str(path), 'exists': False, 'frames': 0, 'controls': 0, 'ocr_items': 0, 'db_bytes': 0}
    expire_stale(path)
    with _connect(path) as con:
        frames = int(con.execute('SELECT COUNT(*) FROM frames').fetchone()[0])
        controls = int(con.execute('SELECT COUNT(*) FROM controls').fetchone()[0])
        ocrs = int(con.execute('SELECT COUNT(*) FROM ocr_items').fetchone()[0])
    return {
        'db_path': str(path),
        'exists': True,
        'frames': frames,
        'controls': controls,
        'ocr_items': ocrs,
        'db_bytes': path.stat().st_size,
    }


def format_search_report(res: dict[str, Any], max_chars: int = 4500) -> str:
    lines = [
        f'FLOWSIGHT PROGRESSIVE SEARCH v1 | query={res["query"]!r} | terms={res["terms"]} | '
        f'matches={res.get("total_matches", len(res["candidates"]))}/{res["total_indexed_frames"]} ({res["elapsed_ms"]}ms)',
        'Provenance: DEMONSTRATED OCR/geometry index hits (bbox=[x0,y0,x1,y1) EXCLUSIVE). Deepen ONLY winning crop(s).',
    ]
    if not res['candidates']:
        lines.append('  (no indexed frames matched the query/filters)')
        return '\n'.join(lines)
    for rank, c in enumerate(res['candidates'], start=1):
        lines.append(
            f'#{rank} score={c["score"]:.3f} cov={c["term_coverage"]:.2f} | {c["source_name"]} '
            f'({c["size"][0]}x{c["size"][1]}, theme={c["theme"]}, sha={c["sha256"]}, mtime={c["mtime_utc"]})'
        )
        lines.append(f'   path={c["path"]}')
        lines.append(f'   deepen_vf="{c["suggested_crop_vf"]},controls,ocr" (union_bbox={c["suggested_crop_bbox"]})')
        for m in c['matched_ocr'][:5]:
            ctl_s = f' ctl={m["control_id"]}' if m['control_id'] else ''
            lines.append(f'   - {m["ocr_id"]} [{m["status"]}] bbox={m["bbox"]} conf={m["conf"]}{ctl_s} text={m["text"]!r}')
        for mc in c['matched_color_controls'][:3]:
            lines.append(f'   - color_ctl {mc["id"]} {mc["color"]} bbox={mc["bbox"]} text={mc["text"]!r}')
    txt = '\n'.join(lines)
    if len(txt) > max_chars:
        txt = txt[:max(0, max_chars - 32)] + '\n[TRUNCATED: lower --limit]'
    return txt


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description='FlowSight M5-A: Progressive Visual Index (SQLite + FTS5)')
    p.add_argument('--db', default=None, help='Override index SQLite path')
    sub = p.add_subparsers(dest='cmd', required=True)

    pa = sub.add_parser('add', help='Explicitly index image file(s) or directory')
    pa.add_argument('paths', nargs='+')
    pa.add_argument('--no-ocr', action='store_true')
    pa.add_argument('--ttl-hours', type=float, default=24.0)
    pa.add_argument('--max-files', type=int, default=64)

    ps = sub.add_parser('search', help='Progressive search over indexed frames')
    ps.add_argument('query', nargs='?', default='')
    ps.add_argument('--theme', default='')
    ps.add_argument('--color', default='')
    ps.add_argument('--since', default='')
    ps.add_argument('--until', default='')
    ps.add_argument('--limit', type=int, default=5)
    ps.add_argument('--json', action='store_true')

    sub.add_parser('stats', help='Show local index statistics')
    sub.add_parser('expire', help='Delete expired entries')

    pp = sub.add_parser('purge', help='Purge index entries explicitly')
    pp.add_argument('--mode', default='all', choices=('expired', 'all', 'sha256', 'path_prefix'))
    pp.add_argument('--sha256', default='')
    pp.add_argument('--path-prefix', default='')

    args = p.parse_args(argv)
    dbp = Path(args.db).expanduser().resolve() if args.db else None

    if args.cmd == 'add':
        res = index_paths(args.paths, use_ocr=not args.no_ocr, ttl_hours=args.ttl_hours, max_files=args.max_files, db_path=dbp)
        print(json.dumps(res, indent=2, ensure_ascii=False))
        return 0
    if args.cmd == 'search':
        res = search_index(args.query, theme=args.theme, color=args.color, since=args.since, until=args.until, limit=args.limit, db_path=dbp)
        if args.json:
            print(json.dumps(res, indent=2, ensure_ascii=False))
        else:
            print(format_search_report(res))
        return 0
    if args.cmd == 'stats':
        print(json.dumps(index_stats(dbp), indent=2, ensure_ascii=False))
        return 0
    if args.cmd == 'expire':
        removed = expire_stale(dbp)
        print(json.dumps({'removed': removed, 'db_path': str(dbp or default_index_db())}, indent=2))
        return 0
    if args.cmd == 'purge':
        res = purge_index(dbp, mode=args.mode, sha256=args.sha256, path_prefix=args.path_prefix)
        print(json.dumps(res, indent=2, ensure_ascii=False))
        return 0
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
