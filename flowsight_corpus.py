#!/usr/bin/env python3
"""FlowSight Private Per-PC Failure Corpus (`flowsight_corpus.py`).

Stores local perception/OCR/layout/diff failure reports confirmed by the user
(e.g. after `ask_user` confirmation in ISyCode TUI) into a strictly local,
per-machine private directory OUTSIDE any git repository:

  ${FLOWSIGHT_PRIVATE_CORPUS_DIR:-${XDG_DATA_HOME:-~/.local/share}/flowsight/private_failures}/

Privacy & isolation invariants:
  - Unique random `machine_id` (UUID4) generated once per PC (never hardware MAC/serial/hostname).
  - Directory mode 0700, files mode 0600 (POSIX).
  - Never writes inside a git repository; never uploads or syncs anywhere.
  - By default stores hashes, bounding boxes, pipeline trace, FlowSight output, and user
    correction WITHOUT copying the user's screenshot bytes (`save_local_copy=False`).
    Passing `save_local_copy=True` keeps the copy strictly inside this PC's private dir.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import uuid
from typing import Any

from PIL import Image

SCHEMA_FAILURE = 'flowsight-private-failure-v1'

VALID_CATEGORIES = (
    'ocr_error',
    'control_missed',
    'control_false_positive',
    'layout_error',
    'diff_identity_error',
    'vlm_probe_error',
    'blind_misinterpretation',
    'other',
)


def default_corpus_dir() -> Path:
    override = os.environ.get('FLOWSIGHT_PRIVATE_CORPUS_DIR', '').strip()
    if override:
        return Path(override).expanduser().resolve()
    xdg = os.environ.get('XDG_DATA_HOME', '').strip()
    base = Path(xdg).expanduser() if xdg else (Path.home() / '.local' / 'share')
    return (base / 'flowsight' / 'private_failures').resolve()


def _ensure_private_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    if os.name == 'posix':
        try:
            os.chmod(path, 0o700)
        except OSError:
            pass
    return path


def _chmod_private_file(path: Path) -> None:
    if os.name == 'posix':
        try:
            os.chmod(path, 0o600)
        except OSError:
            pass


def get_or_create_machine_id(corpus_dir: Path | None = None) -> str:
    """Return a random per-PC UUID4 stored privately in the local corpus dir."""
    root = _ensure_private_dir(corpus_dir or default_corpus_dir())
    id_file = root / 'machine_id.txt'
    if id_file.is_file():
        val = id_file.read_text(encoding='utf-8').strip()
        if val:
            return val
    mid = f'pc-{uuid.uuid4().hex[:16]}'
    id_file.write_text(mid + '\n', encoding='utf-8')
    _chmod_private_file(id_file)
    return mid


def _inspect_image_meta(image_path: str | Path | None) -> tuple[str | None, list[int] | None]:
    if not image_path:
        return None, None
    p = Path(image_path).expanduser()
    if not p.is_file():
        return None, None
    raw = p.read_bytes()
    sha = hashlib.sha256(raw).hexdigest()
    size = None
    try:
        with Image.open(p) as im:
            size = [int(im.width), int(im.height)]
    except Exception:
        size = None
    return sha, size


def log_failure(
    *,
    category: str,
    user_correction: str,
    flowsight_observed: str = '',
    agent_explanation: str = '',
    expected_behavior: str = '',
    image_path: str | Path | None = None,
    vf_pipeline: str = '',
    provider: str = '',
    model: str = '',
    notes: str = '',
    save_local_copy: bool = False,
    corpus_dir: Path | None = None,
) -> dict[str, Any]:
    """Append one structured failure record to this PC's private failure corpus."""
    cat = (category or 'other').strip().lower()
    if cat not in VALID_CATEGORIES:
        cat = 'other'
    corr = (user_correction or '').strip()
    if not corr:
        raise ValueError('user_correction must be non-empty')

    root = _ensure_private_dir(corpus_dir or default_corpus_dir())
    machine_id = get_or_create_machine_id(root)
    ts = datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')
    entry_id = f'fail-{datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")}-{uuid.uuid4().hex[:6]}'

    img_sha, img_size = _inspect_image_meta(image_path)
    local_copy_rel: str | None = None
    if save_local_copy and image_path:
        src = Path(image_path).expanduser()
        if src.is_file():
            img_dir = _ensure_private_dir(root / 'images')
            ext = src.suffix.lower() if src.suffix.lower() in ('.png', '.jpg', '.jpeg', '.webp') else '.png'
            dst = img_dir / f'{entry_id}{ext}'
            shutil.copy2(src, dst)
            _chmod_private_file(dst)
            local_copy_rel = f'images/{dst.name}'

    record: dict[str, Any] = {
        'schema': SCHEMA_FAILURE,
        'entry_id': entry_id,
        'timestamp_utc': ts,
        'machine_id': machine_id,
        'category': cat,
        'source_image_sha256': img_sha,
        'image_size': img_size,
        'vf_pipeline': (vf_pipeline or '').strip()[:300],
        'provider': (provider or '').strip()[:80],
        'model': (model or '').strip()[:120],
        'flowsight_observed': (flowsight_observed or '').strip()[:4000],
        'agent_explanation': (agent_explanation or '').strip()[:2000],
        'user_correction': corr[:2000],
        'expected_behavior': (expected_behavior or '').strip()[:1000],
        'notes': (notes or '').strip()[:1000],
        'local_image_copy': local_copy_rel,
    }

    log_path = root / 'failures.jsonl'
    with log_path.open('a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=False) + '\n')
    _chmod_private_file(log_path)
    return {
        'entry_id': entry_id,
        'machine_id': machine_id,
        'category': cat,
        'corpus_path': str(log_path),
        'source_image_sha256': img_sha,
        'local_image_copy': local_copy_rel,
        'record': record,
    }


def list_failures(
    *,
    limit: int = 20,
    category: str | None = None,
    corpus_dir: Path | None = None,
) -> list[dict[str, Any]]:
    root = corpus_dir or default_corpus_dir()
    log_path = root / 'failures.jsonl'
    if not log_path.is_file():
        return []
    out: list[dict[str, Any]] = []
    cat_filter = (category or '').strip().lower() or None
    for line in log_path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if cat_filter and row.get('category') != cat_filter:
            continue
        out.append(row)
    return out[-max(1, min(200, int(limit))):]


def corpus_stats(corpus_dir: Path | None = None) -> dict[str, Any]:
    root = corpus_dir or default_corpus_dir()
    log_path = root / 'failures.jsonl'
    rows = list_failures(limit=10000, corpus_dir=root) if log_path.is_file() else []
    by_cat: dict[str, int] = {}
    for r in rows:
        c = str(r.get('category', 'other'))
        by_cat[c] = by_cat.get(c, 0) + 1
    return {
        'corpus_dir': str(root),
        'log_exists': log_path.is_file(),
        'total_entries': len(rows),
        'by_category': dict(sorted(by_cat.items())),
    }


def format_failure_list(rows: list[dict[str, Any]], corpus_dir: Path | None = None) -> str:
    root = corpus_dir or default_corpus_dir()
    if not rows:
        return f'PRIVATE FAILURE CORPUS ({root}): 0 entries recorded on this PC.'
    lines = [f'PRIVATE FAILURE CORPUS ({root}) — showing {len(rows)} entries (local per-PC only):']
    for r in rows:
        sha = (r.get('source_image_sha256') or '-')[:12]
        corr = str(r.get('user_correction') or '').replace('\n', ' ')[:120]
        lines.append(
            f'  {r.get("entry_id")} [{r.get("category")}] {r.get("timestamp_utc")} '
            f'model={r.get("model") or "-"} img={sha} | user_correction={corr!r}'
        )
    return '\n'.join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description='FlowSight local per-PC private failure corpus')
    sub = p.add_subparsers(dest='cmd', required=True)

    pl = sub.add_parser('log', help='Log one user-confirmed perception failure locally')
    pl.add_argument('--category', default='other', choices=VALID_CATEGORIES)
    pl.add_argument('--user-correction', required=True)
    pl.add_argument('--observed', default='')
    pl.add_argument('--agent-explanation', default='')
    pl.add_argument('--expected', default='')
    pl.add_argument('--image', default=None)
    pl.add_argument('--vf', default='')
    pl.add_argument('--provider', default='')
    pl.add_argument('--model', default='')
    pl.add_argument('--notes', default='')
    pl.add_argument('--save-local-copy', action='store_true')

    pls = sub.add_parser('list', help='List recorded local failures')
    pls.add_argument('--limit', type=int, default=20)
    pls.add_argument('--category', default=None)

    sub.add_parser('stats', help='Show summary counts of the private failure corpus')

    args = p.parse_args(argv)
    if args.cmd == 'log':
        res = log_failure(
            category=args.category,
            user_correction=args.user_correction,
            flowsight_observed=args.observed,
            agent_explanation=args.agent_explanation,
            expected_behavior=args.expected,
            image_path=args.image,
            vf_pipeline=args.vf,
            provider=args.provider,
            model=args.model,
            notes=args.notes,
            save_local_copy=args.save_local_copy,
        )
        print(json.dumps({k: v for k, v in res.items() if k != 'record'}, indent=2, ensure_ascii=False))
        return 0
    if args.cmd == 'list':
        rows = list_failures(limit=args.limit, category=args.category)
        print(format_failure_list(rows))
        return 0
    if args.cmd == 'stats':
        print(json.dumps(corpus_stats(), indent=2, ensure_ascii=False))
        return 0
    return 1


if __name__ == '__main__':
    raise SystemExit(main())
