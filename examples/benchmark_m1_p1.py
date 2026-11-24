#!/usr/bin/env python3
"""FlowSight M1 Priority 1 benchmark & gate verification over real FLOW programs.

Measures exact index sizes, query character counts (plus approx word/char4 token
estimates clearly labeled as estimates, NOT billing/tokenizer counts), timings,
determinism hashes, and real FLOW VM trace invariance.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import time

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import flowsight as fs

FLOW_ROOT = Path('/home/danny/Development/ISyCo Git/FLOW')
OUT_DIR = ROOT / 'evidence' / 'm1_p1'


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_benchmark():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if str(FLOW_ROOT) not in sys.path:
        sys.path.insert(0, str(FLOW_ROOT))
    import flow.core as flow_core
    from flow.runtime import flow_trace

    targets = [
        ('hello_flow', FLOW_ROOT / 'examples' / 'hello_flow.png', 200),
        ('vortex', FLOW_ROOT / 'examples' / 'vortex.png', 500),
    ]
    results = []

    for name, png, max_ticks in targets:
        db1 = OUT_DIR / f'{name}.sqlite'
        db2 = OUT_DIR / f'{name}.run2.sqlite'
        rebuilt = OUT_DIR / f'{name}_rebuilt.png'

        t0 = time.perf_counter()
        w, h, distinct_b, b_runs, comps, rgba_bytes = fs.index_image(png, db1, mode='flow', force=True)
        index_ms = (time.perf_counter() - t0) * 1000.0

        fs.index_image(png, db2, mode='flow', force=True)
        sqlite_sha1 = sha256_file(db1)
        sqlite_sha2 = sha256_file(db2)
        db2.unlink()

        with fs.open_index(db1) as (db, meta):
            tq = time.perf_counter()
            l0 = fs.summary(db, meta)
            l0_ms = (time.perf_counter() - tq) * 1000.0

            ver = fs.verify(db, meta, str(png))
            fs.save_rgba_png(db, meta, rebuilt, force=True)

            tq2 = time.perf_counter()
            fld = fs.field_query(db, meta)
            ins = fs.instructions_query(db, meta)
            q_ms = (time.perf_counter() - tq2) * 1000.0

        (OUT_DIR / f'{name}_L0.txt').write_text(l0 + '\n', encoding='utf-8')
        (OUT_DIR / f'{name}_field.txt').write_text(fld + '\n', encoding='utf-8')
        (OUT_DIR / f'{name}_instructions.txt').write_text(ins + '\n', encoding='utf-8')

        # Naive JSON per pixel size for comparison
        orig_rgb = Image.open(png).convert('RGB')
        pix = list(orig_rgb.get_flattened_data()) if hasattr(orig_rgb, 'get_flattened_data') else list(orig_rgb.getdata())
        naive_json_bytes = len(json.dumps([
            {'x': i % w, 'y': i // w, 'r': p[0], 'g': p[1], 'b': p[2]}
            for i, p in enumerate(pix)
        ]).encode('utf-8'))

        # Gate 4: real FLOW VM trace invariance (original PNG vs rebuilt PNG)
        tr_orig = flow_trace.run_program(str(png), config=flow_core.FlowConfig(max_ticks=max_ticks, seed=42), seed=42)
        tr_reb = flow_trace.run_program(str(rebuilt), config=flow_core.FlowConfig(max_ticks=max_ticks, seed=42), seed=42)
        vm_exact = (
            tr_orig.ticks == tr_reb.ticks
            and tr_orig.events == tr_reb.events
            and tr_orig.metadata['particles_spawned'] == tr_reb.metadata['particles_spawned']
            and tr_orig.metadata['final_alive'] == tr_reb.metadata['final_alive']
        )
        orig_trace_sha = hashlib.sha256(
            json.dumps(flow_trace.canonical_projection(tr_orig), sort_keys=True, separators=(',', ':')).encode()
        ).hexdigest()

        reb_rgb_exact = orig_rgb.tobytes() == Image.open(rebuilt).convert('RGB').tobytes()
        entry = {
            'name': name,
            'source_png': str(png),
            'source_png_sha256': sha256_file(png),
            'size': [w, h],
            'distinct_rgb': int(meta['distinct_rgb']),
            'distinct_r': int(meta['distinct_r']),
            'distinct_g': int(meta['distinct_g']),
            'distinct_b': distinct_b,
            'spawn_count': int(meta['spawn_count']),
            'b_runs': b_runs,
            'b_components': comps,
            'source_png_bytes': png.stat().st_size,
            'rgba_bytes': rgba_bytes,
            'sqlite_bytes': db1.stat().st_size,
            'sqlite_sha256': sqlite_sha1,
            'sqlite_deterministic': sqlite_sha1 == sqlite_sha2,
            'naive_rgb_json_bytes': naive_json_bytes,
            'l0_chars': len(l0),
            'l0_utf8_bytes': len(l0.encode('utf-8')),
            'l0_est_tokens_div4': round(len(l0) / 4.0, 1),
            'field_query_chars': len(fld),
            'instructions_query_chars': len(ins),
            'index_ms': round(index_ms, 2),
            'l0_ms': round(l0_ms, 2),
            'field_plus_instr_ms': round(q_ms, 2),
            'verify_pass': '0 reconstruction error' in ver,
            'rebuilt_rgb_exact': reb_rgb_exact,
            'vm_trace_identical_orig_vs_rebuilt': vm_exact,
            'vm_orig_canon_sha256': orig_trace_sha,
            'vm_ticks': tr_orig.metadata['ticks'],
            'vm_events': len(tr_orig.events),
        }
        results.append(entry)
        print(
            f"{name:10} {w}x{h} | RGB={entry['distinct_rgb']:3d} B={distinct_b} spawns={entry['spawn_count']} | "
            f"SQLite={entry['sqlite_bytes']}B det={entry['sqlite_deterministic']} | "
            f"L0={entry['l0_chars']}ch (~{entry['l0_est_tokens_div4']} tok) vs naive={naive_json_bytes}B | "
            f"idx={entry['index_ms']}ms l0={entry['l0_ms']}ms | exactRGB={reb_rgb_exact} vmEq={vm_exact}"
        )

    (OUT_DIR / 'benchmark_m1_p1.json').write_text(json.dumps(results, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    run_benchmark()
