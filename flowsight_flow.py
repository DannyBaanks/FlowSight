"""FlowSight native 3-channel FLOW program index (`flowsight-flow-v1`).

Preserves the three real channels of a FLOW program PNG without loss or
quantization:
  - R, G: continuous vector field components stored as exact uint8 arrays
    (`vx = R/127.5 - 1.0`, `vy = G/127.5 - 1.0`), plus deterministic spatial
    summaries (tiles, regional stats, circulation/divergence, directional view).
  - B: exact discrete instructions/scalar values (`0..255`), stored as lossless
    per-row RLE + 4-connected components + spawn table.

NEVER executes or mutates FLOW's VM; read-only peripheral over PNGs.
Coordinate contract: origin top-left (0,0); x right, y down; boxes [x0,y0,x1,y1)
with EXCLUSIVE end coordinates.
"""
from __future__ import annotations

from collections import Counter
from contextlib import closing
import hashlib
import math
import os
from pathlib import Path
import sqlite3
import tempfile

from PIL import Image, UnidentifiedImageError

SCHEMA_FLOW = 'flowsight-flow-v1'
MAX_DIM = 512
MAX_PIXELS = MAX_DIM * MAX_DIM
MAX_COMPONENTS_SAVED = 64
# Threshold below which a vector is treated as effectively zero in directional
# summaries (note: uint8 128 maps to +0.00392, so |v| for (128,128) is ~0.00555).
ZERO_MAG_EPS = 0.02
DIR_MIN_MAG = 0.05
COHERENCE_VORTEX_THRESH = 0.35

OPCODES = {
    0: 'HALT', 1: 'NOP', 2: 'INC', 3: 'DEC', 4: 'READ', 5: 'WRITE',
    6: 'JMP+', 7: 'JMP-', 8: 'SPLIT', 9: 'TURN+', 10: 'TURN-', 11: 'PUSH',
    12: 'POP', 13: 'DUP', 14: 'SWAP', 15: 'ADD', 16: 'SUB', 17: 'MUL',
    18: 'DIV', 19: 'MOD', 20: 'EQ', 21: 'LT', 22: 'RAND', 23: 'COLOR',
    24: 'TRACE', 255: 'SPAWN MARKER (NOP)',
}

MNEMONIC_TO_CODE = {
    'HALT': 0, 'NOP': 1, 'INC': 2, 'DEC': 3, 'READ': 4, 'WRITE': 5,
    'JMP+': 6, 'JMP-': 7, 'SPLIT': 8, 'TURN+': 9, 'TURN-': 10, 'PUSH': 11,
    'POP': 12, 'DUP': 13, 'SWAP': 14, 'ADD': 15, 'SUB': 16, 'MUL': 17,
    'DIV': 18, 'MOD': 19, 'EQ': 20, 'LT': 21, 'RAND': 22, 'COLOR': 23,
    'TRACE': 24, 'SPAWN': 255, 'NOP_SPAWN': 255, 'SPAWN_MARKER': 255,
}

# 8-way compass arrows in screen coordinates (x right, y down):
# angle = atan2(vy, vx) in degrees, 0=right, +90=down, -90=up, +-180=left.
_ARROWS = ['→', '↘', '↓', '↙', '←', '↖', '↑', '↗']
_COMPASS = ['E', 'SE', 'S', 'SW', 'W', 'NW', 'N', 'NE']


def opcode_name(b: int) -> str:
    return OPCODES.get(b, 'NOP (25..254)')


def opcode_short(b: int) -> str:
    if b == 255:
        return 'SPAWN'
    if b in OPCODES:
        return OPCODES[b]
    return f'NOP({b})'


def rg_to_vec(r: int, g: int) -> tuple[float, float]:
    return (r / 127.5) - 1.0, (g / 127.5) - 1.0


def vec_arrow(vx: float, vy: float, mag: float, coherence: float | None = None) -> str:
    if mag < DIR_MIN_MAG:
        return '.'
    if coherence is not None and coherence < COHERENCE_VORTEX_THRESH:
        return '*'
    ang = math.degrees(math.atan2(vy, vx))
    idx = int(math.floor((ang + 22.5) / 45.0)) % 8
    return _ARROWS[idx]


def vec_compass(vx: float, vy: float, mag: float) -> str:
    if mag < ZERO_MAG_EPS:
        return 'ZERO'
    ang = math.degrees(math.atan2(vy, vx))
    idx = int(math.floor((ang + 22.5) / 45.0)) % 8
    return _COMPASS[idx]


def _load_flow_png(path, deny_cls):
    p = Path(path)
    if p.suffix.lower() != '.png':
        raise deny_cls('only PNG is accepted for lossless FLOW program contract')
    try:
        with Image.open(p) as img:
            if img.format != 'PNG':
                raise deny_cls('image content is not PNG')
            w, h = img.size
            if not (1 <= w <= MAX_DIM and 1 <= h <= MAX_DIM and w * h <= MAX_PIXELS):
                raise deny_cls(f'canvas {w}x{h} outside 1..{MAX_DIM} each')
            rgba_img = img.convert('RGBA')
            rgba_raw = rgba_img.tobytes()
            # FLOW programs must be opaque (alpha == 255 everywhere)
            if any(rgba_raw[i] != 255 for i in range(3, len(rgba_raw), 4)):
                raise deny_cls('FLOW program RGB channels must be opaque')
            rgb_img = img.convert('RGB')
            rgb_raw = rgb_img.tobytes()
            r_bytes = rgb_raw[0::3]
            g_bytes = rgb_raw[1::3]
            b_bytes = rgb_raw[2::3]
            return w, h, r_bytes, g_bytes, b_bytes, rgb_raw, rgba_raw
    except (OSError, UnidentifiedImageError, Image.DecompressionBombError) as e:
        raise deny_cls(f'cannot decode PNG: {e}') from e


def _region_field_stats(r_bytes: bytes, g_bytes: bytes, w: int, x0: int, y0: int, x1: int, y1: int) -> dict:
    """Exact statistics of the unquantized R/G vector field inside [x0,y0,x1,y1)."""
    n = (x1 - x0) * (y1 - y0)
    sum_vx = 0.0
    sum_vy = 0.0
    sum_mag = 0.0
    min_mag = float('inf')
    max_mag = 0.0
    active = 0
    min_r = 255
    max_r = 0
    min_g = 255
    max_g = 0

    for y in range(y0, y1):
        row_off = y * w
        for x in range(x0, x1):
            idx = row_off + x
            r = r_bytes[idx]
            g = g_bytes[idx]
            if r < min_r:
                min_r = r
            if r > max_r:
                max_r = r
            if g < min_g:
                min_g = g
            if g > max_g:
                max_g = g
            vx, vy = rg_to_vec(r, g)
            mag = math.hypot(vx, vy)
            sum_vx += vx
            sum_vy += vy
            sum_mag += mag
            if mag < min_mag:
                min_mag = mag
            if mag > max_mag:
                max_mag = mag
            if mag >= ZERO_MAG_EPS:
                active += 1

    mean_vx = sum_vx / n
    mean_vy = sum_vy / n
    mean_mag = sum_mag / n
    net_mag = math.hypot(mean_vx, mean_vy)
    coherence = (net_mag / mean_mag) if mean_mag > 1e-9 else 0.0
    mean_angle = math.degrees(math.atan2(mean_vy, mean_vx)) if net_mag >= 1e-9 else 0.0

    # Discrete divergence and curl over interior finite differences (exact on raw vx,vy)
    bw = x1 - x0
    bh = y1 - y0
    div_sum = 0.0
    curl_sum = 0.0
    diff_n = 0
    if bw >= 2 and bh >= 2:
        for y in range(y0, y1 - 1):
            r0 = y * w
            r1 = (y + 1) * w
            for x in range(x0, x1 - 1):
                vx00, vy00 = rg_to_vec(r_bytes[r0 + x], g_bytes[r0 + x])
                vx10, vy10 = rg_to_vec(r_bytes[r0 + x + 1], g_bytes[r0 + x + 1])
                vx01, vy01 = rg_to_vec(r_bytes[r1 + x], g_bytes[r1 + x])
                dvx_dx = vx10 - vx00
                dvy_dx = vy10 - vy00
                dvx_dy = vx01 - vx00
                dvy_dy = vy01 - vy00
                div_sum += dvx_dx + dvy_dy
                curl_sum += dvy_dx - dvx_dy
                diff_n += 1

    return {
        'n': n,
        'active': active,
        'mean_vx': mean_vx,
        'mean_vy': mean_vy,
        'net_mag': net_mag,
        'mean_mag': mean_mag,
        'min_mag': 0.0 if min_mag == float('inf') else min_mag,
        'max_mag': max_mag,
        'coherence': coherence,
        'angle_deg': mean_angle,
        'r_range': (min_r, max_r),
        'g_range': (min_g, max_g),
        'mean_div': (div_sum / diff_n) if diff_n else None,
        'mean_curl': (curl_sum / diff_n) if diff_n else None,
        'arrow': vec_arrow(mean_vx, mean_vy, mean_mag, coherence),
    }


def index_flow_image(image, dest, force: bool, deny_cls):
    """Index a FLOW program PNG into a lossless 3-channel SQLite database."""
    w, h, r_bytes, g_bytes, b_bytes, rgb_raw, rgba_raw = _load_flow_png(image, deny_cls)
    dest = Path(dest)
    if dest.exists() and not force:
        raise deny_cls(f'index exists: {dest} (use --force intentionally)')
    dest.parent.mkdir(parents=True, exist_ok=True)

    b_freq = Counter(b_bytes)
    distinct_r = len(set(r_bytes))
    distinct_g = len(set(g_bytes))
    distinct_b = len(b_freq)
    distinct_rgb = len({rgb_raw[i:i + 3] for i in range(0, len(rgb_raw), 3)})

    # RLE on discrete channel B
    b_runs = []
    for y in range(h):
        off = y * w
        x = 0
        while x < w:
            bv = b_bytes[off + x]
            x0 = x
            x += 1
            while x < w and b_bytes[off + x] == bv:
                x += 1
            b_runs.append((y, x0, x, bv))

    # 4-connected components on discrete channel B
    seen = bytearray(w * h)
    comps = []
    for start in range(w * h):
        if seen[start]:
            continue
        bv = b_bytes[start]
        seen[start] = 1
        stack = [start]
        n = 0
        xmin, ymin, xmax, ymax = w, h, 0, 0
        while stack:
            i = stack.pop()
            y, x = divmod(i, w)
            n += 1
            if x < xmin:
                xmin = x
            if y < ymin:
                ymin = y
            if x + 1 > xmax:
                xmax = x + 1
            if y + 1 > ymax:
                ymax = y + 1
            if x > 0 and not seen[i - 1] and b_bytes[i - 1] == bv:
                seen[i - 1] = 1
                stack.append(i - 1)
            if x < w - 1 and not seen[i + 1] and b_bytes[i + 1] == bv:
                seen[i + 1] = 1
                stack.append(i + 1)
            if y > 0 and not seen[i - w] and b_bytes[i - w] == bv:
                seen[i - w] = 1
                stack.append(i - w)
            if y < h - 1 and not seen[i + w] and b_bytes[i + w] == bv:
                seen[i + w] = 1
                stack.append(i + w)
        comps.append((bv, xmin, ymin, xmax, ymax, n))
    comps.sort(key=lambda z: (-z[5], z[2], z[1], z[0]))

    # Spawns (B == 255) in deterministic row-major order (matches FlowVM._spawn_initial)
    spawns = []
    for y in range(h):
        off = y * w
        for x in range(w):
            if b_bytes[off + x] == 255:
                r = r_bytes[off + x]
                g = g_bytes[off + x]
                vx, vy = rg_to_vec(r, g)
                mag = math.hypot(vx, vy)
                ang = math.degrees(math.atan2(vy, vx)) if mag >= 1e-9 else 0.0
                spawns.append((len(spawns), x, y, r, g, vx, vy, mag, ang))

    # Global field stats + tile summaries (up to 16x16 tiles)
    gstats = _region_field_stats(r_bytes, g_bytes, w, 0, 0, w, h)
    tcols = min(w, 16)
    trows = min(h, 16)
    tile_rows = []
    for ty in range(trows):
        y0 = ty * h // trows
        y1 = (ty + 1) * h // trows
        for tx in range(tcols):
            x0 = tx * w // tcols
            x1 = (tx + 1) * w // tcols
            st = _region_field_stats(r_bytes, g_bytes, w, x0, y0, x1, y1)
            tile_rows.append((
                ty, tx, x0, y0, x1, y1,
                st['mean_vx'], st['mean_vy'], st['mean_mag'], st['max_mag'],
                st['coherence'], st['arrow'],
            ))

    active_instr = sum(c for b, c in b_freq.items() if (1 <= b <= 24) or b == 255)

    fd, tmp = tempfile.mkstemp(prefix='.flowsight_', suffix='.sqlite', dir=str(dest.parent))
    os.close(fd)
    try:
        with closing(sqlite3.connect(tmp)) as db:
            db.execute('PRAGMA page_size = 4096')
            db.executescript('''
                CREATE TABLE meta(k TEXT PRIMARY KEY, v TEXT NOT NULL);
                CREATE TABLE field_raw(channel TEXT PRIMARY KEY, data BLOB NOT NULL);
                CREATE TABLE b_palette(b INTEGER PRIMARY KEY, mnemonic TEXT NOT NULL, pixels INT NOT NULL);
                CREATE TABLE b_runs(y INT NOT NULL, x0 INT NOT NULL, x1 INT NOT NULL, b INT NOT NULL, PRIMARY KEY(y, x0));
                CREATE INDEX b_runs_b_idx ON b_runs(b, y);
                CREATE TABLE b_components(id INT PRIMARY KEY, b INT NOT NULL, x0 INT NOT NULL, y0 INT NOT NULL, x1 INT NOT NULL, y1 INT NOT NULL, pixels INT NOT NULL);
                CREATE TABLE spawns(id INT PRIMARY KEY, x INT NOT NULL, y INT NOT NULL, r INT NOT NULL, g INT NOT NULL, vx REAL NOT NULL, vy REAL NOT NULL, mag REAL NOT NULL, angle_deg REAL NOT NULL);
                CREATE TABLE field_tiles(ty INT NOT NULL, tx INT NOT NULL, x0 INT NOT NULL, y0 INT NOT NULL, x1 INT NOT NULL, y1 INT NOT NULL, mean_vx REAL NOT NULL, mean_vy REAL NOT NULL, mean_mag REAL NOT NULL, max_mag REAL NOT NULL, coherence REAL NOT NULL, dominant_dir TEXT NOT NULL, PRIMARY KEY(ty, tx));
            ''')
            kv = {
                'schema': SCHEMA_FLOW,
                'mode': 'flow',
                'width': str(w),
                'height': str(h),
                'source_name': Path(image).name,
                'source_sha256': hashlib.sha256(Path(image).read_bytes()).hexdigest(),
                'rgb_sha256': hashlib.sha256(rgb_raw).hexdigest(),
                'rgba_sha256': hashlib.sha256(rgba_raw).hexdigest(),
                'r_sha256': hashlib.sha256(r_bytes).hexdigest(),
                'g_sha256': hashlib.sha256(g_bytes).hexdigest(),
                'b_sha256': hashlib.sha256(b_bytes).hexdigest(),
                'distinct_rgb': str(distinct_rgb),
                'distinct_r': str(distinct_r),
                'distinct_g': str(distinct_g),
                'distinct_b': str(distinct_b),
                'colors': str(distinct_b),
                'spawn_count': str(len(spawns)),
                'active_instruction_cells': str(active_instr),
                'nonzero_vector_cells': str(gstats['active']),
                'field_mean_vx': f'{gstats["mean_vx"]:.6f}',
                'field_mean_vy': f'{gstats["mean_vy"]:.6f}',
                'field_mean_mag': f'{gstats["mean_mag"]:.6f}',
                'field_max_mag': f'{gstats["max_mag"]:.6f}',
                'field_coherence': f'{gstats["coherence"]:.6f}',
                'tile_cols': str(tcols),
                'tile_rows': str(trows),
                'component_count': str(len(comps)),
                'component_saved': str(min(len(comps), MAX_COMPONENTS_SAVED)),
            }
            db.executemany('INSERT INTO meta VALUES (?, ?)', sorted(kv.items()))
            db.executemany('INSERT INTO field_raw VALUES (?, ?)', [
                ('G', sqlite3.Binary(g_bytes)),
                ('R', sqlite3.Binary(r_bytes)),
            ])
            db.executemany(
                'INSERT INTO b_palette VALUES (?, ?, ?)',
                [(b, opcode_name(b), b_freq[b]) for b in sorted(b_freq)],
            )
            db.executemany('INSERT INTO b_runs VALUES (?, ?, ?, ?)', b_runs)
            db.executemany(
                'INSERT INTO b_components VALUES (?, ?, ?, ?, ?, ?, ?)',
                [(i, *c) for i, c in enumerate(comps[:MAX_COMPONENTS_SAVED])],
            )
            db.executemany('INSERT INTO spawns VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)', spawns)
            db.executemany(
                'INSERT INTO field_tiles VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
                tile_rows,
            )
            db.commit()
        os.replace(tmp, dest)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)

    return w, h, distinct_b, len(b_runs), len(comps), len(rgba_raw)


def _get_rg_raw(db, meta, deny_cls) -> tuple[bytes, bytes]:
    w = int(meta['width'])
    h = int(meta['height'])
    try:
        rows = dict(db.execute('SELECT channel, data FROM field_raw'))
    except sqlite3.Error as e:
        raise deny_cls(f'corrupted field_raw table: {e}') from e
    if 'R' not in rows or 'G' not in rows:
        raise deny_cls('missing R or G channel in flow index')
    if not isinstance(rows['R'], (bytes, bytearray, memoryview)) or not isinstance(rows['G'], (bytes, bytearray, memoryview)):
        raise deny_cls('field_raw channel data must be BLOB')
    r_bytes = bytes(rows['R'])
    g_bytes = bytes(rows['G'])
    if len(r_bytes) != w * h or len(g_bytes) != w * h:
        raise deny_cls('R/G channel byte length mismatch')
    if hashlib.sha256(r_bytes).hexdigest() != meta['r_sha256']:
        raise deny_cls('R channel hash mismatch: index is corrupted')
    if hashlib.sha256(g_bytes).hexdigest() != meta['g_sha256']:
        raise deny_cls('G channel hash mismatch: index is corrupted')
    return r_bytes, g_bytes


def _get_b_raw(db, meta, deny_cls) -> bytes:
    w = int(meta['width'])
    h = int(meta['height'])
    out = [-1] * (w * h)
    for y, x0, x1, b in db.execute('SELECT y, x0, x1, b FROM b_runs ORDER BY y, x0'):
        if not (0 <= y < h and 0 <= x0 < x1 <= w and 0 <= b <= 255):
            raise deny_cls('invalid B run in flow index')
        s = y * w + x0
        if any(v != -1 for v in out[s:s + x1 - x0]):
            raise deny_cls('overlapping B runs in flow index')
        out[s:s + x1 - x0] = [b] * (x1 - x0)
    if -1 in out:
        raise deny_cls('missing B pixel coverage in flow index')
    b_bytes = bytes(out)
    if hashlib.sha256(b_bytes).hexdigest() != meta['b_sha256']:
        raise deny_cls('B channel hash mismatch: index is corrupted')
    return b_bytes


def flow_decoded_rgba(db, meta, deny_cls) -> bytes:
    r_bytes, g_bytes = _get_rg_raw(db, meta, deny_cls)
    b_bytes = _get_b_raw(db, meta, deny_cls)
    n = len(r_bytes)
    rgb = bytearray(n * 3)
    rgb[0::3] = r_bytes
    rgb[1::3] = g_bytes
    rgb[2::3] = b_bytes
    if hashlib.sha256(rgb).hexdigest() != meta['rgb_sha256']:
        raise deny_cls('RGB hash mismatch: flow index is corrupted')
    rgba = bytearray(n * 4)
    rgba[0::4] = r_bytes
    rgba[1::4] = g_bytes
    rgba[2::4] = b_bytes
    rgba[3::4] = b'\xff' * n
    raw = bytes(rgba)
    if hashlib.sha256(raw).hexdigest() != meta['rgba_sha256']:
        raise deny_cls('RGBA hash mismatch: flow index is corrupted')
    return raw


def flow_verify(db, meta, input_png, deny_cls) -> str:
    raw = flow_decoded_rgba(db, meta, deny_cls)
    res = (
        f'PASS flow index integrity: '
        f'r_sha256={meta["r_sha256"][:12]} g_sha256={meta["g_sha256"][:12]} '
        f'b_sha256={meta["b_sha256"][:12]} rgb_sha256={meta["rgb_sha256"][:12]}'
    )
    if input_png:
        w, h, r_src, g_src, b_src, rgb_src, rgba_src = _load_flow_png(input_png, deny_cls)
        if (w, h) != (int(meta['width']), int(meta['height'])):
            raise deny_cls('image size mismatch')
        r_idx, g_idx = _get_rg_raw(db, meta, deny_cls)
        b_idx = _get_b_raw(db, meta, deny_cls)
        if r_idx != r_src:
            raise deny_cls('R channel mismatch with source PNG')
        if g_idx != g_src:
            raise deny_cls('G channel mismatch with source PNG')
        if b_idx != b_src:
            raise deny_cls('B channel mismatch with source PNG')
        if raw != rgba_src:
            raise deny_cls('RGBA pixel mismatch with source PNG')
        res += '\nPASS input: exact R, G, B and decoded RGBA byte equality (0 reconstruction error)'
    return res


def flow_summary(db, meta, cols: int, rows: int, max_chars: int, bounded_fn, deny_cls) -> str:
    w = int(meta['width'])
    h = int(meta['height'])
    n = w * h
    r_bytes, g_bytes = _get_rg_raw(db, meta, deny_cls)
    _ = _get_b_raw(db, meta, deny_cls)  # integrity check

    # Vector field directional map (approximate visual summary; raw R/G are exact)
    vcols = max(1, min(cols, w, 24))
    vrows = max(1, min(rows, h, 16))
    tcols = int(meta['tile_cols'])
    trows = int(meta['tile_rows'])

    out = [
        f'FLOWSIGHT v1 | {w}x{h} RGB | mode=flow (native 3-channel lossless) | SHA256(RGB)={meta["rgb_sha256"][:16]}',
        'Coordinates: origin top-left (0,0); x right, y down; boxes [x0,y0,x1,y1) exclusive.',
        'FLOW raw channels: R,G define vector ((value/127.5)-1); B defines instruction. NOT image gradients.',
        (
            f'Vector field R/G (exact unquantized stored): distinct_RGB={meta["distinct_rgb"]} '
            f'(R={meta["distinct_r"]}, G={meta["distinct_g"]}) | '
            f'active(|v|>={ZERO_MAG_EPS})={meta["nonzero_vector_cells"]}/{n} '
            f'({int(meta["nonzero_vector_cells"])*100/n:.1f}%) | '
            f'mean_v=({float(meta["field_mean_vx"]):+.3f},{float(meta["field_mean_vy"]):+.3f}) '
            f'mean_mag={float(meta["field_mean_mag"]):.3f} max_mag={float(meta["field_max_mag"]):.3f} '
            f'coherence={float(meta["field_coherence"]):.3f}'
        ),
        'Channel B discrete instructions/data (exact counts):',
    ]
    b_rows = list(db.execute('SELECT b, mnemonic, pixels FROM b_palette ORDER BY pixels DESC, b ASC'))
    for b, mnem, cnt in b_rows[:12]:
        out.append(f'  B={b:<3d} {mnem:<20s} {cnt}/{n} ({cnt*100/n:.1f}%)')
    if len(b_rows) > 12:
        out.append(f'  ... {len(b_rows)-12} additional B values (use instructions)')

    spawns = list(db.execute('SELECT id, x, y, vx, vy, mag, angle_deg FROM spawns ORDER BY id'))
    out.append(f'Spawns (B=255): {len(spawns)} total:')
    for sid, sx, sy, vx, vy, mag, ang in spawns[:8]:
        out.append(f'  pid={sid} ({sx},{sy}) v=({vx:+.3f},{vy:+.3f}) mag={mag:.3f} deg={ang:+.1f} dir={vec_compass(vx, vy, mag)}')
    if len(spawns) > 8:
        out.append(f'  ... {len(spawns)-8} additional spawns (use instructions --opcode SPAWN)')

    out.append(
        f'Vector field summary map {vcols}x{vrows} (APPROXIMATE tile mean; '
        f'arrows=dir if mean_mag>={DIR_MIN_MAG}, *=low-coherence<{COHERENCE_VORTEX_THRESH}, .=zero):'
    )
    if vcols == tcols and vrows == trows:
        tmap = {(ty, tx): d for ty, tx, d in db.execute('SELECT ty, tx, dominant_dir FROM field_tiles')}
        for ty in range(vrows):
            out.append('  ' + ''.join(tmap[(ty, tx)] for tx in range(vcols)))
    else:
        for ty in range(vrows):
            y0 = ty * h // vrows
            y1 = (ty + 1) * h // vrows
            line = []
            for tx in range(vcols):
                x0 = tx * w // vcols
                x1 = (tx + 1) * w // vcols
                st = _region_field_stats(r_bytes, g_bytes, w, x0, y0, x1, y1)
                line.append(st['arrow'])
            out.append('  ' + ''.join(line))

    # Show non-HALT components first so instructions are immediately visible in L0
    total_c = int(meta['component_count'])
    saved_c = int(meta['component_saved'])
    non_zero_total = db.execute('SELECT COUNT(*) FROM b_components WHERE b != 0').fetchone()[0]
    non_zero = list(db.execute(
        'SELECT id, b, x0, y0, x1, y1, pixels FROM b_components WHERE b != 0 ORDER BY id LIMIT 7'
    ))
    out.append(f'Channel B components (4-connected): total={total_c} (saved={saved_c}); non-HALT shown ({len(non_zero)}/{non_zero_total}):')
    if non_zero:
        for cid, b, x0, y0, x1, y1, cnt in non_zero:
            out.append(f'  id={cid} B={b}({opcode_short(b)}) bbox=[{x0},{y0},{x1},{y1}) area={cnt}')
        if non_zero_total > len(non_zero):
            out.append(f'  [TRUNCATED: {non_zero_total - len(non_zero)} more non-HALT components; use components or instructions]')
    else:
        out.append('  (none: all cells have B=0 HALT)')
    if total_c > saved_c:
        out.append(f'  component detail truncated ({total_c - saved_c} not individually retained)')

    out.append('Queries: point, scanline, region, field, instructions, find, components, edges, diff, rebuild, verify')
    return bounded_fn('\n'.join(out), max_chars)


def flow_point(db, meta, x: int, y: int, validate_box_fn, deny_cls) -> str:
    w = int(meta['width'])
    h = int(meta['height'])
    validate_box_fn(w, h, x, y, 1, 1)
    r_bytes, g_bytes = _get_rg_raw(db, meta, deny_cls)
    row = db.execute('SELECT b FROM b_runs WHERE y=? AND x0<=? AND x1>?', (y, x, x)).fetchone()
    if not row:
        raise deny_cls('point missing from B index')
    b = row[0]
    idx = y * w + x
    r = r_bytes[idx]
    g = g_bytes[idx]
    vx, vy = rg_to_vec(r, g)
    mag = math.hypot(vx, vy)
    ang = math.degrees(math.atan2(vy, vx)) if mag >= 1e-9 else 0.0
    hx = f'#{r:02x}{g:02x}{b:02x}ff'
    return (
        f'({x},{y}) RGB=({r},{g},{b}) {hx} | '
        f'FLOW v=({vx:+.5f},{vy:+.5f}) mag={mag:.5f} deg={ang:+.1f} dir={vec_compass(vx, vy, mag)}; '
        f'B={b} {opcode_name(b)}'
    )


def flow_scanline(db, meta, y: int, max_lines: int, deny_cls) -> str:
    w = int(meta['width'])
    h = int(meta['height'])
    if not 0 <= y < h:
        raise deny_cls('scanline y outside image')
    r_bytes, g_bytes = _get_rg_raw(db, meta, deny_cls)
    data = list(db.execute('SELECT x0, x1, b FROM b_runs WHERE y=? ORDER BY x0', (y,)))
    lines = [f'FLOW SCANLINE y={y} B_runs={len(data)}; x intervals [x0,x1) exclusive (R/G exact stats per run):']
    for a, b_end, bv in data[:max_lines]:
        st = _region_field_stats(r_bytes, g_bytes, w, a, y, b_end, y + 1)
        if b_end - a == 1:
            r = r_bytes[y * w + a]
            g = g_bytes[y * w + a]
            rg_txt = f'RG=({r},{g}) v=({st["mean_vx"]:+.4f},{st["mean_vy"]:+.4f}) mag={st["mean_mag"]:.4f}'
        else:
            rg_txt = (
                f'R=[{st["r_range"][0]}..{st["r_range"][1]}] G=[{st["g_range"][0]}..{st["g_range"][1]}] '
                f'mean_v=({st["mean_vx"]:+.3f},{st["mean_vy"]:+.3f}) mag={st["mean_mag"]:.3f} coh={st["coherence"]:.2f}'
            )
        lines.append(f'  x=[{a},{b_end}) len={b_end - a} B={bv}({opcode_short(bv)}) | {rg_txt}')
    if len(data) > max_lines:
        lines.append(f'[TRUNCATED: {len(data)-max_lines} additional B runs; use region]')
    return '\n'.join(lines)


def flow_region(db, meta, x: int, y: int, bw: int, bh: int, max_lines: int, grid: bool, validate_box_fn, deny_cls) -> str:
    w = int(meta['width'])
    h = int(meta['height'])
    validate_box_fn(w, h, x, y, bw, bh)
    r_bytes, g_bytes = _get_rg_raw(db, meta, deny_cls)
    st = _region_field_stats(r_bytes, g_bytes, w, x, y, x + bw, y + bh)

    counts = Counter()
    rle_lines = []
    total_runs = 0
    for yy, lo, hi, bv in db.execute(
        'SELECT y, x0, x1, b FROM b_runs WHERE y>=? AND y<? AND x1>? AND x0<? ORDER BY y, x0',
        (y, y + bh, x, x + bw),
    ):
        a = max(x, lo)
        b_end = min(x + bw, hi)
        if b_end <= a:
            continue
        counts[bv] += b_end - a
        total_runs += 1
        if len(rle_lines) < max_lines:
            rle_lines.append(f'  y={yy} x=[{a},{b_end}) B={bv}({opcode_short(bv)})')
    if sum(counts.values()) != bw * bh:
        raise deny_cls('B index coverage mismatch in queried region')

    out = [
        f'FLOW REGION [{x},{y},{x + bw},{y + bh}) exact pixels={bw * bh}, B_runs={total_runs}',
        (
            f'FIELD R/G (exact unquantized): active={st["active"]}/{st["n"]} '
            f'R=[{st["r_range"][0]}..{st["r_range"][1]}] G=[{st["g_range"][0]}..{st["g_range"][1]}] '
            f'mean_v=({st["mean_vx"]:+.4f},{st["mean_vy"]:+.4f}) '
            f'mean_mag={st["mean_mag"]:.4f} min_mag={st["min_mag"]:.4f} max_mag={st["max_mag"]:.4f} '
            f'coherence={st["coherence"]:.4f} deg={st["angle_deg"]:+.1f} dir={vec_compass(st["mean_vx"], st["mean_vy"], st["net_mag"])}'
        ),
    ]
    if st['mean_div'] is not None:
        out.append(
            f'FIELD DIFFERENTIALS (exact finite diff): mean_div={st["mean_div"]:+.5f} '
            f'mean_curl={st["mean_curl"]:+.5f} '
            f'(curl>0=CW in screen y-down, curl<0=CCW)'
        )
    out.append(
        'CHANNEL B COVERAGE: ' + ', '.join(
            f'B={k}({opcode_short(k)}):{v} ({v/(bw*bh):.1%})'
            for k, v in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        )
    )

    if grid:
        if bw > 32 or bh > 32:
            raise deny_cls(f'--grid region {bw}x{bh} exceeds 32x32 max; query a smaller box')
        b_bytes = _get_b_raw(db, meta, deny_cls)
        # 1:1 exact arrow grid for R/G
        out.append(f'R/G DIRECTION GRID 1:1 (x={x}..{x + bw - 1}, y={y}..{y + bh - 1}):')
        for yy in range(y, y + bh):
            chars = []
            for xx in range(x, x + bw):
                idx = yy * w + xx
                vx, vy = rg_to_vec(r_bytes[idx], g_bytes[idx])
                chars.append(vec_arrow(vx, vy, math.hypot(vx, vy)))
            out.append(f'  y={yy:3d}: ' + ''.join(chars))
        # 1:1 exact opcode symbol grid for B
        b_syms = {0: '.', 255: 'S'}
        sym_pool = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        next_i = 0
        legend = ['.=0(HALT)', 'S=255(SPAWN)']
        for bv in sorted(counts):
            if bv not in b_syms:
                ch = sym_pool[next_i % len(sym_pool)]
                next_i += 1
                b_syms[bv] = ch
                legend.append(f'{ch}={bv}({opcode_short(bv)})')
        out.append('B OPCODE GRID 1:1 (' + ' '.join(legend) + '):')
        for yy in range(y, y + bh):
            chars = [b_syms[b_bytes[yy * w + xx]] for xx in range(x, x + bw)]
            out.append(f'  y={yy:3d}: ' + ''.join(chars))
    else:
        out.append('B RLE intersections:')
        out.extend(rle_lines)
        if total_runs > max_lines:
            out.append(f'[TRUNCATED {total_runs - max_lines} more runs: split region or use --grid]')
    return '\n'.join(out)


def flow_field(db, meta, box: tuple[int, int, int, int] | None, cols: int, rows: int, validate_box_fn, deny_cls) -> str:
    """Regional vector field query over exact unquantized R/G channels."""
    w = int(meta['width'])
    h = int(meta['height'])
    if box is None:
        x, y, bw, bh = 0, 0, w, h
    else:
        x, y, bw, bh = box
        validate_box_fn(w, h, x, y, bw, bh)
    r_bytes, g_bytes = _get_rg_raw(db, meta, deny_cls)
    st = _region_field_stats(r_bytes, g_bytes, w, x, y, x + bw, y + bh)

    gcols = max(1, min(cols, bw, 24))
    grows = max(1, min(rows, bh, 16))
    out = [
        f'FLOW FIELD [{x},{y},{x + bw},{y + bh}) exact cells={bw * bh} | unquantized R/G',
        (
            f'SUMMARY: active(|v|>={ZERO_MAG_EPS})={st["active"]}/{st["n"]} ({st["active"]*100/st["n"]:.1f}%) | '
            f'R=[{st["r_range"][0]}..{st["r_range"][1]}] G=[{st["g_range"][0]}..{st["g_range"][1]}]'
        ),
        (
            f'VECTOR: mean_v=({st["mean_vx"]:+.5f},{st["mean_vy"]:+.5f}) net_mag={st["net_mag"]:.5f} '
            f'mean_mag={st["mean_mag"]:.5f} min_mag={st["min_mag"]:.5f} max_mag={st["max_mag"]:.5f} '
            f'coherence={st["coherence"]:.5f} deg={st["angle_deg"]:+.1f}'
        ),
    ]
    if st['mean_div'] is not None:
        out.append(
            f'DIFFERENTIALS: mean_div={st["mean_div"]:+.5f} mean_curl={st["mean_curl"]:+.5f} '
            f'(screen y-down: curl>0=CW rotation, curl<0=CCW rotation)'
        )

    # 2x2 quadrant breakdown when region >= 4x4 for quick spatial structure
    if bw >= 4 and bh >= 4:
        mx = x + bw // 2
        my = y + bh // 2
        quads = [
            ('TL', x, y, mx, my),
            ('TR', mx, y, x + bw, my),
            ('BL', x, my, mx, y + bh),
            ('BR', mx, my, x + bw, y + bh),
        ]
        out.append('QUADRANTS (exact means):')
        for name, qx0, qy0, qx1, qy1 in quads:
            qs = _region_field_stats(r_bytes, g_bytes, w, qx0, qy0, qx1, qy1)
            out.append(
                f'  {name} [{qx0},{qy0},{qx1},{qy1}): mean_v=({qs["mean_vx"]:+.3f},{qs["mean_vy"]:+.3f}) '
                f'mag={qs["mean_mag"]:.3f} coh={qs["coherence"]:.3f} dir={vec_compass(qs["mean_vx"], qs["mean_vy"], qs["net_mag"])}'
            )

    out.append(f'DIRECTION GRID {gcols}x{grows} (APPROXIMATE tile mean; *=coh<{COHERENCE_VORTEX_THRESH}, .=mag<{DIR_MIN_MAG}):')
    for ty in range(grows):
        yy0 = y + ty * bh // grows
        yy1 = y + (ty + 1) * bh // grows
        line = []
        for tx in range(gcols):
            xx0 = x + tx * bw // gcols
            xx1 = x + (tx + 1) * bw // gcols
            ts = _region_field_stats(r_bytes, g_bytes, w, xx0, yy0, xx1, yy1)
            line.append(ts['arrow'])
        out.append('  ' + ''.join(line))
    return '\n'.join(out)


def _parse_opcode_filter(tok: str, deny_cls) -> int:
    t = tok.strip().upper()
    if t in MNEMONIC_TO_CODE:
        return MNEMONIC_TO_CODE[t]
    try:
        v = int(t, 10)
    except ValueError:
        raise deny_cls(f'unknown opcode filter: {tok!r} (use 0..255 or mnemonic like SPAWN, INC, WRITE)') from None
    if not 0 <= v <= 255:
        raise deny_cls(f'opcode {v} outside 0..255')
    return v


def flow_instructions(
    db, meta,
    opcode_tok: str | None,
    box: tuple[int, int, int, int] | None,
    include_halt: bool,
    max_lines: int,
    validate_box_fn,
    deny_cls,
) -> str:
    """Query channel B instructions and their spatial coupling to the local R/G vector."""
    w = int(meta['width'])
    h = int(meta['height'])
    if box is None:
        x0, y0, bw, bh = 0, 0, w, h
    else:
        x0, y0, bw, bh = box
        validate_box_fn(w, h, x0, y0, bw, bh)
    x1, y1 = x0 + bw, y0 + bh

    op_filter = _parse_opcode_filter(opcode_tok, deny_cls) if opcode_tok is not None else None
    r_bytes, g_bytes = _get_rg_raw(db, meta, deny_cls)
    b_bytes = _get_b_raw(db, meta, deny_cls)

    matches = []
    by_code = Counter()
    for yy in range(y0, y1):
        off = yy * w
        for xx in range(x0, x1):
            bv = b_bytes[off + xx]
            if op_filter is not None:
                if bv != op_filter:
                    continue
            elif not include_halt and bv == 0:
                continue
            by_code[bv] += 1
            matches.append((xx, yy, bv))

    scope = f'[{x0},{y0},{x1},{y1})'
    filt = f'opcode={op_filter}({opcode_short(op_filter)})' if op_filter is not None else ('all' if include_halt else 'non-HALT (B!=0)')
    out = [
        f'FLOW INSTRUCTIONS scope={scope} filter={filt} matches={len(matches)}',
        'COUNTS: ' + (
            ', '.join(f'B={k}({opcode_short(k)}):{v}' for k, v in sorted(by_code.items(), key=lambda kv: (-kv[1], kv[0])))
            if by_code else '(none)'
        ),
        'CELLS + LOCAL VECTOR + 1-CELL STEP ALONG UNIT VECTOR (static spatial relation, NOT VM simulation):',
    ]
    for xx, yy, bv in matches[:max_lines]:
        idx = yy * w + xx
        r = r_bytes[idx]
        g = g_bytes[idx]
        vx, vy = rg_to_vec(r, g)
        mag = math.hypot(vx, vy)
        if mag >= ZERO_MAG_EPS:
            nx = xx + int(round(vx / mag))
            ny = yy + int(round(vy / mag))
            if 0 <= nx < w and 0 <= ny < h:
                nb = b_bytes[ny * w + nx]
                step_txt = f'-> ({nx},{ny}) B={nb}({opcode_short(nb)})'
            else:
                step_txt = f'-> ({nx},{ny}) OUT_OF_BOUNDS'
        else:
            step_txt = '-> (static, |v|<eps)'
        out.append(
            f'  ({xx:3d},{yy:3d}) B={bv:<3d} {opcode_short(bv):<7s} | '
            f'RG=({r:3d},{g:3d}) v=({vx:+.3f},{vy:+.3f}) mag={mag:.3f} {vec_compass(vx, vy, mag):<4s} {step_txt}'
        )
    if len(matches) > max_lines:
        out.append(f'[TRUNCATED {len(matches) - max_lines} additional cells; filter with --opcode or --region]')
    return '\n'.join(out)


def flow_find(db, meta, token: str, max_lines: int, parse_color_fn, deny_cls) -> str:
    """Find by exact RGB hex (#rrggbb/#rrggbbaa) OR by B opcode (e.g. SPAWN, INC, B=2, 255)."""
    w = int(meta['width'])
    h = int(meta['height'])
    t = token.strip()
    if t.startswith('#'):
        rgba = parse_color_fn(t)
        r_bytes, g_bytes = _get_rg_raw(db, meta, deny_cls)
        b_bytes = _get_b_raw(db, meta, deny_cls)
        rt, gt, bt, at = rgba
        if at != 255:
            return f'#{rt:02x}{gt:02x}{bt:02x}{at:02x} exact matches=0 (FLOW programs are opaque)'
        pts = []
        for yy in range(h):
            off = yy * w
            for xx in range(w):
                idx = off + xx
                if r_bytes[idx] == rt and g_bytes[idx] == gt and b_bytes[idx] == bt:
                    pts.append((xx, yy))
        if not pts:
            return f'#{rt:02x}{gt:02x}{bt:02x}ff exact matches=0'
        x0 = min(p[0] for p in pts)
        x1 = max(p[0] for p in pts) + 1
        y0 = min(p[1] for p in pts)
        y1 = max(p[1] for p in pts) + 1
        lines = [f'#{rt:02x}{gt:02x}{bt:02x}ff exact pixels={len(pts)} bbox=[{x0},{y0},{x1},{y1})']
        for xx, yy in pts[:max_lines]:
            lines.append(f'  ({xx},{yy})')
        if len(pts) > max_lines:
            lines.append(f'[TRUNCATED {len(pts) - max_lines} points; use region]')
        return '\n'.join(lines)

    # Opcode lookup on channel B
    op_tok = t[2:] if t.upper().startswith('B=') else t
    bv = _parse_opcode_filter(op_tok, deny_cls)
    data = list(db.execute('SELECT y, x0, x1 FROM b_runs WHERE b=? ORDER BY y, x0', (bv,)))
    if not data:
        return f'B={bv}({opcode_short(bv)}) exact matches=0'
    count = sum(b_end - a for _, a, b_end in data)
    x0 = min(a for _, a, _ in data)
    x1 = max(b_end for _, _, b_end in data)
    y0 = data[0][0]
    y1 = data[-1][0] + 1
    lines = [f'B={bv}({opcode_short(bv)}) exact pixels={count} bbox=[{x0},{y0},{x1},{y1}) rows/runs={len(data)}']
    lines += [f'  y={yy} x=[{a},{b_end})' for yy, a, b_end in data[:max_lines]]
    if len(data) > max_lines:
        lines.append(f'[TRUNCATED {len(data) - max_lines} runs: use region]')
    return '\n'.join(lines)


def flow_components(db, meta, max_lines: int) -> str:
    total = int(meta['component_count'])
    saved = int(meta['component_saved'])
    lines = [f'FLOW Channel B 4-connected components total={total}; saved={saved} (top {MAX_COMPONENTS_SAVED} saved)']
    for cid, bv, x0, y0, x1, y1, count in db.execute(
        'SELECT id, b, x0, y0, x1, y1, pixels FROM b_components ORDER BY id LIMIT ?',
        (max_lines,),
    ):
        lines.append(f'  id={cid} B={bv}({opcode_short(bv)}) bbox=[{x0},{y0},{x1},{y1}) pixels={count}')
    if total > saved:
        lines.append('WARNING: components list incomplete; only 64 biggest retained')
    return '\n'.join(lines)


def flow_edges(db, meta, box: tuple[int, int, int, int] | None, max_lines: int, validate_box_fn, deny_cls) -> str:
    """Discrete instruction-boundary edges on channel B inside optional bbox."""
    w = int(meta['width'])
    h = int(meta['height'])
    if box is None:
        x0, y0, bw, bh = 0, 0, w, h
    else:
        x0, y0, bw, bh = box
        validate_box_fn(w, h, x0, y0, bw, bh)
    x1, y1 = x0 + bw, y0 + bh
    b_bytes = _get_b_raw(db, meta, deny_cls)
    segments = []
    for x in range(max(1, x0 + 1), x1):
        y = y0
        while y < y1:
            a = b_bytes[y * w + x - 1]
            b = b_bytes[y * w + x]
            if a == b:
                y += 1
                continue
            ys = y
            y += 1
            while y < y1 and b_bytes[y * w + x - 1] == a and b_bytes[y * w + x] == b:
                y += 1
            segments.append(('V', x, ys, y, a, b))
    for y in range(max(1, y0 + 1), y1):
        x = x0
        while x < x1:
            a = b_bytes[(y - 1) * w + x]
            b = b_bytes[y * w + x]
            if a == b:
                x += 1
                continue
            xs = x
            x += 1
            while x < x1 and b_bytes[(y - 1) * w + x] == a and b_bytes[y * w + x] == b:
                x += 1
            segments.append(('H', y, xs, x, a, b))
    lines = [f'FLOW CHANNEL B EDGES scope=[{x0},{y0},{x1},{y1}) exact segments={len(segments)}']
    for orient, at, start, stop, a, b in segments[:max_lines]:
        if orient == 'V':
            lines.append(f'  V x={at} y=[{start},{stop}) B={a}({opcode_short(a)})->B={b}({opcode_short(b)})')
        else:
            lines.append(f'  H y={at} x=[{start},{stop}) B={a}({opcode_short(a)})->B={b}({opcode_short(b)})')
    if len(segments) > max_lines:
        lines.append(f'[TRUNCATED {len(segments) - max_lines} edge segments; narrow --region]')
    return '\n'.join(lines)


def flow_diff(db, meta, other_png, max_lines: int, deny_cls) -> str:
    w = int(meta['width'])
    h = int(meta['height'])
    ww, hh, r2, g2, b2, _, _ = _load_flow_png(other_png, deny_cls)
    if (w, h) != (ww, hh):
        raise deny_cls(f'diff requires same size; got {ww}x{hh} vs {w}x{h}')
    r1, g1 = _get_rg_raw(db, meta, deny_cls)
    b1 = _get_b_raw(db, meta, deny_cls)

    rg_changed = 0
    b_changed = 0
    any_changed = 0
    x0, y0, x1, y1 = w, h, 0, 0
    b_changes = []
    rg_changes = []
    for y in range(h):
        off = y * w
        for x in range(w):
            i = off + x
            drg = (r1[i] != r2[i]) or (g1[i] != g2[i])
            dbv = (b1[i] != b2[i])
            if drg or dbv:
                any_changed += 1
                x0 = min(x0, x)
                x1 = max(x1, x + 1)
                y0 = min(y0, y)
                y1 = max(y1, y + 1)
                if dbv:
                    b_changed += 1
                    if len(b_changes) < max_lines:
                        b_changes.append((x, y, b1[i], b2[i]))
                if drg:
                    rg_changed += 1
                    if len(rg_changes) < max_lines:
                        rg_changes.append((x, y, r1[i], g1[i], r2[i], g2[i]))

    pixels = w * h
    lines = [
        f'FLOW DIFF {w}x{h} total_changed={any_changed}/{pixels} ({any_changed/pixels:.2%}) '
        f'| field_RG_changed={rg_changed} | instr_B_changed={b_changed}'
    ]
    if not any_changed:
        lines.append('EXACT MATCH')
        return '\n'.join(lines)
    lines.append(f'CHANGED BBOX=[{x0},{y0},{x1},{y1})')
    if b_changes:
        lines.append('CHANNEL B TRANSITIONS:')
        for x, y, bo, bn in b_changes:
            lines.append(f'  ({x},{y}) B={bo}({opcode_short(bo)}) -> B={bn}({opcode_short(bn)})')
        if b_changed > len(b_changes):
            lines.append(f'  [TRUNCATED {b_changed - len(b_changes)} more B changes]')
    if rg_changes:
        lines.append('FIELD R/G TRANSITIONS:')
        for x, y, ro, go, rn, gn in rg_changes:
            lines.append(f'  ({x},{y}) RG=({ro},{go}) -> RG=({rn},{gn})')
        if rg_changed > len(rg_changes):
            lines.append(f'  [TRUNCATED {rg_changed - len(rg_changes)} more R/G changes]')
    return '\n'.join(lines)
