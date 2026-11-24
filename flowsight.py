#!/usr/bin/env python3
"""FlowSight PoC v0.1: spatial, lossless, queryable raster index for text-only agents.

Independent sidecar: NEVER executes or changes FLOW, FlowGen, or original images.
Pillow is the only runtime dependency. Storage: SQLite with RLE and palette.
Coordinate contract: top-left origin; x increases right, y down;
all rectangles are [x0,y0,x1,y1) (end EXCLUSIVE).
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import contextmanager, closing
import hashlib
import os
from pathlib import Path
import sqlite3
import sys
import tempfile

from PIL import Image, UnidentifiedImageError

import flowsight_flow as fs_flow

MAX_DIM = 512
MAX_PIXELS = MAX_DIM * MAX_DIM
MAX_COLORS = 64
MAX_COMPONENTS_SAVED = 64
SYMBOLS = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz@#'
OPCODES = {
    0:'HALT', 1:'NOP', 2:'INC', 3:'DEC', 4:'READ', 5:'WRITE',
    6:'JMP+', 7:'JMP-', 8:'SPLIT', 9:'TURN+', 10:'TURN-', 11:'PUSH',
    12:'POP', 13:'DUP', 14:'SWAP', 15:'ADD', 16:'SUB', 17:'MUL',
    18:'DIV', 19:'MOD', 20:'EQ', 21:'LT', 22:'RAND', 23:'COLOR',
    24:'TRACE', 255:'SPAWN MARKER (NOP)'
}

class Deny(ValueError):
    pass


def hex_color(rgba):
    return '#' + bytes(rgba).hex()


def parse_color(token):
    v=token.lstrip('#')
    if len(v)==6: v += 'ff'
    if len(v)!=8:
        raise Deny('COLOR must be #rrggbb or #rrggbbaa')
    try: return tuple(bytes.fromhex(v))
    except ValueError: raise Deny('invalid COLOR hex') from None


def load_png(path):
    p=Path(path)
    if p.suffix.lower() != '.png':
        raise Deny('only PNG is accepted for lossless pixel-art contract')
    try:
        with Image.open(p) as img:
            if img.format != 'PNG': raise Deny('image content is not PNG')
            w,h = img.size
            if not (1 <= w <= MAX_DIM and 1 <= h <= MAX_DIM and w*h <= MAX_PIXELS):
                raise Deny(f'canvas {w}x{h} outside 1..{MAX_DIM} each')
            rgba = img.convert('RGBA')
            raw = rgba.tobytes()
            return w,h,[tuple(raw[i:i+4]) for i in range(0,len(raw),4)],raw
    except (OSError, UnidentifiedImageError, Image.DecompressionBombError) as e:
        raise Deny(f'cannot decode PNG: {e}') from e


def index_image(image, dest, mode='visual', force=False):
    if mode == 'flow':
        return fs_flow.index_flow_image(image, dest, force, Deny)
    w,h,pixels,raw = load_png(image)
    freq=Counter(pixels)
    if len(freq)>MAX_COLORS:
        raise Deny(f'{len(freq)} distinct colors exceed {MAX_COLORS}; refusing silent quantization (photo mode not implemented)')
    palette=sorted(freq, key=lambda x:(-freq[x],x))
    ids={rgba:i for i,rgba in enumerate(palette)}
    sequence=[ids[p] for p in pixels]
    dest=Path(dest)
    if dest.exists() and not force: raise Deny(f'index exists: {dest} (use --force intentionally)')
    dest.parent.mkdir(parents=True, exist_ok=True)
    fd,tmp=tempfile.mkstemp(prefix='.flowsight_',suffix='.sqlite',dir=str(dest.parent)); os.close(fd)
    try:
        with closing(sqlite3.connect(tmp)) as db:
            db.execute('PRAGMA page_size = 4096')
            db.executescript('''
                CREATE TABLE meta(k TEXT PRIMARY KEY, v TEXT NOT NULL);
                CREATE TABLE palette(cid INTEGER PRIMARY KEY, r INT NOT NULL, g INT NOT NULL, b INT NOT NULL, a INT NOT NULL, pixels INT NOT NULL);
                CREATE TABLE runs(y INT NOT NULL, x0 INT NOT NULL, x1 INT NOT NULL, cid INT NOT NULL, PRIMARY KEY(y,x0));
                CREATE INDEX run_color_idx ON runs(cid,y);
                CREATE TABLE components(id INT PRIMARY KEY, cid INT NOT NULL, x0 INT NOT NULL, y0 INT NOT NULL, x1 INT NOT NULL, y1 INT NOT NULL, pixels INT NOT NULL);
            ''')
            kv = {
                'schema':'flowsight-rle-v1','width':str(w),'height':str(h),'mode':mode,
                'source_name':Path(image).name,
                'source_sha256':hashlib.sha256(Path(image).read_bytes()).hexdigest(),
                'rgba_sha256':hashlib.sha256(raw).hexdigest(),
                'colors':str(len(palette))
            }
            db.executemany('INSERT INTO meta VALUES (?,?)', sorted(kv.items()))
            db.executemany('INSERT INTO palette VALUES (?,?,?,?,?,?)',
                    [(i,*c,freq[c]) for i,c in enumerate(palette)])
            runrows=[]
            for y in range(h):
                x=0
                while x<w:
                    cid=sequence[y*w+x]; x0=x; x+=1
                    while x<w and sequence[y*w+x]==cid: x+=1
                    runrows.append((y,x0,x,cid))
            db.executemany('INSERT INTO runs VALUES (?,?,?,?)',runrows)
            # Connected components: exact 4-neighbor connectivity; only largest 64 retained.
            seen=bytearray(w*h); comps=[]
            for start, cid in enumerate(sequence):
                if seen[start]: continue
                seen[start]=1; stack=[start]; n=0
                xmin=w;ymin=h;xmax=0;ymax=0
                while stack:
                    i=stack.pop(); y,x=divmod(i,w); n+=1
                    xmin=min(xmin,x); ymin=min(ymin,y); xmax=max(xmax,x+1); ymax=max(ymax,y+1)
                    for j in ((i-1 if x>0 else -1),(i+1 if x<w-1 else -1),
                              (i-w if y>0 else -1),(i+w if y<h-1 else -1)):
                        if j>=0 and not seen[j] and sequence[j]==cid:
                            seen[j]=1; stack.append(j)
                comps.append((cid,xmin,ymin,xmax,ymax,n))
            comps.sort(key=lambda z:(-z[5],z[2],z[1],z[0]))
            db.executemany('INSERT INTO components VALUES (?,?,?,?,?,?,?)',
                  [(i,*comp) for i,comp in enumerate(comps[:MAX_COMPONENTS_SAVED])])
            db.execute('INSERT INTO meta VALUES (?,?)',('component_count',str(len(comps))))
            db.execute('INSERT INTO meta VALUES (?,?)',('component_saved',str(min(len(comps),MAX_COMPONENTS_SAVED))))
            db.commit()
        os.replace(tmp,dest)
    finally:
        if os.path.exists(tmp): os.unlink(tmp)
    return w,h,len(palette),len(runrows),len(comps),len(raw)


@contextmanager
def open_index(path):
    uri=Path(path).resolve().as_uri()+'?mode=ro'
    try:
        db=sqlite3.connect(uri,uri=True)
        try:
            meta=dict(db.execute('SELECT k,v FROM meta'))
            if meta.get('schema') not in ('flowsight-rle-v1', fs_flow.SCHEMA_FLOW):
                raise Deny('unsupported or invalid index schema')
            yield db,meta
        finally: db.close()
    except sqlite3.Error as e: raise Deny(f'bad index: {e}') from e


def _is_flow_schema(meta) -> bool:
    return meta.get('schema') == fs_flow.SCHEMA_FLOW


def pal(db):
    return {r[0]:tuple(r[1:5]) for r in db.execute('SELECT cid,r,g,b,a FROM palette')}


def get_pixel_ids(db,w,h):
    out=[-1]*(w*h)
    for y,x0,x1,cid in db.execute('SELECT y,x0,x1,cid FROM runs ORDER BY y,x0'):
        if not (0<=y<h and 0<=x0<x1<=w): raise Deny('invalid run in index')
        s=y*w+x0
        if any(c!=-1 for c in out[s:s+x1-x0]): raise Deny('overlapping runs in index')
        out[s:s+x1-x0]=[cid]*(x1-x0)
    if -1 in out: raise Deny('missing pixel coverage in index')
    return out


def decoded_rgba(db,meta):
    if _is_flow_schema(meta):
        return fs_flow.flow_decoded_rgba(db, meta, Deny)
    w=int(meta['width']);h=int(meta['height']); ps=pal(db)
    pixels=get_pixel_ids(db,w,h)
    try: raw=b''.join(bytes(ps[c]) for c in pixels)
    except KeyError: raise Deny('run references missing palette') from None
    sha=hashlib.sha256(raw).hexdigest()
    if sha!=meta['rgba_sha256']: raise Deny('rgba hash mismatch: index is corrupted')
    return raw


def bounded(text, max_chars):
    if len(text)<=max_chars: return text
    suffix='\n[TRUNCATED: request a more targeted query; this is not complete coverage]'
    return text[:max(0,max_chars-len(suffix))]+suffix


def summary(db,meta, cols=24,rows=16, max_chars=3500):
    if _is_flow_schema(meta):
        return fs_flow.flow_summary(db, meta, cols, rows, max_chars, bounded, Deny)
    w=int(meta['width']);h=int(meta['height']);P=pal(db)
    vec=get_pixel_ids(db,w,h)
    cols=max(1,min(cols,w,64));rows=max(1,min(rows,h,64))
    hist=Counter(vec)
    ordered_colors=sorted(hist.items(), key=lambda kv:(-kv[1], kv[0]))

    # Compute minimap first so we know which palette colors were hidden by majority voting
    minimap_lines=[]
    shown_in_minimap=set()
    for ty in range(rows):
        y0=ty*h//rows; y1=(ty+1)*h//rows
        s=''
        for tx in range(cols):
            x0=tx*w//cols; x1=(tx+1)*w//cols
            freq=Counter(vec[y*w+x] for y in range(y0,y1) for x in range(x0,x1))
            best,n=min(freq.items(),key=lambda v:(-v[1],v[0]))
            if n*10 >= (x1-x0)*(y1-y0)*6:
                s+=SYMBOLS[best]
                shown_in_minimap.add(best)
            else:
                s+='?'
        minimap_lines.append('  '+s)

    hidden_cids=[cid for cid,_ in ordered_colors if cid not in shown_in_minimap]
    out=[f'FLOWSIGHT v1 | {w}x{h} RGBA | mode={meta["mode"]} | SHA256(decoded)={meta["rgba_sha256"][:16]}',
         'Coordinates: origin top-left (0,0); x right, y down; regions [x0,y0,x1,y1) exclusive.',
         'Palette (observed colors, no inferred object identity):']
    for cid,count in ordered_colors[:12]:
        tag=' [hidden-in-minimap]' if cid not in shown_in_minimap else ''
        out.append(f'  {SYMBOLS[cid]}=C{cid} {hex_color(P[cid])} {count}/{w*h} ({count*100/(w*h):.1f}%){tag}')
    if len(ordered_colors)>12:out.append(f'  ... {len(ordered_colors)-12} additional colors (use scanline/region)')
    out.append(f'Minimap {cols}x{rows} (tile majority; ? if winning color < 60%):')
    out.extend(minimap_lines)
    if hidden_cids:
        out.append(f'Minority colors absent from minimap ({len(hidden_cids)} total, exact bbox):')
        for cid in hidden_cids[:8]:
            row=db.execute('SELECT MIN(x0), MIN(y), MAX(x1), MAX(y)+1, SUM(x1-x0) FROM runs WHERE cid=?',(cid,)).fetchone()
            out.append(f'  {SYMBOLS[cid]}=C{cid} {hex_color(P[cid])} pixels={row[4]} bbox=[{row[0]},{row[1]},{row[2]},{row[3]})')
        if len(hidden_cids)>8:
            out.append(f'  [TRUNCATED: {len(hidden_cids)-8} more minority colors; use find]')
    total=int(meta['component_count']); saved=int(meta['component_saved'])
    out.append(f'Components (4-connected): {total}; largest {saved} indexed:')
    for _,cid,x0,y0,x1,y1,count in db.execute('SELECT id,cid,x0,y0,x1,y1,pixels FROM components ORDER BY id LIMIT 7'):
        out.append(f'  C{cid} bbox=[{x0},{y0},{x1},{y1}) area={count}')
    if total>saved:out.append(f'  component detail truncated in index ({total-saved} not individually retained)')
    out.append('Queries: point, scanline, region [--grid], find, edges [--region], components, diff, export-flowpaint, rebuild, verify')
    if meta['mode']=='flow':
        out.append('FLOW raw channels: R,G define vector ((value/127.5)-1); B defines instruction. NOT image gradients.')
    return bounded('\n'.join(out),max_chars)


def validate_box(w,h,x,y,bw,bh):
    if bw<1 or bh<1 or x<0 or y<0 or x+bw>w or y+bh>h:
        raise Deny(f'box ({x},{y},{bw},{bh}) outside {w}x{h}')


def scanline(db,meta,y,max_lines=100):
    if _is_flow_schema(meta):
        return fs_flow.flow_scanline(db, meta, y, max_lines, Deny)
    h=int(meta['height'])
    if not 0<=y<h: raise Deny('scanline y outside image')
    P=pal(db)
    data=list(db.execute('SELECT x0,x1,cid FROM runs WHERE y=? ORDER BY x0',(y,)))
    lines=[f'y={y} runs={len(data)}; x intervals [x0,x1) exclusive:']
    lines += [f'  x=[{a},{b}) len={b-a} C{c} {hex_color(P[c])}' for a,b,c in data[:max_lines]]
    if len(data)>max_lines:lines.append(f'[TRUNCATED: {len(data)-max_lines} additional runs; use region]')
    return '\n'.join(lines)


def region(db,meta,x,y,bw,bh,max_lines=50, grid=False):
    if _is_flow_schema(meta):
        return fs_flow.flow_region(db, meta, x, y, bw, bh, max_lines, grid, validate_box, Deny)
    w=int(meta['width']);h=int(meta['height']);validate_box(w,h,x,y,bw,bh)
    if grid and (bw > 64 or bh > 64 or bw * bh > 1024):
        raise Deny(f'--grid region {bw}x{bh} ({bw*bh} px) exceeds 64x64 / 1024 px fovea limit; query a smaller box')
    P=pal(db); counts=Counter(); lines=[];totalruns=0
    grid_rows = [[-1]*bw for _ in range(bh)] if grid else None
    for yy,lo,hi,cid in db.execute('SELECT y,x0,x1,cid FROM runs WHERE y>=? AND y<? AND x1>? AND x0<? ORDER BY y,x0',(y,y+bh,x,x+bw)):
        a=max(x,lo);b=min(x+bw,hi)
        if b<=a:continue
        counts[cid]+=b-a;totalruns+=1
        if grid:
            row_buf=grid_rows[yy-y]
            for xx in range(a,b): row_buf[xx-x]=cid
        elif len(lines)<max_lines:
            lines.append(f'  y={yy} x=[{a},{b}) C{cid}')
    if sum(counts.values())!=bw*bh:raise Deny('index coverage mismatch in queried region')
    ordered_cov=sorted(counts.items(), key=lambda kv:(-kv[1], kv[0]))
    s=[f'REGION [{x},{y},{x+bw},{y+bh}) exact pixels={bw*bh}, intersecting runs={totalruns}',
       'COVERAGE: '+', '.join(f'{SYMBOLS[k]}=C{k}:{v} ({v/(bw*bh):.1%}) {hex_color(P[k])}' for k,v in ordered_cov)]
    if grid:
        s.append(f'GRID 1:1 exact (x={x}..{x+bw-1}, y={y}..{y+bh-1}; 1 char = 1 pixel):')
        x_units=''.join(str((x+i)%10) for i in range(bw))
        s.append(f'  x%10: {x_units}')
        for ry in range(bh):
            s.append(f'  y={y+ry:3d}: '+''.join(SYMBOLS[c] for c in grid_rows[ry]))
    else:
        s.append('RLE intersections:')
        s.extend(lines)
        if totalruns>max_lines:s.append(f'[TRUNCATED {totalruns-max_lines} more runs: split region, lower height, or use --grid]')
    return '\n'.join(s)


def point(db,meta,x,y):
    if _is_flow_schema(meta):
        return fs_flow.flow_point(db, meta, x, y, validate_box, Deny)
    w=int(meta['width']);h=int(meta['height']);validate_box(w,h,x,y,1,1)
    p=db.execute('SELECT cid FROM runs WHERE y=? AND x0<=? AND x1>?',(y,x,x)).fetchone()
    if not p:raise Deny('point missing from index')
    cid=p[0];rgba=pal(db)[cid];out=f'({x},{y}) C{cid} {hex_color(rgba)}'
    if meta['mode']=='flow':
        r,g,b,a=rgba
        out+=f' | FLOW v=({r/127.5-1:.5f},{g/127.5-1:.5f}); B={b} {OPCODES.get(b,"NOP (25..254)")}'
    return out


def find_color(db,meta,color,max_lines=35):
    if _is_flow_schema(meta):
        return fs_flow.flow_find(db, meta, color, max_lines, parse_color, Deny)
    rgba=parse_color(color);P=pal(db)
    match=[cid for cid,c in P.items() if c==rgba]
    if not match:return f'{hex_color(rgba)} exact matches=0'
    cid=match[0]
    data=list(db.execute('SELECT y,x0,x1 FROM runs WHERE cid=? ORDER BY y,x0',(cid,)))
    count=sum(b-a for _,a,b in data)
    x0=min(a for _,a,b in data);x1=max(b for _,a,b in data)
    y0=data[0][0];y1=data[-1][0]+1
    lines=[f'{hex_color(rgba)} exact pixels={count} bbox=[{x0},{y0},{x1},{y1}) rows/runs={len(data)}']
    lines += [f'  y={yy} x=[{a},{b})' for yy,a,b in data[:max_lines]]
    if len(data)>max_lines:lines.append(f'[TRUNCATED {len(data)-max_lines} runs: use region]')
    return '\n'.join(lines)


def components(db,meta,max_lines=15):
    if _is_flow_schema(meta):
        return fs_flow.flow_components(db, meta, max_lines)
    total=int(meta['component_count']);saved=int(meta['component_saved']);P=pal(db)
    lines=[f'4-connected components total={total}; saved={saved} (only top {MAX_COMPONENTS_SAVED} saved)']
    for cid,color,x0,y0,x1,y1,count in db.execute('SELECT id,cid,x0,y0,x1,y1,pixels FROM components ORDER BY id LIMIT ?', (max_lines,)):
        lines.append(f'  id={cid} C{color} {hex_color(P[color])} bbox=[{x0},{y0},{x1},{y1}) pixels={count}')
    if total>saved:lines.append('WARNING: components list incomplete; only 64 biggest retained')
    return '\n'.join(lines)


def edges(db,meta,max_lines=40, box=None):
    """Exact color discontinuity edges as merged horizontal/vertical grid segments.

    This is a discrete boundary vectorization, NOT a photographic/semantic contour.
    V x Y0 Y1 separates left C_a from right C_b.
    H y X0 X1 separates above C_a from below C_b.
    Segments have half-open spans and integer grid-line coordinates.
    """
    if _is_flow_schema(meta):
        return fs_flow.flow_edges(db, meta, box, max_lines, validate_box, Deny)
    w=int(meta['width']);h=int(meta['height'])
    if box is None:
        x0,y0,bw,bh=0,0,w,h
    else:
        x0,y0,bw,bh=box
        validate_box(w,h,x0,y0,bw,bh)
    x1,y1=x0+bw,y0+bh
    seq=get_pixel_ids(db,w,h)
    segments=[]
    # Vertical boundaries strictly inside [x0,x1) x [y0,y1)
    for x in range(max(1,x0+1),x1):
        y=y0
        while y<y1:
            a=seq[y*w+x-1];b=seq[y*w+x]
            if a==b:y+=1;continue
            ys=y;y+=1
            while y<y1 and seq[y*w+x-1]==a and seq[y*w+x]==b:y+=1
            segments.append(('V',x,ys,y,a,b))
    # Horizontal boundaries strictly inside [x0,x1) x [y0,y1)
    for y in range(max(1,y0+1),y1):
        x=x0
        while x<x1:
            a=seq[(y-1)*w+x];b=seq[y*w+x]
            if a==b:x+=1;continue
            xs=x;x+=1
            while x<x1 and seq[(y-1)*w+x]==a and seq[y*w+x]==b:x+=1
            segments.append(('H',y,xs,x,a,b))
    scope=f' scope=[{x0},{y0},{x1},{y1})' if box is not None else ''
    lines=[f'INTERNAL COLOR EDGES{scope} exact segments={len(segments)} (V: left/right; H: above/below).']
    for orient,at,start,stop,a,b in segments[:max_lines]:
        if orient=='V':lines.append(f'  V x={at} y=[{start},{stop}) C{a}->C{b}')
        else:lines.append(f'  H y={at} x=[{start},{stop}) C{a}->C{b}')
    if len(segments)>max_lines:lines.append(f'[TRUNCATED {len(segments)-max_lines} edge segments; use --region to focus]')
    return '\n'.join(lines)

def rectangles(db):
    active={};result=[];current_y=None; row=[]
    def flush(y,row):
        nonlocal active,result
        keys={(x0,x1,cid) for x0,x1,cid in row}
        for key,(sy,hh) in list(active.items()):
            if key not in keys:
                result.append((key[0],sy,key[1]-key[0],hh,key[2]))
                del active[key]
        for key in sorted(keys):
            if key in active:
                sy,hh=active[key];active[key]=(sy,hh+1)
            else: active[key]=(y,1)
    for y,x0,x1,cid in db.execute('SELECT y,x0,x1,cid FROM runs ORDER BY y,x0'):
        if current_y is None:current_y=y
        if y!=current_y:
            flush(current_y,row);row=[];current_y=y
        row.append((x0,x1,cid))
    if current_y is not None:flush(current_y,row)
    for key,(sy,hh) in active.items():result.append((key[0],sy,key[1]-key[0],hh,key[2]))
    result.sort(key=lambda r:(r[1],r[0],r[4]))
    return result


def field_query(db, meta, box=None, cols=16, rows=16):
    if not _is_flow_schema(meta):
        raise Deny('field query requires a --mode flow index')
    return fs_flow.flow_field(db, meta, box, cols, rows, validate_box, Deny)


def instructions_query(db, meta, opcode=None, box=None, include_halt=False, max_lines=60):
    if not _is_flow_schema(meta):
        raise Deny('instructions query requires a --mode flow index')
    return fs_flow.flow_instructions(db, meta, opcode, box, include_halt, max_lines, validate_box, Deny)


def flowpaint_source(db,meta,max_ops=50000, max_undo_mb=64):
    if _is_flow_schema(meta):
        w = int(meta['width']); h = int(meta['height'])
        distinct_rgb = int(meta.get('distinct_rgb', 0))
        if distinct_rgb > MAX_COLORS:
            raise Deny(
                f'FLOW program has {distinct_rgb} distinct RGB values (> {MAX_COLORS}); '
                f'continuous R/G vector fields should be reconstructed via rebuild instead of .flowpaint'
            )
        raw = fs_flow.flow_decoded_rgba(db, meta, Deny)
        pixels = [tuple(raw[i:i + 4]) for i in range(0, len(raw), 4)]
        freq = Counter(pixels)
        palette = sorted(freq, key=lambda x: (-freq[x], x))
        ids = {rgba: i for i, rgba in enumerate(palette)}
        seq = [ids[p] for p in pixels]
        # Build vertical-merged rects in memory
        active = {}; rects = []
        for y in range(h):
            row_keys = set()
            x = 0
            while x < w:
                cid = seq[y * w + x]; x0 = x; x += 1
                while x < w and seq[y * w + x] == cid:
                    x += 1
                row_keys.add((x0, x, cid))
            for key, (sy, hh) in list(active.items()):
                if key not in row_keys:
                    rects.append((key[0], sy, key[1] - key[0], hh, key[2]))
                    del active[key]
            for key in sorted(row_keys):
                if key in active:
                    sy, hh = active[key]; active[key] = (sy, hh + 1)
                else:
                    active[key] = (y, 1)
        for key, (sy, hh) in active.items():
            rects.append((key[0], sy, key[1] - key[0], hh, key[2]))
        rects.sort(key=lambda r: (r[1], r[0], r[4]))
        if len(rects) > max_ops:
            raise Deny(f'{len(rects)} RECT instructions exceed --max-ops {max_ops}; refuse massive export')
        estimated_undo = w * h * 4 * min(len(rects), 1024)
        if estimated_undo > max_undo_mb * 1024 * 1024:
            raise Deny(f'upstream FLOW paint UNDO could require {estimated_undo // (1024 * 1024)} MiB; exceeds {max_undo_mb} MiB limit; use rebuild instead')
        s = [f'# FlowSight v1 | reconstructed from decoded RGBA hash {meta["rgba_sha256"]}',
             '# Exact pixel reconstruction in FLOW paint (CANVAS / COLOR / RECT subset).',
             f'CANVAS {w} {h}']
        for cid, rgba in enumerate(palette):
            s.append(f'COLOR c{cid} ' + ' '.join(map(str, rgba)))
        s += [f'RECT {x} {y} {rw} {rh} c{cid}' for x, y, rw, rh, cid in rects]
        return '\n'.join(s) + '\n', len(rects)
    w=int(meta['width']);h=int(meta['height'])
    rects=rectangles(db)
    if len(rects)>max_ops:raise Deny(f'{len(rects)} RECT instructions exceed --max-ops {max_ops}; refuse massive export')
    # FLOW painter stores up to 1024 full-canvas snapshots for UNDO; avoid OOM.
    estimated_undo=w*h*4*min(len(rects),1024)
    if estimated_undo > max_undo_mb*1024*1024:
        raise Deny(f'upstream FLOW paint UNDO could require {estimated_undo//(1024*1024)} MiB; exceeds {max_undo_mb} MiB limit; use rebuild instead')
    P=pal(db)
    s=[f'# FlowSight v1 | reconstructed from decoded RGBA hash {meta["rgba_sha256"]}',
       '# Exact pixel reconstruction in FLOW paint (CANVAS / COLOR / RECT subset).',
       f'CANVAS {w} {h}']
    for cid in sorted(P):s.append(f'COLOR c{cid} '+ ' '.join(map(str,P[cid])))
    s += [f'RECT {x} {y} {rw} {rh} c{cid}' for x,y,rw,rh,cid in rects]
    return '\n'.join(s)+'\n',len(rects)


def save_rgba_png(db,meta,dest,force=False):
    p=Path(dest)
    if p.exists() and not force:raise Deny(f'file exists: {dest}')
    raw=decoded_rgba(db,meta)
    if _is_flow_schema(meta):
        img=Image.frombytes('RGBA',(int(meta['width']),int(meta['height'])),raw).convert('RGB')
        img.save(p,format='PNG')
    else:
        img=Image.frombytes('RGBA',(int(meta['width']),int(meta['height'])),raw)
        img.save(p,format='PNG')
    return len(raw)


def verify(db,meta,input_png=None):
    if _is_flow_schema(meta):
        return fs_flow.flow_verify(db, meta, input_png, Deny)
    raw=decoded_rgba(db,meta)
    result=f'PASS index integrity: decoded rgba_sha256={hashlib.sha256(raw).hexdigest()}'
    if input_png:
        w,h,_,src=load_png(input_png)
        if (w,h)!=(int(meta['width']),int(meta['height'])):raise Deny('image size mismatch')
        if raw!=src:raise Deny('pixel mismatch with source image')
        result+='\nPASS input: exact decoded RGBA byte equality'
    return result


def diff(db,meta,other,max_rects=20):
    if _is_flow_schema(meta):
        return fs_flow.flow_diff(db, meta, other, max_rects, Deny)
    w,h=int(meta['width']),int(meta['height'])
    ww,hh,_,target=load_png(other)
    if (w,h)!=(ww,hh):raise Deny(f'diff requires same size; got {ww}x{hh} vs {w}x{h}')
    base=decoded_rgba(db,meta)
    P=pal(db)
    rev_pal={rgba:cid for cid,rgba in P.items()}
    def _lbl(px_bytes):
        t=tuple(px_bytes)
        cid=rev_pal.get(t)
        return f'C{cid}({hex_color(t)})' if cid is not None else hex_color(t)
    pixels=w*h; changed=0; x0=w;y0=h;x1=0;y1=0; rowruns=[]; transitions=Counter()
    for y in range(h):
        x=0
        while x<w:
            i=(y*w+x)*4
            old_px=base[i:i+4]; new_px=target[i:i+4]
            if old_px==new_px:x+=1;continue
            start=x
            while x<w:
                j=(y*w+x)*4
                if base[j:j+4]!=old_px or target[j:j+4]!=new_px:
                    break
                x+=1;changed+=1
            rowruns.append((y,start,x,old_px,new_px))
            transitions[(old_px,new_px)]+=x-start
            x0=min(x0,start);x1=max(x1,x);y0=min(y0,y);y1=max(y1,y+1)
    lines=[f'DIFF {w}x{h} decoded RGBA pixel-changes={changed}/{pixels} ({changed/pixels:.2%})']
    if changed:
        lines.append(f'CHANGED BBOX=[{x0},{y0},{x1},{y1})')
        top_t=sorted(transitions.items(), key=lambda kv:(-kv[1], kv[0]))
        lines.append('COLOR TRANSITIONS: '+', '.join(
            f'{_lbl(o)}->{_lbl(n)}:{cnt}px' for (o,n),cnt in top_t[:8]
        ))
        if len(top_t)>8:
            lines.append(f'  [TRUNCATED {len(top_t)-8} additional color transition pairs]')
        lines += [f'  y={yy} x=[{a},{b}) {_lbl(o)}->{_lbl(n)}' for yy,a,b,o,n in rowruns[:max_rects]]
        if len(rowruns)>max_rects: lines.append(f'[TRUNCATED {len(rowruns)-max_rects} changed runs; use smaller crops]')
    else:lines.append('EXACT MATCH')
    return '\n'.join(lines)


def main(argv=None):
    a=argparse.ArgumentParser(description='FlowSight: eyes by indexed raster probes (no neural vision)')
    sub=a.add_subparsers(dest='command',required=True)
    sp=sub.add_parser('index');sp.add_argument('image');sp.add_argument('db');sp.add_argument('--mode',choices=['visual','flow'],default='visual');sp.add_argument('--force',action='store_true')
    sp=sub.add_parser('summary');sp.add_argument('db');sp.add_argument('--cols',type=int,default=24);sp.add_argument('--rows',type=int,default=16);sp.add_argument('--max-chars',type=int,default=3500)
    sp=sub.add_parser('point');sp.add_argument('db');sp.add_argument('x',type=int);sp.add_argument('y',type=int)
    sp=sub.add_parser('scanline');sp.add_argument('db');sp.add_argument('y',type=int);sp.add_argument('--limit',type=int,default=100)
    sp=sub.add_parser('region');sp.add_argument('db');sp.add_argument('x',type=int);sp.add_argument('y',type=int);sp.add_argument('w',type=int);sp.add_argument('h',type=int);sp.add_argument('--limit',type=int,default=50);sp.add_argument('--grid',action='store_true')
    sp=sub.add_parser('field');sp.add_argument('db');sp.add_argument('--region',nargs=4,type=int,metavar=('X','Y','W','H'),default=None);sp.add_argument('--cols',type=int,default=16);sp.add_argument('--rows',type=int,default=16)
    sp=sub.add_parser('instructions');sp.add_argument('db');sp.add_argument('--opcode',default=None);sp.add_argument('--region',nargs=4,type=int,metavar=('X','Y','W','H'),default=None);sp.add_argument('--include-halt',action='store_true');sp.add_argument('--limit',type=int,default=60)
    sp=sub.add_parser('find');sp.add_argument('db');sp.add_argument('color');sp.add_argument('--limit',type=int,default=35)
    sp=sub.add_parser('components');sp.add_argument('db');sp.add_argument('--limit',type=int,default=15)
    sp=sub.add_parser('edges');sp.add_argument('db');sp.add_argument('--region',nargs=4,type=int,metavar=('X','Y','W','H'),default=None);sp.add_argument('--limit',type=int,default=40)
    sp=sub.add_parser('diff');sp.add_argument('db');sp.add_argument('other');sp.add_argument('--limit',type=int,default=20)
    sp=sub.add_parser('rebuild');sp.add_argument('db');sp.add_argument('output');sp.add_argument('--force',action='store_true')
    sp=sub.add_parser('verify');sp.add_argument('db');sp.add_argument('image',nargs='?')
    sp=sub.add_parser('export-flowpaint');sp.add_argument('db');sp.add_argument('output');sp.add_argument('--max-ops',type=int,default=50000);sp.add_argument('--max-undo-mb',type=int,default=64);sp.add_argument('--force',action='store_true')
    args=a.parse_args(argv)
    try:
        if args.command=='index':
            w,h,colors,runs,comps,bytes_raw=index_image(args.image,args.db,args.mode,args.force)
            print(f'INDEX OK {w}x{h} colors={colors} runs={runs} components={comps} rgba_bytes={bytes_raw} sqlite_bytes={Path(args.db).stat().st_size}')
            return 0
        with open_index(args.db) as (db,meta):
            if args.command=='summary':print(summary(db,meta,args.cols,args.rows,max(200,args.max_chars)))
            elif args.command=='point':print(point(db,meta,args.x,args.y))
            elif args.command=='scanline':print(scanline(db,meta,args.y,max(1,args.limit)))
            elif args.command=='region':print(region(db,meta,args.x,args.y,args.w,args.h,max(1,args.limit),grid=args.grid))
            elif args.command=='field':print(field_query(db,meta,tuple(args.region) if args.region else None,args.cols,args.rows))
            elif args.command=='instructions':print(instructions_query(db,meta,args.opcode,tuple(args.region) if args.region else None,args.include_halt,max(1,args.limit)))
            elif args.command=='find':print(find_color(db,meta,args.color,max(1,args.limit)))
            elif args.command=='components':print(components(db,meta,max(1,args.limit)))
            elif args.command=='edges':print(edges(db,meta,max(1,args.limit),box=tuple(args.region) if args.region else None))
            elif args.command=='diff':print(diff(db,meta,args.other,max(1,args.limit)))
            elif args.command=='verify':print(verify(db,meta,args.image))
            elif args.command=='rebuild':
                num=save_rgba_png(db,meta,args.output,args.force);print(f'REBUILD OK {args.output}: {num} RGBA bytes exact')
            elif args.command=='export-flowpaint':
                source,n=flowpaint_source(db,meta,args.max_ops,args.max_undo_mb)
                dest=Path(args.output)
                if dest.exists() and not args.force:raise Deny(f'file exists: {dest}')
                dest.write_text(source,encoding='utf-8')
                print(f'FLOWPAINT OK {dest}: RECT={n} chars={len(source)}')
        return 0
    except (Deny,FileNotFoundError,PermissionError,sqlite3.Error) as e:
        print(f'DENY {e}',file=sys.stderr);return 2

if __name__=='__main__':sys.exit(main())
