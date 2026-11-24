#!/usr/bin/env python3
"""Print actual index/query sizes and checks; no LLM token/billing claims."""
from collections import Counter
from pathlib import Path
import hashlib
import json
import sys
import time

from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import flowsight as fs
from examples.make_examples import ROOT as DEMO


def benchmark():
    results=[]
    for name,mode in [('sprite','visual'),('flow_hello','flow')]:
        png=DEMO/(name+'.png');db=ROOT/'reports'/(name+'.sqlite')
        started=time.perf_counter()
        w,h,pal,runs,components,raw_bytes=fs.index_image(png,db,mode,force=True)
        ms=(time.perf_counter()-started)*1000
        with fs.open_index(db) as (conn,meta):
            l0=fs.summary(conn,meta)
            fs.verify(conn,meta,str(png))
            prog,rects=fs.flowpaint_source(conn,meta)
            out=ROOT/'reports'/(name+'_restored.png')
            fs.save_rgba_png(conn,meta,out,force=True)
            raw=fs.decoded_rgba(conn,meta)
        rgba=[tuple(x) for x in Image.open(png).convert('RGBA').get_flattened_data()]
        naive_json=len(json.dumps([{'x':i%w,'y':i//w,'rgba':p} for i,p in enumerate(rgba)]))
        stats={'name':name,'w':w,'h':h,'colors':pal,'runs':runs,'components':components,
               'source_png_bytes':png.stat().st_size,'rgba_bytes':raw_bytes,
               'sqlite_bytes':db.stat().st_size, 'naive_pixel_json_bytes':naive_json,
               'l0_utf8_bytes':len(l0.encode()),'flowpaint_utf8_bytes':len(prog.encode()),
               'rects':rects,'index_ms':round(ms,2),'verified_rgba_hash':hashlib.sha256(raw).hexdigest(),
               'match':Image.open(out).convert('RGBA').tobytes()==raw}
        results.append(stats)
        (ROOT/'reports'/(name+'_L0.txt')).write_text(l0+'\n',encoding='utf-8')
        (ROOT/'reports'/(name+'.flowpaint')).write_text(prog,encoding='utf-8')
    with fs.open_index(ROOT/'reports'/'sprite.sqlite') as (db,meta):
        (ROOT/'reports'/'queries.txt').write_text('\n\n'.join([
            '=== POINT ===\n'+fs.point(db,meta,15,24),
            '=== FIND GREEN ===\n'+fs.find_color(db,meta,'#2ecc71'),
            '=== REGION ===\n'+fs.region(db,meta,12,20,12,10),
            '=== SCANLINE ===\n'+fs.scanline(db,meta,24),
            '=== VECTOR EDGES ===\n'+fs.edges(db,meta,20),
            '=== DIFF ===\n'+fs.diff(db,meta,str(DEMO/'sprite_edited.png')),
        ])+'\n',encoding='utf-8')
    (ROOT/'reports'/'benchmark.json').write_text(json.dumps(results,indent=2)+'\n',encoding='utf-8')
    for s in results:
        print(f"{s['name']:10} {s['w']}x{s['h']} | {s['colors']} colors | {s['runs']} runs | {s['rects']} RECT | "
              f"L0 {s['l0_utf8_bytes']} B vs naive JSON {s['naive_pixel_json_bytes']} B | "
              f"index {s['sqlite_bytes']} B | {s['index_ms']} ms | pixel_exact={s['match']}")

if __name__=='__main__':benchmark()
