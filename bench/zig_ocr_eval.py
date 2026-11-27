"""Synthetic Zig OCR eval: CER over 7 fonts (2 not in the atlas: NimbusS*/NimbusM*) x 12/13/14/16px x dark/light,
plus hallucination counts on no-text textures. Run from repo root: python3 bench/zig_ocr_eval.py [-v]"""
import sys, numpy as np, unicodedata
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, __import__('os').path.join(__import__('os').path.dirname(__file__), '..'))
import flowsight_ocr as f
LINES=["localhost:8080 ready","key:value pairs","std:vector and std:map","ratio 16:9 here","error:file not found",
 "Cosas Sobre Otros","Configuración del sistema","Open Settings Window","Copy Paste Save","Origen y Salida",
 "Valor Siguiente Ventana","Zona Xilofono Whisky","Usuario: admin","Sistema operativo","Click OK to continue"]
FONTS={'DejaVu':"/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",'Ubuntu':"/usr/share/fonts/truetype/ubuntu/Ubuntu-R.ttf",
 'DVMono':"/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",'Liber':"/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
 'Noto':"/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf",
 'NimbusS*':"/usr/share/fonts/opentype/urw-base35/NimbusSans-Regular.otf",'NimbusM*':"/usr/share/fonts/opentype/urw-base35/NimbusMonoPS-Regular.otf"}
def lev(a,b):
    p=list(range(len(b)+1))
    for i,ca in enumerate(a,1):
        c=[i]
        for j,cb in enumerate(b,1): c.append(min(p[j]+1,c[j-1]+1,p[j-1]+(ca!=cb)))
        p=c
    return p[-1]
def ocr(img):
    return f.run_ocr(img,0,0,'e',backend='zig')
def cer_case(fp,size,dark):
    H=len(LINES)*32+20; im=Image.new('RGB',(560,H),(30,30,30) if dark else (245,245,245)); d=ImageDraw.Draw(im); ft=ImageFont.truetype(fp,size)
    for i,t in enumerate(LINES): d.text((14,10+i*32),t,font=ft,fill=(230,230,230) if dark else (20,20,20))
    r=ocr(np.asarray(im)); rows=[[] for _ in LINES]
    for it in r['items']:
        y=(it['bbox'][1]+it['bbox'][3])/2; k=int(round((y-10-size*0.6)/32)); 
        if 0<=k<len(LINES): rows[k].append(it)
    E=N=0; bad=[]
    for k,t in enumerate(LINES):
        got=' '.join(x['text'] for x in sorted(rows[k],key=lambda x:x['bbox'][0]))
        e=lev(got,t); E+=e; N+=len(t)
        if e: bad.append(got)
    return E/N,bad
rng=np.random.default_rng(7)
def textures():
    out={}
    for s in (8,24,64):
        x=rng.integers(0,256,(1080//s+2,1920//s+2,3)).astype(np.uint8)
        out[f'smooth{s}']=np.asarray(Image.fromarray(x).resize((1920,1080),Image.BICUBIC))
    out['grain']=np.clip(rng.normal(60,20,(1080,1920,3)),0,255).astype(np.uint8)
    out['noise']=rng.integers(0,256,(1080,1920,3),dtype=np.uint8)
    yy,xx=np.mgrid[0:1080,0:1920]; out['waves']=np.stack([(128+100*np.sin(xx/37.0+np.sin(yy/53.0)*3)).astype(np.uint8)]*3,-1)
    return out
if __name__=='__main__':
    verbose='-v' in sys.argv
    tot={}
    for fn,fp in FONTS.items():
        for size in (12,13,14,16):
            for dark in (True,False):
                c,bad=cer_case(fp,size,dark); tot.setdefault(fn,[]).append(c)
                if verbose and bad: print(f"  {fn} {size} {'D' if dark else 'L'} cer={c:.3f} :: "+' | '.join(bad)[:300])
    allc=[c for v in tot.values() for c in v]
    print('CER '+'  '.join(f"{k}={np.mean(v)*100:.1f}" for k,v in tot.items())+f"  | mean={np.mean(allc)*100:.2f}%")
    print('TEXTURE items: '+'  '.join(f"{k}={len(ocr(v)['items'])}" for k,v in textures().items()))
