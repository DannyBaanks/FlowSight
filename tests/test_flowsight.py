import hashlib
from pathlib import Path
from contextlib import closing
import sqlite3
import tempfile
import unittest
import sys

from PIL import Image, ImageDraw
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import flowsight as f


def demo_image(path, size=(32,24)):
    im=Image.new('RGBA',size,(10,10,18,255))
    d=ImageDraw.Draw(im)
    d.rectangle((3,5,21,10),fill=(46,204,113,255))
    d.rectangle((19,2,26,6),fill=(231,76,60,255))
    d.point((8,7),fill=(250,250,250,0))
    im.save(path)
    return im


def interpret_subset(source):
    """Independent oracle for the exact CANVAS/COLOR/RECT subset FLOW paint accepts.
    This is NOT the upstream FLOW parser; optional real-FLOW test below.
    """
    im=None; colors={}
    for line in source.splitlines():
        if not line or line.startswith('#'):continue
        op,*params=line.split()
        if op=='CANVAS':
            w,h=map(int,params);im=Image.new('RGBA',(w,h),(0,0,0,0))
        elif op=='COLOR':
            name,*rgba=params;colors[name]=tuple(map(int,rgba))
        elif op=='RECT':
            x,y,w,h=map(int,params[:4]);cid=params[4]
            ImageDraw.Draw(im).rectangle((x,y,x+w-1,y+h-1),fill=colors[cid])
        else:raise AssertionError(op)
    return im


class FlowSightTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
        self.png=self.root/'sprite.png';demo_image(self.png)
        self.db=self.root/'scene.sqlite'
        self.args=f.index_image(self.png,self.db)

    def tearDown(self):self.tmp.cleanup()

    def test_index_and_roundtrip_rebuild(self):
        with f.open_index(self.db) as (db,meta):
            self.assertIn('PASS', f.verify(db,meta,str(self.png)))
            out=self.root/'out.png';f.save_rgba_png(db,meta,out)
            self.assertEqual(Image.open(self.png).convert('RGBA').tobytes(),Image.open(out).tobytes())

    def test_summary_is_bounded_and_has_minimap(self):
        with f.open_index(self.db) as (db,meta):
            out=f.summary(db,meta,max_chars=1800)
            self.assertLessEqual(len(out),1800)
            self.assertIn('Minimap',out)
            self.assertIn('top-left',out)

    def test_region_coverage_exact(self):
        with f.open_index(self.db) as (db,meta):
            txt=f.region(db,meta,3,5,19,6)
            self.assertIn('pixels=114',txt)
            self.assertIn('COVERAGE',txt)
            with self.assertRaises(f.Deny):f.region(db,meta,-1,0,5,5)

    def test_point_and_scanline(self):
        with f.open_index(self.db) as (db,meta):
            self.assertIn('#2ecc71ff',f.point(db,meta,4,7))
            self.assertIn('x=[',f.scanline(db,meta,7))
            with self.assertRaises(f.Deny):f.point(db,meta,32,0)

    def test_vector_edges_are_grounded(self):
        with f.open_index(self.db) as (db,meta):
            out=f.edges(db,meta,20)
            self.assertIn('INTERNAL COLOR EDGES exact segments=',out)
            self.assertIn('V x=',out)
            self.assertIn('H y=',out)
            self.assertIn('[TRUNCATED',f.edges(db,meta,1))

    def test_find_exact_palette(self):
        with f.open_index(self.db) as (db,meta):
            txt=f.find_color(db,meta,'#e74c3c')
            self.assertIn('bbox=[19,2,27,7)',txt)
            self.assertIn('exact pixels=',txt)
            self.assertIn('exact matches=0',f.find_color(db,meta,'#123456'))

    def test_export_flowpaint_exact_pixel_reconstruction(self):
        with f.open_index(self.db) as (db,meta):
            src,n=f.flowpaint_source(db,meta)
            self.assertIn('CANVAS 32 24',src)
            self.assertTrue(n<50)
            self.assertEqual(Image.open(self.png).convert('RGBA').tobytes(),interpret_subset(src).tobytes())
            with self.assertRaises(f.Deny):f.flowpaint_source(db,meta,max_ops=0)

    def test_diff_exact(self):
        other=self.root/'modified.png';im=Image.open(self.png).convert('RGBA')
        im.putpixel((1,2),(255,0,0,255));im.putpixel((2,2),(255,0,0,255));im.save(other)
        with f.open_index(self.db) as (db,meta):
            self.assertIn('pixel-changes=2/',f.diff(db,meta,str(other)))
            self.assertIn('BBOX=[1,2,3,3)',f.diff(db,meta,str(other)))
            self.assertIn('EXACT MATCH',f.diff(db,meta,str(self.png)))

    def test_fail_closed_high_color_image(self):
        noisy=self.root/'many.png';im=Image.new('RGB',(9,9))
        im.putdata([(i,i^1,i^2) for i in range(81)]);im.save(noisy)
        out=self.root/'blocked.sqlite'
        with self.assertRaises(f.Deny):f.index_image(noisy,out)
        self.assertFalse(out.exists())

    def test_flow_program_decode(self):
        path=self.root/'program.png'
        im=Image.new('RGB',(4,2),(128,128,0))
        im.putpixel((0,0),(255,128,255));im.putpixel((1,0),(255,128,2));im.save(path)
        dbp=self.root/'flow.sqlite';f.index_image(path,dbp,mode='flow')
        with f.open_index(dbp) as (db,meta):
            self.assertIn('SPAWN MARKER',f.point(db,meta,0,0))
            self.assertIn('INC',f.point(db,meta,1,0))
            self.assertIn('NOT image gradients',f.summary(db,meta))

    def test_no_overwrite_without_force(self):
        with self.assertRaises(f.Deny):f.index_image(self.png,self.db)

    def test_corrupted_index_detected(self):
        with closing(sqlite3.connect(self.db)) as conn:
            conn.execute('UPDATE runs SET cid=999 WHERE y=0 AND x0=0')
            conn.commit()
        with f.open_index(self.db) as (db,meta):
            with self.assertRaises(f.Deny):f.verify(db,meta)


if __name__=='__main__':unittest.main()
