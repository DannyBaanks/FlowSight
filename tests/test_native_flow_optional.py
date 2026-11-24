"""Native FLOW paint integration: runs when the real upstream FLOW is installed.

This deliberately does not vendor/replace the upstream parser. On a machine
without FLOW, unittest reports SKIP, NOT PASS for this integration gate.
"""
from pathlib import Path
import sys
import tempfile
import unittest
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import flowsight as fs

try:
    from flow.paint.painter import run_paint_program
except ImportError:
    run_paint_program=None

@unittest.skipUnless(run_paint_program is not None,'FLOW painter not installed here; native gate NOT_DEMONSTRATED')
class NativeFlowIntegration(unittest.TestCase):
    def test_generated_flowpaint_exact_rgba(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'img.png'
            im=Image.new('RGBA',(16,12),(2,5,8,255))
            for y in range(3,7):
                for x in range(4,10): im.putpixel((x,y),(40,200,30,255))
            im.save(p)
            dbp=Path(tmp)/'i.sqlite';fs.index_image(p,dbp)
            with fs.open_index(dbp) as (db,meta):
                paint,_=fs.flowpaint_source(db,meta)
            result=run_paint_program(paint)
            self.assertTrue(result.ok, result.error)
            from io import BytesIO
            self.assertEqual(Image.open(BytesIO(result.png_bytes)).convert('RGBA').tobytes(),im.tobytes())

if __name__=='__main__':unittest.main()
