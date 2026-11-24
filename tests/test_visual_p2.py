"""Tests for FlowSight M1 Priority 2: visual fovea upgrades.

Covers:
  - region --grid (exact 1:1 symbolic grid + fail-closed limit >1024px)
  - minority/hidden-in-minimap color markers with exact bboxes in summary()
  - edges --region (scoped internal boundaries without global scan truncation)
  - color transitions in diff() (old -> new palette/hex + per-run labels)
"""
from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import flowsight as fs


class VisualP2Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.png = self.root / 'scene.png'
        im = Image.new('RGBA', (64, 64), (10, 12, 18, 255))
        d = ImageDraw.Draw(im)
        d.rectangle((4, 8, 24, 28), fill=(46, 204, 113, 255))
        d.rectangle((30, 8, 50, 28), fill=(231, 76, 60, 255))
        # Single minority pixel hidden from 24x16 majority minimap
        im.putpixel((55, 52), (241, 196, 15, 255))
        im.save(self.png)
        self.db = self.root / 'scene.sqlite'
        fs.index_image(self.png, self.db, mode='visual')

    def tearDown(self):
        self.tmp.cleanup()

    def test_region_grid_exact_and_fovea_limit(self):
        with fs.open_index(self.db) as (db, meta):
            out = fs.region(db, meta, 53, 50, 5, 5, grid=True)
            self.assertIn('GRID 1:1 exact', out)
            self.assertIn('y= 52: 00300', out)  # pixel (55,52) is C3 ('3') surrounded by C0 ('0')
            # Exceeding 1024 px fovea limit fails closed
            with self.assertRaises(fs.Deny):
                fs.region(db, meta, 0, 0, 40, 40, grid=True)

    def test_minority_color_marker_in_summary(self):
        with fs.open_index(self.db) as (db, meta):
            l0 = fs.summary(db, meta)
            self.assertIn('[hidden-in-minimap]', l0)
            self.assertIn('Minority colors absent from minimap', l0)
            self.assertIn('#f1c40fff pixels=1 bbox=[55,52,56,53)', l0)

    def test_edges_scoped_by_region(self):
        with fs.open_index(self.db) as (db, meta):
            scoped = fs.edges(db, meta, max_lines=40, box=(53, 50, 5, 5))
            self.assertIn('scope=[53,50,58,55)', scoped)
            self.assertIn('exact segments=4', scoped)
            self.assertIn('V x=55 y=[52,53) C0->C3', scoped)
            self.assertIn('V x=56 y=[52,53) C3->C0', scoped)
            self.assertIn('H y=52 x=[55,56) C0->C3', scoped)
            self.assertIn('H y=53 x=[55,56) C3->C0', scoped)
            with self.assertRaises(fs.Deny):
                fs.edges(db, meta, box=(60, 60, 10, 10))

    def test_diff_reports_exact_color_transitions(self):
        other = self.root / 'edited.png'
        im = Image.open(self.png).convert('RGBA')
        # Change 2 green pixels to magenta (new color) and 1 red pixel to existing yellow C3
        im.putpixel((10, 12), (255, 0, 255, 255))
        im.putpixel((11, 12), (255, 0, 255, 255))
        im.putpixel((35, 12), (241, 196, 15, 255))
        im.save(other)

        with fs.open_index(self.db) as (db, meta):
            out = fs.diff(db, meta, str(other))
            self.assertIn('COLOR TRANSITIONS:', out)
            self.assertIn('C1(#2ecc71ff)->#ff00ffff:2px', out)
            self.assertIn('C2(#e74c3cff)->C3(#f1c40fff):1px', out)
            self.assertIn('y=12 x=[10,12) C1(#2ecc71ff)->#ff00ffff', out)
            self.assertIn('y=12 x=[35,36) C2(#e74c3cff)->C3(#f1c40fff)', out)


if __name__ == '__main__':
    unittest.main()
