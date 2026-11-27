"""Tests for FlowSight Screen real layout, controls, and report formatting fixes."""
from __future__ import annotations

from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import flowsight_screen as fscr


def _draw_fake_text(d: ImageDraw.ImageDraw, x0: int, y0: int, n_chars: int, fg=(220, 228, 238), h=14) -> None:
    x = x0
    for i in range(n_chars):
        if i % 6 == 5:
            x += 8
            continue
        d.rectangle((x, y0, x + 1, y0 + h), fill=fg)
        d.rectangle((x + 4, y0, x + 5, y0 + h), fill=fg)
        x += 9


class ScreenRealLayoutAndControlsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_divider_ranking_keeps_strong_bottom_divider_over_weak_top_lines(self):
        """1. _f_layout ranks dividers by coverage strength before keeping top 8, then sorts by position."""
        im = Image.new('RGB', (1000, 1000), (20, 22, 30))
        d = ImageDraw.Draw(im)
        # Draw 10 medium-coverage horizontal lines in the top half (coverage ~0.50: x=200..700)
        for i in range(10):
            y = 40 + i * 35
            d.line((200, y, 700, y), fill=(90, 95, 110), width=2)
        # Draw 1 full-width strong bottom status bar divider at y=920 (coverage 1.00)
        d.line((0, 920, 999, 920), fill=(160, 170, 190), width=2)
        p = self.root / 'ranked_divs.png'
        im.save(p, format='PNG')

        ctx = fscr.run_pipeline(p, vf='layout')
        h_divs = ctx.artifacts['layout']['h_dividers_global_y']
        self.assertEqual(len(h_divs), 8)
        self.assertEqual(h_divs, sorted(h_divs), "h_dividers must be sorted by position")
        self.assertTrue(
            any(abs(y - 920) <= 3 for y in h_divs),
            f"Strong bottom divider at y=920 must not be dropped by [:8] top-truncation; got {h_divs}",
        )

    def test_xy_cut_does_not_slice_sidebar_with_main_pane_divider(self):
        """2. _f_layout uses recursive XY-cut so a horizontal divider inside main pane never crosses the sidebar."""
        im = Image.new('RGB', (1200, 800), (18, 22, 30))
        d = ImageDraw.Draw(im)
        # Full-height left sidebar: x=0..240
        d.rectangle((0, 0, 239, 799), fill=(12, 16, 22))
        d.line((240, 0, 240, 799), fill=(110, 120, 140), width=2)
        # Strong horizontal divider ONLY inside the main pane (x=242..1199, y=400)
        d.line((242, 400, 1199, 400), fill=(130, 140, 165), width=2)
        p = self.root / 'sidebar_xycut.png'
        im.save(p, format='PNG')

        ctx = fscr.run_pipeline(p, vf='layout')
        panels = ctx.artifacts['layout']['panels']
        # No panel should cross the vertical sidebar boundary x=240
        for pan in panels:
            x0, y0, x1, y1 = pan['global_bbox']
            crosses_sidebar = (x0 < 200 and x1 > 280)
            self.assertFalse(
                crosses_sidebar,
                f"Panel {pan['id']} {pan['global_bbox']} illegally crossed sidebar divider x=240",
            )
        # And the left sidebar should remain a single un-sliced full-height panel
        sidebar_panels = [pan for pan in panels if pan['global_bbox'][2] <= 245]
        self.assertEqual(len(sidebar_panels), 1, f"Sidebar should not be sliced by main-pane H-divider: {sidebar_panels}")
        self.assertEqual(sidebar_panels[0]['position_hint'], 'left-sidebar-band')

    def test_wide_thin_bordered_input_detected_and_not_treated_as_layout_divider(self):
        """3 & 4. A 95%-wide 1px thin-bordered input box is detected as input-or-bar-box and not a layout divider."""
        im = Image.new('RGB', (1200, 800), (10, 12, 16))
        d = ImageDraw.Draw(im)
        # 95% wide thin 1px low-contrast bordered input at [30, 700, 1170, 742] (width=1140 = 95% of 1200)
        d.rectangle((30, 700, 1170, 742), fill=(14, 16, 22), outline=(38, 44, 56), width=1)
        _draw_fake_text(d, 48, 714, 24, fg=(150, 160, 175), h=14)
        p = self.root / 'wide_input.png'
        im.save(p, format='PNG')

        ctx = fscr.run_pipeline(p, vf='layout,controls')
        h_divs = ctx.artifacts['layout']['h_dividers_global_y']
        self.assertFalse(
            any(695 <= y <= 747 for y in h_divs),
            f"Thin input box borders [700, 742] must not become full-width layout dividers: {h_divs}",
        )

        ctls = ctx.artifacts['controls']['controls']
        matched = [
            c for c in ctls
            if c['kind'] == 'input-or-bar-box'
            and abs(c['global_bbox'][0] - 30) <= 8
            and abs(c['global_bbox'][1] - 700) <= 8
            and abs(c['global_bbox'][2] - 1170) <= 8
            and abs(c['global_bbox'][3] - 742) <= 8
        ]
        self.assertTrue(matched, f"Expected 95%-wide 1px input box to be detected, got: {ctls}")

    def test_noisy_colorful_banner_does_not_starve_real_inputs_and_reports_total_detected(self):
        """5 & 6. Saturated noise inside a colorful pixel-art banner does not starve bordered inputs; total_detected is real."""
        rng = np.random.default_rng(42)
        im = Image.new('RGB', (1200, 800), (12, 14, 18))
        d = ImageDraw.Draw(im)

        # Top colorful pixel-art banner [40, 40, 1160, 240] with 35 small saturated blobs
        d.rectangle((40, 40, 1160, 240), fill=(25, 30, 35))
        palette = [
            (240, 60, 60), (60, 220, 90), (60, 140, 250), (245, 190, 40),
            (210, 70, 230), (50, 220, 220), (250, 120, 40), (140, 240, 60),
        ]
        for idx in range(35):
            bx = 55 + (idx % 12) * 90
            by = 55 + (idx // 12) * 60
            col = palette[idx % len(palette)]
            d.rectangle((bx, by, bx + 42, by + 24), fill=col)

        # Real wide bordered input at bottom [40, 710, 1160, 754]
        d.rectangle((40, 710, 1160, 754), fill=(16, 18, 24), outline=(70, 80, 100), width=1)
        _draw_fake_text(d, 60, 725, 22, fg=(190, 200, 215), h=14)
        p = self.root / 'banner_and_input.png'
        im.save(p, format='PNG')

        # Limit max controls to 10 so truncation definitely happens
        ctx = fscr.run_pipeline(p, vf='layout,controls=max=10')
        ct_art = ctx.artifacts['controls']
        self.assertGreater(ct_art['total_detected'], 10, "total_detected must reflect pre-truncation count")
        self.assertEqual(len(ct_art['controls']), 10)

        # Even with max=10 and 35 saturated blobs in the banner, the real input box must be in the top 10!
        has_input = any(
            c['kind'] == 'input-or-bar-box' and abs(c['global_bbox'][1] - 710) <= 8
            for c in ct_art['controls']
        )
        self.assertTrue(has_input, f"Real bordered input was starved by noisy banner blobs: {ct_art['controls']}")

        rep = fscr.format_pipeline_report(ctx)
        self.assertIn('[TRUNCATED', rep)

    def test_report_priority_ocr_limit_and_line_boundary_truncation(self):
        """7. format_pipeline_report orders OCR & ui_map before text_bands, honors ocr_limit, and trims on line boundaries."""
        im = Image.new('RGB', (800, 600), (20, 24, 32))
        d = ImageDraw.Draw(im)
        for i in range(6):
            _draw_fake_text(d, 40, 40 + i * 40, 20, fg=(220, 230, 240), h=14)
        p = self.root / 'rep_test.png'
        im.save(p, format='PNG')

        def fake_ocr(rgb, offset_x=0, offset_y=0, source_frame='', lang='eng'):
            items = [
                {
                    'id': f'O{i}',
                    'text': f'Synthetic OCR Line {i}',
                    'bbox': [40, 20 + i * 15, 300, 34 + i * 15],
                    'confidence': 95.0,
                    'status': 'OCR_OBSERVED',
                }
                for i in range(35)
            ]
            return {
                'status': 'OCR_OBSERVED',
                'engine': 'synthetic',
                'lang': lang,
                'confidence_scale': 'synthetic',
                'items': items,
                'elapsed_ms': 1.0,
            }

        ctx = fscr.run_pipeline(p, vf='theme,layout,controls,text_bands,ocr,ui_map=32:12', ocr_provider=fake_ocr)
        rep = fscr.format_pipeline_report(ctx, max_chars=6500, ocr_limit=10)

        # Section order: OCR and UI WIREFRAME MAP must appear BEFORE TEXT BANDS
        idx_ocr = rep.index('OCR:')
        idx_map = rep.index('UI WIREFRAME MAP')
        idx_tb = rep.index('TEXT BANDS')
        self.assertLess(idx_ocr, idx_map)
        self.assertLess(idx_map, idx_tb)

        # Explicit OCR truncation marker when 35 items > ocr_limit=10
        self.assertIn('[TRUNCATED 25 of 35 additional OCR items', rep)

        # Line-boundary trimming under tight max_chars budget
        tight = fscr.format_pipeline_report(ctx, max_chars=650, ocr_limit=25)
        self.assertLessEqual(len(tight), 650)
        lines = tight.splitlines()
        self.assertEqual(lines[-1], '[TRUNCATED: narrow scope with crop=x:y:w:h or select fewer filters in -vf]')
        # Every line before the truncation footer must be a complete line from the full report
        full_lines = set(fscr.format_pipeline_report(ctx, max_chars=50000, ocr_limit=25).splitlines())
        for ln in lines[:-1]:
            self.assertIn(ln, full_lines, f"Line was sliced mid-line: {ln!r}")


if __name__ == '__main__':
    unittest.main()
