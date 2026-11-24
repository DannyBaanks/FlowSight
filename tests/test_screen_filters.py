"""Tests for FlowSight M3: ffmpeg-style screenshot filter pipeline + MCP screen tools."""
from __future__ import annotations

import asyncio
from pathlib import Path
import sys
import tempfile
import unittest

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import flowsight_screen as fscr

REPO_ROOT = Path(__file__).resolve().parents[1]


def _draw_fake_text_line(d: ImageDraw.ImageDraw, x0: int, y0: int, n_chars: int, fg=(225, 232, 240), h=14):
    """Draw deterministic high-contrast vertical glyph strokes mimicking rendered UI text."""
    x = x0
    for i in range(n_chars):
        if i % 6 == 5:
            x += 8  # space between words
            continue
        # Simple 2-stroke glyph (6px wide)
        d.rectangle((x, y0, x + 1, y0 + h), fill=fg)
        d.rectangle((x + 4, y0 + (2 if i % 2 else 0), x + 5, y0 + h), fill=fg)
        if i % 3 == 0:
            d.rectangle((x, y0 + h // 2, x + 5, y0 + h // 2 + 1), fill=fg)
        x += 9


def make_1080p_desktop_ui(path: Path, after_path: Path | None = None) -> Path:
    """Realistic 1920x1080 dark desktop IDE/app screenshot with exact ground truth."""
    im = Image.new('RGB', (1920, 1080), (18, 20, 28))
    d = ImageDraw.Draw(im)

    # Top bar: y=0..56
    d.rectangle((0, 0, 1919, 55), fill=(28, 32, 44))
    d.line((0, 56, 1919, 56), fill=(68, 76, 98), width=2)
    # Search input inside topbar: [520, 12, 1280, 44)
    d.rectangle((520, 12, 1279, 43), fill=(16, 18, 26), outline=(92, 104, 134), width=2)
    _draw_fake_text_line(d, 536, 21, 18, fg=(180, 190, 210), h=12)
    # Primary blue button in topbar: [1740, 10, 1890, 46) -> center (1815, 28)
    d.rectangle((1740, 10, 1889, 45), fill=(37, 99, 235))
    _draw_fake_text_line(d, 1765, 21, 10, fg=(248, 250, 252), h=12)

    # Left sidebar: x=0..280, y=58..1038
    d.rectangle((0, 58, 279, 1038), fill=(24, 27, 37))
    d.line((280, 56, 280, 1040), fill=(68, 76, 98), width=2)
    for row_i in range(8):
        _draw_fake_text_line(d, 24, 92 + row_i * 42, 16 + (row_i % 4) * 3, fg=(195, 204, 220), h=13)

    # Main pane: x=282..1920, y=58..1038 (dark bg + code lines + modal card + 2 action buttons)
    for line_i in range(10):
        _draw_fake_text_line(d, 330, 100 + line_i * 36, 35 + (line_i % 5) * 8, fg=(210, 218, 230), h=14)

    # Action buttons in lower-right of main pane:
    # Green Run button: [1340, 860, 1520, 912) -> center (1430, 886)
    d.rectangle((1340, 860, 1519, 911), fill=(22, 163, 74))
    _draw_fake_text_line(d, 1378, 878, 10, fg=(250, 252, 250), h=14)
    # Red Cancel button: [1550, 860, 1730, 912) -> center (1640, 886)
    d.rectangle((1550, 860, 1729, 911), fill=(220, 38, 38))
    _draw_fake_text_line(d, 1584, 878, 11, fg=(252, 248, 248), h=14)

    # Bottom status bar: y=1040..1080
    d.line((0, 1040, 1919, 1040), fill=(68, 76, 98), width=2)
    d.rectangle((0, 1042, 1919, 1079), fill=(30, 36, 52))
    _draw_fake_text_line(d, 20, 1054, 22, fg=(190, 200, 220), h=12)

    im.save(path, format='PNG')

    if after_path is not None:
        ed = im.copy()
        de = ImageDraw.Draw(ed)
        # Green button turns amber/orange (running state) + central modal alert appears
        de.rectangle((1340, 860, 1519, 911), fill=(234, 148, 24))
        _draw_fake_text_line(de, 1378, 878, 10, fg=(20, 20, 20), h=14)
        de.rectangle((700, 420, 1220, 620), fill=(46, 26, 34), outline=(239, 68, 68), width=3)
        _draw_fake_text_line(de, 740, 470, 28, fg=(250, 220, 220), h=16)
        ed.save(after_path, format='PNG')
    return path


class ScreenFilterEngineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.before = self.root / 'ui_before.png'
        self.after = self.root / 'ui_after.png'
        make_1080p_desktop_ui(self.before, self.after)

    def tearDown(self):
        self.tmp.cleanup()

    def test_parse_vf_and_fail_closed(self):
        steps = fscr.parse_vf('crop=10:20:300:200,theme,layout,ui_map=40:16')
        self.assertEqual([s[0] for s in steps], ['crop', 'theme', 'layout', 'ui_map'])
        with self.assertRaises(fscr.FilterDeny):
            fscr.parse_vf('nonexistent_filter')
        with self.assertRaises(fscr.FilterDeny):
            fscr.parse_vf('')

    def test_full_1080p_ui_pipeline(self):
        ctx = fscr.run_pipeline(
            self.before,
            vf='theme,layout,controls,text_bands,ocr,ui_map=48:20',
        )
        # Theme polarity dark + accent colors present
        self.assertEqual(ctx.artifacts['theme']['polarity'], 'dark')

        # Layout detects topbar, bottom bar, sidebar divider
        lay = ctx.artifacts['layout']
        self.assertTrue(any(abs(y - 56) <= 4 for y in lay['h_dividers_global_y']), lay['h_dividers_global_y'])
        self.assertTrue(any(abs(y - 1040) <= 4 for y in lay['h_dividers_global_y']), lay['h_dividers_global_y'])
        self.assertTrue(any(abs(x - 280) <= 4 for x in lay['v_dividers_global_x']), lay['v_dividers_global_x'])
        hints = {p['position_hint'] for p in lay['panels']}
        self.assertIn('top-bar-band', hints)
        self.assertIn('bottom-bar-band', hints)
        self.assertIn('main-pane', hints)

        # Controls detect the 3 accent buttons with accurate click centers
        ctls = ctx.artifacts['controls']['controls']
        def _has_click_near(tx, ty, tol=12):
            return any(abs(c['click_center'][0] - tx) <= tol and abs(c['click_center'][1] - ty) <= tol for c in ctls)
        self.assertTrue(_has_click_near(1815, 28), ctls)   # blue topbar button
        self.assertTrue(_has_click_near(1430, 886), ctls)  # green Run button
        self.assertTrue(_has_click_near(1640, 886), ctls)  # red Cancel button

        # Text bands detected geometrically
        self.assertGreaterEqual(ctx.artifacts['text_bands']['total_detected'], 10)

        # OCR is explicitly OCR_OBSERVED/OCR_UNCERTAIN or OCR_ABSENT (never crashes or fakes labels)
        self.assertIn(ctx.artifacts['ocr']['status'], ('OCR_OBSERVED', 'OCR_UNCERTAIN', 'OCR_ABSENT'))

        # Wireframe ASCII map compact
        rep = fscr.format_pipeline_report(ctx)
        self.assertIn('FLOWSIGHT SCREEN v1', rep)
        self.assertIn('UI WIREFRAME MAP 48x20', rep)
        self.assertLessEqual(len(rep), 6500)

    def test_crop_preserves_global_click_coordinates_and_ui_diff(self):
        # Crop around the two lower-right action buttons; coordinates must remain global 1920x1080
        ctx = fscr.run_pipeline(
            self.before,
            vf='crop=1300:830:480:110,controls,text_bands',
        )
        self.assertEqual(ctx.global_bbox, [1300, 830, 1780, 940])
        ctls = ctx.artifacts['controls']['controls']
        self.assertTrue(any(abs(c['click_center'][0] - 1430) <= 10 and abs(c['click_center'][1] - 886) <= 10 for c in ctls), ctls)
        self.assertTrue(any(abs(c['click_center'][0] - 1640) <= 10 and abs(c['click_center'][1] - 886) <= 10 for c in ctls), ctls)

        # UI diff detects both the button color change and the new central modal
        dctx = fscr.run_pipeline(
            self.after,
            vf=f'layout,controls,ui_diff={self.before}',
        )
        diff = dctx.artifacts['ui_diff']
        self.assertGreaterEqual(len(diff['regions']), 2)
        self.assertTrue(any(r['global_bbox'][0] <= 710 and r['global_bbox'][2] >= 1210 for r in diff['regions']), diff['regions'])
        self.assertTrue(any(r['global_bbox'][0] <= 1350 and r['global_bbox'][2] >= 1510 for r in diff['regions']), diff['regions'])

    def test_mcp_screen_tools_e2e(self):
        async def _run():
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client

            params = StdioServerParameters(
                command=sys.executable,
                args=[str(REPO_ROOT / 'flowsight_mcp.py')],
                cwd=str(REPO_ROOT),
            )
            async with stdio_client(params) as (r, w):
                async with ClientSession(r, w) as session:
                    await session.initialize()
                    tools = {t.name for t in (await session.list_tools()).tools}
                    self.assertIn('flowsight_screen_filters', tools)
                    self.assertIn('flowsight_screen_run', tools)

                    cat = (await session.call_tool('flowsight_screen_filters', {})).content[0].text
                    self.assertIn('layout', cat)
                    self.assertIn('controls', cat)
                    self.assertIn('ui_diff', cat)

                    out = (await session.call_tool('flowsight_screen_run', {
                        'image_path': str(self.before),
                        'vf': 'theme,layout,controls,text_bands,ui_map=48:18',
                    })).content[0].text
                    self.assertIn('FLOWSIGHT SCREEN v1', out)
                    self.assertIn('UI WIREFRAME MAP 48x18', out)
                    self.assertIn('top-bar-band', out)

        asyncio.run(_run())


if __name__ == '__main__':
    unittest.main()
