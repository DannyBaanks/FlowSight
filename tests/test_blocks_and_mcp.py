"""Tests for FlowSight M2: 1920x1080 block partitioning, stitching, and real MCP stdio server."""
from __future__ import annotations

import asyncio
from pathlib import Path
import sys
import tempfile
import unittest

from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import flowsight_blocks as fb

REPO_ROOT = Path(__file__).resolve().parents[1]
VORTEX_PNG = Path('/home/danny/Development/ISyCo Git/FLOW/examples/vortex.png')


def _make_1920x1080_scene(path: Path) -> Path:
    """Deterministic 1920x1080 multi-color photo-like test scene with known cross-seam ground truth.

    Ground truth:
      - Top sky gradient (blue) across y=[0, 420), spanning TL and TR.
      - Bottom ground band (green) across y=[700, 1080), spanning BL and BR.
      - Warm orange/yellow sun disk in top-left quadrant TL around (320, 220), radius 110.
      - Large crimson/red central monolith crossing ALL 4 quadrants: [760, 340, 1160, 740).
      - High-contrast white horizontal bar inside BR: [1280, 820, 1780, 880).
    """
    im = Image.new('RGB', (1920, 1080), (42, 62, 88))
    d = ImageDraw.Draw(im)
    # Sky gradient in strips (>64 distinct colors!)
    for y in range(0, 420, 4):
        t = y / 420.0
        col = (int(45 + 75 * t), int(110 + 75 * t), int(215 + 25 * t))
        d.rectangle((0, y, 1919, y + 3), fill=col)
    # Mid haze gradient y=420..700
    for y in range(420, 700, 4):
        t = (y - 420) / 280.0
        col = (int(120 - 40 * t), int(165 - 45 * t), int(190 - 70 * t))
        d.rectangle((0, y, 1919, y + 3), fill=col)
    # Ground band y=700..1080
    for y in range(700, 1080, 4):
        t = (y - 700) / 380.0
        col = (int(48 + 20 * t), int(135 - 30 * t), int(58 + 10 * t))
        d.rectangle((0, y, 1919, y + 3), fill=col)
    # Sun in TL
    d.ellipse((210, 110, 430, 330), fill=(250, 205, 55))
    # Central crimson monolith crossing all 4 quadrants around (960, 540)
    d.rectangle((760, 340, 1159, 739), fill=(195, 38, 46))
    # High-contrast white horizontal bar in BR
    d.rectangle((1280, 820, 1779, 879), fill=(244, 246, 250))
    im.save(path, format='PNG')
    return path


class BlocksPartitionAndStitchTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.scene = _make_1920x1080_scene(self.root / 'scene_1080p.png')
        self.sdir = self.root / 'sessions'

    def tearDown(self):
        self.tmp.cleanup()

    def test_1920x1080_quarters_and_halves_and_subdivide(self):
        sess = fb.map_image_blocks(self.scene, split='quarters', palette_k=10, session_dir=self.sdir)
        self.assertEqual(sess['full_size'], [1920, 1080])
        self.assertEqual(sess['block_order'], ['TL', 'TR', 'BL', 'BR'])
        self.assertEqual(sess['blocks']['TL']['bbox'], [0, 0, 960, 540])
        self.assertEqual(sess['blocks']['TR']['bbox'], [960, 0, 1920, 540])
        self.assertEqual(sess['blocks']['BL']['bbox'], [0, 540, 960, 1080])
        self.assertEqual(sess['blocks']['BR']['bbox'], [960, 540, 1920, 1080])
        self.assertGreater(sess['quantization_error']['psnr_db'], 20.0)

        # Halves horizontal and vertical
        sh = fb.map_image_blocks(self.scene, split='halves_h', palette_k=8, session_dir=self.sdir)
        self.assertEqual(sh['blocks']['L']['bbox'], [0, 0, 960, 1080])
        self.assertEqual(sh['blocks']['R']['bbox'], [960, 0, 1920, 1080])

        sv = fb.map_image_blocks(self.scene, split='halves_v', palette_k=8, session_dir=self.sdir)
        self.assertEqual(sv['blocks']['T']['bbox'], [0, 0, 1920, 540])
        self.assertEqual(sv['blocks']['B']['bbox'], [0, 540, 1920, 1080])

        # Recursive subdivision of TL -> 4 blocks of 480x270 with global coordinates
        child = fb.subdivide_block(sess['session_id'], 'TL', split='quarters', session_dir=self.sdir)
        self.assertEqual(child['root_box'], [0, 0, 960, 540])
        self.assertEqual(child['blocks']['BR']['bbox'], [480, 270, 960, 540])

    def test_stitch_detects_cross_seam_regions_and_aligns_notes(self):
        sess = fb.map_image_blocks(self.scene, split='quarters', palette_k=10, session_dir=self.sdir)
        notes = {
            'TL': 'Blue upper band, bright yellow disk at top-left, red corner at bottom-right.',
            'TR': 'Blue upper band continuing from left, red corner at bottom-left.',
            'BL': 'Green lower band, red corner at top-right.',
            'BR': 'Green lower band, red corner at top-left, bright white horizontal bar.',
        }
        stitched = fb.stitch_blocks(sess['session_id'], block_notes=notes, session_dir=self.sdir)
        report = fb.format_stitch_report(stitched)

        # Real seam continuity across all 4 seams should be high (>80%)
        for sm in stitched['seams']:
            self.assertGreater(sm['continuity_pct'], 75.0, f'low seam continuity: {sm}')

        # Must merge cross-seam regions spanning multiple quadrants (including central red block)
        cross = [r for r in stitched['unified_regions'] if r['crosses_seam']]
        self.assertGreaterEqual(len(cross), 3)
        red_cross = [r for r in cross if 'red' in r['family']]
        self.assertTrue(red_cross, f'central red cross-seam region not merged: {cross}')
        self.assertEqual(set(red_cross[0]['spans']), {'TL', 'TR', 'BL', 'BR'})

        # Report keeps DEMONSTRATED vs AGENT_NOTES / INFERRED clearly separated
        self.assertIn('SEAM CONTINUITY (DEMONSTRATED', report)
        self.assertIn('UNIFIED GLOBAL REGIONS (DEMONSTRATED', report)
        self.assertIn('ALIGNED BLOCK OBSERVATIONS (AGENT_NOTES / INFERRED', report)
        self.assertIn('bright white horizontal bar', report)

    def test_blocks_determinism(self):
        s1 = fb.map_image_blocks(self.scene, split='quarters', palette_k=10, session_dir=self.sdir)
        s2 = fb.map_image_blocks(self.scene, split='quarters', palette_k=10, session_dir=self.sdir)
        self.assertEqual(s1['session_id'], s2['session_id'])
        self.assertEqual(fb.format_map_summary(s1), fb.format_map_summary(s2))
        self.assertEqual(
            fb.format_stitch_report(fb.stitch_blocks(s1['session_id'], session_dir=self.sdir)),
            fb.format_stitch_report(fb.stitch_blocks(s2['session_id'], session_dir=self.sdir)),
        )


class MCPServerStdioE2ETests(unittest.TestCase):
    def test_mcp_stdio_roundtrip(self):
        async def _run():
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client

            with tempfile.TemporaryDirectory() as tmp:
                scene = _make_1920x1080_scene(Path(tmp) / 'e2e_1080p.png')
                params = StdioServerParameters(
                    command=sys.executable,
                    args=[str(REPO_ROOT / 'flowsight_mcp.py')],
                    cwd=str(REPO_ROOT),
                )
                async with stdio_client(params) as (read_stream, write_stream):
                    async with ClientSession(read_stream, write_stream) as session:
                        await session.initialize()
                        tools = await session.list_tools()
                        tool_names = {t.name for t in tools.tools}
                        self.assertTrue({
                            'flowsight_map_blocks',
                            'flowsight_get_block',
                            'flowsight_subdivide_block',
                            'flowsight_stitch_blocks',
                            'flowsight_inspect_lossless',
                        }.issubset(tool_names))

                        # 1. Map 1920x1080 into quarters
                        r_map = await session.call_tool('flowsight_map_blocks', {
                            'image_path': str(scene),
                            'split': 'quarters',
                            'palette_k': 10,
                        })
                        txt_map = r_map.content[0].text
                        self.assertIn('FLOWSIGHT BLOCKS v1', txt_map)
                        self.assertIn('(1920x1080)', txt_map)
                        sid = txt_map.split('session=')[1].split(' ')[0]

                        # 2. Get one block
                        r_blk = await session.call_tool('flowsight_get_block', {
                            'session_id': sid,
                            'block_id': 'TL',
                        })
                        self.assertIn('BLOCK TL', r_blk.content[0].text)
                        self.assertIn('EDGE MAP', r_blk.content[0].text)

                        # 3. Subdivide TL
                        r_sub = await session.call_tool('flowsight_subdivide_block', {
                            'session_id': sid,
                            'block_id': 'TL',
                            'split': 'quarters',
                        })
                        self.assertIn('global bbox=[0, 0, 960, 540]', r_sub.content[0].text)

                        # 4. Stitch with notes
                        r_st = await session.call_tool('flowsight_stitch_blocks', {
                            'session_id': sid,
                            'block_notes': {'TL': 'Yellow disk top-left + red bottom-right'},
                        })
                        self.assertIn('FLOWSIGHT STITCH v1', r_st.content[0].text)
                        self.assertIn('CROSS-SEAM', r_st.content[0].text)

                        # 5. Lossless M1 tool over real FLOW vortex.png if available
                        if VORTEX_PNG.exists():
                            r_fl = await session.call_tool('flowsight_inspect_lossless', {
                                'image_path': str(VORTEX_PNG),
                                'mode': 'flow',
                                'query': 'summary',
                            })
                            self.assertIn('native 3-channel lossless', r_fl.content[0].text)
                            self.assertIn('Spawns (B=255): 8 total', r_fl.content[0].text)

        asyncio.run(_run())


if __name__ == '__main__':
    unittest.main()
