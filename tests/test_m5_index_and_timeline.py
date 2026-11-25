"""FlowSight M5 tests: Progressive Visual Index (SQLite+FTS5+TTL), Real MP4 Timeline, and MCP 14-tools E2E."""
from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
import os
from pathlib import Path
import sqlite3
import stat
import sys
import tempfile
import unittest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'examples'))

import benchmark_m5
import flowsight_index as fidx
import flowsight_screen as fscr
import flowsight_timeline as ftl


class M5ProgressiveIndexTests(unittest.TestCase):
    def test_seven_screenshots_progressive_search_savings_dedup_and_lifecycle(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            shots = benchmark_m5.build_seven_screenshots(Path(td) / 'shots')
            db_path = Path(td) / 'priv' / 'idx.sqlite'

            # 1. Cold index + private permissions + NO pixel blobs in DB
            r1 = fidx.index_paths(shots, use_ocr=True, ttl_hours=12.0, db_path=db_path)
            self.assertEqual(r1['indexed_new'], 7)
            self.assertEqual(r1['reused_cached'], 0)
            self.assertEqual(r1['errors'], [])
            if os.name == 'posix':
                self.assertEqual(stat.S_IMODE(db_path.parent.stat().st_mode), 0o700)
                self.assertEqual(stat.S_IMODE(db_path.stat().st_mode), 0o600)

            # Verify zero PNG bytes stored in SQLite
            raw_db = db_path.read_bytes()
            self.assertNotIn(b'\x89PNG\r\n\x1a\n', raw_db)

            # 2. Second pass hits SHA-256 dedup cache (0 new OCR runs!)
            r2 = fidx.index_paths(shots, use_ocr=True, ttl_hours=12.0, db_path=db_path)
            self.assertEqual(r2['indexed_new'], 0)
            self.assertEqual(r2['reused_cached'], 7)

            # 3. Progressive search for "UFW 3128" ranks shot_03_ufw_terminal.png #1 with exact bbox
            s = fidx.search_index('UFW 3128', theme='dark', limit=3, db_path=db_path)
            self.assertGreaterEqual(len(s['candidates']), 1)
            top = s['candidates'][0]
            self.assertEqual(top['source_name'], 'shot_03_ufw_terminal.png')
            self.assertIn('3128', top['matched_terms'])
            self.assertIn('ufw', top['matched_terms'])
            self.assertTrue(top['suggested_crop_vf'].startswith('crop='))

            # 4. Deepen ONLY the winning crop and verify huge char savings vs 7 full reports
            search_txt = fidx.format_search_report(s)
            ctx_zoom = fscr.run_pipeline(top['path'], vf=f'{top["suggested_crop_vf"]},controls,ocr')
            zoom_txt = fscr.format_pipeline_report(ctx_zoom)
            self.assertIn('3128/tcp', zoom_txt)

            full_chars = sum(
                len(fscr.format_pipeline_report(fscr.run_pipeline(p, vf='theme,layout,controls,text_bands,ui_map=44:16')))
                for p in shots
            )
            self.assertLess(len(search_txt) + len(zoom_txt), full_chars * 0.60)

            # 5. Color filter narrows down to orange button screenshot
            s_orange = fidx.search_index('', color='orange', limit=5, db_path=db_path)
            self.assertEqual([c['source_name'] for c in s_orange['candidates']], ['shot_07_chat_summary.png'])

            # 6. TTL expiration & explicit purge
            future = datetime.now(timezone.utc) + timedelta(hours=24)
            removed_exp = fidx.expire_stale(db_path, now=future)
            self.assertEqual(removed_exp, 7)
            self.assertEqual(fidx.index_stats(db_path)['frames'], 0)

            # Re-add 2 and purge all
            fidx.index_paths(shots[:2], use_ocr=False, db_path=db_path)
            p_res = fidx.purge_index(db_path, mode='all')
            self.assertEqual(p_res['removed'], 2)
            self.assertEqual(p_res['remaining'], 0)


class M5VideoTimelineTests(unittest.TestCase):
    def test_real_mp4_video_skips_static_frames_and_detects_temporal_events(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            vid = benchmark_m5.build_synthetic_timeline_video(Path(td) / 'ui_flow.mp4', Path(td))
            tl = ftl.scan_video(vid, sample_fps=2.0, max_frames=30, scene_thresh=6.0, use_ocr=True)

            # 4 seconds at 2 fps = 8 sampled frames across 4 distinct 1-second phases -> 4 skipped static!
            self.assertEqual(tl['total_sampled_frames'], 8)
            self.assertEqual(tl['keyframes_analyzed'], 4)
            self.assertEqual(tl['frames_skipped_static'], 4)

            events = tl['events']
            # Event 1 (~1.0s): Merge button green -> orange with STATE_CHANGED_CANDIDATE INFERRED
            color_evs = [e for e in events if e['event_type'] == 'CONTROL_COLOR_CHANGED']
            self.assertTrue(color_evs, f'expected CONTROL_COLOR_CHANGED in {events}')
            self.assertIn('STATE_CHANGED_CANDIDATE', color_evs[0].get('inferred_flags', []))
            self.assertAlmostEqual(color_evs[0]['t_sec'], 1.0, delta=0.6)

            # Event 2 (~2.0s): Dismiss modal button appears
            dismiss_evs = ftl.filter_timeline_events(tl, 'Dismiss')
            self.assertTrue(dismiss_evs, f'expected Dismiss appearance in {events}')
            self.assertAlmostEqual(dismiss_evs[0]['t_sec'], 2.0, delta=0.6)

            # Event 3 (~3.0s): ERROR: exit=56 appears (as red banner control + new OCR text)
            err_evs = ftl.filter_timeline_events(tl, 'exit=56')
            self.assertTrue(err_evs, f'expected exit=56 appearance in {events}')
            self.assertIn('TEXT_APPEARED', {e['event_type'] for e in err_evs})
            self.assertIn('CONTROL_APPEARED', {e['event_type'] for e in err_evs})
            self.assertAlmostEqual(err_evs[0]['t_sec'], 3.0, delta=0.6)

            rep = ftl.format_timeline_report(tl, query='exit=56')
            self.assertIn('TEXT_APPEARED', rep)
            self.assertIn('exit=56', rep)


class M5MCPStdioTests(unittest.TestCase):
    def test_mcp_stdio_exposes_14_tools_index_and_timeline(self) -> None:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        async def _run() -> None:
            with tempfile.TemporaryDirectory() as td:
                shots = benchmark_m5.build_seven_screenshots(Path(td) / 'shots')
                vid = benchmark_m5.build_synthetic_timeline_video(Path(td) / 'clip.mp4', Path(td))
                db_path = Path(td) / 'mcp_idx.sqlite'

                env = dict(os.environ)
                env['FLOWSIGHT_INDEX_DB'] = str(db_path)
                params = StdioServerParameters(
                    command=sys.executable,
                    args=[str(REPO / 'flowsight_mcp.py')],
                    cwd=str(REPO),
                    env=env,
                )
                async with stdio_client(params) as (read, write):
                    async with ClientSession(read, write) as session:
                        await session.initialize()
                        tools = await session.list_tools()
                        names = {t.name for t in tools.tools}
                        self.assertEqual(len(names), 14)
                        for expected in (
                            'flowsight_index_add',
                            'flowsight_index_search',
                            'flowsight_index_purge',
                            'flowsight_timeline_scan',
                        ):
                            self.assertIn(expected, names)

                        res_add = await session.call_tool(
                            'flowsight_index_add',
                            {'paths': [str(shots[2]), str(shots[3])], 'use_ocr': True, 'ttl_hours': 4.0},
                        )
                        self.assertIn('INDEXED new=2', res_add.content[0].text)

                        res_s = await session.call_tool(
                            'flowsight_index_search',
                            {'query': '3128', 'theme': 'dark', 'limit': 2},
                        )
                        self.assertIn('shot_03_ufw_terminal.png', res_s.content[0].text)

                        res_tl = await session.call_tool(
                            'flowsight_timeline_scan',
                            {'inputs': [str(vid)], 'sample_fps': 2.0, 'scene_thresh': 2.0, 'query': 'exit=56'},
                        )
                        self.assertIn('exit=56', res_tl.content[0].text)

                        res_p = await session.call_tool('flowsight_index_purge', {'mode': 'all'})
                        self.assertIn('removed=2 remaining=0', res_p.content[0].text)

        asyncio.run(_run())


if __name__ == '__main__':
    unittest.main()
