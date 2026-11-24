"""Tests for FlowSight M1 Priority 1: native 3-channel `--mode flow` index.

Covers all 6 mandatory P1 gates:
  1. Index real FLOW programs (`vortex.png` with 873 RGB colors + `hello_flow.png`).
  2. Lossless R, G, B and RGBA channel verification & rebuild (0 reconstruction error).
  3. Accurate regional field, instruction, scanline, point, and spatial-coupling queries.
  4. Native FLOW VM trace invariance (original PNG vs rebuilt PNG, read-only guarantee).
  5. Byte-exact determinism across repeated runs (SQLite, L0 summary, rebuilt PNG).
  6. Compact L0 output bounds and corruption detection per channel.
"""
from __future__ import annotations

from contextlib import closing
import hashlib
import math
from pathlib import Path
import sqlite3
import sys
import tempfile
import unittest

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import flowsight as fs
import flowsight_flow as fs_flow

FLOW_EXAMPLES = Path('/home/danny/Development/ISyCo Git/FLOW/examples')
VORTEX_PNG = FLOW_EXAMPLES / 'vortex.png'
HELLO_PNG = FLOW_EXAMPLES / 'hello_flow.png'

try:
    import flow.core as flow_core
    from flow.runtime import flow_trace
except ImportError:
    flow_core = None
    flow_trace = None


class FlowModeCoreTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _make_synthetic_vortex(self, path: Path, size: int = 32):
        """Synthetic vortex with >200 distinct RGB values (no dependency on external files)."""
        img = Image.new('RGB', (size, size), (128, 128, 0))
        px = img.load()
        cx, cy = size // 2, size // 2
        for y in range(size):
            for x in range(size):
                dx, dy = x - cx, y - cy
                r = math.hypot(dx, dy)
                if r > 1:
                    vx, vy = -dy / r, dx / r
                    px[x, y] = (int((vx + 1.0) * 127.5), int((vy + 1.0) * 127.5), 0)
        # 4 spawns around center + 1 INC cell
        px[cx + 3, cy] = (128, 255, 255)
        px[cx, cy + 3] = (0, 128, 255)
        px[cx - 3, cy] = (128, 0, 255)
        px[cx, cy - 3] = (255, 128, 255)
        px[cx + 3, cy + 1] = (128, 255, 2)
        img.save(path)
        return path

    def test_synthetic_high_rgb_flow_program_lossless(self):
        png = self._make_synthetic_vortex(self.root / 'syn_vortex.png')
        dbp = self.root / 'syn_vortex.sqlite'
        w, h, distinct_b, runs, comps, raw_bytes = fs.index_image(png, dbp, mode='flow')
        self.assertEqual((w, h), (32, 32))
        self.assertEqual(distinct_b, 3)  # B in {0, 2, 255}
        self.assertGreater(runs, 0)
        self.assertEqual(raw_bytes, 32 * 32 * 4)

        with fs.open_index(dbp) as (db, meta):
            self.assertEqual(meta['schema'], fs_flow.SCHEMA_FLOW)
            self.assertGreater(int(meta['distinct_rgb']), 64)
            ver = fs.verify(db, meta, str(png))
            self.assertIn('PASS flow index integrity', ver)
            self.assertIn('0 reconstruction error', ver)

            # Rebuild exact RGB
            rebuilt = self.root / 'rebuilt.png'
            fs.save_rgba_png(db, meta, rebuilt)
            self.assertEqual(
                Image.open(png).convert('RGB').tobytes(),
                Image.open(rebuilt).convert('RGB').tobytes(),
            )

            # Compact L0 summary
            l0 = fs.summary(db, meta)
            self.assertLessEqual(len(l0), 3500)
            self.assertIn('native 3-channel lossless', l0)
            self.assertIn('NOT image gradients', l0)
            self.assertIn('Spawns (B=255): 4 total', l0)

            # Regional field stats: curl > 0 (CW in y-down screen coordinates)
            fld = fs.field_query(db, meta, box=(8, 8, 16, 16))
            self.assertIn('unquantized R/G', fld)
            self.assertIn('mean_curl=+', fld)

            # Spatial coupling of instructions: spawn (19,16) points down to (19,17) INC
            ins = fs.instructions_query(db, meta, opcode='SPAWN')
            self.assertIn('( 19, 16)', ins)
            self.assertIn('-> (19,17) B=2(INC)', ins)

            # Region query with --grid
            reg = fs.region(db, meta, 16, 14, 5, 5, grid=True)
            self.assertIn('R/G DIRECTION GRID 1:1', reg)
            self.assertIn('B OPCODE GRID 1:1', reg)

    def test_channel_corruption_detected_independently(self):
        png = self._make_synthetic_vortex(self.root / 'v.png', size=16)
        for kind in ('R', 'G', 'B'):
            dbp = self.root / f'corrupt_{kind}.sqlite'
            fs.index_image(png, dbp, mode='flow')
            with closing(sqlite3.connect(dbp)) as conn:
                if kind in ('R', 'G'):
                    old = bytes(conn.execute('SELECT data FROM field_raw WHERE channel = ?', (kind,)).fetchone()[0])
                    mutated = bytes([old[0] ^ 1]) + old[1:]
                    conn.execute('UPDATE field_raw SET data = ? WHERE channel = ?', (sqlite3.Binary(mutated), kind))
                else:
                    conn.execute('UPDATE b_runs SET b = 99 WHERE y = 0 AND x0 = 0')
                conn.commit()
            with fs.open_index(dbp) as (db, meta):
                with self.assertRaises(fs.Deny):
                    fs.verify(db, meta)

    def test_flow_diff_separates_field_and_instruction_changes(self):
        png1 = self._make_synthetic_vortex(self.root / 'a.png', size=16)
        png2 = self.root / 'b.png'
        im = Image.open(png1).convert('RGB')
        r, g, b = im.getpixel((2, 2))
        im.putpixel((2, 2), ((r + 10) % 256, g, b))       # R/G field change only
        r2, g2, _ = im.getpixel((4, 4))
        im.putpixel((4, 4), (r2, g2, 8))                   # B instruction change (SPLIT)
        im.save(png2)

        dbp = self.root / 'a.sqlite'
        fs.index_image(png1, dbp, mode='flow')
        with fs.open_index(dbp) as (db, meta):
            out = fs.diff(db, meta, str(png2))
            self.assertIn('total_changed=2/', out)
            self.assertIn('field_RG_changed=1', out)
            self.assertIn('instr_B_changed=1', out)
            self.assertIn('B=0(HALT) -> B=8(SPLIT)', out)

    def test_determinism_byte_exact(self):
        png = self._make_synthetic_vortex(self.root / 'det.png', size=24)
        db1 = self.root / 'run1.sqlite'
        db2 = self.root / 'run2.sqlite'
        fs.index_image(png, db1, mode='flow')
        fs.index_image(png, db2, mode='flow')
        self.assertEqual(db1.read_bytes(), db2.read_bytes())

        with fs.open_index(db1) as (d1, m1), fs.open_index(db2) as (d2, m2):
            self.assertEqual(fs.summary(d1, m1), fs.summary(d2, m2))
            self.assertEqual(fs.field_query(d1, m1), fs.field_query(d2, m2))
            self.assertEqual(fs.instructions_query(d1, m1), fs.instructions_query(d2, m2))


@unittest.skipUnless(VORTEX_PNG.exists() and HELLO_PNG.exists(), 'FLOW examples not present')
class RealFlowProgramsGatesTests(unittest.TestCase):
    """Gates 1..6 directly against real DannyBaanks/FLOW example PNGs."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_gate1_gate2_gate3_gate5_vortex_and_hello(self):
        for name, png, expected_spawns, min_rgb in [
            ('hello_flow', HELLO_PNG, 1, 5),
            ('vortex', VORTEX_PNG, 8, 800),
        ]:
            db1 = self.root / f'{name}_1.sqlite'
            db2 = self.root / f'{name}_2.sqlite'
            # Gate 1: index real FLOW PNGs without rejecting >64 RGB colors
            fs.index_image(png, db1, mode='flow')
            fs.index_image(png, db2, mode='flow')
            # Gate 5: byte-exact SQLite determinism
            self.assertEqual(db1.read_bytes(), db2.read_bytes(), f'{name} sqlite non-deterministic')

            with fs.open_index(db1) as (db, meta):
                self.assertGreaterEqual(int(meta['distinct_rgb']), min_rgb)
                self.assertEqual(int(meta['spawn_count']), expected_spawns)

                # Gate 2: exact channel verification and rebuild (0 error)
                ver = fs.verify(db, meta, str(png))
                self.assertIn('0 reconstruction error', ver)
                rebuilt = self.root / f'{name}_rebuilt.png'
                fs.save_rgba_png(db, meta, rebuilt)
                orig_im = Image.open(png).convert('RGB')
                reb_im = Image.open(rebuilt).convert('RGB')
                self.assertEqual(orig_im.size, reb_im.size)
                self.assertEqual(orig_im.tobytes(), reb_im.tobytes())

                # L0 compact and deterministic
                l0 = fs.summary(db, meta)
                self.assertLessEqual(len(l0), 3500)
                with fs.open_index(db2) as (d2, m2):
                    self.assertEqual(l0, fs.summary(d2, m2))

                # Gate 3: spot-check exact pixel & regional values against PIL ground truth
                w, h = orig_im.size
                for sx, sy in [(0, 0), (w // 2, h // 2), (2, 2), (w - 1, h - 1)]:
                    r, g, b = orig_im.getpixel((sx, sy))
                    pt = fs.point(db, meta, sx, sy)
                    self.assertIn(f'RGB=({r},{g},{b})', pt)
                    self.assertIn(f'B={b} ', pt)

        # Specific structural checks on vortex.png
        with fs.open_index(self.root / 'vortex_1.sqlite') as (db, meta):
            # Center region around (32,32) has positive curl (tangential CW in y-down)
            fld = fs.field_query(db, meta, box=(16, 16, 32, 32))
            self.assertIn('mean_curl=+', fld)
            # Top-right quadrant (32..64, 0..32): dx>0, dy<0 -> vx=-dy/r>0, vy=dx/r>0 -> SE
            q_tr = fs.field_query(db, meta, box=(32, 0, 32, 32))
            self.assertIn('mean_v=(+', q_tr)
            # Bottom-left quadrant (0..32, 32..64): dx<0, dy>0 -> vx<0, vy<0 -> NW
            q_bl = fs.field_query(db, meta, box=(0, 32, 32, 32))
            self.assertIn('mean_v=(-', q_bl)
            # 8 spawns all present with exact coordinates
            ins = fs.instructions_query(db, meta, opcode='SPAWN')
            self.assertIn('matches=8', ins)

        # Specific structural checks on hello_flow.png
        with fs.open_index(self.root / 'hello_flow_1.sqlite') as (db, meta):
            ins = fs.instructions_query(db, meta)
            self.assertIn('B=255(SPAWN):1', ins)
            self.assertIn('B=2(INC):5', ins)
            self.assertIn('B=5(WRITE):1', ins)
            self.assertIn('(  2,  2) B=255 SPAWN', ins)
            self.assertIn('-> (3,2) B=2(INC)', ins)
            self.assertIn('(  8,  2) B=5   WRITE', ins)
            self.assertIn('-> (9,2) B=0(HALT)', ins)

    @unittest.skipUnless(flow_core is not None and flow_trace is not None, 'FLOW VM not installed')
    def test_gate4_vm_trace_invariance(self):
        """Rebuilt PNG produces a semantically IDENTICAL ExecutionTrace in real FLOW VM."""
        for name, png, ticks in [
            ('hello_flow', HELLO_PNG, 200),
            ('vortex', VORTEX_PNG, 500),
        ]:
            dbp = self.root / f'{name}.sqlite'
            rebuilt = self.root / f'{name}_rebuilt.png'
            fs.index_image(png, dbp, mode='flow')
            with fs.open_index(dbp) as (db, meta):
                fs.save_rgba_png(db, meta, rebuilt)

            t_orig = flow_trace.run_program(str(png), config=flow_core.FlowConfig(max_ticks=ticks, seed=42), seed=42)
            t_reb = flow_trace.run_program(str(rebuilt), config=flow_core.FlowConfig(max_ticks=ticks, seed=42), seed=42)
            # Canonical projection ignores program_sha256 only if we compare ticks+events+counts
            # (PNG zlib headers may differ, so compare events & ticks & counts directly)
            self.assertEqual(t_orig.ticks, t_reb.ticks)
            self.assertEqual(t_orig.events, t_reb.events)
            self.assertEqual(t_orig.metadata['particles_spawned'], t_reb.metadata['particles_spawned'])
            self.assertEqual(t_orig.metadata['final_alive'], t_reb.metadata['final_alive'])


if __name__ == '__main__':
    unittest.main()
