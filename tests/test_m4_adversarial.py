"""FlowSight M4 Adversarial & Gate Verification Suite (M4-A, M4-B, M4-C, MCP).

Covers all 13 required cases in COMPOSE §7 + negative controls:
  1. Rectangular button with known text ("Merge pull request")
  2. Two nearby buttons with distinct labels ("Save", "Cancel")
  3. External labels associated to inputs ("Email:" left-of, "Target Branch" above)
  4. Small / low-contrast text -> OCR_UNCERTAIN (never promoted to OCR_OBSERVED)
  5. Overlapping / nested controls + partial border straddle ambiguity
  6. Dark & light theme interfaces
  7. Modal insertion renumbering local W-IDs without shifting persistent K-tracks
  8. Color change (green -> orange) preserving persistent K-track + COLOR_CHANGED + INFERRED STATE_CHANGED_CANDIDATE
  9. Button movement (+140, -50) -> MOVED_CONTROL preserving track, plus RESIZED_CONTROL & TEXT_CHANGED
 10. Disappearance + far appearance of similar control -> REMOVED_CONTROL + NEW_CONTROL (no false match)
 11. Real 1920x1080 Linux desktop screenshot (if present on host, read-only)
 12. No OCR installed -> explicit OCR_ABSENT, NO_TEXT_EVIDENCE, zero invented labels
 13. Visually identical controls forcing AMBIGUOUS_MATCH (temporal) and AMBIGUOUS_ASSOCIATION (OCR)
"""
from __future__ import annotations

import asyncio
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'examples'))

import corpus_m4
import flowsight_diff as fdiff
import flowsight_ocr as focr
import flowsight_screen as fscr
import flowsight_vlm as fvlm

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_CAPS_DIR = Path('/home/danny/Imágenes/Capturas de pantalla')


def _synthetic_test_ocr_provider(items: list[dict]):
    """Unit-test helper ONLY (tagged engine='TEST_SYNTHETIC_OCR'); NEVER used to claim real OCR."""
    def _prov(rgb, offset_x=0, offset_y=0, source_frame='test', lang='eng'):
        out = []
        for idx, it in enumerate(items):
            conf = float(it.get('confidence', 92.0))
            st = it.get('status') or (focr.OCR_OBSERVED if conf >= 60.0 else focr.OCR_UNCERTAIN)
            out.append({
                'id': f'O{idx}',
                'text': it['text'],
                'bbox': list(it['bbox']),
                'confidence': conf,
                'min_word_confidence': conf,
                'confidence_scale': 'test_synthetic_0_100',
                'engine': 'TEST_SYNTHETIC_OCR',
                'source_frame': source_frame,
                'status': st,
            })
        return {
            'status': focr.OCR_OBSERVED if any(x['status'] == focr.OCR_OBSERVED for x in out) else focr.OCR_UNCERTAIN,
            'engine': 'TEST_SYNTHETIC_OCR',
            'confidence_scale': 'test_synthetic_0_100',
            'source_frame': source_frame,
            'lang': lang,
            'preprocessing': 'test_synthetic',
            'detail': 'synthetic unit test fixture (NOT real Tesseract)',
            'items': out,
        }
    return _prov


class M4AdversarialTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.corpus_dir = Path(cls.tmp.name) / 'corpus'
        cls.manifest = corpus_m4.build_corpus(cls.corpus_dir)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_case1_2_3_6_ocr_association_and_dark_light_themes(self):
        """Cases 1, 2, 3, 6: known button text, 2 nearby buttons, external input labels, dark+light."""
        for theme in ('dark', 'light'):
            info = self.manifest['cases'][f'pr_{theme}']
            # Build exact OCR fixture matching ground-truth text positions (testing geometric association logic)
            ocr_items = []
            for c in info['controls']:
                tb = c.get('text_bbox') or c.get('label_bbox')
                ocr_items.append({'text': c['expected_text'], 'bbox': tb, 'confidence': 93.0})

            ctx = fscr.run_pipeline(
                info['path'],
                vf='theme,layout,controls,ocr',
                ocr_provider=_synthetic_test_ocr_provider(ocr_items),
            )
            self.assertEqual(ctx.artifacts['theme']['polarity'], theme)
            ctls = ctx.artifacts['controls']['controls']
            by_text = {c.get('text'): c for c in ctls if c.get('text')}

            # Case 1: Merge pull request button
            self.assertIn('Merge pull request', by_text, ctls)
            self.assertEqual(by_text['Merge pull request']['association'], focr.ASSOC_INSIDE)
            self.assertEqual(by_text['Merge pull request']['text_source'], focr.OCR_OBSERVED)

            # Case 2: Two nearby buttons keep distinct labels
            self.assertIn('Save', by_text, ctls)
            self.assertIn('Cancel', by_text, ctls)
            self.assertEqual(by_text['Save']['association'], focr.ASSOC_INSIDE)
            self.assertEqual(by_text['Cancel']['association'], focr.ASSOC_INSIDE)
            self.assertNotEqual(by_text['Save']['id'], by_text['Cancel']['id'])

            # Case 3: External labels associated to input boxes (left_of and above)
            self.assertIn('Email:', by_text, ctls)
            self.assertEqual(by_text['Email:']['association'], focr.ASSOC_LABEL)
            self.assertIn('Target Branch', by_text, ctls)
            self.assertEqual(by_text['Target Branch']['association'], focr.ASSOC_LABEL)

    def test_case4_5_low_contrast_uncertain_nested_and_partial_overlap_conflict(self):
        """Cases 4 & 5: low-contrast -> OCR_UNCERTAIN; nested card+button -> innermost; straddle -> AMBIGUOUS."""
        info = self.manifest['cases']['lowcontrast_nested']
        ocr_items = [
            # Case 4: low-contrast text gets low confidence (42.0) -> OCR_UNCERTAIN
            {'text': 'Advanced options', 'bbox': info['low_contrast_text_bbox'], 'confidence': 42.0},
            # Case 5a: text inside button nested inside card -> should associate to innermost button
            {'text': 'Confirm', 'bbox': info['inner_text_bbox'], 'confidence': 95.0},
        ]
        ctx = fscr.run_pipeline(
            info['path'],
            vf='layout,controls,ocr',
            ocr_provider=_synthetic_test_ocr_provider(ocr_items),
        )
        ctls = ctx.artifacts['controls']['controls']
        by_text = {c.get('text'): c for c in ctls if c.get('text')}
        self.assertIn('Confirm', by_text, ctls)
        self.assertEqual(by_text['Confirm']['association'], focr.ASSOC_INSIDE)
        # Inner button width is ~160px, not the 460px outer card
        self.assertLess(by_text['Confirm']['size'][0], 220)
        if 'Advanced options' in by_text:
            self.assertEqual(by_text['Advanced options']['text_source'], focr.OCR_UNCERTAIN)

        # Case 5b: Direct conflict test — text straddling border of two adjacent non-nested controls
        c_left = {'id': 'W0', 'kind': 'bordered-control', 'global_bbox': [100, 100, 200, 140], 'click_center': [150, 120], 'inside_panel': 'P0'}
        c_right = {'id': 'W1', 'kind': 'bordered-control', 'global_bbox': [205, 100, 305, 140], 'click_center': [255, 120], 'inside_panel': 'P0'}
        straddle_ocr = {
            'status': focr.OCR_OBSERVED,
            'items': [{'id': 'O0', 'text': 'Shared?', 'bbox': [160, 110, 245, 130], 'confidence': 90.0, 'status': focr.OCR_OBSERVED}],
        }
        assoc = focr.associate_text_to_controls([c_left, c_right], straddle_ocr)
        self.assertTrue(assoc['ambiguities'])
        for c in assoc['controls']:
            self.assertEqual(c['association'], focr.ASSOC_AMBIGUOUS)

    def test_case7_8_modal_renumbering_and_color_change_preserve_identity(self):
        """Cases 7 & 8 (Core M3 bug fix): modal appears & shifts W indices + green button turns orange."""
        info = self.manifest['cases']['modal_and_color_change']
        ctx = fscr.run_pipeline(
            info['frame_b'],
            vf=f'layout,controls,ui_diff={info["frame_a"]}',
        )
        tr = ctx.artifacts['ui_diff']['tracking']
        tracks = tr['tracks']

        # Zero false REMOVED_CONTROL! Every original control in Frame A must stay tracked
        removed = [e for e in tracks if e['classification'] == fdiff.CLS_REMOVED]
        self.assertEqual(len(removed), 0, f'false REMOVED_CONTROL after modal insertion: {removed}')

        # New modal button/card must be classified as NEW_CONTROL
        new_ctls = [e for e in tracks if e['classification'] == fdiff.CLS_NEW]
        self.assertGreaterEqual(len(new_ctls), 1, tracks)

        # Merge pull request button [260,520,510,568) MUST be COLOR_CHANGED + INFERRED STATE_CHANGED_CANDIDATE
        color_ch = [e for e in tracks if fdiff.CLS_COLOR_CHANGED in e['flags']]
        self.assertEqual(len(color_ch), 1, tracks)
        e_merge = color_ch[0]
        self.assertIn(fdiff.CLS_STATE_CANDIDATE, e_merge['inferred_flags'])
        self.assertIn('green', e_merge['color_transition'])
        self.assertIn('orange', e_merge['color_transition'])
        # Confirm both Frame A and Frame B evidence preserved
        self.assertIsNotNone(ctx.artifacts['ui_diff']['frame_a']['controls'])
        self.assertIsNotNone(ctx.artifacts['ui_diff']['frame_b']['controls'])

    def test_case9_moved_resized_and_text_changed(self):
        """Case 9: button move (+140, -50) -> MOVED_CONTROL preserving track + unit test for RESIZED & TEXT_CHANGED."""
        info = self.manifest['cases']['move_button']
        ctx = fscr.run_pipeline(info['frame_b'], vf=f'layout,controls,ui_diff={info["frame_a"]}')
        tracks = ctx.artifacts['ui_diff']['tracking']['tracks']
        moved = [e for e in tracks if fdiff.CLS_MOVED in e['flags']]
        self.assertEqual(len(moved), 1, tracks)
        self.assertEqual(len([e for e in tracks if e['classification'] == fdiff.CLS_NEW]), 0)
        self.assertEqual(len([e for e in tracks if e['classification'] == fdiff.CLS_REMOVED]), 0)
        dx, dy = moved[0]['delta_center_px']
        self.assertAlmostEqual(dx, 140, delta=4)
        self.assertAlmostEqual(dy, -50, delta=4)

        # Direct unit check for RESIZED_CONTROL + TEXT_CHANGED on same persistent track
        ca = [{'id': 'W0', 'kind': 'accent-control', 'global_bbox': [200, 200, 340, 244], 'click_center': [270, 222],
               'hex': '#16a34a', 'family': 'green', 'inside_panel': 'P0', 'text': 'Submit', 'text_source': focr.OCR_OBSERVED}]
        cb = [{'id': 'W5', 'kind': 'accent-control', 'global_bbox': [200, 200, 420, 244], 'click_center': [310, 222],
               'hex': '#16a34a', 'family': 'green', 'inside_panel': 'P0', 'text': 'Submitting...', 'text_source': focr.OCR_OBSERVED}]
        p = [{'id': 'P0', 'position_hint': 'main-pane', 'global_bbox': [0, 0, 1280, 720]}]
        res = fdiff.match_controls_temporal(ca, cb, p, p, (1280, 720), prev_track_map={'W0': 'K7'})
        e = res['tracks'][0]
        self.assertEqual(e['track'], 'K7')  # multi-frame track persistence!
        self.assertIn(fdiff.CLS_RESIZED, e['flags'])
        self.assertIn(fdiff.CLS_TEXT_CHANGED, e['flags'])

    def test_case10_negative_control_disappear_and_far_similar_new(self):
        """Case 10: similar blue button disappears top-left and another appears far bottom-right -> REMOVED + NEW."""
        info = self.manifest['cases']['disappear_and_new_far']
        ctx = fscr.run_pipeline(info['frame_b'], vf=f'layout,controls,ui_diff={info["frame_a"]}')
        tracks = ctx.artifacts['ui_diff']['tracking']['tracks']
        cls_set = {e['classification'] for e in tracks}
        self.assertIn(fdiff.CLS_REMOVED, cls_set, tracks)
        self.assertIn(fdiff.CLS_NEW, cls_set, tracks)

    def test_case12_ocr_absent_never_invents_labels(self):
        """Case 12: when requested OCR backend is absent, returns OCR_ABSENT + NO_TEXT_EVIDENCE and 0 invented texts."""
        tess, _ = focr.detect_tesseract_engine()
        if tess is not None:
            self.skipTest('tesseract is installed on this host')
        info = self.manifest['cases']['pr_dark']
        ctx = fscr.run_pipeline(info['path'], vf='layout,controls,ocr=backend=tesseract')
        self.assertEqual(ctx.artifacts['ocr']['status'], focr.OCR_ABSENT)
        for c in ctx.artifacts['controls']['controls']:
            self.assertIsNone(c['text'])
            self.assertEqual(c['association'], focr.ASSOC_NONE)
            self.assertEqual(c['association_reason'], focr.OCR_ABSENT)

    def test_gate_b_real_ocr_no_mocks_pr_dark_light_and_uidiff(self):
        """Gate B REAL OCR (ZERO MOCKS): reads rendered labels and binds them to controls & temporal tracks."""
        rap_ok, _ = focr.detect_rapidocr_engine()
        tess, _ = focr.detect_tesseract_engine()
        if not rap_ok and not tess:
            self.skipTest('no local OCR engine installed')

        for theme in ('dark', 'light'):
            info = self.manifest['cases'][f'pr_{theme}']
            ctx = fscr.run_pipeline(info['path'], vf='theme,layout,controls,ocr')
            oc = ctx.artifacts['ocr']
            self.assertEqual(oc['status'], focr.OCR_OBSERVED, oc)
            self.assertNotEqual(oc['engine'], 'TEST_SYNTHETIC_OCR')

            ctls = ctx.artifacts['controls']['controls']
            by_norm = {focr.normalize_text(c.get('text') or ''): c for c in ctls if c.get('text')}
            self.assertIn('merge pull request', by_norm, ctls)
            self.assertEqual(by_norm['merge pull request']['association'], focr.ASSOC_INSIDE)
            self.assertIn('save', by_norm, ctls)
            self.assertIn('cancel', by_norm, ctls)
            self.assertIn('target branch', by_norm, ctls)
            self.assertEqual(by_norm['target branch']['association'], focr.ASSOC_LABEL)

        # Real OCR inside temporal ui_diff (Case 7+8): Merge pull request keeps track AND text across color change!
        c78 = self.manifest['cases']['modal_and_color_change']
        ctx_d = fscr.run_pipeline(c78['frame_b'], vf=f'layout,controls,ocr,ui_diff={c78["frame_a"]}')
        tracks = ctx_d.artifacts['ui_diff']['tracking']['tracks']
        color_ch = [e for e in tracks if fdiff.CLS_COLOR_CHANGED in e['flags']]
        self.assertEqual(len(color_ch), 1, tracks)
        self.assertEqual(focr.normalize_text(color_ch[0]['frame_a']['text'] or ''), 'merge pull request')
        self.assertEqual(focr.normalize_text(color_ch[0]['frame_b']['text'] or ''), 'merge pull request')
        # Dismiss button inside new modal read via real OCR!
        new_texts = {focr.normalize_text(e['frame_b']['text'] or '') for e in tracks if e['classification'] == fdiff.CLS_NEW}
        self.assertIn('dismiss', new_texts, tracks)

    def test_case13_ambiguous_match_and_ambiguous_external_label(self):
        """Case 13: visually identical controls force AMBIGUOUS_MATCH & equidistant label forces AMBIGUOUS_ASSOCIATION."""
        info = self.manifest['cases']['ambiguous_identical']
        ctx = fscr.run_pipeline(info['frame_b'], vf=f'layout,controls,ui_diff={info["frame_a"]}')
        tracks = ctx.artifacts['ui_diff']['tracking']['tracks']
        ambig = [e for e in tracks if e['classification'] == fdiff.CLS_AMBIGUOUS]
        self.assertGreaterEqual(len(ambig), 1, f'expected AMBIGUOUS_MATCH for symmetric identical buttons: {tracks}')

        # Equidistant external label above two identical adjacent inputs -> AMBIGUOUS_ASSOCIATION
        i1 = {'id': 'W0', 'kind': 'input-or-bar-box', 'global_bbox': [200, 200, 340, 236], 'click_center': [270, 218], 'inside_panel': 'P0'}
        i2 = {'id': 'W1', 'kind': 'input-or-bar-box', 'global_bbox': [350, 200, 490, 236], 'click_center': [420, 218], 'inside_panel': 'P0'}
        lbl_ocr = {
            'status': focr.OCR_OBSERVED,
            'items': [{'id': 'O0', 'text': 'Name', 'bbox': [305, 174, 385, 194], 'confidence': 94.0, 'status': focr.OCR_OBSERVED}],
        }
        assoc = focr.associate_text_to_controls([i1, i2], lbl_ocr)
        self.assertTrue(assoc['ambiguities'])
        self.assertEqual(assoc['controls'][0]['association'], focr.ASSOC_AMBIGUOUS)
        self.assertEqual(assoc['controls'][1]['association'], focr.ASSOC_AMBIGUOUS)

    def test_case11_real_desktop_1080p_screenshot_readonly(self):
        """Case 11: real 1920x1080 desktop screenshot (if present on host, read-only)."""
        caps = sorted(REAL_CAPS_DIR.glob('*.png')) if REAL_CAPS_DIR.is_dir() else []
        if not caps:
            self.skipTest('no real desktop screenshots present on host')
        ctx = fscr.run_pipeline(caps[-1], vf='theme,layout,controls,text_bands,ocr,ui_map=48:18')
        self.assertEqual((ctx.full_w, ctx.full_h), (1920, 1080))
        self.assertIn('layout', ctx.artifacts)
        self.assertIn('controls', ctx.artifacts)
        self.assertIn('ocr', ctx.artifacts)
        self.assertLessEqual(len(fscr.format_pipeline_report(ctx)), 6500)

    def test_m4c_vlm_probe_disabled_by_default_and_policy_guards(self):
        """M4-C contract: disabled by default, rejects full-frame, tags output VLM_INFERRED."""
        info = self.manifest['cases']['pr_dark']
        # Default: disabled
        ctx = fscr.run_pipeline(info['path'], vf='crop=250:510:270:70,vlm_probe=question=What button is this?')
        self.assertEqual(ctx.artifacts['vlm_probe']['status'], fvlm.VLM_DISABLED)

        # Enabled policy refuses full-frame without crop
        pol = fvlm.VLMProbePolicy(enabled=True, require_crop_subregion=True)
        ctx_full = fscr.run_pipeline(info['path'], vf='vlm_probe=question=Describe full screen', vlm_policy=pol)
        self.assertEqual(ctx_full.artifacts['vlm_probe']['status'], fvlm.VLM_DENIED_POLICY)

        # Enabled policy with crop but no backend -> VLM_UNAVAILABLE (no implicit network call)
        ctx_nob = fscr.run_pipeline(info['path'], vf='crop=250:510:270:70,vlm_probe=question=Check icon', vlm_policy=pol)
        self.assertEqual(ctx_nob.artifacts['vlm_probe']['status'], fvlm.VLM_UNAVAILABLE)

        # Enabled policy + explicit mock backend -> tagged VLM_INFERRED with exact global_bbox
        ctx_ok = fscr.run_pipeline(
            info['path'],
            vf='crop=250:510:270:70,vlm_probe=question=Check icon',
            vlm_policy=pol,
            vlm_backend=lambda raw, req, p: 'Green rectangular button labeled Merge pull request',
        )
        vp = ctx_ok.artifacts['vlm_probe']
        self.assertEqual(vp['status'], fvlm.VLM_INFERRED)
        self.assertEqual(vp['provenance'], fvlm.VLM_INFERRED)
        self.assertEqual(vp['global_bbox'], [250, 510, 520, 580])

    def test_cer_wer_metric_math(self):
        self.assertEqual(focr.compute_cer_wer('Merge pull request', 'Merge pull request'), {'cer': 0.0, 'wer': 0.0})
        m = focr.compute_cer_wer('Merge pull request', 'Merge pul request')
        self.assertGreater(m['cer'], 0.0)
        self.assertAlmostEqual(m['wer'], 1 / 3, places=3)

    def test_filter_registry_metadata_complete_and_mcp_query_tool(self):
        """§4 FFmpeg filter metadata & MCP backward-compat + new flowsight_screen_query tool."""
        for name, spec in fscr.FILTERS.items():
            self.assertTrue(spec.summary)
            self.assertTrue(spec.syntax)
            self.assertIn(spec.coord_space, ('global', 'local'))
            self.assertIsInstance(spec.requires_network, bool)
            self.assertIsInstance(spec.deterministic, bool)

        info = self.manifest['cases']['modal_and_color_change']
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
                    self.assertTrue({
                        'flowsight_map_blocks',
                        'flowsight_get_block',
                        'flowsight_subdivide_block',
                        'flowsight_stitch_blocks',
                        'flowsight_inspect_lossless',
                        'flowsight_screen_filters',
                        'flowsight_screen_run',
                        'flowsight_screen_query',
                    }.issubset(tools))

                    q_out = (await session.call_tool('flowsight_screen_query', {
                        'image_path': info['frame_b'],
                        'selector': 'K0',
                        'vf': f'layout,controls,ui_diff={info["frame_a"]}',
                    })).content[0].text
                    self.assertIn('"track": "K0"', q_out)

        asyncio.run(_run())


if __name__ == '__main__':
    unittest.main()
