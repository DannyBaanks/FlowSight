"""Tests for opt-in VLM probe CLI/MCP adapter and local per-PC private failure corpus."""
from __future__ import annotations

import asyncio
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import os
from pathlib import Path
import stat
import subprocess
import sys
import tempfile
import threading
import unittest

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import flowsight_corpus as fcorp
import flowsight_screen as fscr
import flowsight_vlm as fvlm


def _make_sample_png(path: Path) -> Path:
    arr = np.full((240, 320, 3), 24, dtype=np.uint8)
    arr[40:100, 50:190] = [22, 163, 74]
    Image.fromarray(arr).save(path)
    return path


class _LocalVLMStub:
    """In-process loopback OpenAI-compatible Vision HTTP stub (zero external network)."""

    def __init__(self) -> None:
        self.requests: list[dict] = []
        self.auth_headers: list[str | None] = []
        outer = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, format, *args):  # noqa: A002
                pass

            def do_POST(self):
                length = int(self.headers.get('Content-Length', '0'))
                raw = self.rfile.read(length).decode('utf-8')
                outer.auth_headers.append(self.headers.get('Authorization'))
                body = json.loads(raw)
                outer.requests.append(body)
                resp = {
                    'id': 'stub-vlm-1',
                    'choices': [{
                        'index': 0,
                        'message': {
                            'role': 'assistant',
                            'content': 'Green rounded CTA rectangle on dark background.',
                        },
                    }],
                }
                enc = json.dumps(resp).encode('utf-8')
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(enc)))
                self.end_headers()
                self.wfile.write(enc)

        self.server = HTTPServer(('127.0.0.1', 0), Handler)
        self.port = self.server.server_port
        self.url = f'http://127.0.0.1:{self.port}/v1/chat/completions'
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self) -> _LocalVLMStub:
        self.thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self.server.shutdown()
        self.server.server_close()


class VLMProbeRuntimeTests(unittest.TestCase):
    def test_vlm_probe_disabled_by_default_and_blocks_remote_without_optin(self) -> None:
        pol, backend = fvlm.resolve_runtime_vlm_config(env={})
        self.assertFalse(pol.enabled)
        self.assertIsNone(backend)

        with tempfile.TemporaryDirectory() as td:
            img = _make_sample_png(Path(td) / 'ui.png')
            pol_rem, back_rem = fvlm.resolve_runtime_vlm_config(env={
                'FLOWSIGHT_VLM_ENABLED': '1',
                'FLOWSIGHT_VLM_URL': 'https://api.example.invalid/v1/chat/completions',
                'FLOWSIGHT_VLM_MODEL': 'vision-test',
                'FLOWSIGHT_VLM_ALLOW_REMOTE': '0',
            })
            ctx = fscr.run_pipeline(
                img,
                vf='crop=40:30:160:80,vlm_probe=question=What icon?',
                vlm_policy=pol_rem,
                vlm_backend=back_rem,
            )
            vp = ctx.artifacts['vlm_probe']
            self.assertEqual(vp['status'], fvlm.VLM_DENIED_POLICY)
            self.assertIn('allow_remote=False', vp['detail'])

    def test_vlm_probe_loopback_http_crop_only_and_cli_integration(self) -> None:
        with tempfile.TemporaryDirectory() as td, _LocalVLMStub() as stub:
            img = _make_sample_png(Path(td) / 'ui.png')
            secret = 'sk-test-never-leak-999'

            # 1. Full-frame transfer MUST be refused even with valid loopback endpoint
            pol, backend = fvlm.resolve_runtime_vlm_config(
                enable=True,
                url=stub.url,
                model='local-vlm',
                api_key=secret,
                allow_remote=False,
            )
            ctx_full = fscr.run_pipeline(
                img,
                vf='vlm_probe=question=Describe full screen',
                vlm_policy=pol,
                vlm_backend=backend,
            )
            self.assertEqual(ctx_full.artifacts['vlm_probe']['status'], fvlm.VLM_DENIED_POLICY)
            self.assertEqual(len(stub.requests), 0)

            # 2. Cropped subregion succeeds via CLI with --enable-vlm and never leaks secret
            env = dict(os.environ)
            env['FLOWSIGHT_VLM_API_KEY'] = secret
            proc = subprocess.run(
                [
                    sys.executable,
                    str(REPO / 'flowsight_screen.py'),
                    '-i', str(img),
                    '-vf', 'crop=45:35:150:70,vlm_probe=question=Describe cropped button',
                    '--enable-vlm',
                    '--vlm-url', stub.url,
                    '--vlm-model', 'local-vlm',
                ],
                capture_output=True,
                text=True,
                env=env,
                timeout=30,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            self.assertIn('VLM_PROBE [VLM_INFERRED] provenance=VLM_INFERRED', proc.stdout)
            self.assertIn('Green rounded CTA rectangle', proc.stdout)
            self.assertNotIn(secret, proc.stdout)
            self.assertNotIn(secret, proc.stderr)
            self.assertEqual(len(stub.requests), 1)
            self.assertEqual(stub.auth_headers[0], f'Bearer {secret}')
            user_content = stub.requests[0]['messages'][1]['content']
            self.assertTrue(user_content[1]['image_url']['url'].startswith('data:image/png;base64,'))


class PrivateFailureCorpusTests(unittest.TestCase):
    def test_private_corpus_per_pc_permissions_and_cli(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            cdir = Path(td) / 'private_corpus'
            img = _make_sample_png(Path(td) / 'shot.png')

            r1 = fcorp.log_failure(
                category='ocr_error',
                user_correction='El botón dice Publicar, no Publico.',
                flowsight_observed="W2 text='Publico'",
                agent_explanation='Dije que era Publico',
                image_path=img,
                vf_pipeline='theme,layout,controls,ocr',
                provider='nvidia',
                model='nemotron-3.5',
                save_local_copy=False,
                corpus_dir=cdir,
            )
            r2 = fcorp.log_failure(
                category='control_missed',
                user_correction='Faltó detectar el icono de configuración arriba a la derecha.',
                image_path=img,
                save_local_copy=True,
                corpus_dir=cdir,
            )
            self.assertEqual(r1['machine_id'], r2['machine_id'])
            self.assertTrue(r1['machine_id'].startswith('pc-'))
            self.assertIsNone(r1['local_image_copy'])
            self.assertIsNotNone(r2['local_image_copy'])
            self.assertTrue((cdir / r2['local_image_copy']).is_file())

            if os.name == 'posix':
                self.assertEqual(stat.S_IMODE(cdir.stat().st_mode), 0o700)
                self.assertEqual(stat.S_IMODE((cdir / 'failures.jsonl').stat().st_mode), 0o600)
                self.assertEqual(stat.S_IMODE((cdir / 'machine_id.txt').stat().st_mode), 0o600)

            rows = fcorp.list_failures(limit=10, corpus_dir=cdir)
            self.assertEqual(len(rows), 2)
            ocr_only = fcorp.list_failures(limit=10, category='ocr_error', corpus_dir=cdir)
            self.assertEqual(len(ocr_only), 1)
            self.assertEqual(ocr_only[0]['user_correction'], 'El botón dice Publicar, no Publico.')

            stats = fcorp.corpus_stats(cdir)
            self.assertEqual(stats['total_entries'], 2)
            self.assertEqual(stats['by_category']['ocr_error'], 1)
            self.assertEqual(stats['by_category']['control_missed'], 1)

    def test_mcp_stdio_exposes_failure_corpus_and_optin_vlm(self) -> None:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        async def _run() -> None:
            with tempfile.TemporaryDirectory() as td, _LocalVLMStub() as stub:
                cdir = Path(td) / 'mcp_corpus'
                img = _make_sample_png(Path(td) / 'ui.png')
                env = dict(os.environ)
                env['FLOWSIGHT_PRIVATE_CORPUS_DIR'] = str(cdir)
                env['FLOWSIGHT_VLM_ENABLED'] = '1'
                env['FLOWSIGHT_VLM_URL'] = stub.url
                env['FLOWSIGHT_VLM_MODEL'] = 'local-vlm'

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
                        self.assertIn('flowsight_log_failure', names)
                        self.assertIn('flowsight_list_failures', names)
                        self.assertGreaterEqual(len(names), 10)

                        # Call vlm_probe via MCP over cropped region
                        res_vlm = await session.call_tool(
                            'flowsight_screen_run',
                            {
                                'image_path': str(img),
                                'vf': 'crop=45:35:150:70,vlm_probe=question=Describe crop',
                            },
                        )
                        txt_vlm = res_vlm.content[0].text
                        self.assertIn('VLM_PROBE [VLM_INFERRED]', txt_vlm)
                        self.assertIn('Green rounded CTA rectangle', txt_vlm)

                        # Log a user-confirmed failure and list it back
                        res_log = await session.call_tool(
                            'flowsight_log_failure',
                            {
                                'category': 'ocr_error',
                                'user_correction': 'Decía Enviar, no Envia',
                                'flowsight_observed': "W0 text='Envia'",
                                'image_path': str(img),
                            },
                        )
                        self.assertIn('LOGGED fail-', res_log.content[0].text)

                        res_list = await session.call_tool('flowsight_list_failures', {'limit': 5})
                        self.assertIn('Decía Enviar, no Envia', res_list.content[0].text)

        asyncio.run(_run())


if __name__ == '__main__':
    unittest.main()
