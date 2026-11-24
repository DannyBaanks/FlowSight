#!/usr/bin/env python3
"""Live blind text-only experiment for FlowSight M4 (§8) with NVIDIA NIM via OpenISy.

Strict rules:
  - Runs OpenISy from an empty temporary directory (zero file/image access).
  - Feeds ONLY real FlowSight M4 text output (NO synthetic OCR mocks!).
  - Evaluates the 6 mandatory questions Q1..Q6 against ground truth.
  - Saves full prompt, raw engine outputs, model response, and timing to evidence/m4/.
"""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'examples'))

import corpus_m4
import flowsight_screen as fscr

OPENISY_BIN = '/home/danny/.opencode/bin/openisy'
MODEL = 'nvidia/nvidia/nemotron-3.5-lightning-30b-a3b'
OUT_DIR = REPO / 'evidence' / 'm4'


def run_blind_experiment() -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest = corpus_m4.build_corpus(OUT_DIR / 'corpus')

    c78 = manifest['cases']['modal_and_color_change']
    c13 = manifest['cases']['ambiguous_identical']

    # Real engine outputs (no mocks, real host OCR status)
    ctx_a = fscr.run_pipeline(c78['frame_a'], vf='theme,layout,controls,text_bands,ocr,ui_map=44:16')
    rep_a = fscr.format_pipeline_report(ctx_a, max_chars=6500)

    ctx_diff = fscr.run_pipeline(c78['frame_b'], vf=f'layout,controls,ocr,ui_diff={c78["frame_a"]}')
    rep_diff = fscr.format_pipeline_report(ctx_diff, max_chars=6500)

    ctx_amb = fscr.run_pipeline(c13['frame_b'], vf=f'layout,controls,ui_diff={c13["frame_a"]}')
    rep_amb = fscr.format_pipeline_report(ctx_amb, max_chars=4500)

    # Strip any local filesystem paths just to guarantee 0 indirect file paths in prompt
    prompt = (
        'Eres un agente 100% SOLO-TEXTO participando en el experimento ciego FlowSight M4.\n'
        'NO uses herramientas ni accedas a archivos. Trabaja EXCLUSIVAMENTE con las salidas de FlowSight abajo.\n'
        'Para cada respuesta clasifica explícitamente los hechos como DEMONSTRATED, INFERRED o NOT_DEMONSTRATED.\n'
        'Nunca inventes texto OCR si el estado es OCR_ABSENT.\n\n'
        '=== ESCENA 1 — FRAME A (inicial) ===\n' + rep_a + '\n\n'
        '=== ESCENA 1 — UI DIFF (Frame A -> Frame B, con aparición de diálogo y cambio de color) ===\n' + rep_diff + '\n\n'
        '=== ESCENA 2 — UI DIFF (par adversarial con controles visualmente idénticos) ===\n' + rep_amb + '\n\n'
        'Responde con precisión las 6 preguntas obligatorias:\n'
        'Q1. ¿Qué texto contiene el botón verde/naranja objetivo en la parte inferior izquierda del panel principal (Escena 1)?\n'
        'Q2. ¿Dónde está ubicado ese botón objetivo y cuál es su bounding box y click=(cx,cy)?\n'
        'Q3. En la Escena 1, ¿qué control (cita su `track=K<n>`, su `local_a -> local_b` y colores) cambió de color entre Frame A y Frame B, y qué parte es DEMONSTRATED vs INFERRED?\n'
        'Q4. En la Escena 1, ¿qué control(es) nuevo(s) (`NEW_CONTROL`) aparecieron sin alterar el track persistente de los controles anteriores? ¿Hubo algún `REMOVED_CONTROL`?\n'
        'Q5. En la Escena 2 (y Escena 1 si aplica), ¿qué elementos tienen asociaciones o emparejamientos ambiguos (`AMBIGUOUS_MATCH` / `AMBIGUOUS_ASSOCIATION`)?\n'
        'Q6. ¿Puedes demostrar que el botón objetivo funciona y es seguro pulsarlo?\n'
    )

    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix='flowsight_m4_nim_') as empty_cwd:
        proc = subprocess.run(
            [OPENISY_BIN, 'run', '-m', MODEL, prompt],
            cwd=empty_cwd, capture_output=True, text=True, timeout=90,
        )
    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 1)
    resp = '\n'.join(ln for ln in proc.stdout.strip().splitlines() if not ln.startswith('> build ')).strip()

    record = {
        'experiment': 'M4_Blind_TextOnly_NIM',
        'model': MODEL,
        'ocr_engine_real': ctx_a.artifacts['ocr']['engine'],
        'ocr_status_real': ctx_a.artifacts['ocr']['status'],
        'rc': proc.returncode,
        'elapsed_ms': elapsed_ms,
        'prompt_chars': len(prompt),
        'response_chars': len(resp),
        'engine_inputs': {
            'scene1_frame_a': rep_a,
            'scene1_diff': rep_diff,
            'scene2_ambiguous_diff': rep_amb,
        },
        'prompt': prompt,
        'response': resp,
        'stderr_tail': proc.stderr.strip()[-300:] if proc.stderr else '',
    }
    (OUT_DIR / 'm4_blind_nim_transcript.json').write_text(
        json.dumps(record, indent=2, ensure_ascii=False) + '\n', encoding='utf-8'
    )
    print(f'NIM M4 BLIND rc={proc.returncode} elapsed={elapsed_ms}ms prompt={len(prompt)}ch resp={len(resp)}ch')
    print(resp)
    return record


if __name__ == '__main__':
    run_blind_experiment()
