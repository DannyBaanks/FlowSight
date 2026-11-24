#!/usr/bin/env python3
"""Live blind experiment: FlowSight M2 (1920x1080 quarters + stitch) with a real
text-only NVIDIA NIM model via OpenISy (`nvidia/nvidia/nemotron-3.5-lightning-30b-a3b`).

Zero image access: every OpenISy call runs from an empty temporary directory with
no file paths to the source image and receives ONLY FlowSight text output.
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
sys.path.insert(0, str(REPO / 'tests'))

import flowsight_blocks as fb
from test_blocks_and_mcp import _make_1920x1080_scene

OPENISY_BIN = '/home/danny/.opencode/bin/openisy'
MODEL = 'nvidia/nvidia/nemotron-3.5-lightning-30b-a3b'
OUT_DIR = REPO / 'evidence' / 'm2_blocks'


def call_nim_blind(prompt: str, timeout_s: int = 60) -> dict:
    """Invoke text-only NVIDIA NIM model via OpenISy from an isolated empty temp cwd."""
    t0 = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix='flowsight_blind_cwd_') as empty_cwd:
        proc = subprocess.run(
            [OPENISY_BIN, 'run', '-m', MODEL, prompt],
            cwd=empty_cwd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
    elapsed_ms = round((time.perf_counter() - t0) * 1000.0, 1)
    # Strip the leading "> build · ..." header line emitted by openisy CLI
    raw = proc.stdout.strip()
    lines = raw.splitlines()
    clean_lines = [ln for ln in lines if not ln.startswith('> build ')]
    text = '\n'.join(clean_lines).strip()
    return {
        'model': MODEL,
        'rc': proc.returncode,
        'elapsed_ms': elapsed_ms,
        'prompt_chars': len(prompt),
        'response_chars': len(text),
        'response': text,
        'stderr_tail': proc.stderr.strip()[-300:] if proc.stderr else '',
    }


def run_scene_experiment() -> dict:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sdir = OUT_DIR / 'sessions'
    scene_png = _make_1920x1080_scene(OUT_DIR / 'scene_1080p.png')

    t_map = time.perf_counter()
    sess = fb.map_image_blocks(scene_png, split='quarters', palette_k=10, session_dir=sdir)
    map_ms = round((time.perf_counter() - t_map) * 1000.0, 1)
    map_txt = fb.format_map_summary(sess)
    (OUT_DIR / 'scene_1080p_map_L0.txt').write_text(map_txt + '\n', encoding='utf-8')

    # Shared palette header (no file path!)
    pal_lines = ['Shared Global Palette:'] + [
        f'  {p["symbol"]}=C{p["cid"]} {p["hex"]} {p["family"]} L={p["luma"]} ({p["pct"]}%)'
        for p in sess['palette']
    ]
    pal_txt = '\n'.join(pal_lines)

    # Step 1: 4 isolated blind calls — one per 960x540 quarter
    block_notes = {}
    block_runs = {}
    for bid in sess['block_order']:
        b_txt = fb.format_block_detail(sess, bid, include_edge_map=True)
        # Sanitize filename just in case (keep only abstract id)
        (OUT_DIR / f'scene_1080p_block_{bid}.txt').write_text(b_txt + '\n', encoding='utf-8')
        prompt = (
            'Eres un agente SOLO-TEXTO participando en un experimento de visión por bloques (FlowSight).\n'
            'NO uses herramientas ni accedas a archivos. Solo analiza el texto del bloque abajo.\n'
            'En máximo 3 frases breves y factuales describe:\n'
            '1) Colores dominantes y su distribución vertical/horizontal en este bloque.\n'
            '2) Regiones/formas notables (posición dentro del bloque y bordes en EDGE MAP).\n'
            '3) Qué regiones tocan las costuras (LEFT/RIGHT/TOP/BOTTOM) para unirlo después.\n'
            'No inventes identidad semántica de objetos reales; solo describe geometría, color y bordes.\n\n'
            f'{pal_txt}\n\n{b_txt}'
        )
        res = call_nim_blind(prompt)
        block_runs[bid] = res
        block_notes[bid] = res['response']
        print(f'[BLOCK {bid}] rc={res["rc"]} {res["elapsed_ms"]}ms prompt={res["prompt_chars"]}ch resp={res["response_chars"]}ch')

    # Step 2: Stitch the 4 blocks + their notes with flowsight_stitch_blocks
    t_st = time.perf_counter()
    stitched = fb.stitch_blocks(sess['session_id'], block_notes=block_notes, session_dir=sdir)
    stitch_ms = round((time.perf_counter() - t_st) * 1000.0, 1)
    stitch_txt = fb.format_stitch_report(stitched)
    (OUT_DIR / 'scene_1080p_stitched.txt').write_text(stitch_txt + '\n', encoding='utf-8')

    # Step 3: Global synthesis by text-only NVIDIA NIM model using ONLY the stitched report
    synth_prompt = (
        'Eres un agente SOLO-TEXTO. NO uses herramientas ni archivos.\n'
        'Has recibido la salida de `flowsight_stitch_blocks`, que unió 4 bloques (TL, TR, BL, BR) '
        'de una imagen 1920x1080 midiendo continuidad real en las costuras y alineando las notas de cada bloque.\n\n'
        f'{pal_txt}\n\n{stitch_txt}\n\n'
        'Responde con precisión y brevedad las 6 preguntas siguientes, citando coordenadas/regiones G0..Gn/costuras '
        'y distinguiendo entre lo DEMONSTRATED (medido) y lo NOT_DEMONSTRATED:\n'
        'Q1. ¿Cuáles son las dimensiones globales y qué bandas horizontales principales (familia de color y zona vertical) cruzan de izquierda a derecha?\n'
        'Q2. ¿Hay una región/forma que cruce los 4 cuadrantes (TL, TR, BL, BR) en el centro? Indica su familia de color, G-id y global_bbox.\n'
        'Q3. ¿En qué cuadrante está la región amarilla clara (pale-yellow) y en qué parte de ese cuadrante aparece?\n'
        'Q4. ¿En qué cuadrante hay una banda/región con bordes horizontales fuertes (orient=horizontal / edge map) en la mitad inferior?\n'
        'Q5. ¿Qué niveles de continuidad de costura (continuity_pct) se midieron en las 4 costuras?\n'
        'Q6. ¿Puedes demostrar con certeza que esta imagen es una fotografía tomada en París con una cámara DSLR?\n'
    )
    synth_res = call_nim_blind(synth_prompt, timeout_s=75)
    print(f'[SYNTHESIS 1080p] rc={synth_res["rc"]} {synth_res["elapsed_ms"]}ms prompt={synth_res["prompt_chars"]}ch resp={synth_res["response_chars"]}ch')
    print(synth_res['response'])

    record = {
        'experiment': 'M2-A_1920x1080_NIM_Blind',
        'model': MODEL,
        'session_id': sess['session_id'],
        'map_ms': map_ms,
        'stitch_ms': stitch_ms,
        'quantization_error': sess['quantization_error'],
        'map_L0_chars': len(map_txt),
        'stitch_report_chars': len(stitch_txt),
        'block_runs': block_runs,
        'synthesis_run': synth_res,
    }
    (OUT_DIR / 'nim_1080p_transcript.json').write_text(json.dumps(record, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return record


if __name__ == '__main__':
    run_scene_experiment()
