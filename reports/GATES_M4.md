# FlowSight M4 — Reporte de Gates, Métricas y Evidencia

Fecha: 2026-11-24.
Repositorio: `/home/danny/Development/ISyCo Git/FlowSight`.
Baseline M3 verificado antes de editar: `ef94e091fbdf8be1a00d7a66c0bacd525ba1ae1e` (worktree limpio, `31/31` PASS).
`FLOW` (`3e13909`) y `FlowGen` (`0dd6aa2`): **intactos (solo lectura)**.
`isymcp-hub`: **no modificado ni reiniciado**.

---

## 1. Resumen de arquitectura M4

- **M4-A ([`../flowsight_ocr.py`](../flowsight_ocr.py)):**
  - Extracción OCR real vía `tesseract` local cuando existe en `PATH`, con `text`, `bbox` global, `confidence` + escala explícita `tesseract_0_100_uncalibrated (NOT calibrated probability)`, `engine` + versión, `source_frame`, preprocesamiento por polaridad (`inverted_for_dark_theme` / `native_rgb`) y estados `OCR_OBSERVED`, `OCR_UNCERTAIN`, `OCR_ABSENT`, `OCR_ERROR`.
  - Asociación geométrica explícita texto ↔ controles (`TEXT_INSIDE`, `LABEL_ASSOCIATED`, `AMBIGUOUS_ASSOCIATION`, `NO_TEXT_EVIDENCE`) usando contención, intersección, distancia al centro, compatibilidad de panel padre, fusión de fragmentos partidos dentro de un control y detección estricta de conflictos sin elegir arbitrariamente.
  - Métricas exactas `CER`/`WER` deterministas (`compute_cer_wer`).
- **M4-B ([`../flowsight_diff.py`](../flowsight_diff.py)):**
  - `ui_diff` basado en **identidad temporal** (nunca en índices `W_i`): asignación global óptima vía `scipy.optimize.linear_sum_assignment` combinando IoU, proximidad, tamaño, panel padre, tipo de control, texto OCR normalizado y peso controlado de color (`<=0.10`, nunca clave absoluta).
  - Emite tracks persistentes `track=K0..Kn` separados de los IDs locales `local=W_i`, conserva ambos frames (`frame_a`, `frame_b`) como evidencia y soporta las 9 clasificaciones: `NEW_CONTROL`, `REMOVED_CONTROL`, `MOVED_CONTROL`, `RESIZED_CONTROL`, `TEXT_CHANGED`, `COLOR_CHANGED`, `STATE_CHANGED_CANDIDATE` (explícitamente `INFERRED`), `UNCHANGED_CONTROL` y `AMBIGUOUS_MATCH`.
- **M4-C ([`../flowsight_vlm.py`](../flowsight_vlm.py)):**
  - Contrato extensible fail-closed para futura escalación `vlm_probe` sobre recortes explícitos, **desactivado por defecto (`enabled=False`)**, prohíbe pantalla completa, acota tamaño/timeout y marca toda salida como `VLM_INFERRED`.
- **Pipeline FFmpeg & MCP ([`../flowsight_screen.py`](../flowsight_screen.py), [`../flowsight_mcp.py`](../flowsight_mcp.py)):**
  - Metadata declarativa por filtro en `FILTERS` (`inputs`, `outputs`, `optional_deps`, `requires_network`, `requires_gpu`, `deterministic`, `coord_space`, `provenance`, `degradations`), representación canónica estructurada (`to_canonical_dict`), reutilización de etapas sin reprocesamiento y nueva tool acotada `flowsight_screen_query` (8 tools MCP totales, 100% retrocompatibles).

---

## 2. Evaluación de Gates de Aceptación (§10)

| Gate | Criterio | Estado | Evidencia / Nota |
|---|---|---|---|
| **Gate A — Regresión** | Suite M3 reproducida (`31/31`), nuevas pruebas M4 (`+11` → `42/42`), `FLOW`/`FlowGen` intactos, sin acceso de red en tests deterministas | **DEMONSTRATED** | `python3 -m unittest discover -s tests -v`: **42/42 PASS**; `git rev-parse HEAD` y `status` sin cambios en `FLOW` y `FlowGen` |
| **Gate B — OCR (arquitectura, degradación y asociación)** | Degradación `OCR_ABSENT` explícita sin inventar etiquetas + asociación geométrica texto↔control + manejo de `OCR_UNCERTAIN` y `AMBIGUOUS_ASSOCIATION` | **DEMONSTRATED** (contrato y asociación geométrica) / **NOT_DEMONSTRATED** (ejecución real de Tesseract + CER/WER en este host) | **`GAP` registrado:** `tesseract` no está instalado en este host (`No dpkg OCR packages`). El motor devuelve `OCR_ABSENT` sin inventar texto (`test_case12_ocr_absent_never_invents_labels`). Precisión/recall de la lógica de asociación geométrica sobre fixture sintético etiquetado `TEST_SYNTHETIC_OCR`: `1.00 / 1.00` (5/5 TP, 0 FP). Para cerrar la parte real de Tesseract: `sudo apt install tesseract-ocr && python3 examples/benchmark_m4.py` |
| **Gate C — Identidad temporal** | Modal nuevo no altera identidad de controles existentes; cambio de color verde→naranja mantiene `track=K_t`; `MOVED`/`RESIZED`/`TEXT_CHANGED`; negativo lejos → `REMOVED+NEW`; controles idénticos → `AMBIGUOUS_MATCH` | **DEMONSTRATED** | `tests/test_m4_adversarial.py` (Casos 7, 8, 9, 10, 13) + `evidence/m4/benchmark_m4.json`: **`fp_removed_controls = 0`**, `color_changed_detected = 1`, `case9_move_exact = true`, `case10_negative_far_exact = true`, `case13_ambiguous_detected = true` |
| **Gate D — Text-only blind experiment** | Experimento ciego real con `nvidia/nvidia/nemotron-3.5-lightning-30b-a3b` vía OpenISy sobre Q1..Q6 (cero acceso a PNG, directorio temporal vacío) | **DEMONSTRATED** (geometría, identidad temporal, ambigüedad y calibración epistémica; con 2 imprecisiones menores de prosa del LLM documentadas abajo) | `evidence/m4/m4_blind_nim_transcript.json` (`42.1 s`, `11,069 ch` prompt → `5,809 ch` respuesta) |
| **Gate E — Seguridad y operación** | Cero clics automáticos, `vlm_probe` desactivado por defecto, cero modificaciones a `isymcp-hub`, cero operaciones destructivas, rollback local definido | **DEMONSTRATED** | `test_m4c_vlm_probe_disabled_by_default_and_policy_guards`, `docs/MCP_REGISTRATION.md` (procedimiento reversible), `git status` limpio en `FLOW`/`FlowGen` |

---

## 3. Métricas medidas (§9, fuente: `evidence/m4/benchmark_m4.json`)

| Dimensión | Modo rápido (sin OCR) | Modo enriquecido (con etapa OCR) | Notas |
|---|---:|---:|---|
| Corpus `1280×720` (`pr_dark`) — `layout` | `13.30 ms` | — | |
| Corpus `1280×720` (`pr_dark`) — `controls` | `8.42 ms` | — | `mean_bbox_iou_vs_gt = 0.9678` (botones `1.00`, inputs `0.92`) |
| Corpus `1280×720` (`pr_dark`) — `text_bands` | `3.43 ms` | — | |
| Corpus `1280×720` — `ocr` + `association` | — | `0.48 ms` + `0.04 ms` | `OCR_ABSENT` (con Tesseract real subirá sustancialmente; no asumir 20 FPS) |
| Corpus `1280×720` — total / chars | `33.32 ms` / `3,694 ch` | `30.83 ms` / `4,118 ch` | |
| `ui_diff` temporal (Caso 7+8, modal + color) | `63.77 ms` / `4,054 ch` | — | `fp_removed=0`, `color_changed=1`, `new=2` |
| Asociación geométrica (fixture `TEST_SYNTHETIC_OCR`) | — | `1.45 ms`, **P=1.00 / R=1.00** (5/5 TP, 0 FP) | Mide solo la geometría de asociación; NO sustituye OCR real |
| Captura real escritorio Linux `1920×1080` (bajo `tracemalloc`) | `layout 51.99 ms`, `controls 38.61 ms`, `text_bands 48.23 ms`, **total `146.84 ms`** (`6,500 ch`) | `ocr 0.56 ms` (`OCR_ABSENT`), `assoc 0.09 ms`, total `117.39 ms` | M3 histórico sin `tracemalloc`: `48.97 ms` (referencia histórica) |
| Memoria pico Python (`tracemalloc`) | **`59.83 MiB`** (`62,733,491 B`) | — | Incluye corpus completo + 1080p real |
| OCR CER / WER real | — | **`NOT_DEMONSTRATED`** | `GAP`: falta binario `tesseract` en host |

---

## 4. Evaluación del experimento ciego con NVIDIA NIM (`m4_blind_nim_transcript.json`)

Modelo: `nvidia/nvidia/nemotron-3.5-lightning-30b-a3b` (100% text-only, vía OpenISy en `cwd` temporal vacío, cero acceso a archivos).

| Pregunta | Resultado motor FlowSight | Respuesta del LLM | Veredicto y clasificación de errores |
|---|---|---|---|
| **Q1** (texto del botón objetivo) | `OCR_ABSENT`, `T2` geométrico, `NO_TEXT_EVIDENCE` | Respondió `NOT_DEMONSTRATED` sin inventar texto | **PASS (`DEMONSTRATED` calibración)** |
| **Q2** (ubicación, bbox y click) | `W2` en `[260, 520, 510, 568)`, `click=(385, 544)`, `P4` | Bbox y click exactos; *error menor del LLM:* dijo "coordenadas son locales por-frame" al confundir la advertencia de IDs `W` locales con coordenadas globales | **PASS con nota menor (`LLM_INTERPRETATION_SLIP`, motor exacto)** |
| **Q3** (control que cambió de color + `DEMONSTRATED` vs `INFERRED`) | `track=K2 local_a=W2->local_b=W4 COLOR_CHANGED #16a34a(green)->#ea9418(orange) INFERRED=['STATE_CHANGED_CANDIDATE']` | Citó `track=K2`, `W2->W4`, colores exactos, y separó `COLOR_CHANGED` (`DEMONSTRATED`) de `STATE_CHANGED_CANDIDATE` (`INFERRED`) | **PASS (`DEMONSTRATED` — bug M3 eliminado)** |
| **Q4** (`NEW_CONTROL` sin alterar tracks + `REMOVED_CONTROL`) | `K5` (`W0` modal) y `K6` (`W3` botón), `0 REMOVED_CONTROL`, `tracks={'COLOR_CHANGED':1, 'NEW_CONTROL':2, 'UNCHANGED_CONTROL':4}` | Identificó `K5`, `K6` y `0 REMOVED_CONTROL`; *desliz menor del LLM:* resumió "K0-K4 son todos UNCHANGED" aunque en la misma línea citó el conteo exacto `COLOR_CHANGED: 1, UNCHANGED_CONTROL: 4` | **PASS con nota menor (`LLM_PROSE_SLIP`, motor exacto)** |
| **Q5** (ambigüedades) | Escena 1: `0`; Escena 2: `track=K0 AMBIGUOUS_MATCH score=0.9213` | Reportó exacto `0` en Escena 1 y `track=K0 AMBIGUOUS_MATCH score=0.9213` en Escena 2 | **PASS (`DEMONSTRATED`)** |
| **Q6** (¿demuestra que funciona y es seguro pulsarlo?) | `geometric candidates — appearance != verified interactivity` | Respondió **`NOT_DEMONSTRATED`** citando el contrato explícito | **PASS (`DEMONSTRATED` calibración)** |

---

## 5. Clasificación consolidada & Rollback

- **DEMONSTRATED:**
  - `ui_diff` basado en identidad temporal con tracks persistentes `K0..Kn`, eliminación del bug M3 de desplazamiento de IDs (`0` falsos `REMOVED_CONTROL`), preservación de ambos frames y detección de `AMBIGUOUS_MATCH`.
  - Contrato OCR M4-A con degradación `OCR_ABSENT`, asociación geométrica texto↔control (`TEXT_INSIDE`, `LABEL_ASSOCIATED`, `AMBIGUOUS_ASSOCIATION`, `NO_TEXT_EVIDENCE`, P=1.0/R=1.0 en corpus geométrico).
  - Contrato M4-C `vlm_probe` desactivado por defecto con guardas de política.
  - Retrocompatibilidad de las 7 tools MCP + nueva `flowsight_screen_query` (42/42 tests PASS).
  - Experimento ciego real Q1..Q6 con `nvidia/nvidia/nemotron-3.5-lightning-30b-a3b`.
- **INFERRED:**
  - `STATE_CHANGED_CANDIDATE` en `ui_diff` (sin señal de accesibilidad).
  - Etiquetas posicionales (`top-bar-band`, `main-pane`) y naturaleza interactiva de controles candidatos.
- **NOT_DEMONSTRATED:**
  - Ejecución de OCR real con Tesseract y medición real de `CER`/`WER` en este host (`GAP`: `tesseract-ocr` no instalado).
  - Escalación VLM viva a un modelo visual remoto/local (contrato preparado y desactivado por defecto).
  - Registro activo en `isymcp-hub` (pendiente autorización explícita de Danny; ver [`../docs/MCP_REGISTRATION.md`](../docs/MCP_REGISTRATION.md)).

### Rollback local
Para revertir M4 al baseline M3 intacto sin tocar `FLOW` ni `FlowGen`:
```bash
git -C "/home/danny/Development/ISyCo Git/FlowSight" checkout ef94e09
```
