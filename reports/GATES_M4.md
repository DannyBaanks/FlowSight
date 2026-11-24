# FlowSight M4 — Reporte de Gates, Métricas y Evidencia

Fecha: 2026-11-24.
Repositorio: `/home/danny/Development/ISyCo Git/FlowSight`.
Baseline M3 verificado antes de editar: `ef94e091fbdf8be1a00d7a66c0bacd525ba1ae1e` (worktree limpio, `31/31` PASS).
`FLOW` (`3e13909`) y `FlowGen` (`0dd6aa2`): **intactos (solo lectura)**.
`isymcp-hub`: **no modificado ni reiniciado**.

---

## 1. Resumen de arquitectura M4

- **M4-A ([`../flowsight_ocr.py`](../flowsight_ocr.py)):**
  - Extracción OCR local real (cero red, CPU) con selección `backend=auto|tesseract|rapidocr`:
    - Si `tesseract` está en `PATH`, usa Tesseract local (`tesseract_0_100_uncalibrated`).
    - Si no, usa **`rapidocr-onnxruntime==1.4.4`** (PP-OCRv4 ONNX instalado en user-space local, escala explícita `rapidocr_0_100_uncalibrated (raw 0..1 x100; NOT calibrated probability)`).
    - Si el backend solicitado no existe, degrada fail-closed a `OCR_ABSENT` sin inventar texto.
  - Para cada texto emite `text`, `bbox` global `[x0,y0,x1,y1)`, `confidence` + escala documentada, `engine`, `source_frame`, preprocesamiento por polaridad y estado (`OCR_OBSERVED`, `OCR_UNCERTAIN`, `OCR_ABSENT`, `OCR_ERROR`).
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
| **Gate A — Regresión** | Suite M3 reproducida (`31/31`), nuevas pruebas M4 (`+12` → `43/43`), `FLOW`/`FlowGen` intactos, sin acceso de red en tests deterministas | **DEMONSTRATED** | `python3 -m unittest discover -s tests -v`: **43/43 PASS**; `git rev-parse HEAD` y `status` sin cambios en `FLOW` y `FlowGen` |
| **Gate B — OCR y asociación semántica real** | OCR local real sin mocks + `CER`/`WER` medidos + asociación texto↔control (`TEXT_INSIDE`, `LABEL_ASSOCIATED`, `AMBIGUOUS_ASSOCIATION`) + degradación `OCR_ABSENT` explícita y `OCR_UNCERTAIN` | **DEMONSTRATED** | Backend local real `rapidocr-onnxruntime 1.4.4` (PP-OCRv4 ONNX, CPU, 0 red): **`mean_cer = 0.0`**, **`mean_wer = 0.0`**, **`association_precision = 1.0`**, **`association_recall = 1.0`** (`10/10 TP, 0 FP` en `pr_dark` + `pr_light`, controles `Email:`, `Target Branch`, `Merge pull request`, `Save`, `Cancel`). `test_gate_b_real_ocr_no_mocks_pr_dark_light_and_uidiff` + `test_case12_ocr_absent_never_invents_labels` (`backend=tesseract` → `OCR_ABSENT`). |
| **Gate C — Identidad temporal** | Modal nuevo no altera identidad de controles existentes; cambio de color verde→naranja mantiene `track=K_t` y texto `'Merge pull request'`; `MOVED`/`RESIZED`/`TEXT_CHANGED`; negativo lejos → `REMOVED+NEW`; controles idénticos → `AMBIGUOUS_MATCH` | **DEMONSTRATED** | `tests/test_m4_adversarial.py` (Casos 7, 8, 9, 10, 13) + `evidence/m4/benchmark_m4.json`: **`fp_removed_controls = 0`**, `color_changed_detected = 1`, `new_controls_detected = 2` (`K5` modal + `K6` `'Dismiss'`), `case9_move_exact = true`, `case10_negative_far_exact = true`, `case13_ambiguous_detected = true` |
| **Gate D — Text-only blind experiment** | Experimento ciego real con `nvidia/nvidia/nemotron-3.5-lightning-30b-a3b` vía OpenISy sobre Q1..Q6 con OCR real activo (cero acceso a PNG, directorio temporal vacío) | **DEMONSTRATED** | `evidence/m4/m4_blind_nim_transcript.json` (`33.0 s`, `12,169 ch` prompt → `6,263 ch` respuesta): **6/6 PASS** (`Q1='Merge pull request'`, `Q2=[260,520,510,568)/(385,544)`, `Q3=K2 W2->W4 green->orange + INFERRED`, `Q4=K5+K6('Dismiss') + 0 REMOVED`, `Q5=K0 AMBIGUOUS_MATCH`, `Q6=NOT_DEMONSTRATED`). |
| **Gate E — Seguridad y operación** | Cero clics automáticos, `vlm_probe` desactivado por defecto, cero modificaciones a `isymcp-hub`, cero operaciones destructivas, rollback local definido | **DEMONSTRATED** | `test_m4c_vlm_probe_disabled_by_default_and_policy_guards`, `docs/MCP_REGISTRATION.md` (procedimiento reversible), `git status` limpio en `FLOW`/`FlowGen` |

---

## 3. Métricas medidas (§9, fuente: `evidence/m4/benchmark_m4.json`)

| Dimensión | Modo rápido (sin OCR) | Modo enriquecido (OCR real RapidOCR) | Notas |
|---|---:|---:|---|
| Corpus `1280×720` (`pr_dark`) — `layout` | `13.39 ms` | `12.78 ms` | |
| Corpus `1280×720` (`pr_dark`) — `controls` | `11.06 ms` | `10.24 ms` | `mean_bbox_iou_vs_gt = 0.9678` (botones `1.00`, inputs `0.92`) |
| Corpus `1280×720` (`pr_dark`) — `text_bands` | `4.31 ms` | `4.33 ms` | |
| Corpus `1280×720` (`pr_dark`) — `ocr` + `association` | — | `802.26 ms` (`assoc 1.17 ms`) | Primera llamada en frío (warm en `ui_diff`: `~300 ms`) |
| Corpus `1280×720` — total / chars | `36.14 ms` / `3,696 ch` | `838.20 ms` / `4,590 ch` | Separación explícita de costo modo rápido vs modo OCR |
| **OCR real (`pr_dark` + `pr_light`, 10 controles)** | — | **`CER = 0.0`, `WER = 0.0`, `P = 1.0`, `R = 1.0`** (`10/10 TP, 0 FP`) | Confianzas `97.9–99.9` (`rapidocr_0_100_uncalibrated`), 0 mocks |
| `ui_diff` temporal con OCR real (Caso 7+8, modal + color) | — | `684.71 ms` total (`ocr 302.15 ms` + `ui_diff 358.14 ms`), `4,685 ch` | `fp_removed=0`, `color_changed=1` (`K2` `'Merge pull request'`), `new=2` (`K5`, `K6` `'Dismiss'`) |
| Captura real escritorio Linux `1920×1080` (bajo `tracemalloc`) | `layout 22.02 ms`, `controls 34.22 ms`, `text_bands 45.12 ms`, **total `111.88 ms`** (`6,500 ch`) | `ocr 1770.11 ms` (`OCR_OBSERVED`), `assoc 60.29 ms`, **total `1871.35 ms`** | Costo real medido en CPU sin GPU; ~0.53 FPS con OCR 1080p completo vs ~8.9 FPS modo rápido bajo `tracemalloc` |
| Memoria pico Python (`tracemalloc`) | — | **`139.35 MiB`** (`146,116,382 B`) | Incluye modelos ONNX PP-OCRv4 + corpus completo + 1080p real |

---

## 4. Evaluación del experimento ciego con NVIDIA NIM (`m4_blind_nim_transcript.json`, con OCR real)

Modelo: `nvidia/nvidia/nemotron-3.5-lightning-30b-a3b` (100% text-only, vía OpenISy en `cwd` temporal vacío, cero acceso a archivos ni imágenes).
Motor OCR real activo durante el experimento: `rapidocr-onnxruntime installed (PP-OCRv4 ONNX)` (`status=OCR_OBSERVED`).

| Pregunta | Resultado motor FlowSight | Respuesta del LLM | Veredicto y notas de auditoría |
|---|---|---|---|
| **Q1** (texto del botón objetivo) | `W2` / `O2`: `text='Merge pull request'`, `conf=99.9`, `association=TEXT_INSIDE` | Respondió **`'Merge pull request'`** citando `O2`, `W2`, `T2`, `conf=99.9` y `bbox=[286, 531, 485, 557]` | **PASS (`DEMONSTRATED` — OCR real cerrado)** |
| **Q2** (ubicación, bbox y click) | `W2` en `[260, 520, 510, 568)`, `click=(385, 544)`, origen top-left, fin exclusivo | Bbox `[260, 520, 510, 568]` y `click=(385, 544)` exactos + sistema de coordenadas correcto | **PASS (`DEMONSTRATED`)** |
| **Q3** (control que cambió de color + `DEMONSTRATED` vs `INFERRED`) | `track=K2 local_a=W2->local_b=W4 COLOR_CHANGED #16a34a(green)->#ea9418(orange) text='Merge pull request' INFERRED=['STATE_CHANGED_CANDIDATE']` | Citó `track=K2`, `W2->W4`, colores exactos, texto `'Merge pull request'` y separó `COLOR_CHANGED` (`DEMONSTRATED`) de `STATE_CHANGED_CANDIDATE` (`INFERRED`) | **PASS (`DEMONSTRATED` — bug M3 eliminado)** *(nota menor de prosa LLM: escribió `99,9/99,8` en vez de `99.9/99.9`)* |
| **Q4** (`NEW_CONTROL` sin alterar tracks + `REMOVED_CONTROL`) | `K5` (`W0`, modal `#341822`) y `K6` (`W3`, botón `#d946ef`, texto `'Dismiss'`), `0 REMOVED_CONTROL` | Reportó `K5`, `K6` con `'Dismiss'` y `0 REMOVED_CONTROL` exactos | **PASS (`DEMONSTRATED`)** |
| **Q5** (ambigüedades) | Escena 1: `0`; Escena 2: `track=K0 AMBIGUOUS_MATCH score=0.9213` | Reportó exacto `0` en Escena 1 y `track=K0 AMBIGUOUS_MATCH score=0.9213` + competing candidate en Escena 2 | **PASS (`DEMONSTRATED`)** *(nota menor de prosa LLM: en comentario final atribuyó `10/10 panels` a Escena 2 en vez de Escena 1 Frame B)* |
| **Q6** (¿demuestra que funciona y es seguro pulsarlo?) | `geometric candidates — appearance != verified interactivity` | Respondió **`NOT_DEMONSTRATED`** citando el contrato explícito | **PASS (`DEMONSTRATED` calibración epistémica)** |

---

## 5. Clasificación consolidada & Rollback

- **DEMONSTRATED:**
  - **OCR local real (`rapidocr-onnxruntime 1.4.4`, CPU, sin red, sin mocks)** con `CER=0.0`, `WER=0.0` y asociación geométrica `P=1.0 / R=1.0` (`10/10` controles en dark y light) + degradación fail-closed `OCR_ABSENT` verificada cuando un backend falta (`backend=tesseract`).
  - `ui_diff` basado en identidad temporal con tracks persistentes `K0..Kn`, eliminación del bug M3 de desplazamiento de IDs (`0` falsos `REMOVED_CONTROL`), preservación de ambos frames, preservación de texto OCR (`'Merge pull request'`) a través del cambio de color y detección de `AMBIGUOUS_MATCH`.
  - Contrato M4-C `vlm_probe` desactivado por defecto con guardas de política.
  - Retrocompatibilidad de las 7 tools MCP + nueva `flowsight_screen_query` (`43/43` tests PASS).
  - Experimento ciego real Q1..Q6 (6/6 PASS) con `nvidia/nvidia/nemotron-3.5-lightning-30b-a3b` leyendo OCR real.
- **INFERRED:**
  - `STATE_CHANGED_CANDIDATE` en `ui_diff` (sin señal de accesibilidad).
  - Etiquetas posicionales (`top-bar-band`, `main-pane`) y naturaleza interactiva de controles candidatos.
- **NOT_DEMONSTRATED:**
  - Escalación VLM viva a un modelo visual remoto/local (contrato preparado y desactivado por defecto).
  - Ejecución con `tesseract` en este host específico (`tesseract` no instalado en `/usr/bin`; código de adaptador presente y degradación `OCR_ABSENT` probada; OCR real cubierto por `rapidocr-onnxruntime`).
  - Registro activo en `isymcp-hub` (pendiente autorización explícita de Danny; ver [`../docs/MCP_REGISTRATION.md`](../docs/MCP_REGISTRATION.md)).

### Rollback local
Para revertir M4 al baseline M3 intacto sin tocar `FLOW` ni `FlowGen`:
```bash
git -C "/home/danny/Development/ISyCo Git/FlowSight" checkout ef94e09
```
