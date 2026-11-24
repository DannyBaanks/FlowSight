# FlowSight M1 — Reporte de Gates, Evidencia y Experimentos Ciegos

Fecha: 2026-11-24.
Repositorio independiente: `/home/danny/Development/ISyCo Git/FlowSight`.
`FLOW` (`3e13909`) y `FlowGen` (`0dd6aa2`): **intactos (solo lectura)**.
Línea base **M0** y toda su evidencia original: preservadas en [`../evidence/m0/`](../evidence/m0/) y en el commit `1e7ec93`.

---

## 1. Prioridad 1 — Índice nativo `--mode flow` (`flowsight-flow-v1`)

### Gates obligatorios

| # | Gate | Clasificación | Evidencia directa |
|---|---|---|---|
| G1 | Indexar `FLOW/examples/vortex.png` (873 RGB) y `hello_flow.png` (5 RGB) sin rechazar por >64 colores | **DEMONSTRATED** | `tests/test_flow_mode.py::test_gate1_gate2_gate3_gate5_vortex_and_hello`, `evidence/m1_p1/benchmark_m1_p1.json` |
| G2 | Verificar canales `R`, `G`, `B` y `RGBA` contra el PNG original (0 error de reconstrucción, sin cuantizar `R/G`) | **DEMONSTRATED** | `verify` PASS en ambos (`r_sha256`, `g_sha256`, `b_sha256`, `rgb_sha256`), `rebuilt_rgb_exact=true`, detección independiente de corrupción por canal (`test_channel_corruption_detected_independently`) |
| G3 | Consultas regionales exactas (`point`, `scanline`, `region`, `field`, `instructions`, `find`, `components`, `edges`, `diff`) | **DEMONSTRATED** | Verificación contra ground truth PIL (`test_flow_mode.py`), divergencia/rotacional en diferencias finitas exactas y acoplamiento espacial instrucción → vector local → celda siguiente |
| G4 | Invarianza de la VM real de `FLOW` y sus trazas (sin modificar `FLOW` ni `FlowGen`) | **DEMONSTRATED** | `test_gate4_vm_trace_invariance` PASS: trazas de `flow.runtime.flow_trace.run_program` (seed=42) idénticas en ticks, eventos y conteos entre PNG original y PNG reconstruido (`hello_flow`: `768ea223...`, `vortex`: `cbcf6794...`); `git rev-parse HEAD` y `git status` sin cambios en `FLOW` y `FlowGen` |
| G5 | Determinismo byte a byte en ejecuciones repetidas | **DEMONSTRATED** | `sqlite_deterministic=true` (mismo SHA-256 del archivo `.sqlite` en dos corridas), `L0`, `field` e `instructions` idénticos (`test_determinism_byte_exact`, `test_gate1_gate2_gate3_gate5_vortex_and_hello`) |
| G6 | Medición de tamaño de índice, caracteres/tokens entregados al modelo y tiempos | **DEMONSTRATED** | Tabla abajo + `evidence/m1_p1/benchmark_m1_p1.json` (estimado `chars/4` marcado explícitamente como estimado, no tokenizer facturado) |

### Métricas medidas (Gate 6)

| Programa | Tamaño | RGB / R / G / B distintos | Spawns | PNG | SQLite | Naive JSON (RGB/px) | L0 chars (est. `ch/4`) | Reducción L0 vs Naive | Tiempo index / L0 | Exact RGB & VM Eq |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `hello_flow.png` | 32×16 | 5 / 2 / 1 / 4 | 1 | 150 B | 77,824 B | 24,098 B | 1,606 ch (~401.5 tok) | ~15× | 27.6 ms / 0.85 ms | `True` / `True` |
| `vortex.png` | 64×64 | 873 / 249 / 249 / 2 | 8 | 2,000 B | 90,112 B | 190,837 B | 2,241 ch (~560.2 tok) | ~85× | 20.2 ms / 5.41 ms | `True` / `True` |

*(Nota: al igual que en M0, SQLite pesa más en disco que el PNG comprimido; el ahorro real está en el texto entregado al contexto del agente frente a volcar píxeles en JSON).*

---

## 2. Prioridad 2 — Mejoras de fóvea visual (`--mode visual`)

Implementadas únicamente después de cerrar todos los gates de Prioridad 1 (`tests/test_visual_p2.py`, 4/4 PASS; suite completa 23/23 PASS incluyendo los 13 tests intactos de M0):

| Capacidad | Clasificación | Evidencia |
|---|---|---|
| `region X Y W H --grid` (matriz simbólica 1:1 con límite fail-closed ≤64×64 / ≤1024 px) | **DEMONSTRATED** | `test_region_grid_exact_and_fovea_limit` + uso efectivo en M1-C (`visual_p2`) y M1-B (`hello_flow`) |
| Marcadores `[hidden-in-minimap]` + sección de colores minoritarios con `bbox` exacto en `summary()` | **DEMONSTRATED** | `test_minority_color_marker_in_summary` + M1-C V1 resuelta en 0 consultas adicionales |
| `edges --region X Y W H` (segmentos `V`/`H` acotados a caja) | **DEMONSTRATED** | `test_edges_scoped_by_region` + M1-C V2 (22 segmentos exactos sin truncamiento) |
| Transiciones de color exactas en `diff` (`C_old(#hex) -> C_new(#hex)`) | **DEMONSTRATED** | `test_diff_reports_exact_color_transitions` + M1-C V3 (6/6 píxeles y 2 pares exactos) |

---

## 3. Experimentos ciegos M1 (agentes solo-texto, sin acceso a PNG ni archivos)

Todos los subagentes corrieron con `blind-agent` (`Flash`) sin acceso a rutas de archivo y fueron auditados mediante `transcript.jsonl` (`evidence/m1_blind/*_transcript.jsonl`): **todos los `tool_calls` registrados fueron exclusivamente `send_message`** (0 `read_file`, 0 `run_command`, 0 accesos a PNG).

| Experimento | Target | Consultas / Chars | Auditoría ceguera (`tool_calls`) | Resultado | Clasificación |
|---|---|---|---|---|---|
| **M1-A (intento 1 — control negativo de race)** | `FLOW/examples/vortex.png` (`--mode flow`) | 6/6 (3,237 ch) | Solo `send_message` (0 archivos) | F2, F3, F4, F5 correctas; en **F1** el subagente emitió `FINAL:` antes de leer la respuesta de su 6ta consulta y alucinó un grid 3×3 `(128,128)` inexistente. Transcript preservado en `m1_a_vortex_transcript.jsonl`. | **Fallo operacional del protocolo de turnos (registrado)** |
| **M1-A (repetición limpia)** | `FLOW/examples/vortex.png` (`--mode flow`) | 6/6 (3,367 ch) | Solo `send_message` (0 archivos) | **5/5 exactas**: campo rotacional CW (`mean_curl=+0.11357`, `mean_div=-0.00002`), las 5 coordenadas exactas de la cruz nula `(32,31),(31,32),(32,32),(33,32),(32,33)`, 8 spawns tangenciales → `B=0(HALT)`, ausencia de `SPLIT`/`TRACE`, valores RGB exactos en `(32,32)` y `(32,29)`, y **F5 = `NO DEMOSTRADO`** (sin ejecutar VM). Transcript: `m1_a_vortex_clean_transcript.jsonl`. | **DEMONSTRATED** (estructura espacial/vectorial y calibración epistémica) |
| **M1-B** | `FLOW/examples/hello_flow.png` (`--mode flow`) | 6/6 (2,155 ch) | Solo `send_message` (0 archivos) | **4/4 exactas**: cadena completa `(2,2) SPAWN → 5×INC → (8,2) WRITE → (9,2) HALT`, rango activo exacto `x=[2,11)` (`R=255`), valores RGB exactos en `(8..11,2)`, ausencia de saltos/giros/splits, rechazo geométrico de que dibuje las letras `"HELLO"`. Transcript: `m1_b_hello_flow_transcript.jsonl`. | **DEMONSTRATED** (estructura lineal de opcodes/campo y calibración) |
| **M1-C** | `visual_p2.png` (`64×64`, `--mode visual` P2) | 3/6 (1,406 ch) | Solo `send_message` (0 archivos) | **4/4 exactas**: detectó los 2 píxeles minoritarios ocultos `(12,14)` y `(50,50)` directo desde L0; verificó oclusión (`C2` encima de `C1`) y grosor exacto `2px` usando `region --grid` + `edges --region`; reportó las 2 transiciones exactas de `diff`; respondió **`NO DEMOSTRADO`** a "plano arquitectónico de una casa". Transcript: `m1_c_visual_p2_transcript.jsonl`. | **DEMONSTRATED** (geometría/color/oclusión/diff y calibración) |

---

## 4. Clasificación epistémica estricta

- **DEMONSTRATED:**
  - Lectura, indexación lossless 3 canales, reconstrucción RGB exacta (0 error) y preservación de trazas de `FlowVM` para `vortex.png` (873 colores RGB) y `hello_flow.png`.
  - Comprensión estructural de programas FLOW (topología del campo rotacional/lineal, zona nula, disposición tangencial de spawns, cadena de opcodes en `B` y verificación de ausencia de `SPLIT`/`TRACE`) por un agente solo-texto acotado a ~1.4–3.4 KB de texto.
  - Fóvea visual P2 (`region --grid`, colores minoritarios en L0, `edges --region`, transiciones de color en `diff`).
  - Rechazo calibrado de afirmaciones semánticas o dinámicas no respaldadas por datos estáticos (`NO DEMOSTRADO` ante "plano arquitectónico" y ante "cuántas partículas vivas a 500 ticks sin ejecutar la VM").
- **INFERRED:**
  - Aproximaciones de tokens `chars / 4` (no medidas con un tokenizer específico de proveedor).
  - En `vortex.png`, el sentido de giro a partir de `mean_curl` y cuadrantes es matemática exacta del campo estático; inferir el comportamiento dinámico real de partículas a `t=500` sigue requiriendo ejecutar `FlowVM`.
- **NOT_DEMONSTRATED:**
  - Reconocimiento semántico de objetos del mundo real en fotografías o escenas naturales (ningún test demostró visión perceptual humana; solo geometría, color y estructura de campo/opcodes).
  - Imágenes visuales `>64` colores o `>512×512` (screenshots complejos, UI con antialiasing, fotos) — continúan rechazadas fail-closed en `--mode visual` hasta un futuro modo cuantizado explícito con métricas de error.
  - Desempeño sobre modelos text-only pequeños/locales (7B–8B); los experimentos ciegos actuales se ejecutaron con `Flash` sin acceso a archivos/imágenes.
  - Exposición como servidor MCP en `isymcp-hub` (fuera del alcance de M1).
