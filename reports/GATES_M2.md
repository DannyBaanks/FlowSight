# FlowSight M2 — Reporte de Gates: Bloques 1920×1080, Stitching, Servidor MCP y Experimento Ciego NVIDIA NIM

Fecha: 2026-11-24.
Repositorio independiente: `/home/danny/Development/ISyCo Git/FlowSight`.
`FLOW` (`3e13909`) y `FlowGen` (`0dd6aa2`): **intactos (solo lectura)**.
`isymcp-hub` (daemon global): **no modificado** (snippet de registro incluido en `README.md` para activación explícita por Danny).

---

## 1. Capacidades M2 implementadas

- **[`../flowsight_blocks.py`](../flowsight_blocks.py)** (`flowsight-blocks-v1`, explícitamente **APPROXIMATE QUANTIZED**, nunca lossless):
  - Parte imágenes de alta resolución (PNG/JPG/WEBP hasta 8192px, probado en `1920×1080` y foto real `960×1280`) en **cuartos (`quarters`: `TL, TR, BL, BR`)** o **mitades (`halves_h`: `L, R`; `halves_v`: `T, B`)**, con **subdivisión recursiva** (quadtree zoom) preservando coordenadas globales exactas `[x0, y0, x1, y1)`.
  - Paleta global compartida (`median-cut` + refinamiento Lloyd k-means determinista de 4 pasadas en `int32`), métricas reales de error (`MAE_RGB`, `RMSE_RGB`, `PSNR_dB`, `within_delta24_pct`), mapa de color + **mapa de densidad/orientación de bordes Sobel** por bloque y regiones locales proyectadas a bbox global.
  - **Unión estructural y semántica (`stitch_blocks`)**: mide continuidad píxel a píxel sobre las fronteras reales a resolución completa (`x=W/2`, `y=H/2`), fusiona regiones trans-frontera vía Union-Find (`G0..Gn [CROSS-SEAM]`), re-ensambla mapas globales con costuras marcadas y alinea notas textuales por bloque (`AGENT_NOTES / INFERRED`) separándolas estrictamente de las mediciones (`DEMONSTRATED`).
- **[`../flowsight_mcp.py`](../flowsight_mcp.py)** (servidor MCP stdio en `mcp 2.x` `MCPServer`):
  - Expone 5 tools: `flowsight_map_blocks`, `flowsight_get_block`, `flowsight_subdivide_block`, `flowsight_stitch_blocks` y `flowsight_inspect_lossless` (acceso directo al motor lossless M1 `--mode flow` / `--mode visual`).

---

## 2. Gates M2 verificados

| Gate | Clasificación | Evidencia |
|---|---|---|
| Partición `1920×1080` en cuartos (`960×540`) y mitades (`960×1080` / `1920×540`) + subdivisión recursiva (`480×270`) | **DEMONSTRATED** | `tests/test_blocks_and_mcp.py::test_1920x1080_quarters_and_halves_and_subdivide` (`217.6 ms`, `MAE_RGB=5.9`, `PSNR=26.63 dB`, `within_d24=96.7%`) |
| Unión trans-frontera (`stitch_blocks`) midiendo costuras reales + fusión Union-Find | **DEMONSTRATED** | `test_stitch_detects_cross_seam_regions_and_aligns_notes`: 4 costuras al `100.0%`, fusión del bloque rojo central en `G6 [CROSS-SEAM] spans=[TL,TR,BL,BR] bbox=[760,338,1160,742]` (ground truth `[760,340,1160,740)` = error ≤2 px sobre 1080) |
| Determinismo de mapeo y stitching | **DEMONSTRATED** | `test_blocks_determinism` PASS |
| Servidor MCP real sobre stdio (`mcp 2.x`) E2E con las 5 tools | **DEMONSTRATED** | `tests/test_blocks_and_mcp.py::test_mcp_stdio_roundtrip` PASS (cliente `mcp.client.stdio` real ejecutando `list_tools` + las 5 tools, incluyendo `vortex.png` lossless) |
| Mapeo y stitching sobre foto real de disco (`5163877233851043130.jpg`, `960×1280`) | **DEMONSTRATED** | `219.2 ms`, `K=12`, `MAE_RGB=3.81`, `PSNR=33.28 dB`, `within_d24=99.7%`, `stitch_report=3,165 ch`, `3` regiones `CROSS-SEAM` |
| Experimento ciego en vivo con modelo 100% text-only de **NVIDIA NIM** vía OpenISy (`nvidia/nvidia/nemotron-3.5-lightning-30b-a3b`) | **DEMONSTRATED** (geometría, color, costuras y calibración epistémica) | `examples/experiment_nim_blocks.py` + transcript completo en `evidence/m2_blocks/nim_1080p_transcript.json` |

---

## 3. Experimento ciego en vivo con NVIDIA NIM (`nvidia/nvidia/nemotron-3.5-lightning-30b-a3b`)

Cada invocación de `openisy run -m nvidia/nvidia/nemotron-3.5-lightning-30b-a3b` corrió desde un directorio temporal vacío (`tempfile.TemporaryDirectory`) sin acceso ni ruta a la imagen `1920×1080`.

- **Etapa 1 (4 análisis aislados de `960×540`):**
  - `TL`: `14.1 s` (`2,767 ch` prompt → `762 ch` nota)
  - `TR`: `15.4 s` (`2,657 ch` prompt → `836 ch` nota)
  - `BL`: `14.1 s` (`2,628 ch` prompt → `945 ch` nota)
  - `BR`: `24.0 s` (`2,787 ch` prompt → `807 ch` nota)
- **Etapa 2 (`flowsight_stitch_blocks`):** unió los 4 bloques + notas + costuras en un informe compacto.
- **Etapa 3 (síntesis global en `nemotron-3.5-lightning-30b-a3b`, `26.1 s`, `8,291 ch` prompt → `1,604 ch` respuesta):**
  - **Q1 (bandas horizontales `1920×1080`):** exacta (`G2/G3/G1` azul `y=0..424`, cian en `y=540`, `G0` verde `y=704..1080`).
  - **Q2 (objeto central cruzando los 4 cuadrantes):** exacta (`G6 [CROSS-SEAM] spans=[TL,TR,BL,BR]`, rojo `#c3262e`, `bbox=[760,338,1160,742]` vs ground truth `[760,340,1160,740)`).
  - **Q3 (forma amarilla en `TL`):** exacta (`G7`, `pale-yellow`, `bbox=[210,116,430,328]` vs ground truth `[210,110,430,330]`).
  - **Q4 (barra horizontal fuerte en `BR`):** localizó exactamente los bordes fuertes `#######` en fila 12 de `BR` (`G10 [1280,820,1780,878]` con `edges=36.7%`), pero señaló honestamente que la orientación dominante por bloque no venía impresa en el informe unido. *Fallo acotado detectado y corregido de inmediato en `flowsight_blocks.py` añadiendo la sección `0. PER-BLOCK MEASURED STATS (DEMONSTRATED)` con `orient` y `edges` por bloque.*
  - **Q5 (continuidad de las 4 costuras):** exacta (`100.0%` en `TL<->TR`, `BL<->BR`, `TL<->BL`, `TR<->BR`).
  - **Q6 (calibración epistémica — "foto tomada en París con DSLR"):** exacta (**`NOT_DEMONSTRATED`**).

---

## 4. Clasificación epistémica estricta (M2)

- **DEMONSTRATED:**
  - Partición determinista de imágenes `1920×1080` (y fotos reales) en mitades/cuartos con paleta global compartida y métricas reales de error (`MAE`, `RMSE`, `PSNR`).
  - Detección determinista de continuidad trans-frontera y fusión Union-Find de regiones entre cuadrantes.
  - Exposición y ejecución E2E sobre protocolo MCP 2.x stdio (`flowsight_mcp.py`).
  - Observación multi-bloque y síntesis global por un modelo **realmente solo-texto** de NVIDIA NIM (`nemotron-3.5-lightning-30b-a3b`) con calibración epistémica correcta.
- **INFERRED:**
  - Las notas en lenguaje natural generadas por cada bloque (`block_notes`) y cualquier hipótesis semántica sobre qué objeto del mundo real representa una mancha de color o contorno.
- **NOT_DEMONSTRATED:**
  - Reconocimiento semántico fino en fotografías complejas (caras, texto pequeño/OCR, escenas densas con decenas de objetos) sin subdivisión recursiva adicional o un extractor OCR explícito.
  - Registro activo en el daemon global `isymcp-hub` (pendiente de autorización explícita de Danny, porque reiniciar el hub desconecta las TUIs abiertas).
