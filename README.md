# FlowSight — Percepción visual estructurada, componible y verificable para agentes solo-texto

**«No dije que quisiera que fuera lo mismo. Quiero que funcione».**

**FlowSight** es un periférico local, modular y de solo lectura (CLI + servidor MCP stdio con **14 tools**) que permite a modelos **100 % solo-texto** inspeccionar capturas de pantalla, secuencias temporales, videos de interfaces gráficas, imágenes de alta resolución (`1920×1080`) y programas visuales de [FLOW](https://github.com/DannyBaanks/FLOW) sin volcar JSONs gigantes por píxel y sin depender de visión multimodal nativa.

> **Principio epistémico:** FlowSight **no** finge visión humana ni reemplaza medidas exactas con conjeturas. Toda salida separa explícitamente hechos medidos (`DEMONSTRATED`: geometría, coordenadas globales, colores, OCR observado, hashes) de hipótesis (`INFERRED`: p. ej. `STATE_CHANGED_CANDIDATE`, `VLM_INFERRED`) y límites (`NOT_DEMONSTRATED`). **Nunca ejecuta clics** y mantiene `FLOW` y `FlowGen` completamente intactos.

```text
capturas / video ─▶ filtros (-vf) ─▶ índice SQLite+FTS5 (TTL) ─▶ búsqueda (top-K + bbox) ─▶ crop exacto ─▶ respuesta
```

---

## Capacidades actuales (M0 → M5)

### 1. Motor de filtros componibles estilo `ffmpeg` para capturas (`flowsight_screen.py`)
Pipeline determinista `-vf "f1=...,f2,..."` (OpenCV + NumPy + SciPy + Pillow) con **10 filtros declarativos** (`--filters`), coordenadas globales `[x0, y0, x1, y1)` exclusivas y centros `click=(cx, cy)`:

- **`crop=x:y:w:h`** — zoom a una subregión preservando coordenadas globales exactas.
- **`theme`** — polaridad (`dark`/`light`/`mixed`), luminancia mediana, superficies dominantes y colores acento.
- **`layout`** — divisores estructurales `H`/`V` y paneles `P0..Pn` con `area_pct`, fondo y pista posicional.
- **`controls`** — detección geométrica de controles candidatos `W0..Wn` (botones acento, inputs, badges, tarjetas/modales) con `click=(cx, cy)`.
- **`text_bands`** — detección geométrica de bandas de texto `T0..Tn` (sin OCR).
- **`ocr`** ([`flowsight_ocr.py`](flowsight_ocr.py)) — **OCR local real en CPU sin red** (`backend=auto|zig|rapidocr|tesseract`), con confianza y escala documentada (`OCR_OBSERVED` / `OCR_UNCERTAIN` / `OCR_ABSENT` fail-closed) + **asociación geométrica texto ↔ controles** (`TEXT_INSIDE`, `LABEL_ASSOCIATED`, `AMBIGUOUS_ASSOCIATION`, `NO_TEXT_EVIDENCE`) y métricas exactas `CER`/`WER`:
  - **`zig`** ([`zig_ocr/`](zig_ocr/)) — **Zig 0.16 SIMD Shape OCR** nativo para texto de interfaz de escritorio: atlas binario `FSOA v3` (`zig_ocr/build_atlas.py`), filtro grueso `4×8` → fino `12×16` SoA `i8` SIMD con workers paralelos por bandas, umbralización adaptativa por tile, separación por valles para glifos en contacto, compuerta de nitidez de bordes contra texturas suaves (cero alucinaciones en degradados/fotos) y worker persistente binario `--serve` (`FSQ1`).
  - **`rapidocr` / `tesseract`** — backends alternativos (`rapidocr-onnxruntime` PP-OCRv4 ONNX o `tesseract` del sistema).
- **Preset `--one-shot` y estructura de chrome superior** (`flowsight_screen.py`) — inspección de pantalla completa en **una sola llamada** (`theme,layout,controls,ocr(zig si está compilado),ui_map`, `max_chars>=16000`, `ocr_limit>=150`), con detección `TOP CHROME STRUCTURE` (distingue barra de título de ventana vs barra de pestañas) y marcado explícito de `GRAPHIC REGIONS (NO READABLE TEXT)` (`NOT_DEMONSTRATED`) para evitar conjeturas en zonas puramente gráficas.
- **`ui_map`** — mapa wireframe ASCII compacto de divisores, controles y texto.
- **`ui_diff=before.png`** ([`flowsight_diff.py`](flowsight_diff.py)) — **diff basado en identidad temporal** con asignación global óptima (`scipy.optimize.linear_sum_assignment`), tracks persistentes `track=K0..Kn` separados de los índices locales `W_i`, preservación de ambos frames y 9 clasificaciones (`NEW_CONTROL`, `REMOVED_CONTROL`, `MOVED_CONTROL`, `RESIZED_CONTROL`, `TEXT_CHANGED`, `COLOR_CHANGED`, `STATE_CHANGED_CANDIDATE` [`INFERRED`], `UNCHANGED_CONTROL`, `AMBIGUOUS_MATCH`).
- **`vlm_probe`** ([`flowsight_vlm.py`](flowsight_vlm.py)) — escalación opcional a un VLM **apagada por defecto (fail-closed)**, estrictamente limitada a recortes (`crop=x:y:w:h` obligatorio, `<= 1024 px/lado`, `<= 512×512 px`), loopback por defecto (`allow_remote=False`), adaptador OpenAI-compatible en CLI (`--enable-vlm`) y MCP (`FLOWSIGHT_VLM_ENABLED=1`), salida siempre marcada `VLM_INFERRED`.
- **`blocks`** — acopla el particionado en mitades/cuartos y costuras del motor M2.

### 2. Búsqueda visual progresiva con geometría verificable (`flowsight_index.py`, M5-A)
Evita volcar múltiples capturas completas al contexto del LLM:
- **Indexado explícito y efímero:** SQLite + **FTS5** (`unicode61` preservando tokens técnicos `-_./:` como `3128/tcp` o `br-575398bb8361`), guardado fuera de git en `~/.local/share/flowsight/progressive_index.sqlite` (`0700`/`0600`), **0 bytes de píxeles almacenados**, deduplicación por `SHA-256` y ciclo de vida explícito (`--ttl-hours`, `expire`, `purge`). Nunca escanea carpetas en segundo plano.
- **Ranking + recorte listo para profundizar:** `search` filtra por texto, `theme`, `color` y fechas, devuelve top-K con fragmentos OCR, `bbox` global exacto y `deepen_vf="crop=x:y:w:h,controls,ocr"`.
- **Ahorro medido** ([`reports/GATES_M5.md`](reports/GATES_M5.md)): en el benchmark de 7 capturas `1280×720` (`query="UFW 3128"`), pasó de `21,931 chars` (7 reportes completos) a `2,037 chars` (`search + crop` ganador) — **`90.71 %` menos contexto** y re-indexado cacheado en `1.90 ms`.

### 3. Percepción temporal sobre videos y secuencias (`flowsight_timeline.py`, M5-B)
Extiende la analogía con FFmpeg a secuencias ordenadas de capturas y videos `.mp4/.mkv/.webm/.avi` (`cv2.VideoCapture`):
- **Salto de fotogramas estáticos por bloques (`8×6` tiles):** ignora frames repetidos/ruido de compresión pero detecta cambios localizados (~1 % de pantalla, como un solo botón cambiando de color o un banner de error).
- **Eventos con timestamp y `bbox` global:** encadena `track=K0..Kn` entre keyframes y emite `CONTROL_APPEARED`, `CONTROL_DISAPPEARED`, `CONTROL_COLOR_CHANGED` (+ `INFERRED=['STATE_CHANGED_CANDIDATE']`), `CONTROL_MOVED`, `CONTROL_RESIZED`, `CONTROL_TEXT_CHANGED`, `TEXT_APPEARED` y `TEXT_DISAPPEARED`, filtrables con `--query`.

### 4. Partición por bloques `1920×1080` y unión estructural (`flowsight_blocks.py`, M2)
Para fotografías e imágenes grandes (>64 colores, hasta `8192 px`):
- Partición en mitades (`halves_h`, `halves_v`) o cuartos (`quarters`: `TL/TR/BL/BR`) + **subdivisión recursiva (quadtree)** con paleta compartida cuantizada (`median-cut` + Lloyd k-means determinista) y métricas reales de error (`MAE_RGB`, `RMSE_RGB`, `PSNR_dB`).
- Mapa de color + mapa de bordes Sobel por bloque y `stitch_blocks` que mide continuidad real píxel a píxel en costuras, fusiona regiones `CROSS-SEAM` (`G0..Gn`) y alinea notas del agente.

### 5. Modo lossless SQLite para programas FLOW y pixel art (`flowsight.py` + `flowsight_flow.py`, M0/M1)
- **`--mode flow` (`flowsight-flow-v1`):** índice nativo de 3 canales sin pérdida ni cuantización para programas de `FLOW` (soporta >64 colores RGB como `vortex.png` y `hello_flow.png`): `R/G` campo vectorial continuo exacto (magnitud, coherencia, divergencia, curl) y `B` instrucciones/datos discretos en RLE exacto + `spawns` (`B=255`). Reconstrucción byte a byte con verificación SHA-256 e invariancia demostrada de trazas en la VM real de `FLOW`.
- **`--mode visual` (`flowsight-rle-v1`):** raster discreto lossless (≤64 colores RGBA exactos) con `summary` L0, marcadores `[hidden-in-minimap]` para colores minoritarios, `region --grid` 1:1, `edges --region`, `diff` con transiciones de color y `export-flowpaint`.

### 6. Corpus privado de fallos por PC (`flowsight_corpus.py`) e integración con ISyCode TUI
- Registra fallos de percepción confirmados por el usuario (p. ej. tras confirmación vía `ask_user`) en `~/.local/share/flowsight/private_failures/` (`0700`/`0600`, `machine_id` UUID4 aleatorio único por PC, fuera de git y sin copiar capturas salvo `--save-local-copy`).
- Integrado con el fallback para modelos sin visión de [ISyCode](https://github.com/DannyBaanks/ISyCode) ([PR #123](https://github.com/DannyBaanks/ISyCode/pull/123)): al pegar una imagen en un modelo sin visión nativa, el agente inspecciona con FlowSight, explica `DEMONSTRATED` vs `INFERRED`, confirma con `ask_user` y registra errores reales en el corpus privado local.

---

## Servidor MCP (`flowsight_mcp.py` — 14 tools stdio)

Expone **14 tools** (solo lectura sobre imágenes/videos; compatible con `mcp 2.x` y configuración local `~/.config/isycode/mcp.json` o `isymcp-hub` — ver guía detallada en [`docs/MCP_REGISTRATION.md`](docs/MCP_REGISTRATION.md)):

| Grupo | Tools |
|---|---|
| **Filtros de pantalla (M3/M4)** | `flowsight_screen_filters`, `flowsight_screen_run`, `flowsight_screen_query` |
| **Índice progresivo (M5-A)** | `flowsight_index_add`, `flowsight_index_search`, `flowsight_index_purge` |
| **Timeline / Video (M5-B)** | `flowsight_timeline_scan` |
| **Bloques 1920×1080 (M2)** | `flowsight_map_blocks`, `flowsight_get_block`, `flowsight_subdivide_block`, `flowsight_stitch_blocks` |
| **Lossless FLOW / Visual (M1)** | `flowsight_inspect_lossless` |
| **Corpus privado de fallos** | `flowsight_log_failure`, `flowsight_list_failures` |

---

## Instalación rápida y uso CLI

Requisitos: Python ≥3.10, `Pillow`, `numpy`, `opencv-python`, `scipy`, `mcp` (para el servidor MCP); opcionalmente **Zig 0.16** para compilar el backend nativo `zig_ocr` (recomendado para `--one-shot`), o `rapidocr-onnxruntime` / `tesseract` para OCR alternativo.

```bash
python3 -m pip install -r requirements.txt

# --- (Opcional, recomendado) Compilar backend nativo Zig 0.16 Shape OCR ---
(cd zig_ocr && zig build -Doptimize=ReleaseFast && python3 build_atlas.py)

# --- Capturas de pantalla estilo ffmpeg (M3/M4) y preset --one-shot ---
python3 flowsight_screen.py --filters
python3 flowsight_screen.py -i screen.png --one-shot
python3 flowsight_screen.py -i screen.png -vf "theme,layout,controls,text_bands,ocr,ui_map=48:20" --ocr-backend zig
python3 flowsight_screen.py -i after.png  -vf "layout,controls,ocr,ui_diff=before.png"

# Escalación VLM opt-in SOLO sobre recorte explícito (apagada por defecto)
python3 flowsight_screen.py -i screen.png \
  -vf "crop=250:510:270:68,vlm_probe=question=Describe el icono y estado visual" \
  --enable-vlm --vlm-url http://127.0.0.1:11434/v1/chat/completions --vlm-model qwen2.5vl

# --- Búsqueda visual progresiva (M5-A) ---
python3 flowsight_index.py add ./capturas/ --ttl-hours 24
python3 flowsight_index.py search "UFW 3128" --theme dark --limit 3
python3 flowsight_index.py purge --mode expired

# --- Timeline sobre video o secuencia de capturas (M5-B) ---
python3 flowsight_timeline.py grabacion.mp4 --fps 2 --query "exit=56"
python3 flowsight_timeline.py frame1.png frame2.png frame3.png

# --- Bloques 1920x1080 + stitch (M2) ---
python3 flowsight_blocks.py map foto_1080p.png --split quarters --colors 12
python3 flowsight_blocks.py block <session_id> TL
python3 flowsight_blocks.py stitch <session_id>

# --- Programas FLOW y pixel art lossless (M1) ---
python3 flowsight.py index /ruta/a/FLOW/examples/vortex.png vortex.sqlite --mode flow
python3 flowsight.py summary vortex.sqlite
python3 flowsight.py field vortex.sqlite --region 16 16 32 32 --cols 8 --rows 8
python3 flowsight.py verify vortex.sqlite /ruta/a/FLOW/examples/vortex.png

# --- Corpus privado de fallos local por PC ---
python3 flowsight_corpus.py list
python3 flowsight_corpus.py stats

# --- Servidor MCP stdio ---
python3 flowsight_mcp.py
```

### Ejecutar pruebas y evaluación de OCR

```bash
# Tests unitarios del motor nativo Zig OCR
(cd zig_ocr && zig build test)

# Suite Python (pytest o unittest; con FLOW en PYTHONPATH se ejecutan también los gates nativos de VM)
pytest -q tests
PYTHONPATH="/ruta/a/FLOW:." python3 -m unittest discover -s tests -v

# Evaluación sintética de CER y rechazo de texturas del backend Zig OCR
python3 bench/zig_ocr_eval.py -v
```

---

## Garantías de seguridad, privacidad y coordenadas

- **Coordenadas:** origen `(0, 0)` en esquina superior izquierda; `x` hacia la derecha, `y` hacia abajo; cajas `[x0, y0, x1, y1)` con extremo final **exclusivo**.
- **Sin acciones destructivas ni clics automáticos:** FlowSight informa coordenadas y evidencia; nunca hace clic ni modifica imágenes originales.
- **Privacidad local por defecto:** tanto el índice progresivo (`~/.local/share/flowsight/progressive_index.sqlite`) como el corpus privado de fallos (`~/.local/share/flowsight/private_failures/`) viven fuera de git con permisos `0700`/`0600` y jamás copian capturas privadas sin consentimiento explícito.
- **Límites explícitos (`NOT_DEMONSTRATED`):** no reemplaza visión multimodal nativa universal sobre escenas naturales del mundo físico, no verifica interactividad real solo por apariencia visual, no ejecuta la VM dinámica de `FLOW` y mantiene `vlm_probe` apagado por defecto.

---

## Evidencia y reportes por milestone

| Milestone | Reporte de gates | Evidencia hasheada |
|---|---|---|
| **M0** (PoC raster lossless) | [`evidence/m0/GATES.md`](evidence/m0/GATES.md) | [`evidence/m0/`](evidence/m0/) |
| **M1** (`--mode flow` 3 canales + fóvea visual + ciego) | [`reports/GATES_M1.md`](reports/GATES_M1.md) | [`evidence/m1_p1/`](evidence/m1_p1/), [`evidence/m1_blind/`](evidence/m1_blind/) |
| **M2** (bloques 1920×1080 + stitch + MCP + NIM) | [`reports/GATES_M2.md`](reports/GATES_M2.md) | [`evidence/m2_blocks/`](evidence/m2_blocks/) |
| **M3** (filtros `-vf` estilo `ffmpeg` para pantallas) | [`reports/GATES_M3.md`](reports/GATES_M3.md) | [`evidence/m3_screen/`](evidence/m3_screen/) |
| **M4** (OCR local real + `ui_diff` temporal + `vlm_probe` + ciego Q1..Q6) | [`reports/GATES_M4.md`](reports/GATES_M4.md) | [`evidence/m4/`](evidence/m4/) |
| **M5** (índice progresivo SQLite+FTS5 + timeline video/secuencias) | [`reports/GATES_M5.md`](reports/GATES_M5.md) | [`evidence/m5/`](evidence/m5/) |

## Licencia

CC0 / dominio público, alineado con [FLOW](https://github.com/DannyBaanks/FLOW) y [FlowGen](https://github.com/DannyBaanks/FlowGen).
