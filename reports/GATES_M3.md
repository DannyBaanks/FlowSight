# FlowSight M3 — Motor de Filtros Componibles estilo `ffmpeg` para Capturas de Pantalla

Fecha: 2026-11-24.
Repositorio independiente: `/home/danny/Development/ISyCo Git/FlowSight`.
`FLOW` (`3e13909`) y `FlowGen` (`0dd6aa2`): **intactos (solo lectura)**.

---

## 1. Arquitectura `ffmpeg`-style ([`../flowsight_screen.py`](../flowsight_screen.py))

Pipeline declarativo `-vf "f1=arg,f2=...,f3"` sobre `FilterContext` con registro extensible `@register_filter(...)` (OpenCV + NumPy + Pillow, cero LLM dentro de filtros):

| Filtro | Sintaxis `-vf` | Qué produce |
|---|---|---|
| `crop` | `crop=x:y:w:h` | Recorte de fóvea preservando coordenadas globales de `1920×1080` |
| `theme` | `theme` | Polaridad (`dark`/`light`/`mixed`), superficies dominantes y colores de acento |
| `layout` | `layout[=max_panels]` | Divisores reales `H/V` y paneles (`P0..Pn`) con `global_bbox`, fondo y pista posicional |
| `controls` | `controls[=max]` | Controles clicables candidatos (`accent-control`, `bordered-control`, `input-or-bar-box`, `badge`) con **`click=(cx, cy)`** y `panel=P_k` |
| `text_bands` | `text_bands[=max]` | Bandas horizontales de texto puramente geométricas (sin OCR) enlazadas a panel/control |
| `ocr` | `ocr[=lang]` | Plugin opcional fail-closed (`tesseract` si está en PATH; `ABSENT` si no, sin alucinar texto) |
| `ui_map` | `ui_map[=cols:rows]` | Wireframe ASCII compacto (`48×20` ≈ 1 KB) de divisores, controles y bandas de texto |
| `ui_diff` | `ui_diff=before.png` | Agrupación de cambios UI por región (`D0..Dn`), transición hex, panel y controles afectados |
| `blocks` | `blocks[=quarters\|halves_h\|halves_v:colors]` | Integra partición/stitching M2 dentro del mismo grafo `-vf` |

Expuesto además en [`../flowsight_mcp.py`](../flowsight_mcp.py) con `flowsight_screen_filters` y `flowsight_screen_run` (total: **7 tools MCP**).

---

## 2. Rendimiento y Gates verificados

- **Suite completa:** `31/31 PASS` (M0 + M1 + M2 + M3 + E2E MCP stdio).
- **Captura sintética `1920×1080` (`ui_1080p_before.png`):**
  - Pipeline `theme -> layout -> controls -> text_bands -> ocr -> ui_map=48:20`: **`67.57 ms`** (`layout` 33.5 ms, `controls` 11.9 ms, `text_bands` 6.7 ms, `ui_map` 3.8 ms).
  - Detectó 4/4 paneles exactos (`H_div_y=[58, 1042]`, `V_div_x=[282]`), 3/3 botones de acento con `click=(1815,28)`, `(1430,886)`, `(1640,886)` exactos al píxel, 22 bandas de texto y `OCR=ABSENT`.
  - `ui_diff` (`155.8 ms`): detectó los 2 cambios exactos (`D0` modal central `[695,415,1226,626]`, `D1` botón verde→naranja `[1335,855,1525,917]`).
- **Captura REAL de escritorio Linux `1920×1080`** (solo lectura en `/home/danny/Imágenes/Capturas de pantalla/`, nunca copiada al repo):
  - Pipeline `theme -> layout -> controls -> text_bands -> ui_map=48:18`: **`48.97 ms` (~20 FPS en CPU pura)**, `polarity=dark`, `12` paneles, `18` controles y `67` bandas de texto en `6,500 chars` acotados.

---

## 3. Experimento ciego en vivo con NVIDIA NIM (`nvidia/nvidia/nemotron-3.5-lightning-30b-a3b`)

Transcript completo en [`../evidence/m3_screen/m3_screen_nim_transcript.json`](../evidence/m3_screen/m3_screen_nim_transcript.json) (`17.0 s`, `6,646 ch` prompt, directorio temporal vacío, cero acceso a PNG):

- **S1 (tema + paneles):** **DEMONSTRATED** — acertó `dark`, `P0..P3` y todos los divisores/bboxes exactos.
- **S2 (coordenadas click de 3 botones):** **DEMONSTRATED** — `(1815,28)`, `(1430,886)` y `(1640,886)` exactos al píxel.
- **S3 (diff de UI):** **DEMONSTRATED (parcial en lectura del modelo)** — ubicó ambos bboxes `D0` y `D1` y las transiciones en `P0`, pero el modelo confundió el color inicial de `D0` (`#12141c` vs `#16a34a`) porque `W1` se renumeró entre `before` y `after` al aparecer un control nuevo. *Mejora identificada para M3.1:* incluir `old_family->new_family` y etiquetar controles `NEW` vs `PERSISTENT` en `ui_diff`.
- **S4 (palabra exacta dentro del botón verde sin OCR):** **DEMONSTRATED (calibración)** — respondió **`NOT_DEMONSTRATED`** citando `OCR: status=ABSENT` y distinguiendo `has_inner_text/icon` de lectura real de caracteres.

---

## 4. Clasificación epistémica (M3)

- **DEMONSTRATED:** motor de filtros componibles `-vf` determinista a ~49–68 ms sobre `1920×1080` reales, coordenadas `click=(cx,cy)` exactas preservadas bajo `crop`, wireframe ASCII compacto, `ui_diff` agrupado, E2E MCP stdio y uso ciego por NVIDIA NIM solo-texto.
- **INFERRED:** etiquetas posicionales (`top-bar-band`, `main-pane`, `accent-control`) como hipótesis estructurales de UI.
- **NOT_DEMONSTRATED:** lectura de texto real en pantalla (requiere instalar `tesseract-ocr` para activar el filtro `ocr`), y detección de controles con contraste muy sutil sin borde ni saturación (e.g. botones fantasma minimalistas).
