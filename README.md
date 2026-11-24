# FlowSight — periférico de inspección espacial y estructural para agentes solo-texto

**M1 (repositorio independiente, periférico de solo lectura sobre `FLOW` y `FlowGen`).**
`FlowSight` traduce imágenes PNG en una memoria espacial SQLite lossless y consultable por fóvea (L0 global acotado → L1 consultas regionales → L2 reconstrucción/verificación exacta), sin modificar los núcleos de `FLOW` ni `FlowGen` y sin usar modelos visuales ni APIs remotas.

> La línea base **M0** y toda su evidencia original se conservan íntegras en [`evidence/m0/`](evidence/m0/) y en el commit inicial del repositorio.

---

## Dos modos lossless explícitos (M1) + Modo bloques alta resolución & MCP (M2)

### 0. Modo bloques alta resolución (`flowsight_blocks.py`) y Servidor MCP (`flowsight_mcp.py`)
Para fotos y screenshots grandes (`1920×1080` o cualquier PNG/JPG/WEBP hasta `8192px`, >64 colores) donde el modo lossless rechaza por diseño:
- **Explícitamente aproximado (`photo-blocks-v1`):** paleta global compartida (`median-cut` + refinamiento Lloyd k-means determinista) con métricas reales de error (`MAE_RGB`, `RMSE_RGB`, `PSNR_dB`, `within_d24%`).
- **Partición en mitades o cuartos** (`quarters`: `TL/TR/BL/BR`; `halves_h`: `L/R`; `halves_v`: `T/B`) + **subdivisión recursiva** (quadtree zoom) con mapa de color y **mapa de bordes Sobel** por bloque.
- **Unión estructural y semántica (`stitch_blocks`):** mide continuidad píxel a píxel en las costuras reales, fusiona regiones trans-frontera (`G0..Gn [CROSS-SEAM]`), re-ensambla el mapa global y alinea notas textuales por bloque (`AGENT_NOTES / INFERRED`).

```bash
# CLI directo de bloques 1920x1080
python3 flowsight_blocks.py map foto_1080p.png --split quarters --colors 12
python3 flowsight_blocks.py block <session_id> TL
python3 flowsight_blocks.py subdivide <session_id> TL --split quarters
python3 flowsight_blocks.py stitch <session_id> --notes-json '{"TL":"...","TR":"..."}'

# Arrancar servidor MCP stdio (mcp 2.x)
python3 flowsight_mcp.py
```

Snippet listo para registrar en `~/.codex-web-http/hub/servers.json` (cuando decidas habilitarlo en `isymcp-hub`):
```json
"flowsight": {
  "command": "python3",
  "args": ["/home/danny/Development/ISyCo Git/FlowSight/flowsight_mcp.py"],
  "cwd": "/home/danny/Development/ISyCo Git/FlowSight",
  "isolation": "shared"
}
```

### 1. `--mode flow` — Índice nativo de 3 canales para programas FLOW (`flowsight-flow-v1`)
Los programas FLOW (`DannyBaanks/FLOW`) codifican **dos capas distintas en la misma imagen RGB**:
- **Canales `R, G` (campo vectorial continuo):** almacenados **sin pérdida ni cuantización** como arrays `uint8` exactos (`vx = R/127.5 - 1`, `vy = G/127.5 - 1`), acompañados de resúmenes espaciales deterministas por tiles, estadísticas regionales exactas (media, magnitud, coherencia, divergencia y rotacional/curl en diferencias finitas exactas) y grillas direccionales.
- **Canal `B` (instrucciones y datos discretos `0..255`):** almacenado en RLE exacto por fila, paleta de opcodes, componentes 4-conexos y tabla directa de `spawns` (`B=255`).
- **Sin rechazo por >64 colores RGB:** programas reales como `FLOW/examples/vortex.png` (873 colores RGB distintos) y `FLOW/examples/hello_flow.png` se indexan y reconstruyen con **0 errores en R, G y B**, manteniendo invariantes las trazas de la VM real de FLOW.
- **Aclaración epistémica:** el mapa de flechas por tile en L0 y `field` es un *resumen aproximado* de la media local; los valores exactos sin cuantizar se consultan con `point`, `scanline`, `region`, `field` e `instructions`. FlowSight observa el programa estático y **nunca ejecuta la VM de FLOW**.

### 2. `--mode visual` — Índice raster discreto (`flowsight-rle-v1`) + mejoras de fóvea M1
Para pixel art, sprites y diagramas discretos (PNG, 1..512 por lado, ≤64 colores RGBA exactos; rechaza >64 colores sin cuantizar silenciosamente):
- `summary` (L0): paleta en orden determinista, minimapa `24×16` por mayoría (≥60% o `?`), **marcadores explícitos `[hidden-in-minimap]` con `bbox` exacto para colores minoritarios** ausentes del minimapa, y top componentes 4-conexos.
- `region X Y W H --grid`: vista simbólica **1:1 exacta** para fóveas pequeñas (límite fail-closed: ≤64×64 y ≤1024 px).
- `edges --region X Y W H`: fronteras discretas `V`/`H` acotadas a una caja sin truncamiento global.
- `diff`: reporta `CHANGED BBOX` y **transiciones exactas de color** (`C_old(#hex) -> C_new(#hex)`).
- `export-flowpaint`: exporta al subconjunto `CANVAS / COLOR / RECT` compatible con `flow.paint.painter.run_paint_program`.

---

## Instalación y pruebas

Requisitos: Python ≥3.10 y `Pillow`.

```bash
python3 -m pip install -r requirements.txt

# Suite completa (con FLOW en PYTHONPATH se ejecutan también los gates nativos)
PYTHONPATH="/home/danny/Development/ISyCo Git/FLOW:." python3 -m unittest discover -s tests -v

# Benchmark y gates de M1 Prioridad 1 sobre programas FLOW reales
python3 examples/benchmark_m1_p1.py
```

---

## CLI

```bash
# --- Programas FLOW (nativo 3 canales, lossless) ---
python3 flowsight.py index "/home/danny/Development/ISyCo Git/FLOW/examples/vortex.png" vortex.sqlite --mode flow
python3 flowsight.py summary vortex.sqlite
python3 flowsight.py field vortex.sqlite --region 16 16 32 32 --cols 8 --rows 8
python3 flowsight.py instructions vortex.sqlite --limit 20
python3 flowsight.py region vortex.sqlite 30 30 5 5 --grid
python3 flowsight.py point vortex.sqlite 32 29
python3 flowsight.py scanline vortex.sqlite 29
python3 flowsight.py verify vortex.sqlite "/home/danny/Development/ISyCo Git/FLOW/examples/vortex.png"
python3 flowsight.py rebuild vortex.sqlite vortex_rebuilt.png

# --- Imágenes visuales discretas (<=64 colores RGBA) ---
python3 flowsight.py index examples/sprite.png sprite.sqlite
python3 flowsight.py summary sprite.sqlite
python3 flowsight.py region sprite.sqlite 12 20 12 10 --grid
python3 flowsight.py edges sprite.sqlite --region 12 20 12 10
python3 flowsight.py diff sprite.sqlite examples/sprite_edited.png
python3 flowsight.py export-flowpaint sprite.sqlite sprite.flowpaint
```

---

## Sistema de coordenadas y garantías fail-closed

- Origen `(0,0)` en esquina superior izquierda; `x` aumenta hacia la derecha, `y` hacia abajo; cajas `[x0, y0, x1, y1)` con extremos finales **exclusivos**.
- Verificación de integridad SHA-256 por canal (`r_sha256`, `g_sha256`, `b_sha256`, `rgb_sha256`, `rgba_sha256`) al abrir y reconstruir.
- Escritura atómica vía archivo temporal + `os.replace`; lectura en SQLite `?mode=ro`; nunca sobrescribe sin `--force`.
- Determinismo byte a byte en índices SQLite, resúmenes L0 y salidas de consulta.
- **Límites explícitos (`NOT_DEMONSTRATED`):** no hace reconocimiento semántico de fotos/objetos humanos, no hace OCR, no cuantiza imágenes >64 colores en `--mode visual`, ni predice la ejecución dinámica multi-tick de la VM de FLOW sin ejecutarla.

---

## Evidencia y reportes

- [`evidence/m0/`](evidence/m0/) — Evidencia original M0 intacta (`GATES.md`, `M0_SHA256SUMS.txt`, benchmarks y artefactos).
- [`evidence/m1_p1/`](evidence/m1_p1/) — Benchmark, índices, L0 y verificación de gates M1 Prioridad 1 sobre `hello_flow.png` y `vortex.png`.
- [`evidence/m1_blind/`](evidence/m1_blind/) — Transcripts completos y auditoría de los experimentos ciegos M1 (`m1_a_vortex`, `m1_b_hello_flow`, `m1_c_visual_p2`).
- [`evidence/m2_blocks/`](evidence/m2_blocks/) — Artefactos 1920×1080, salidas de bloques/stitch y transcript real con NVIDIA NIM (`nemotron-3.5-lightning-30b-a3b`).
- [`reports/GATES_M1.md`](reports/GATES_M1.md) — Matriz consolidada M1 (`DEMONSTRATED`, `INFERRED`, `NOT_DEMONSTRATED`).
- [`reports/GATES_M2.md`](reports/GATES_M2.md) — Matriz consolidada M2 (bloques 1920×1080, stitching, MCP stdio E2E y experimento NVIDIA NIM).

## Licencia

CC0 / dominio público, alineado con `FLOW` y `FlowGen`.
