# Registro MCP de FlowSight en `isymcp-hub` (Manual y Reversible)

> **Importante:** Este documento NO ha sido aplicado al hub global. Registrar o reiniciar `isymcp-hub.service` desconecta momentáneamente las sesiones TUI activas; hazlo solo cuando quieras activarlo.

## Tools expuestas por `flowsight_mcp.py` (8 tools, solo lectura sobre imágenes)

1. `flowsight_map_blocks` — partición `1920×1080` en mitades/cuartos con paleta global y error medido.
2. `flowsight_get_block` — inspección compacta de un bloque (color + bordes Sobel + regiones).
3. `flowsight_subdivide_block` — zoom recursivo preservando coordenadas globales.
4. `flowsight_stitch_blocks` — continuidad real en costuras + fusión `CROSS-SEAM` + alineación de `block_notes`.
5. `flowsight_inspect_lossless` — modo lossless M1 (`--mode flow` / `--mode visual`).
6. `flowsight_screen_filters` — catálogo declarativo estilo `ffmpeg -filters`.
7. `flowsight_screen_run` — pipeline `-vf` para capturas (`theme,layout,controls,text_bands,ocr,ui_map,ui_diff,vlm_probe`).
8. `flowsight_screen_query` — consulta acotada por selector (`P0`, `W1`, `K0`, `ambiguities`, `ocr`, `ui_diff`).

## Paso 1 — Respaldo previo (reversible)

```bash
cp ~/.codex-web-http/hub/servers.json ~/.codex-web-http/hub/servers.json.bak-pre-flowsight
```

## Paso 2 — Entrada a agregar en `~/.codex-web-http/hub/servers.json`

```json
"flowsight": {
  "command": "python3",
  "args": ["/home/danny/Development/ISyCo Git/FlowSight/flowsight_mcp.py"],
  "cwd": "/home/danny/Development/ISyCo Git/FlowSight",
  "isolation": "shared"
}
```

## Paso 3 — Verificación antes de reiniciar el hub

```bash
cd "/home/danny/Development/ISyCo Git/FlowSight"
PYTHONPATH="/home/danny/Development/ISyCo Git/FLOW:." python3 -m unittest tests/test_blocks_and_mcp.py tests/test_screen_filters.py tests/test_m4_adversarial.py -v
```

## Paso 4 — Backends locales de OCR

- **Activo en este host (CPU, sin red):** `rapidocr-onnxruntime==1.4.4` (PP-OCRv4 ONNX) usado automáticamente por `ocr` (`backend=auto` o `backend=rapidocr`).
- **Opcional adicional (`tesseract`):** si también deseas habilitar Tesseract como backend alternativo (`backend=tesseract`):

```bash
sudo apt update && sudo apt install -y tesseract-ocr tesseract-ocr-eng tesseract-ocr-spa
python3 "/home/danny/Development/ISyCo Git/FlowSight/examples/benchmark_m4.py"
```

## Rollback

```bash
cp ~/.codex-web-http/hub/servers.json.bak-pre-flowsight ~/.codex-web-http/hub/servers.json
systemctl --user restart isymcp-hub.service
```
