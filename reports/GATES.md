# FlowSight M0 — reporte de gates

Fecha de creación: 2026-11-24. Pruebas realizadas **en el contenedor de trabajo de ChatGPT**, NO en el workspace `/home/danny/Development/ISyCo`.

| Gate | Resultado | Evidencia |
|---|---|---|
| Índice PNG (píxeles exactos) | DEMONSTRATED | `test_index_and_roundtrip_rebuild`; `verify reports/sprite.sqlite examples/sprite.png` |
| Consultas point / region / scanline / find / diff | DEMONSTRATED | `tests/test_flowsight.py` y `reports/queries.txt` |
| Bordes vectoriales discretos | DEMONSTRATED | `test_vector_edges_are_grounded` (segmentos de discontinuidad de color, no objetos semánticos) |
| L0 acotado | DEMONSTRATED | `reports/benchmark.json`: 1242 bytes para `sprite` y 1243 bytes para `flow_hello` |
| `.flowpaint` CANVAS/COLOR/RECT exacto en oráculo independiente | DEMONSTRATED | `test_export_flowpaint_exact_pixel_reconstruction` |
| `.flowpaint` exacto con `flow.paint.painter` verdadero | NOT_DEMONSTRATED | `tests/test_native_flow_optional.py` salta si FLOW no instalado |
| Experimento con modelo real solo-texto | NOT_DEMONSTRATED | `EXPERIMENTO_CIEGO.md` es protocolo pendiente |
| Fotos, OCR, comprensión de objetos humanos | NOT_DEMONSTRATED | M0 rechaza >64 colores sin cuantizar |
| Escritura/commit en repositorios de Danny | NOT_PERFORMED | puente local ISyMCP informó `Session terminated`; no se modificaron archivos remotos |

En benchmark, el tamaño de la tabla SQLite fue 36,864 bytes para cada ejemplo, superior al PNG sintético. **No se promete reducción de almacenamiento en disco.** La ganancia medida es en **texto de consulta entregado al agente** respecto de un JSON ingenuo por píxel.

La salida `summary` contiene un minimapa aproximado: la exactitud solo se garantiza consultando RLE/region/point y reconstrución RGBA. No confundir minimapa, componentes u otros resúmenes con visión humana.
