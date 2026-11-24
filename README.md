# FlowSight — primer experimento de «visión» sin modelo visual

**M0, experimental, no integrado ni fusionado con FLOW.** Un traductor de imágenes PNG de colores discretos a una memoria espacial SQLite consultable por un agente **solo-texto**. Puede preguntar por colores, posiciones, regiones y cambios sin recibir un JSON por píxel. Incluye exportación al subconjunto **CANVAS / COLOR / RECT** de `.flowpaint` y reconstrucción exacta del PNG a nivel de píxeles RGBA decodificados.

## Contratos que conservamos

- FLOW (repo `DannyBaanks/FLOW`) usa **R/G como campo vectorial** y **B como instrucciones**, por lo que una imagen visual cualquiera **no** es directamente un programa FLOW. `index --mode flow` explica estos canales, pero no ejecuta la VM.
- `flow.paint.painter.run_paint_program` ya admite `CANVAS`, `COLOR`, `RECT`. Este prototipo exporta ese subconjunto. **Validado aquí con un intérprete independiente del subconjunto; todavía falta ejecutarlo contra el painter real de FLOW en la máquina del usuario.**
- Sin extracción semántica de objetos, OCR ni embeddings; «verde» es dato RGB, «ojo» sería una inferencia no demostrada. **No garantiza visión perceptual humana.**
- Sin modificación de FLOW, FlowGen, ISyMCP ni ISyMem. Sin red, sin LLM, sin APIs remotas.

## Instalación

Python ≥3.10 y Pillow (el mismo tipo de dependencia que FLOW):

```bash
python -m pip install Pillow
python examples/make_examples.py
python examples/benchmark.py
python -m unittest discover -s tests -v
```

## CLI (fóvea en lugar de JSON gigante)

```bash
python flowsight.py index examples/sprite.png demo.sqlite
python flowsight.py summary demo.sqlite
python flowsight.py point demo.sqlite 15 24
python flowsight.py scanline demo.sqlite 24
python flowsight.py region demo.sqlite 12 20 12 10
python flowsight.py find demo.sqlite '#2ecc71'
python flowsight.py components demo.sqlite
python flowsight.py edges demo.sqlite --limit 40
python flowsight.py diff demo.sqlite examples/sprite_edited.png
python flowsight.py verify demo.sqlite examples/sprite.png
python flowsight.py rebuild demo.sqlite reconstructed.png
python flowsight.py export-flowpaint demo.sqlite reconstructed.flowpaint

# Programas FLOW: sus canales son CÓDIGO y campo, no color visual arbitrario
python flowsight.py index examples/flow_hello.png hello.sqlite --mode flow
python flowsight.py point hello.sqlite 2 2
```

La salida global `summary` es un **minimapa textual** (24×16 como máximo), una leyenda RGB y un máximo de siete cajas de regiones. Los demás detalles se consultan por coordenadas; `edges` extrae segmentos discretos verticales/horizontales exactamente en fronteras de color (no son contornos semánticos de objetos). La paleta y los RLE completos quedan en SQLite: **la salida textual no necesita contenerlos todos**. En cada comando se indican truncamientos si excede el límite de líneas.

## Sistema de coordenadas y precisión

Origen (0,0) en esquina superior izquierda; `x` aumenta hacia la derecha, `y` hacia abajo. Cajas `[x0,y0,x1,y1)` con extremos finales *exclusivos*; `region` acepta `x y ancho alto`. Colores RGBA exactos, incluyendo transparencia. La comparación y SHA-256 se realizan sobre **RGBA descomprimido**, no sobre los bytes PNG, que pueden diferir por metadata y compresión.

## Reglas de seguridad y límites explícitos

- Solo PNG, dimensiones 1..512 y hasta **64 colores RGBA diferentes**. Si supera ese límite se **rechaza** sin cuantizar silenciosamente. Foto y screenshots complejos quedan para M1.
- Una consulta nunca inventa identidad de objetos ni coordenadas. Objetos geométricos = componentes **4-conexos**, no clasificación semántica.
- Solo los 64 componentes conectados más grandes se almacenan individualmente; `summary` informa recuento total y truncamiento.
- `export-flowpaint` limita por defecto a 50,000 instrucciones RECT **y a 64 MiB de snapshots UNDO potenciales**: el painter actual de FLOW conserva hasta 1024 snapshots completos y podría agotar RAM con canvas grandes. Para reconstruir PNG grande usa `rebuild`, que no usa el painter.
- Rechaza rutas de salida existentes salvo `--force`; trabaja sobre copia de datos. SQLite se crea de manera temporal y se publica al final del indexado.
- Las métricas de ahorro son bytes de salida del prototipo comparados con JSON por píxel, **no tokens medidos por tokenizer ni ahorro facturado**.

## Evidencia y experimentación con un LLM text-only

Ejecuta `python examples/benchmark.py`: genera `reports/benchmark.json`, `reports/*_L0.txt`, `reports/queries.txt`, `.flowpaint` y PNG reconstruidos. Los PNG reconstruidos se comparan por píxeles en los tests. Después sigue [EXPERIMENTO_CIEGO.md](EXPERIMENTO_CIEGO.md) para evaluar al agente sin mostrarle el PNG, comparando sus respuestas con un oráculo de consultas deterministas.

## Próximos gates (todavía NO DEMOSTRADOS)

1. Ejecutar `.flowpaint` exportado con el `run_paint_program` real de FLOW en la máquina de Danny; comparar bytes RGBA decodificados contra original.
2. Probar con un **modelo realmente text-only** en una sesión sin acceso a imágenes: preguntas de color, posición, simetría e identidad de objetos y reportar la matriz correcta/incorrecta/indeterminable.
3. Incorporar `crop`, contornos, quadtree y detección de primitivas (no solo componentes), garantizando límites de salida por operación.
4. Para fotos y UI: segmentación/agrupamiento **aproximados**, métricas de error y OCR opcional explícito; NO declarar pixel-exacto si hay cuantización.
5. Contrato MCP para preguntas L0/L1/L2 y trazas de consultas, sin modificar el core de FLOW.

## Licencia

CC0 / dominio público, alineado con FLOW y FlowGen.
