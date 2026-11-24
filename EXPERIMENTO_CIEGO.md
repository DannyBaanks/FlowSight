# M0 — Experimento falsable de imagen para agente **solo-texto**

**Hipótesis:** para pixel art de hasta 64 colores, un agente sin visión que reciba solo la salida textual de FlowSight podrá responder preguntas geométricas y cromáticas exactas bajo consultas delimitadas, sin ingerir píxeles individuales de la imagen entera. No implica clasificación semántica de fotos.

## Setup

1. Genera `examples/sprite.png` y su índice usando `python examples/benchmark.py`.
2. Crea una sesión de modelo text-only sin herramientas de visualización ni acceso al PNG, a las vistas previas o a `reports/benchmark.json`. Dale **solo** `reports/sprite_L0.txt`.
3. Ofrece como herramientas exclusivas de lectura las consultas CLI `point`, `scanline`, `region`, `find`, `components` y `diff`; el humano ejecuta cada consulta textual y retorna solo stdout. Establece presupuesto de **6 consultas** y **8000 caracteres de salida total**.
4. Pregunta, en orden aleatorio, para evitar memorizar prompts fijos:
   - ¿Cuáles son las dimensiones, origen y dirección de los ejes?
   - ¿Cuál es la caja exacta del color verde `#2ecc71ff`?
   - ¿Qué color exacto hay en el píxel `(15,24)`?
   - ¿Qué colores/coberturas hay en región `(12,20,12,10)`?
   - ¿Qué caja cambió en `examples/sprite_edited.png`? (solo stdout de `diff`, no imagen)
   - ¿La imagen representa con certeza un gato? **Respuesta esperada: NO DEMOSTRADO.**
5. Puntúa separado: exactitud geométrica (0..5), calibración semántica (rechazo de identificación no respaldada), llamadas y caracteres emitidos.

## Gates

- **G0:** indexación y reconstrucción RGBA byte-a-byte == original.
- **G1:** L0 <= 3500 caracteres; la base SQLite puede ser mayor porque no se introduce en el contexto.
- **G2:** 5 preguntas objetivas exactas con <=6 consultas, <=8000 caracteres, sin acceso al PNG.
- **G3:** No afirma conocer objetos no deducibles de valores/relaciones visibles.
- **G4:** exportación `.flowpaint` ejecutada en el painter nativo de FLOW con coincidencia RGBA exacta; **PENDIENTE**, no reemplazar con la interpretación de prueba.

## Falsificación / condiciones de parada

- Si el agente necesita abrir el PNG para responder, **no es un test text-only válido**.
- Si datos con transparencia, paleta o RLE no se reconstruyen exactamente, **FAIL G0**.
- Si un resumen truncado se interpreta como observación completa, **FAIL G3**.
- Si una foto excede 64 colores, **DENY BY DESIGN** en M0, no se registra como éxito de percepción.
- Sin verificación en el painter real, no declarar integración con FLOW demostrada.

## Qué registrar

`modelo`, `versión`, `prompt completo`, `herramientas concedidas`, `preguntas`, `cada stdout`, `respuestas`, `veredicto`, `bytes`, `tiempo`, `sha256 de índice/PNG` y límites de contexto. No atribuir semántica humana a estructuras de píxeles sin evidencia.
