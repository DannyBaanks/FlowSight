# FlowSight M5 — Búsqueda Visual Progresiva y Percepción Temporal sobre Video/Secuencias

Fecha: 2026-11-24.
Repositorio: `/home/danny/Development/ISyCo Git/FlowSight` (`https://github.com/DannyBaanks/FlowSight`).
`FLOW` (`3e13909`) y `FlowGen` (`0dd6aa2`): **intactos (solo lectura)**.
`isymcp-hub`: **no modificado ni reiniciado**.

---

## 1. Arquitectura M5 (`capturas / video -> filtros -> índice -> búsqueda -> recortes -> respuesta`)

- **M5-A — Índice Visual Progresivo con Geometría Verificable ([`../flowsight_index.py`](../flowsight_index.py)):**
  - SQLite local + **FTS5** (`unicode61` preservando `-_./:` para tokens técnicos como `3128/tcp` o `br-575398bb8361`).
  - **Privacidad y ciclo de vida explícito:**
    - Vive fuera de cualquier repositorio git (`~/.local/share/flowsight/progressive_index.sqlite`, directorio `0700`, archivo `0600`).
    - **Nunca** escanea carpetas en segundo plano ni almacena bytes/píxeles de imágenes dentro de la base de datos (verificado en test: `0` cabeceras PNG en el SQLite).
    - Efímero por defecto (`ttl_hours=24.0`), expiración automática (`expire_stale`) y purga explícita (`purge_index`: `expired | all | sha256 | path_prefix`).
    - Deduplicación por `SHA-256 + ocr_enabled` para no re-ejecutar OCR en capturas ya indexadas.
  - **4 pasos progresivos:** `index_paths` → `search_index` (BM25 + coincidencia exacta + filtros `theme`/`color`/`since`/`until`) → top-K con evidencia OCR y `bbox` global exacto + `deepen_vf="crop=x:y:w:h,controls,ocr"` → profundizar solo el recorte ganador.

- **M5-B — Percepción Temporal sobre Secuencias y Videos ([`../flowsight_timeline.py`](../flowsight_timeline.py)):**
  - Lee secuencias ordenadas de PNG/JPG/WEBP **o** videos `.mp4/.mkv/.webm/.avi` vía `cv2.VideoCapture`.
  - **Salto de fotogramas estáticos por bloques (`8×6` tiles, `scene_thresh=6.0`):** ignora el ruido de compresión MP4 en frames repetidos (`delta < 2`) pero detecta de inmediato cambios localizados (~1 % de pantalla, como un botón cambiando de color o un banner de error).
  - Encadena `flowsight_diff.match_controls_temporal` con `prev_track_map` continuo (`K0..Kn`) y emite eventos con timestamp y `bbox` global: `CONTROL_APPEARED`, `CONTROL_DISAPPEARED`, `CONTROL_COLOR_CHANGED` (+ `INFERRED=['STATE_CHANGED_CANDIDATE']`), `CONTROL_MOVED`, `CONTROL_RESIZED`, `CONTROL_TEXT_CHANGED`, `TEXT_APPEARED`, `TEXT_DISAPPEARED`.

- **M5-C — 4 nuevas tools MCP ([`../flowsight_mcp.py`](../flowsight_mcp.py), 14 tools totales):**
  - `flowsight_index_add`, `flowsight_index_search`, `flowsight_index_purge`, `flowsight_timeline_scan` (100 % retrocompatible con las 10 tools de M1–M4).

---

## 2. Métricas reales medidas (`evidence/m5/benchmark_m5.json`, OCR real sin mocks)

### 2.1 Búsqueda visual progresiva sobre 7 capturas `1280×720` (`query="UFW 3128"`, `theme="dark"`)

| Métrica | Valor medido | Nota |
|---|---:|---|
| Enfoque ingenuo: 7 reportes completos al contexto del LLM | `21,931 chars` | Sin índice progresivo |
| `flowsight_index_search` (ranking + evidencia + `bbox` + `crop`) | `912 chars` (`0.89 ms`) | `#1 shot_03_ufw_terminal.png` (`score=0.900`, `cov=1.00`, términos `['3128', 'ufw']`) |
| Profundización solo sobre recorte ganador (`crop=0:0:854:270,controls,ocr`) | `1,125 chars` | Conserva coordenadas globales exactas |
| **Total progresivo (`search + crop`)** | **`2,037 chars`** | **Ahorro real medido: `90.71 %` (`10.76×` menos contexto)** |
| Indexado en frío (7 capturas con RapidOCR real) | `3,439.34 ms` | SQLite resultante: `4,096 B` (+ WAL), `0` bytes de imagen |
| Re-indexado con dedup `SHA-256` | **`1.90 ms`** | **`1,810×` más rápido** (`7/7` reutilizadas sin re-correr OCR) |

### 2.2 Timeline sobre video real `demo_timeline.mp4` (`4.0 s @ 10 FPS`, muestreo `2 FPS`)

| Métrica / Evento | Valor medido | Clasificación |
|---|---|---|
| Fotogramas muestreados / analizados / estáticos saltados | `8` muestreados → `4` keyframes analizados, **`4` saltados** (`1,029.31 ms` total) | **DEMONSTRATED** |
| `t=1.00s` (f10): botón `Merge pull request` verde → naranja | `CONTROL_COLOR_CHANGED` `track=K2` `#14a249(green)->#e89417(orange)` + `INFERRED=['STATE_CHANGED_CANDIDATE']` | Color: **DEMONSTRATED** / Estado: **INFERRED** |
| `t=2.00s` (f20): aparición de diálogo modal y botón `'Dismiss'` | `CONTROL_APPEARED` `K5` (`[700, 100, 1160, 290]`) y `K6` (`text='Dismiss'`) + `TEXT_APPEARED` `'Dismiss'` | **DEMONSTRATED** |
| `t=3.00s` (f30): aparición de banner de error rojo | `CONTROL_APPEARED` `K7` (`#b61b19(red)`, `[260, 320, 681, 386]`) + `TEXT_APPEARED` `'ERR0R:exit=56'` | **DEMONSTRATED** |

> **Observación honesta sobre compresión con pérdida MP4 (`mp4v`):** la compresión lossy introdujo ligeras fluctuaciones OCR en fuentes delgadas (`'Target Branch'` → `'TargetBranch'`, `'ERROR'` → `'ERR0R'`, `'ALLOW'` → `'ALL0W'`) y desplazó 3–4 px los bordes Canny de los inputs (`CONTROL_RESIZED`). Se conservan tal cual sin maquillar para mantener fidelidad empírica.

---

## 3. Clasificación epistémica consolidada M5

- **DEMONSTRATED:**
  - Índice progresivo SQLite+FTS5 con permisos `0700`/`0600`, cero bytes de imagen almacenados, dedup SHA-256 (`1.90 ms`), TTL/expiración y purga explícita.
  - Ahorro medido del **`90.71 %`** de caracteres frente a entregar 7 reportes completos, manteniendo `bbox` global verificable y zoom `crop=x:y:w:h`.
  - Escaneo de video `.mp4` real con salto de fotogramas estáticos por tiles y detección temporal con tracks persistentes `K0..Kn` de cambios de color, aparición de modales y aparición de textos de error.
  - Servidor MCP stdio con **14 tools** verificadas E2E (`50/50` tests PASS en toda la suite de FlowSight).
- **INFERRED:**
  - `STATE_CHANGED_CANDIDATE` en eventos `CONTROL_COLOR_CHANGED` del timeline.
- **NOT_DEMONSTRATED:**
  - Escaneo automático en segundo plano de directorios privados del sistema (prohibido por diseño: siempre opt-in explícito).
  - Comprensión semántica universal de video natural/escenas del mundo físico (el motor está especializado en interfaces gráficas, terminales y flujos UI).
