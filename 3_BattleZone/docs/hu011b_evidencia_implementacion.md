# HU011B — Evidencia de implementación

## Estado

**HU011B IMPLEMENTADA — MATERIALIZACIÓN DE ARTEFACTOS PENDIENTE**

## IMPLEMENTATION EVIDENCE

- Fuente HU011 explícita: `battlezone-dqn-20260903-001628-b7c33d5-255e`, 1.000.000 global steps, DQN `reference_v1`, Git SHA `b7c33d58f6c896da3bea824537cd810a83932ee0`.
- `src/model_artifact.py`: exportación atómica del modelo compacto, checksum SHA256, metadata de linaje, recarga round-trip, agente mínimo sin Replay/optimizer/Target Network y resolución local-first con fallback de run explícito.
- `src/reporting.py`: lectura desde eventos TensorBoard reales, validación de tags/global steps/valores y generación de las tres figuras obligatorias; API de explotación preparada para registros estructurados HU013.
- `src/video.py`: MP4 reproducible con seed/FPS explícitos, frames RGB, overlay, cierre seguro y metadata. El video de proceso exige un checkpoint intermedio y el post-entrenamiento exige el modelo entregable recién cargado con epsilon 0.
- `src/delivery.py`: layout determinista, `delivery_manifest.json` separado del manifest de entrenamiento y `HU011B_DELIVERY_GATE` estricto.
- Notebook: celdas HU011B protegidas con `RUN_HU011B_DELIVERY=False`, checkpoint intermedio real explícito de 750.000 steps, visualización inline y verificación autónoma independiente de entrenamiento.
- Bootstrap de desarrollo: mientras PR #36 permanece abierto, el notebook resuelve `feature/battlezone-hu011b-delivery-artifacts` sin fijar un SHA mutable de desarrollo; el checkout ocurre antes de importar los módulos HU011B.
- Materialización local-first: la misma exportación validada se copia atómicamente, con sidecars y verificación SHA/metadata, tanto a `<PERSISTENT_ROOT>/models/<run_id>/` como a `3_BattleZone/`. La carga resuelve `LOCAL_DELIVERY` antes de `PERSISTENT_FALLBACK`.
- El delivery manifest registra por separado `delivery_model_path`, `persistent_model_path` y `local_project_model_path`; el gate exige que las tres copias compartan SHA256 y falla si falta el modelo local.
- Idempotencia: modelo y JSON se escriben mediante reemplazo atómico; las rutas están aisladas por `run_id`; no se modifica ningún checkpoint fuente.
- No se ejecutó otro entrenamiento, tuning, evaluación formal HU013 ni código Assault.
- Validación local tras auditoría: `compileall` PASS; 36 tests focales PASS; suite BattleZone 160 passed, 1 skipped, 0 failed.

## REAL DELIVERY EVIDENCE

`PENDING`

Codex no tuvo acceso al Google Drive que contiene la corrida. Por ello no se fabricaron modelo, gráficas, eventos ni videos finales. Deben materializarse en Colab ejecutando únicamente las celdas HU011B con Drive montado.

Evidencia que deberá registrarse tras esa ejecución:

- SHA256 y tamaño real de `battlezone_dqn_model.pt`;
- paths de las tres figuras derivadas de TensorBoard;
- paths, reward sanity y metadata de ambos videos;
- `delivery_manifest.json` del run real;
- `HU011B_DELIVERY_GATE=PASS`.

La figura `exploitation_reward.png` definitiva permanece `PENDING_HU013` hasta disponer de al menos 10 episodios formales.
