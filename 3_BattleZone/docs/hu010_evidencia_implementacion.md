# HU010 — Evidencia de implementación

## 1. Estado

La capa técnica de trazabilidad ligera está implementada y validada. PR #32 corresponde a la definición HU010 ya mergeada; un PR de seguimiento integra exclusivamente la implementación técnica HU010.

## 2. Rama y commit de base

- Rama: `feature/battlezone-hu010-experiment-traceability`.
- Commit al capturar la evidencia: `d6f73c1b921cc90866d53a77cab654ce9c53b9ad`.
- El SHA final de implementación es el commit posterior que contiene este documento.

## 3. Archivos modificados

- `.gitignore`.
- `3_BattleZone/configs/battlezone_config.yaml`.
- `3_BattleZone/src/experiment.py`.
- `3_BattleZone/tests/test_experiment_tracking.py`.
- `3_BattleZone/docs/hu010_evidencia_implementacion.md`.

La reejecución del notebook quedó identificada en el commit `1c4421b` y se retiró del delta del PR de seguimiento; `3_BattleZone/pipeline_battlezone.ipynb` es idéntico a `origin/main`.

## 4. Tracking config

Se añadió `tracking`, separada de `training`, `checkpointing`, `tensorboard` y `smoke`, habilitada con `results_dir=3_BattleZone/results`, `manifest_filename=run_manifest.json`, schema 1 y gate Git estricto.

## 5. Manifest schema

`run_manifest.json` usa `schema_version=1` y conserva proyecto/algoritmo, estado, modo, timestamps, Git, entorno, snapshot/hash de configuración, runtime, hardware, progreso, artefactos, resume, sesiones y notas. La carga rechaza JSON corrupto, campos críticos ausentes y schemas incompatibles.

## 6. run_id y results path

- Ejemplo ejecutado: `battlezone-dqn-20260902-211209-d6f73c1-97a4`.
- Ruta canónica: `3_BattleZone/results/<run_id>/run_manifest.json`.
- Se crea además `summaries/`; una colisión se rechaza con `FileExistsError`.

## 7. Configuración y Git

- SHA-256 real del YAML: `46568ae7f7e29a6ce4f370d043c92f018ac30859d16a5ffac1ff32996a8d46f5`.
- Git SHA capturado: `d6f73c1b921cc90866d53a77cab654ce9c53b9ad`.
- Branch capturada: `feature/battlezone-hu010-experiment-traceability`.
- Dirty real: `true`, debido al notebook local preexistente y a los cambios HU010 todavía no confirmados durante la captura.

## 8. Runtime

Python 3.12.11; Gymnasium 1.1.1; ALE-Py 0.10.1; PyTorch 2.13.0; TensorBoard 2.21.0; macOS arm64. Los paquetes ausentes se representan con `null`.

## 9. Hardware

Device `mps`; CUDA `false`; GPU CUDA `null`; CPU `arm`; RAM 32.0 GiB. La detección soporta CPU, CUDA y MPS sin asumir acelerador.

## 10. NEW session

La validación controlada creó sesión 1, modo `new`, `start_global_step=0`, `end_global_step=32`, checkpoint de salida `..._step_32.pt`, log dir explícito y estado `interrupted` porque el objetivo lógico aún no se alcanzó.

## 11. RESUME_FULL

La validación reabrió explícitamente el mismo manifest/run_id, creó sesión 2, restauró desde `..._step_32.pt`, avanzó `32→48`, registró `replay_restored=true` y checkpoint de salida `..._step_48.pt`.

## 12. RESUME_LIGHTWEIGHT

El test controlado valida `mode=resume_lightweight` y `replay_restored=false`, sin seleccionar automáticamente runs o checkpoints.

## 13. Session history y continuidad

Las sesiones 1 y 2 quedaron preservadas con índices monotónicos. Se rechaza cualquier resume cuyo inicio no coincida con el `end_global_step` anterior.

## 14. Checkpoint y TensorBoard lineage

Input/output checkpoints y `tensorboard_log_dir` se guardan tanto en sesión como en los campos agregados del manifest. HU007 y los tags HU008 no se modificaron.

## 15. Persistencia segura

La escritura usa archivo temporal UTF-8, `flush`, `fsync` y `os.replace`. Un fallo forzado de replace preservó byte por byte el manifest válido y eliminó el temporal.

## 16. READY_FOR_LONG_TRAINING

Con un checkout limpio inyectado, el gate dio `True`: tracking, results writable, run_id, config/hash, Git SHA/clean, TensorBoard, checkpointing, manifest writable, DQN y BattleZone pasaron. En el worktree local real dio `False` únicamente por `git_clean_when_required`, como exige el modo estricto y sin limpiar Git automáticamente.

## 17. Tests

- `compileall`: PASS.
- HU010: `13 passed`.
- Regresión focal: `79 passed, 1 skipped`.
- Suite completa BattleZone: `104 passed, 1 skipped`.
- Fallos: 0.
- Skip: el smoke real ALE marcado por HU009 cuando ROM/runtime no está disponible; no se repitió porque HU010 no modifica entorno/trainer.
- `git diff --check`: PASS.

## 18. CA01–CA18

CA01–CA18: PASS; no se añadió lógica Assault, servicios externos, entrenamiento largo, tuning ni evaluación formal.

## 19. AV01–AV18

AV01–AV18: PASS técnico mediante inspección de historial, tests focales/completos, flujo temporal controlado, hash real, captura Git/runtime/hardware, persistencia atómica y scope checks. El dirty flag se verificó por captura real e inyección, sin modificar Git.

## 20. Scope

BattleZone mantiene DQN clásico. No se modificaron `2_Assault/`, `trainer.py`, el contrato HU007, tags HU008 ni el notebook. No se ejecutó entrenamiento largo, tuning, evaluación formal ni smoke ALE adicional.

## 21. Limitaciones

El manifest vincula rutas de checkpoints/logs; no duplica artefactos. RAM puede ser `null` si `psutil` no está instalado. La continuidad del estado interno ALE entre procesos sigue el contrato HU007: se preserva progreso y el entorno se resetea.

## 22. Linaje administrativo

PR #32 (`https://github.com/j-mauro-r/reinforcement_learning_reto_1/pull/32`) mergeó la definición documental. El PR de seguimiento autorizado contiene la implementación HU010 y debe permanecer abierto hasta su auditoría final.
