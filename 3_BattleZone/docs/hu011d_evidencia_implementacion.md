# HU011D — Evidencia de implementación

## Archivos modificados

- `3_BattleZone/configs/battlezone_config.yaml`
- `3_BattleZone/src/training_run.py`
- `3_BattleZone/tests/test_full_training.py`
- `3_BattleZone/pipeline_battlezone.ipynb`
- `3_BattleZone/docs/hu011d_evidencia_implementacion.md`

## Valores efectivos

```text
profile: improved_v2
algorithm: DQN
total_timesteps: 1000000
replay_buffer_capacity: 16384
replay_sampling: uniform
learning_rate: 0.0001
epsilon: 1.0 -> 0.05
epsilon_decay_steps: 750000
batch_size: 32
train_frequency: 4
target_sync_interval: 10000
```

`reference_v1` continúa siendo aceptado por
`resolve_long_training_config(...)`.

## Pruebas ejecutadas

- Tests focales de `test_full_training.py`: `20 passed`.
- Smoke y regresiones relevantes de training, agent, trainer, checkpoints,
  TensorBoard, evaluación y modelo: `99 passed, 1 skipped`.
- Automatización focal HU011/HU011B/HU011D, delivery, modelo, video y smoke:
  `65 passed, 1 skipped`.
- Suite BattleZone: `166 passed, 1 skipped`.

## Smoke test

El smoke test existente pasó. El único caso omitido fue el smoke ALE real que
requiere habilitación explícita; no se creó un smoke alternativo.

## Preflight de memoria

El preflight existente se ejecutó con 32 GiB de RAM declarada y CUDA disponible:

```text
REPLAY_GIB: 6.0001983642578125
RAM_FRACTION: 0.18750619888305664
FULL_CHECKPOINT_READY: True
PREFLIGHT_READY: True
ERRORS: ()
```

No se modificó ni forzó el memory gate.

## Alcance confirmado

- DQN clásico sin PER ni DDQN.
- Preprocessing sin cambios: RGB 128×128, stack 4.
- CNN, Replay Buffer, trainer, optimizer Adam, Smooth L1, reward y gamma sin
  modificaciones.
- Sin dependencias, tracking, persistencia ni notebooks nuevos.
- No se ejecutó entrenamiento largo ni se generaron resultados de desempeño.

## Estado

**IMPLEMENTADA — LISTA PARA ENTRENAMIENTO REAL**

## Automatización HU011B

- HU011B consume directamente el `result` producido por HU011 en la misma
  ejecución y reutiliza el SHA efectivo del bootstrap.
- El último checkpoint FULL con `0 < checkpoint_step < final_step` se resuelve
  automáticamente; si no existe, el flujo falla con un mensaje explícito.
- El epsilon del video intermedio usa `LinearEpsilonSchedule` y la configuración
  efectiva de entrenamiento, incluido `decay_steps=750000` para `improved_v2`.
- El notebook no contiene run ID, SHA, step intermedio ni rutas históricas que
  el profesor deba editar. El flujo está preparado para ejecutarse con Run all.
- Validación focal de HU011/HU011B/HU011D, delivery, modelo, video y smoke:
  `65 passed, 1 skipped`.
