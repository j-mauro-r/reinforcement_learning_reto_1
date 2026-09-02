# Evidencia de implementación — HU005 Núcleo del agente DQN (BattleZone)

## 1. Identificación

- HU: HU005
- PR: #23
- Rama histórica del PR: `feature/battlezone-hu005-ddqn-agent-core`
- Estado actual: **[COMPLETADA] — implementación y validación técnica cerradas con dependencias mergeadas**
- Algoritmo vigente: `DQN`
- Dependencia externa de cierre: PR #24 (corrección HU004) mergeado.

## 2. Fecha y entorno de ejecución

- Fecha/hora UTC: `2026-09-02T17:39:49.353066+00:00`
- Runtime local observado:
	- python: `3.12.11`
	- platform: `macOS-26.5.2-arm64-arm-64bit`
	- cpu_count: `12`
	- ram_gb: `32.0`
	- gymnasium: `1.1.1`
	- ale_py: `0.10.1`
	- numpy: `2.5.2`
	- pillow: `12.3.0`
	- pyyaml: `6.0.3`
	- torch: `2.13.0`
	- gpu_available: `False`

## 3. Contrato HU003 preservado

Sin cambios:

- `environment: ALE/BattleZone-v5`
- `action_space: Discrete(18)`
- `observation_shape: (4,128,128,3)`
- `dtype: uint8`
- `color: RGB`
- `frame_stack: 4`
- `crop: none`
- `frameskip: 4`
- `repeat_action_probability: 0.25`
- `reward_transform: none`

## 4. Implementación DQN

### Q-Network

Archivo: `3_BattleZone/src/network.py`.

- clase `BattleZoneQNetwork`;
- acepta entrada individual `(4,128,128,3)` y batch `(N,4,128,128,3)`;
- conversión explícita `uint8 -> float32`;
- escalado explícito `/255.0`;
- reordenamiento explícito a NCHW con `12` canales efectivos (`4*3`);
- salida exacta `(batch_size, 18)`.

### Replay Buffer

Archivo: `3_BattleZone/src/replay_buffer.py`.

- uniforme;
- CPU RAM;
- estado en `uint8`;
- transición `state, action, reward, next_state, done`;
- capacidad configurable;
- muestreo uniforme sin reemplazo;
- validación de shape y dtype;
- error explícito si `batch_size > len(buffer)`;
- sin PER, prioridades, SumTree, ni importance-sampling.

### Agente

Archivo: `3_BattleZone/src/agent.py`.

- clase `DQNAgent`;
- `from_config(...)`;
- `select_action(...)`;
- `store_transition(...)`;
- `sample_batch(...)`;
- `compute_targets(...)`;
- `update(...)`;
- `sync_target_network(...)`;
- `state_dict(...)`;
- `load_state_dict(...)`.

Se verificó que:

- Online y Target son objetos distintos;
- sincronizan pesos al inicio;
- optimizer actualiza solo Online;
- Target tiene gradientes desactivados.

## 5. Regla DQN implementada

Se implementa DQN clásico:

```text
next_q = max_a Q_target(next_state, a)
target = reward + gamma * (1 - done) * next_q
```

No se usa selección de acción con `argmax(Q_online(next_state))` para el target.

## 6. Configuración centralizada

Archivo: `3_BattleZone/configs/battlezone_config.yaml`.

- `algorithm: "DQN"`
- sección `dqn` con baseline de implementación por validar:
	- `gamma`
	- `learning_rate`
	- `batch_size`
	- `replay_buffer.capacity`
	- parámetros de red
	- `optimizer` y `loss`
	- `epsilon` de API

## 7. Comandos ejecutados y resultados reales

Comandos:

```bash
python -m compileall -q 3_BattleZone/src 3_BattleZone/tests
PYTHONPATH=3_BattleZone python -m pytest 3_BattleZone/tests/test_network.py -q
PYTHONPATH=3_BattleZone python -m pytest 3_BattleZone/tests/test_replay_buffer.py -q
PYTHONPATH=3_BattleZone python -m pytest 3_BattleZone/tests/test_agent.py -q
PYTHONPATH=3_BattleZone python -m pytest 3_BattleZone/tests -q
```

Resultados:

- compileall: `PASS`
- `test_network.py`: `6 passed in 0.66s`
- `test_replay_buffer.py`: `6 passed in 0.06s`
- `test_agent.py`: `13 passed in 1.45s`
- suite completa `3_BattleZone/tests`: `34 passed in 2.86s`

## 8. Evidencia técnica controlada (salida real)

- `FORWARD_SHAPE (2, 18)`
- `FORWARD_FINITE True`
- `FORWARD_DTYPE torch.float32`
- `INIT_ONLINE_TARGET_EQUAL True`
- `INIT_ONLINE_TARGET_DISTINCT_OBJECTS True`
- `DQN_TARGET_VALUE 50.5`
- `DQN_EXPECTED_VALUE 50.5`
- `DDQN_REFERENCE_VALUE 4.96`
- `TERMINAL_TARGET_DONE_TRUE 2.5`
- `TERMINAL_TARGET_DONE_FALSE 48.5`
- `UPDATE_LOSS 0.2001556009054184`
- `ONLINE_CHANGED_AFTER_UPDATE True`
- `TARGET_UNCHANGED_AFTER_UPDATE True`
- `TARGET_GRADS_NONE True`
- `SYNC_RESTORES_EQUALITY True`
- `SAVE_LOAD_ONLINE_RESTORED True`
- `SAVE_LOAD_TARGET_RESTORED True`
- `SAVE_LOAD_GAMMA_RESTORED True`
- `SAVE_LOAD_GREEDY_ACTION_MATCH True`
- `SAVE_LOAD_GREEDY_ACTION 15`
- `STRUCTURAL_MISMATCH_REJECTED True`
- `EPSILON_ONE_ACTION_MIN 0`
- `EPSILON_ONE_ACTION_MAX 17`
- `EPSILON_ONE_UNIQUE_COUNT 17`

## 9. CA01–CA16

| CA | Estado | Evidencia |
|---|---|---|
| CA01 HU004 corregida a DQN y PR #24 mergeado | PASS | PR #24 mergeado y decisión DQN consolidada en `main`. |
| CA02 Q-Network compatible | PASS | Salida `(batch,18)` en tests y evidencia controlada. |
| CA03 dtype/layout | PASS | Conversión explícita `uint8->float32`, `/255`, NCHW. |
| CA04 Online/Target independientes | PASS | Igualdad inicial con objetos distintos. |
| CA05 Replay uniforme CPU/uint8 | PASS | Tests de add/sample/shape/dtype/capacidad. |
| CA06 epsilon-greedy | PASS | Tests epsilon `0`, `1`, e inválido. |
| CA07 target DQN clásico | PASS | Caso controlado usa `max(Target)` y difiere de referencia DDQN. |
| CA08 terminal masking | PASS | `done=True -> target=reward` validado. |
| CA09 update real de Online | PASS | `UPDATE_LOSS` finita y Online cambia. |
| CA10 Target protegido | PASS | Target inmutable durante update y sin gradientes. |
| CA11 sync explícito | PASS | `SYNC_RESTORES_EQUALITY True`. |
| CA12 save/load consistente | PASS | Restaura redes, optimizer y gamma; rechaza incompatibilidad estructural. |
| CA13 configuración centralizada | PASS | `algorithm: DQN` y sección `dqn` versionada. |
| CA14 tests focalizados | PASS | `test_network`, `test_replay_buffer`, `test_agent` en verde. |
| CA15 alcance sin HU006+/Assault/MLflow/PER | PASS | Alcance verificado por diff y búsquedas. |
| CA16 evidencia DQN real | PASS | Esta evidencia usa solo ejecuciones DQN re-ejecutadas. |

## 10. AV01–AV16

| AV | Estado | Evidencia |
|---|---|---|
| AV01 Dependencias | PASS | HU003 intacta y PR #24 mergeado con decisión DQN vigente. |
| AV02 Forward | PASS | `FORWARD_SHAPE (2, 18)`, finito. |
| AV03 Entrada individual | PASS | Test de forward individual. |
| AV04 Sincronización inicial | PASS | Online/Target iguales y distintos objetos. |
| AV05 Epsilon-greedy | PASS | epsilon `0`, `1`, inválido. |
| AV06 Replay add/sample | PASS | Tests de buffer con shapes/dtypes/error controlado. |
| AV07 Replay CPU/uint8 | PASS | Estados en arrays `uint8` de CPU. |
| AV08 Target DQN | PASS | `DQN_TARGET_VALUE=50.5` vs `DDQN_REFERENCE_VALUE=4.96`. |
| AV09 Terminal mask | PASS | target terminal igual a reward. |
| AV10 Update real | PASS | loss finita y cambio Online. |
| AV11 Target inmutable | PASS | sin cambio/sin gradientes durante update. |
| AV12 Sync | PASS | target se realinea con sync. |
| AV13 Save/load | PASS | restaura gamma/parámetros y rechaza incompatibilidad. |
| AV14 Configuración | PASS | construcción desde `DQNAgent.from_config(...)`. |
| AV15 Scope | PASS | sin Assault, PER, trainer E2E, TensorBoard, checkpoint completo, MLflow. |
| AV16 Anti-alucinación | PASS | resultados registrados provienen de ejecución local real. |

## 11. Limitaciones reales

- no se ejecutó entrenamiento E2E de BattleZone;
- no se afirma que DQN ya aprende BattleZone;
- no hay comparación de rendimiento contra baseline aleatorio en esta HU;
- memoria/VRAM y throughput de entrenamiento largo: `NO MEDIDO`.

## 12. Estado de cierre HU005

HU005 queda en estado:

**[COMPLETADA]**

Condiciones de cierre verificadas:

- PR #24 mergeado.
- PR #23 mergeado.
- Validación técnica DQN en verde.
