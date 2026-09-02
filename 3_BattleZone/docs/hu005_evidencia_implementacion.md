# Evidencia de implementación — HU005 Núcleo del agente DQN (BattleZone)

## 1. Identificación

- HU: HU005
- PR: #23
- Rama histórica del PR: `feature/battlezone-hu005-ddqn-agent-core`
- Estado: **realineada a DQN — pendiente de reejecución local y revisión**
- Algoritmo vigente: `DQN`
- Dependencia: la corrección documental HU004 del PR #24 debe estar mergeada antes del cierre formal de HU005.

## 2. Motivo de la corrección

La primera implementación de HU005 se construyó como DDQN porque HU004 lo había seleccionado como ganador técnico. Posteriormente se identificó una restricción global del reto: deben utilizarse al menos dos métodos distintos entre los tres ejercicios. Dado que DDQN ya se usa en LunarLander y Assault, BattleZone se realinea a `DQN`, segunda alternativa de la matriz HU004 y mejor candidato elegible.

Los resultados numéricos y tests ejecutados previamente sobre DDQN se consideran **evidencia histórica y no evidencia válida de la implementación DQN actual**.

## 3. Contrato HU003 preservado

Sin cambios:

- `ALE/BattleZone-v5`
- `Discrete(18)`
- observación `(4,128,128,3)`
- `uint8`
- RGB
- `frame_stack=4`
- `frameskip=4`
- `repeat_action_probability=0.25`
- `reward_transform=none`

## 4. Implementación DQN actual

### Q-Network

`3_BattleZone/src/network.py`

- CNN propia de BattleZone.
- Entrada individual o batch HU003.
- Conversión explícita `uint8 -> float32` y `/255`.
- Layout `(N,T,H,W,C) -> (N,T*C,H,W)`.
- Salida `[batch,18]`.
- Parámetros estructurales recibidos explícitamente desde configuración/constructor; no quedan como defaults reutilizables duplicados.

### Replay Buffer

`3_BattleZone/src/replay_buffer.py`

- uniforme;
- CPU RAM;
- estados `uint8`;
- capacidad configurable;
- sin PER/prioridades/IS-weights.

### DQNAgent

`3_BattleZone/src/agent.py`

- clase `DQNAgent`;
- Online Network y Target Network independientes;
- Target fuera del optimizer y sin gradientes;
- epsilon-greedy;
- Replay Buffer uniforme integrado;
- `from_config()` para construir desde configuración versionada;
- update controlado;
- sync explícito;
- save/load básico con validación estructural.

## 5. Regla DQN implementada

La implementación actual usa:

```text
next_q = max_a Q_target(next_state, a)
target = reward + gamma * (1 - done) * next_q
```

Esto reemplaza explícitamente la regla DDQN anterior de selección con Online + evaluación con Target.

El test DQN construye un caso donde el argmax de Online difiere del argmax de Target y verifica que el target utilice `max(Target)`.

## 6. Configuración centralizada

`3_BattleZone/configs/battlezone_config.yaml` ahora contiene:

```text
algorithm: DQN
dqn: ...
```

La sección `dqn` conserva como **baseline de implementación por validar**:

- device;
- gamma;
- learning rate;
- batch size;
- Replay Buffer;
- arquitectura de red;
- optimizer;
- loss;
- epsilon de API.

Los módulos reutilizables reciben estos valores explícitamente; no se consideran hiperparámetros optimizados.

## 7. Correcciones derivadas de la auditoría previa

### Configuración

Se eliminó la dependencia de defaults algorítmicos duplicados en `DQNAgent` y `BattleZoneQNetwork`. `DQNAgent.from_config()` permite construir el núcleo desde la configuración versionada.

### Save/load

`state_dict()` exporta parámetros y metadatos. `load_state_dict()` ahora:

- restaura `gamma`;
- restaura Online/Target/optimizer;
- valida `action_dim`, `state_shape` y `batch_size`;
- rechaza estados estructuralmente incompatibles.

## 8. Tests actualizados

Los tests fueron reescritos para validar DQN:

- `test_network.py` usa configuración explícita de arquitectura;
- `test_agent.py` importa `DQNAgent`;
- prueba de target DQN con `max(Q_target)`;
- terminal masking;
- Online cambia / Target no cambia;
- sync explícito;
- save/load restaura `gamma`;
- save/load rechaza incompatibilidad estructural;
- construcción desde configuración versionada.

## 9. Estado de ejecución después de la conversión

**PENDIENTE DE REEJECUCIÓN LOCAL.**

Los resultados DDQN anteriores (`30 passed`, loss y targets registrados previamente) **no deben reutilizarse** para marcar DQN como PASS.

Antes de cerrar HU005 se debe ejecutar nuevamente, como mínimo:

```bash
python -m compileall -q 3_BattleZone/src 3_BattleZone/tests
PYTHONPATH=3_BattleZone python -m pytest 3_BattleZone/tests/test_network.py -q
PYTHONPATH=3_BattleZone python -m pytest 3_BattleZone/tests/test_replay_buffer.py -q
PYTHONPATH=3_BattleZone python -m pytest 3_BattleZone/tests/test_agent.py -q
PYTHONPATH=3_BattleZone python -m pytest 3_BattleZone/tests -q
```

Los outputs reales deberán sustituir esta sección antes del cierre.

## 10. CA01–CA16

| CA | Estado actual |
|---|---|
| CA01 HU004 corregida a DQN y PR #24 mergeado | PENDIENTE |
| CA02 Q-Network compatible | IMPLEMENTADO — pendiente reejecución |
| CA03 dtype/layout | IMPLEMENTADO — pendiente reejecución |
| CA04 Online/Target independientes | IMPLEMENTADO — pendiente reejecución |
| CA05 Replay uniforme CPU/uint8 | IMPLEMENTADO — pendiente reejecución |
| CA06 epsilon-greedy | IMPLEMENTADO — pendiente reejecución |
| CA07 target DQN clásico | IMPLEMENTADO — pendiente reejecución |
| CA08 terminal masking | IMPLEMENTADO — pendiente reejecución |
| CA09 update real de Online | IMPLEMENTADO — pendiente reejecución |
| CA10 Target protegido | IMPLEMENTADO — pendiente reejecución |
| CA11 sync explícito | IMPLEMENTADO — pendiente reejecución |
| CA12 save/load consistente | IMPLEMENTADO — pendiente reejecución |
| CA13 configuración centralizada | IMPLEMENTADO — pendiente reejecución |
| CA14 tests focalizados | PENDIENTE REEJECUCIÓN |
| CA15 alcance sin HU006+/Assault/MLflow/PER | IMPLEMENTADO — pendiente auditoría final |
| CA16 evidencia DQN real | PENDIENTE REEJECUCIÓN |

## 11. AV01–AV16

Todas las AV quedan **pendientes de validación final** hasta completar dos gates:

1. merge del PR #24;
2. reejecución local del código DQN actualizado.

No se marca ninguna AV técnica como PASS usando resultados DDQN históricos.

## 12. Limitaciones

- No se ejecutó entrenamiento E2E.
- No existe evidencia todavía de que DQN supere el baseline aleatorio.
- No se midió memoria/VRAM o throughput de entrenamiento largo.
- Los hiperparámetros siguen siendo baseline de implementación.
- El nombre de la rama conserva `ddqn` por trazabilidad histórica del PR; el contenido vigente es DQN.

## 13. Pendientes antes del cierre

1. Merge del PR #24.
2. Reejecutar tests localmente.
3. Registrar outputs DQN reales en este documento.
4. Auditar nuevamente PR #23.
5. Solo entonces marcar HU005 `[COMPLETADA]` y hacer merge.
