# HU005 — Núcleo del agente DQN para BattleZone

## 1. Identificación

- **ID:** HU005
- **Nombre:** Núcleo del agente DQN para BattleZone
- **Estado:** Implementada y validada técnicamente — pendiente de cierre formal (PR #24 sin merge) y revisión/merge de PR #23
- **Dependencia previa:** HU004 — Selección formal del algoritmo
- **Dependencia correctiva:** PR #24 debe quedar mergeado antes del cierre formal de HU005.
- **Habilita:** HU006 — Ciclo de entrenamiento
- **Algoritmo vigente para BattleZone:** `DQN`
- **Fuentes de verdad:** `enunciado_reto_1.txt`, `3_BattleZone/docs/implementacion.md`, `3_BattleZone/docs/lineamientos.md`, `3_BattleZone/docs/arquitectura.md`, HU003, `3_BattleZone/docs/hu004_decision_algoritmo.md` una vez incorporada la corrección del PR #24, y `3_BattleZone/configs/battlezone_config.yaml`.

## 2. Contexto y corrección

La versión inicial de HU005 implementó DDQN porque HU004 lo había seleccionado como ganador técnico. Posteriormente se identificó una restricción global del reto: deben utilizarse al menos dos métodos distintos entre los tres problemas. Como DDQN ya se usa en LunarLander y Assault, HU004 fue corregida para seleccionar `DQN` como mejor alternativa elegible para BattleZone.

HU005 debe por tanto implementar **DQN clásico**, preservando el contrato HU003:

- `ALE/BattleZone-v5`;
- `Discrete(18)`;
- observación `(4,128,128,3) uint8`;
- RGB, `frame_stack=4`, sin crop;
- `frameskip=4`;
- `repeat_action_probability=0.25`;
- reward sin clipping/shaping.

No se ejecuta entrenamiento E2E en esta HU.

## 3. Objetivo verificable

Construir y validar el núcleo reusable del agente DQN con:

1. Q-Network convolucional compatible con HU003;
2. Online Network y Target Network independientes;
3. Replay Buffer uniforme en CPU/`uint8`;
4. epsilon-greedy;
5. target DQN clásico;
6. optimizer solo sobre Online;
7. una actualización real sobre batch controlado;
8. sincronización explícita Online → Target;
9. save/load básico consistente;
10. configuración centralizada y tests focalizados.

## 4. Decisión algorítmica obligatoria

### DT01 — DQN clásico

El target debe calcularse como:

```text
next_q = max_a Q_target(next_state, a)
target = reward + gamma * (1 - done) * next_q
```

No debe usarse la separación DDQN de selección con Online y evaluación con Target.

### DT02 — Online y Target

- misma arquitectura;
- instancias diferentes;
- pesos sincronizados inicialmente;
- Target fuera del optimizer;
- Target sin gradientes durante el cálculo del target;
- Target cambia únicamente mediante sync explícito.

### DT03 — Replay Buffer

- uniforme;
- CPU RAM;
- estados `uint8`;
- capacidad configurable;
- sin PER, prioridades, SumTree ni importance-sampling weights.

### DT04 — Configuración centralizada

Los parámetros reutilizables del agente/red deben provenir de configuración versionada. El código reusable no debe duplicar como defaults los valores de `gamma`, learning rate, batch size, capacidad de replay o arquitectura CNN.

### DT05 — Save/load básico

El estado exportado debe restaurarse de forma coherente. Metadatos estructurales incompatibles deben rechazarse explícitamente y `gamma` debe restaurarse cuando forma parte del estado serializado.

## 5. Componentes

### `src/network.py`

- entrada individual o batch HU003;
- conversión explícita `uint8 -> float32`;
- escalado `/255`;
- layout `(N,T,H,W,C) -> (N,T*C,H,W)`;
- salida `[batch,18]`;
- parámetros estructurales recibidos explícitamente.

### `src/replay_buffer.py`

- transición `state, action, reward, next_state, done`;
- add/sample/len;
- almacenamiento CPU/uint8;
- muestreo uniforme.

### `src/agent.py`

Contrato esperado:

```python
class DQNAgent:
    @classmethod
    def from_config(cls, config): ...
    def select_action(self, state, epsilon): ...
    def compute_targets(self, batch): ...
    def update(self, batch): ...
    def sync_target_network(self): ...
    def state_dict(self): ...
    def load_state_dict(self, state): ...
```

## 6. Tareas

- **T01:** verificar HU003 y la decisión DQN de HU004/PR #24.
- **T02:** mantener Q-Network compatible con HU003.
- **T03:** mantener Replay Buffer uniforme.
- **T04:** reemplazar lógica DDQN por target DQN clásico.
- **T05:** mantener epsilon-greedy.
- **T06:** validar terminal masking.
- **T07:** ejecutar update real controlado.
- **T08:** validar Online cambia y Target no cambia durante update.
- **T09:** validar sync explícito.
- **T10:** corregir save/load para restaurar/validar estado coherentemente.
- **T11:** centralizar configuración mediante `algorithm: DQN` y sección `dqn`.
- **T12:** actualizar tests y evidencia sin afirmar aprendizaje real.

## 7. Criterios de aceptación

- **CA01:** HU004 corregida selecciona DQN y PR #24 está mergeado antes del cierre.
- **CA02:** Q-Network produce exactamente 18 Q-values por muestra.
- **CA03:** dtype/layout se convierten explícitamente sin alterar HU003.
- **CA04:** Online/Target son independientes e inicialmente iguales.
- **CA05:** Replay Buffer es uniforme, CPU/uint8 y sin PER.
- **CA06:** epsilon-greedy devuelve acciones válidas; epsilon 0 es greedy y epsilon 1 exploratorio.
- **CA07:** target DQN usa `max(Q_target(next_state))` y no la regla DDQN.
- **CA08:** terminales no reciben bootstrap.
- **CA09:** update controlado produce loss finita y modifica Online.
- **CA10:** Target no recibe gradientes ni cambia durante optimizer step.
- **CA11:** sync explícito realinea Target con Online.
- **CA12:** save/load restaura estado compatible y rechaza metadatos estructurales incompatibles.
- **CA13:** configuración DQN está centralizada; módulos reusables reciben valores explícitos/configurados.
- **CA14:** tests focalizados BattleZone pasan después de la conversión a DQN.
- **CA15:** no hay entrenamiento E2E, Assault, PER, TensorBoard, checkpoints completos ni MLflow.
- **CA16:** evidencia versionada distingue resultados re-ejecutados de resultados históricos DDQN.

## 8. Autovalidaciones oficiales

- **AV01 Dependencias:** HU003 intacta y HU004 corregida a DQN.
- **AV02 Forward:** `[batch,18]`, finito.
- **AV03 Entrada individual:** adapter inequívoco.
- **AV04 Sincronización inicial:** Online/Target iguales pero independientes.
- **AV05 Epsilon-greedy:** 0 greedy, 1 exploratorio válido.
- **AV06 Replay:** add/sample/shapes/dtypes/error de sample inválido.
- **AV07 CPU/uint8:** estados persistidos eficientemente.
- **AV08 Target DQN:** caso controlado donde Online y Target tienen argmax diferente demuestra uso de `max(Target)`.
- **AV09 Terminal mask:** `done=True` implica target=reward.
- **AV10 Update real:** loss finita y Online cambia.
- **AV11 Target protegido:** sin gradientes/cambio durante update.
- **AV12 Sync:** Target vuelve a coincidir con Online.
- **AV13 Save/load:** gamma/params restaurados; incompatibilidad estructural rechazada.
- **AV14 Configuración:** construcción desde config versionada y sin defaults algorítmicos duplicados.
- **AV15 Scope:** sin Assault/PER/trainer/TensorBoard/checkpoint completo/MLflow.
- **AV16 Anti-alucinación:** no reutilizar resultados DDQN como evidencia DQN; re-ejecutar tests antes del cierre.

## 9. Definition of Done

HU005 puede cerrarse únicamente cuando:

- PR #24 esté mergeado y DQN sea la decisión vigente en `main`;
- `DQNAgent` y target DQN estén implementados;
- configuración `DQN`/`dqn` esté versionada;
- tests focalizados y suite BattleZone sean re-ejecutados y pasen;
- evidencia HU005 se actualice con outputs DQN reales;
- CA01–CA16 y AV01–AV16 estén en PASS;
- no se adelante HU006+;
- PR #23 sea auditado y mergeado.

## 10. Fuera de alcance

No implementar entrenamiento E2E, schedules temporales completos, checkpoints/resume completos, TensorBoard, `run_manifest`, tuning, PER, REINFORCE, reward shaping, cambios HU003, MLflow ni código/imports desde `2_Assault/`.

## 11. Resultado esperado para HU006

HU006 recibirá un núcleo DQN con Q-Network, Online/Target, Replay Buffer uniforme, epsilon-greedy, target DQN, optimizer, update unitario, sync explícito, configuración centralizada y tests en verde. HU006 integrará la recolección continua de experiencia y el ciclo temporal de entrenamiento.
