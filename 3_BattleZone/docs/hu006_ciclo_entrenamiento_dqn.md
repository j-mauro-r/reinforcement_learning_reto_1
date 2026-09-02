# HU006 — Ciclo de entrenamiento DQN para BattleZone

## 1. Identificación

- **ID:** HU006
- **Nombre:** Ciclo de entrenamiento DQN para BattleZone
- **Estado:** Lista para implementación
- **Dependencia previa:** HU005 — Núcleo del agente DQN
- **Habilita:** HU007 — Checkpoints, reanudación e idempotencia
- **Algoritmo vigente:** `DQN`
- **Fuentes de verdad:**
  - `enunciado_reto_1.txt`;
  - `3_BattleZone/docs/implementacion.md`;
  - `3_BattleZone/docs/lineamientos.md`;
  - `3_BattleZone/docs/arquitectura.md`;
  - `3_BattleZone/docs/hu003_pipeline_reproducible_entorno.md`;
  - `3_BattleZone/docs/hu004_decision_algoritmo.md`;
  - `3_BattleZone/docs/hu005_nucleo_agente_dqn.md`;
  - `3_BattleZone/configs/battlezone_config.yaml`.

---

## 2. Contexto

HU003 congeló el contrato del entorno y preprocessing. HU004 seleccionó `DQN` como algoritmo final elegible para BattleZone. HU005 implementó el núcleo reusable del agente: Q-Network, Online/Target Networks, Replay Buffer uniforme, epsilon-greedy, target DQN clásico, optimizer, actualización controlada y save/load básico.

HU006 debe integrar esos componentes en un **ciclo temporal de entrenamiento controlado**, sin introducir todavía infraestructura de recuperación, observabilidad, manifests ni entrenamiento largo.

La HU debe respetar el principio transversal del proyecto:

> validar barato antes de entrenar caro.

El objetivo de HU006 no es demostrar aún que DQN aprende BattleZone ni superar el baseline de HU002. El objetivo es demostrar que el flujo de entrenamiento está correctamente conectado y puede avanzar durante una ejecución corta y controlada.

---

## 3. Historia de usuario

**Como** equipo responsable del agente BattleZone,  
**quiero** integrar entorno, política epsilon-greedy, Replay Buffer y actualizaciones DQN dentro de un trainer desacoplado,  
**para** disponer de un ciclo de aprendizaje reproducible y verificable que pueda recibir checkpointing, observabilidad y smoke tests en HUs posteriores.

---

## 4. Objetivo verificable

Implementar un `trainer` reusable capaz de:

1. crear/recibir un entorno BattleZone ya configurado;
2. resetear el entorno con seed explícita;
3. seleccionar acciones mediante `DQNAgent.select_action`;
4. ejecutar `env.step()`;
5. almacenar transiciones en Replay Buffer;
6. aplicar un schedule de epsilon configurable;
7. respetar `learning_starts`;
8. ejecutar updates DQN con frecuencia configurable;
9. sincronizar Target Network con frecuencia configurable;
10. mantener contadores explícitos de timestep y episodio;
11. manejar `terminated` y `truncated` de forma no ambigua;
12. devolver un resumen estructurado de una corrida corta/controlada.

**Resultado mínimo obligatorio:** una ejecución corta del ciclo de entrenamiento debe completar pasos reales de entorno, poblar Replay Buffer, ejecutar al menos una actualización DQN cuando se cumplan los gates y producir métricas estructuradas verificables, sin checkpointing, TensorBoard ni entrenamiento largo.

---

## 5. Contratos previos que no se pueden alterar

### 5.1 Contrato HU003

Debe permanecer intacto:

- `ALE/BattleZone-v5`;
- `Discrete(18)`;
- observación `(4,128,128,3)`;
- `uint8`;
- RGB;
- `frame_stack=4`;
- sin crop;
- `frameskip=4` aplicado una sola vez;
- `repeat_action_probability=0.25`;
- reward sin clipping, normalización ni shaping.

### 5.2 Contrato HU005

HU006 debe consumir el núcleo DQN existente y no reimplementar su lógica.

Debe reutilizar contratos equivalentes a:

```python
DQNAgent.select_action(...)
DQNAgent.store_transition(...)
DQNAgent.sample_batch(...)
DQNAgent.update(...)
DQNAgent.sync_target_network(...)
```

La regla DQN permanece:

```text
next_q = max_a Q_target(next_state, a)
target = reward + gamma * (1 - done) * next_q
```

HU006 no debe introducir lógica DDQN, PER o REINFORCE.

---

## 6. Alcance

### 6.1 Incluido

#### `src/trainer.py`

Debe concentrar el ciclo temporal de entrenamiento.

Responsabilidades mínimas:

- reset inicial;
- seed de entrenamiento explícita;
- selección de acción con epsilon actual;
- `env.step(action)`;
- acumulación de recompensa por episodio;
- almacenamiento de experiencia;
- gate de `learning_starts`;
- gate de frecuencia de update;
- muestreo de Replay Buffer;
- llamada a `agent.update(...)`;
- sincronización periódica de Target Network;
- incremento de `global_step`;
- contador de episodios;
- reinicio del entorno al finalizar episodio;
- tratamiento explícito de `terminated` y `truncated`;
- retorno de resumen estructurado de la corrida.

#### Configuración

Actualizar `3_BattleZone/configs/battlezone_config.yaml` solo con parámetros necesarios para HU006.

Debe incluir, como baseline de implementación por validar:

- `training.total_timesteps` o parámetro equivalente para ejecución controlada;
- `training.learning_starts`;
- `training.train_frequency`;
- `training.target_sync_interval`;
- estrategia de epsilon;
- `epsilon.start`;
- `epsilon.end`;
- `epsilon.decay_steps`;
- seed de entrenamiento cuando no exista ya una fuente única apropiada.

Los valores deben permanecer configurables y no presentarse como hiperparámetros óptimos.

#### Tests

Crear tests focalizados del trainer y preservar la suite HU003/HU005.

#### Evidencia

Crear:

`3_BattleZone/docs/hu006_evidencia_implementacion.md`

con resultados reales de las validaciones.

### 6.2 Fuera de alcance

HU006 **no debe** implementar:

- checkpointing completo;
- resume de entrenamiento;
- idempotencia de corridas;
- persistencia de Replay Buffer;
- TensorBoard;
- callbacks de observabilidad;
- `run_manifest.json`;
- MLflow;
- evaluación formal;
- video;
- entrenamiento largo;
- comparación contra baseline;
- tuning de hiperparámetros;
- PER;
- DDQN;
- REINFORCE;
- reward shaping/clipping;
- modificación del preprocessing HU003;
- creación de lógica duplicada dentro del notebook.

Estas responsabilidades pertenecen a HU007–HU014.

---

## 7. Decisiones técnicas obligatorias

### DT01 — Trainer desacoplado

El ciclo de entrenamiento debe vivir en `src/trainer.py` o módulo equivalente de responsabilidad única.

El notebook no debe contener una segunda implementación del loop.

### DT02 — DQNAgent es la única fuente de lógica del algoritmo

El trainer no debe calcular targets DQN, losses ni Q-values manualmente.

Debe depender del contrato del agente.

### DT03 — Schedule de epsilon configurable

Implementar una función/clase simple para obtener epsilon a partir del `global_step`.

Requisito mínimo:

- epsilon inicia en `epsilon.start`;
- decae de forma explícita hasta `epsilon.end`;
- alcanza/no cruza el mínimo configurado;
- `decay_steps` debe ser configurable;
- la fórmula debe ser determinista para un `global_step` dado.

Se permite decaimiento lineal como baseline de implementación, siempre que se documente como decisión inicial por validar.

### DT04 — `learning_starts`

Antes de `learning_starts`:

- se interactúa con el entorno;
- se almacenan transiciones;
- no se ejecuta optimizer step.

Después de alcanzarlo, solo se aprende si el Replay Buffer contiene al menos `batch_size` transiciones.

### DT05 — Frecuencia de entrenamiento

Las actualizaciones solo deben dispararse cuando se cumpla el gate configurable de `train_frequency`.

El comportamiento debe ser verificable por timestep.

### DT06 — Sincronización Target

`sync_target_network()` debe ejecutarse únicamente al alcanzar el intervalo configurable `target_sync_interval`.

No debe sincronizarse Target después de cada update salvo que la configuración lo indique explícitamente.

### DT07 — `terminated` y `truncated`

Gymnasium diferencia terminación MDP de truncación externa.

Para el control del episodio:

```text
episode_done = terminated or truncated
```

Para el target DQN, la transición almacenada debe conservar una señal de terminalidad coherente y explícita.

Decisión HU006:

- `terminated=True` bloquea bootstrap;
- `truncated=True` finaliza el rollout/episodio, pero no se tratará automáticamente como terminal MDP para el target;
- si la API actual de `ReplayBuffer` solo dispone de `done`, HU006 debe realizar la adaptación mínima y documentada necesaria sin romper HU005.

Esta decisión debe contar con test específico.

### DT08 — Recompensa real

El trainer debe almacenar y acumular la recompensa entregada por el entorno sin clipping, normalización ni shaping.

### DT09 — Contadores explícitos

Mantener al menos:

- `global_step`;
- `episode_index`;
- `episode_step`;
- `episode_reward`.

Estos contadores deben quedar preparados para HU007, pero HU006 no implementa persistencia/resume.

### DT10 — Resultado estructurado

Una corrida debe devolver un objeto/dataclass/dict estructurado con al menos:

- timesteps ejecutados;
- episodios completados;
- número de updates ejecutados;
- número de sincronizaciones Target;
- epsilon inicial;
- epsilon final;
- rewards de episodios completados o resumen equivalente;
- tamaño final del Replay Buffer;
- última loss disponible si existió actualización.

No se requiere todavía persistir este resultado en disco.

---

## 8. Contratos esperados

Los nombres exactos pueden variar manteniendo claridad y responsabilidades.

Ejemplo:

```python
@dataclass
class TrainingState:
    global_step: int = 0
    episode_index: int = 0
    episode_step: int = 0
    episode_reward: float = 0.0

@dataclass
class TrainingSummary:
    total_steps: int
    completed_episodes: int
    updates: int
    target_syncs: int
    initial_epsilon: float
    final_epsilon: float
    replay_size: int
    episode_rewards: list[float]
    last_loss: float | None

class LinearEpsilonSchedule:
    def value(self, step: int) -> float: ...

class DQNTrainer:
    def train(self, total_timesteps: int, seed: int) -> TrainingSummary: ...
```

No crear una jerarquía genérica de trainers para múltiples algoritmos.

---

## 9. Flujo esperado

```text
reset(seed)
   ↓
state
   ↓
epsilon(global_step)
   ↓
agent.select_action(state, epsilon)
   ↓
env.step(action)
   ↓
store_transition(...)
   ↓
¿learning_starts cumplido?
   ├── no → continuar
   └── sí
        ↓
    ¿train_frequency cumplida?
        ├── no → continuar
        └── sí
             ↓
       sample_batch()
             ↓
        agent.update()
   ↓
¿target_sync_interval cumplido?
   ├── sí → sync_target_network()
   └── no
   ↓
actualizar contadores
   ↓
terminated or truncated?
   ├── sí → registrar episodio + reset
   └── no → continuar
```

---

## 10. Tareas

### T01 — Gate de dependencias

- confirmar HU003 `[COMPLETADA]`;
- confirmar HU004 `[COMPLETADA — DQN]`;
- confirmar HU005 mergeada y DQN vigente en `main` antes del cierre de HU006;
- confirmar suite BattleZone en verde.

### T02 — Diseñar configuración temporal

Definir baseline configurable para:

- epsilon start/end/decay;
- learning starts;
- train frequency;
- target sync interval;
- total timesteps de validación controlada.

### T03 — Implementar epsilon schedule

- fórmula simple;
- límites correctos;
- tests de inicio, mitad, final y pasos posteriores al decay.

### T04 — Implementar trainer

Integrar entorno y agente sin duplicar lógica.

### T05 — Integrar Replay Buffer

- almacenar cada transición;
- aprender únicamente con batch suficiente;
- no alterar almacenamiento CPU/uint8 de HU005.

### T06 — Integrar gates de aprendizaje

- `learning_starts`;
- `train_frequency`;
- conteo de updates.

### T07 — Integrar Target sync

- sincronización según intervalo;
- conteo de sincronizaciones;
- sin sync implícito adicional.

### T08 — Manejar episodios

- `terminated`;
- `truncated`;
- reset;
- reward acumulada;
- episode length/step.

### T09 — Resumen estructurado

Retornar evidencia suficiente para tests y HU007/HU008.

### T10 — Tests focalizados

Validar comportamiento temporal sin entrenamiento largo.

### T11 — Ejecución corta real

Ejecutar una corrida local/controlada de pocos timesteps suficiente para demostrar integración real.

No buscar performance.

### T12 — Evidencia y alcance

Actualizar evidencia HU006 y verificar que no se adelantó infraestructura futura.

---

## 11. Criterios de aceptación

### CA01 — Dependencias satisfechas

HU003 y HU004 están cerradas; HU005 DQN está mergeada antes del cierre formal de HU006.

### CA02 — Trainer desacoplado

Existe un módulo reusable de entrenamiento y no hay loop duplicado en notebook.

### CA03 — Entorno compartido

El trainer utiliza la fábrica/configuración de HU003 y no llama a `gymnasium.make("ALE/BattleZone-v5")` directamente.

### CA04 — Epsilon schedule correcto

El schedule es configurable, reproducible y respeta límites start/end.

### CA05 — Recolección de experiencia

Cada step válido puede producir una transición compatible con Replay Buffer HU005.

### CA06 — Learning starts respetado

No ocurre update antes del gate configurado.

### CA07 — Batch suficiente

No se intenta aprender con Replay Buffer menor que `batch_size`.

### CA08 — Train frequency respetada

Los updates ocurren únicamente en timesteps habilitados por configuración.

### CA09 — Target sync respetado

La Target Network se sincroniza exclusivamente según el intervalo configurado.

### CA10 — Terminal/truncation explícitos

El cierre de episodio y la señal utilizada para bootstrap están definidos y testeados sin ambigüedad.

### CA11 — Reward sin transformación

El trainer conserva la recompensa real del entorno.

### CA12 — Contadores correctos

`global_step`, episodios y métricas básicas evolucionan de forma verificable.

### CA13 — Resumen estructurado

La ejecución retorna un resumen con pasos, episodios, updates, syncs, epsilon, replay size, rewards y última loss cuando exista.

### CA14 — Ejecución controlada real

Una corrida corta real del entorno completa sin error y ejecuta al menos una actualización cuando la configuración de validación lo permite.

### CA15 — Tests BattleZone en verde

Pasan los tests HU003–HU006 relevantes.

### CA16 — Alcance preservado

No se introducen checkpoints, resume, TensorBoard, manifests, evaluación, MLflow, PER, DDQN ni entrenamiento largo.

---

## 12. Autovalidaciones obligatorias

### AV01 — Dependencias
**PASS:** HU003/HU004 cerradas y HU005 DQN disponible como dependencia vigente.

### AV02 — Configuración
**PASS:** parámetros HU006 centralizados y etiquetados como baseline por validar.

### AV03 — Epsilon schedule
**PASS:** valores start/intermedios/end/post-decay son correctos y acotados.

### AV04 — Reset y step
**PASS:** trainer obtiene observación HU003 válida y ejecuta acción `0..17`.

### AV05 — Replay integration
**PASS:** transitions almacenadas aumentan Replay Buffer con shapes/dtypes esperados.

### AV06 — Learning starts
**PASS:** contador de updates permanece en cero antes del gate.

### AV07 — Batch gate
**PASS:** trainer no llama update sin batch suficiente.

### AV08 — Train frequency
**PASS:** número/timesteps de updates coinciden con la política configurada.

### AV09 — Target sync
**PASS:** número/timesteps de sync coinciden con intervalo configurado.

### AV10 — Terminated
**PASS:** finaliza episodio y bloquea bootstrap.

### AV11 — Truncated
**PASS:** finaliza rollout sin convertir silenciosamente truncation en terminal MDP.

### AV12 — Reward passthrough
**PASS:** reward acumulada/almacenada coincide con entorno controlado.

### AV13 — Contadores
**PASS:** `global_step`, episodio y episode_step son consistentes.

### AV14 — Summary
**PASS:** salida estructurada contiene métricas mínimas y valores coherentes.

### AV15 — Corrida real corta
**PASS:** ejecución local/controlada completa y, con configuración adecuada, produce al menos un update DQN real con loss finita.

### AV16 — Scope
**PASS:** no Assault, no PER/DDQN/REINFORCE, no MLflow, no checkpoint/resume, no TensorBoard, no manifest, no evaluación formal, no entrenamiento largo.

---

## 13. Tests mínimos esperados

Crear preferiblemente:

`3_BattleZone/tests/test_trainer.py`

Debe cubrir como mínimo:

1. epsilon schedule start/end/clamp;
2. no update antes de `learning_starts`;
3. no update con Replay Buffer insuficiente;
4. update según `train_frequency`;
5. Target sync según `target_sync_interval`;
6. transición almacenada por step;
7. reward acumulada;
8. reset al terminar episodio;
9. tratamiento de `terminated`;
10. tratamiento de `truncated`;
11. acciones siempre válidas;
12. contadores;
13. summary estructurado;
14. integración con `DQNAgent` real en una ejecución controlada;
15. preservación de los tests HU003/HU005.

Los tests unitarios pueden usar entornos/agentes controlados para verificar temporalidad sin costo Atari.

La evidencia final debe incluir además una corrida corta con el entorno BattleZone real.

---

## 14. Definition of Done

HU006 puede marcarse `[COMPLETADA]` únicamente cuando:

- [ ] HU005 DQN está mergeada y vigente en `main`;
- [ ] existe `src/trainer.py` o módulo equivalente con responsabilidad clara;
- [ ] epsilon schedule configurable está implementado;
- [ ] `learning_starts` está implementado y testeado;
- [ ] `train_frequency` está implementado y testeado;
- [ ] `target_sync_interval` está implementado y testeado;
- [ ] Replay Buffer HU005 se integra sin duplicación;
- [ ] `terminated` y `truncated` tienen tratamiento explícito;
- [ ] reward se conserva sin transformación;
- [ ] contadores globales/episódicos son correctos;
- [ ] existe un summary estructurado;
- [ ] una corrida corta real completa sin errores;
- [ ] al menos una actualización DQN real ocurre bajo configuración controlada;
- [ ] loss observada es finita;
- [ ] tests focalizados y suite BattleZone relevante pasan;
- [ ] existe `hu006_evidencia_implementacion.md` con outputs reales;
- [ ] AV01–AV16 están en PASS o excepción explícitamente aprobada;
- [ ] no se adelantó HU007+;
- [ ] no se modificó Assault;
- [ ] no se introdujo MLflow;
- [ ] PR revisado y listo para merge.

---

## 15. Evidencias esperadas

La evidencia HU006 debe registrar como mínimo:

1. rama y commit;
2. configuración temporal utilizada;
3. epsilon inicial/final de la corrida;
4. timesteps solicitados/ejecutados;
5. episodios completados;
6. Replay Buffer final;
7. número de updates;
8. número de Target syncs;
9. última loss finita si hubo update;
10. rewards de episodios completados o resumen equivalente;
11. tratamiento observado de `terminated`/`truncated`;
12. comandos ejecutados;
13. resultados de tests;
14. CA01–CA16;
15. AV01–AV16;
16. limitaciones y pendientes para HU007.

No se debe presentar recompensa de esta corrida corta como evidencia de performance.

---

## 16. Riesgos y mitigaciones

### R01 — Entrenar antes de poblar Replay
**Mitigación:** gates de `learning_starts` y `batch_size`.

### R02 — Actualizar demasiado frecuentemente
**Mitigación:** `train_frequency` centralizada y testeada.

### R03 — Sincronizar Target accidentalmente en cada update
**Mitigación:** método explícito + intervalo + contador verificable.

### R04 — Confundir truncation con terminal MDP
**Mitigación:** semántica explícita y test separado.

### R05 — Duplicar lógica de DQN en trainer
**Mitigación:** trainer solo orquesta contratos de `DQNAgent`.

### R06 — Convertir HU006 en smoke/entrenamiento largo
**Mitigación:** ejecución corta, sin criterio de reward/performance.

### R07 — Adelantar HU007/HU008
**Mitigación:** no checkpoints, resume, callbacks ni TensorBoard en esta HU.

### R08 — Parámetros iniciales tratados como óptimos
**Mitigación:** etiquetarlos como baseline de implementación por validar; tuning pertenece a HU012.

### R09 — Contaminar BattleZone con Assault
**Mitigación:** diff scope obligatorio y cero imports desde `2_Assault/`.

---

## 17. Resultado esperado para HU007

HU007 debe recibir un ciclo DQN funcional con:

- trainer reusable;
- entorno HU003 integrado;
- agente HU005 integrado;
- Replay Buffer funcionando en flujo real;
- epsilon schedule;
- learning starts;
- train frequency;
- Target sync interval;
- contadores globales y episódicos;
- manejo explícito de terminated/truncated;
- summary estructurado;
- tests en verde;
- evidencia de una corrida corta real.

HU007 añadirá persistencia completa, checkpoints, resume e idempotencia sin rediseñar el ciclo temporal básico de HU006.
