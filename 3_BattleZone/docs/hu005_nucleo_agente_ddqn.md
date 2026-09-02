# HU005 — Núcleo del agente DDQN para BattleZone

## 1. Identificación

- **ID:** HU005
- **Nombre:** Núcleo del agente DDQN para BattleZone
- **Estado:** Lista para implementación
- **Dependencia previa:** HU004 — Selección formal del algoritmo `[COMPLETADA]`
- **Habilita:** HU006 — Ciclo de entrenamiento
- **Algoritmo fijado por HU004:** `DDQN`
- **Fuentes de verdad:**
  - `enunciado_reto_1.txt`;
  - `3_BattleZone/docs/implementacion.md`;
  - `3_BattleZone/docs/lineamientos.md`;
  - `3_BattleZone/docs/arquitectura.md`;
  - `3_BattleZone/docs/hu003_pipeline_reproducible_entorno.md`;
  - `3_BattleZone/docs/hu003_evidencia_implementacion.md`;
  - `3_BattleZone/docs/hu004_decision_algoritmo.md`;
  - `3_BattleZone/configs/battlezone_config.yaml`.

---

## 2. Contexto y problema

HU003 congeló el contrato perceptual de BattleZone y HU004 seleccionó formalmente `DDQN` como mejor hipótesis inicial para el agente.

El contrato de entrada vigente es:

- entorno `ALE/BattleZone-v5`;
- action space `Discrete(18)`;
- observación final `(4, 128, 128, 3)`;
- `uint8`;
- RGB;
- `frame_stack=4`;
- sin crop;
- `frameskip=4` aplicado una sola vez;
- `repeat_action_probability=0.25`;
- reward sin clipping, normalización ni shaping.

HU002 mostró además una señal de recompensa muy sparse y alta variabilidad. HU004 concluyó que DDQN ofrece el mejor balance esperado entre estabilidad, complejidad y costo para la primera implementación, pero esa ventaja todavía no está validada por entrenamiento real.

HU005 debe construir únicamente el **núcleo reusable y verificable del agente DDQN**, sin integrar todavía el ciclo completo entorno → experiencia → aprendizaje continuo, responsabilidad de HU006.

---

## 3. Historia de usuario

**Como** equipo responsable del agente BattleZone,  
**quiero** disponer de un núcleo DDQN modular, probado y compatible con el contrato de HU003,  
**para** que HU006 pueda integrar un ciclo de entrenamiento sin descubrir errores básicos de red, memoria, exploración o actualización de pesos durante una corrida costosa.

---

## 4. Objetivo verificable

Implementar y validar, de forma independiente del ciclo completo de entrenamiento:

1. una Q-Network convolucional compatible con el estado visual de HU003;
2. Online Network y Target Network;
3. Replay Buffer uniforme en CPU;
4. selección de acciones epsilon-greedy;
5. cálculo correcto del target DDQN;
6. optimizer y una actualización real de pesos sobre un batch controlado;
7. interfaces básicas de estado/save-load necesarias para persistencia posterior;
8. tests focalizados que demuestren contratos, shapes, independencia Online/Target y aprendizaje unitario.

**Resultado mínimo obligatorio:** un forward pass válido y al menos una actualización real del agente sobre datos controlados, sin ejecutar entrenamiento E2E en BattleZone.

---

## 5. Alcance

### 5.1 Incluido

#### `src/network.py`

Debe contener la arquitectura neuronal reusable del agente DDQN.

Requisitos:

- entrada compatible con el contrato HU003;
- salida de tamaño `18`, una Q-value por acción;
- conversión explícita y controlada desde el layout del entorno a un layout compatible con PyTorch;
- conversión de `uint8` a tensor de punto flotante dentro del límite agente/red, sin modificar el contrato del entorno;
- escalado de píxeles documentado y probado si se utiliza;
- arquitectura CNN razonable para entrada visual, sin sobredimensionamiento no justificado;
- cálculo del tamaño del feature map sin números mágicos frágiles;
- forward determinista para un estado fijo en modo `eval()`;
- docstrings estilo Google para APIs públicas.

La arquitectura exacta de capas puede definirse en HU005, pero debe quedar documentada y justificada. No debe copiarse de Assault.

#### `src/replay_buffer.py`

Debe implementar Replay Buffer **uniforme**, porque HU004 no seleccionó PER.

Requisitos:

- almacenamiento en CPU;
- observaciones conservadas como `uint8` mientras sea razonable;
- capacidad configurable;
- inserción de transiciones;
- muestreo uniforme de batch;
- contrato claro para `state`, `action`, `reward`, `next_state`, `terminated`/`done`;
- shapes y dtypes consistentes;
- error explícito si se intenta muestrear más elementos de los disponibles;
- no incluir prioridades, importance-sampling weights ni estructuras PER.

HU005 no debe fijar una capacidad de entrenamiento grande sin evidencia de memoria. Los tests deben usar capacidades pequeñas/controladas. La capacidad real de entrenamiento deberá quedar configurable y validarse en HU006/HU009.

#### `src/agent.py`

Debe encapsular la lógica DDQN propia del agente.

Requisitos mínimos:

- construcción de Online Network;
- construcción de Target Network con arquitectura equivalente;
- sincronización inicial Online → Target;
- Target Network desacoplada del optimizer;
- selección epsilon-greedy;
- acción greedy mediante `argmax` de Online Network;
- acción aleatoria válida dentro de `0..17`;
- cálculo de targets DDQN:
  - Online Network selecciona la mejor acción para `next_state`;
  - Target Network evalúa esa acción;
- bootstrap anulado para transiciones terminales;
- cálculo de loss para Q-learning;
- optimizer asociado únicamente a Online Network;
- método de una actualización (`learn`, `update` o equivalente) sobre batch controlado;
- sincronización explícita de Target Network mediante método separado;
- API mínima para exponer estado necesario a save/load posterior.

La pérdida exacta y optimizer pueden definirse como configuración inicial de ingeniería, pero no deben presentarse como hiperparámetros optimizados. Deben quedar centralizados y documentados.

#### Configuración

Actualizar `3_BattleZone/configs/battlezone_config.yaml` únicamente con los parámetros necesarios para construir y probar el núcleo DDQN.

Como mínimo puede incorporar, cuando corresponda:

- `algorithm: DDQN`;
- parámetros estructurales de red;
- `gamma`;
- learning rate;
- batch size para pruebas/unit update;
- Replay Buffer configurable;
- epsilon inicial/valor usado por API;
- parámetros de optimizer/loss si son configurables;
- dispositivo `auto` o equivalente.

Reglas:

- no introducir constantes mágicas dispersas;
- no hacer tuning;
- no fijar todavía `total_timesteps`, schedules completos, checkpoint intervals o logging de entrenamiento salvo que ya existan por otra HU;
- cualquier valor inicial nuevo debe quedar identificado como **baseline de implementación por validar**, no como valor óptimo.

#### Tests focalizados

Crear/actualizar pruebas bajo `3_BattleZone/tests/` para validar el núcleo del agente sin entrenamiento largo.

### 5.2 Fuera de alcance

HU005 **no debe** implementar:

- ciclo completo `env.reset()` / `env.step()` de entrenamiento;
- recolección continua de experiencias desde BattleZone;
- warm-up/`learning_starts` integrado;
- schedule temporal completo de epsilon;
- frecuencia de aprendizaje dentro del loop;
- Target Network sync por timestep dentro de un trainer;
- trainer E2E;
- entrenamiento de múltiples episodios;
- evaluación formal;
- comparación contra baseline;
- checkpoints completos/resume;
- idempotencia de entrenamientos;
- TensorBoard;
- `run_manifest.json`;
- MLflow;
- optimización de hiperparámetros;
- PER;
- reward clipping/shaping;
- cambios al preprocessing HU003;
- reducción del action space;
- código o imports desde `2_Assault/`.

Estas responsabilidades pertenecen a HU006–HU014 según `implementacion.md`.

---

## 6. Decisiones técnicas obligatorias

### DT01 — DDQN, no DQN clásico

El target debe implementar la separación DDQN:

1. `argmax_a Q_online(next_state, a)` selecciona la acción;
2. `Q_target(next_state, argmax_online)` evalúa esa acción.

No usar directamente `max(Q_target(next_state))` como target principal, porque eso correspondería al patrón DQN clásico y violaría HU004.

### DT02 — Online y Target son redes distintas

- misma arquitectura;
- parámetros inicialmente sincronizados;
- objetos independientes;
- Target no recibe gradientes durante el cálculo del target;
- optimizer solo actualiza Online.

### DT03 — Contrato del entorno permanece intacto

La observación de HU003 continúa siendo `(4,128,128,3) uint8`.

Si PyTorch necesita `NCHW`, la transformación debe ocurrir explícitamente dentro del límite del agente/red, sin cambiar `environment.py` ni el contrato de HU003.

### DT04 — Replay uniforme y eficiente

- no PER;
- CPU RAM;
- `uint8` para estados almacenados;
- transferencia a GPU solo al construir el batch requerido para aprender;
- evitar copias CPU↔GPU innecesarias.

### DT05 — Separación de responsabilidades

- `network.py`: red;
- `replay_buffer.py`: almacenamiento/muestreo;
- `agent.py`: política y aprendizaje DDQN;
- `trainer.py`: no se implementa en HU005.

### DT06 — Configuración centralizada

Los valores que afecten el comportamiento del agente no deben quedar duplicados entre tests, notebook y módulos salvo fixtures/control values explícitos de prueba.

### DT07 — Save/load básico, no checkpointing completo

HU005 debe exponer un contrato básico que permita obtener/restaurar estado del agente o guardar/cargar pesos si resulta necesario para probar el núcleo.

HU007 será responsable del checkpoint completo, resume, optimizer/progreso/buffer cuando corresponda e idempotencia.

### DT08 — No prometer performance

Una actualización de pesos exitosa demuestra corrección mecánica básica, **no aprendizaje efectivo de BattleZone**. La evidencia de HU005 no debe afirmar que DDQN supera DQN, PER, REINFORCE o el baseline aleatorio.

---

## 7. Contratos mínimos esperados

Los nombres exactos pueden variar si mantienen claridad, pero el diseño deberá proporcionar contratos equivalentes a:

```python
class BattleZoneQNetwork(...):
    def forward(self, observations): ...

class ReplayBuffer:
    def add(self, state, action, reward, next_state, done): ...
    def sample(self, batch_size): ...
    def __len__(self): ...

class DDQNAgent:
    def select_action(self, state, epsilon): ...
    def compute_targets(self, batch): ...
    def update(self, batch): ...
    def sync_target_network(self): ...
    def state_dict(self): ...
    def load_state_dict(self, state): ...
```

No crear abstracciones genéricas para múltiples algoritmos si DDQN es el único algoritmo permitido por HU004 para esta implementación.

---

## 8. Tareas

### T01 — Validar gates previos

- confirmar HU004 `[COMPLETADA]` en `main`;
- confirmar `DDQN` como algoritmo seleccionado;
- confirmar contrato HU003 sin cambios;
- registrar evidencia de estos gates.

### T02 — Diseñar Q-Network

- definir arquitectura CNN;
- justificar profundidad/tamaño de forma pragmática;
- soportar batch y observación individual según contrato decidido;
- garantizar salida `[batch, 18]`;
- probar conversión de layout/dtype.

### T03 — Implementar Replay Buffer uniforme

- contrato de transición;
- almacenamiento CPU/uint8;
- capacidad configurable;
- muestreo uniforme;
- validaciones de tamaño, shape y dtype.

### T04 — Implementar Online/Target Network

- crear ambas redes;
- sincronización inicial;
- independencia de objetos/parámetros;
- Target fuera del optimizer.

### T05 — Implementar epsilon-greedy

- epsilon `0` produce selección greedy;
- epsilon `1` permite acciones aleatorias válidas;
- valores fuera del rango permitido deben validarse explícitamente.

### T06 — Implementar target DDQN

- selección de acción con Online;
- evaluación con Target;
- bootstrap `0` cuando `done=True`;
- cálculo sin gradiente sobre Target.

### T07 — Implementar actualización real

Sobre un batch sintético/controlado compatible con HU003:

- forward;
- gather de Q-value de acciones ejecutadas;
- target DDQN;
- loss;
- `zero_grad`;
- `backward`;
- optimizer step;
- verificar que al menos un parámetro de Online cambia;
- verificar que Target no cambia durante ese update.

### T08 — Implementar contrato básico de estado/save-load

- serialización mínima coherente del núcleo;
- restauración que preserve outputs/parámetros esperados;
- sin convertirlo en checkpoint/resume completo de HU007.

### T09 — Centralizar configuración DDQN

- agregar solo parámetros necesarios para HU005;
- documentar que son baseline de implementación;
- no realizar tuning.

### T10 — Crear tests focalizados

Cubrir CA/AV definidos en esta HU.

### T11 — Crear evidencia de implementación

Crear:

`3_BattleZone/docs/hu005_evidencia_implementacion.md`

Debe registrar resultados reales, comandos, tests, shapes, dtypes, loss controlada y verificaciones Online/Target.

### T12 — Revisar alcance del PR

- cero cambios en `2_Assault/`;
- cero MLflow;
- cero trainer E2E;
- cero cambios en contrato HU003;
- diff concentrado en núcleo DDQN, config, tests y evidencia HU005.

---

## 9. Criterios de aceptación

### CA01 — Gate HU004 satisfecho

HU004 está `[COMPLETADA]` y el algoritmo implementado es exactamente `DDQN`.

### CA02 — Q-Network compatible

Una observación/batch compatible con HU003 produce exactamente 18 Q-values por muestra y no modifica el contrato del entorno.

### CA03 — Manejo correcto de dtype/layout

La entrada `uint8` se transforma explícitamente al formato requerido por la red; no existen conversiones implícitas ambiguas ni cambio del preprocessing HU003.

### CA04 — Online/Target independientes

Ambas redes inician sincronizadas pero son instancias distintas. Una actualización de Online no modifica Target hasta ejecutar sincronización explícita.

### CA05 — Replay Buffer uniforme

Almacena y devuelve transiciones válidas, conserva estados eficientemente en CPU y no incluye lógica PER.

### CA06 — Epsilon-greedy válido

La política siempre devuelve una acción dentro de `Discrete(18)` y diferencia correctamente modo greedy y exploratorio.

### CA07 — Target DDQN correcto

La acción siguiente se selecciona con Online y se evalúa con Target. Las transiciones terminales no reciben bootstrap.

### CA08 — Actualización real de Online

Una actualización sobre batch controlado produce loss finita y modifica al menos un parámetro entrenable de Online.

### CA09 — Target protegido

Target no recibe gradientes ni cambia durante el optimizer step de Online.

### CA10 — Save/load básico consistente

El estado del núcleo puede restaurarse y conservar parámetros/output esperado dentro de tolerancias apropiadas.

### CA11 — Configuración centralizada

Los parámetros DDQN necesarios para HU005 se encuentran versionados y no se duplican como constantes mágicas en lógica reusable.

### CA12 — Tests focalizados aprobados

Todos los tests de BattleZone relevantes para HU005 pasan localmente.

### CA13 — Sin entrenamiento E2E

El PR no integra loops de múltiples pasos/episodios ni adelanta HU006.

### CA14 — Independencia de Assault

No hay modificaciones, imports ni copia de código desde `2_Assault/`.

### CA15 — Sin MLflow y sin infraestructura futura

No se introduce MLflow, TensorBoard, checkpoints completos ni manifiestos de ejecución.

### CA16 — Evidencia versionada

Existe `3_BattleZone/docs/hu005_evidencia_implementacion.md` con evidencia verificable de la implementación y autovalidaciones.

---

## 10. Autovalidaciones obligatorias

### AV01 — Dependencias
**Procedimiento:** revisar HU004/HU003 en `main`.  
**PASS:** HU004 completada, DDQN seleccionado y contrato HU003 vigente.

### AV02 — Forward pass
**Procedimiento:** crear batch controlado compatible con HU003 y ejecutar Online Network.  
**PASS:** salida exacta `[batch_size, 18]`, valores finitos.

### AV03 — Entrada individual
**Procedimiento:** pasar una observación individual por la API soportada.  
**PASS:** resultado inequívoco y documentado; si la API exige batch dimension, el error/adapter es explícito.

### AV04 — Sincronización inicial
**Procedimiento:** comparar parámetros Online/Target inmediatamente después de construir agente.  
**PASS:** mismos valores, objetos/almacenamiento independientes.

### AV05 — Epsilon greedy
**Procedimiento:** probar epsilon `0` y `1` con seed controlada cuando aplique.  
**PASS:** epsilon `0` usa greedy; epsilon `1` solo produce acciones válidas `0..17`.

### AV06 — Replay add/sample
**Procedimiento:** insertar transiciones controladas y muestrear batch.  
**PASS:** shapes/dtypes correctos; no se permite sample mayor a elementos disponibles.

### AV07 — Replay en CPU/uint8
**Procedimiento:** inspeccionar almacenamiento.  
**PASS:** estados no se mantienen permanentemente en GPU ni se expanden innecesariamente a float32 dentro del buffer.

### AV08 — Target DDQN
**Procedimiento:** usar redes/batch controlados donde pueda verificarse selección Online y evaluación Target.  
**PASS:** cálculo coincide con fórmula DDQN y no con `max` directo de Target.

### AV09 — Terminal mask
**Procedimiento:** batch con transiciones terminales/no terminales.  
**PASS:** terminal no incluye bootstrap; no terminal sí aplica `gamma`.

### AV10 — Update real
**Procedimiento:** capturar parámetros Online antes/después de un optimizer step.  
**PASS:** loss finita y al menos un parámetro cambia.

### AV11 — Target inmutable durante update
**Procedimiento:** capturar Target antes/después del update sin sync.  
**PASS:** parámetros idénticos.

### AV12 — Sync explícito
**Procedimiento:** ejecutar método de sincronización luego de modificar Online.  
**PASS:** Target vuelve a coincidir con Online.

### AV13 — Save/load básico
**Procedimiento:** guardar/restaurar estado en instancia nueva o equivalente.  
**PASS:** parámetros restaurados y output reproducible para input fijo en `eval()`.

### AV14 — Tests focalizados
**Procedimiento:** ejecutar `python -m pytest 3_BattleZone/tests -q` o comando equivalente documentado.  
**PASS:** suite BattleZone relevante aprobada; cualquier test no relacionado debe distinguirse explícitamente.

### AV15 — Scope diff
**Procedimiento:** revisar diff contra `main`.  
**PASS:** no Assault, no MLflow, no trainer E2E, no TensorBoard/checkpointing completo, no cambio del contrato HU003.

### AV16 — Anti-alucinación
**Procedimiento:** revisar evidencia HU005.  
**PASS:** solo se reportan outputs realmente ejecutados; no se atribuye performance de aprendizaje a una actualización sintética.

---

## 11. Definition of Done

HU005 puede considerarse `[COMPLETADA]` únicamente cuando:

- [ ] HU004 está cerrada y DDQN continúa siendo la decisión vigente;
- [ ] existe `3_BattleZone/src/network.py` con Q-Network propia de BattleZone;
- [ ] existe `3_BattleZone/src/replay_buffer.py` con Replay Buffer uniforme;
- [ ] existe `3_BattleZone/src/agent.py` con lógica DDQN;
- [ ] Online y Target están implementadas como redes independientes;
- [ ] la sincronización inicial Online → Target está validada;
- [ ] epsilon-greedy está implementado y validado;
- [ ] target DDQN está implementado correctamente;
- [ ] terminal masking está validado;
- [ ] existe optimizer funcional sobre Online;
- [ ] una actualización real sobre batch controlado produce loss finita;
- [ ] al menos un parámetro Online cambia después del update;
- [ ] Target permanece inmutable hasta sync explícito;
- [ ] Replay Buffer conserva estados eficientemente en CPU;
- [ ] save/load básico del núcleo está validado;
- [ ] configuración DDQN necesaria está centralizada;
- [ ] tests focalizados BattleZone pasan;
- [ ] existe `3_BattleZone/docs/hu005_evidencia_implementacion.md`;
- [ ] AV01–AV16 están en PASS o cualquier excepción aprobada está documentada;
- [ ] no se implementó entrenamiento E2E;
- [ ] no se modificó el contrato HU003;
- [ ] no se modificó ni reutilizó código de `2_Assault/`;
- [ ] no se introdujo MLflow;
- [ ] PR focalizado, revisable y listo para merge.

La HU no debe marcarse `[COMPLETADA]` en `main` antes de que su implementación sea revisada y mergeada.

---

## 12. Evidencias esperadas

La implementación debe conservar como mínimo:

1. arquitectura final de Q-Network y justificación;
2. shape/dtype de entrada y salida;
3. prueba de Online/Target sincronizadas inicialmente;
4. prueba de independencia Online/Target;
5. prueba epsilon-greedy;
6. prueba Replay Buffer add/sample;
7. prueba de almacenamiento CPU/uint8;
8. ejemplo verificable del cálculo DDQN;
9. prueba de terminal masking;
10. loss de una actualización real sobre batch controlado;
11. evidencia de cambio de parámetros Online;
12. evidencia de Target sin cambios hasta sync;
13. prueba save/load básico;
14. salida de tests focalizados;
15. diff de PR demostrando alcance;
16. tabla AV01–AV16 con estado real.

No se requiere reward promedio entrenado, TensorBoard, checkpoint, video, evaluación de 10 episodios ni evidencia de superar baseline para cerrar HU005.

---

## 13. Riesgos y mitigaciones

### R01 — Confundir DDQN con DQN clásico
**Riesgo:** calcular `max` directamente sobre Target Network.  
**Mitigación:** AV08 con batch/red controlados y verificación explícita de selección Online + evaluación Target.

### R02 — Alterar HU003 para acomodar la CNN
**Riesgo:** cambiar shape/layout del entorno.  
**Mitigación:** adaptar layout/dtype dentro del agente/red y bloquear cambios en environment/config perceptual.

### R03 — Replay Buffer consume demasiada memoria
**Riesgo:** estados RGB apilados `(4,128,128,3)` son costosos.  
**Mitigación:** almacenar `uint8` en CPU, capacidad configurable, usar buffers pequeños en HU005 y medir antes de escalar en HU006/HU009.

### R04 — Target recibe gradientes
**Riesgo:** romper la separación DDQN y actualizar ambas redes accidentalmente.  
**Mitigación:** no incluir Target en optimizer, usar `no_grad`/detach y AV09–AV12.

### R05 — Sobre-ingeniería
**Riesgo:** crear frameworks genéricos para múltiples algoritmos ya descartados.  
**Mitigación:** implementar contratos simples específicos a DDQN siguiendo SOLID de forma pragmática.

### R06 — Adelantar HU006/HU007/HU008
**Riesgo:** integrar trainer, checkpointing o TensorBoard dentro del núcleo.  
**Mitigación:** scope bloqueante y AV15.

### R07 — Valores iniciales tratados como óptimos
**Riesgo:** fijar learning rate/gamma/batch y presentarlos como decisión validada.  
**Mitigación:** etiquetar parámetros como baseline de implementación; tuning pertenece a HU012.

### R08 — Falsa evidencia de aprendizaje
**Riesgo:** interpretar una loss finita o cambio de pesos como éxito del agente en BattleZone.  
**Mitigación:** evidencia HU005 debe limitar la conclusión a corrección mecánica; aprendizaje real se valida en HUs posteriores.

### R09 — Contaminación con Assault
**Riesgo:** copiar una CNN/agente previo por similitud Atari.  
**Mitigación:** implementación independiente, revisión de diff y AV15.

### R10 — Save/load invade checkpointing
**Riesgo:** convertir HU005 en HU007.  
**Mitigación:** limitar HU005 a contrato básico de estado/pesos; resume completo queda fuera de alcance.

---

## 14. Resultado esperado para HU006

HU006 debe recibir un núcleo DDQN estable con:

- Q-Network validada para el contrato HU003;
- Online/Target Network;
- Replay Buffer uniforme;
- API epsilon-greedy;
- API para insertar/muestrear experiencia;
- función/método de actualización DDQN funcional;
- sincronización Target explícita;
- optimizer operativo;
- configuración centralizada mínima;
- tests del núcleo en verde;
- evidencia de una actualización controlada;
- interfaces suficientes para que HU006 implemente recolección continua de experiencia y ciclo de entrenamiento.

HU006 será responsable de decidir e integrar temporalmente `learning_starts`, frecuencia de actualización, schedule de exploración, target-sync dentro del loop y duración del entrenamiento de smoke/full runs según las HUs posteriores.