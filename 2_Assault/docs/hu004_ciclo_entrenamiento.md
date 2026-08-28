# HU004 — Ciclo de entrenamiento + Preflight previo al entrenamiento

## 1. Identificación

- **ID:** HU004
- **Nombre:** Ciclo de entrenamiento + Preflight previo al entrenamiento
- **Estado:** Lista para implementación
- **Dependencia previa:** HU003 — Núcleo DDQN `[COMPLETADA]`
- **Dependencias técnicas:** HU002/HU002B mantienen pendiente su validación formal en Colab, pero sus contratos locales de entorno/bootstrap se consideran disponibles para continuar el desarrollo controlado.
- **Habilita:** HU005 — Checkpoints + reanudación + idempotencia
- **Fuentes de verdad:**
  - `2_Assault/docs/implementacion.md`
  - `2_Assault/docs/arquitectura.md`
  - `2_Assault/docs/linemientos.md`
  - `2_Assault/docs/ficha_tecnica.md`
  - `2_Assault/docs/hu002_pipeline_reproducible_entorno.md`
  - `2_Assault/docs/hu002b_pipeline_ejecucion_local_github_colab.md`
  - `2_Assault/docs/hu003_nucleo_ddqn.md`
  - `2_Assault/configs/ddqn_config.yaml`
  - `2_Assault/src/environment.py`
  - `2_Assault/src/network.py`
  - `2_Assault/src/replay_buffer.py`
  - `2_Assault/src/agent.py`
  - `2_Assault/assault_ddqn.ipynb`
  - `enunciado_reto_1.txt`

---

## 2. Contexto y problema

HU003 dejó implementado y validado el núcleo DDQN de Assault:

- observaciones `(4, 84, 84)` en `uint8`;
- Q-Network con 7 Q-values;
- Online Network y Target Network independientes;
- Replay Buffer uniforme;
- política epsilon-greedy;
- target DDQN con selección Online y evaluación Target;
- actualización de Online Network;
- sincronización explícita de Target;
- save/load básico.

El siguiente riesgo ya no está en una pieza aislada sino en su **integración temporal** con `ALE/Assault-v5`.

El proyecto necesita comprobar que, durante varios timesteps consecutivos, el sistema puede ejecutar correctamente:

```text
reset
  ↓
seleccionar acción epsilon-greedy
  ↓
env.step()
  ↓
guardar transición
  ↓
avanzar global_step
  ↓
muestrear Replay Buffer cuando corresponda
  ↓
actualizar DDQN
  ↓
decay de epsilon
  ↓
sincronizar Target periódicamente
  ↓
acumular métricas mínimas
  ↓
continuar
```

Sin embargo, antes de iniciar incluso una corrida corta, el proyecto debe detectar fallos básicos de integración. Por ello HU004 incorpora un **Preflight obligatorio**.

El Preflight funciona como checklist previo al despegue: valida rápidamente que entorno, red, Replay Buffer y agente pueden trabajar juntos en el runtime actual antes de gastar tiempo de CPU/GPU en entrenamiento.

---

## 3. Historia de usuario

> **Como** equipo que desarrolla el agente DDQN de Assault, **quiero** ejecutar primero una verificación automática de integración y luego un ciclo corto de entrenamiento controlado por timesteps, **para** detectar rápidamente errores de runtime y demostrar que el agente puede aprender mediante actualizaciones reales antes de implementar persistencia, observabilidad avanzada o entrenamientos largos.

---

## 4. Objetivo verificable

Al finalizar HU004 debe ser posible ejecutar este flujo:

```text
Runtime local o Colab
        ↓
Preflight automático
        ↓
PASS / FAIL explícito
        ↓
si PASS
        ↓
crear entorno + agente + Replay Buffer
        ↓
training loop corto por timesteps
        ↓
recoger transiciones
        ↓
learning_starts
        ↓
updates DDQN periódicos
        ↓
epsilon decay
        ↓
Target sync periódico
        ↓
métricas mínimas
        ↓
resultado verificable
```

HU004 debe demostrar concretamente que:

1. el Preflight puede detectar el runtime y validar la integración HU002+HU003;
2. si el Preflight falla, el entrenamiento no comienza;
3. si el Preflight pasa, el trainer puede ejecutar un número pequeño y configurable de timesteps;
4. `global_step` aumenta exactamente con cada decisión del agente;
5. cada interacción válida se almacena en Replay Buffer;
6. el entrenamiento no comienza antes de `learning_starts`;
7. luego de `learning_starts`, se ejecutan actualizaciones según `train_frequency`;
8. epsilon disminuye de forma reproducible según la configuración;
9. Target Network se sincroniza según `target_update_frequency`;
10. al menos un update produce loss finita;
11. los pesos de Online Network cambian durante la corrida;
12. el trainer registra métricas mínimas suficientes para verificar comportamiento;
13. episodios terminados o truncados se reinician correctamente sin reiniciar `global_step`;
14. el ciclo puede ejecutarse tanto en CPU como en GPU cuando esté disponible;
15. no se implementan todavía checkpoints, TensorBoard, MLflow ni entrenamiento largo.

Resultado esperado:

```text
PREFLIGHT PASS
      ↓
short training run
      ↓
transitions > 0
      ↓
updates > 0
      ↓
finite loss
      ↓
Online weights changed
      ↓
Target synced at expected steps
      ↓
HU004 PASS
```

---

## 5. Alcance

### 5.1 Preflight obligatorio

Crear una verificación rápida previa al entrenamiento, preferiblemente en:

`2_Assault/src/preflight.py`

Responsabilidad principal:

- integrar componentes existentes;
- ejecutar checks baratos;
- devolver un resultado estructurado;
- abortar claramente cuando exista un fallo bloqueante.

El Preflight **no sustituye `pytest`**. Los tests unitarios validan componentes y contratos de forma aislada; el Preflight valida que los componentes versionados realmente pueden trabajar juntos dentro del runtime desde el que se va a entrenar.

Debe comprobar como mínimo:

```text
Environment              PASS/FAIL
Observation contract     PASS/FAIL
QNetwork                 PASS/FAIL
Replay Buffer            PASS/FAIL
DDQN target/update       PASS/FAIL
Target sync              PASS/FAIL
Save/load básico         PASS/FAIL
Device                   PASS/FAIL
```

Debe producir un resumen similar a:

```text
===== DDQN PRE-FLIGHT =====
Runtime: local / Google Colab
Device: cpu / cuda
Observation: PASS (4,84,84) uint8
QNetwork: PASS -> (1,7)
ReplayBuffer: PASS
DDQN update: PASS
Loss finite: PASS
Target stable: PASS
Target sync: PASS
Save/load: PASS

READY_FOR_TRAINING=True
```

Si un check obligatorio falla:

```text
READY_FOR_TRAINING=False
```

el trainer no debe comenzar.

### 5.2 `src/preflight.py`

Implementar una interfaz simple, por ejemplo:

```python
report = run_preflight_checks(config, device=...)
```

El resultado deberá ser inspeccionable mediante una estructura ligera, por ejemplo dataclass o diccionario, que contenga como mínimo:

- `passed`;
- runtime/device;
- checks individuales;
- observaciones o errores relevantes.

Evitar implementar un framework genérico de checks.

El Preflight puede utilizar archivos temporales para comprobar save/load y debe limpiarlos al finalizar.

### 5.3 Configuración del entrenamiento

Extender `2_Assault/configs/ddqn_config.yaml` con los parámetros mínimos necesarios para HU004.

Como mínimo:

```yaml
training:
  total_timesteps: <valor corto/configurable>
  learning_starts: <valor>
  train_frequency: <valor>
  target_update_frequency: <valor>
  epsilon_decay_steps: <valor>
```

Debe reutilizar:

```yaml
agent:
  epsilon_start: ...
  epsilon_final: ...

replay_buffer:
  capacity: ...
  batch_size: ...
```

Los nombres definitivos pueden adaptarse manteniendo semántica clara.

`total_timesteps` será configurable y durante autovalidaciones se utilizará un valor pequeño. HU004 no fija todavía el entrenamiento largo final.

### 5.4 `src/trainer.py`

Crear el ciclo de entrenamiento.

Responsabilidades:

- recibir entorno/agente/buffer/configuración o crear únicamente lo que le corresponda según arquitectura;
- mantener `global_step`;
- ejecutar selección de acción;
- ejecutar `env.step()`;
- almacenar transición;
- detectar límites de episodio;
- ejecutar updates cuando corresponde;
- calcular epsilon actual;
- sincronizar Target cuando corresponde;
- acumular métricas mínimas;
- devolver un resumen estructurado de la corrida.

El trainer no debe implementar:

- CNN;
- target DDQN;
- Replay Buffer interno duplicado;
- creación alternativa de `ALE/Assault-v5`;
- TensorBoard;
- MLflow;
- persistencia de checkpoints.

### 5.5 Control por timesteps

El entrenamiento debe estar gobernado por `global_step`, no por número de episodios.

Cada llamada válida a `env.step(action)` representa una decisión del agente y debe incrementar `global_step` una vez.

El trainer debe poder continuar atravesando múltiples episodios hasta alcanzar el número objetivo de timesteps.

### 5.6 `learning_starts`

Antes de `learning_starts`:

- el agente interactúa;
- el Replay Buffer se llena;
- no se ejecuta `agent.update(...)`.

Después de `learning_starts`, solo se actualiza cuando:

- existe al menos `batch_size` en el Replay Buffer;
- el timestep cumple `train_frequency`.

Esto evita aprender con un buffer vacío o insuficiente.

### 5.7 `train_frequency`

Definir de forma explícita cuántas decisiones del agente ocurren entre actualizaciones.

Ejemplo conceptual:

```text
global_step % train_frequency == 0
→ update
```

No introducir entrenamiento por episodios como sustituto.

### 5.8 Sincronización Target

Target Network debe sincronizarse únicamente cuando corresponda según:

`target_update_frequency`

Ejemplo conceptual:

```text
global_step % target_update_frequency == 0
→ agent.sync_target_network()
```

El trainer debe registrar cuándo ocurre la sincronización para poder autovalidarla.

### 5.9 Epsilon decay

HU004 debe implementar el scheduling temporal de epsilon.

Debe usar:

- `epsilon_start`;
- `epsilon_final`;
- `epsilon_decay_steps`.

Se recomienda decay lineal simple y determinista:

```text
step = 0                 → epsilon_start
step >= decay_steps      → epsilon_final
entre ambos              → interpolación lineal
```

Epsilon nunca debe quedar fuera del rango configurado.

La política de selección sigue perteneciendo a `DDQNAgent`; el trainer únicamente calcula/entrega el epsilon correspondiente al timestep.

### 5.10 Replay Buffer

HU004 reutiliza `ReplayBuffer` de HU003.

Cada transición debe incluir como mínimo:

```text
state
action
reward
next_state
done_for_bootstrap
```

#### Semántica `terminated` / `truncated`

Gymnasium devuelve:

```text
terminated
truncated
```

Se deben distinguir dos conceptos:

```text
episode_ended = terminated OR truncated
```

para decidir cuándo ejecutar `reset()`.

Para el bootstrap DDQN se debe utilizar como terminal real:

```text
done_for_bootstrap = terminated
```

salvo que exista evidencia específica del entorno que justifique tratar una truncación como terminal real.

Esto evita eliminar bootstrap únicamente porque un episodio terminó por límite externo de tiempo.

El Replay Buffer puede continuar almacenando el campo `done` existente, asignándole `done_for_bootstrap` sin necesidad de ampliar su contrato en HU004.

### 5.11 Recompensa

Guardar la recompensa real entregada por el entorno.

HU004 no debe introducir reward clipping salvo decisión explícita posterior respaldada por una HU/configuración y sin afectar la evaluación final raw.

### 5.12 Métricas mínimas

Sin TensorBoard todavía, el trainer debe acumular/devolver métricas mínimas en memoria o logging sencillo:

- `global_step`;
- episodios completados;
- recompensa de cada episodio terminado;
- longitud de episodio;
- epsilon actual/final;
- número de transiciones almacenadas;
- número de updates;
- última loss;
- loss media de la corrida si aplica;
- número/timesteps de Target sync;
- duración básica opcional si ya existe helper reutilizable.

No crear todavía dashboards, TensorBoard writers ni MLflow runs.

### 5.13 Notebook

Actualizar `2_Assault/assault_ddqn.ipynb` únicamente como **orquestador**.

Debe incorporar, en orden:

```text
bootstrap HU002B
↓
config + hardware
↓
HU002 validation existente
↓
PRE-FLIGHT HU004
↓
si PASS
↓
short training run HU004
↓
resumen de métricas
```

El notebook no debe duplicar implementación de `trainer.py` ni `preflight.py`.

Debe permitir ejecutar el mismo flujo:

- localmente en PC;
- en Google Colab;

usando la infraestructura de bootstrap ya existente.

---

## 6. Fuera de alcance

HU004 **no** debe implementar:

- entrenamiento largo/final;
- checkpoints persistentes;
- resume de entrenamiento;
- persistencia completa del Replay Buffer;
- Google Drive para checkpoints;
- TensorBoard;
- MLflow;
- callbacks avanzados;
- selección del mejor modelo;
- evaluación formal sobre ≥10 episodios;
- video;
- optimización de hiperparámetros;
- PER;
- Dueling DQN;
- Rainbow;
- Noisy Nets;
- n-step returns;
- reward clipping no aprobado;
- nuevas arquitecturas neuronales;
- GitHub Actions para entrenamiento;
- automatización remota Codex → Colab.

La persistencia/reanudación pertenece a HU005.
TensorBoard pertenece a HU006.
El smoke E2E completo con esas capacidades pertenece a HU007.
MLflow pertenece a HU008.
El primer entrenamiento largo pertenece a HU009.

---

## 7. Decisiones y restricciones técnicas

### 7.1 Separación de responsabilidades

```text
environment.py
→ entorno/preprocessing

network.py
→ CNN

replay_buffer.py
→ memoria de experiencias

agent.py
→ lógica DDQN/update

preflight.py
→ gate rápido de integración

trainer.py
→ secuencia temporal de entrenamiento

assault_ddqn.ipynb
→ orquestación/evidencia
```

No duplicar responsabilidades.

### 7.2 Preflight fail-fast

El Preflight debe fallar antes de entrenamiento cuando exista un problema material.

No debe ocultar excepciones críticas convirtiéndolas silenciosamente en warnings.

El notebook puede mostrar un reporte amigable, pero el código debe impedir continuar si `passed=False`.

### 7.3 Preflight barato

Debe durar lo mínimo razonable y no debe llenar el Replay Buffer final ni ejecutar entrenamiento prolongado.

Usar:

- pocas observaciones/transiciones;
- batches pequeños cuando sea apropiado;
- un solo update o cantidad mínima necesaria;
- archivos temporales para save/load.

### 7.4 Tests vs Preflight

`pytest` y Preflight tienen objetivos distintos:

```text
pytest
→ contratos/componentes/regresión

Preflight
→ integración real en el runtime antes de entrenar
```

No invocar toda la suite `pytest` desde `preflight.py`.

El notebook puede opcionalmente mostrar una celda independiente para tests, pero el gate de runtime debe depender de los checks de integración definidos en `preflight.py`.

### 7.5 Uso de GPU

El trainer debe usar el dispositivo configurado en el agente.

No mover tensores manualmente en el trainer si `DDQNAgent.update` ya encapsula esa responsabilidad.

No exigir GPU para cerrar HU004 si la corrida corta funciona localmente en CPU. GPU se validará cuando el notebook se ejecute en Colab.

### 7.6 Reproducibilidad

Reutilizar seed del proyecto para:

- entorno;
- Replay Buffer;
- agente/PyTorch;
- acciones aleatorias cuando corresponda.

No prometer identidad bit-a-bit entre CPU/GPU/ALE cuando la plataforma no la garantice.

### 7.7 Total timesteps de smoke

Los tests y autovalidaciones deben usar un total pequeño que:

- supere `learning_starts`;
- permita varios updates;
- permita al menos una sincronización Target;
- finalice rápidamente.

No reducir artificialmente la configuración final si eso mezcla parámetros de prueba con parámetros objetivo. Los tests pueden sobrescribir configuración en memoria para sus escenarios pequeños.

### 7.8 Sin persistencia anticipada

HU004 puede devolver `global_step` y métricas, pero no debe diseñar todavía el esquema completo de checkpoint/resume. Esa responsabilidad es de HU005.

### 7.9 Docstrings

Funciones y clases reutilizables deben seguir docstrings estilo Google definidos en `linemientos.md`.

---

## 8. Plan de implementación / tareas

### T01 — Extender configuración de entrenamiento

**Archivo:** `2_Assault/configs/ddqn_config.yaml`

Agregar los parámetros mínimos de HU004.

**Resultado:** trainer y epsilon scheduling no dependen de constantes mágicas.

### T02 — Implementar scheduling de epsilon

**Ubicación:** `src/trainer.py` o helper pequeño si existe una razón clara de reutilización.

Implementar función determinista, por ejemplo:

```python
compute_epsilon(global_step, start, final, decay_steps)
```

**Resultado:** valores correctos en inicio, mitad y final del decay.

### T03 — Implementar Preflight

**Archivo recomendado:** `2_Assault/src/preflight.py`

Integrar:

- entorno HU002;
- QNetwork/DDQNAgent HU003;
- Replay Buffer;
- un update corto;
- Target estable/sync;
- save/load temporal;
- device.

**Resultado:** reporte `PASS/FAIL` explícito y `READY_FOR_TRAINING`.

### T04 — Implementar `Trainer`

**Archivo:** `2_Assault/src/trainer.py`

Crear el loop gobernado por timesteps.

**Resultado:** interacción continua con episodios múltiples hasta alcanzar `total_timesteps`.

### T05 — Integrar Replay Buffer

Guardar una transición por decisión del agente.

**Resultado:** buffer aumenta correctamente y respeta su contrato HU003.

### T06 — Implementar `learning_starts` + `train_frequency`

**Resultado:** no hay updates anticipados y luego ocurren en timesteps esperados.

### T07 — Integrar `agent.update`

**Resultado:** se generan losses finitas y Online Network cambia.

### T08 — Integrar Target sync periódico

**Resultado:** sincronizaciones exactamente en los timesteps configurados.

### T09 — Manejar episodios

Mantener:

```text
episode_reward
episode_length
```

Resetear cuando `terminated or truncated`, sin reiniciar `global_step`.

Usar `terminated` como máscara terminal de bootstrap según sección 5.10.

### T10 — Acumular métricas mínimas

Devolver un resumen estructurado de la corrida corta.

### T11 — Integrar notebook

Agregar:

1. Preflight;
2. gate fail-fast;
3. short training run;
4. resumen final HU004.

No duplicar lógica.

### T12 — Tests focalizados

Agregar tests para:

- epsilon schedule;
- Preflight;
- trainer;
- learning_starts;
- train frequency;
- Target sync;
- episode reset;
- terminal/truncated semantics;
- short training run.

### T13 — Ejecutar smoke local

Ejecutar Preflight + entrenamiento corto con Assault real.

### T14 — Validación notebook

Ejecutar las secciones HU004 localmente y, cuando el usuario lo haga en Colab, registrar evidencia real de runtime/GPU sin inventar resultados.

### T15 — Actualizar documentación

Registrar resultados reales en `2_Assault/docs/implementacion.md` únicamente después de ejecutar las autovalidaciones.

---

## 9. Criterios de aceptación

### CA01 — Preflight integrado

**Dado** un runtime válido, **cuando** se ejecuta `run_preflight_checks`, **entonces** devuelve un reporte explícito de checks HU002/HU003 y marca `passed=True` únicamente si todos los checks obligatorios pasan.

### CA02 — Preflight bloqueante

**Dado** un Preflight fallido, **cuando** se intenta iniciar el training desde el notebook/orquestador, **entonces** el entrenamiento no comienza.

### CA03 — Entorno real en Preflight

**Dado** `create_assault_env`, **cuando** el Preflight obtiene una observación, **entonces** valida `(4,84,84)`, `uint8` y acción válida.

### CA04 — Núcleo DDQN en Preflight

**Dado** el agente HU003, **cuando** el Preflight ejecuta su integración mínima, **entonces** QNetwork, Replay Buffer, update, Target sync y save/load básico pasan sin errores.

### CA05 — Loop por timesteps

**Dado** `total_timesteps=N`, **cuando** finaliza `Trainer.train`, **entonces** `global_step == N` independientemente del número de episodios ocurridos.

### CA06 — Una transición por step

**Dado** un `env.step()` exitoso, **cuando** avanza el loop, **entonces** se agrega exactamente una transición al Replay Buffer.

### CA07 — Learning starts

**Dado** `learning_starts=L`, **cuando** `global_step < L`, **entonces** no ocurre ningún update del agente.

### CA08 — Batch suficiente

**Dado** un Replay Buffer con menos elementos que `batch_size`, **cuando** llega un timestep entrenable, **entonces** no se intenta muestrear ni actualizar inválidamente.

### CA09 — Train frequency

**Dado** `train_frequency=F`, **cuando** se supera `learning_starts`, **entonces** los updates ocurren únicamente en los timesteps que cumplen la frecuencia configurada.

### CA10 — Epsilon inicial

**Dado** `global_step=0`, **cuando** se calcula epsilon, **entonces** es igual a `epsilon_start`.

### CA11 — Epsilon final

**Dado** `global_step >= epsilon_decay_steps`, **cuando** se calcula epsilon, **entonces** es igual a `epsilon_final`.

### CA12 — Epsilon acotado

**Dado** cualquier timestep válido, **cuando** se calcula epsilon, **entonces** permanece entre `epsilon_final` y `epsilon_start`.

### CA13 — Update real

**Dado** un buffer suficiente después de `learning_starts`, **cuando** ocurre un timestep de entrenamiento, **entonces** `agent.update` produce una loss finita y al menos un parámetro Online cambia durante la corrida.

### CA14 — Target sync

**Dado** `target_update_frequency=T`, **cuando** `global_step` alcanza un múltiplo válido de `T`, **entonces** se ejecuta sincronización Target y el evento queda registrado.

### CA15 — Episodio terminado

**Dado** `terminated=True`, **cuando** finaliza el step, **entonces** la transición se almacena como terminal para bootstrap y el entorno se resetea para continuar.

### CA16 — Episodio truncado

**Dado** `truncated=True` y `terminated=False`, **cuando** finaliza el step, **entonces** el episodio se resetea pero la transición no se fuerza como terminal MDP para el target DDQN.

### CA17 — Global step continuo

**Dado** que termina un episodio, **cuando** se ejecuta `reset()`, **entonces** `global_step` conserva su valor y continúa hasta `total_timesteps`.

### CA18 — Métricas mínimas

**Dada** una corrida corta, **cuando** finaliza, **entonces** existe evidencia de `global_step`, episodios, rewards, lengths, updates, loss, epsilon y Target syncs.

### CA19 — Compatibilidad CPU

**Dado** un entorno sin CUDA, **cuando** se ejecuta Preflight + short training, **entonces** el flujo funciona en CPU sin device mismatch.

### CA20 — Compatibilidad GPU opcional

**Dado** CUDA disponible, **cuando** se ejecuta el flujo, **entonces** el agente utiliza GPU sin mezclar dispositivos. La ausencia de GPU local no bloquea HU004.

### CA21 — Notebook orquestador

**Dado** `assault_ddqn.ipynb`, **cuando** se revisa, **entonces** llama a módulos `src` y no duplica implementación de Preflight, DDQN o trainer.

### CA22 — Sin scope creep

**Dado** el PR HU004, **cuando** se revisa, **entonces** no contiene checkpoints/resume completos, TensorBoard, MLflow, entrenamiento largo ni extensiones no aprobadas de DDQN.

---

## 10. Autovalidaciones obligatorias

### AV01 — Imports

**Procedimiento:** importar `Trainer`, función de epsilon y Preflight desde la estructura real del proyecto.

**Esperado:** imports limpios.

### AV02 — Suite completa

**Procedimiento:**

```bash
python -m pytest 2_Assault/tests -q
```

**Esperado:** todos los tests previos + HU004 pasan; solo se permiten skips justificados por hardware opcional.

### AV03 — Epsilon schedule

**Procedimiento:** probar inicio, punto intermedio, final y timestep posterior al decay.

**Esperado:** interpolación/valores exactos según configuración y límites correctos.

### AV04 — Preflight PASS

**Procedimiento:** ejecutar Preflight con componentes reales en runtime local.

**Esperado:** todos los checks obligatorios pasan y `READY_FOR_TRAINING=True`.

### AV05 — Preflight FAIL controlado

**Procedimiento:** inyectar/controlar un componente inválido en test sin modificar código productivo.

**Esperado:** `passed=False` o excepción bloqueante documentada; el training no se inicia.

### AV06 — No update antes de learning_starts

**Procedimiento:** corrida controlada menor que `learning_starts` o mocks focalizados.

**Esperado:** `update_count == 0`.

### AV07 — Update después de learning_starts

**Procedimiento:** corrida pequeña que supere `learning_starts` y tenga buffer suficiente.

**Esperado:** `update_count > 0`, loss finita.

### AV08 — Train frequency

**Procedimiento:** usar frecuencia conocida en test y registrar timesteps de updates.

**Esperado:** todos coinciden con la regla configurada.

### AV09 — Target frequency

**Procedimiento:** usar frecuencia pequeña de prueba.

**Esperado:** syncs ocurren únicamente en timesteps esperados.

### AV10 — Episode boundaries

**Procedimiento:** entorno/mocks controlados con `terminated` y `truncated`.

**Esperado:** ambos reinician episodio; solo `terminated` marca terminal de bootstrap.

### AV11 — Global step

**Procedimiento:** entrenar `N` timesteps atravesando al menos un reset.

**Esperado:** `global_step == N`.

### AV12 — Pesos Online

**Procedimiento:** comparar pesos antes/después de corrida con updates.

**Esperado:** al menos un parámetro cambia.

### AV13 — Métricas

**Procedimiento:** inspeccionar resultado del trainer.

**Esperado:** contiene métricas mínimas definidas en 5.12 con valores coherentes.

### AV14 — Smoke real Assault

**Procedimiento:** ejecutar Preflight y short training sobre `ALE/Assault-v5` usando la fábrica HU002.

**Esperado:** completa sin errores de shape, dtype, acción, buffer, optimizer o device.

### AV15 — Notebook local

**Procedimiento:** ejecutar celdas HU002 + Preflight + HU004 short run en orden localmente.

**Esperado:** resumen final de Preflight PASS y training corto PASS sin modificaciones manuales de código.

### AV16 — Notebook Colab futura

**Procedimiento:** cuando el usuario ejecute el notebook en runtime Colab, correr las mismas celdas.

**Esperado:** bootstrap usa `/content`, dependencias/imports correctos, Preflight pasa y short training funciona en el dispositivo seleccionado.

**Regla:** si esta validación no se ejecuta durante la implementación de HU004, debe quedar documentada como pendiente; no inventar resultados de Colab.

---

## 11. Evidencias requeridas

El PR HU004 debe incluir o referenciar:

- salida de `pytest`;
- configuración de entrenamiento agregada;
- resultado estructurado del Preflight;
- runtime/device utilizado;
- `READY_FOR_TRAINING`;
- `total_timesteps` del smoke;
- `global_step` final;
- tamaño/transiciones del Replay Buffer;
- timestep del primer update;
- cantidad de updates;
- loss final/media;
- evidencia de cambio de pesos Online;
- timesteps de Target sync;
- epsilon inicial/final observado;
- episodios/recompensas/longitudes de la corrida corta;
- evidencia de manejo `terminated`/`truncated`;
- resultado de ejecución local del notebook;
- resultado Colab únicamente si fue ejecutado realmente;
- commit Git;
- confirmación explícita de scope excluido.

No se requiere demostrar todavía mejora significativa de recompensa sobre baseline.

---

## 12. Definition of Done

HU004 se considera terminada únicamente cuando:

- [ ] configuración HU004 centralizada en YAML;
- [ ] epsilon scheduling implementado y probado;
- [ ] `src/preflight.py` o equivalente simple implementado;
- [ ] Preflight valida entorno + núcleo DDQN;
- [ ] Preflight produce PASS/FAIL explícito;
- [ ] entrenamiento queda bloqueado si Preflight falla;
- [ ] `src/trainer.py` implementado;
- [ ] loop controlado por timesteps;
- [ ] transición almacenada por decisión;
- [ ] `learning_starts` respetado;
- [ ] `batch_size` mínimo respetado;
- [ ] `train_frequency` respetada;
- [ ] update DDQN real ejecutado;
- [ ] loss finita;
- [ ] Online Network cambia durante corrida con updates;
- [ ] epsilon decay funciona;
- [ ] Target sync periódico funciona;
- [ ] `terminated`/`truncated` manejados correctamente;
- [ ] `global_step` no se reinicia entre episodios;
- [ ] métricas mínimas disponibles;
- [ ] smoke real con Assault pasa;
- [ ] notebook integra Preflight + short training sin duplicar lógica;
- [ ] ejecución local del notebook validada;
- [ ] tests previos continúan pasando;
- [ ] nuevos tests HU004 pasan;
- [ ] no existen errores bloqueantes conocidos;
- [ ] documentación/evidencia actualizada;
- [ ] PR limitado a HU004;
- [ ] no se implementaron checkpoints/resume completos, TensorBoard, MLflow ni entrenamiento largo.

La validación Colab puede registrarse como pendiente si no fue ejecutada por el usuario durante HU004, siempre que no se afirme falsamente que pasó.

---

## 13. Riesgos y consideraciones

### 13.1 Replay Buffer y RAM

La configuración actual de Replay Buffer visual puede reservar varios GB si se inicializa con capacidad grande y arrays preasignados.

Durante tests/smoke se deben usar capacidades reducidas en configuración de prueba o sobrescrituras en memoria. No redimensionar silenciosamente la configuración objetivo solo para hacer pasar tests.

Antes de HU009 deberá comprobarse que la capacidad final es razonable para RAM disponible en Colab.

### 13.2 Preflight demasiado costoso

Un Preflight que tarda demasiado pierde su propósito. Debe ejecutar únicamente lo necesario para detectar incompatibilidades de integración.

### 13.3 Error en frecuencia

Off-by-one en `global_step`, `learning_starts`, train frequency o Target sync puede cambiar significativamente el algoritmo. Los tests deben registrar timesteps exactos.

### 13.4 Truncation vs termination

Tratar toda truncación como terminal puede introducir targets incorrectos. HU004 debe separar el final operativo del episodio de la máscara terminal usada por DDQN.

### 13.5 Exploración

Un epsilon decay demasiado rápido puede impedir explorar; demasiado lento puede retrasar aprendizaje. HU004 solo implementa el mecanismo, no optimiza todavía estos valores.

### 13.6 CPU vs GPU

Una corrida corta local en CPU demuestra funcionalidad, no rendimiento. El entrenamiento final sigue orientado a GPU Colab.

### 13.7 Notebook local vs filesystem Colab

El bootstrap HU002B debe seguir siendo la única fuente para resolver repo/ref/SHA/imports remotos. HU004 no debe introducir rutas paralelas o hardcodeadas adicionales.

### 13.8 Métricas sin TensorBoard

HU004 solo acumula métricas mínimas. No anticipar HU006 creando infraestructura de observabilidad avanzada.

### 13.9 HU002/HU002B pendientes

La decisión de continuar no convierte sus validaciones Colab pendientes en aprobadas. Mantener la documentación explícita hasta que el usuario ejecute esas validaciones.

---

## 14. Resultado esperado y gate para HU005

HU005 solo debe comenzar cuando HU004 demuestre:

```text
HU002 environment
      +
HU003 DDQN core
      ↓
PRE-FLIGHT PASS
      ↓
training loop por timesteps
      ↓
Replay Buffer poblado
      ↓
learning_starts respetado
      ↓
updates DDQN reales
      ↓
epsilon decay
      ↓
Target sync periódico
      ↓
episode reset correcto
      ↓
finite losses + métricas
      ↓
HU004 PASS
```

El objetivo de HU004 no es obtener todavía una recompensa alta. Su objetivo es demostrar que el sistema **puede entrenar correctamente** durante una corrida corta y observable antes de añadir persistencia en HU005.