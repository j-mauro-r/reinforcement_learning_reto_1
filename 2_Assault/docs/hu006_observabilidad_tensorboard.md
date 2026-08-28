# HU006 — Observabilidad con TensorBoard

## 1. Identificación

- **ID:** HU006
- **Nombre:** Observabilidad con TensorBoard
- **Estado:** Lista para implementación
- **Dependencia previa:** HU005 — Checkpoints, reanudación e idempotencia `[COMPLETADA]`
- **Dependencias técnicas:** HU002/HU002B mantienen pendiente la validación formal en Colab, pero sus contratos locales continúan disponibles para el desarrollo controlado.
- **Habilita:** HU007 — Smoke test end-to-end
- **Fuentes de verdad:**
  - `2_Assault/docs/implementacion.md`
  - `2_Assault/docs/arquitectura.md`
  - `2_Assault/docs/linemientos.md`
  - `2_Assault/docs/ficha_tecnica.md`
  - `2_Assault/docs/hu003_nucleo_ddqn.md`
  - `2_Assault/docs/hu004_ciclo_entrenamiento.md`
  - `2_Assault/docs/hu005_checkpoints_reanudacion_idempotencia.md`
  - `2_Assault/configs/ddqn_config.yaml`
  - `2_Assault/src/agent.py`
  - `2_Assault/src/trainer.py`
  - `2_Assault/src/checkpointing.py`
  - `2_Assault/src/preflight.py`
  - `2_Assault/assault_ddqn.ipynb`
  - `enunciado_reto_1.txt`

---

## 2. Contexto y problema

HU004 demostró que el agente DDQN puede entrenar durante una corrida corta y HU005 añadió continuidad entre sesiones mediante checkpoints y reanudación. El sistema ya puede modificar pesos, continuar desde un `global_step` restaurado y preservar estado relevante del entrenamiento.

El siguiente riesgo es operar el entrenamiento como una **caja negra**. Saber que el proceso no lanza excepciones no permite responder preguntas esenciales:

```text
¿La recompensa está mejorando?
¿La loss es estable o diverge?
¿Epsilon está decayendo como se esperaba?
¿Los Q-values están creciendo de forma razonable?
¿El agente continúa actualizando después de resume?
¿La longitud de episodios cambia con el aprendizaje?
¿En qué timestep ocurrió un comportamiento anómalo?
```

HU006 debe hacer observable el entrenamiento con **TensorBoard**, sin convertir TensorBoard en la fuente de verdad del experimento ni adelantar MLflow.

La responsabilidad conceptual queda separada así:

```text
Trainer
  │
  ├── produce eventos/métricas
  │
  ▼
TensorBoard callback/logger
  │
  ├── scalars por timestep
  ├── scalars por episodio
  └── event files
        │
        ▼
TensorBoard UI
```

TensorBoard servirá para inspección detallada de curvas durante entrenamiento. La comparación formal de experimentos, hardware, commits y artefactos seguirá perteneciendo a MLflow en HU008.

---

## 3. Historia de usuario

> **Como** equipo que entrena el agente DDQN de Assault, **quiero** registrar automáticamente las métricas esenciales del aprendizaje en TensorBoard, **para** detectar rápidamente ausencia de aprendizaje, inestabilidad, divergencia o problemas de exploración antes de ejecutar entrenamientos costosos.

---

## 4. Objetivo verificable

Al finalizar HU006 debe ser posible ejecutar:

```text
Preflight PASS
      ↓
new / resume
      ↓
training corto
      ↓
TensorBoard event files
      ↓
EventAccumulator / TensorBoard
      ↓
métricas esperadas disponibles
```

HU006 debe demostrar que:

1. TensorBoard se inicializa desde configuración centralizada;
2. los logs se agrupan por `run_id`;
3. el trainer puede funcionar con observabilidad habilitada o deshabilitada;
4. una corrida corta genera event files válidos;
5. se registra `train/epsilon` por timestep según una frecuencia configurable;
6. se registra `train/loss` cuando existe un update DDQN;
7. se registra `train/q_mean` o métrica Q equivalente útil cuando existe un update;
8. se registra `train/learning_rate` cuando corresponde;
9. se registra `episode/reward` al finalizar un episodio;
10. se registra `episode/reward_mean` mediante una ventana móvil configurable;
11. se registra `episode/length` al finalizar un episodio;
12. todos los scalars usan `global_step` como eje temporal común cuando sea técnicamente coherente;
13. no se generan métricas falsas en timesteps donde no existe dato real;
14. un resume continúa escribiendo con pasos globales correctos sin reiniciar silenciosamente a cero;
15. los logs de una corrida no se mezclan accidentalmente con otro `run_id`;
16. el notebook permite visualizar TensorBoard sin duplicar lógica del trainer;
17. los tests pueden leer los event files y verificar tags/steps/valores;
18. TensorBoard puede deshabilitarse sin cambiar la lógica del algoritmo;
19. no se implementa todavía MLflow;
20. no se ejecuta todavía el smoke E2E completo de HU007 ni entrenamiento largo.

Resultado esperado:

```text
short training
   ↓
events.out.tfevents...
   ↓
train/loss
train/epsilon
train/q_mean
episode/reward
episode/reward_mean
episode/length
train/learning_rate
   ↓
TENSORBOARD PASS
```

---

## 5. Alcance

### 5.1 Dependencia TensorBoard

Agregar la dependencia necesaria en:

`2_Assault/requirements.txt`

Preferir el soporte estándar de PyTorch mediante:

```python
from torch.utils.tensorboard import SummaryWriter
```

Agregar `tensorboard` explícitamente si todavía no está en dependencias.

No introducir frameworks alternativos de tracking.

---

## 5.2 Configuración centralizada

Extender `2_Assault/configs/ddqn_config.yaml` con una sección equivalente a:

```yaml
tensorboard:
  enabled: true
  directory: logs/tensorboard
  log_frequency_steps: 4
  reward_window_episodes: 10
  flush_frequency_steps: 24
```

Los nombres pueden adaptarse si mejoran claridad.

La configuración debe permitir como mínimo:

- habilitar/deshabilitar TensorBoard;
- definir directorio raíz;
- frecuencia de logging por timestep;
- ventana para recompensa media móvil;
- política simple de flush cuando aporte valor.

No hardcodear rutas ni frecuencias en varios archivos.

---

## 5.3 `src/callbacks.py`

Crear o completar:

`2_Assault/src/callbacks.py`

La arquitectura del proyecto establece que este módulo centraliza observabilidad y persistencia periódica. En HU006 su alcance se limita a **TensorBoard y medición simple asociada a observabilidad**.

No mover ni reimplementar el checkpointing funcional de HU005 salvo que exista una refactorización mínima, segura y claramente justificada.

Crear una interfaz simple, por ejemplo:

```python
logger = TensorBoardLogger(...)
logger.log_step(...)
logger.log_update(...)
logger.log_episode(...)
logger.close()
```

Los nombres pueden variar, pero las responsabilidades deben permanecer claras.

Debe encapsular `SummaryWriter` para evitar que `trainer.py` conozca detalles de TensorBoard.

### Restricciones

`callbacks.py` no debe:

- calcular DDQN targets;
- seleccionar acciones;
- modificar optimizer;
- crear entornos;
- implementar Replay Buffer;
- decidir cuándo entrenar;
- contener MLflow;
- cambiar hiperparámetros.

---

## 5.4 Directorio por `run_id`

Los event files deben separarse por ejecución.

Ejemplo:

```text
logs/
└── tensorboard/
    └── assault_ddqn_exp_001/
        └── events.out.tfevents...
```

Reglas:

- el `run_id` debe ser el mismo definido para checkpointing;
- un `resume_full` o `resume_light` debe conservar el mismo `run_id`;
- una corrida `new` diferente debe utilizar otro `run_id`;
- no mezclar silenciosamente runs diferentes en la misma carpeta;
- crear directorios con `exist_ok=True`;
- los logs no deben versionarse rutinariamente en GitHub.

---

## 5.5 Métricas obligatorias

### `train/epsilon`

Registrar el epsilon utilizado por el agente.

Debe corresponder al timestep real de selección de acción.

No registrar un epsilon calculado para un timestep diferente al utilizado.

### `train/loss`

Registrar únicamente cuando `agent.update(...)` haya ocurrido realmente.

No rellenar timesteps sin update con cero.

Esto debe permitir detectar:

- divergencia;
- NaN/Inf;
- oscilaciones fuertes;
- ausencia de updates.

### `train/q_mean`

Registrar Q-value medio o una métrica equivalente útil asociada a los estados/batch utilizados en entrenamiento.

Debe provenir de una medición real y no de un valor sintético.

Preferencia: exponer desde `DDQNAgent.update(...)` una métrica adicional calculada sin duplicar forward passes costosos innecesariamente.

Ejemplo:

```python
{
    "loss": ...,
    "q_mean": ...,
}
```

Si se utiliza otra métrica Q, debe documentarse por qué.

### `train/learning_rate`

Registrar el learning rate efectivo del optimizer.

Aunque HU006 no implemente scheduler, el valor actual debe quedar observable y la arquitectura debe admitir que cambie posteriormente.

### `episode/reward`

Registrar recompensa raw acumulada de cada episodio finalizado.

No aplicar clipping ni transformar la recompensa para TensorBoard.

### `episode/reward_mean`

Registrar media móvil sobre una ventana configurable de episodios completados.

Ejemplo:

```text
reward_window_episodes = 10
```

Para menos de 10 episodios, calcular sobre los episodios disponibles.

### `episode/length`

Registrar longitud del episodio finalizado en decisiones del agente.

---

## 5.6 Global step como eje principal

Para scalars de entrenamiento se debe usar:

```text
global_step
```

como eje X.

Ejemplo:

```text
train/epsilon       @ global_step 40
train/loss          @ global_step 40
train/q_mean        @ global_step 40
train/learning_rate @ global_step 40
```

Para métricas de episodios se recomienda también utilizar el `global_step` en el que terminó el episodio, permitiendo correlacionar reward/length con loss y epsilon.

No reiniciar pasos al hacer resume.

---

## 5.7 Integración con `DDQNAgent`

HU006 puede extender mínimamente `DDQNAgent.update(...)` para devolver métricas adicionales útiles.

Actualmente HU003/HU004 requieren al menos:

```python
{"loss": value}
```

HU006 puede evolucionar a:

```python
{
    "loss": value,
    "q_mean": value,
    "learning_rate": value,
}
```

Mantener compatibilidad con tests y consumidores existentes.

No introducir lógica TensorBoard dentro de `agent.py`.

El agente produce métricas; el callback/logger decide cómo persistirlas.

---

## 5.8 Integración con `Trainer`

Extender `Trainer` para aceptar un observador/logger opcional.

Ejemplo conceptual:

```python
Trainer(
    ...,
    metrics_logger=logger,
)
```

o interfaz equivalente.

El Trainer debe emitir eventos en momentos correctos:

```text
cada timestep configurado
→ epsilon

cada update DDQN
→ loss
→ q_mean
→ learning_rate

cada episodio terminado
→ reward
→ reward_mean
→ length
```

La lógica de entrenamiento debe funcionar igual cuando:

```text
metrics_logger=None
```

TensorBoard no puede convertirse en dependencia funcional del aprendizaje.

### Fail-safe

Si TensorBoard está deshabilitado, el entrenamiento debe continuar sin crear event files.

Los errores materiales al inicializar una ruta inválida deben ser explícitos; no ocultar fallos silenciosamente si el usuario solicitó logging habilitado.

---

## 5.9 Resume + TensorBoard

HU005 permite reanudar desde un `global_step` restaurado.

HU006 debe demostrar:

```text
session A logs steps 1..N
checkpoint @ N

session B resume same run_id @ N
logs steps N+1..T
```

Reglas:

- no reiniciar `global_step` de TensorBoard;
- conservar `run_id`;
- permitir escribir nuevos event files dentro de la misma carpeta del run;
- no requerir un único archivo físico de events;
- TensorBoard debe poder leer todos los event files de esa carpeta como una misma serie temporal;
- no borrar event files previos durante resume.

HU006 no necesita guardar el estado de `SummaryWriter` dentro del checkpoint.

---

## 5.10 Idempotencia de logs

La ejecución repetida debe ser segura.

Reglas:

- `new` con nuevo `run_id` → carpeta nueva;
- `resume` → reutiliza carpeta del mismo `run_id` sin borrar logs previos;
- no ejecutar `rmtree()` sobre logs existentes;
- no truncar event files previos;
- no seleccionar automáticamente otro run;
- el notebook debe mostrar la ruta exacta de TensorBoard utilizada.

Si el usuario ejecuta accidentalmente `new` con un `run_id` ya existente, la protección de HU005 sobre la corrida debe seguir aplicando.

---

## 5.11 Persistencia local y Colab

### Local

Por defecto:

```text
2_Assault/logs/tensorboard/<run_id>/
```

### Colab

Los event files necesarios solo para observación de una sesión pueden vivir temporalmente bajo la copia del proyecto, pero para continuidad real entre sesiones debe poder configurarse una ruta persistente, idealmente bajo Google Drive.

Ejemplo:

```text
/content/drive/MyDrive/reinforcement_learning_reto_1/logs/tensorboard/<run_id>/
```

No automatizar OAuth ni montaje de Drive en HU006.

El notebook debe permitir override por variable de entorno o configuración equivalente, por ejemplo:

```text
ASSAULT_TENSORBOARD_DIR
```

No acoplar el logger a `/content`.

---

## 5.12 Notebook

Actualizar:

`2_Assault/assault_ddqn.ipynb`

manteniéndolo como orquestador.

Secuencia objetivo:

```text
bootstrap
↓
config/runtime
↓
HU002 validation
↓
Preflight
↓
new/resume state HU005
↓
TensorBoard config + log dir
↓
crear logger
↓
training corto HU006
↓
flush/close logger
↓
verificar event files
↓
mostrar TensorBoard
↓
resumen
```

El notebook no debe duplicar `SummaryWriter` logic si esta ya está encapsulada en `callbacks.py`.

### Visualización

En Colab puede utilizar:

```python
%load_ext tensorboard
%tensorboard --logdir <ruta>
```

si es compatible con el runtime.

En local debe quedar claro el comando equivalente:

```bash
tensorboard --logdir 2_Assault/logs/tensorboard
```

La ausencia de interfaz gráfica en tests automatizados no bloquea HU006 si los event files se validan programáticamente.

---

## 5.13 Validación programática de TensorBoard

No depender únicamente de inspección visual.

Los tests deben poder leer event files, preferiblemente mediante APIs de TensorBoard como `EventAccumulator`, y confirmar que existen tags esperados.

Conceptualmente:

```python
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
```

Debe validarse al menos:

```text
train/epsilon
train/loss
train/q_mean
train/learning_rate
episode/reward
episode/reward_mean
episode/length
```

Cuando un smoke real sea demasiado corto para finalizar un episodio, los tests controlados deben generar cierres de episodio para verificar los tags `episode/*`.

No inventar episodios en el smoke real únicamente para producir métricas.

---

## 5.14 Frecuencia y costo

TensorBoard no debe introducir un costo desproporcionado.

Evitar:

- imágenes de frames por timestep;
- histogramas de todos los pesos en cada update;
- embeddings;
- graph tracing repetitivo;
- logging de grandes tensors.

HU006 registra principalmente scalars.

Objetivo: observabilidad suficiente con bajo overhead.

---

## 6. Fuera de alcance

HU006 **no** debe implementar:

- MLflow;
- evaluación formal sobre ≥10 episodios;
- comparación automática contra baseline;
- entrenamiento largo/final;
- selección automática del mejor modelo;
- video;
- dashboards personalizados;
- Prometheus/Grafana;
- Weights & Biases;
- logging de frames/imágenes por timestep;
- histogramas masivos;
- profiling avanzado de GPU;
- optimización de hiperparámetros;
- PER;
- Dueling DQN;
- Rainbow;
- Noisy Nets;
- n-step returns;
- automatización Codex → Colab.

HU007 validará TensorBoard + checkpoint + restore + entrenamiento conjuntamente en GPU.

HU008 implementará MLflow.

HU009 realizará el entrenamiento largo.

---

## 7. Decisiones y restricciones técnicas

### 7.1 TensorBoard vs MLflow

Mantener objetivos separados:

```text
TensorBoard
→ curvas detalladas durante entrenamiento
→ diagnóstico temporal

MLflow (HU008)
→ comparación entre experimentos
→ parámetros, hardware, commit, artefactos
```

No duplicar en HU006 la responsabilidad futura de MLflow.

### 7.2 Scalars sobre imágenes

Priorizar scalars que permitan diagnosticar aprendizaje.

No registrar datos únicamente porque TensorBoard lo permita.

### 7.3 Reward raw

`episode/reward` debe usar la recompensa raw del entorno.

### 7.4 Loss finita

El trainer ya bloquea loss no finita. TensorBoard registra la métrica real; no sustituye validaciones funcionales.

### 7.5 Q-value

`train/q_mean` debe ser interpretable y derivarse del batch real utilizado en el update.

### 7.6 Run ID

Checkpointing y TensorBoard deben utilizar el mismo `run_id`.

### 7.7 Docstrings

Clases y funciones reutilizables deben seguir estilo Google según `linemientos.md`.

### 7.8 SOLID/DRY

- `agent.py` produce métricas del update;
- `trainer.py` conoce cuándo ocurrieron eventos;
- `callbacks.py` persiste observabilidad;
- notebook orquesta y visualiza;
- no duplicar cálculos costosos.

---

## 8. Plan de implementación / tareas

### T01 — Dependencia TensorBoard

Actualizar `requirements.txt`.

**Resultado:** `SummaryWriter` y lectura de event files funcionan.

### T02 — Configuración TensorBoard

Extender `ddqn_config.yaml`.

**Resultado:** logging configurable y centralizado.

### T03 — Implementar logger/callback

Crear `src/callbacks.py` con wrapper simple de TensorBoard.

**Resultado:** interfaz desacoplada de `SummaryWriter`.

### T04 — Exponer métricas del agente

Extender `DDQNAgent.update(...)` para devolver `q_mean` y learning rate si corresponde, manteniendo `loss`.

**Resultado:** el trainer no recalcula métricas innecesariamente.

### T05 — Integrar Trainer

Agregar logger opcional y emitir métricas en eventos correctos.

**Resultado:** entrenamiento funciona con logger `None` o TensorBoard.

### T06 — Recompensa media móvil

Calcular/registrar `episode/reward_mean` con ventana configurable.

**Resultado:** curva suavizada disponible sin alterar reward raw.

### T07 — Run ID y paths

Crear logs bajo `<tensorboard.directory>/<run_id>`.

**Resultado:** aislamiento de runs.

### T08 — Resume

Validar que un resume continúa `global_step` y conserva carpeta del run.

**Resultado:** serie temporal no reinicia a cero.

### T09 — Notebook

Integrar creación, cierre y visualización TensorBoard sin duplicar lógica.

### T10 — Tests unitarios/focalizados

Crear preferiblemente:

```text
2_Assault/tests/test_tensorboard.py
```

### T11 — Validación programática event files

Usar EventAccumulator o equivalente oficial.

### T12 — Smoke real Assault

Ejecutar corrida corta con event files reales.

### T13 — Actualizar documentación

Registrar evidencia real en `implementacion.md`.

---

## 9. Criterios de aceptación

### CA01 — Configuración

TensorBoard se configura únicamente desde YAML/overrides explícitos.

### CA02 — Logger desacoplado

La lógica de `SummaryWriter` no está embebida directamente en el algoritmo DDQN.

### CA03 — Logging opcional

El Trainer funciona correctamente con TensorBoard deshabilitado.

### CA04 — Event files

Una corrida con TensorBoard habilitado genera al menos un event file válido.

### CA05 — Epsilon

`train/epsilon` contiene valores reales asociados al timestep usado para seleccionar acciones.

### CA06 — Loss

`train/loss` se registra únicamente en steps donde ocurre update.

### CA07 — Q mean

`train/q_mean` existe, es finito y proviene del cálculo real del update.

### CA08 — Learning rate

`train/learning_rate` refleja el optimizer actual.

### CA09 — Episode reward

`episode/reward` registra recompensa raw al finalizar episodios.

### CA10 — Reward mean

`episode/reward_mean` calcula correctamente la ventana configurada.

### CA11 — Episode length

`episode/length` coincide con la longitud real del episodio.

### CA12 — Global step

Los scalars utilizan global steps correctos y monotónicos dentro de cada segmento de entrenamiento.

### CA13 — Resume

Después de cargar checkpoint en step `N`, los nuevos scalars se registran con steps posteriores a `N`, no desde cero.

### CA14 — Run ID

Logs y checkpoints pueden asociarse al mismo `run_id`.

### CA15 — Aislamiento de runs

Dos `run_id` diferentes escriben en directorios distintos.

### CA16 — Idempotencia

Resume no borra logs previos.

### CA17 — Lectura programática

Los event files pueden leerse con TensorBoard y contienen tags esperados.

### CA18 — CPU/GPU

El logger no causa device mismatch ni requiere copiar grandes tensors innecesariamente.

### CA19 — Notebook

El notebook sigue siendo orquestador y puede mostrar la ruta/comando de TensorBoard.

### CA20 — Scope

No se introduce MLflow ni entrenamiento largo.

---

## 10. Autovalidaciones obligatorias

### AV01 — Imports

Validar imports de `callbacks.py`, Trainer y TensorBoard.

### AV02 — Suite completa

```bash
python -m pytest 2_Assault/tests -q
```

Todos los tests previos deben seguir pasando.

### AV03 — Compile

```bash
python -m compileall -q 2_Assault/src
```

### AV04 — Logger disabled

Entrenar con TensorBoard deshabilitado.

**Esperado:** entrenamiento PASS, sin event files obligatorios.

### AV05 — Logger enabled

Entrenar con logger activo en directorio temporal.

**Esperado:** event file generado.

### AV06 — Epsilon tags

Leer events y verificar `train/epsilon`.

### AV07 — Loss tags

Verificar que `train/loss` aparece únicamente en update steps esperados.

### AV08 — Q-value tags

Verificar `train/q_mean` finito.

### AV09 — Learning-rate tag

Verificar `train/learning_rate`.

### AV10 — Episode tags

Con entorno controlado, verificar:

- `episode/reward`;
- `episode/reward_mean`;
- `episode/length`.

### AV11 — Reward mean

Validar matemáticamente la ventana móvil con rewards conocidos.

### AV12 — Run isolation

Dos run IDs crean carpetas diferentes.

### AV13 — Resume steps

Ejecutar segmento A → checkpoint → nuevos objetos → resume → segmento B.

**Esperado:** nuevos event steps continúan después del `global_step` restaurado.

### AV14 — Existing logs preserved

Validar que resume no elimina event files previos.

### AV15 — Real Assault smoke

Ejecutar:

```text
Preflight
→ Assault
→ DDQN training corto
→ TensorBoard events
```

### AV16 — Notebook local

Ejecutar celdas automatizables localmente cuando sea viable.

### AV17 — TensorBoard parser

EventAccumulator o equivalente carga los logs sin errores.

### AV18 — Scope audit

Confirmar ausencia de MLflow, evaluación formal y entrenamiento largo.

---

## 11. Evidencias requeridas

El PR HU006 debe incluir o referenciar:

- versión de TensorBoard;
- configuración YAML agregada;
- ruta de logs utilizada;
- `run_id`;
- lista de event files generados;
- tamaño básico de los event files;
- tags detectados programáticamente;
- steps de `train/epsilon`;
- steps de `train/loss`;
- valores finitos de `train/q_mean`;
- learning rate registrado;
- episodio reward/mean/length en test controlado;
- resultado resume y continuidad de steps;
- evidencia de logs previos preservados;
- smoke Assault real;
- salida de pytest;
- dispositivo utilizado;
- commit Git;
- confirmación de ausencia de MLflow/scope creep.

La inspección visual de TensorBoard puede añadirse como evidencia, pero no sustituye la validación programática.

---

## 12. Definition of Done

HU006 se considera terminada únicamente cuando:

- [ ] TensorBoard está declarado en dependencias;
- [ ] configuración TensorBoard está centralizada;
- [ ] `callbacks.py` implementa observabilidad desacoplada;
- [ ] Trainer acepta observabilidad opcional;
- [ ] agente expone métricas necesarias sin duplicación costosa;
- [ ] event files se generan correctamente;
- [ ] `train/epsilon` está disponible;
- [ ] `train/loss` está disponible;
- [ ] `train/q_mean` está disponible;
- [ ] `train/learning_rate` está disponible;
- [ ] `episode/reward` está disponible;
- [ ] `episode/reward_mean` está disponible;
- [ ] `episode/length` está disponible;
- [ ] tags/steps se validan programáticamente;
- [ ] global steps no reinician tras resume;
- [ ] logs previos no se eliminan al reanudar;
- [ ] runs diferentes permanecen aislados;
- [ ] TensorBoard puede deshabilitarse sin romper entrenamiento;
- [ ] notebook sigue siendo orquestador;
- [ ] smoke real Assault genera logs válidos;
- [ ] tests previos y nuevos pasan;
- [ ] documentación/evidencia está actualizada;
- [ ] no existen errores bloqueantes conocidos;
- [ ] no se implementó MLflow ni scope HU007+.

---

## 13. Riesgos y consideraciones

### 13.1 Logging excesivo

Registrar demasiado puede reducir throughput y aumentar I/O. HU006 debe priorizar scalars y frecuencias razonables.

### 13.2 Steps inconsistentes

Un error común es usar contador local de episodio o segmento en lugar de `global_step`. Resume debe conservar eje temporal.

### 13.3 Epsilon off-by-one

HU004 identificó una diferencia posible entre epsilon usado en la última acción y epsilon calculado después de incrementar el contador. TensorBoard debe registrar específicamente el epsilon **usado para seleccionar la acción** en el timestep correspondiente.

### 13.4 Q-value costoso

No realizar un segundo forward únicamente para logging si el update ya dispone de Q-values suficientes para calcular una media útil.

### 13.5 Runs mezclados

Usar el mismo directorio para diferentes `run_id` vuelve ambiguas las curvas. Cada run debe quedar aislado.

### 13.6 Logs efímeros en Colab

Si se requiere continuidad entre sesiones, el directorio debe poder apuntar a almacenamiento persistente. No automatizar Google Drive todavía.

### 13.7 Event files múltiples

Un mismo run puede producir múltiples event files tras reanudar. Esto es válido si comparten logdir/run_id y conservan pasos globales coherentes.

---

## 14. Resultado esperado y gate para HU007

HU007 solo debe comenzar cuando HU006 demuestre:

```text
HU005 resumable training
       ↓
TensorBoard logger
       ↓
short training
       ↓
valid event files
       ↓
expected scalar tags
       ↓
correct global steps
       ↓
resume preserves timeline
       ↓
logs readable
       ↓
HU006 PASS
```

No es necesario demostrar todavía mejora estadísticamente significativa de reward. HU006 valida **observabilidad correcta**, no desempeño final del agente.
