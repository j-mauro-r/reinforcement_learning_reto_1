# HU008 — MLflow y trazabilidad de experimentos

## 1. Identificación

- **ID:** HU008
- **Nombre:** MLflow y trazabilidad de experimentos
- **Estado:** Implementada con validaciones locales; lista para validacion Colab multisesion.
- **Dependencia previa:** HU007 — Smoke test end-to-end `[COMPLETADA]`
- **Dependencias funcionales:** HU002/HU002B, HU003, HU004, HU005, HU006 y HU007.
- **Habilita:** HU009 — Entrenamiento DDQN completo.
- **Entorno objetivo:** desarrollo/validación local y ejecución principal en Google Colab GPU.
- **Componente principal esperado:** `2_Assault/src/tracking.py`.
- **Fuentes de verdad:**
  - todos los documentos de `2_Assault/docs/`;
  - `2_Assault/docs/implementacion.md` como mapa maestro;
  - `2_Assault/docs/arquitectura.md`;
  - `2_Assault/docs/linemientos.md`;
  - `2_Assault/docs/ficha_tecnica.md`;
  - `2_Assault/docs/hu005_checkpoints_reanudacion_idempotencia.md`;
  - `2_Assault/docs/hu006_observabilidad_tensorboard.md`;
  - `2_Assault/docs/hu007_smoke_test_end_to_end.md`;
  - `2_Assault/configs/ddqn_config.yaml`;
  - `2_Assault/assault_ddqn.ipynb`;
  - `enunciado_reto_1.txt`.

---

## 2. Contexto y problema

El pipeline técnico de Assault ya permite:

```text
GitHub / bootstrap reproducible
        ↓
entorno + preprocessing
        ↓
DDQN
        ↓
Trainer
        ↓
TensorBoard
        ↓
checkpoint + resume
        ↓
smoke E2E Colab GPU
```

HU007 demostró que estos componentes funcionan conjuntamente en el runtime objetivo. Sin embargo, todavía falta una capa distinta: **poder identificar, reconstruir y comparar los experimentos que produzcan modelos candidatos durante HU009 y HU010**.

TensorBoard permite observar qué ocurre dentro de una corrida, pero no responde por sí solo preguntas como:

- ¿qué commit produjo este modelo?;
- ¿qué hiperparámetros se utilizaron?;
- ¿qué GPU y versiones ejecutaron la corrida?;
- ¿desde qué timestep empezó y en cuál terminó?;
- ¿fue un entrenamiento nuevo o una reanudación?;
- ¿qué checkpoint corresponde al resultado?;
- ¿qué evaluación corta o formal obtuvo?;
- ¿qué experimento debe compararse contra otro?;

HU008 introduce **MLflow únicamente como sistema de tracking y comparación de experimentos**, manteniendo la filosofía MLOps ligera del proyecto.

Flujo conceptual:

```text
GitHub SHA + configuración
          ↓
Training / resume
          ├────────────→ TensorBoard
          │              curvas internas
          │
          ├────────────→ Checkpoints
          │              continuidad
          │
          └────────────→ MLflow
                         identidad + parámetros
                         métricas agregadas
                         hardware/versiones
                         referencias a artefactos
                         comparación entre runs
```

MLflow **no sustituye** TensorBoard, checkpointing, GitHub ni el notebook.

---

## 3. Historia de usuario

> **Como** equipo que va a ejecutar entrenamientos DDQN costosos y compararlos posteriormente, **quiero** que cada experimento relevante quede registrado en MLflow con su código, configuración, hardware, progreso, métricas y artefactos asociados, **para** poder reproducirlo, compararlo y justificar técnicamente qué configuración produjo cada resultado.

---

## 4. Objetivo verificable

HU008 debe demostrar que una corrida corta y controlada puede registrarse en MLflow de forma reproducible y desacoplada, incluyendo como mínimo:

1. `algorithm=DDQN`;
2. identificador lógico del experimento/run;
3. commit Git ejecutado;
4. configuración del entorno;
5. preprocessing;
6. seed;
7. hiperparámetros relevantes;
8. versiones principales;
9. hardware/runtime;
10. timestep inicial;
11. timestep final;
12. tiempo de entrenamiento;
13. métricas agregadas de entrenamiento útiles;
14. métricas de evaluación disponibles;
15. referencia explícita al checkpoint/modelo;
16. configuración utilizada como artefacto;
17. resumen de evaluación como artefacto;
18. posibilidad de consultar nuevamente el run desde MLflow;
19. separación entre runs distintos;
20. continuidad/trazabilidad de una ejecución reanudada;
21. funcionamiento del Trainer aunque MLflow esté deshabilitado;
22. coexistencia con TensorBoard sin duplicar responsabilidades.

Resultado esperado:

```text
MLflow store configured
↓
experiment created/resolved
↓
run registered
↓
params + tags logged
↓
training summary logged
↓
evaluation summary logged
↓
artifact references logged
↓
run queried back successfully
↓
MLFLOW_TRACKING_PASS=True
```

---

## 5. Alcance

HU008 es una **capa de tracking**, no una modificación del algoritmo DDQN.

Debe reutilizar el pipeline existente y agregar únicamente la responsabilidad de MLflow.

Archivos esperados:

```text
2_Assault/
├── configs/ddqn_config.yaml
├── src/
│   └── tracking.py
├── tests/
│   └── test_tracking.py
├── assault_ddqn.ipynb
├── requirements.txt
└── docs/
    ├── implementacion.md
    └── hu008_mlflow_trazabilidad_experimentos.md
```

No crear componentes adicionales sin una responsabilidad clara.

---

## 5.1 Responsabilidad de `src/tracking.py`

`tracking.py` debe encapsular el uso de MLflow.

Responsabilidades permitidas:

- configurar/resolver tracking URI;
- crear o resolver experimentos MLflow;
- iniciar/reanudar/cerrar un run de tracking;
- registrar parámetros y tags;
- registrar métricas agregadas;
- registrar artefactos livianos;
- registrar referencias a checkpoints/modelos;
- consultar/validar un run creado;
- manejar modo `enabled=false` sin romper entrenamiento.

No debe:

- seleccionar acciones;
- calcular targets DDQN;
- ejecutar optimizer;
- crear el entorno;
- administrar Replay Buffer;
- implementar TensorBoard;
- decidir cuándo sincronizar Target Network;
- reemplazar `CheckpointManager`;
- contener lógica de evaluación.

Interfaz conceptual permitida:

```python
tracker = MLflowTracker.from_config(
    config=config,
    run_id=run_id,
    git_commit=git_commit,
)

tracker.start(...)
tracker.log_training_summary(...)
tracker.log_evaluation_summary(...)
tracker.log_checkpoint_reference(...)
tracker.finish()
```

Los nombres concretos pueden adaptarse si el diseño mantiene la separación de responsabilidades.

---

## 5.2 MLflow como tracking, no deployment

La implementación debe usar funciones de tracking de MLflow.

Queda fuera del alcance:

- Model Registry como requisito;
- MLflow Serving;
- deployments;
- Kubernetes;
- servidor MLflow administrado;
- autenticación empresarial;
- pipelines CI/CD de MLflow;
- feature store.

La solución debe seguir siendo ejecutable por un estudiante desde local/Colab.

---

## 5.3 Backend / tracking URI

El backend de MLflow debe ser **configurable**.

Prioridades:

1. permitir override mediante variable de entorno, por ejemplo:

```text
ASSAULT_MLFLOW_TRACKING_URI
```

2. permitir configuración central en YAML;
3. disponer de una opción local basada en filesystem para tests y desarrollo;
4. permitir una ruta persistente montada en Google Drive para entrenamientos repartidos entre sesiones de Colab.

Ejemplo conceptual:

```yaml
mlflow:
  enabled: true
  experiment_name: assault_ddqn
  tracking_uri: null
  local_directory: logs/mlflow
  log_checkpoint_binary: false
```

Los nombres pueden adaptarse.

Cuando `tracking_uri` no se establezca explícitamente, `tracking.py` puede construir un URI local seguro a partir del directorio configurado.

No hardcodear rutas personales de Windows ni `/content/drive/...` dentro de módulos Python.

---

## 5.4 Persistencia en Colab

HU009 podrá distribuirse entre varias sesiones de Colab. Por tanto, el tracking no debe depender obligatoriamente del filesystem efímero de `/content` para experimentos importantes.

HU008 debe demostrar que el tracking URI puede apuntar a una ruta externa/configurable, por ejemplo un directorio de Google Drive ya montado por el usuario.

No automatizar OAuth ni montaje de Drive.

El notebook puede aceptar una variable como:

```text
ASSAULT_MLFLOW_TRACKING_URI=file:///content/drive/MyDrive/.../mlruns
```

La ubicación exacta queda bajo configuración del usuario.

Para tests locales, usar directorios temporales.

---

## 5.5 Identidad: proyecto `run_id` vs MLflow `run_id`

El proyecto ya utiliza un `run_id` lógico, por ejemplo:

```text
assault_ddqn_exp_001
```

MLflow genera además su propio identificador técnico.

No deben confundirse.

Usar conceptos explícitos equivalentes a:

```text
project_run_id = assault_ddqn_exp_001
mlflow_run_id = <ID generado por MLflow>
```

El `project_run_id` debe quedar registrado como tag o parámetro reservado del run MLflow.

El `mlflow_run_id` debe quedar disponible para trazabilidad y para poder reabrir explícitamente el mismo tracking run cuando corresponda.

No utilizar únicamente el nombre visible del run como identificador técnico.

---

## 5.6 Experimento MLflow

Se recomienda un experimento MLflow principal:

```text
assault_ddqn
```

Los runs representan ejecuciones/configuraciones concretas:

```text
assault_ddqn_exp_001
assault_ddqn_exp_002
...
```

No crear un experimento MLflow distinto para cada pequeño segmento de una misma corrida lógica salvo que exista una justificación documentada.

Esto permite comparar runs bajo un mismo experimento.

---

## 5.7 Continuidad entre sesiones y resume

HU005 permite reanudar entrenamiento. HU008 debe conservar trazabilidad cuando una corrida lógica continúa en otra sesión.

Objetivo conceptual:

```text
project_run_id = assault_ddqn_exp_001
MLflow run = ABC123

session A
step 0 → N
checkpoint @ N
↓
sesión termina
↓
session B
explicit resume checkpoint @ N
MLflow run = ABC123
step N → T
```

La implementación debe permitir reabrir **explícitamente** el mismo `mlflow_run_id` cuando se reanuda el mismo experimento lógico.

No debe seleccionar automáticamente un run ambiguo únicamente por ser “el último”.

El estado necesario para recuperar el `mlflow_run_id` debe persistirse de manera desacoplada y liviana, por ejemplo mediante metadata/sidecar de tracking asociado al `project_run_id`, o mecanismo equivalente claramente documentado.

### Restricción de arquitectura

No acoplar `CheckpointManager` directamente a la API de MLflow.

Si se decide extender metadata transversal, debe hacerse sin convertir checkpointing en responsable de tracking.

---

## 5.8 Idempotencia

Reejecutar una celda del notebook no debe crear silenciosamente múltiples runs MLflow para el mismo intento cuando el usuario esperaba continuar uno existente.

Debe existir distinción explícita entre:

```text
tracking_mode = new
tracking_mode = resume
```

o contrato equivalente.

### `new`

- crea un MLflow run nuevo;
- devuelve/persiste su `mlflow_run_id`;
- no reutiliza otro run automáticamente.

### `resume`

- requiere `mlflow_run_id` explícito o metadata inequívoca;
- reabre ese run;
- conserva la identidad del experimento;
- registra el timestep inicial restaurado.

No usar heurísticas tipo “último run del experimento”.

---

## 5.9 Parámetros obligatorios

Registrar como parámetros o tags, según convenga, al menos:

### Identidad

- `algorithm=DDQN`;
- `project_run_id`;
- experiment name;
- seed.

### Git / reproducibilidad

- commit SHA ejecutado;
- ref solicitada cuando esté disponible;
- runtime (`local` / `Google Colab`).

### Entorno

- environment ID;
- observation type;
- action space / número de acciones;
- effective frameskip;
- repeat action probability;
- full action space.

### Preprocessing

- grayscale;
- resize height;
- resize width;
- frame stack;
- dtype;
- normalización.

### DDQN

- gamma;
- learning rate;
- epsilon start;
- epsilon final;
- epsilon decay steps;
- batch size;
- Replay Buffer capacity;
- learning starts;
- train frequency;
- Target Network update frequency;
- total timesteps objetivo.

### Versiones

- Python;
- Gymnasium;
- ALE-Py;
- PyTorch;
- MLflow.

### Hardware

Como tags/params:

- device;
- GPU disponible;
- nombre GPU si aplica;
- VRAM total cuando esté disponible;
- CPU;
- RAM total cuando esté disponible.

No registrar listas/dicts complejos directamente como params sin serialización controlada.

---

## 5.10 Métricas obligatorias

HU008 debe permitir registrar métricas agregadas que sean útiles para comparar experimentos.

Como mínimo, cuando estén disponibles:

### Entrenamiento

- `train/initial_global_step`;
- `train/final_global_step`;
- `train/duration_seconds`;
- `train/updates_count`;
- `train/last_loss`;
- `train/mean_loss`;
- `train/last_q_mean`;
- `train/mean_q_mean`;
- `train/final_epsilon`;
- episodios completados;
- mejor recompensa de entrenamiento disponible si existe evidencia real.

### Evaluación

- `eval/episodes`;
- `eval/mean_reward`;
- `eval/median_reward`;
- `eval/std_reward`;
- `eval/min_reward`;
- `eval/max_reward`;
- `eval/mean_episode_length`;
- `eval/epsilon`.

Usar únicamente valores reales existentes.

No registrar `0` artificialmente para una métrica ausente.

---

## 5.11 MLflow no debe duplicar TensorBoard paso a paso

TensorBoard ya registra métricas temporales detalladas:

- loss por update;
- epsilon;
- Q-value;
- recompensa por episodio;
- learning rate;
- timestep global.

HU008 **no debe duplicar automáticamente cada scalar de TensorBoard en MLflow**.

MLflow se usará principalmente para:

- parámetros;
- tags;
- métricas agregadas/comparables;
- estado inicial/final;
- evaluación;
- artefactos/referencias.

Esto evita ruido, almacenamiento innecesario y doble instrumentación del Trainer.

Si posteriormente se decide registrar alguna serie temporal MLflow por una razón analítica concreta, debe justificarse.

---

## 5.12 Artefactos obligatorios

Registrar como artefactos livianos al menos:

### Configuración

Una copia exacta de la configuración usada, por ejemplo:

```text
config/ddqn_config.yaml
```

### Runtime / metadata

Un JSON equivalente a:

```text
metadata/runtime.json
```

con versiones, hardware, commit y contexto de ejecución.

### Training summary

```text
summaries/training_summary.json
```

### Evaluation summary

Cuando exista evaluación:

```text
summaries/evaluation_summary.json
```

### Checkpoint reference

Registrar metadata o archivo liviano:

```text
artifacts/checkpoint_reference.json
```

incluyendo como mínimo:

- path/reference;
- run_id lógico;
- checkpoint step;
- tamaño;
- modo de persistencia;
- hash si se implementa sin costo relevante.

---

## 5.13 Checkpoint binario y modelos pesados

No subir automáticamente cada checkpoint completo a MLflow.

El proyecto ya dispone de `CheckpointManager`, y `resume_full` puede generar artefactos grandes por incluir Replay Buffer.

Default recomendado:

```text
mlflow.log_checkpoint_binary = false
```

Registrar referencia y metadata del checkpoint.

Puede existir una opción explícita para registrar un modelo/checkpoint seleccionado cuando sea razonable, pero no debe provocar duplicación masiva ni ralentizar cada checkpoint periódico.

HU009/HU012 decidirán qué modelo final debe preservarse como artefacto de entrega.

---

## 5.14 Relación con TensorBoard

Debe mantenerse esta separación:

```text
TensorBoard
→ evolución interna de una corrida
→ diagnóstico temporal
→ curvas

MLflow
→ identidad del experimento
→ configuración
→ métricas agregadas
→ comparación entre runs
→ trazabilidad de artefactos
```

Una falla de MLflow no debe corromper los logs TensorBoard ya existentes.

No modificar `TensorBoardLogger` para que dependa de MLflow.

---

## 5.15 Relación con Trainer

El Trainer debe conservar baja dependencia respecto a MLflow.

Preferencia arquitectónica:

```text
Trainer
↓
TrainingSummary
↓
tracking.py
↓
MLflow
```

En lugar de:

```text
Trainer
↓
mlflow.log_* en múltiples líneas internas
```

HU008 debe aprovechar `TrainingSummary`, `EvaluationSummary`, configuración y metadata existentes.

Solo introducir hooks dentro del Trainer si existe una necesidad demostrable que no pueda resolverse con los resúmenes ya producidos.

No cambiar el comportamiento de aprendizaje por habilitar/deshabilitar MLflow.

---

## 5.16 Relación con evaluator

`evaluator.py` continúa siendo responsable únicamente de evaluación.

Flujo esperado:

```text
evaluate_agent(...)
↓
EvaluationSummary
↓
MLflowTracker.log_evaluation_summary(...)
```

No importar MLflow dentro de `evaluator.py`.

---

## 5.17 Run terminado vs corrida reanudable

HU008 debe distinguir entre:

- cerrar temporalmente una sesión de cómputo;
- finalizar lógicamente un experimento.

MLflow permite reabrir un run por ID. La implementación puede terminar la sesión MLflow al finalizar un proceso y posteriormente reabrir el mismo `mlflow_run_id` durante `resume`, siempre que el identificador se preserve explícitamente.

El estado MLflow (`FINISHED`, etc.) no debe confundirse con el estado del entrenamiento DDQN.

La fuente de verdad del progreso de entrenamiento sigue siendo el checkpoint/progreso restaurado.

---

## 5.18 Fallos de tracking

MLflow es importante para trazabilidad, pero no debe producir corrupción silenciosa del entrenamiento.

La política debe ser explícita:

### Antes de entrenamiento largo

Si `mlflow.enabled=true` y el tracking store no puede inicializarse:

```text
FAIL FAST
```

porque HU009 exige trazabilidad completa.

### Durante logging

Errores de escritura deben propagarse de forma visible y dejar claro que la corrida no cumple trazabilidad.

No ocultar errores con `except Exception: pass`.

### Cuando MLflow está deshabilitado

El pipeline debe seguir funcionando para tests o ejecución focalizada que explícitamente no requiera tracking.

---

## 5.19 Configuración centralizada

Extender `ddqn_config.yaml` con un bloque equivalente a:

```yaml
mlflow:
  enabled: true
  experiment_name: assault_ddqn
  tracking_uri: null
  local_directory: logs/mlflow
  tracking_mode: new
  mlflow_run_id: null
  log_checkpoint_binary: false
```

Los nombres pueden adaptarse.

Variables de entorno pueden sobrescribir rutas/IDs sensibles al runtime, por ejemplo:

```text
ASSAULT_MLFLOW_TRACKING_URI
ASSAULT_MLFLOW_RUN_ID
ASSAULT_MLFLOW_TRACKING_MODE
```

No hardcodear experimentos, rutas o IDs dentro de `tracking.py`.

---

## 5.20 Notebook

Actualizar:

`2_Assault/assault_ddqn.ipynb`

manteniéndolo como orquestador.

Secuencia HU008 objetivo:

```text
bootstrap GitHub
↓
runtime/config
↓
resolver MLflow tracking URI
↓
crear/resolver experiment
↓
new/resume tracking run explícito
↓
registrar params/tags/runtime
↓
ejecutar corrida corta controlada
↓
registrar TrainingSummary
↓
evaluación corta
↓
registrar EvaluationSummary
↓
registrar configuración + metadata + checkpoint reference
↓
consultar run MLflow
↓
resumen
MLFLOW_TRACKING_PASS=True
```

No ejecutar entrenamiento largo dentro de HU008.

El notebook debe imprimir al menos:

```text
MLflow tracking URI
MLflow experiment name
project_run_id
mlflow_run_id
Git SHA
initial_global_step
final_global_step
checkpoint reference
evaluation mean reward
MLFLOW_TRACKING_PASS=True / False
```

---

## 5.21 Visualización MLflow

HU008 debe documentar cómo inspeccionar la UI MLflow cuando sea práctico.

Localmente puede utilizarse un comando equivalente a:

```bash
mlflow ui --backend-store-uri <tracking-uri>
```

En Colab, la UI no debe convertirse en requisito técnico si el entorno no permite exponerla de forma simple.

La validación programática mediante `MlflowClient` es obligatoria y suficiente para demostrar que el tracking funciona.

No introducir túneles externos únicamente para visualizar la UI.

---

## 6. Fuera de alcance

HU008 **no** debe implementar:

- entrenamiento DDQN largo/final;
- HU009;
- optimización de hiperparámetros;
- HU010;
- evaluación formal de ≥10 episodios;
- comparación formal contra baseline;
- HU011;
- video final;
- Model Registry obligatorio;
- MLflow Serving;
- deployment;
- Kubernetes;
- servidor cloud MLflow administrado;
- dashboard diferente de TensorBoard/MLflow UI;
- PER;
- Dueling DQN;
- Rainbow;
- Noisy Nets;
- n-step returns;
- reward clipping no aprobado;
- distributed training;
- cambios al algoritmo DDQN;
- automatización Codex → Colab.

---

## 7. Decisiones técnicas

### 7.1 MLflow y TensorBoard son complementarios

No duplicar todas las series de TensorBoard en MLflow. MLflow registra identidad, configuración, métricas agregadas y artefactos comparables.

### 7.2 Tracking file-based es válido

Para este reto académico, un backend filesystem local/persistente es suficiente si cumple trazabilidad y puede apuntar a almacenamiento persistente en Colab.

No se necesita un servidor MLflow dedicado para aprobar HU008.

### 7.3 `project_run_id` y `mlflow_run_id` son diferentes

Ambos deben quedar visibles y asociados.

### 7.4 Resume explícito

Una reanudación debe conocer el `mlflow_run_id` exacto. No buscar automáticamente “el último run”.

### 7.5 Checkpoint como referencia por defecto

Evitar duplicar Replay Buffer/checkpoints grandes dentro de MLflow. Registrar metadata/referencia es suficiente para HU008.

### 7.6 Configuración completa como artefacto

Además de params planos, conservar el YAML exacto utilizado evita perder estructura y facilita reproducción.

### 7.7 Métricas ausentes no son cero

No fabricar valores para cumplir esquemas.

### 7.8 Fail fast para HU009

Si tracking está habilitado y falla, un entrenamiento largo no debe arrancar como si fuera trazable.

---

## 8. Plan de implementación / tareas

### T01 — Revisar toda la base documental

Leer `2_Assault/docs/` completo antes de modificar código y confirmar contratos de HU002–HU007.

### T02 — Agregar dependencia MLflow

Agregar versión compatible a `requirements.txt` y validar instalación local/Colab.

### T03 — Configuración MLflow

Agregar bloque central mínimo y overrides por entorno.

### T04 — Implementar `tracking.py`

Encapsular creación/resume, params, tags, métricas, artefactos y consulta.

### T05 — Flatten/serialización segura

Transformar configuración a parámetros MLflow sin exceder contratos ni registrar estructuras inválidas.

### T06 — Runtime metadata

Reutilizar `utils.py`/bootstrap para registrar versiones, hardware y SHA sin duplicar lógica.

### T07 — Training summary

Registrar métricas agregadas provenientes de `TrainingSummary`.

### T08 — Evaluation summary

Registrar métricas provenientes de `EvaluationSummary`.

### T09 — Artefactos

Registrar configuración exacta, runtime metadata, summaries y checkpoint reference.

### T10 — Idempotencia new/resume

Validar nuevo run y reanudación explícita del mismo MLflow run.

### T11 — Run isolation

Crear dos runs distintos y confirmar que parámetros/métricas no se mezclan.

### T12 — Modo disabled

Confirmar que MLflow deshabilitado no modifica Trainer/evaluator.

### T13 — Tests

Agregar tests focalizados usando tracking store temporal.

### T14 — Integración notebook

Orquestar una corrida corta con tracking y consulta programática.

### T15 — Validación Colab

Ejecutar el flujo HU008 con GPU/Colab cuando sea necesario para confirmar compatibilidad del tracking en el runtime objetivo, sin entrenamiento largo.

### T16 — Actualizar documentación

Registrar únicamente evidencia realmente ejecutada en `implementacion.md`.

---

## 9. Criterios de aceptación

### CA01 — Dependency

MLflow está declarado reproduciblemente en `requirements.txt`.

### CA02 — Tracking module

Existe `src/tracking.py` con responsabilidad exclusiva de MLflow.

### CA03 — Optionality

Con `mlflow.enabled=false`, entrenamiento/evaluación existentes continúan funcionando sin cambios funcionales.

### CA04 — Config central

Experiment name, tracking URI/directorio, modo y opciones se resuelven desde configuración/overrides, no constantes dispersas.

### CA05 — Experiment

Se puede crear/resolver el experimento `assault_ddqn` o nombre configurado.

### CA06 — Identity

Cada run registra `project_run_id` y expone su `mlflow_run_id` técnico.

### CA07 — Git traceability

Commit SHA ejecutado queda registrado.

### CA08 — Environment traceability

Environment ID, acciones, frameskip y repeat action probability quedan registrados.

### CA09 — Preprocessing traceability

Resize, grayscale, frame stack, dtype y normalización quedan registrados.

### CA10 — Hyperparameters

Parámetros DDQN/training relevantes quedan registrados.

### CA11 — Runtime versions

Python, Gymnasium, ALE-Py, PyTorch y MLflow quedan registrados.

### CA12 — Hardware

Device/CPU/GPU/RAM disponibles quedan registrados sin asumir hardware específico.

### CA13 — Training metrics

TrainingSummary produce métricas MLflow válidas y finitas cuando corresponda.

### CA14 — Evaluation metrics

EvaluationSummary registra media, mediana, std, min, max, episodios y epsilon.

### CA15 — Config artifact

La configuración exacta usada queda registrada como artefacto.

### CA16 — Runtime artifact

Metadata de runtime/reproducibilidad queda registrada como artefacto.

### CA17 — Summary artifacts

Training/evaluation summaries quedan registradas como JSON o formato equivalente.

### CA18 — Checkpoint reference

El run contiene referencia inequívoca al checkpoint/modelo asociado.

### CA19 — No binary duplication by default

Los checkpoints completos no se duplican automáticamente en MLflow.

### CA20 — Query back

`MlflowClient` puede recuperar el run y verificar params, metrics, tags y artefactos esperados.

### CA21 — Run isolation

Dos `project_run_id` distintos producen runs MLflow distintos y no mezclan datos.

### CA22 — Explicit resume

Una continuación puede reabrir explícitamente el mismo `mlflow_run_id`.

### CA23 — Resume continuity

El run reanudado registra `initial_global_step=N` y un `final_global_step>T`/posterior sin cambiar su identidad MLflow.

### CA24 — No ambiguous auto-resume

No existe selección automática basada únicamente en “latest run”.

### CA25 — TensorBoard independence

TensorBoard funciona sin depender del tracker MLflow y viceversa.

### CA26 — Trainer decoupling

No se dispersan llamadas `mlflow.log_*` por la lógica interna del Trainer sin necesidad justificada.

### CA27 — Evaluator decoupling

`evaluator.py` no importa ni conoce MLflow.

### CA28 — Colab compatibility

Tracking URI puede configurarse para filesystem local o ruta persistente montada en Colab.

### CA29 — Fail fast

Con MLflow habilitado, un tracking URI inválido/fallo material produce error visible antes de considerar válida la corrida.

### CA30 — Scope

No se introduce entrenamiento largo, HPO, evaluación formal ni despliegue MLflow.

---

## 10. Autovalidaciones obligatorias

### AV01 — Imports

Validar imports de `tracking.py` y módulos HU002–HU008.

### AV02 — Compile

```bash
python -m compileall -q 2_Assault/src
```

### AV03 — Suite completa

```bash
python -m pytest 2_Assault/tests -q
```

Todos los tests previos deben seguir pasando.

### AV04 — Temporary tracking store

Usar `tmp_path`/directorio temporal y confirmar creación de experiment/run sin depender de servicio externo.

### AV05 — Params/tags

Crear run controlado y validar parámetros/tags obligatorios mediante `MlflowClient`.

### AV06 — Runtime metadata

Validar versiones/hardware/commit disponibles sin inventar campos ausentes.

### AV07 — Training metrics

Registrar un `TrainingSummary` controlado y consultar nuevamente métricas.

### AV08 — Evaluation metrics

Registrar un `EvaluationSummary` controlado y consultar media/mediana/std/min/max.

### AV09 — Artifact config

Confirmar que el YAML/config usado existe dentro de los artefactos del run.

### AV10 — Artifact summaries

Confirmar training/evaluation summary artifacts.

### AV11 — Checkpoint reference

Confirmar metadata del checkpoint sin subir binario cuando `log_checkpoint_binary=false`.

### AV12 — Run isolation

Crear run A y run B; confirmar IDs distintos y ausencia de contaminación cruzada.

### AV13 — Explicit resume

```text
new MLflow run
→ persist mlflow_run_id
→ close session
→ reopen same mlflow_run_id explicitly
→ log second segment
```

Esperado: mismo MLflow run técnico.

### AV14 — Global-step resume metadata

Registrar segmento A `0→N`, reabrir run y registrar segmento B `N→T`. Validar identidad y métricas finales.

### AV15 — Ambiguous resume rejection

Modo resume sin `mlflow_run_id`/metadata inequívoca debe fallar de manera explícita.

### AV16 — Disabled mode

Con MLflow deshabilitado, una corrida corta continúa sin crear tracking artifacts.

### AV17 — Invalid backend fail-fast

Con MLflow habilitado y backend inválido/no escribible, la inicialización debe fallar de forma visible.

### AV18 — TensorBoard coexistence

Corrida corta genera TensorBoard y MLflow simultáneamente sin interferencia.

### AV19 — Real Assault short run

Ejecutar entrenamiento corto real de Assault y registrar TrainingSummary + checkpoint reference + evaluación corta.

No prolongar entrenamiento para obtener desempeño.

### AV20 — Notebook local

Ejecutar celdas automatizables localmente con tracking store temporal y comprobar:

```text
MLFLOW_TRACKING_PASS=True
```

### AV21 — Colab compatibility

Ejecutar notebook en Colab con tracking filesystem/configurable y verificar que run/artifacts pueden consultarse desde MLflow.

No requiere servidor remoto ni UI pública.

### AV22 — Persistence configuration

Demostrar que el tracking URI puede configurarse hacia una ruta persistente externa a `/content`; no es obligatorio automatizar el montaje.

### AV23 — Scope audit

Confirmar ausencia de entrenamiento largo, HPO, evaluación formal y deployment MLflow.

---

## 11. Evidencias requeridas

Registrar para la validación de HU008:

- branch/ref;
- commit SHA;
- versión MLflow;
- tracking URI usado (sin secretos);
- experiment name;
- experiment ID;
- `project_run_id`;
- `mlflow_run_id`;
- tracking mode (`new`/`resume`);
- runtime;
- device/hardware;
- seed;
- environment/preprocessing;
- hiperparámetros principales;
- initial/final global step;
- training duration;
- TrainingSummary registrado;
- EvaluationSummary registrado;
- checkpoint reference;
- lista de artefactos;
- resultado de consulta `MlflowClient`;
- evidencia de run isolation;
- evidencia de resume explícito;
- resultado TensorBoard coexistente;
- resultado pytest;
- resultado compileall;
- resultado notebook local;
- resultado Colab cuando se ejecute;
- `MLFLOW_TRACKING_PASS`;
- limitaciones/warnings reales.

No incluir tokens, credenciales o secretos en tags, params, logs, notebook outputs o GitHub.

---

## 12. Definition of Done

HU008 se considera terminada únicamente cuando:

- [ ] se leyó y respetó la base documental `2_Assault/docs/`;
- [ ] MLflow está declarado en dependencias;
- [ ] existe `src/tracking.py` desacoplado;
- [ ] existe configuración central MLflow;
- [ ] tracking URI es configurable;
- [ ] filesystem temporal funciona para tests;
- [ ] ruta persistente Colab puede configurarse;
- [ ] experiment se crea/resuelve correctamente;
- [ ] `project_run_id` y `mlflow_run_id` se distinguen;
- [ ] params/tags obligatorios quedan registrados;
- [ ] Git SHA queda registrado;
- [ ] environment/preprocessing quedan registrados;
- [ ] hiperparámetros quedan registrados;
- [ ] versiones/hardware quedan registrados;
- [ ] TrainingSummary queda registrado;
- [ ] EvaluationSummary queda registrado;
- [ ] configuración exacta queda como artefacto;
- [ ] runtime metadata queda como artefacto;
- [ ] summaries quedan como artefactos;
- [ ] checkpoint/model reference queda registrado;
- [ ] checkpoint binario no se duplica por defecto;
- [ ] run se consulta programáticamente con `MlflowClient`;
- [ ] runs distintos permanecen aislados;
- [ ] resume explícito reutiliza el mismo `mlflow_run_id`;
- [ ] resume ambiguo se rechaza;
- [ ] MLflow disabled no rompe pipeline;
- [ ] TensorBoard y MLflow coexisten;
- [ ] Trainer no queda acoplado innecesariamente a MLflow;
- [ ] evaluator no depende de MLflow;
- [ ] real Assault short tracking funciona;
- [ ] notebook orquesta tracking sin duplicar lógica;
- [ ] suite completa pasa;
- [ ] compileall pasa;
- [ ] `MLFLOW_TRACKING_PASS=True`;
- [ ] documentación/evidencia real queda actualizada;
- [ ] no existen blockers conocidos;
- [ ] no se implementó scope HU009+.

---

## 13. Riesgos y consideraciones

### 13.1 Filesystem de Colab

`/content` es efímero. Para HU009, un experimento que deba sobrevivir sesiones debe usar tracking store persistente o copiarse/persistirse antes de finalizar la sesión.

### 13.2 Google Drive

Drive puede utilizarse como filesystem persistente, pero I/O puede ser más lento. HU008 no debe convertir esto en infraestructura compleja.

### 13.3 Checkpoints grandes

`resume_full` puede incluir Replay Buffer y producir archivos grandes. Duplicarlos como artefactos MLflow en cada checkpoint sería costoso.

### 13.4 Cantidad de parámetros

MLflow params son strings/escalares. La configuración estructurada debe además conservarse como YAML completo.

### 13.5 Cambios de configuración en resume

HU005 valida compatibilidad del checkpoint. HU008 debe registrar la configuración realmente utilizada y no ocultar diferencias entre sesiones.

### 13.6 Experimentos duplicados

Reejecutar el notebook sin distinguir `new`/`resume` puede crear duplicados. El contrato debe ser explícito y visible.

### 13.7 Métricas parciales

Un segmento de entrenamiento no equivale a evaluación final. No interpretar la evaluación corta de HU008 como desempeño del reto.

### 13.8 MLflow unavailable

Antes de HU009, tracking habilitado pero no disponible debe impedir iniciar una corrida que luego no pueda reconstruirse.

### 13.9 Seguridad

No registrar variables de entorno completas, tokens, paths sensibles con secretos ni credenciales.

---

## 14. Resultado esperado y gate

HU008 debe cerrar con evidencia equivalente a:

```text
GitHub SHA verified
        ↓
MLflow tracking URI resolved
        ↓
experiment assault_ddqn
        ↓
project_run_id ↔ mlflow_run_id
        ↓
params/tags/runtime logged
        ↓
short Assault training
        ↓
TrainingSummary logged
        ↓
checkpoint reference logged
        ↓
short evaluation
        ↓
EvaluationSummary logged
        ↓
config + metadata artifacts
        ↓
MlflowClient query PASS
        ↓
explicit resume same MLflow run PASS
        ↓
TensorBoard coexistence PASS
        ↓
MLFLOW_TRACKING_PASS=True
```

### Gate hacia HU009

HU009 no debe iniciar el primer entrenamiento DDQN largo hasta que HU008 demuestre que una corrida relevante puede:

- identificarse;
- reconstruirse;
- persistir su tracking;
- reanudarse manteniendo trazabilidad;
- asociarse inequívocamente con código, configuración, checkpoint y métricas.

**Habilita:** HU009 — Entrenamiento DDQN completo.

---

## 15. Enmienda de trazabilidad por sesión de ejecución

> **Prioridad normativa:** esta sección complementa y, cuando exista conflicto, tiene prioridad sobre las secciones anteriores de HU008 respecto a `new`, `resume`, artefactos y evidencia entre sesiones.

### 15.1 Problema que debe resolver la implementación

Un `project_run_id` representa un experimento lógico y un `mlflow_run_id` representa su run técnico en MLflow, pero una misma corrida puede ejecutarse en varias sesiones de cómputo independientes, por ejemplo dos runtimes distintos de Google Colab.

La implementación debe poder responder inequívocamente:

- qué ocurrió en cada sesión;
- qué Git SHA y hardware usó cada sesión;
- desde qué `global_step` inició y en cuál terminó;
- qué checkpoint cargó o produjo;
- qué métricas pertenecen a cada sesión;
- qué evaluación se ejecutó en cada sesión;
- en qué orden ocurrieron las sesiones;
- y confirmar que todas pertenecen al mismo `project_run_id` y `mlflow_run_id` cuando se trata de un `resume` real.

No es suficiente conservar únicamente el último `training_summary.json`, `evaluation_summary.json` o `runtime.json` con nombres fijos.

### 15.2 Nueva identidad: `tracking_session_id`

Además de:

```text
project_run_id
mlflow_run_id
```

debe existir un identificador explícito por sesión:

```text
tracking_session_id
```

Ejemplo:

```text
project_run_id = assault_ddqn_exp_001
mlflow_run_id = ABC123

tracking_session_id = session_001
tracking_session_id = session_002
```

El nombre concreto puede adaptarse (`session_id`, `execution_session_id`, etc.), pero el concepto debe ser explícito y persistente.

Cada ejecución del notebook que represente una sesión de cómputo debe utilizar un `tracking_session_id` único dentro del mismo `project_run_id`.

Debe poder suministrarse explícitamente mediante configuración o variable de entorno, por ejemplo:

```text
ASSAULT_MLFLOW_SESSION_ID=session_001
```

No inferir silenciosamente una sesión existente como “la última”. Si se implementa generación automática para `new`, el identificador generado debe imprimirse y quedar persistido antes de iniciar el trabajo relevante.

### 15.3 Un MLflow run lógico, múltiples sesiones trazables

Para una reanudación del mismo experimento:

```text
MLflow experiment: assault_ddqn
└── mlflow_run_id = ABC123
    ├── project_run_id = assault_ddqn_exp_001
    ├── session_001
    │   └── step 0 → N
    └── session_002
        └── step N → T
```

No crear un segundo `mlflow_run_id` para representar únicamente una nueva sesión del mismo entrenamiento reanudado.

Un nuevo `mlflow_run_id` corresponde a un experimento/run lógico distinto, no a una sesión adicional del mismo experimento.

### 15.4 Artefactos inmutables por sesión

Los artefactos variables entre sesiones deben almacenarse en un namespace específico de sesión.

Estructura objetivo equivalente:

```text
sessions/
  session_001/
    session_metadata.json
    runtime.json
    training_summary.json
    evaluation_summary.json
    checkpoint_reference.json
  session_002/
    session_metadata.json
    runtime.json
    training_summary.json
    evaluation_summary.json
    checkpoint_reference.json
```

Los nombres pueden adaptarse, pero una sesión posterior **no debe sobrescribir** los artefactos históricos de otra sesión.

Artefactos realmente invariantes del experimento, como una configuración base que MLflow no permite cambiar como param, pueden conservarse en una ubicación común si existe evidencia de igualdad/compatibilidad.

### 15.5 `session_metadata.json`

Cada sesión debe registrar como mínimo:

```text
tracking_session_id
project_run_id
mlflow_run_id
tracking_mode
started_at
ended_at o duration
runtime
Git SHA
Git ref
device
GPU/CPU
initial_global_step
final_global_step
checkpoint_input_reference
checkpoint_output_reference
```

Cuando una propiedad no aplique, usar `null`/ausencia explícita y no inventar valores.

### 15.6 Métricas globales vs métricas de sesión

MLflow puede conservar historial de métricas repetidas, pero HU008 debe evitar depender únicamente del “último valor” de una misma key para reconstruir sesiones.

Debe existir una estrategia explícita equivalente a una de estas opciones:

1. registrar métricas con `step=global_step` cuando la semántica sea temporal y útil; y/o
2. registrar métricas namespaced por sesión; y siempre
3. conservar summaries de sesión como artefactos inmutables.

Para comparación de runs pueden mantenerse métricas agregadas canónicas como:

```text
train/final_global_step
eval/mean_reward
```

pero la evidencia histórica completa debe seguir disponible en `sessions/<tracking_session_id>/...`.

### 15.7 Reglas de `new`

Una sesión `new` debe:

- crear un nuevo `mlflow_run_id`;
- usar un `project_run_id` explícito;
- crear/registrar un `tracking_session_id` único;
- registrar `tracking_mode=new`;
- registrar `initial_global_step=0` cuando efectivamente aplique;
- persistir artefactos bajo `sessions/<tracking_session_id>/`;
- devolver e imprimir `mlflow_run_id` y `tracking_session_id`.

### 15.8 Reglas de `resume`

Una sesión `resume` debe:

- recibir explícitamente el mismo `project_run_id`;
- recibir explícitamente el `mlflow_run_id` a reabrir;
- usar un `tracking_session_id` nuevo y distinto al anterior;
- verificar que el `project_run_id` del run existente coincide;
- registrar `tracking_mode=resume`;
- registrar el checkpoint de entrada;
- registrar `initial_global_step` restaurado;
- registrar el `final_global_step` alcanzado;
- guardar nuevos artefactos bajo el namespace de la nueva sesión;
- preservar sin modificación los artefactos de sesiones anteriores.

### 15.9 Detección de colisiones

Si se intenta registrar una nueva sesión con un `tracking_session_id` que ya existe para ese MLflow run, el comportamiento por defecto debe ser **fail-fast**.

No sobrescribir silenciosamente:

```text
sessions/session_001/*
```

El usuario debe escoger explícitamente otro `tracking_session_id` o un modo de corrección documentado fuera del flujo normal.

### 15.10 Trazabilidad de checkpoint entre sesiones

Para demostrar continuidad real, la sesión reanudada debe poder asociar:

```text
session_001.checkpoint_output_reference
        ↓
session_002.checkpoint_input_reference
```

Cuando HU005 produzca el checkpoint usado para resume, la metadata de sesión debe registrar esta relación sin acoplar `CheckpointManager` a MLflow.

### 15.11 API esperada en `tracking.py`

Codex puede adaptar la API existente, pero debe soportar un contrato equivalente a:

```python
metadata = tracker.start_run(
    project_run_id=project_run_id,
    tracking_mode="new" | "resume",
    mlflow_run_id=mlflow_run_id,
    tracking_session_id=tracking_session_id,
)

tracker.log_session_metadata(...)
tracker.log_training_summary(..., tracking_session_id=tracking_session_id)
tracker.log_evaluation_summary(..., tracking_session_id=tracking_session_id)
tracker.log_checkpoint_reference(..., tracking_session_id=tracking_session_id)
```

No es obligatorio usar exactamente estas firmas si el diseño resultante conserva la misma trazabilidad y separación de responsabilidades.

### 15.12 Notebook

El notebook debe imprimir de forma visible:

```text
project_run_id
mlflow_run_id
tracking_session_id
tracking_mode
initial_global_step
final_global_step
checkpoint input/output
MLFLOW_TRACKING_PASS
```

Para validar HU008 en Colab se esperan dos sesiones independientes:

```text
runtime Colab A
tracking_mode=new
tracking_session_id=session_001
↓
MLflow run ABC123
↓
cerrar runtime

runtime Colab B limpio
tracking_mode=resume
mlflow_run_id=ABC123
tracking_session_id=session_002
↓
mismo MLflow run ABC123
```

El tracking URI debe ser persistente entre ambos runtimes.

### 15.13 Nuevas tareas de implementación

Agregar al alcance de Codex:

- **T17 — Session identity:** implementar `tracking_session_id` explícito.
- **T18 — Session artifact namespace:** almacenar artefactos variables bajo `sessions/<tracking_session_id>/`.
- **T19 — Session metadata:** registrar metadata estructurada por sesión.
- **T20 — Collision protection:** rechazar reutilización accidental de un session ID existente.
- **T21 — Resume linkage:** relacionar checkpoint de salida de una sesión con checkpoint de entrada de la siguiente.
- **T22 — Session tests:** validar dos sesiones sobre un mismo `mlflow_run_id` con artefactos históricos preservados.
- **T23 — Notebook session orchestration:** exponer/configurar `tracking_session_id` y evidencia por sesión.

### 15.14 Criterios de aceptación adicionales

- **CA31 — Session identity:** cada sesión posee un `tracking_session_id` explícito y único dentro del run.
- **CA32 — Session artifact isolation:** artefactos de una sesión no sobrescriben otra.
- **CA33 — Session metadata:** cada sesión conserva runtime, SHA, modo, steps y checkpoint references propios.
- **CA34 — Same run resume:** `session_001` y `session_002` reutilizan el mismo `mlflow_run_id` durante resume real.
- **CA35 — Historical preservation:** tras session_002, `MlflowClient` permite comprobar que siguen existiendo los artefactos de session_001.
- **CA36 — Collision fail-fast:** reutilizar accidentalmente un session ID existente falla explícitamente.
- **CA37 — Checkpoint linkage:** la continuidad checkpoint output → checkpoint input queda trazable cuando aplica.
- **CA38 — Session queryability:** la implementación permite enumerar/consultar las sesiones registradas de un run mediante artefactos/metadata sin heurísticas ambiguas.

### 15.15 Autovalidaciones adicionales

- **AV24 — Two-session local run:** crear session_001 y session_002 sobre el mismo `mlflow_run_id` usando backend temporal.
- **AV25 — Artifact preservation:** después de session_002 comprobar que ambos namespaces existen y sus contenidos corresponden a cada sesión.
- **AV26 — Session collision:** intentar repetir session_001 y esperar fail-fast.
- **AV27 — Same run identity:** validar programáticamente que ambas sesiones pertenecen al mismo `project_run_id` y `mlflow_run_id`.
- **AV28 — Session metadata continuity:** validar `session_001.final_global_step == session_002.initial_global_step` cuando el resume de entrenamiento use ese checkpoint.
- **AV29 — Persistent Colab store:** ejecutar session_001 y session_002 en runtimes Colab separados usando el mismo tracking URI persistente.

### 15.16 Evidencia adicional obligatoria

La revisión final de HU008 debe incluir:

- `tracking_session_id` de cada sesión;
- mismo `project_run_id` entre sesiones de resume;
- mismo `mlflow_run_id` entre sesiones de resume;
- Git SHA por sesión;
- runtime/hardware por sesión;
- `initial_global_step` y `final_global_step` por sesión;
- checkpoint input/output por sesión;
- listado de artefactos de `session_001`;
- listado de artefactos de `session_002`;
- evidencia de que session_002 no sobrescribió session_001;
- resultado de consulta programática de ambas sesiones;
- evidencia de persistencia entre runtimes Colab separados.

### 15.17 Definition of Done adicional

HU008 **no puede marcarse `[COMPLETADA]`** hasta que además se cumpla:

- [ ] existe `tracking_session_id` explícito;
- [ ] las sesiones tienen artefactos independientes e inmutables;
- [ ] `new` crea session_001 y un nuevo MLflow run;
- [ ] `resume` crea session_002 reutilizando el mismo MLflow run;
- [ ] session_001 permanece íntegra después de session_002;
- [ ] existe metadata estructurada por sesión;
- [ ] existe protección contra colisión/overwrite de session IDs;
- [ ] checkpoints de entrada/salida quedan asociados a la sesión cuando aplican;
- [ ] tests automatizados de dos sesiones pasan;
- [ ] dos runtimes Colab separados pueden consultar el mismo tracking store persistente;
- [ ] la evidencia final permite reconstruir individualmente cada sesión.

El estado posterior a implementación local de Codex debe seguir siendo:

```text
HU008 [IMPLEMENTADA — VALIDACIONES LOCALES COMPLETADAS — VALIDACIÓN COLAB MULTISESIÓN PENDIENTE]
```

hasta ejecutar y verificar las dos sesiones reales en Colab.

### 15.18 Correccion tecnica local de resume real, params estables y evaluacion corta

La correccion tecnica del 2026-08-28 agrega evidencia local para cerrar los hallazgos pendientes de HU008 sin cambiar el alcance de HU009:

- `tracking.py` separa params globales/inmutables de metadata variable por sesion. Los params globales conservan identidad, seed, entorno, preprocessing y configuracion core DDQN compatible.
- Git SHA/ref, runtime, device, hardware, versiones y `training.total_timesteps` se conservan por sesion y no bloquean una reanudacion valida con el mismo `mlflow_run_id`.
- `training.total_timesteps` se registra como `session_target_timesteps` dentro de `session_metadata.json`.
- `config/base_config.json` representa la configuracion logica base del run; `sessions/<tracking_session_id>/effective_config.json` registra la configuracion efectiva usada por cada sesion.
- `tracking.py` deja de ocultar errores de `MlflowClient.list_artifacts(...)` al validar colisiones de `tracking_session_id`; fallos reales de backend se propagan fail-fast.
- `tracking.py` registra evidencia explicita de restauracion en `session_metadata.json`: `checkpoint_input_loaded`, `restored_checkpoint_path`, `restored_global_step`, `replay_buffer_restored`, `resume_mode`, `session_target_timesteps` y `effective_config_artifact`.
- `training_session.py` orquesta una unica sesion HU008 reanudable usando `CheckpointManager.load(..., mode="resume_full")`; no implementa una segunda ruta de restauracion.
- En modo `resume`, la sesion valida que el checkpoint externo fue cargado, que `restored_global_step` coincide con `initial_global_step`, que el Replay Buffer fue restaurado en `resume_full` y que el entrenamiento continua de `N` a `T`.
- La evaluacion corta posterior al entrenamiento se restaura usando `evaluate_agent(...)` desde `src/evaluator.py`, con `epsilon=0.0` y episodios cortos configurables. No es evaluacion formal HU011 ni comparacion contra baseline.
- El notebook soporta explicitamente:
  - `tracking_mode=new`, `tracking_session_id=session_001`, sin checkpoint de entrada, `initial_global_step=0`;
  - `tracking_mode=resume`, `tracking_session_id=session_002`, mismo `mlflow_run_id`, `ASSAULT_MLFLOW_CHECKPOINT_INPUT` explicito, restore real del checkpoint y continuacion con `final_global_step > initial_global_step`.
- Las salidas visibles del notebook incluyen `session_target_timesteps`, `checkpoint_input_loaded`, `restored_global_step`, `replay_buffer_restored`, `checkpoint_output_reference`, `evaluation_episodes`, `evaluation_mean_reward`, `evaluation_epsilon`, `MULTISESSION_CHECKPOINT_RESUME_PASS` y `MLFLOW_TRACKING_PASS`.
- Tests locales cubren resume externo real A=0->N y B=N->T, hardware/runtime distinto entre sesiones, target distinto por sesion, fallo por cambio core `ddqn.gamma`, evaluacion corta, artefacto `evaluation_summary.json`, artefacto `effective_config.json`, lineage checkpoint output/input, duplicado de sesion y propagacion de fallo real del backend MLflow.

Esta correccion no ejecuta Colab desde Codex ni inventa resultados remotos; HU008 queda lista para la validacion final de dos runtimes Colab separados con tracking URI persistente.
