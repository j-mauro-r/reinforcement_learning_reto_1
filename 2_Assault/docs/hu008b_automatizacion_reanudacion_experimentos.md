# HU008B — Automatización de arranque y reanudación de experimentos

## 1. Identificación

- **ID:** HU008B
- **Nombre:** Automatización de arranque y reanudación de experimentos
- **Estado:** PENDIENTE
- **Dependencia previa:** HU008 — MLflow y trazabilidad de experimentos `[COMPLETADA antes de implementar HU008B]`
- **Habilita:** HU009 — Entrenamiento DDQN completo
- **Entorno objetivo:** Google Colab GPU, con Google Drive como almacenamiento persistente y GitHub como fuente de verdad.
- **Objetivo principal:** eliminar la configuración manual sensible a errores que hoy se realiza en la primera celda del notebook para iniciar o reanudar sesiones de entrenamiento.

---

## 2. Contexto y problema

HU008 demostró el flujo multisesión real:

```text
session_001
new
0 -> 48
checkpoint
MLflow run
        ↓
runtime eliminado
        ↓
session_002
resume
48 -> 64
mismo MLflow run
```

La validación confirmó que la continuidad del entrenamiento depende de reconstruir correctamente varios datos persistentes:

- ubicación de Google Drive;
- `project_run_id`;
- `mlflow_run_id`;
- `tracking_session_id`;
- `tracking_mode` (`new` o `resume`);
- checkpoint de entrada;
- tracking URI de MLflow;
- directorios persistentes de checkpoints y TensorBoard;
- target global de timesteps;
- modo de restore;
- commit/ref exacto de GitHub.

Actualmente estos valores se configuran manualmente mediante variables de entorno en la primera celda del notebook. Ese procedimiento ya produjo errores reales durante HU008, entre ellos:

- uso accidental de placeholders como `<MLFLOW_RUN_ID_SESSION_001>`;
- reanudación contra un checkpoint incompatible;
- mezcla entre configuraciones de distintos flujos;
- uso de un commit distinto al cargado en memoria;
- riesgo de seleccionar un `mlflow_run_id` o checkpoint equivocado;
- necesidad de editar manualmente el notebook entre `session_001` y `session_002`.

Antes de comenzar HU009, donde los entrenamientos serán más largos y costosos, este proceso debe quedar automatizado y fail-fast.

---

## 3. Historia de usuario

> **Como** responsable de ejecutar entrenamientos DDQN en múltiples sesiones de Colab, **quiero** iniciar o reanudar un experimento indicando únicamente la intención y un identificador lógico, **para** que el sistema resuelva automáticamente el estado persistente correcto, elimine configuraciones manuales y evite continuar un experimento inconsistente.

---

## 4. Objetivo verificable

HU008B debe permitir que una nueva sesión de Colab pueda reconstruir automáticamente el contexto de ejecución de un experimento sin editar manualmente IDs, rutas o checkpoints.

El flujo objetivo debe ser equivalente a:

```python
session = prepare_training_session(
    project_run_id="assault_ddqn_exp_002",
    mode="auto",
    target_timesteps=1_000_000,
)
```

El sistema debe determinar automáticamente si corresponde:

```text
NO existe experimento persistente
        ↓
mode = new
tracking_session_id = session_001
checkpoint_input = None
MLflow run nuevo
```

O:

```text
Existe experimento persistente válido
        ↓
mode = resume
mismo project_run_id
mismo mlflow_run_id
tracking_session_id = siguiente sesión
último checkpoint compatible
restore real
```

El usuario no debe copiar manualmente un `mlflow_run_id`, construir una ruta a un checkpoint ni decidir el número de sesión.

---

## 5. Alcance funcional

### 5.1 Bootstrap de sesión reutilizable

Implementar un componente reutilizable, por ejemplo:

```text
2_Assault/src/session_bootstrap.py
```

Responsabilidad única: resolver y validar el contexto necesario para iniciar una sesión de entrenamiento.

Interfaz conceptual:

```python
context = prepare_training_session(
    base_path=...,
    project_run_id=...,
    target_timesteps=...,
    requested_mode="auto",
)
```

Debe retornar un objeto explícito, por ejemplo `TrainingSessionContext`, con al menos:

```text
project_run_id
tracking_mode
mlflow_run_id
tracking_session_id
tracking_uri
checkpoint_root
tensorboard_root
checkpoint_input
resume_mode
target_timesteps
bootstrap_ref
bootstrap_commit
```

No debe contener lógica DDQN, selección de acciones, entrenamiento ni evaluación.

---

### 5.2 Manifest persistente del experimento

Cada experimento lógico debe disponer de un manifest persistente y legible por máquina, almacenado fuera del runtime efímero de Colab.

Ubicación conceptual:

```text
<BASE>/experiments/<project_run_id>/experiment_state.json
```

Debe contener únicamente estado de orquestación necesario para reconstruir la siguiente sesión, por ejemplo:

```json
{
  "schema_version": 1,
  "project_run_id": "...",
  "mlflow_run_id": "...",
  "latest_tracking_session_id": "session_002",
  "latest_checkpoint": ".../checkpoint_step_000064.pt",
  "latest_global_step": 64,
  "resume_mode": "resume_full",
  "bootstrap_commit": "...",
  "config_fingerprint": "...",
  "updated_at": "..."
}
```

El manifest no reemplaza MLflow ni el checkpoint. Es un índice de orquestación para localizar el estado persistente correcto.

Debe actualizarse únicamente después de que una sesión termine de forma válida y sus artefactos/checkpoint hayan sido confirmados.

---

### 5.3 Resolución automática `new` / `resume`

`requested_mode="auto"` debe comportarse así:

1. Si no existe manifest válido para `project_run_id`:
   - crear `session_001`;
   - `tracking_mode=new`;
   - sin `mlflow_run_id` previo;
   - sin checkpoint de entrada.

2. Si existe manifest válido:
   - verificar que el MLflow run exista;
   - verificar que `identity.project_run_id` coincida;
   - localizar el último checkpoint declarado;
   - verificar que el checkpoint exista físicamente;
   - leer metadata suficiente para validar compatibilidad;
   - calcular siguiente `tracking_session_id`;
   - `tracking_mode=resume`;
   - reutilizar el mismo `mlflow_run_id`;
   - usar el checkpoint más reciente válido.

No seleccionar silenciosamente un "latest run" global de MLflow. La resolución siempre debe partir de un `project_run_id` explícito.

---

## 6. Compatibilidad y fail-fast

Antes de entrenar en `resume`, el bootstrap debe validar como mínimo:

- existencia física del checkpoint;
- `project_run_id` del manifest = `project_run_id` solicitado;
- MLflow run existente;
- `identity.project_run_id` en MLflow coincidente;
- checkpoint perteneciente al experimento esperado;
- `global_step` del checkpoint coherente con el manifest;
- Replay Buffer presente cuando `resume_mode=resume_full`;
- capacidad y shape del Replay Buffer compatibles;
- configuración inmutable compatible;
- environment/preprocessing compatibles;
- seed y parámetros DDQN invariantes compatibles según contrato HU008;
- target global solicitado mayor al timestep restaurado;
- `tracking_session_id` no utilizado previamente;
- commit/ref de ejecución resuelto explícitamente.

Si alguna validación falla, el entrenamiento debe abortar antes de modificar el MLflow run o sobrescribir estado persistente.

No se permiten correcciones silenciosas de configuración para hacer compatible un checkpoint.

---

## 7. Fingerprint de configuración

Para prevenir errores como el observado en HU008 (`Replay buffer capacity 128 != 1024`), generar un fingerprint determinista de los campos invariantes necesarios para un `resume_full`.

Debe incluir al menos:

```text
environment id
obs_type
frameskip
repeat_action_probability
preprocessing
network input/actions
gamma
learning_rate
epsilon policy contract
batch_size
replay_buffer capacity
train_frequency
target_update_frequency
seed
```

El fingerprint debe almacenarse en el manifest y/o metadata de sesión.

En `resume`, un fingerprint incompatible debe producir error fail-fast antes del entrenamiento.

---

## 8. Gestión automática de sesiones

El sistema debe calcular el siguiente identificador de sesión de forma determinista:

```text
session_001
session_002
session_003
...
```

Debe comprobar simultáneamente:

- manifest persistente;
- artefactos `sessions/<id>/` en MLflow.

Si el siguiente ID ya existe en cualquiera de los dos lados, abortar y pedir resolución explícita; no sobrescribir ni incrementar silenciosamente hasta encontrar uno libre.

---

## 9. Notebook

`2_Assault/assault_ddqn.ipynb` debe quedar como orquestador ligero.

La primera celda no debe contener IDs concretos de ejecuciones anteriores ni rutas construidas manualmente.

Configuración objetivo conceptual:

```python
from google.colab import drive

drive.mount("/content/drive")

PROJECT_RUN_ID = "assault_ddqn_exp_002"
TARGET_TIMESTEPS = 1_000_000
REQUESTED_MODE = "auto"
```

Después del bootstrap versionado:

```python
session_context = prepare_training_session(
    base_path=BASE,
    project_run_id=PROJECT_RUN_ID,
    target_timesteps=TARGET_TIMESTEPS,
    requested_mode=REQUESTED_MODE,
)
```

El notebook puede mostrar el contexto resuelto, pero no debe pedir al usuario copiar manualmente:

- `mlflow_run_id`;
- checkpoint path;
- `tracking_session_id`;
- `tracking_mode` cuando se use `auto`;
- rutas internas de MLflow/checkpoints/TensorBoard.

---

## 10. Commit/ref reproducible

Mantener el principio de HU002B:

- GitHub es fuente de verdad;
- Colab ejecuta un SHA conocido;
- no usar `git pull` como mecanismo de ejecución;
- bootstrap debe registrar SHA real ejecutado.

Para entrenamientos formales HU009, el commit debe poder ser fijado explícitamente.

El bootstrap automatizado no puede esconder qué versión de código se está ejecutando.

---

## 11. Persistencia y atomicidad

La actualización de `experiment_state.json` debe ser segura frente a sesiones fallidas.

Secuencia requerida:

```text
training termina
↓
checkpoint guardado
↓
checkpoint validado
↓
MLflow artifacts/metrics registrados
↓
MLflow session cerrada correctamente
↓
manifest persistente actualizado de forma atómica
```

Si el runtime muere antes del último paso, el manifest no debe apuntar a un checkpoint incompleto o a una sesión no cerrada.

Preferir escritura temporal + rename/reemplazo atómico cuando el filesystem lo permita.

---

## 12. Recuperación ante fallo parcial

Debe existir una forma explícita de inspeccionar estado si:

- existe checkpoint pero el manifest no fue actualizado;
- MLflow tiene la sesión pero falta checkpoint;
- manifest apunta a un archivo inexistente;
- una sesión quedó `FAILED` o incompleta.

El modo automático no debe "adivinar" cuál evidencia es válida.

Proveer una operación de diagnóstico, por ejemplo:

```python
inspect_experiment_state(project_run_id)
```

que reporte inconsistencias y recomiende la acción requerida sin modificar datos.

---

## 13. Observabilidad de arranque

Antes de entrenar, imprimir un resumen corto y auditable:

```text
SESSION_BOOTSTRAP_READY=True
project_run_id=...
tracking_mode=new|resume
mlflow_run_id=...
tracking_session_id=...
checkpoint_input=...
restored_expected_step=...
target_global_step=...
bootstrap_commit=...
config_fingerprint=...
```

En `new`, checkpoint y restored step deben ser `None`.

En `resume`, deben estar resueltos antes de abrir la sesión de entrenamiento.

---

## 14. Tests obligatorios

Agregar pruebas focales para, como mínimo:

1. nuevo experimento -> `new/session_001`;
2. experimento existente -> `resume/session_002`;
3. resolución del mismo `mlflow_run_id`;
4. selección del checkpoint correcto;
5. checkpoint inexistente -> fail-fast;
6. Replay Buffer incompatible -> fail-fast antes de entrenar;
7. config fingerprint incompatible -> fail-fast;
8. `target_timesteps <= restored_step` -> fail-fast;
9. sesión duplicada -> fail-fast;
10. MLflow run de otro `project_run_id` -> fail-fast;
11. manifest corrupto -> fail-fast;
12. manifest ausente -> comportamiento `new`;
13. actualización atómica del manifest;
14. fallo antes de cierre -> manifest anterior permanece intacto;
15. MLflow deshabilitado, si el contrato del proyecto lo permite, no debe romper el Trainer;
16. notebook sin IDs/run/checkpoint hardcodeados de ejecuciones previas.

Mantener además toda la suite existente en verde.

---

## 15. Criterios de aceptación

HU008B se considera completada únicamente si:

- [ ] el usuario puede iniciar un experimento nuevo indicando `project_run_id` y target sin copiar un `mlflow_run_id`;
- [ ] el usuario puede destruir el runtime y reanudar desde otro sin copiar manualmente el `mlflow_run_id`;
- [ ] el checkpoint de entrada se resuelve automáticamente desde estado persistente validado;
- [ ] `tracking_session_id` se determina automáticamente;
- [ ] `tracking_mode` se determina automáticamente con `mode=auto`;
- [ ] el mismo `project_run_id` reutiliza el mismo MLflow run;
- [ ] la reanudación hace `CheckpointManager.load`, no metadata-only resume;
- [ ] `resume_full` restaura Replay Buffer;
- [ ] configuración incompatible aborta antes del entrenamiento;
- [ ] el manifest solo cambia después de una sesión completada correctamente;
- [ ] existe diagnóstico no destructivo para estado inconsistente;
- [ ] el notebook no contiene IDs concretos de runs/checkpoints de pruebas anteriores;
- [ ] GitHub SHA/ref ejecutado sigue siendo visible y reproducible;
- [ ] TensorBoard, MLflow y checkpoints mantienen sus responsabilidades separadas;
- [ ] tests focales y suite completa pasan;
- [ ] una prueba real en dos runtimes independientes de Colab confirma el flujo automático.

---

## 16. Prueba de cierre obligatoria en Colab

Ejecutar una validación corta equivalente a HU008, pero sin editar manualmente el contexto entre runtimes.

### Runtime A

Entrada manual permitida:

```text
project_run_id = hu008b_auto_validation_001
target = 48
mode = auto
```

Esperado:

```text
tracking_mode=new
tracking_session_id=session_001
initial_global_step=0
final_global_step=48
SESSION_BOOTSTRAP_READY=True
MLFLOW_TRACKING_PASS=True
```

Eliminar completamente el runtime.

### Runtime B

Entrada manual permitida:

```text
project_run_id = hu008b_auto_validation_001
target = 64
mode = auto
```

No introducir manualmente `mlflow_run_id`, `session_002` ni checkpoint path.

Esperado:

```text
tracking_mode=resume
tracking_session_id=session_002
checkpoint_input_loaded=True
restored_global_step=48
replay_buffer_restored=True
initial_global_step=48
final_global_step=64
SESSION_BOOTSTRAP_READY=True
MULTISESSION_CHECKPOINT_RESUME_PASS=True
MLFLOW_TRACKING_PASS=True
```

---

## 17. Cierre esperado antes de HU009

Al finalizar HU008B, el flujo operativo debe quedar:

```text
Usuario define
project_run_id + target + mode=auto
        ↓
session bootstrap
        ↓
resuelve Git SHA + persistencia
        ↓
new o resume
        ↓
MLflow + checkpoint + TensorBoard
        ↓
training
        ↓
evaluation
        ↓
validación
        ↓
manifest actualizado
```

Y deben desaparecer del procedimiento normal los siguientes pasos manuales:

```text
copiar mlflow_run_id
copiar checkpoint path
editar tracking_session_id
editar tracking_mode entre sesiones
construir rutas de Drive
buscar manualmente el último checkpoint
```

La única entrada operacional necesaria para HU009 debe reducirse, idealmente, a:

```text
qué experimento ejecutar
hasta qué timestep global entrenar
qué commit/ref versionado usar, cuando se requiera fijarlo explícitamente
```

---

## 18. Fuera de alcance

HU008B no implementa:

- entrenamiento completo HU009;
- HPO HU010;
- scheduler externo de Colab;
- ejecución automática de nuevos runtimes de Colab;
- servidor MLflow remoto administrado;
- Model Registry;
- deployment;
- CI/CD de modelos.

La automatización cubre la **reconstrucción segura del contexto dentro de un runtime iniciado por el usuario**, no la creación automática de infraestructura Colab.

---

## 19. Habilita

HU008B debe quedar `[COMPLETADA]` antes de iniciar el entrenamiento costoso de HU009.

La razón es operacional: HU009 debe consumir GPU sobre un flujo de reanudación ya estable, reproducible y sin pasos manuales propensos a error.
