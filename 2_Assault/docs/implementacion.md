# Plan de implementación — Assault con DDQN

## 1. Objetivo

Definir el **orden obligatorio de implementación** para desarrollar, entrenar, evaluar y entregar el agente DDQN de `ALE/Assault-v5` siguiendo la arquitectura del proyecto, la ficha técnica y una filosofía de MLOps ligera.

Este documento funciona como mapa maestro de ejecución. Cada HU debe implementarse únicamente cuando las dependencias y validaciones de las HUs anteriores estén satisfechas, salvo que exista una decisión técnica documentada que justifique lo contrario.

Fuentes de verdad relacionadas:

- `2_Assault/docs/arquitectura.md`
- `2_Assault/docs/ficha_tecnica.md`
- `2_Assault/docs/linemientos.md`
- `enunciado_reto_1.txt`

---

## 2. Principios del plan

1. Validar primero lo barato antes de consumir GPU en entrenamientos largos.
2. Separar entorno, agente, entrenamiento, evaluación y observabilidad.
3. Mantener el notebook como orquestador y reporte, no como contenedor de toda la lógica.
4. Aplicar SOLID y DRY sin crear abstracciones innecesarias.
5. Toda ejecución relevante debe ser reproducible y trazable.
6. GitHub será la fuente de verdad; Colab será un runner de cómputo, no un entorno primario de desarrollo.
7. Los checkpoints deben permitir continuar entrenamiento entre sesiones de Google Colab.
8. TensorBoard se utilizará para observar el entrenamiento y MLflow para comparar experimentos.
9. La evaluación final debe estar separada del entrenamiento y ejecutarse sobre al menos 10 episodios independientes.
10. El baseline aleatorio y la recompensa promedio definidos en la ficha técnica serán la referencia principal de evaluación.
11. Ninguna HU se considera terminada únicamente porque el código exista: debe superar sus autovalidaciones y producir evidencia verificable.

---

## 3. Mapa de HUs en orden de implementación

```text
HU001  EDA + baseline aleatorio                      [COMPLETADA]
  ↓
HU002  Pipeline reproducible del entorno             [COMPLETADA]
  ↓
HU002B Pipeline de ejecución Local → GitHub → Colab [COMPLETADA]
  ↓
HU003  Núcleo DDQN                                   [COMPLETADA]
  ↓
HU004  Ciclo de entrenamiento                        [COMPLETADA]
  ↓
HU005  Checkpoints + reanudación + idempotencia      [COMPLETADA]
  ↓
HU006  Observabilidad con TensorBoard                [COMPLETADA]
  ↓
HU007  Smoke test end-to-end                         [COMPLETADA]
  ↓
HU008  MLflow y trazabilidad de experimentos         [COMPLETADA]
  ↓
HU008B Automatización de reanudación de experimentos [IMPLEMENTADA - VALIDACIONES LOCALES COMPLETADAS - VALIDACIÓN COLAB MULTISESIÓN AUTOMÁTICA PENDIENTE]
  ↓
HU009  Entrenamiento DDQN completo                   [COMPLETADA]
  ↓
HU009C Artefactos de entrega: modelo, gráficas, video y reporte [PENDIENTE]
  ↓
HU010  Optimización controlada de hiperparámetros
  ↓
HU011  Evaluación formal contra baseline
  ↓
HU012  Evidencias y entrega final
```

La secuencia es deliberada: primero se construye y valida el sistema y el flujo reproducible de ejecución; después se consume cómputo en entrenamientos largos y finalmente se consolidan los artefactos académicos de entrega.

---

## 4. HUs

### HU001 — Experimento 0: EDA y baseline aleatorio

**Estado:** completada.

**Propósito:** caracterizar empíricamente Assault, validar observaciones, acciones, vidas, variables de `info`, comportamiento temporal y construir el baseline aleatorio que servirá como referencia de desempeño.

**Entregables principales:**

- `2_Assault/experimento_0_assault.ipynb`
- actualización de `2_Assault/docs/ficha_tecnica.md`

**Habilita:** HU002.

---

### HU002 — Pipeline reproducible del entorno

**Estado:** [COMPLETADA].

**Propósito:** construir la única fábrica/configuración de `ALE/Assault-v5` que será utilizada por entrenamiento y evaluación.

Debe implementar:

- configuración central en `configs/ddqn_config.yaml`;
- creación reproducible del entorno;
- seeds;
- preprocessing Atari;
- grayscale;
- resize objetivo;
- frame stacking;
- manejo correcto de `frameskip` sin duplicarlo;
- distinción entre entorno de entrenamiento y evaluación;
- detección básica de hardware de Colab.

**Resultado esperado:** para una misma configuración y seed, el pipeline crea observaciones con dimensiones y tipos esperados y puede ejecutar episodios sin errores.

**Entregables principales:**

- `2_Assault/configs/ddqn_config.yaml`
- `2_Assault/src/environment.py`
- `2_Assault/src/utils.py`
- `2_Assault/tests/test_smoke.py`
- `2_Assault/assault_ddqn.ipynb`
- `2_Assault/requirements.txt`

**Evidencia de autovalidación HU002:**

- Rama validada localmente: `feature/hu002-pipeline-reproducible-entorno`, PR #3.
- Entorno virtual limpio en Windows: `.venv` creado con `python -m venv .venv`.
- Instalación desde cero validada con `.venv\Scripts\python -m pip install -r 2_Assault\requirements.txt`.
- `.venv\Scripts\python -m pytest 2_Assault\tests -q` -> `6 passed`.
- Observación procesada validada: shape `(4, 84, 84)`, dtype `uint8`.
- Espacio de acciones validado: `Discrete(7)` con `NOOP`, `FIRE`, `UP`, `RIGHT`, `LEFT`, `RIGHTFIRE`, `LEFTFIRE`.
- `frameskip` efectivo validado: tras 100 `step()`, `episode_frame_number=400`.
- Train/eval comparten fábrica, preprocessing y contrato.
- Reproducibilidad local validada con seed `42`.
- Hardware local registrado; GPU no requerida para HU002.
- `assault_ddqn.ipynb` ejecutado localmente con resultado `HU002 validations passed`.

**Cierre posterior:** las ejecuciones reales posteriores en Google Colab, incluido el flujo full de HU009, verificaron el bootstrap desde GitHub, import de `src.environment` desde `/content/reinforcement_learning_reto_1/2_Assault/src/environment.py`, observación `(4,84,84)` `uint8`, `Discrete(7)`, frameskip contractual y preflight exitoso. La evidencia histórica que aparece en HU002B sobre AV09 pendiente corresponde al estado anterior a estas ejecuciones.

**Habilita:** HU002B.

---

### HU002B — Pipeline de ejecución Local → GitHub → Colab

**Estado:** [COMPLETADA].

**Propósito:** establecer un flujo reproducible de ejecución que separe desarrollo/validación local, versionamiento en GitHub y ejecución remota en Google Colab, garantizando que Colab ejecute una rama o commit conocido del repositorio.

Debe implementar:

- detección explícita de contexto local o Colab;
- bootstrap idempotente del repositorio en Colab;
- `clone` cuando no exista copia y sincronización segura cuando ya exista;
- GitHub como fuente de verdad;
- selección explícita de rama o commit;
- registro del commit SHA ejecutado;
- instalación reproducible desde `2_Assault/requirements.txt`;
- configuración correcta del working directory e imports;
- protección frente a imports obsoletos al cambiar de commit;
- compatibilidad con notebook abierto desde VS Code y kernel remoto Colab;
- ejecución de las validaciones HU002 desde runtime limpio de Colab.

**Restricción:** HU002B no implementa entrenamiento DDQN ni infraestructura MLflow remota. MLflow remoto y trazabilidad avanzada permanecen en HU008.

**Resultado esperado:** el ciclo `VS/local → tests → commit/push → GitHub → Colab → commit verificado → notebook reproducible` funciona sin modificaciones manuales al código.

**Entregable de definición:** `2_Assault/docs/hu002b_pipeline_ejecucion_local_github_colab.md`.

**Habilita:** HU003.

**Evidencia de implementación HU002B (2026-08-27, rama `feature/hu002b-pipeline-local-github-colab`):**

- Archivos implementados/modificados:
  - `2_Assault/assault_ddqn.ipynb`
  - `2_Assault/src/execution_bootstrap.py`
  - `2_Assault/tests/test_execution_bootstrap.py`
  - `2_Assault/docs/implementacion.md`
- Bootstrap agregado al inicio del notebook:
  - detecta ejecución local vs Google Colab;
  - en local usa el checkout Git existente;
  - en Colab usa `/content/reinforcement_learning_reto_1`;
  - si la copia Colab no existe, ejecuta `git clone`;
  - si existe, ejecuta `git fetch --prune origin`;
  - resuelve rama/ref o commit explícito;
  - usa checkout detached al SHA resuelto;
  - no ejecuta `git pull`, merge, commit ni push desde Colab;
  - instala dependencias desde `2_Assault/requirements.txt` de la copia seleccionada;
  - configura `PROJECT_ROOT`, `ASSAULT_DIR` e imports después del bootstrap;
  - verifica el origen de `src.environment`;
  - bloquea cambios de commit cuando ya existen imports `src.*` cargados.
- Resultado del bootstrap local:
  - runtime detectado: `local`;
  - repo usado: `D:\Users\Usuario\Documents\ENTROPY_LAB\maestria_ia\reinforcement_learning_reto_1`;
  - `ASSAULT_DIR`: `D:\Users\Usuario\Documents\ENTROPY_LAB\maestria_ia\reinforcement_learning_reto_1\2_Assault`;
  - ref solicitada: `feature/hu002b-pipeline-local-github-colab`;
  - commit pin validado localmente: `8a17b7f9b5e5ae12dc6bfacfc36e9d2bdef926d2`;
  - idempotencia local: dos ejecuciones resolvieron el mismo SHA;
  - origen real de import: `D:\Users\Usuario\Documents\ENTROPY_LAB\maestria_ia\reinforcement_learning_reto_1\2_Assault\src\environment.py`.
- Validaciones locales ejecutadas:
  - `python -m pytest 2_Assault/tests -q` -> `10 passed in 11.46s`;
  - ejecución local de celdas de código de `2_Assault/assault_ddqn.ipynb` con `ASSAULT_INSTALL_DEPENDENCIES=0` -> `NOTEBOOK_CODE_CELLS_OK`;
  - observación procesada: `(4, 84, 84)`;
  - dtype: `uint8`;
  - action space: `Discrete(7)`;
  - acciones: `NOOP`, `FIRE`, `UP`, `RIGHT`, `LEFT`, `RIGHTFIRE`, `LEFTFIRE`;
  - `frameskip` efectivo validado: `4`;
  - interacción corta HU002: 100 `step()` sin errores.

**Evidencia histórica de bloqueo inicial de Colab:**

- En 2026-08-27 Codex no tenía una sesión Colab autenticada desde Windows/WSL y no podía completar AV09 remotamente por sí mismo.
- Ese bloqueo era de acceso remoto de Codex, no del pipeline del proyecto.
- Las ejecuciones posteriores realizadas directamente por el usuario en Google Colab resolvieron esta validación operacional.

**Cierre posterior en Google Colab:**

- runtime real detectado: `Google Colab`;
- repositorio ejecutado bajo `/content/reinforcement_learning_reto_1`;
- bootstrap con GitHub `main` y SHA resuelto explícitamente;
- requirements instalados desde `2_Assault/requirements.txt`;
- `src.environment` cargado desde la copia versionada en `/content`;
- pipeline usado exitosamente por las validaciones HU007–HU009 y por la corrida full de `250000` timesteps.

---

### HU003 — Núcleo DDQN

**Estado:** completada.

**Propósito:** implementar los componentes propios del algoritmo seleccionado sin incorporar aún el ciclo completo de entrenamiento.

Debe implementar:

- CNN Q-Network;
- Online Network;
- Target Network;
- Replay Buffer uniforme;
- política epsilon-greedy;
- cálculo del target DDQN;
- actualización de la Online Network;
- sincronización de la Target Network;
- optimizer;
- interfaces básicas de `save` y `load` del agente.

**Restricción:** no implementar Prioritized Experience Replay, ya que el algoritmo seleccionado es DDQN con Experience Replay uniforme.

**Resultado esperado:** los componentes reciben batches sintéticos/reales con las dimensiones esperadas, producen Q-values para las 7 acciones y ejecutan al menos un paso de optimización sin errores.

**Habilita:** HU004.

**Evidencia de implementación HU003 (2026-08-27, rama `feature/hu003-nucleo-ddqn`):**

- Archivos creados/modificados:
  - `2_Assault/src/network.py`
  - `2_Assault/src/replay_buffer.py`
  - `2_Assault/src/agent.py`
  - `2_Assault/tests/test_network.py`
  - `2_Assault/tests/test_replay_buffer.py`
  - `2_Assault/tests/test_agent.py`
  - `2_Assault/configs/ddqn_config.yaml`
  - `2_Assault/requirements.txt`
  - `2_Assault/docs/implementacion.md`
- Configuración agregada:
  - `network.input_channels=4`
  - `network.num_actions=7`
  - `agent.gamma=0.99`
  - `agent.learning_rate=0.0001`
  - `agent.epsilon_start=1.0`
  - `agent.epsilon_final=0.01`
  - `replay_buffer.capacity=100000`
  - `replay_buffer.batch_size=32`
- Dependencia agregada:
  - `torch>=2.0`
- Arquitectura QNetwork:
  - CNN Atari/DQN simple con `Conv2d(4,32,8,stride=4)`, `Conv2d(32,64,4,stride=2)`, `Conv2d(64,64,3,stride=1)`, `Linear(3136,512)` y salida `Linear(512,7)`.
  - Entrada `(batch,4,84,84)`; acepta `uint8`, convierte a `float32` y normaliza a `[0,1]`.
  - Salida validada: `(batch,7)` con valores finitos.
- Replay Buffer:
  - capacidad fija;
  - escritura circular/FIFO;
  - estados y siguientes estados almacenados como `uint8`;
  - muestreo uniforme sin reemplazo;
  - sin PER ni prioridades.
- DDQNAgent:
  - Online Network y Target Network independientes;
  - Target inicializada con los mismos pesos de Online;
  - Target sin gradientes;
  - `select_action` epsilon-greedy para `epsilon=0` y `epsilon=1`;
  - `compute_ddqn_targets` implementa selección con Online y evaluación con Target;
  - `update` usa Adam y `SmoothL1Loss`;
  - `sync_target_network` explícito;
  - `save/load` básico de Online, Target, optimizer y metadatos de red/agente.
- Validaciones ejecutadas:
  - `python -m pytest 2_Assault/tests -q` -> `21 passed, 1 skipped in 9.10s`.
  - `python -m compileall -q 2_Assault/src` -> PASS.
- Ambiente local:
  - PyTorch `2.4.1+cpu`.
  - CUDA local: `False`; test GPU omitido correctamente con `skip`.
- Smoke HU003:
  - `qnetwork_shape=(2, 7)`;
  - `qnetwork_finite=True`;
  - `update_loss=2.8990249633789062`;
  - `loss_finite=True`;
  - `target_stable_after_update=True`;
  - `target_equals_online_after_sync=True`;
  - `save_load_outputs_match=True`.
- Test específico DDQN:
  - Online seleccionó la acción `1`.
  - Target evaluó esa acción con Q-value `10.0`.
  - Target DDQN calculado: `10.899999618530273`.
  - El target DQN clásico con `max(Target(next_state))` habría sido `50.5`; el test detectaría esa regresión.
  - Transición terminal validada sin bootstrap.
- Smoke con Assault real:
  - observación real desde `create_assault_env`: `(4, 84, 84)`;
  - dtype: `uint8`;
  - Q-values reales: `(1, 7)`;
  - acción seleccionada válida: `6`.
- Desviaciones respecto del DWP:
  - Ninguna desviación funcional.
  - Validación GPU queda para Colab/futuro runtime con CUDA disponible, tal como permite AV11.
- Scope excluido confirmado:
  - No se implementó trainer completo, ciclo de entrenamiento, TensorBoard, MLflow, callbacks, checkpoints avanzados, resume, evaluación formal, videos, PER, Dueling DQN, Rainbow, n-step ni Noisy Nets.

---

### HU004 — Ciclo de entrenamiento

**Estado:** completada.

**Propósito:** integrar entorno, agente y Replay Buffer en un ciclo de entrenamiento controlado por timesteps.

Debe implementar:

- `reset` y `step` del entorno;
- selección epsilon-greedy;
- almacenamiento de transiciones;
- `learning_starts`;
- muestreo por batches;
- actualización DDQN;
- decay de epsilon;
- sincronización periódica de Target Network;
- registro de métricas básicas en memoria/log;
- control por `global_step`.

**Resultado esperado:** el sistema puede entrenar durante un número pequeño de timesteps y modificar los pesos de la Online Network de forma verificable.

**Evidencia de implementación HU004 (2026-08-27, rama `feature/hu004-ciclo-entrenamiento`):**

- Commit de implementación registrado:
  - `910d7ea2df9f88b45dbe386e5bff54856ff1ae73` (`Implement HU004 training preflight and loop`).
- Archivos creados/modificados:
  - `2_Assault/src/preflight.py`
  - `2_Assault/src/trainer.py`
  - `2_Assault/tests/test_preflight.py`
  - `2_Assault/tests/test_trainer.py`
  - `2_Assault/configs/ddqn_config.yaml`
  - `2_Assault/assault_ddqn.ipynb`
  - `2_Assault/docs/implementacion.md`
- Configuración HU004 agregada:
  - `training.total_timesteps=48`
  - `training.learning_starts=32`
  - `training.train_frequency=4`
  - `training.target_update_frequency=16`
  - `training.epsilon_decay_steps=48`
  - `replay_buffer.capacity=1024`
  - `replay_buffer.batch_size=32`
- Decisión de memoria:
  - `replay_buffer.capacity` se ajustó de `100000` a `1024` para HU004 porque el buffer visual preasignado con `uint8` reserva memoria al inicializarse y esta HU valida una corrida corta, no el entrenamiento largo de HU009.
  - El dimensionamiento final de Replay Buffer para entrenamiento largo queda pendiente de HU009/validación de RAM en Colab.
- Preflight:
  - archivo: `2_Assault/src/preflight.py`;
  - interfaz: `run_preflight_checks(config, device=...)`;
  - resultado estructurado: `PreflightReport(passed, runtime, device, checks, errors, details)`;
  - checks obligatorios implementados: `Device`, `Environment`, `Observation`, `QNetwork`, `ReplayBuffer`, `DDQN update`, `Loss finite`, `Target stable`, `Target sync`, `Save/load`;
  - `READY_FOR_TRAINING=True` en validación local CPU;
  - save/load usa archivo temporal y confirma `temporary_file_cleaned=True`.
- Trainer:
  - archivo: `2_Assault/src/trainer.py`;
  - interfaz principal: `Trainer(env, agent, replay_buffer, config).train()`;
  - resumen estructurado: `TrainingSummary`;
  - control por `global_step`;
  - una transición almacenada por cada `env.step(action)`;
  - `learning_starts` respetado antes de updates;
  - updates solo si `global_step >= learning_starts`, `len(replay_buffer) >= batch_size` y `global_step % train_frequency == 0`;
  - epsilon decay lineal determinista con `compute_epsilon`;
  - Target sync solo en múltiplos de `target_update_frequency`;
  - `terminated or truncated` reinicia episodio;
  - solo `terminated` se guarda como `done` para bootstrap DDQN.
- Notebook:
  - `2_Assault/assault_ddqn.ipynb` queda como orquestador;
  - secuencia: bootstrap HU002B, config/runtime, validación HU002, Preflight HU004, gate `READY_FOR_TRAINING`, short training HU004, resumen de métricas;
  - no duplica CNN, Replay Buffer, lógica DDQN ni trainer.
- Validaciones ejecutadas:
  - `python -m pytest 2_Assault/tests -q` -> `33 passed, 2 skipped in 11.50s`;
  - `python -m compileall -q 2_Assault/src` -> PASS;
  - imports HU004 -> `HU004 imports OK`;
  - ejecución local de celdas de código del notebook con `ASSAULT_INSTALL_DEPENDENCIES=0` y `ASSAULT_BOOTSTRAP_REF=feature/hu004-ciclo-entrenamiento` -> `NOTEBOOK_CODE_CELLS_OK`.
- Ambiente local:
  - PyTorch `2.4.1+cpu`;
  - CUDA local: `False`;
  - dispositivo usado: `cpu`;
  - tests GPU omitidos correctamente por ausencia de CUDA.
- Smoke integrado HU004 con Assault real:
  - flujo: `create_assault_env -> Preflight -> DDQNAgent -> ReplayBuffer -> Trainer -> short run`;
  - observación real: `(4, 84, 84)`;
  - dtype real: `uint8`;
  - `preflight_passed=True`;
  - `READY_FOR_TRAINING=True`;
  - checks Preflight: todos `True`;
  - `global_step=48`;
  - `transitions_stored=48`;
  - `updates_count=5`;
  - `first_update_step=32`;
  - `last_loss=0.0008681566687300801`;
  - `mean_loss=0.512887254380621`;
  - `loss_finite=True`;
  - `epsilon_initial=1.0`;
  - `epsilon_final=0.01`;
  - `target_sync_steps=[16, 32, 48]`;
  - `online_weights_changed=True`;
  - `episodes_completed=0` durante la corrida corta real de 48 timesteps.
- Evidencia `terminated/truncated`:
  - tests con entorno controlado validan que `terminated=True` reinicia y guarda `done=True`;
  - `truncated=True` reinicia episodio pero guarda `done=False` para el target DDQN;
  - `global_step` continúa tras `reset()` y se detiene exactamente en `total_timesteps`.
- Desviaciones respecto del DWP:
  - No se ejecutó Colab/GPU desde Codex porque HU002B sigue sin canal remoto autenticado disponible; queda como validación futura/manual.
  - No se registró evidencia remota ni se declara GPU como aprobada.
- Scope excluido confirmado:
  - No se implementó entrenamiento largo, checkpoints persistentes, resume, Replay Buffer persistente, Google Drive, TensorBoard, MLflow, callbacks avanzados, mejor modelo, evaluación formal, video, optimización de hiperparámetros, PER, Dueling, Rainbow, Noisy Nets, n-step, reward clipping, GitHub Actions ni automatización Codex -> Colab.

**Habilita:** HU005.

---

### HU005 — Checkpoints, reanudación e idempotencia

**Estado:** completada.

**Propósito:** asegurar continuidad entre sesiones de Google Colab y evitar pérdida de progreso.

El checkpoint debe guardar como mínimo:

- Online Network;
- Target Network;
- optimizer;
- timestep global;
- estado/valor de epsilon o información suficiente para reconstruirlo;
- configuración del experimento;
- métricas mínimas de continuidad;
- Replay Buffer cuando se utilice modo de resume completo.

Debe soportar explícitamente:

1. entrenamiento nuevo;
2. resume completo;
3. resume liviano cuando el Replay Buffer no pueda persistirse.

**Resultado esperado:** entrenar → guardar → reiniciar proceso → cargar → continuar desde el timestep correcto sin reiniciar silenciosamente el entrenamiento.

**Evidencia de implementación HU005 (2026-08-27, rama `feature/hu005-checkpoints-resume`):**

- Commit de implementación registrado:
  - `be461f1a0f68880ae5a6335ae36414cb85f93050` (`Implement HU005 checkpoint resume support`).
- Archivos creados/modificados:
  - `2_Assault/src/checkpointing.py`
  - `2_Assault/src/replay_buffer.py`
  - `2_Assault/src/trainer.py`
  - `2_Assault/tests/test_checkpointing.py`
  - `2_Assault/configs/ddqn_config.yaml`
  - `2_Assault/assault_ddqn.ipynb`
  - `2_Assault/docs/implementacion.md`
- Configuración checkpointing agregada:
  - `checkpointing.enabled=true`
  - `checkpointing.interval_steps=24`
  - `checkpointing.directory=checkpoints`
  - `checkpointing.mode=new`
  - `checkpointing.run_id=assault_ddqn_exp_001`
  - `checkpointing.resume_checkpoint=null`
  - `checkpointing.save_replay_buffer=true`
- Arquitectura `checkpointing.py`:
  - `CheckpointManager(directory, run_id, repo_path=".")`;
  - rutas por `run_id`: `checkpoints/<run_id>/checkpoint_step_000048.pt`;
  - `save(...)` con guardado atómico mediante archivo temporal y `os.replace`;
  - `load(..., mode="resume_full" | "resume_light")` con checkpoint explícito;
  - `ensure_new_run()` bloquea `new` si ya existen checkpoints para el `run_id`;
  - `CheckpointMetadata` y `CheckpointState` como resultados estructurados;
  - `reconstruct_epsilon(global_step, config)` reconstruye epsilon desde `global_step + config`.
- Contenido de checkpoint validado:
  - `schema_version=1`;
  - `run_id`;
  - `created_at`;
  - `checkpoint_step`;
  - `git_commit`;
  - `config`;
  - `online_network`;
  - `target_network`;
  - `optimizer`;
  - `global_step`;
  - `epsilon_state`;
  - `training_metrics`;
  - `resume_mode_capabilities`;
  - `replay_buffer_state` cuando `save_replay_buffer=True`.
- Replay Buffer serializable:
  - `ReplayBuffer.state_dict()`;
  - `ReplayBuffer.load_state_dict(...)`;
  - conserva `capacity`, `state_shape`, `size`, `position`, arrays válidos y estado RNG;
  - serializa solo posiciones válidas, no slots vacíos.
- Trainer extendido:
  - soporta `initial_global_step`;
  - soporta `initial_metrics`;
  - `training.total_timesteps` sigue siendo objetivo global;
  - checkpoint periódico opcional por `checkpoint_interval_steps`;
  - resume desde `N` hasta `T` termina en `T`, no en `N+T`.
- Modos soportados:
  - `new`: agente nuevo, optimizer nuevo, Replay Buffer vacío, `global_step=0`, `run_id` explícito y no-overwrite por defecto;
  - `resume_full`: restaura Online, Target, optimizer, `global_step`, config, métricas y Replay Buffer;
  - `resume_light`: restaura Online, Target, optimizer, `global_step`, config y métricas; Replay Buffer inicia vacío.
- Validaciones ejecutadas:
  - `python -m pytest 2_Assault/tests -q` -> `48 passed, 2 skipped in 17.14s`;
  - `python -m compileall -q 2_Assault/src` -> PASS;
  - imports HU005 -> `HU005 imports OK`;
  - notebook local con `ASSAULT_INSTALL_DEPENDENCIES=0`, `ASSAULT_BOOTSTRAP_REF=feature/hu005-checkpoints-resume`, `ASSAULT_CHECKPOINT_DIR=<temp>` -> `NOTEBOOK_CODE_CELLS_OK`.
- Notebook:
  - `2_Assault/assault_ddqn.ipynb` queda como orquestador;
  - expone `RUN_MODE = new | resume_full | resume_light`;
  - exige `CHECKPOINT_PATH` explícito para `resume_full` y `resume_light`;
  - imprime modo, `run_id`, checkpoint elegido, `global_step` inicial, epsilon reconstruido y si Replay Buffer fue restaurado;
  - guarda checkpoints periódicos y reutiliza explícitamente el checkpoint final si ya fue creado por el intervalo.
- Smoke real Assault HU005:
  - flujo validado: `Preflight PASS -> train N -> save full/light -> recrear objetos -> load -> resume`;
  - PyTorch `2.4.1+cpu`;
  - CUDA local: `False`;
  - dispositivo usado: `cpu`;
  - `READY_FOR_TRAINING=True`;
  - `new_global_step=8`;
  - `new_updates=3`;
  - `new_epsilon=0.01`;
  - `buffer_size_at_save=8`;
  - checkpoint full size: `27476290` bytes;
  - checkpoint light size: `27022210` bytes;
  - `resume_full_loaded_step=8`;
  - `resume_full_epsilon=0.01`;
  - `resume_full_buffer_size_after_load=8`;
  - `resume_full_final_global_step=12`;
  - `resume_full_updates=5`;
  - `resume_full_last_loss=0.0002125417668139562`;
  - `resume_light_loaded_step=8`;
  - `resume_light_epsilon=0.01`;
  - `resume_light_buffer_size_after_load=0`;
  - `resume_light_final_global_step=11`;
  - `resume_light_new_updates=0`;
  - `resume_light_buffer_size_after_refill=3`.
- Idempotencia validada:
  - guardar dos veces el mismo step con `overwrite=False` falla con `FileExistsError`;
  - `overwrite=True` permite sobrescritura explícita;
  - `new` con `run_id` existente y checkpoint previo falla con `FileExistsError`;
  - `resume_full/resume_light` requieren checkpoint explícito;
  - no se implementa selección automática de `latest`.
- Compatibilidad validada:
  - carga en CPU con `map_location`;
  - error explícito ante `schema_version` incompatible;
  - error explícito ante config crítica incompatible (`environment.id`, preprocessing, `network.input_channels`, `network.num_actions`);
  - `resume_full` falla si el checkpoint no fue guardado con Replay Buffer.
- Desviaciones respecto del DWP:
  - No se ejecutó Colab/GPU desde Codex porque HU002B sigue sin canal remoto autenticado disponible; queda como validación futura/manual.
  - No se automatizó montaje OAuth/Google Drive; el notebook permite configurar `ASSAULT_CHECKPOINT_DIR` para una ruta persistente.
  - No se serializa estado interno de ALE; al reanudar se inicia un episodio nuevo conservando agente, optimizer, `global_step`, epsilon, métricas y buffer según modo.
- Scope excluido confirmado:
  - No se implementó TensorBoard, MLflow, entrenamiento largo, evaluación formal, video, best-model selection, hyperparameter optimization, PER, Dueling, Rainbow, Noisy Nets, n-step, GitHub Actions ni automatización Codex -> Colab.

**Habilita:** HU006.

---

### HU006 — Observabilidad con TensorBoard

**Estado:** completada.

**Propósito:** hacer observable el proceso de aprendizaje durante entrenamiento.

Registrar como mínimo:

- recompensa por episodio;
- recompensa media móvil;
- longitud del episodio;
- loss;
- epsilon;
- Q-value medio o equivalente útil;
- timestep global;
- learning rate si cambia.

**Resultado esperado:** una corrida corta genera logs válidos que TensorBoard puede visualizar y que permiten detectar si el agente está aprendiendo, divergiendo o dejó de explorar.

**Evidencia de implementación HU006 (2026-08-28, rama `feature/hu006-tensorboard`):**

- Archivos modificados:
  - `2_Assault/requirements.txt`: agrega `tensorboard>=2.14`.
  - `2_Assault/configs/ddqn_config.yaml`: agrega `tensorboard.enabled`, `directory`, `log_frequency_steps`, `reward_window_episodes` y `flush_frequency_steps`.
  - `2_Assault/src/callbacks.py`: crea `TensorBoardLogger` y `load_tensorboard_scalars`.
  - `2_Assault/src/agent.py`: `DDQNAgent.update()` conserva `loss` y añade `q_mean` y `learning_rate` reales.
  - `2_Assault/src/trainer.py`: integra `metrics_logger` opcional sin cambiar el flujo DDQN cuando es `None`.
  - `2_Assault/tests/test_tensorboard.py`: valida event files, tags, pasos, valores finitos, episodios, resume, runs separados, modo deshabilitado y smoke Assault.
  - `2_Assault/assault_ddqn.ipynb`: orquesta HU006 y muestra comandos TensorBoard local/Colab.
- Arquitectura:
  - `Trainer` decide cuándo existen eventos reales;
  - `TensorBoardLogger` encapsula `SummaryWriter`;
  - `callbacks.py` no selecciona acciones, no calcula targets, no toca optimizer, no crea entornos y no implementa MLflow.
- Configuración centralizada aplicada:
  - `enabled: true`;
  - `directory: logs/tensorboard`;
  - `log_frequency_steps: 4`;
  - `reward_window_episodes: 10`;
  - `flush_frequency_steps: 24`.
- Estructura de logs:
  - `2_Assault/logs/tensorboard/<run_id>/events.out.tfevents.*`;
  - el smoke local usó `run_id=hu006_validation_local`;
  - los logs quedan fuera de Git por `2_Assault/logs/`.
- Tags implementados y validados:
  - `train/epsilon`;
  - `train/loss`;
  - `train/q_mean`;
  - `train/learning_rate`;
  - `episode/reward`;
  - `episode/reward_mean`;
  - `episode/length`.
- Smoke real Assault:
  - `Preflight PASS`;
  - dispositivo: `cpu`;
  - Python `3.8.10`;
  - Torch `2.4.1`;
  - ALE `0.10.1`;
  - `global_step=48`;
  - `updates_count=5`;
  - `update_steps=[32, 36, 40, 44, 48]`;
  - `last_loss=0.0008681566687300801`;
  - `last_q_mean=0.02830067276954651`;
  - `last_learning_rate=0.0001`;
  - event files: `1`;
  - tags reales leídos: `train/epsilon`, `train/loss`, `train/q_mean`, `train/learning_rate`;
  - conteos: `train/epsilon=12`, `train/loss=5`, `train/q_mean=5`, `train/learning_rate=5`.
- Evidencia de episodios:
  - entorno controlado valida `episode/reward` en pasos `3` y `6`;
  - `episode/reward_mean` usa ventana móvil configurable;
  - `episode/length` registra decisiones del agente;
  - la corrida real corta de Assault no terminó episodio, por eso no inventa métricas `episode/*`.
- Evidencia de epsilon:
  - con frecuencia `4`, se registran pasos `4` y `8` en test controlado;
  - el valor corresponde al epsilon usado para la acción antes de incrementar el timestep, registrado sobre el `global_step` posterior de esa transición.
- Evidencia de updates:
  - `train/loss`, `train/q_mean` y `train/learning_rate` solo aparecen en pasos con update real;
  - no se registran ceros falsos en timesteps sin actualización.
- Evidencia de resume:
  - al reusar `run_id=resume_run`, los eventos continúan con pasos `[4, 8, 12]`;
  - el segundo tramo arranca con `initial_global_step=8` y no reinicia silenciosamente a cero;
  - el mismo directorio de run puede contener múltiples event files sin truncarse.
- AV13 - Resume real mediante checkpoint:
  - test agregado: `test_tensorboard_resume_from_checkpoint_load_preserves_logs_and_continues_after_restored_step`;
  - `run_id=checkpoint_resume_run`;
  - segmento A entrena hasta `checkpoint @ N=8`;
  - checkpoint guardado con `CheckpointManager.save(..., save_replay_buffer=True)`;
  - se cierran logger, entorno y objetos del segmento A;
  - segmento B crea nuevos `environment`, `DDQNAgent`, `ReplayBuffer`, `TensorBoardLogger` y `Trainer`;
  - resume ejecutado con `CheckpointManager.load(..., mode="resume_full")`;
  - `restored global_step=8`;
  - `Trainer(..., initial_global_step=restored_state.global_step, initial_metrics=restored_state.training_metrics)`;
  - `final global_step=12`;
  - `train/epsilon` antes del resume: `[4, 8]`;
  - `train/epsilon` despues del resume en la serie completa: `[4, 8, 12]`;
  - nuevos `train/epsilon` steps `> N`: `[12]`;
  - `train/loss` antes del resume: `[4, 6, 8]`;
  - `train/loss` despues del resume en la serie completa: `[4, 6, 8, 10, 12]`;
  - nuevos `train/loss` steps `> N`: `[10, 12]`;
  - no se valida pasando manualmente `initial_global_step=N`; el valor proviene del checkpoint cargado.
- AV14 - Preservacion de logs:
  - mismo directorio usado: `<tensorboard_root>/checkpoint_resume_run/`;
  - event files antes del resume: `1`;
  - event files despues del resume: `2`;
  - archivos previos preservados: `True`;
  - eventos `train/loss` antes/despues: `3 -> 5`;
  - EventAccumulator lee una serie temporal completa sin eliminar ni truncar logs previos.
- Evidencia de aislamiento:
  - `run_a` y `run_b` escriben en directorios separados;
  - los scalars no se mezclan entre `run_id`.
- Evidencia de modo deshabilitado:
  - con `tensorboard.enabled=false`, `Trainer` completa la corrida;
  - no se crea directorio de run ni event files.
- Autovalidaciones ejecutadas:
  - AV03: `python -m compileall -q 2_Assault/src` -> PASS sin salida ni errores;
  - `python -m pytest 2_Assault/tests/test_tensorboard.py -q` -> `11 passed`;
  - `python -m pytest 2_Assault/tests/test_tensorboard.py 2_Assault/tests/test_checkpointing.py -q` -> `26 passed`;
  - AV16 notebook local: celdas automatizables de `2_Assault/assault_ddqn.ipynb` ejecutadas con `ASSAULT_INSTALL_DEPENDENCIES=0`, `ASSAULT_BOOTSTRAP_REF=feature/hu006-tensorboard`, `ASSAULT_CHECKPOINT_DIR=<temp>`, `ASSAULT_TENSORBOARD_DIR=<temp>`, `ASSAULT_RUN_ID=<run temporal>` -> `NOTEBOOK_CODE_CELLS_OK`; celdas ejecutadas: `11`; celdas omitidas: `[]`;
  - `python -m pytest 2_Assault/tests -q` -> `59 passed, 2 skipped`.
- Limitaciones históricas:
  - en el momento de implementación no se ejecutó runtime remoto de Colab desde Codex por la restricción ya diagnosticada;
  - no se implementó MLflow, evaluación formal, video, entrenamiento largo ni selección de mejor modelo en esta HU.

**Habilita:** HU007.

---

### HU007 — Smoke test end-to-end

**Estado:** completada.

**Propósito:** validar todo el pipeline antes de gastar recursos en un entrenamiento largo.

Debe ejecutar una corrida corta con GPU y verificar conjuntamente:

- creación del entorno;
- preprocessing;
- inferencia de la red;
- Replay Buffer;
- aprendizaje;
- actualización de Target Network;
- TensorBoard;
- checkpoint;
- restauración del checkpoint;
- continuidad del entrenamiento;
- evaluación corta del modelo resultante.

**Resultado esperado:** el pipeline completo funciona sin errores funcionales ni problemas evidentes de dimensiones, dispositivo, memoria o persistencia.

**Gate:** no iniciar HU009 si HU007 no está aprobada.

**Evidencia:** HU007 fue cerrada posteriormente con ejecución real Colab/GPU `E2E_SMOKE_PASS=True`, checkpoint/restauración y continuidad observada. La evidencia local histórica anterior permanece en commits/notebook.

**Habilita:** HU008.

---

### HU008 — MLflow y trazabilidad de experimentos

**Estado:** [COMPLETADA].

**Propósito:** registrar de forma comparable las ejecuciones que sí importan para tomar decisiones y extender HU002B con tracking persistente cuando corresponda.

Cada run relevante debe registrar como mínimo:

- algoritmo;
- hiperparámetros;
- configuración del entorno;
- preprocessing;
- seed;
- versiones principales;
- hardware;
- commit Git;
- timestep inicial y final;
- tiempo de entrenamiento;
- métricas de evaluación;
- referencia al checkpoint/modelo.

**Resultado esperado:** dos corridas pueden compararse en MLflow y es posible identificar exactamente qué código y configuración produjo cada resultado.

**Diseño implementado:**

- `2_Assault/src/tracking.py` encapsula MLflow sin introducir dependencias directas desde Trainer/Evaluator/Agent.
- TensorBoard conserva curvas densas; MLflow registra identidad, params estables, métricas agregadas, runtime y referencias a artefactos.
- Identidad separada mediante `project_run_id`, `mlflow_run_id` y `tracking_session_id`.
- Artefactos variables por sesión bajo `sessions/<tracking_session_id>/`.
- `training_session.py` permite una sesión nueva o reanudada con checkpoint externo.

**Evidencia de cierre Colab multi-sesión:**

- `project_run_id=assault_ddqn_hu008_colab_002`;
- mismo `mlflow_run_id=e641bd92682d4fa9a800013cb0df989c` entre runtimes;
- `session_001`: `new`, `0 -> 48`, checkpoint persistente;
- runtime destruido y recreado;
- `session_002`: `resume`, checkpoint step 48 cargado, `restored_global_step=48`, `replay_buffer_restored=True`, continuación `48 -> 64`;
- `MULTISESSION_CHECKPOINT_RESUME_PASS=True`;
- `MLFLOW_TRACKING_PASS=True`;
- artefactos por sesión presentes y mismo run lógico conservado.

**Habilita:** HU008B/HU009.

---

### HU008B - Automatización de arranque y reanudación de experimentos

**Estado:** IMPLEMENTADA - VALIDACIONES LOCALES COMPLETADAS - VALIDACIÓN COLAB MULTISESIÓN AUTOMÁTICA PENDIENTE.

**Propósito:** automatizar el arranque y la reanudación de experimentos DDQN multisesión para que el usuario indique solo `project_run_id`, `target_timesteps` y `requested_mode=auto`, sin copiar manualmente `mlflow_run_id`, `tracking_session_id`, checkpoint de entrada ni rutas internas.

**Diseño final implementado:**

- `2_Assault/src/session_bootstrap.py` resuelve `new` o `resume` desde `<BASE>/experiments/<project_run_id>/experiment_state.json`.
- Si no existe manifest, `auto` produce `tracking_mode=new`, `tracking_session_id=session_001`, `mlflow_run_id=None` y `checkpoint_input=None`.
- Si existe manifest válido, `auto` produce `tracking_mode=resume`, reutiliza el mismo `mlflow_run_id`, calcula el siguiente `session_NNN` y resuelve el checkpoint persistente del manifest.
- El manifest contiene estado de orquestación y usa actualización atómica posterior al cierre exitoso de la sesión.
- El fingerprint determinístico protege invariantes de entorno, preprocessing, network, gamma, learning rate, epsilon, batch size, Replay Buffer, frecuencias de entrenamiento/target y seed.

**Validaciones locales:**

- `python -m pytest 2_Assault/tests/test_session_bootstrap.py -q` -> `17 passed, 1 warning`.
- Cubren new/resume, mismo MLflow run, checkpoint, Replay Buffer, fingerprint, target, sesión duplicada, identidad, manifest corrupto/ausente, atomicidad y notebook sin IDs históricos.

**Pendiente específico:**

- demostrar en dos runtimes independientes que `requested_mode=auto` descubre por sí solo la sesión previa y reanuda sin introducir manualmente `mlflow_run_id`, `tracking_session_id` ni checkpoint.
- Este pendiente no invalida que HU009 haya completado una corrida full continua; valida una capacidad operacional distinta.

---

### HU009 - Entrenamiento DDQN completo

**Estado:** [COMPLETADA].

**Propósito:** ejecutar el primer entrenamiento DDQN completo y prolongado para `ALE/Assault-v5`, reutilizando entorno reproducible, DDQN, Trainer, checkpoints, TensorBoard y MLflow.

**Diseño implementado:**

- `2_Assault/src/training_profiles.py` resuelve perfiles `smoke|full`.
- Perfil `full`: `250000` timesteps, Replay Buffer `50000`, `learning_starts=10000`, `train_frequency=4`, `target_update_frequency=1000`, `epsilon_decay_steps=200000`, checkpoint cada `25000` y TensorBoard persistente.
- `FULL_TRAINING_READY` valida Colab, CUDA, preflight, observación, action space, fingerprint, storage, target y RAM antes de iniciar cómputo prolongado.
- El perfil full no modifica silenciosamente el contrato DDQN ni el entorno.

**Evidencia de cierre de la corrida full `assault_ddqn_full_001`:**

- runtime: Google Colab;
- GPU: NVIDIA A100-SXM4-40GB;
- `TRAINING_PROFILE=full`;
- `FULL_TRAINING_READY=True`;
- `initial_global_step=0`;
- `final_global_step=250000`;
- `episodes_completed=417`;
- `transitions_stored=250000`;
- `updates_count=60001`;
- `online_weights_changed=True`;
- `epsilon_final=0.01`;
- `last_loss=0.6625979542732239`;
- `mean_loss=0.8943741276802306`;
- `last_q_mean=22.840946197509766`;
- `mean_q_mean=11.862329530789939`;
- `duration_seconds=656.916490947`;
- Replay Buffer final `50000`;
- checkpoint final `/content/drive/MyDrive/reinforcement_learning_reto_1/checkpoints/assault_ddqn_full_001/checkpoint_step_250000.pt`;
- checkpoint size `2881543669` bytes (~2.88 GB), incluyendo Replay Buffer para resume;
- evaluación técnica/final ejecutada sobre `10` episodios con `epsilon=0.0`;
- `evaluation_mean_reward=569.1`;
- `median=556.5`, `std=76.55644976094437`, `min=462.0`, `max=735.0`;
- `MLFLOW_TRACKING_PASS=True`;
- checkpoint, MLflow y TensorBoard persistentes.

La corrida full demuestra entrenamiento real: hubo `60001` updates, los pesos cambiaron y las métricas loss/Q se mantuvieron finitas. No se requiere repetir 250000 timesteps para cerrar HU009.

**Habilita:** HU009C.

---

### HU009C — Artefactos de entrega: modelo compacto, gráficas, video y reporte técnico

**Estado:** PENDIENTE.

**Propósito:** convertir la corrida full ya validada en artefactos académicos de entrega reproducibles sin repetir el entrenamiento prolongado.

Debe implementar:

- modelo compacto de inferencia derivado del checkpoint final, sin Replay Buffer ni optimizer, conservando metadata y trazabilidad;
- carga y validación del modelo compacto desde un agente/runtime nuevo;
- máximo **3 figuras TensorBoard no redundantes**:
  1. recompensa por episodio + media móvil;
  2. loss DDQN;
  3. `q_mean` + epsilon en una visualización conjunta y legible;
- video MP4 reproducible que combine evidencia breve del proceso de entrenamiento con gameplay del agente usando `epsilon=0.0`;
- reporte técnico profesional dentro de `2_Assault/assault_ddqn.ipynb`, con algoritmo, entorno/preprocessing, hiperparámetros, librerías/versiones, hardware, tiempo, entrenamiento, evaluación de al menos 10 episodios, comparación con baseline, comportamiento aprendido, limitaciones, conclusión y referencias a artefactos.

**Fuente de verdad / DWP ejecutable:**

- `2_Assault/docs/hu009c_artefactos_entrega_modelo_tensorboard_video_reporte.md`

**Restricciones principales:**

- no repetir automáticamente el entrenamiento full;
- no modificar el checkpoint original;
- no subir binarios grandes a Git;
- notebook como orquestador/reporte, lógica reutilizable en `src/`;
- no inventar métricas si faltan artifacts;
- máximo tres figuras de entrenamiento.

**Resultado esperado:** modelo compacto + evidencia TensorBoard + video + reporte alineados al mismo `project_run_id` y checkpoint fuente.

**Habilita:** HU010 y contribuye directamente a HU011/HU012.

---

### HU010 — Optimización controlada de hiperparámetros

**Propósito:** mejorar el desempeño sin realizar una búsqueda exhaustiva costosa.

Solo deben modificarse parámetros con una hipótesis clara, por ejemplo:

- learning rate;
- gamma;
- batch size;
- tamaño de Replay Buffer;
- learning starts;
- epsilon decay;
- frecuencia de aprendizaje;
- frecuencia de sincronización Target Network.

Cada variante debe compararse con el experimento anterior usando MLflow y un protocolo de evaluación consistente.

**Resultado esperado:** seleccionar justificadamente el mejor modelo/configuración candidata para evaluación formal.

**Habilita:** HU011.

---

### HU011 — Evaluación formal contra baseline

**Propósito:** ejecutar la medición oficial del desempeño del agente.

Debe:

- cargar el modelo seleccionado;
- separar completamente evaluación de entrenamiento;
- ejecutar al menos 10 episodios independientes;
- utilizar recompensa real del entorno;
- desactivar exploración o utilizar epsilon de evaluación explícitamente documentado;
- calcular media, mediana, desviación estándar, mínimo y máximo;
- registrar duración/vidas cuando aporten al análisis;
- comparar contra el baseline aleatorio de la ficha técnica.

**Métrica principal:** recompensa promedio sobre al menos 10 episodios independientes.

**Criterio interno mínimo:** recompensa promedio del agente superior al baseline aleatorio bajo un protocolo comparable.

**Resultado esperado:** evidencia cuantitativa de que el agente aprendió un comportamiento superior a la política aleatoria.

**Habilita:** HU012.

---

### HU012 — Evidencias y entrega final

**Propósito:** consolidar todos los artefactos requeridos por el reto académico.

Debe producir/verificar:

- `assault_ddqn.ipynb` ejecutable en Google Colab;
- instalación explícita de dependencias;
- modelo entrenado correspondiente a la ejecución documentada;
- video del entrenamiento/comportamiento aprendido;
- hiperparámetros;
- versiones de librerías;
- hardware utilizado;
- tiempo de entrenamiento;
- gráficas de TensorBoard;
- evaluación sobre al menos 10 episodios;
- comparación contra baseline;
- análisis del comportamiento aprendido;
- conclusión técnica.

**Resultado esperado:** entrega reproducible, consistente y trazable entre notebook, modelo, métricas, video y código Git.

---

## 5. Reglas de transición entre HUs

Una HU posterior no debe utilizarse para ocultar una validación fallida de una HU anterior.

Antes de avanzar:

1. todos los criterios de aceptación de la HU deben estar satisfechos;
2. todas las autovalidaciones obligatorias deben ejecutarse correctamente;
3. la evidencia debe quedar disponible en el PR, notebook, logs o artefactos según corresponda;
4. los criterios de finalización deben estar completos;
5. cualquier desviación debe quedar documentada explícitamente;
6. el PR debe ser revisable y limitarse al alcance de la HU.

Para cambios de implementación se debe seguir el flujo:

```text
main
  ↓
feature/HU-xxx
  ↓
implementación
  ↓
autovalidaciones
  ↓
Pull Request
  ↓
revisión
  ↓
merge a main
```

---

## 6. Estándar obligatorio para construir cada HU

Cada nueva HU deberá redactarse como un **Deep Work Plan (DWP) ejecutable**, de forma que otro desarrollador o agente pueda implementarla sin tener que reinterpretar el objetivo.

Como mínimo debe contener las siguientes secciones.

### 6.1 Identificación

- ID y nombre de la HU.
- Estado.
- Dependencias previas.
- Archivos/documentos fuente de verdad.

### 6.2 Contexto y problema

Explicar:

- qué problema existe;
- por qué debe resolverse ahora;
- qué decisión o capacidad habilita la HU;
- qué información de `arquitectura.md`, `ficha_tecnica.md` o del enunciado condiciona la solución.

### 6.3 Historia de usuario

Formato recomendado:

> **Como** [actor], **quiero** [capacidad], **para** [resultado/valor].

### 6.4 Objetivo verificable

Definir el resultado técnico concreto que debe existir al finalizar. Debe ser observable y verificable, no una intención genérica.

### 6.5 Alcance

Listar explícitamente:

- componentes que deben crearse/modificarse;
- comportamientos requeridos;
- integraciones necesarias;
- datos/configuración involucrados.

### 6.6 Fuera de alcance

Indicar lo que **no** debe implementarse en la HU para evitar sobreingeniería, scope creep y duplicación con historias posteriores.

### 6.7 Decisiones y restricciones técnicas

Documentar únicamente las decisiones necesarias para implementar la historia, por ejemplo:

- interfaces esperadas;
- módulos responsables;
- reglas de arquitectura;
- compatibilidad con Colab/GPU;
- idempotencia;
- persistencia;
- principios SOLID/DRY aplicables;
- restricciones del algoritmo DDQN.

La HU no debe contradecir `2_Assault/docs/arquitectura.md`. Si requiere una excepción, esta debe justificarse explícitamente.

### 6.8 Plan de implementación / tareas

Dividir la implementación en tareas pequeñas y ordenadas. Cada tarea debe indicar qué cambia, dónde cambia y resultado esperado.

### 6.9 Criterios de aceptación

Los criterios deben ser objetivos y verificables. Preferir formato Given/When/Then y cubrir comportamiento funcional, integración, casos de borde, reproducibilidad y restricciones arquitectónicas.

### 6.10 Criterios de finalización / Definition of Done

Debe existir una checklist explícita que incluya implementación, criterios de aceptación, autovalidaciones, errores bloqueantes, documentación, evidencia y alcance del PR.

### 6.11 Autovalidaciones obligatorias

Toda HU debe especificar **cómo comprobar automática o semiautomáticamente que la implementación funciona**.

Cada autovalidación debe definir:

1. comando o procedimiento;
2. resultado esperado;
3. criterio de éxito/fallo.

Cuando una validación solo pueda ejecutarse en Google Colab/GPU, debe quedar marcada explícitamente como **validación Colab pendiente de ejecución por el usuario** y no debe sustituirse con resultados inventados.

### 6.12 Evidencias esperadas

Definir qué evidencia demuestra el éxito: tests, métricas, TensorBoard, MLflow, checkpoint, notebook ejecutado, resultados de evaluación o artefactos según aplique.

### 6.13 Riesgos y consideraciones

Registrar únicamente riesgos materiales para la HU, como RAM/VRAM, duración de Colab, pérdida de checkpoints, versiones, shapes, frameskip o desviaciones de comparabilidad.

---

## 7. Regla de cierre de una HU

Una HU se considera **implementada** cuando el código existe.

Una HU se considera **terminada** únicamente cuando:

```text
Implementación
    +
Criterios de aceptación
    +
Autovalidaciones exitosas
    +
Evidencia verificable
    +
Definition of Done completa
    =
HU CERRADA
```

Si una autovalidación obligatoria depende de Google Colab y todavía no fue ejecutada, la HU debe mantenerse como **implementada pendiente de validación**, no como completada.

---

## 8. Evidencia de implementación HU009C

**Estado:** [IMPLEMENTADA - VALIDACIÓN COLAB PENDIENTE].

**Fecha local:** 2026-08-30.

**Rama:** `feature/hu009c-delivery-artifacts`.

**Alcance implementado:**

- `2_Assault/src/model_artifact.py` agrega `export_inference_model(...)`, `load_inference_model(...)` y `compute_sha256(...)`.
- El artefacto compacto guarda solo `online_network`, schema, arquitectura `QNetwork`, contrato de entorno/preprocessing y metadata de lineage.
- El artefacto compacto excluye Replay Buffer, optimizer, Target Network e históricos de entrenamiento.
- La exportación calcula SHA-256 del checkpoint fuente y del modelo compacto, escribe sidecars `.sha256` y `.metadata.json`, y aplica guardrail `<100 MiB`.
- `2_Assault/src/reporting.py` prepara exactamente tres figuras de entrenamiento desde TensorBoard real: recompensa + media móvil, loss DDQN, y `q_mean` + `epsilon`.
- `2_Assault/src/video.py` genera MP4 con intro de evidencia de entrenamiento y gameplay desde un agente cargado para inferencia, usando frames RGB y `epsilon` explícito.
- `2_Assault/assault_ddqn.ipynb` queda como reporte/orquestador HU009C; el entrenamiento HU009 queda protegido por `ASSAULT_RUN_TRAINING=1` y no corre por defecto.
- `2_Assault/requirements.txt` declara `imageio[ffmpeg]>=2.34` para MP4 portable en Colab.

**Tests agregados:**

- `2_Assault/tests/test_model_artifact.py`
- `2_Assault/tests/test_reporting.py`
- `2_Assault/tests/test_video.py`
- `2_Assault/tests/test_notebook_hu009c.py`

**Validaciones locales ejecutadas y observadas:**

- `python -m compileall -q 2_Assault/src` -> PASS sin salida.
- `python -m pytest 2_Assault/tests/test_model_artifact.py -q` -> `4 passed`.
- `python -m pytest 2_Assault/tests/test_reporting.py 2_Assault/tests/test_video.py -q` -> `5 passed`.
- `python -m pytest 2_Assault/tests/test_notebook_hu009c.py -q` -> `3 passed`.

**Validaciones pendientes por artefactos reales Colab/Drive:**

- AV06: exportar modelo compacto real desde `/content/drive/MyDrive/reinforcement_learning_reto_1/checkpoints/assault_ddqn_full_001/checkpoint_step_250000.pt`, imprimir tamaño y SHA-256, validar `<100 MiB` y smoke.
- AV07: evaluar al menos 10 episodios desde el modelo compacto cargado desde disco con `epsilon=0.0`.
- AV08: generar las tres figuras desde event files TensorBoard reales de `assault_ddqn_full_001`.
- AV09: generar y reproducir MP4 real desde `render_mode="rgb_array"`.
- AV10: confirmar consistencia `checkpoint source -> compact model + checksum -> evaluation >=10 episodes -> TensorBoard figures -> video -> notebook report`.

No se ejecutó ni se repitió entrenamiento full de `250000` timesteps durante HU009C local.

### Corrección PR #15 - blocker de bootstrap HU009C

**Fecha local:** 2026-08-30.

**Estado:** [IMPLEMENTADA - VALIDACIÓN COLAB PENDIENTE].

**Causa corregida:**

- `2_Assault/assault_ddqn.ipynb` ejecutaba `prepare_training_session(...)` antes de decidir si `ASSAULT_RUN_TRAINING=0`.
- Para una corrida full ya terminada, con `target_timesteps` igual al `latest_global_step`, ese bootstrap podía fallar correctamente con `target_timesteps must be greater than the restored global_step`.
- La validación de `session_bootstrap.py` se mantiene intacta; la corrección vive en la orquestación HU009C.

**Cambio implementado:**

- `2_Assault/src/hu009c_delivery.py` agrega `resolve_hu009c_execution_mode(...)`.
- En `ASSAULT_RUN_TRAINING=0`, el notebook entra en modo post-training, no llama `prepare_training_session(...)`, no crea nueva sesión MLflow, no modifica manifest y continúa hacia export/modelo/TensorBoard/evaluación/video.
- En `ASSAULT_RUN_TRAINING=1`, el notebook conserva el camino original: `prepare_training_session(...)`, `FULL_TRAINING_READY`, tracking, entrenamiento, checkpointing y manifest.

**Validaciones locales ejecutadas y observadas para esta corrección:**

- `python -m compileall -q 2_Assault/src` -> PASS sin salida.
- `python -m pytest 2_Assault/tests/test_notebook_hu009c.py -q` -> `6 passed`.
- `python -m pytest 2_Assault/tests/test_session_bootstrap.py -q` -> `17 passed, 1 warning`.
- `python -m pytest 2_Assault/tests/test_training_profiles.py -q` -> `19 passed`.
- `python -m pytest 2_Assault/tests -q` -> `129 passed, 2 skipped, 1 warning`.

AV06-AV10 reales siguen pendientes de ejecución en Colab/Drive.

### Mejora PR #15 - visualizaci?n inline del MP4 HU009C

**Fecha local:** 2026-08-30.

**Estado:** [IMPLEMENTADA - VALIDACI?N COLAB PENDIENTE].

**Cambio implementado:**

- `2_Assault/assault_ddqn.ipynb` muestra inline el MP4 generado por HU009C inmediatamente despu?s de `generate_assault_demo_video(...)`.
- La visualizaci?n usa `IPython.display.Video` y `display` cuando `VIDEO_PATH.exists()` y el archivo tiene tama?o mayor que cero.
- La celda imprime `VIDEO_READY=True`, ruta, reward, steps, seed, epsilon, `project_run_id` y checksum del modelo.
- Si IPython/Colab no puede renderizar inline, se imprime `VIDEO_INLINE_WARNING` y se conserva la ruta al MP4 para reproducci?n manual.
- No se cambia entrenamiento, DDQN, checkpointing, TensorBoard, MLflow, evaluator ni `src/video.py`.

HU009C contin?a como [IMPLEMENTADA - VALIDACI?N COLAB PENDIENTE] hasta ejecutar AV06-AV10 reales.

### Correcci?n PR #15 - orquestaci?n AUTO para Colab limpio

**Fecha local:** 2026-08-30.

**Estado:** [IMPLEMENTADA - VALIDACI?N COLAB PENDIENTE].

**Causa corregida:**

- El notebook usaba por defecto `ASSAULT_RUN_TRAINING=0`, por lo que un Run All con Drive vac?o saltaba directamente a delivery y no entrenaba, no generaba checkpoint, TensorBoard, modelo compacto ni video.
- Esto hac?a que la entrega dependiera de artefactos previos en un Drive personal.

**Cambio implementado:**

- `ASSAULT_EXECUTION_MODE=auto` es ahora el default del notebook.
- `AUTO_RESOLUTION=NEW` cuando no existe checkpoint final ni manifest y se debe iniciar entrenamiento desde `global_step=0`.
- `AUTO_RESOLUTION=RESUME` cuando no existe checkpoint final pero `prepare_training_session(...)` resuelve una sesi?n parcial v?lida.
- `AUTO_RESOLUTION=DELIVERY` cuando existe el checkpoint final esperado y no debe llamarse `prepare_training_session(...)`.
- `ASSAULT_EXECUTION_MODE=delivery` falla claramente si no existe checkpoint final.
- `ASSAULT_EXECUTION_MODE=train` conserva el flujo de entrenamiento existente.
- `2_Assault/src/training_session.py` ahora inyecta `CheckpointManager`, `checkpoint_interval_steps` y `checkpoint_save_replay_buffer` al `Trainer` para checkpoints peri?dicos persistentes; si el checkpoint final ya fue guardado peri?dicamente, se reutiliza y no se intenta guardarlo dos veces.

**Validaciones locales ejecutadas y observadas para esta correcci?n:**

- `python -m compileall -q 2_Assault/src` -> PASS sin salida.
- `python -m pytest 2_Assault/tests/test_notebook_hu009c.py -q` -> `11 passed`.
- `python -m pytest 2_Assault/tests/test_training_session.py -q` -> `2 passed`.
- `python -m pytest 2_Assault/tests/test_session_bootstrap.py -q` -> `17 passed, 1 warning`.
- `python -m pytest 2_Assault/tests/test_training_profiles.py -q` -> `19 passed`.
- `python -m pytest 2_Assault/tests/test_checkpointing.py -q` -> `15 passed`.
- `python -m pytest 2_Assault/tests/test_reporting.py 2_Assault/tests/test_video.py -q` -> `5 passed`.

**Validaci?n real pendiente:**

- `VALIDACI?N REAL CLEAN COLAB + EMPTY DRIVE PENDIENTE`: ejecutar Run All en Colab limpio con Drive vac?o y GPU disponible para demostrar `AUTO_RESOLUTION=NEW -> checkpoint final -> HU009C_ARTIFACTS_READY`.
