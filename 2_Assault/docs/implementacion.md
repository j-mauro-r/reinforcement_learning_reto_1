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
HU002  Pipeline reproducible del entorno              [IMPLEMENTADA — VALIDACIÓN LOCAL COMPLETADA — AV09 COLAB PENDIENTE]
  ↓
HU002B Pipeline de ejecución Local → GitHub → Colab
  ↓
HU003  Núcleo DDQN                                      [COMPLETADA]
  ↓
HU004  Ciclo de entrenamiento                              [COMPLETADA]
  ↓
HU005  Checkpoints + reanudación + idempotencia
  ↓
HU006  Observabilidad con TensorBoard
  ↓
HU007  Smoke test end-to-end
  ↓
HU008  MLflow y trazabilidad de experimentos
  ↓
HU009  Entrenamiento DDQN completo
  ↓
HU010  Optimización controlada de hiperparámetros
  ↓
HU011  Evaluación formal contra baseline
  ↓
HU012  Evidencias y entrega final
```

La secuencia es deliberada: primero se construye y valida el sistema y el flujo reproducible de ejecución; después se consume cómputo en entrenamientos largos.

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

**Estado:** implementada — validación local completada — AV09 Colab pendiente.

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
- AV09 queda **prevalidada localmente**; la ejecución real desde runtime limpio de Google Colab continúa pendiente.

**Habilita:** HU002B.

---

### HU002B — Pipeline de ejecución Local → GitHub → Colab

**Estado:** implementada — validaciones automatizables locales completadas — validación Colab pendiente.

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

**Gate:** el cierre de HU002B debe completar la evidencia AV09 de HU002. Si la ejecución limpia en Colab es exitosa, HU002 y HU002B pueden pasar a `[COMPLETADA]`.

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
- Resultado Gate T00 Codex -> Colab:
  - mecanismo intentado: CLI local oficial/equivalente para Colab y puentes Jupyter disponibles;
  - comandos/procedimientos:
    - `Get-Command colab -ErrorAction SilentlyContinue`;
    - `Get-Command google-colab -ErrorAction SilentlyContinue`;
    - `python -m pip show google-colab colab-cli colabcode jupyter_http_over_ws`;
    - `jupyter server list`;
    - `jupyter notebook list`;
    - `jupyter kernelspec list`;
  - resultado real:
    - no existe comando local `colab`;
    - no existe comando local `google-colab`;
    - no hay paquetes Python de CLI Colab instalados;
    - Jupyter local solo expone subcomandos `kernel`, `kernelspec`, `migrate`, `run`, `troubleshoot`;
    - `jupyter-server` y `jupyter-notebook` no están disponibles;
    - kernels locales detectados: `myenv`, `venv`, `python3`;
  - conclusión: Codex no tiene desde este entorno un canal remoto automatizable para ejecutar comandos contra un runtime activo de Google Colab, por lo que no se ejecutaron Python remoto, GPU remota, smoke remoto ni AV09 en Colab. No se inventan resultados remotos.
- Estado AV09 HU002:
  - `HU002 — IMPLEMENTADA — VALIDACIÓN LOCAL COMPLETADA — AV09 COLAB PENDIENTE`.
- Estado HU002B:
  - `HU002B — IMPLEMENTADA — VALIDACIONES AUTOMATIZABLES COMPLETADAS — VALIDACIÓN COLAB PENDIENTE`.
- Desviación respecto del DWP:
  - El notebook incluye un pre-bootstrap mínimo antes de importar `src.execution_bootstrap`, porque en un runtime Colab limpio el repositorio y sus helpers todavía no existen bajo `/content`.
  - Justificación: ese pre-bootstrap solo trae la copia versionada desde GitHub; la validación del contrato, el SHA, la instalación, imports e idempotencia se delegan al helper versionado del repositorio.

**Instrucciones manuales para cerrar AV09 Colab:**

1. Hacer merge del PR o, para validar la rama antes del merge, mantener `BOOTSTRAP_REF = "feature/hu002b-pipeline-local-github-colab"` en el notebook.
2. Abrir `2_Assault/assault_ddqn.ipynb` en Google Colab con runtime limpio.
3. Activar GPU en Colab si se desea registrar disponibilidad CUDA: `Runtime > Change runtime type > GPU`.
4. Ejecutar todas las celdas en orden.
5. Verificar que el bootstrap imprima:
   - runtime `Google Colab`;
   - repo `/content/reinforcement_learning_reto_1`;
   - ref solicitada;
   - SHA resuelto;
   - requirements bajo `/content/reinforcement_learning_reto_1/2_Assault/requirements.txt`;
   - `src.environment` bajo `/content/reinforcement_learning_reto_1/2_Assault/src/environment.py`.
6. Confirmar que aparece `HU002 validations passed.`.
7. Registrar stdout/stderr, versión Python, GPU disponible y SHA ejecutado para completar AV09 de HU002 y la validación Colab de HU002B.

**Reintento Gate T00 con CLI oficial de Colab (2026-08-27, PR #4):**

- Cambio aplicado antes del reintento:
  - `2_Assault/assault_ddqn.ipynb` ahora deja como default `BOOTSTRAP_REF = os.environ.get("ASSAULT_BOOTSTRAP_REF", "main")`.
  - Para validar el PR sin usar `main` como código objetivo, se usó explícitamente `ASSAULT_BOOTSTRAP_REF=feature/hu002b-pipeline-local-github-colab`.
- CLI/mecanismo inspeccionado:
  - En Windows no existe `colab` ni `google-colab`.
  - La CLI oficial `google-colab-cli` documenta `colab exec` para ejecución remota y su README indica soporte actual para Linux y macOS; Windows no está soportado.
  - Se inspeccionó WSL2 Ubuntu disponible en la máquina y se instaló la CLI oficial allí con `uv tool install google-colab-cli`.
- Sintaxis real confirmada:
  - `/root/.local/bin/colab --help` -> PASS, muestra comandos `new`, `sessions`, `status`, `exec`, `run`, `log`, etc.
  - `/root/.local/bin/colab exec --help` -> PASS, muestra `colab exec [OPTIONS]`, `--session`, `--file`, `--output-image`, `--timeout`.
  - `/root/.local/bin/colab version` -> `Version: 0.6.0`.
- Comando de conectividad remoto intentado:
  - `printf 'import sys\nprint(sys.version)\n' | /root/.local/bin/colab exec --timeout 30`
- Resultado real:
  - stdout/stderr: `[colab] Error: No active sessions found. Create one with 'colab new'.`
  - No se obtuvo versión Python remota porque no existía sesión Colab activa accesible para la CLI.
- Comandos adicionales:
  - `/root/.local/bin/colab sessions`
  - `/root/.local/bin/colab status`
  - `/root/.local/bin/colab new -s hu002b-pr4 --gpu T4`
- Bloqueo real:
  - Los comandos `sessions`, `status` y `new` solicitan autorización OAuth interactiva con un código de Google.
  - Codex no tiene un código de autorización ya disponible en la sesión y no debe escribir tokens, passwords ni secretos en archivos o Git.
  - `colab new -s hu002b-pr4 --gpu T4` imprimió la URL OAuth y terminó en `Aborted.`
- Conclusión:
  - La CLI oficial sí está identificada y disponible vía WSL, pero la ejecución remota real no fue posible desde Codex porque no hay sesión activa autenticada ni credenciales interactivas disponibles.
  - No se ejecutaron en Colab el bootstrap remoto, Python remoto, GPU remota, validaciones HU002 remotas ni notebook remoto.
  - No se declara HU002 ni HU002B como completada.
- Validaciones locales posteriores al ajuste:
  - `ASSAULT_BOOTSTRAP_REF=feature/hu002b-pipeline-local-github-colab python -m pytest 2_Assault/tests -q` -> `10 passed in 11.54s`.
  - Celdas de código de `2_Assault/assault_ddqn.ipynb` ejecutadas localmente con `ASSAULT_BOOTSTRAP_REF=feature/hu002b-pipeline-local-github-colab` y `ASSAULT_INSTALL_DEPENDENCIES=0` -> `NOTEBOOK_CODE_CELLS_OK`.
  - SHA local resuelto para la rama del PR durante esa validación: `d265329a4f1f30f00dbc9a1fe224b799dae9e03c`.
- Estado tras el reintento:
  - `HU002 — IMPLEMENTADA — VALIDACIÓN LOCAL COMPLETADA — AV09 COLAB PENDIENTE`.
  - `HU002B — IMPLEMENTADA — VALIDACIONES AUTOMATIZABLES COMPLETADAS — VALIDACIÓN COLAB PENDIENTE`.
- Pasos manuales exactos para desbloquear:
  1. En una terminal WSL con la CLI instalada, ejecutar `/root/.local/bin/colab new -s hu002b-pr4 --gpu T4`.
  2. Abrir la URL OAuth que imprime la CLI, autorizar con la cuenta de Google/Colab y pegar el código en la terminal.
  3. Ejecutar `printf 'import sys\nprint(sys.version)\n' | /root/.local/bin/colab exec -s hu002b-pr4 --timeout 30`.
  4. Ejecutar `printf 'import os\nprint(os.getcwd())\n' | /root/.local/bin/colab exec -s hu002b-pr4 --timeout 30`.
  5. Ejecutar `printf 'import torch\nprint("CUDA available:", torch.cuda.is_available())\nif torch.cuda.is_available():\n    print(torch.cuda.get_device_name(0))\n' | /root/.local/bin/colab exec -s hu002b-pr4 --timeout 30`.
  6. Ejecutar el notebook o script equivalente con `ASSAULT_BOOTSTRAP_REF=feature/hu002b-pipeline-local-github-colab` y capturar SHA, ruta `/content/reinforcement_learning_reto_1`, origen de `src.environment`, resultado `HU002 validations passed.` y logs stdout/stderr.

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

**Habilita:** HU006.

---

### HU006 — Observabilidad con TensorBoard

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

**Habilita:** HU007.

---

### HU007 — Smoke test end-to-end

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

**Habilita:** HU008.

---

### HU008 — MLflow y trazabilidad de experimentos

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

**Habilita:** HU009.

---

### HU009 — Entrenamiento DDQN completo

**Propósito:** ejecutar el primer entrenamiento largo del agente usando la arquitectura validada.

Debe:

- usar GPU de Colab;
- entrenar por timesteps;
- persistir checkpoints fuera del almacenamiento efímero cuando corresponda;
- permitir varias sesiones;
- conservar logs de TensorBoard;
- registrar el experimento en MLflow;
- guardar modelos candidatos;
- registrar tiempo acumulado de entrenamiento.

**Resultado esperado:** producir al menos un modelo DDQN entrenado y evaluable, con trazabilidad completa y evidencia de evolución del aprendizaje.

**Habilita:** HU010.

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
