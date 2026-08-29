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
HU005  Checkpoints + reanudación + idempotencia        [COMPLETADA]
  ↓
HU006  Observabilidad con TensorBoard                  [COMPLETADA]
  ↓
HU007  Smoke test end-to-end                         [COMPLETADA]
  ↓
HU008  MLflow y trazabilidad de experimentos         [IMPLEMENTADA - VALIDACIONES LOCALES COMPLETADAS - LISTA PARA VALIDACION COLAB MULTISESION]
  ↓
HU008B Automatizacion de reanudacion de experimentos [IMPLEMENTADA - VALIDACIONES LOCALES COMPLETADAS - VALIDACION COLAB MULTISESION AUTOMATICA PENDIENTE]
  â†“
HU009  Entrenamiento DDQN completo                 [IMPLEMENTADA - VALIDACIONES LOCALES COMPLETADAS - G0 COLAB + ENTRENAMIENTO FULL PENDIENTES]
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
- Limitaciones y pendientes:
  - no se ejecutó runtime remoto de Colab desde Codex por la restricción ya diagnosticada en HU002B/HU005;
  - no se implementó MLflow, evaluación formal, video, entrenamiento largo ni selección de mejor modelo;
  - los tags `episode/*` dependen de que finalice un episodio real; en corridas muy cortas de Assault pueden no aparecer.

**Habilita:** HU007.

---

### HU007 — Smoke test end-to-end

**Estado:** implementada - validaciones locales completadas - Colab/GPU pendiente.

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

**Evidencia de implementación HU007 (2026-08-28, rama `feature/hu007-e2e-smoke`):**

- Commit final del PR: último commit de la rama `feature/hu007-e2e-smoke`.
- Archivos implementados/modificados:
  - `2_Assault/configs/ddqn_config.yaml`: agrega bloque `e2e_smoke` con `enabled`, timesteps cortos, evaluación corta, replay buffer reducido y `require_cuda=true` por defecto para el cierre real en Colab/GPU.
  - `2_Assault/src/evaluator.py`: agrega evaluación separada del entrenamiento, con epsilon configurable, recompensas crudas, longitudes y restauración de modos `train/eval`.
  - `2_Assault/src/e2e_smoke.py`: orquesta el smoke end-to-end con preflight, entorno real, DDQN, Replay Buffer, entrenamiento A, checkpoint, restauración, entrenamiento B, TensorBoard y evaluación.
  - `2_Assault/src/callbacks.py`: estabiliza la lectura de scalars de TensorBoard ordenando eventos por `global_step`.
  - `2_Assault/tests/test_e2e_smoke.py`: cubre evaluador, contrato de no-mutación, fail-fast CUDA y smoke real Assault en CPU local.
  - `2_Assault/assault_ddqn.ipynb`: integra la ejecución HU007 y muestra gates/resultados.
  - `2_Assault/docs/implementacion.md`: registra esta evidencia.
- Contrato de ejecución implementado:
  - por configuración, HU007 exige CUDA (`require_cuda=true`) para poder declarar `E2E_SMOKE_PASS=True`;
  - en ejecución local sin CUDA se permite validación reducida con `ASSAULT_E2E_REQUIRE_CUDA=0`, pero el gate formal permanece pendiente;
  - `E2E_SMOKE_PASS` solo puede ser `True` en Google Colab con dispositivo CUDA;
  - `LOCAL_E2E_SMOKE_PASS=True` solo certifica que el pipeline corre localmente de extremo a extremo sin GPU.
- Ejecución local automatizada del notebook:
  - variables usadas: `ASSAULT_INSTALL_DEPENDENCIES=0`, `ASSAULT_BOOTSTRAP_REF=feature/hu007-e2e-smoke`, `ASSAULT_CHECKPOINT_DIR=<temp>`, `ASSAULT_TENSORBOARD_DIR=<temp>`, `ASSAULT_RUN_ID=<run temporal>`, `ASSAULT_E2E_REQUIRE_CUDA=0`;
  - celdas de código ejecutadas: `10`;
  - celdas omitidas: `[]`;
  - resultado: `NOTEBOOK_CODE_CELLS_OK`.
- Evidencia local del smoke end-to-end:
  - runtime detectado: `local`;
  - dispositivo usado: `cpu`;
  - preflight: `READY_FOR_TRAINING=True`;
  - observación procesada: `(4, 84, 84)`, dtype `uint8`;
  - action space: `Discrete(7)`;
  - segmento A: `segment_a_steps=48`, `segment_a_updates=5`;
  - checkpoint: `checkpoint_step_000048.pt`, tamaño aproximado `29,741,058` bytes;
  - restauración: `restored_global_step=48`, `replay_buffer_restored=True`;
  - segmento B: `segment_b_final_step=64`, `segment_b_updates_total=9`;
  - TensorBoard: `tensorboard_event_files=2`;
  - TensorBoard tags preservados: `train/epsilon`, `train/loss`, `train/q_mean`, `train/learning_rate`;
  - pasos post-resume validados para tags principales: `[52, 56, 60, 64]`;
  - evaluación corta separada: `evaluation_rewards=[0.0, 0.0]`, `evaluation_mean_reward=0.0`, `evaluation_lengths=[256, 256]`;
  - memoria local: RSS aproximado `324.56 MB -> 375.28 MB`, RAM disponible aproximada `0.98 GB -> 0.93 GB`;
  - verificación de evaluación: no muta pesos online, pesos target, estado del optimizador, replay buffer ni `global_step`;
  - resultado local: `LOCAL_E2E_SMOKE_PASS=True`;
  - resultado formal HU007: `E2E_SMOKE_PASS=False` por no haberse ejecutado en Colab/GPU.
- Autovalidaciones ejecutadas:
  - `python -c "import sys; sys.path.insert(0, '2_Assault'); from src.evaluator import evaluate_agent; from src.e2e_smoke import run_e2e_smoke; print('HU007 imports OK')"` -> `HU007 imports OK`;
  - `python -m compileall -q 2_Assault/src` -> PASS sin salida ni errores;
  - `python -m pytest 2_Assault/tests/test_e2e_smoke.py -q` -> `5 passed`;
  - `python -m pytest 2_Assault/tests/test_tensorboard.py::test_resume_reuses_run_id_and_continues_global_steps 2_Assault/tests/test_e2e_smoke.py -q` -> `6 passed`;
  - `python -m pytest 2_Assault/tests -q` -> `64 passed, 2 skipped`.
- Corrección post revisión de pares - liberación de memoria entre segmentos (2026-08-28, rama `fix/hu007-release-gpu-memory`):
  - problema detectado: el flujo E2E creaba `agent_b` y `buffer_b` mientras `agent_a` y `buffer_a` seguían referenciados después del checkpoint del segmento A;
  - impacto potencial: en Colab GPU podían coexistir dos agentes completos en VRAM durante el segmento B, debilitando el objetivo de HU007 de detectar problemas de memoria antes de HU009;
  - causa: después de `checkpoint`, `logger_a.flush()`, `logger_a.close()` y `env_a.close()`, el código conservaba referencias Python a objetos pesados del segmento A;
  - solución: capturar primero la evidencia liviana del segmento A (`TrainingSummary`, `CheckpointMetadata`, event files, scalars y snapshots), luego ejecutar explícitamente `del agent_a`, `del buffer_a`, `del logger_a`, `del env_a`, `gc.collect()` y `torch.cuda.empty_cache()` solo si CUDA está disponible;
  - orden implementado: Segment A -> checkpoint -> flush/close logger -> close env -> scalars/event files -> `memory_after_segment_a` -> liberar referencias A -> `gc.collect()` -> `torch.cuda.empty_cache()` condicional -> `memory_after_release` -> crear `agent_b`, `buffer_b`, `logger_b`, `env_b` -> `resume_full`;
  - snapshot agregado: `memory_after_release`, tomado después de liberar referencias y antes de crear las instancias B;
  - snapshots expuestos por HU007: `memory_before`, `memory_after_segment_a`, `memory_after_release`, `memory_after`;
  - CUDA cleanup: `torch.cuda.empty_cache()` no se usa como prueba de ausencia de referencias; primero se eliminan referencias Python, luego se ejecuta GC y finalmente se limpia caché CUDA si aplica;
  - archivos modificados: `2_Assault/src/e2e_smoke.py`, `2_Assault/tests/test_e2e_smoke.py`, `2_Assault/assault_ddqn.ipynb`, `2_Assault/docs/implementacion.md`;
  - tests agregados/extendidos: validación de snapshot post-release, campos CUDA `None` en CPU y llamada condicional a `torch.cuda.empty_cache()` mediante monkeypatch sin falsear una ejecución GPU completa;
  - resultados locales de la corrección: `python -m compileall -q 2_Assault/src` -> PASS; `python -m pytest 2_Assault/tests/test_e2e_smoke.py -q` -> `6 passed`; `python -m pytest 2_Assault/tests -q` -> `65 passed, 2 skipped`;
  - notebook local automatizable ejecutado con `ASSAULT_BOOTSTRAP_REF=fix/hu007-release-gpu-memory`, directorios temporales y `ASSAULT_E2E_REQUIRE_CUDA=0` -> `NOTEBOOK_CODE_CELLS_OK`, `LOCAL_E2E_SMOKE_PASS=True`, `E2E_SMOKE_PASS=False`;
  - limitación: Codex no ejecutó Colab GPU para esta corrección; la reducción real de VRAM y el cierre formal `E2E_SMOKE_PASS=True` quedan pendientes de validación por el usuario en Google Colab GPU.
- Limitaciones y pendientes:
  - no se ejecutó runtime remoto de Google Colab desde Codex;
  - no se inventan resultados GPU ni Colab;
  - queda pendiente ejecutar el notebook en Google Colab con GPU y `ASSAULT_E2E_REQUIRE_CUDA=1` para obtener `E2E_SMOKE_PASS=True`;
  - HU007 bloquea HU009 hasta completar esa evidencia remota;
  - no se implementó MLflow, entrenamiento largo, optimización de hiperparámetros, videos ni evaluación formal contra baseline, porque pertenecen a HU008-HU012.

**Habilita:** HU008.

---

### HU008 — MLflow y trazabilidad de experimentos

**Estado:** implementada - validaciones locales completadas - validacion Colab pendiente.

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

**Evidencia de implementación HU008 (2026-08-28, rama `feature/hu008-mlflow-tracking`):**

- Archivos creados/modificados:
  - `2_Assault/src/tracking.py`: capa única que importa y encapsula MLflow.
  - `2_Assault/tests/test_tracking.py`: validaciones de tracking con backend filesystem temporal.
  - `2_Assault/configs/ddqn_config.yaml`: bloque `mlflow` centralizado.
  - `2_Assault/requirements.txt`: dependencia `mlflow>=2.16,<3.0`.
  - `2_Assault/assault_ddqn.ipynb`: orquestación HU008 con new/resume explícito y logging posterior al smoke.
  - `2_Assault/src/utils.py`: agrega `cuda_version` reutilizable en runtime metadata.
  - `.gitignore`: evita versionar stores locales `mlruns/`.
  - `2_Assault/docs/implementacion.md`: registra evidencia real.
- Arquitectura implementada:
  - `Trainer`, `Evaluator`, `CheckpointManager`, `agent`, `network`, `environment` y `replay_buffer` permanecen sin imports ni llamadas directas a MLflow.
  - El notebook actúa como orquestador y usa `MLflowTracker` para iniciar/cerrar runs y registrar resúmenes.
  - TensorBoard conserva las curvas densas; MLflow registra identidad, parámetros, métricas agregadas, artefactos livianos y referencias.
- Configuración MLflow:
  - `enabled: true`;
  - `experiment_name: assault_ddqn`;
  - `tracking_uri: null`;
  - `local_directory: logs/mlflow`;
  - `tracking_mode: new`;
  - `mlflow_run_id: null`;
  - `artifact_location: null`;
  - `log_checkpoint_binary: false`.
- Overrides soportados:
  - `ASSAULT_MLFLOW_TRACKING_URI`;
  - `ASSAULT_MLFLOW_EXPERIMENT`;
  - `ASSAULT_MLFLOW_TRACKING_MODE`;
  - `ASSAULT_MLFLOW_RUN_ID`;
  - `ASSAULT_MLFLOW_ARTIFACT_LOCATION`.
- Identidad:
  - se diferencia `project_run_id` del proyecto/checkpoint/TensorBoard frente a `mlflow_run_id` técnico de MLflow;
  - ambos quedan disponibles mediante `MLflowRunMetadata`;
  - `project_run_id` se registra como tag y como param `identity.project_run_id`.
- API principal:
  - `MLflowTracker.from_config(...)`;
  - `start_run(project_run_id, tracking_mode, mlflow_run_id, ...)`;
  - `log_run_context(...)`;
  - `log_training_summary(...)`;
  - `log_evaluation_summary(...)`;
  - `log_checkpoint_reference(...)`;
  - `log_config_snapshot(...)`;
  - `log_runtime_metadata(...)`;
  - `get_run(...)`;
  - `end_run(...)`;
  - context manager soportado.
- Parámetros/tags registrados:
  - identidad: `DDQN`, `project_run_id`, experimento, seed;
  - Git/runtime: commit, ref, runtime local/Colab, device;
  - entorno: id, obs type, action space, frameskip, repeat action probability, full action space;
  - preprocessing: grayscale, resize, frame stack, dtype, normalización en entorno;
  - DDQN/training: gamma, learning rate, epsilon start/final/decay, batch size, Replay Buffer capacity, learning starts, train frequency, Target update frequency, total timesteps;
  - versiones/hardware: Python, Gymnasium, ALE-Py, PyTorch, CUDA, MLflow, CPU, RAM, GPU disponible, nombre GPU y VRAM cuando aplica.
- Métricas agregadas registradas:
  - entrenamiento: initial/final global step, duration, updates count, last/mean loss, last/mean q mean, final epsilon, replay buffer size, episodes completed y best episode reward cuando existe;
  - evaluación: episodes, mean/median/std/min/max reward, mean episode length y epsilon;
  - checkpoint: step y size bytes.
- Artefactos livianos:
  - `config/ddqn_config.json`;
  - `metadata/runtime.json`;
  - `summaries/train_summary.json`;
  - `summaries/eval_summary.json`;
  - `summaries/e2e_smoke_summary.json` desde notebook;
  - `artifacts/checkpoint_reference.json`.
- Checkpoint reference:
  - registra `checkpoint_path`, `project_run_id`, `checkpoint_step`, `checkpoint_size_bytes`, `resume_mode`, `save_replay_buffer` y `checkpoint_binary_logged=false`;
  - no duplica por defecto checkpoints `resume_full` con Replay Buffer.
- Evidencia de new run:
  - backend temporal local;
  - experimento `assault_ddqn_tests`;
  - `MlflowClient` recupera la run creada;
  - params, metrics, tags y artefactos esperados verificados en `test_tracking.py`.
- Evidencia de resume:
  - sesión A crea run y registra métricas;
  - sesión B crea otro objeto `MLflowTracker`, reabre explícitamente el mismo `mlflow_run_id` y registra métricas adicionales;
  - se valida mismo `mlflow_run_id`, mismo `project_run_id`, métrica de sesión A preservada y métrica de sesión B agregada.
- Evidencia de coexistencia TensorBoard + MLflow:
  - `TensorBoardLogger` escribe event files para el mismo `project_run_id`;
  - `MLflowTracker` registra resumen agregado sin modificar TensorBoard ni depender de sus internals.
- Fail-fast:
  - tracking URI local inválido/no escribible produce `RuntimeError` visible cuando `mlflow.enabled=true`;
  - `tracking_mode="resume"` sin `mlflow_run_id` explícito produce `ValueError`;
  - parámetros inmutables de MLflow se validan para no cambiar identidad al reanudar.
- Modo disabled:
  - con `mlflow.enabled=false`, el tracker es no-op, no importa MLflow y no crea store local.
- Validaciones locales ejecutadas:
  - `python -m compileall -q 2_Assault/src` -> PASS;
  - `python -m pytest 2_Assault/tests/test_tracking.py -q` -> `8 passed, 1 warning`;
  - `python -m pytest 2_Assault/tests -q` -> `73 passed, 2 skipped, 1 warning`.
- Notebook local automatizable:
  - variables usadas: `ASSAULT_INSTALL_DEPENDENCIES=0`, `ASSAULT_BOOTSTRAP_REF=feature/hu008-mlflow-tracking`, `ASSAULT_E2E_REQUIRE_CUDA=0`, `ASSAULT_RUN_ID=assault_ddqn_hu008_notebook_local`, `ASSAULT_CHECKPOINT_DIR=<temp>`, `ASSAULT_TENSORBOARD_DIR=<temp>`, `ASSAULT_MLFLOW_TRACKING_URI=<temp>/mlruns`, `ASSAULT_MLFLOW_EXPERIMENT=assault_ddqn_hu008_notebook_local`;
  - celdas ejecutadas: `[2, 3, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22]`;
  - resultado: `NOTEBOOK_CODE_CELLS_OK`;
  - `MLFLOW_TRACKING_PASS=True`;
  - `project_run_id=assault_ddqn_hu008_notebook_local`;
  - `mlflow_run_id=86068e5989aa480da6df72a927d8922e`;
  - tracking URI temporal: `file:///C:/Users/Usuario/AppData/Local/Temp/.../mlruns`;
  - `LOCAL_E2E_SMOKE_PASS=True`;
  - `E2E_SMOKE_PASS=False` porque la validación fue local CPU, no Colab GPU.
- Limitaciones y pendientes:
  - no se ejecutó runtime remoto de Google Colab desde Codex;
  - no se inventan resultados Colab/GPU;
  - queda pendiente ejecutar el notebook en Google Colab con tracking URI persistente/configurable y conservar evidencia de `MLFLOW_TRACKING_PASS=True`;
  - HU008 no implementa entrenamiento largo, HPO, evaluación formal, video, Model Registry, serving ni MLflow server.

**Correccion tecnica HU008 multi-sesion, params estables y evaluacion corta (2026-08-28, rama `feature/hu008-mlflow-tracking`):**

- Se agrega `tracking_session_id` como tercer identificador explicito junto con `project_run_id` y `mlflow_run_id`.
- `MLflowRunMetadata` expone `tracking_session_id` y el notebook imprime `project_run_id`, `mlflow_run_id`, `tracking_session_id`, modo, pasos inicial/final y referencias de checkpoint.
- La configuracion central incluye `mlflow.tracking_session_id: session_001`.
- Overrides soportados para multi-sesion: `ASSAULT_MLFLOW_SESSION_ID`, `ASSAULT_MLFLOW_CHECKPOINT_INPUT`, `ASSAULT_MLFLOW_SESSION_TARGET_TIMESTEPS` y `ASSAULT_RESUME_MODE`.
- `log_run_context(...)` queda limitado a params globales/inmutables: identidad, seed, entorno, preprocessing y configuracion core DDQN compatible entre sesiones.
- Git SHA/ref, runtime, device, versiones, RAM/GPU y `training.total_timesteps` dejan de registrarse como params inmutables y se conservan por sesion.
- `training.total_timesteps` se trata como `session_target_timesteps` cuando HU008 ejecuta segmentos multi-sesion.
- Con MLflow habilitado, iniciar un run sin `tracking_session_id` produce `ValueError`.
- En modo `resume`, reusar un `tracking_session_id` que ya contiene `sessions/<tracking_session_id>/session_metadata.json` dentro del mismo `mlflow_run_id` produce `RuntimeError` fail-fast.
- La deteccion de colision de sesion ya no oculta errores reales de backend: si `MlflowClient.list_artifacts(...)` falla por permisos, I/O o store invalido, el error se propaga.
- Se agrega `src/training_session.py` como orquestador HU008 de una sesion reanudable: crea agente/buffer nuevos, usa `CheckpointManager.load(..., mode="resume_full")` para cargar el checkpoint externo, valida `restored_global_step == initial_global_step`, exige `replay_buffer_restored=True` en resume completo y continua de `N` a `T`.
- El notebook usa el orquestador HU008 para sesiones `new` y `resume`; `new` produce un checkpoint final de `session_001` y `resume` carga explicitamente `ASSAULT_MLFLOW_CHECKPOINT_INPUT` para `session_002` sobre el mismo `mlflow_run_id`.
- La evaluacion corta HU008 se restaura despues del entrenamiento de cada sesion usando `evaluate_agent(...)` con `epsilon=0.0` y episodios cortos configurables; no es evaluacion formal HU011.
- Los artefactos variables se registran bajo `sessions/<tracking_session_id>/`: `effective_config.json`, `session_metadata.json`, `runtime.json`, `training_summary.json`, `evaluation_summary.json`, `checkpoint_reference.json` y `training_session_summary.json`.
- `config/base_config.json` permanece como artefacto global de configuracion logica base; la configuracion efectiva real vive por sesion en `sessions/<tracking_session_id>/effective_config.json`.
- `session_metadata.json` registra al menos identidad completa, modo, inicio/fin, runtime, Git ref/SHA, device, CPU/GPU, pasos inicial/final, `session_target_timesteps`, checkpoint input/output, `checkpoint_input_loaded`, `restored_global_step`, `replay_buffer_restored`, `resume_mode`, `effective_config_artifact` y versiones disponibles.
- `checkpoint_reference.json` incluye `checkpoint_input_reference` y `checkpoint_output_reference` para reconstruir lineage entre sesiones.
- Las metricas agregadas conservan nombres canonicos globales; cuando aplica se registran con step final de entrenamiento.
- Tags actualizados: `latest_tracking_session_id` y `tracking_mode`.
- Validaciones locales ejecutadas:
  - `python -m compileall -q 2_Assault/src` -> PASS;
  - `python -m pytest 2_Assault/tests/test_tracking.py -q` -> `13 passed, 1 warning`;
  - `python -m pytest 2_Assault/tests -q` -> `78 passed, 2 skipped, 1 warning`;
  - tests agregados validan resume externo real A=0->N, checkpoint A, agente/buffer/trainer B nuevos, restore `resume_full` y continuacion N->T;
  - tests agregados validan lineage `session_001.checkpoint_output_reference == session_002.checkpoint_input_reference`;
  - tests agregados validan que hardware/runtime y target por sesion pueden cambiar sin `MLflow param mismatch`;
  - tests agregados validan que un cambio core real como `ddqn.gamma` sigue fallando por param inmutable;
  - tests agregados validan evaluacion corta, metricas `eval/*` y `evaluation_summary.json`;
  - tests agregados validan `effective_config.json` independiente por sesion;
  - tests agregados validan que un fallo real de `MlflowClient.list_artifacts(...)` se propaga y no se interpreta como lista vacia;
  - validacion estatica del notebook -> JSON parseable y celdas HU008 actualizadas para `new`/`resume` con evaluacion corta;
  - `nbclient`/`nbconvert` no estan disponibles en el entorno local Codex, por lo que no se declara Run All local; se ejecuto flujo equivalente real.
- Evidencia del smoke multi-sesion local:
  - `project_run_id=assault_ddqn_hu008_multisession_final`;
  - `mlflow_run_id=d8d73f89e4f34e0aa64b4c0e23239821` en store temporal local;
  - un mismo `mlflow_run_id` contiene `sessions/session_001/` y `sessions/session_002/`;
  - `session_001.final_global_step == session_002.initial_global_step`;
  - `session_002.checkpoint_input_loaded=True`;
  - `session_002.replay_buffer_restored=True`;
  - `session_002.final_global_step > session_002.initial_global_step`;
  - evaluacion corta: `evaluation_episodes=2`, `evaluation_mean_reward=5.0`, `evaluation_epsilon=0.0`;
  - `evaluation_artifacts_present=True`;
  - `effective_config_artifacts_present=True`;
  - `MULTISESSION_CHECKPOINT_RESUME_PASS=True`;
  - `MLFLOW_TRACKING_PASS=True`;
  - intento duplicado de `session_001` sobre el mismo run falla como se espera.
- Evidencia equivalente local del notebook:
  - modo `new`: `tracking_mode=new`, `tracking_session_id=session_001`, `checkpoint_input_loaded=False`, `initial_global_step=0`;
  - modo `resume`: `tracking_mode=resume`, `tracking_session_id=session_002`, mismo `mlflow_run_id`, `checkpoint_input_loaded=True`, `restored_global_step=N`, `replay_buffer_restored=True`, `final_global_step>T`;
  - artefactos verificados bajo `sessions/<tracking_session_id>/`: `effective_config.json`, `session_metadata.json`, `runtime.json`, `training_summary.json`, `evaluation_summary.json`, `checkpoint_reference.json`, `training_session_summary.json`.
- Limitacion documentada: no se ejecuto runtime remoto de Google Colab desde Codex y no se inventan resultados Colab/GPU; HU008 queda lista para validar dos sesiones Colab persistentes con `MLFLOW_TRACKING_PASS=True`.

**Habilita:** HU009.

---

### HU008B - Automatizacion de arranque y reanudacion de experimentos

**Estado:** IMPLEMENTADA - VALIDACIONES LOCALES COMPLETADAS - VALIDACION COLAB MULTISESION AUTOMATICA PENDIENTE.

**Proposito:** automatizar el arranque y la reanudacion de experimentos DDQN multisesion para que el usuario indique solo `project_run_id`, `target_timesteps` y `requested_mode=auto`, sin copiar manualmente `mlflow_run_id`, `tracking_session_id`, checkpoint de entrada ni rutas internas.

**Diseno final implementado:**

- Se agrega `2_Assault/src/session_bootstrap.py` como componente reutilizable de orquestacion previa/posterior a la sesion.
- `prepare_training_session(...)` resuelve `new` o `resume` desde `<BASE>/experiments/<project_run_id>/experiment_state.json`.
- Si no existe manifest, `auto` produce `tracking_mode=new`, `tracking_session_id=session_001`, `mlflow_run_id=None` y `checkpoint_input=None`.
- Si existe manifest valido, `auto` produce `tracking_mode=resume`, reutiliza el mismo `mlflow_run_id`, calcula el siguiente `session_NNN` y resuelve el checkpoint persistente del manifest.
- El manifest contiene solo estado de orquestacion: schema, project id, MLflow run id, ultima sesion, ultimo checkpoint, ultimo global step, resume mode, commit, fingerprint y timestamp.
- La actualizacion del manifest usa escritura temporal + reemplazo atomico mediante `update_experiment_state_after_success(...)`.
- El manifest se actualiza unicamente despues de entrenamiento OK, checkpoint existente, logging MLflow OK, cierre `FINISHED` y llamada explicita posterior.
- `inspect_experiment_state(...)` permite diagnostico no destructivo de manifest, checkpoint y MLflow.
- El fingerprint deterministico cubre invariantes de entorno, preprocessing, network, gamma, learning rate, politica epsilon, batch size, Replay Buffer, frecuencias de entrenamiento/target y seed.
- El modulo no contiene logica DDQN, seleccion de acciones, Trainer ni evaluacion.

**Fail-fast implementado:**

- checkpoint fisico inexistente;
- `project_run_id` inconsistente entre manifest/checkpoint/MLflow;
- MLflow run ausente o perteneciente a otro experimento logico;
- checkpoint que no pertenece al experimento;
- `global_step` incoherente;
- Replay Buffer ausente o incompatible en `resume_full`;
- fingerprint/config incompatible;
- `target_timesteps <= restored_global_step`;
- `tracking_session_id` duplicado en artefactos MLflow;
- manifest corrupto o incompleto;
- commit/ref no explicitos.

**Notebook HU008B:**

- `2_Assault/assault_ddqn.ipynb` queda como orquestador ligero.
- Entrada operacional normal:
  - `ASSAULT_PROJECT_RUN_ID` o default logico nuevo;
  - `ASSAULT_TARGET_TIMESTEPS`;
  - `ASSAULT_REQUESTED_MODE=auto`.
- El notebook llama `prepare_training_session(...)` despues del bootstrap versionado.
- El notebook imprime `SESSION_BOOTSTRAP_READY=True`, `project_run_id`, `tracking_mode`, `mlflow_run_id`, `tracking_session_id`, `checkpoint_input`, `restored_expected_step`, `target_global_step`, `bootstrap_commit` y `config_fingerprint`.
- Ya no requiere copiar manualmente `mlflow_run_id`, `tracking_session_id` ni checkpoint input entre runtimes.
- Sigue usando `MLflowTracker`, `run_training_session`, `evaluate_agent` y `execution_bootstrap`.
- No se hardcodean IDs reales de MLflow, checkpoints historicos ni SHA de validaciones anteriores como valores operativos.

**Archivos creados/modificados:**

- `2_Assault/src/session_bootstrap.py`;
- `2_Assault/tests/test_session_bootstrap.py`;
- `2_Assault/assault_ddqn.ipynb`;
- `2_Assault/docs/implementacion.md`.

**Validaciones locales HU008B ejecutadas:**

- `python -m pytest 2_Assault/tests/test_session_bootstrap.py -q` -> `17 passed, 1 warning`.
- Los tests focales cubren: `new -> session_001`, `resume -> session_002`, mismo `mlflow_run_id`, checkpoint correcto, checkpoint inexistente, Replay Buffer incompatible, fingerprint incompatible, target no mayor al restored step, sesion duplicada, MLflow run de otro `project_run_id`, manifest corrupto, manifest ausente, escritura atomica, fallo previo a update sin modificar manifest, MLflow disabled, diagnostico no destructivo y notebook sin IDs historicos.

**Pendientes:**

- Ejecutar la prueba real en dos runtimes independientes de Google Colab con almacenamiento persistente.
- No se declara HU008B `[COMPLETADA]` hasta observar en Colab: Runtime A `new/session_001 0->48`, destruccion del runtime, Runtime B `resume/session_002 48->64`, mismo `project_run_id`, mismo `mlflow_run_id`, checkpoint cargado, Replay Buffer restaurado, `SESSION_BOOTSTRAP_READY=True`, `MULTISESSION_CHECKPOINT_RESUME_PASS=True` y `MLFLOW_TRACKING_PASS=True`.

**Habilita:** HU009 solo despues de la validacion Colab multisesion automatica.

---

### HU009 - Entrenamiento DDQN completo

**Estado:** IMPLEMENTADA - VALIDACIONES LOCALES COMPLETADAS - G0 COLAB + ENTRENAMIENTO FULL PENDIENTES.

**Proposito:** preparar el primer flujo de entrenamiento DDQN completo y prolongado para `ALE/Assault-v5`, reutilizando entorno reproducible, DDQN, Trainer, checkpoints/resume, TensorBoard, MLflow y automatizacion HU008B, sin ejecutar entrenamiento largo desde Codex.

**Diseno final implementado:**

- Se agrega `2_Assault/src/training_profiles.py` para resolver perfiles de entrenamiento y gates de proteccion de computo.
- `resolve_training_profile(...)` aplica `ASSAULT_TRAINING_PROFILE=smoke|full` desde `2_Assault/configs/ddqn_config.yaml` y devuelve una configuracion efectiva unica para entorno, preflight, bootstrap, Trainer, TensorBoard y MLflow.
- `estimate_replay_buffer_memory(...)` calcula memoria aproximada sin instanciar el Replay Buffer, considerando `states`, `next_states`, `actions`, `rewards` y `dones`.
- `evaluate_full_training_ready(...)` implementa el gate `FULL_TRAINING_READY` sin iniciar entrenamiento.
- `assert_training_can_start(...)` centraliza el fail-fast de HU009: si el perfil es `full` y `FULL_TRAINING_READY=False`, aborta antes de abrir o modificar MLflow, iniciar Trainer, crear checkpoints o actualizar el manifest.
- `expected_epsilon_at_step(...)` valida que epsilon continue segun `global_step` y `epsilon_decay_steps` del perfil activo.
- El notebook conserva `prepare_training_session(...)` de HU008B y actualiza el manifest solo despues de entrenamiento, checkpoint, logging MLflow y cierre `FINISHED`.

**Perfiles configurados:**

- `smoke`: conserva contrato barato previo con `target_timesteps=48`, Replay Buffer `1024`, `learning_starts=32`, `train_frequency=4`, `target_update_frequency=16`, `epsilon_decay_steps=48`, checkpoint cada `24` steps y TensorBoard cada `4` steps.
- `full`: perfil conservador para primera corrida HU009 en Colab GPU con `FULL_TRAINING_TARGET_TIMESTEPS=250000`, Replay Buffer `50000`, `learning_starts=10000`, `train_frequency=4`, `target_update_frequency=1000`, `epsilon_decay_steps=200000`, checkpoint cada `25000` steps, TensorBoard cada `1000` steps y flush cada `5000` steps.
- El target full es configurable mediante `ASSAULT_TARGET_TIMESTEPS`; no esta hardcodeado en `Trainer` ni requiere editar manualmente el notebook.

**Replay Buffer y memoria:**

- Shape estimado: `(4, 84, 84)` con `uint8`.
- Por transicion se consideran dos observaciones visuales (`states` y `next_states`) mas `action int64`, `reward float32` y `done bool`.
- Perfil `full` con capacidad `50000`: memoria estimada aproximada `2.63 GiB` para arrays principales del Replay Buffer.
- Margen requerido para full: `4.0 GiB` adicionales para entorno, Python, PyTorch, MLflow y estructuras auxiliares.
- Si `ram_available_gb < replay_buffer_estimate + margin`, el gate falla antes de entrenar. No se reduce capacidad automaticamente.

**Gate `FULL_TRAINING_READY`:**

Solo es `True` si se cumple:

- `READY_FOR_TRAINING=True`;
- runtime `Google Colab`;
- device `cuda`;
- observacion `(4,84,84)` `uint8`;
- action space `Discrete(7)`;
- `SESSION_BOOTSTRAP_READY=True` y fingerprint consistente;
- tracking store, checkpoint root y TensorBoard root accesibles;
- perfil `full`;
- target mayor que `restored_expected_step`;
- memoria RAM suficiente.

**Notebook HU009:**

- Entrada operacional normal:
  - `ASSAULT_PROJECT_RUN_ID`;
  - `ASSAULT_TRAINING_PROFILE=smoke|full`;
  - `ASSAULT_TARGET_TIMESTEPS`;
  - `ASSAULT_REQUESTED_MODE=auto`.
- No solicita `mlflow_run_id`, `tracking_session_id`, checkpoint path ni `tracking_mode` manual.
- Antes de entrenar imprime `TRAINING_PROFILE`, `SESSION_BOOTSTRAP_READY`, `FULL_TRAINING_READY`, identidad, checkpoint input, restored step, target, SHA, fingerprint, capacidad/memoria del Replay Buffer, RAM disponible y device.
- En perfil `full`, si el gate no pasa se aborta antes de `MLflowTracker.from_config(...)`, `MLflowTracker.start_run(...)`, logging, `run_training_session(...)`, checkpoint y manifest.
- En perfil `smoke`, el flujo sigue disponible para G0 HU009/HU008B sin requerir CUDA local.

**Archivos creados/modificados:**

- `2_Assault/src/training_profiles.py`;
- `2_Assault/tests/test_training_profiles.py`;
- `2_Assault/configs/ddqn_config.yaml`;
- `2_Assault/assault_ddqn.ipynb`;
- `2_Assault/docs/implementacion.md`.

**Validaciones locales HU009 ejecutadas:**

- `python -m compileall -q 2_Assault/src` -> OK.
- `python -m pytest 2_Assault/tests/test_training_profiles.py -q` -> `19 passed`.
- `python -m pytest 2_Assault/tests/test_session_bootstrap.py -q` -> `17 passed, 1 warning`.
- `python -m pytest 2_Assault/tests -q` -> `114 passed, 2 skipped, 1 warning`.
- `git diff --check` -> OK.
- Validacion estatica del notebook -> `NOTEBOOK_HU009_FULL_GATE_ORDER_PASS=True`.

**Correccion PR #14:**

- El gate `FULL_TRAINING_READY` ahora se evalua y aborta antes de abrir/modificar MLflow o iniciar entrenamiento.
- `FULL_TRAINING_READY=False` en perfil `full` no crea ni reabre run de MLflow, no registra parametros/tags/artifacts, no ejecuta Trainer, no crea checkpoint y no modifica manifest.
- Perfil `smoke` conserva el flujo G0/HU008B y no queda bloqueado por `FULL_TRAINING_READY=False`.

**Pendientes:**

- Ejecutar G0 en dos runtimes independientes de Colab: `hu009_preflight_001`, `48 -> 64`, `mode=auto`, sin copiar IDs ni checkpoint.
- Si G0 pasa, actualizar evidencia real y cerrar HU008B antes de iniciar entrenamiento prolongado.
- Ejecutar la primera corrida full HU009 en Colab GPU con almacenamiento persistente.
- No se declara HU009 `[COMPLETADA]` sin entrenamiento real, checkpoint final cargable, TensorBoard/MLflow persistentes y evaluacion tecnica registrada.

**Habilita:** HU010 solo despues de G0 Colab y entrenamiento full HU009 validado.

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
