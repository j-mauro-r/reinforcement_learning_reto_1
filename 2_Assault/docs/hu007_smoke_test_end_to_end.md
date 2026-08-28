# HU007 — Smoke test end-to-end

## 1. Identificación

- **ID:** HU007
- **Nombre:** Smoke test end-to-end
- **Estado:** Lista para implementación
- **Dependencia previa:** HU006 — Observabilidad con TensorBoard `[COMPLETADA]`
- **Dependencias funcionales:** HU002/HU002B, HU003, HU004, HU005 y HU006.
- **Habilita:** HU008 — MLflow y trazabilidad de experimentos.
- **Gate crítico:** no iniciar entrenamiento largo HU009 mientras HU007 no esté aprobada.
- **Entorno objetivo de cierre:** Google Colab con GPU real habilitada.
- **Fuentes de verdad:**
  - `2_Assault/docs/implementacion.md`
  - `2_Assault/docs/arquitectura.md`
  - `2_Assault/docs/linemientos.md`
  - `2_Assault/docs/ficha_tecnica.md`
  - `2_Assault/docs/hu002_pipeline_reproducible_entorno.md`
  - `2_Assault/docs/hu002b_pipeline_ejecucion_local_github_colab.md`
  - `2_Assault/docs/hu003_nucleo_ddqn.md`
  - `2_Assault/docs/hu004_ciclo_entrenamiento.md`
  - `2_Assault/docs/hu005_checkpoints_reanudacion_idempotencia.md`
  - `2_Assault/docs/hu006_observabilidad_tensorboard.md`
  - `2_Assault/configs/ddqn_config.yaml`
  - `2_Assault/assault_ddqn.ipynb`
  - `enunciado_reto_1.txt`

---

## 2. Contexto y problema

Las HUs anteriores han validado los componentes principales del pipeline de Assault de forma incremental:

```text
HU002/HU002B
entorno + preprocessing + bootstrap
          ↓
HU003
núcleo DDQN
          ↓
HU004
training loop + Preflight
          ↓
HU005
checkpoint + resume
          ↓
HU006
TensorBoard
```

Cada componente ha sido validado aisladamente o mediante smokes controlados. Sin embargo, antes de iniciar un entrenamiento largo existe un riesgo diferente: que **la combinación completa falle cuando todos los componentes se ejecutan juntos en el runtime real de entrenamiento**.

Los fallos de integración típicos incluyen:

- incompatibilidad CPU/GPU;
- tensores en devices distintos;
- falta de memoria;
- observaciones con dimensiones inesperadas;
- Replay Buffer incompatible con el entrenamiento real;
- checkpoints que guardan pero no restauran correctamente en otro segmento;
- TensorBoard que no continúa después de resume;
- rutas efímeras de Colab;
- errores de bootstrap o imports al ejecutar desde un commit limpio;
- problemas de continuidad de `global_step`;
- evaluación acoplada accidentalmente al entrenamiento;
- notebook que funciona localmente pero no en Colab.

HU007 debe ejecutar una **corrida E2E corta y deliberadamente barata** que demuestre que el pipeline completo puede operar en Google Colab con GPU antes de consumir recursos en HU009.

Flujo objetivo:

```text
GitHub commit conocido
       ↓
Google Colab limpio + GPU
       ↓
bootstrap HU002B
       ↓
Preflight PASS
       ↓
new run
       ↓
training segmento A
       ↓
TensorBoard + checkpoint
       ↓
restore checkpoint
       ↓
training segmento B
       ↓
continuidad de TensorBoard
       ↓
evaluación corta independiente
       ↓
resumen E2E
       ↓
E2E_SMOKE_PASS=True
```

---

## 3. Historia de usuario

> **Como** equipo que está a punto de invertir tiempo y GPU en el entrenamiento largo del agente DDQN de Assault, **quiero** ejecutar un smoke test end-to-end en Google Colab con GPU que valide entrenamiento, observabilidad, persistencia, reanudación y evaluación corta, **para** detectar fallos de integración antes de iniciar experimentos costosos.

---

## 4. Objetivo verificable

HU007 debe demostrar en una misma ejecución controlada que:

1. Colab ejecuta un commit/ref explícito desde GitHub;
2. GPU está realmente disponible y PyTorch la utiliza;
3. `ALE/Assault-v5` se crea con la configuración esperada;
4. preprocessing produce `(4, 84, 84)` `uint8`;
5. action space continúa siendo `Discrete(7)`;
6. Preflight completo pasa en GPU;
7. Online y Target Network están en el device esperado;
8. Replay Buffer recibe transiciones reales;
9. el agente ejecuta updates DDQN reales;
10. la Online Network modifica pesos;
11. Target Network se sincroniza según configuración;
12. loss y Q-values registrados son finitos;
13. TensorBoard genera event files válidos;
14. un checkpoint real se guarda durante el smoke;
15. el checkpoint puede restaurarse en nuevas instancias;
16. `global_step` continúa desde el valor restaurado;
17. `resume_full` restaura Replay Buffer cuando se usa;
18. TensorBoard conserva la línea temporal después del resume;
19. el entrenamiento continúa produciendo updates después del restore;
20. una evaluación corta independiente puede cargar/usar el agente resultante;
21. la evaluación usa reward raw;
22. la evaluación usa política sin exploración adicional (`epsilon=0`) salvo justificación explícita;
23. entrenamiento y evaluación usan la misma fábrica/preprocessing con modos separados;
24. no existen errores evidentes de dimensiones, device, memoria, paths o persistencia;
25. el notebook puede ejecutar el flujo completo en Colab sin modificar código manualmente entre etapas;
26. no se implementa todavía MLflow ni entrenamiento largo.

Resultado esperado:

```text
GPU detected
Preflight PASS
segment A training PASS
checkpoint PASS
restore PASS
segment B training PASS
TensorBoard PASS
short evaluation PASS
E2E_SMOKE_PASS=True
```

---

## 5. Alcance

### 5.1 Tipo de prueba

HU007 es una **prueba de integración end-to-end**, no una nueva capa arquitectónica.

Debe reutilizar los módulos existentes:

- `environment.py`;
- `network.py`;
- `replay_buffer.py`;
- `agent.py`;
- `trainer.py`;
- `checkpointing.py`;
- `callbacks.py`;
- `preflight.py`;
- `execution_bootstrap.py`.

Solo se deben crear componentes nuevos cuando exista una responsabilidad faltante claramente necesaria para el smoke.

La principal responsabilidad nueva esperada es la evaluación corta desacoplada.

---

## 5.2 Evaluación corta independiente

La arquitectura define `src/evaluator.py` como responsable de evaluación separada del entrenamiento. Si todavía no existe, HU007 debe crear una implementación mínima y reutilizable.

Archivo esperado:

`2_Assault/src/evaluator.py`

Responsabilidades en HU007:

- recibir un agente/modelo ya entrenado;
- crear o recibir entorno de evaluación mediante la fábrica existente;
- ejecutar pocos episodios configurables;
- utilizar `epsilon=0.0` por defecto;
- acumular reward raw;
- registrar longitud por episodio;
- producir resultado estructurado;
- no modificar pesos;
- no ejecutar optimizer;
- no escribir Replay Buffer;
- no cambiar Target Network;
- cerrar correctamente el entorno cuando corresponda.

Interfaz conceptual permitida:

```python
evaluation = evaluate_agent(
    env=eval_env,
    agent=agent,
    episodes=2,
    epsilon=0.0,
)
```

Resultado conceptual:

```python
{
    "episodes": 2,
    "rewards": [...],
    "mean_reward": ...,
    "median_reward": ...,
    "std_reward": ...,
    "min_reward": ...,
    "max_reward": ...,
    "episode_lengths": [...],
}
```

### Importante

HU007 **no reemplaza HU011**.

En HU007 la evaluación es únicamente un smoke funcional con pocos episodios. La evaluación formal del reto continúa siendo ≥10 partidas independientes y comparación contra baseline.

---

## 5.3 GPU obligatoria para cierre

El plan maestro exige una corrida corta con GPU.

Por tanto HU007 no puede cerrarse únicamente con evidencia CPU local.

La validación final debe ejecutarse en Google Colab con un runtime GPU real y demostrar al menos:

```python
torch.cuda.is_available() is True
```

y que el agente usa:

```text
device=cuda
```

Registrar:

- modelo de GPU cuando sea detectable;
- versión CUDA expuesta por PyTorch;
- versión PyTorch;
- memoria GPU total/libre cuando sea simple obtenerla;
- device de parámetros Online/Target.

No exigir un modelo específico de GPU de Colab.

Si Colab no asigna GPU temporalmente, HU007 queda pendiente y no debe falsificarse evidencia.

---

## 5.4 Bootstrap y commit conocido

La ejecución Colab debe usar el flujo HU002B.

Debe quedar registrado:

- repo clonado/usado;
- ref solicitada;
- commit SHA realmente ejecutado;
- path del repositorio;
- path de `2_Assault`;
- origen real de imports críticos.

GitHub continúa siendo fuente de verdad.

No editar módulos directamente en `/content` para hacer que el smoke pase.

Si se prueba un PR antes del merge, usar ref/commit explícito.

---

## 5.5 Preflight como gate

Antes de iniciar el segmento A:

```text
READY_FOR_TRAINING=True
```

debe ser obligatorio.

En Colab GPU, Preflight debe confirmar:

- device;
- environment;
- observation;
- QNetwork;
- Replay Buffer;
- DDQN update;
- loss finita;
- Target estable durante update;
- Target sync;
- save/load.

Si Preflight falla:

```text
E2E training MUST NOT START
```

No ignorar errores para continuar el notebook.

---

## 5.6 Configuración específica del smoke

Extender configuración central solo si es necesario.

Se recomienda una sección explícita, por ejemplo:

```yaml
e2e_smoke:
  enabled: true
  segment_a_timesteps: <valor corto>
  final_timesteps: <valor mayor que segmento A>
  evaluation_episodes: 2
  evaluation_epsilon: 0.0
  require_cuda: true
```

Los nombres pueden adaptarse.

La configuración del smoke debe ser suficientemente pequeña para validar integración y suficientemente grande para producir al menos:

- varias transiciones;
- al menos un update DDQN;
- al menos una sincronización Target;
- al menos un checkpoint;
- eventos TensorBoard.

No convertir estos valores de smoke en hiperparámetros definitivos de HU009.

---

## 5.7 Memoria y Replay Buffer

HU007 debe comprobar comportamiento de memoria antes de HU009.

El Replay Buffer visual puede ser costoso. Para el smoke:

- usar capacidad pequeña/controlada apropiada al número de timesteps;
- no reservar 100k slots visuales únicamente para una prueba corta;
- registrar capacidad usada;
- registrar RAM del proceso/sistema antes y después cuando sea viable;
- registrar memoria GPU antes/después cuando sea viable;
- detectar OOM como fallo bloqueante.

HU007 no decide todavía el tamaño final óptimo del Replay Buffer de HU009, pero debe dejar evidencia útil para esa decisión.

No alterar silenciosamente la configuración objetivo de entrenamiento largo para ocultar problemas de memoria.

---

## 5.8 Segmento A

El segmento A debe ejecutar un `new run` real.

Secuencia:

```text
new
↓
run_id explícito
↓
Preflight PASS
↓
TensorBoard logger
↓
Trainer
↓
N timesteps
↓
DDQN updates
↓
Target sync
↓
checkpoint @ N
```

Debe registrarse:

- `initial_global_step=0`;
- `final_global_step=N`;
- transitions;
- updates;
- first/last update step;
- loss;
- q_mean;
- epsilon;
- target sync steps;
- buffer size;
- event files;
- checkpoint path/size.

---

## 5.9 Restore real

Después del segmento A deben cerrarse o descartarse las instancias relevantes y crearse nuevas:

- environment;
- DDQNAgent;
- ReplayBuffer;
- Trainer;
- TensorBoardLogger.

Luego cargar mediante `CheckpointManager.load(...)`.

Preferencia para HU007:

```text
resume_full
```

porque valida conjuntamente persistencia del Replay Buffer.

Debe demostrarse:

```text
checkpoint_step = N
restored_global_step = N
restored_run_id = original_run_id
replay_buffer_restored = True
```

No simular resume pasando manualmente `initial_global_step=N` sin cargar checkpoint.

---

## 5.10 Segmento B

Después del restore:

```text
resume_full
↓
initial_global_step=N
↓
training continúa
↓
final_global_step=T
```

con:

```text
T > N
```

Debe demostrarse:

- nuevas transiciones;
- nuevos updates DDQN;
- `global_step` no reinicia;
- epsilon se reconstruye correctamente;
- Target sync continúa usando global steps;
- TensorBoard escribe steps > N;
- logs previos se conservan;
- checkpointing no sobrescribe accidentalmente artefactos previos.

---

## 5.11 TensorBoard E2E

HU006 ya validó TensorBoard en CPU/local. HU007 debe confirmar que sigue funcionando dentro del flujo GPU E2E.

Usar `EventAccumulator` o equivalente para verificar programáticamente:

- event files válidos;
- `train/epsilon`;
- `train/loss`;
- `train/q_mean`;
- `train/learning_rate`;
- steps antes y después del restore;
- valores finitos.

Si durante el smoke termina un episodio real, validar también `episode/*`.

Si no termina episodio, no alargar artificialmente el entrenamiento solo para generar esas métricas; HU006 ya posee evidencia controlada para dichos tags.

---

## 5.12 Evaluación posterior al entrenamiento

Al finalizar segmento B:

1. crear un entorno de evaluación con `mode="eval"`;
2. no reutilizar el entorno de training activo;
3. ejecutar pocos episodios (`evaluation_episodes`);
4. usar epsilon de evaluación explícito, por defecto `0.0`;
5. no modificar el Replay Buffer de entrenamiento;
6. no ejecutar `agent.update()`;
7. no cambiar parámetros de Online/Target;
8. acumular reward raw;
9. devolver métricas estructuradas.

Debe existir una prueba que compare parámetros antes/después de evaluación y confirme que no cambiaron.

La evaluación corta demuestra que el modelo resultante es **evaluable**, no que ya supera el baseline.

---

## 5.13 Notebook

Actualizar:

`2_Assault/assault_ddqn.ipynb`

manteniéndolo como orquestador.

Secuencia objetivo:

```text
bootstrap GitHub
↓
runtime + GPU info
↓
config
↓
HU002 environment checks
↓
Preflight GPU
↓
HU005 new state
↓
HU006 TensorBoard
↓
segment A
↓
checkpoint
↓
close/discard objects
↓
restore full
↓
segment B
↓
TensorBoard validation
↓
short evaluation
↓
E2E summary
```

El notebook no debe duplicar:

- DDQN;
- Replay Buffer;
- checkpointing;
- TensorBoard writer;
- evaluator logic.

El notebook debe imprimir explícitamente:

```text
E2E_SMOKE_PASS=True / False
```

según validaciones reales.

---

## 5.14 Ejecución local vs Colab

### Local

Codex/desarrollo puede ejecutar tests y un smoke CPU pequeño para validar código antes de hacer push.

La evidencia local sirve como prevalidación, pero **no cierra HU007**.

### Google Colab

El usuario ejecutará manualmente el notebook con GPU.

HU007 se cierra únicamente cuando exista evidencia real de esa ejecución.

No declarar:

```text
GPU PASS
Colab PASS
```

sin ejecución real.

---

## 5.15 Persistencia de artefactos en Colab

Para el smoke, checkpoint y logs pueden usar directorios temporales si el objetivo es únicamente probar funcionalidad dentro de una sesión.

Sin embargo, debe validarse que las rutas continúan siendo configurables para almacenamiento persistente externo a `/content`, conforme HU005/HU006.

Si se utiliza Google Drive durante la prueba, documentar la ruta; no es requisito automatizar OAuth.

No introducir almacenamiento cloud adicional.

---

## 6. Fuera de alcance

HU007 **no** debe implementar:

- MLflow;
- entrenamiento largo/final;
- evaluación formal de ≥10 episodios;
- comparación formal contra baseline;
- optimización de hiperparámetros;
- selección automática del mejor modelo;
- video final;
- PER;
- Dueling DQN;
- Rainbow;
- Noisy Nets;
- n-step returns;
- reward clipping no aprobado;
- distributed training;
- GitHub Actions para entrenamiento;
- automatización Codex → Colab;
- dashboards diferentes de TensorBoard.

HU008 implementará MLflow.

HU009 ejecutará el entrenamiento largo.

HU011 realizará la evaluación formal contra baseline.

---

## 7. Decisiones técnicas

### 7.1 E2E sobre GPU

CPU no es suficiente como evidencia final porque HU007 existe precisamente para validar el runtime que utilizará HU009.

### 7.2 Evaluación sin exploración

Usar por defecto:

```text
epsilon=0.0
```

para probar la política greedy aprendida. Cualquier epsilon diferente debe justificarse explícitamente.

### 7.3 Reward raw

La evaluación corta debe conservar reward raw del entorno.

### 7.4 Misma fábrica, modos separados

Training y evaluación deben usar `create_assault_env(...)` o fábrica equivalente compartida, diferenciando `mode="train"` y `mode="eval"`.

### 7.5 No medir desempeño todavía

Un reward bajo en este smoke no implica fallo de HU007 siempre que el pipeline funcione correctamente. El agente solo habrá entrenado pocos timesteps.

### 7.6 Fail fast

Cualquier error de:

- GPU requerida no disponible;
- Preflight;
- shape/dtype;
- NaN/Inf;
- checkpoint restore;
- TensorBoard corrupto;
- discontinuidad de `global_step`;
- evaluación que modifica pesos;
- OOM;

debe marcar:

```text
E2E_SMOKE_PASS=False
```

y detener el flujo cuando corresponda.

---

## 8. Plan de implementación / tareas

### T01 — Revisar integración HUs previas

Confirmar contratos reales de HU002–HU006 antes de introducir cambios.

### T02 — Configuración E2E smoke

Agregar configuración central mínima si hace falta.

### T03 — Implementar evaluator mínimo

Crear `src/evaluator.py` si todavía no existe.

### T04 — Tests evaluator

Validar reward raw, epsilon evaluación, episodios, métricas y ausencia de updates.

### T05 — Integración GPU/device

Asegurar que agent, batches y redes operan correctamente en CUDA.

### T06 — Segmento A

Ejecutar new run corto con TensorBoard y checkpoint.

### T07 — Restore completo

Recrear objetos y cargar checkpoint real con Replay Buffer.

### T08 — Segmento B

Continuar desde `global_step` restaurado hasta target global mayor.

### T09 — TensorBoard continuity

Validar events antes/después de restore.

### T10 — Evaluación corta

Ejecutar evaluator sobre entorno independiente.

### T11 — Memoria/hardware

Registrar hardware y ausencia de OOM evidente.

### T12 — Notebook

Integrar flujo E2E manteniendo orquestación limpia.

### T13 — Tests/smoke local

Validar barato antes de Colab.

### T14 — Ejecución real Colab GPU

Usuario ejecuta notebook limpio con GPU y conserva evidencia.

### T15 — Actualizar documentación

Registrar resultados reales en `implementacion.md` únicamente después de las validaciones.

---

## 9. Criterios de aceptación

### CA01 — Bootstrap reproducible

El smoke ejecuta un commit/ref GitHub conocido y registra SHA.

### CA02 — CUDA disponible

La ejecución final demuestra CUDA disponible y usada por el agente.

### CA03 — Environment contract

Observación `(4,84,84)` `uint8`, `Discrete(7)` y preprocessing esperado.

### CA04 — Preflight GPU

Preflight completo pasa usando CUDA.

### CA05 — Replay Buffer real

Se almacenan transiciones reales de Assault.

### CA06 — DDQN update

Existe al menos un update DDQN con loss finita.

### CA07 — Online cambia

Los parámetros Online cambian durante entrenamiento.

### CA08 — Target sync

Existe al menos una sincronización de Target en el smoke.

### CA09 — TensorBoard

Se generan event files válidos con métricas de entrenamiento.

### CA10 — Checkpoint

Segmento A produce checkpoint válido, no vacío y legible.

### CA11 — Restore real

Nuevas instancias restauran estado mediante `CheckpointManager.load()`.

### CA12 — Replay restore

`resume_full` restaura Replay Buffer y metadatos necesarios.

### CA13 — Global-step continuity

Si checkpoint está en `N`, segmento B termina en `T>N` sin reiniciar contador.

### CA14 — Epsilon continuity

Epsilon del segmento B corresponde al `global_step` restaurado.

### CA15 — Updates post-resume

Existe al menos un update DDQN después del restore.

### CA16 — TensorBoard post-resume

Existen nuevos event steps `>N` y logs anteriores se preservan.

### CA17 — Evaluation separation

Evaluación usa entorno separado y no modifica entrenamiento.

### CA18 — Evaluation greedy

La evaluación utiliza `epsilon=0.0` salvo justificación documentada.

### CA19 — Evaluation raw reward

Rewards de evaluación corresponden a rewards raw acumulados del entorno.

### CA20 — Evaluation metrics

Resultado de evaluación incluye rewards y estadísticas básicas.

### CA21 — Agent immutable during evaluation

Pesos Online/Target permanecen iguales antes/después de evaluar.

### CA22 — No Replay mutation during eval

La evaluación no modifica Replay Buffer de entrenamiento.

### CA23 — Memory sanity

No se produce OOM ni crecimiento evidentemente incorrecto durante el smoke.

### CA24 — Notebook orchestration

Notebook consume módulos reutilizables y ejecuta flujo completo.

### CA25 — Fail-fast

Un fallo material produce error/estado E2E fallido y no se oculta.

### CA26 — Scope

No se introduce MLflow, entrenamiento largo ni evaluación formal.

---

## 10. Autovalidaciones obligatorias

### AV01 — Imports

Validar imports de todos los módulos HU002–HU007.

### AV02 — Suite completa local

```bash
python -m pytest 2_Assault/tests -q
```

Todos los tests previos y nuevos deben pasar.

### AV03 — Compile

```bash
python -m compileall -q 2_Assault/src
```

### AV04 — Evaluator unit test

Entorno controlado: episodios, rewards y estadísticas correctas.

### AV05 — Evaluator no learning

Verificar que parámetros del agente no cambian durante evaluación.

### AV06 — Evaluator epsilon

Confirmar `epsilon=0.0` en evaluación por defecto.

### AV07 — Local E2E pre-smoke

Ejecutar flujo corto CPU para detectar fallos de código antes de Colab.

No cierra HU007.

### AV08 — Bootstrap Colab

Runtime limpio usa GitHub y registra SHA real.

### AV09 — CUDA

Validar `torch.cuda.is_available()` y device del agente.

### AV10 — Preflight CUDA

`READY_FOR_TRAINING=True` usando GPU.

### AV11 — Segment A

Entrenamiento real Assault produce transitions, update(s), sync y métricas.

### AV12 — TensorBoard A

EventAccumulator lee eventos válidos del segmento A.

### AV13 — Checkpoint A

Checkpoint de N existe, tiene tamaño >0 y metadatos correctos.

### AV14 — Destroy/recreate

Cerrar/destruir objetos del segmento A y crear nuevos objetos.

### AV15 — Resume full

Cargar checkpoint real y restaurar Replay Buffer/global step/run ID.

### AV16 — Segment B

Continuar hasta `T>N` y producir update(s) nuevos.

### AV17 — TensorBoard continuity

Verificar events anteriores + nuevos steps >N.

### AV18 — Evaluation real Assault

Ejecutar pocos episodios de evaluación independiente.

### AV19 — Evaluation immutability

Comparar parámetros antes/después de la evaluación.

### AV20 — Memory/hardware evidence

Registrar GPU/RAM y confirmar ausencia de OOM.

### AV21 — Notebook Run All Colab

Ejecutar todas las celdas en orden en Colab con GPU y sin cambios manuales de código entre etapas.

### AV22 — E2E final flag

Notebook finaliza con:

```text
E2E_SMOKE_PASS=True
```

### AV23 — Scope audit

Confirmar ausencia de MLflow, entrenamiento largo y evaluación formal.

---

## 11. Evidencias requeridas

El PR/registro de HU007 debe incluir:

- branch/ref ejecutada;
- commit SHA;
- runtime `Google Colab`;
- Python;
- Gymnasium;
- ALE-Py;
- PyTorch;
- CUDA disponible;
- versión CUDA;
- nombre/modelo GPU;
- RAM disponible/uso aproximado;
- GPU memory aproximada cuando esté disponible;
- observation shape/dtype;
- action space;
- Preflight report;
- `run_id`;
- Segment A: start/end step;
- updates;
- loss;
- q_mean;
- epsilon;
- target syncs;
- buffer size;
- TensorBoard event files/tags;
- checkpoint path;
- checkpoint size;
- restore mode;
- restored global step;
- restored Replay Buffer size;
- Segment B final step;
- updates post-resume;
- nuevos TensorBoard steps;
- evaluación: número de episodios;
- rewards por episodio;
- media/mediana/std/min/max;
- longitudes;
- epsilon de evaluación;
- evidencia de pesos sin cambios durante evaluación;
- tiempo total del smoke;
- resultado pytest local;
- resultado compileall;
- resultado Run All Colab;
- `E2E_SMOKE_PASS`;
- limitaciones o warnings observados.

No inventar evidencia ausente.

---

## 12. Definition of Done

HU007 se considera terminada únicamente cuando:

- [ ] tests previos siguen pasando;
- [ ] evaluator mínimo existe y está desacoplado;
- [ ] evaluación corta produce métricas estructuradas;
- [ ] evaluación no modifica pesos ni Replay Buffer;
- [ ] smoke CPU/local prevalidado;
- [ ] bootstrap real Colab validado;
- [ ] commit SHA ejecutado registrado;
- [ ] GPU real disponible;
- [ ] agente realmente usa CUDA;
- [ ] Preflight pasa en CUDA;
- [ ] entorno/preprocessing correctos;
- [ ] segmento A ejecutado sobre Assault real;
- [ ] Replay Buffer recibe transiciones;
- [ ] existe update DDQN real;
- [ ] loss/Q-value finitos;
- [ ] Online Network cambia;
- [ ] Target sync ocurre;
- [ ] TensorBoard genera events válidos;
- [ ] checkpoint segmento A guardado;
- [ ] objetos recreados antes de restore;
- [ ] resume_full restaura agente/optimizer/buffer/progreso;
- [ ] `global_step` continúa correctamente;
- [ ] existen updates post-resume;
- [ ] TensorBoard continúa después de resume;
- [ ] evaluación corta se ejecuta sobre entorno independiente;
- [ ] reward de evaluación es raw;
- [ ] pesos permanecen inmutables durante evaluación;
- [ ] no existe OOM;
- [ ] notebook ejecuta Run All en Colab GPU;
- [ ] `E2E_SMOKE_PASS=True`;
- [ ] documentación/evidencia real actualizada;
- [ ] no existen blockers conocidos;
- [ ] no se implementó scope HU008+.

La falta de GPU/Colab real mantiene HU007 como **pendiente**, incluso si todos los tests locales pasan.

---

## 13. Riesgos y consideraciones

### 13.1 GPU Colab no garantizada

La disponibilidad depende del servicio. Si no hay GPU, registrar el hecho y reintentar cuando exista disponibilidad; no sustituir la evidencia con CPU.

### 13.2 OOM

HU007 debe ser pequeña. Un OOM en el smoke indica un problema que debe resolverse antes de HU009.

### 13.3 Replay Buffer

La capacidad de smoke no representa la capacidad final. El dimensionamiento definitivo sigue siendo decisión de HU009, usando la evidencia de memoria obtenida aquí.

### 13.4 Checkpoint size

`resume_full` puede producir archivos grandes. Registrar tamaño y tiempo de save/load para anticipar impacto en sesiones reales.

### 13.5 Evaluación corta variable

Assault es estocástico (`repeat_action_probability=0.25`). Los rewards del smoke pueden variar. HU007 no impone umbral de desempeño.

### 13.6 Epsilon

El entrenamiento conserva el schedule global. La evaluación usa epsilon separado, por defecto `0.0`.

### 13.7 Episode duration

Una evaluación de Assault puede durar significativamente más que el segmento corto de entrenamiento. Mantener pocos episodios para el smoke.

### 13.8 Persistencia de Colab

El smoke funcional puede usar almacenamiento temporal, pero HU009 deberá persistir checkpoints relevantes fuera de `/content` cuando el entrenamiento se distribuya entre sesiones.

### 13.9 Notebook state

El resultado debe ser reproducible desde un runtime limpio. No depender de variables creadas manualmente fuera de las celdas versionadas.

---

## 14. Resultado esperado y gate

HU007 debe cerrar con una evidencia equivalente a:

```text
GitHub SHA verified
        ↓
Google Colab GPU
        ↓
CUDA PASS
        ↓
Preflight PASS
        ↓
Assault train segment A
        ↓
DDQN update + Target sync
        ↓
TensorBoard PASS
        ↓
checkpoint @ N
        ↓
new objects
        ↓
resume_full @ N
        ↓
train segment B → T
        ↓
TensorBoard continuity
        ↓
short greedy evaluation
        ↓
weights unchanged during eval
        ↓
no OOM / no device mismatch
        ↓
E2E_SMOKE_PASS=True
```

Con HU007 aprobada, el pipeline queda habilitado para integrar MLflow en HU008 y posteriormente realizar el entrenamiento completo en HU009.