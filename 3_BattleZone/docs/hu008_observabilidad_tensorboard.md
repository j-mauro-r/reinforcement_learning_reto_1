# HU008 — Observabilidad del entrenamiento con TensorBoard para BattleZone

## 1. Identificación

- **ID:** HU008
- **Nombre:** Observabilidad del entrenamiento con TensorBoard
- **Estado:** Lista para implementación
- **Dependencias previas:** HU006 — Ciclo de entrenamiento DQN; HU007 — Checkpoints, reanudación e idempotencia
- **Habilita:** HU009 — Smoke test end-to-end; HU010 — Trazabilidad ligera de experimentos; HU011 — Entrenamiento completo
- **Algoritmo vigente:** `DQN`
- **Fuentes de verdad:** `enunciado_reto_1.txt`, `3_BattleZone/docs/implementacion.md`, `3_BattleZone/docs/lineamientos.md`, `3_BattleZone/docs/arquitectura.md`, HU005, HU006, HU007 y `3_BattleZone/configs/battlezone_config.yaml`.

## 2. Contexto

HU006 dejó un trainer DQN funcional y HU007 añadió continuidad mediante checkpoints full/lightweight y resume explícito. Antes del smoke test end-to-end y del entrenamiento largo, el proyecto necesita observabilidad suficiente para detectar fallos de aprendizaje sin depender de prints o inspección manual de objetos.

Los lineamientos del proyecto establecen que TensorBoard es la herramienta principal de observabilidad y debe permitir diagnosticar al menos:

- ausencia de aprendizaje;
- inestabilidad o divergencia de loss;
- colapso prematuro de exploración;
- mejora o estancamiento de recompensa;
- comportamiento anómalo de Q-values.

HU008 introduce logging estructurado y desacoplado del algoritmo, sin implementar todavía manifests completos, evaluación formal, tuning ni entrenamiento largo.

## 3. Objetivo verificable

Implementar observabilidad con TensorBoard sobre el ciclo DQN vigente, de forma que una corrida corta pueda producir event files válidos con métricas útiles y con `global_step` coherente tanto en NEW como en RESUME.

Resultado esperado:

```text
DQNTrainer
  ↓ emits training/episode events
TensorBoard callback/logger
  ↓
logs/<run_id>/
  ↓
events.out.tfevents.*
  ↓
TensorBoard puede leer scalars
```

La HU valida **observabilidad técnica**, no aprendizaje ni performance.

## 4. Decisiones técnicas obligatorias

### DT01 — Observabilidad desacoplada

Crear preferiblemente:

```text
3_BattleZone/src/callbacks.py
```

con responsabilidades de observabilidad periódica.

`trainer.py` no debe contener llamadas directas dispersas a `SummaryWriter.add_scalar(...)` para cada métrica. Debe depender de un contrato pequeño de callback/logger.

No introducir un framework genérico de callbacks si una interfaz simple es suficiente.

### DT02 — TensorBoard como herramienta principal

Usar PyTorch TensorBoard, preferiblemente:

```python
from torch.utils.tensorboard import SummaryWriter
```

No usar MLflow, WandB, Neptune u otros trackers.

### DT03 — Métricas mínimas obligatorias

Registrar cuando corresponda:

#### Por episodio

- `train/episode_reward`;
- `train/episode_reward_mean` — media móvil configurable;
- `train/episode_length`;
- `train/episode_index` si aporta valor y no duplica semántica.

#### Por actualización

- `train/loss`;
- `train/q_value_mean`.

#### Por progreso

- `train/epsilon`;
- `train/replay_size`;
- `train/global_step` solo si se necesita como scalar explícito; el step del evento sigue siendo la referencia principal.

#### Operativa justificada

- `train/learning_rate` si es útil para comprobar configuración, aunque sea constante;
- `train/steps_per_second` o métrica equivalente solo si puede medirse sin complejidad innecesaria.

No registrar métricas solo porque estén disponibles.

### DT04 — Q-value medio debe provenir del agente

El trainer no debe inspeccionar detalles internos de la CNN para calcular Q-values.

HU008 puede añadir a `DQNAgent` un método pequeño y controlado, por ejemplo:

```python
def mean_q_value(self, states: np.ndarray | Tensor) -> float:
    ...
```

O ampliar `DQNUpdateResult` para devolver una métrica calculada durante el update, por ejemplo:

```python
@dataclass(frozen=True)
class DQNUpdateResult:
    loss: float
    q_value_mean: float
```

Preferir reutilizar el forward ya realizado durante `update()` para no añadir cómputo innecesario.

No modificar la fórmula DQN ni el comportamiento de aprendizaje.

### DT05 — Reward media móvil

La media móvil debe ser explícita y configurable.

Ejemplo:

```yaml
tensorboard:
  reward_window: 10
```

Semántica recomendada:

```text
mean(last min(window, completed_episodes) rewards)
```

No usar episodios incompletos.

### DT06 — Frecuencias de logging

Centralizar en config valores equivalentes a:

```yaml
tensorboard:
  enabled: true
  log_dir: "3_BattleZone/logs"
  scalar_log_interval_steps: 4
  reward_window: 10
  flush_interval_steps: 64
```

Los valores son baseline de implementación, no óptimos.

Evitar logging por frame si no aporta valor y genera overhead excesivo.

### DT07 — Global step continuo en resume

Todo scalar por timestep debe usar el `global_step` real del trainer.

Ejemplo:

```text
checkpoint global_step = 32
resume
next TensorBoard scalar step > 32
```

Está prohibido reiniciar TensorBoard a step 0 durante RESUME.

### DT08 — Política de log directory

HU008 debe admitir una ruta explícita de logs.

No debe seleccionar automáticamente directorios ambiguos.

Puede soportar:

- NEW: nuevo directorio de logs;
- RESUME: continuar en el mismo directorio si el caller lo indica explícitamente.

No implementar todavía `run_id`/manifest completo de HU010.

Puede usarse un identificador simple controlado en tests, pero no construir toda la trazabilidad experimental.

### DT09 — Idempotencia de logs

Reejecutar tests o validaciones no debe destruir logs previos por defecto.

En tests usar `tmp_path`.

No borrar recursivamente `logs/`.

No sobrescribir archivos event existentes deliberadamente.

### DT10 — Lifecycle del writer

El writer debe:

- crearse explícitamente;
- hacer `flush()` cuando corresponda;
- cerrarse explícitamente;
- soportar uso como context manager o `close()` idempotente si aporta claridad.

El cierre debe ocurrir incluso si la corrida termina normalmente.

No dejar recursos abiertos en tests.

### DT11 — No contaminar lógica de checkpoint

HU007 permanece como fuente de verdad de persistencia.

HU008 no debe:

- modificar schema de checkpoint salvo necesidad mínima y justificada;
- persistir TensorBoard writer dentro del checkpoint;
- guardar event files dentro de `.pt`;
- alterar full/lightweight semantics.

La continuidad de TensorBoard se basa en `global_step` restaurado y ruta de logs explícita.

### DT12 — Sin inferencias de performance

La presencia de curvas TensorBoard no demuestra aprendizaje.

HU008 solo puede afirmar:

> Las métricas necesarias para observar y diagnosticar el entrenamiento DQN se generan correctamente y pueden ser consumidas por TensorBoard.

## 5. Alcance funcional esperado

### 5.1 `src/callbacks.py`

Implementar una clase o contrato equivalente a:

```python
class TensorBoardTrainingLogger:
    def on_training_start(...): ...
    def on_step(...): ...
    def on_update(...): ...
    def on_episode_end(...): ...
    def on_training_end(...): ...
    def close(...): ...
```

No es obligatorio usar exactamente estos nombres si una API menor es más simple.

Debe encapsular `SummaryWriter`.

### 5.2 Integración con trainer

`DQNTrainer` debe aceptar opcionalmente observabilidad, por ejemplo:

```python
logger: TrainingLogger | None = None
```

El trainer debe seguir funcionando sin TensorBoard:

```text
logger=None → comportamiento HU006/HU007 sin cambios
```

Esto es obligatorio para no convertir TensorBoard en dependencia rígida de todos los tests.

### 5.3 Métricas de update

El update DQN debe exponer al menos:

```text
loss
q_value_mean
```

sin duplicar forward innecesariamente.

### 5.4 Episodios

Al finalizar un episodio, registrar:

```text
episode_reward
episode_reward_mean
episode_length
```

usando el `global_step` del final del episodio como x-axis recomendado.

### 5.5 Epsilon y Replay

Registrar periódicamente según `scalar_log_interval_steps`:

```text
epsilon
replay_size
```

No registrar cada step si la frecuencia configurada es mayor que 1.

## 6. Configuración

Actualizar `3_BattleZone/configs/battlezone_config.yaml` con una sección equivalente a:

```yaml
tensorboard:
  enabled: true
  baseline_note: "baseline de implementacion por validar"
  log_dir: "3_BattleZone/logs"
  scalar_log_interval_steps: 4
  reward_window: 10
  flush_interval_steps: 64
```

Si se requiere un parámetro adicional pequeño, documentarlo.

No introducir rutas absolutas locales.

## 7. Tags TensorBoard

Usar nombres estables y documentados.

Mínimo recomendado:

```text
train/episode_reward
train/episode_reward_mean
train/episode_length
train/loss
train/epsilon
train/q_value_mean
train/replay_size
train/learning_rate
```

No cambiar tags entre NEW/RESUME.

No mezclar métricas de evaluación formal dentro de `train/*`.

## 8. Semántica temporal

### 8.1 Por update

Cuando haya optimizer step en `global_step=N`:

```text
train/loss @ step N
train/q_value_mean @ step N
```

### 8.2 Por episodio

Cuando termina episodio en `global_step=N`:

```text
train/episode_reward @ step N
train/episode_reward_mean @ step N
train/episode_length @ step N
```

### 8.3 Epsilon/replay

Cuando el step cumpla frecuencia:

```text
train/epsilon @ global_step
train/replay_size @ global_step
```

## 9. Resume

Validar como mínimo:

```text
NEW:
  log step 4,8,...,32
  checkpoint at 32

RESUME:
  restore global_step=32
  same or explicit log_dir
  next scalars use steps 36,40,...
```

No debe aparecer una nueva serie reiniciada en 0 por error de trainer.

No se exige garantizar continuidad visual perfecta si TensorBoard ordena múltiples event files del mismo directorio; sí se exige coherencia del step.

## 10. TensorBoard event files

La validación debe demostrar que:

- se crea al menos un `events.out.tfevents.*`;
- su tamaño es > 0;
- TensorBoard/EventAccumulator puede leer los scalars esperados.

Preferir usar en tests:

```python
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
```

para verificar tags/steps/values sin levantar un servidor web.

## 11. Tests obligatorios

Crear preferiblemente:

```text
3_BattleZone/tests/test_callbacks.py
```

y actualizar tests de `agent.py`/`trainer.py` solo cuando corresponda.

Cubrir como mínimo:

1. logger crea directorio;
2. writer produce event file;
3. tags mínimos existen;
4. episode reward se registra;
5. moving average correcto;
6. episode length correcto;
7. loss correcto;
8. q_value_mean finito;
9. epsilon correcto;
10. replay_size correcto;
11. learning_rate correcto;
12. scalar interval respetado;
13. flush interval o flush explícito funciona;
14. close funciona;
15. logger=None no rompe trainer;
16. NEW usa steps correctos;
17. RESUME no reinicia step;
18. full resume + TensorBoard conserva step;
19. lightweight resume + TensorBoard conserva step;
20. event accumulator lee valores;
21. no MLflow;
22. no Assault;
23. no evaluación formal;
24. no manifest HU010.

## 12. Validación integrada obligatoria

Además de tests unitarios, ejecutar una corrida corta real o controlada con BattleZone que produzca eventos TensorBoard.

Preferible:

```text
ALE/BattleZone-v5
NEW hasta N pequeño
→ TensorBoard logs
→ guardar checkpoint FULL
→ recrear agent/trainer/logger
→ restore
→ continuar hasta M>N
→ verificar eventos con steps > N
```

No es necesario completar un episodio real si la corrida corta no alcanza uno; sin embargo, los tests con fake env sí deben demostrar métricas de episodio.

Registrar como mínimo:

```text
N
M
log_dir
event_files
tags
loss_count
q_value_count
epsilon_count
max_logged_step
resume_step_continuity
```

## 13. Evidencia requerida

Crear:

```text
3_BattleZone/docs/hu008_evidencia_implementacion.md
```

Debe contener únicamente resultados reales:

- rama;
- SHA;
- archivos modificados;
- configuración TensorBoard;
- tags registrados;
- comandos ejecutados;
- tests y resultados;
- ruta temporal/controlada usada;
- event files creados;
- tamaño si se mide;
- scalar counts;
- steps mínimo/máximo;
- NEW/RESUME continuity;
- CA01–CA16;
- AV01–AV16;
- limitaciones.

No versionar event files de pruebas.

## 14. Criterios de aceptación

- **CA01:** HU007 está mergeada y resume es la base vigente.
- **CA02:** observabilidad vive separada del algoritmo, preferiblemente en `callbacks.py`.
- **CA03:** TensorBoard puede deshabilitarse sin romper trainer.
- **CA04:** se generan event files válidos.
- **CA05:** se registran reward, media móvil y episode length.
- **CA06:** se registran loss y q_value_mean por update.
- **CA07:** se registran epsilon y replay_size con frecuencia configurable.
- **CA08:** learning rate está disponible como métrica cuando corresponda.
- **CA09:** tags son estables y documentados.
- **CA10:** scalars usan global_step correcto.
- **CA11:** RESUME no reinicia step de TensorBoard.
- **CA12:** EventAccumulator puede leer los scalars.
- **CA13:** writer hace flush/close correctamente.
- **CA14:** tests específicos de observabilidad pasan.
- **CA15:** suite BattleZone completa permanece verde.
- **CA16:** alcance preservado: sin MLflow, Assault, manifests, evaluación formal, tuning ni entrenamiento largo.

## 15. Autovalidaciones

- **AV01:** dependencias HU006/HU007 presentes en `main`.
- **AV02:** `tensorboard` config centralizada.
- **AV03:** `SummaryWriter` encapsulado fuera del trainer tanto como sea razonable.
- **AV04:** logger opcional.
- **AV05:** episode metrics correctas.
- **AV06:** moving average correcto.
- **AV07:** update metrics correctas.
- **AV08:** q_value_mean finito y proveniente del update/agente.
- **AV09:** frecuencia de epsilon/replay respetada.
- **AV10:** tags correctos.
- **AV11:** event files válidos.
- **AV12:** EventAccumulator lee scalars.
- **AV13:** NEW step sequence correcta.
- **AV14:** RESUME step sequence continúa desde checkpoint.
- **AV15:** regresión y suite completa PASS.
- **AV16:** scope sin Assault/MLflow/HU009+.

## 16. Definition of Done

HU008 puede cerrarse únicamente si:

1. existe implementación TensorBoard reusable;
2. config está centralizada;
3. trainer funciona con y sin logger;
4. event files reales son producidos;
5. tags obligatorios están presentes cuando aplican;
6. tests verifican valores y steps, no solo existencia de archivos;
7. resume conserva global_step en logs;
8. suite BattleZone completa pasa;
9. evidencia real queda documentada;
10. auditoría confirma alcance;
11. PR se mergea a `main`.

## 17. Fuera de alcance

No implementar en HU008:

- `run_manifest.json`;
- `run_id` definitivo de HU010;
- comparación formal de experimentos;
- evaluator;
- evaluación ≥10 episodios;
- video;
- selección de mejor modelo;
- entrenamiento largo;
- tuning;
- PER;
- DDQN;
- REINFORCE;
- MLflow;
- cambios de preprocessing;
- cambios de recompensa;
- cambios de arquitectura CNN salvo una corrección independiente y justificada.

## 18. Riesgos

### R01 — Overhead por logging excesivo

Mitigación: logging interval configurable y métricas por evento relevante.

### R02 — Duplicar forward para Q-values

Mitigación: obtener `q_value_mean` del forward ya ejecutado durante update.

### R03 — Step inconsistente después de resume

Mitigación: usar exclusivamente `TrainingState.global_step` como step TensorBoard.

### R04 — Acoplar trainer a TensorBoard

Mitigación: logger/callback opcional con interfaz pequeña.

### R05 — Confundir logs con trazabilidad completa

Mitigación: HU010 sigue siendo responsable de run_id/manifests y metadata completa.

## 19. Resultado esperado

Al finalizar HU008, BattleZone contará con observabilidad suficiente para ejecutar HU009 y diagnosticar de forma objetiva si el pipeline completo produce métricas de entrenamiento coherentes antes de gastar cómputo en Colab GPU.