# HU011B — Entregables técnicos post-entrenamiento: modelo, gráficas y videos

## 1. Propósito

Completar los artefactos técnicos obligatorios de entrega que deben derivarse del entrenamiento real de BattleZone antes de iniciar optimización sistemática o evaluación académica final.

HU011 produce la corrida larga, checkpoints, TensorBoard, manifiesto y un estado final recuperable del entrenamiento. HU011B debe convertir esa corrida en artefactos **entregables, reproducibles y verificables de forma autónoma**:

1. un modelo DQN compacto de inferencia que el profesor pueda cargar y ejecutar sin Replay Buffer ni optimizer;
2. gráficas reproducibles derivadas de la evidencia real de entrenamiento;
3. un video corto que evidencie el proceso de entrenamiento mediante un checkpoint intermedio real;
4. un video corto que evidencie el comportamiento del agente entrenado en modo de explotación;
5. integración explícita de estos artefactos en `pipeline_battlezone.ipynb`.

La HU existe para cerrar la brecha entre "el agente fue entrenado" y "el agente puede ser entregado y verificado por un tercero".

---

## 2. Justificación académica

El enunciado del Reto 1 exige para cada problema:

- notebook ejecutable en Google Colab;
- modelo entrenado del agente para verificar correcta ejecución y comportamiento aprendido;
- video corto con evidencia del proceso de entrenamiento;
- video corto/evidencia del comportamiento aprendido una vez entrenado;
- gráficas de evolución de recompensa durante entrenamiento y explotación;
- análisis posterior sustentado en evidencia cuantitativa.

HU011 declaró fuera de alcance la generación de video final y no define todavía un artefacto compacto, autónomo y local-first equivalente al entregable requerido para revisión del profesor.

HU011B corrige esa brecha sin adelantar HU012 ni sustituir HU013.

---

## 3. Referencia metodológica de Assault

`2_Assault/` puede consultarse únicamente como referencia metodológica.

El patrón útil ya probado en Assault incluye responsabilidades separadas para:

- exportar un modelo compacto de inferencia desde un checkpoint;
- validar checksum, metadata y contrato de preprocessing;
- cargar el modelo en una instancia nueva del agente;
- resolver primero un modelo incluido en la entrega y usar almacenamiento persistente solo como fallback;
- generar videos MP4 reproducibles con metadata del run/modelo;
- producir un video de entrenamiento/exploración y otro de explotación;
- reconstruir gráficas desde TensorBoard;
- validar por tests que notebook, modelo, videos y gráficas estén conectados.

**Prohibido importar o depender operativamente de `2_Assault/`.**

BattleZone debe tener módulos propios, nombres propios, contratos propios y tests propios.

---

## 4. Posición en el roadmap

HU011B se ejecuta después de HU011 y antes de HU012:

```text
HU011  Entrenamiento completo DQN
  ↓
HU011B Entregables técnicos: modelo + gráficas + videos
  ↓
HU012  Optimización controlada de hiperparámetros
  ↓
HU013  Evaluación formal contra baseline (>=10 episodios)
  ↓
HU014  Reporte técnico y entrega final
```

La implementación de código de HU011B puede desarrollarse y probarse con fixtures/checkpoints controlados antes de que finalice la corrida de 1.000.000 steps. Sin embargo, HU011B solo puede marcarse `[COMPLETADA]` cuando los artefactos finales se hayan generado desde una corrida real elegible y se haya validado su linaje.

---

## 5. Dependencias

HU011B debe apoyarse en los contratos existentes de:

- `3_BattleZone/src/agent.py`;
- `3_BattleZone/src/network.py`;
- `3_BattleZone/src/environment.py`;
- `3_BattleZone/src/persistence.py`;
- `3_BattleZone/src/callbacks.py`;
- `3_BattleZone/src/experiment.py`;
- `3_BattleZone/src/training_run.py`;
- `3_BattleZone/configs/battlezone_config.yaml`;
- `3_BattleZone/docs/hu008_observabilidad_tensorboard.md`;
- `3_BattleZone/docs/hu010_trazabilidad_ligera_experimentos.md`;
- `3_BattleZone/docs/hu011_entrenamiento_completo_dqn.md`.

No crear un segundo sistema de checkpoints ni duplicar lógica de entrenamiento.

---

## 6. Fuera de alcance

HU011B NO debe:

- modificar el algoritmo DQN;
- introducir DDQN, PER o REINFORCE;
- cambiar hiperparámetros para mejorar desempeño;
- ejecutar grid/random/Bayesian search;
- declarar que el agente resolvió formalmente BattleZone;
- realizar la comparación académica definitiva contra HU002;
- sustituir la evaluación formal de al menos 10 episodios de HU013;
- seleccionar el mejor modelo entre múltiples configuraciones;
- redactar el reporte final completo de HU014;
- introducir MLflow, W&B o Neptune;
- depender de código de Assault.

Una demostración greedy usada para video o smoke de entrega **no equivale a la evaluación formal HU013**.

---

# PARTE A — MODELO ENTRENADO ENTREGABLE

## 7. Problema a resolver

El checkpoint de entrenamiento contiene información destinada a continuar aprendiendo y puede incluir:

- online network;
- target network;
- optimizer;
- Replay Buffer;
- estado del trainer;
- metadata operativa.

Ese artefacto no debe ser el contrato que reciba el profesor para ejecutar el agente.

HU011B debe crear un **artefacto compacto de inferencia**.

---

## 8. Nombre canónico del modelo

El modelo entregable debe llamarse:

```text
battlezone_dqn_model.pt
```

Ruta local-first de entrega:

```text
3_BattleZone/battlezone_dqn_model.pt
```

Ruta persistente de respaldo durante la generación:

```text
<PERSISTENT_ROOT>/models/<run_id>/battlezone_dqn_model.pt
```

El notebook debe resolver el modelo en este orden:

```text
1. 3_BattleZone/battlezone_dqn_model.pt
2. <PERSISTENT_ROOT>/models/<run_id>/battlezone_dqn_model.pt
```

La evaluación del profesor no debe depender obligatoriamente de tener acceso al Google Drive del equipo.

---

## 9. Contenido permitido del modelo de inferencia

El artefacto debe contener solo lo necesario para recrear el agente y ejecutar inferencia.

Contenido esperado:

```text
schema_version
created_at
online_network weights
network architecture metadata
action_dim
state_shape
preprocessing contract
environment contract
algorithm = DQN
project_run_id
source checkpoint step
source checkpoint SHA256
Git commit/config fingerprint relevantes
seed relevante
```

No debe incluir:

```text
Replay Buffer
optimizer state
target network como requisito de inferencia
training batches
TensorBoard events
estado mutable del trainer
```

Puede recrearse Target Network al cargar el agente si la clase actual lo requiere internamente, pero sus pesos no deben duplicarse en el artefacto entregable si no son necesarios para seleccionar acciones.

---

## 10. Módulo de artefacto recomendado

Crear preferiblemente:

```text
3_BattleZone/src/model_artifact.py
```

API mínima esperada:

```python
export_inference_model(...)
load_inference_model(...)
resolve_delivery_model_path(...)
compute_sha256(...)
```

Los nombres exactos pueden ajustarse justificadamente, pero las responsabilidades deben permanecer separadas del trainer.

---

## 11. Exportación

`export_inference_model(...)` debe:

1. recibir explícitamente el checkpoint/final state fuente;
2. comprobar que corresponde a DQN BattleZone;
3. comprobar compatibilidad con el contrato vigente del entorno y red;
4. extraer la online network;
5. incluir metadata de linaje;
6. guardar de forma atómica;
7. calcular SHA256;
8. producir sidecar de checksum;
9. producir metadata JSON legible;
10. comprobar que el artefacto puede volver a cargarse.

No exportar un modelo desde un checkpoint incompatible solo porque `torch.load()` funcione.

---

## 12. Linaje mínimo

El metadata del modelo debe permitir responder:

```text
¿qué corrida produjo este modelo?
¿de qué checkpoint proviene?
¿en qué global_step?
¿qué Git commit/configuración produjo la corrida?
¿qué preprocessing necesita?
¿qué environment/action space necesita?
¿cuál es el SHA256 del modelo?
```

Mínimo:

```text
project_run_id
source_checkpoint_step
source_checkpoint_path/identity
source_checkpoint_sha256
model_sha256
environment id
frameskip
repeat_action_probability
observation shape/dtype
frame_stack
action_dim
algorithm
network architecture/config
```

---

## 13. Tamaño del modelo

El artefacto debe ser compacto y apto para entrega académica.

Objetivo:

```text
< 100 MiB
```

No introducir Git LFS automáticamente.

Si el archivo supera el límite, primero verificar que realmente se esté exportando **solo la online network y metadata mínima**.

HU011B no puede declararse completada dejando como única opción un checkpoint gigantesco con Replay/optimizer.

---

## 14. Carga autónoma por el profesor

`load_inference_model(...)` debe:

1. cargar el archivo con `map_location` configurable;
2. funcionar por defecto en CPU cuando CUDA no esté disponible;
3. reconstruir un `DQNAgent` compatible;
4. cargar estrictamente la online network;
5. colocar la red en modo evaluación;
6. no crear dependencia de Replay para seleccionar acciones;
7. verificar checksum opcional;
8. verificar `project_run_id` opcional;
9. fallar ante contrato incompatible.

El profesor debe poder abrir el notebook, instalar dependencias, cargar `battlezone_dqn_model.pt` y ejecutar el agente **sin reentrenar**.

---

## 15. Gate autónomo de entrega

El notebook debe contener una sección independiente del entrenamiento llamada conceptualmente:

```text
VERIFICACIÓN AUTÓNOMA DEL MODELO ENTREGADO
```

Debe demostrar:

```text
modelo localizado
checksum válido
modelo cargado
entorno creado
observación compatible
acción greedy producida
al menos una ejecución controlada del agente sin optimizer/replay/training
```

No depender de variables creadas horas antes en otras celdas.

Idealmente la sección debe poder ejecutarse desde un runtime limpio después del bootstrap/dependencias.

---

# PARTE B — GRÁFICAS

## 16. Fuente de verdad de gráficas

Las gráficas no deben construirse desde valores escritos manualmente en el notebook.

Fuente principal:

```text
TensorBoard logs del mismo run_id
```

Tags existentes relevantes de BattleZone:

```text
train/episode_reward
train/episode_reward_mean
train/loss
train/epsilon
train/q_value_mean
train/replay_size
train/learning_rate
train/episode_length
```

---

## 17. Módulo recomendado de reporting

Crear preferiblemente:

```text
3_BattleZone/src/reporting.py
```

Responsabilidades:

```text
cargar scalars TensorBoard
validar tags requeridos
normalizar global_step
calcular media móvil cuando aplique
preparar specs de figuras
renderizar figuras sin acoplarse al entrenamiento
```

---

## 18. Figuras mínimas no redundantes

HU011B debe producir como mínimo estas figuras de entrenamiento:

### Figura 1 — Recompensa

```text
train/episode_reward
+
train/episode_reward_mean
```

Eje X:

```text
global_step
```

### Figura 2 — Loss

```text
train/loss
+
opcional media móvil
```

### Figura 3 — Q-value medio y epsilon

```text
train/q_value_mean
train/epsilon
```

Puede utilizar doble eje Y cuando mejore legibilidad.

Evitar gráficas redundantes que no ayuden al análisis.

---

## 19. Recompensa durante explotación

El enunciado también exige evidencia de recompensa durante explotación.

HU011B debe implementar la capacidad de graficar rewards obtenidos por una política de explotación (`epsilon=0.0`) a partir de una lista estructurada de episodios/seeds.

Sin embargo:

- los episodios de video/sanity de HU011B no sustituyen HU013;
- la gráfica académica final de explotación debe alimentarse posteriormente con los >=10 episodios formales de HU013;
- HU011B debe dejar la API y la sección de notebook preparadas para consumir esos resultados sin refactor adicional.

Puede generarse una gráfica preliminar etiquetada claramente como `DELIVERY_SANITY_ONLY`, pero nunca presentarla como evaluación final.

---

## 20. Persistencia de figuras

Las figuras deben:

- mostrarse inline en notebook;
- poder guardarse como PNG;
- quedar asociadas al mismo `run_id`.

Ruta sugerida:

```text
<PERSISTENT_ROOT>/delivery/<run_id>/figures/
```

Nombres sugeridos:

```text
training_reward.png
training_loss.png
training_q_epsilon.png
exploitation_reward.png   # finalizada con HU013
```

---

# PARTE C — VIDEO DEL PROCESO DE ENTRENAMIENTO

## 21. Principio

El video de entrenamiento debe evidenciar un **estado intermedio real del agente**, no simplemente reproducir el modelo final con epsilon alto y llamarlo entrenamiento.

Fuente preferida:

```text
checkpoint periódico del mismo run_id con global_step < final_global_step
```

Ejemplos válidos según disponibilidad:

```text
25_000
250_000
500_000
750_000
```

No hardcodear un step que pueda no existir. La selección debe ser explícita y trazable.

---

## 22. Generación del video de entrenamiento

Crear preferiblemente:

```text
3_BattleZone/src/video.py
```

Debe existir una API equivalente conceptualmente a:

```python
generate_training_process_demo_video(...)
```

Debe:

1. cargar pesos de un checkpoint intermedio real;
2. crear env con `render_mode="rgb_array"` manteniendo contrato BattleZone;
3. ejecutar una demostración corta;
4. usar epsilon coherente con ese checkpoint o valor explícito documentado;
5. producir MP4;
6. incluir overlay/intro con `run_id`, checkpoint step y epsilon;
7. producir metadata JSON;
8. no mutar los parámetros de la red durante la generación.

---

## 23. Semántica del video de entrenamiento

Metadata mínima:

```text
video_kind = training_process
project_run_id
checkpoint_step
checkpoint_sha256
epsilon
seed
reward
steps
fps
```

Debe quedar inequívocamente identificado como:

```text
EVIDENCIA DEL PROCESO DE ENTRENAMIENTO
```

No declarar que representa "antes de entrenar" si en realidad se usa un checkpoint intermedio.

---

# PARTE D — VIDEO POST-ENTRENAMIENTO

## 24. Video del comportamiento aprendido

Debe generarse desde el **modelo entregable `battlezone_dqn_model.pt`**, no desde una instancia del agente que quedó viva en memoria después del entrenamiento.

Esto prueba simultáneamente:

- que el artefacto entregable es cargable;
- que el preprocessing es compatible;
- que el agente puede ejecutar acciones;
- que el video corresponde al modelo que recibe el profesor.

---

## 25. Política del video final

Para el video post-entrenamiento:

```text
epsilon = 0.0
```

Usar seed explícito y reproducible.

No seleccionar manualmente acciones.

No modificar rewards.

No alterar el entorno respecto del contrato de evaluación salvo `render_mode="rgb_array"`.

---

## 26. Metadata del video post-entrenamiento

Mínimo:

```text
video_kind = post_training_exploitation
project_run_id
model_sha256
seed
epsilon = 0.0
reward
steps
fps
environment contract
```

Nombre sugerido:

```text
battlezone_dqn_post_training.mp4
```

Nombre sugerido del video de entrenamiento:

```text
battlezone_dqn_training_process.mp4
```

Ruta persistente sugerida:

```text
<PERSISTENT_ROOT>/delivery/<run_id>/videos/
```

---

## 27. Requisitos técnicos de video

Los videos deben:

- ser MP4 reproducibles;
- usar frames RGB reales del entorno;
- tener tamaño no vacío;
- cerrarse correctamente aun ante excepciones;
- no alterar los pesos del agente;
- incluir metadata sidecar JSON;
- poder mostrarse inline en Colab mediante `IPython.display.Video`.

Añadir `imageio`/ffmpeg solo si es estrictamente necesario y de manera compatible con Colab.

---

# PARTE E — NOTEBOOK

## 28. Sección HU011B en `pipeline_battlezone.ipynb`

Agregar una sección claramente separada del entrenamiento largo:

```text
HU011B — ARTEFACTOS DE ENTREGA
```

Orden esperado:

```text
1. Resolver run_id elegible
2. Verificar manifest/checkpoint fuente
3. Exportar battlezone_dqn_model.pt
4. Verificar checksum y recarga autónoma
5. Generar gráficas TensorBoard
6. Seleccionar checkpoint intermedio explícito
7. Generar video de proceso de entrenamiento
8. Cargar nuevamente battlezone_dqn_model.pt
9. Generar video post-entrenamiento epsilon=0
10. Mostrar videos inline
11. Mostrar/guardar figuras
12. Ejecutar artifact delivery gate
```

No requerir volver a entrenar para regenerar figuras/videos si los artefactos fuente ya existen.

---

## 29. Delivery gate

Crear un gate estructurado que devuelva PASS únicamente si existe evidencia suficiente.

Ejemplo conceptual:

```text
HU011B_DELIVERY_GATE

model_exists = True
model_loadable = True
model_sha256_valid = True
model_lineage_valid = True
training_figures_ready = True
training_video_exists = True
training_video_metadata_valid = True
post_training_video_exists = True
post_training_video_metadata_valid = True
professor_smoke_pass = True
```

Resultado:

```text
HU011B_DELIVERY_GATE=PASS
```

No imprimir PASS si falta un archivo obligatorio.

---

# PARTE F — TESTS

## 30. Tests del modelo

Crear preferiblemente:

```text
3_BattleZone/tests/test_model_artifact.py
```

Cubrir como mínimo:

1. exportación desde checkpoint compatible;
2. rechazo de checkpoint incompatible;
3. online network incluida;
4. Replay excluido;
5. optimizer excluido;
6. target network no requerida en el payload entregable;
7. metadata mínima presente;
8. checksum estable;
9. corrupción/checksum incorrecto falla;
10. load recrea DQNAgent;
11. pesos recargados equivalen al origen;
12. inferencia CPU válida;
13. contrato environment/preprocessing incompatible falla;
14. resolución local-first del modelo;
15. Drive fallback solo si modelo de entrega local no existe;
16. modelo faltante falla con rutas buscadas explícitas.

---

## 31. Tests de reporting

Crear preferiblemente:

```text
3_BattleZone/tests/test_reporting.py
```

Cubrir:

- tags requeridos;
- reward + moving average;
- loss;
- q_value_mean + epsilon;
- global_step ordenado;
- NaN/Inf rechazado;
- duplicados problemáticos rechazados;
- figura preliminar/final de explotación desde rewards estructurados;
- máximo razonable de figuras no redundantes.

---

## 32. Tests de video

Crear preferiblemente:

```text
3_BattleZone/tests/test_video.py
```

Cubrir:

- MP4 producido;
- metadata producido;
- seed/epsilon/run_id preservados;
- entrenamiento usa checkpoint intermedio explícito;
- video final usa epsilon 0.0;
- agent weights no mutan;
- render_mode inválido falla;
- frame inválido falla;
- writer se cierra ante excepción;
- salida vacía falla.

Usar mocks/fakes para unit tests; no requerir GPU ni episodios Atari largos.

---

## 33. Tests del notebook

Agregar tests estáticos/estructurales que garanticen:

- importa módulos propios BattleZone;
- exporta exactamente un delivery model final en el flujo;
- no depende de `2_Assault`;
- muestra gráficas;
- genera ambos videos;
- muestra ambos videos inline;
- el video final carga desde el artifact entregable;
- existe sección autónoma para el profesor;
- existe `HU011B_DELIVERY_GATE`;
- no ejecuta entrenamiento de 1M como efecto lateral al generar entregables.

---

# PARTE G — VALIDACIÓN E2E BARATA

## 34. Validación antes de usar artefactos reales

Codex debe validar localmente con checkpoints/configs controlados:

```text
checkpoint fake/pequeño compatible
  ↓
export model
  ↓
load model
  ↓
select_action
  ↓
TensorBoard scalars controlados
  ↓
figures
  ↓
fake/render env
  ↓
training MP4
  ↓
post-training MP4
  ↓
delivery gate
```

No ejecutar el entrenamiento real de 1.000.000 steps como parte de la implementación de esta HU.

---

## 35. Validación con corrida real

Cuando HU011 disponga de una corrida elegible:

```text
manifest real
checkpoint intermedio real
checkpoint/final source real
TensorBoard real
  ↓
export modelo real
  ↓
video entrenamiento real
  ↓
video explotación real
  ↓
gráficas reales
  ↓
delivery gate PASS
```

Registrar evidencia en:

```text
3_BattleZone/docs/hu011b_evidencia_implementacion.md
```

---

# PARTE H — CRITERIOS DE ACEPTACIÓN

## 36. Criterios funcionales

- **CA01:** existe módulo propio de BattleZone para exportar/cargar modelo de inferencia.
- **CA02:** `battlezone_dqn_model.pt` contiene pesos suficientes para inferencia pero no Replay/optimizer.
- **CA03:** el modelo tiene checksum SHA256 y metadata de linaje.
- **CA04:** el modelo puede cargarse en CPU desde un runtime limpio.
- **CA05:** el profesor puede ejecutar el agente sin reentrenar y sin acceso obligatorio al Drive del equipo.
- **CA06:** el modelo local de entrega tiene prioridad sobre el fallback persistente.
- **CA07:** se generan gráficas de entrenamiento desde TensorBoard real.
- **CA08:** existe API para graficar rewards de explotación y queda preparada para los >=10 episodios de HU013.
- **CA09:** se genera video MP4 del proceso de entrenamiento desde un checkpoint intermedio real.
- **CA10:** se genera video MP4 post-entrenamiento desde `battlezone_dqn_model.pt` con epsilon=0.0.
- **CA11:** ambos videos contienen metadata de run/modelo/seed/política.
- **CA12:** videos y gráficas se muestran en el notebook.
- **CA13:** los artefactos persistentes se guardan fuera de `/content` durante Colab.
- **CA14:** GitHub sigue siendo source of truth del código.
- **CA15:** `HU011B_DELIVERY_GATE=PASS` solo cuando todos los artefactos obligatorios son válidos.

## 37. Criterios de calidad

- **CA16:** no hay dependencia operativa con `2_Assault/`.
- **CA17:** no se introduce MLflow.
- **CA18:** no se modifica DQN para "mejorar" resultados en esta HU.
- **CA19:** no se adelanta evaluación formal HU013.
- **CA20:** unit tests no requieren GPU.
- **CA21:** suite BattleZone existente continúa verde.
- **CA22:** notebook no requiere volver a entrenar para verificar un modelo ya entregado.
- **CA23:** los archivos derivados no se presentan como válidos si su linaje no coincide con el run/modelo esperado.

---

# PARTE I — DEFINICIÓN DE TERMINADO

## 38. Estado de implementación

Mientras solo exista código/tests sin artefactos reales:

```text
HU011B IMPLEMENTADA — ARTEFACTOS REALES PENDIENTES
```

HU011B solo puede marcarse:

```text
[COMPLETADA]
```

cuando exista evidencia real de:

```text
battlezone_dqn_model.pt cargable
SHA256/metadata válidos
training figures desde logs reales
training-process MP4 desde checkpoint intermedio real
post-training MP4 desde delivery model real
professor autonomous smoke PASS
HU011B_DELIVERY_GATE=PASS
```

La evaluación formal de >=10 episodios sigue pendiente hasta HU013.

---

# PARTE J — INSTRUCCIONES PARA CODEX

## 39. Flujo Git

Codex debe implementar HU011B en una rama nueva creada desde `main` actualizado.

Nombre sugerido:

```text
feature/battlezone-hu011b-delivery-artifacts
```

Debe abrir un PR nuevo contra `main`.

No hacer merge automático.

No modificar Assault.

---

## 40. Archivos esperados

Principalmente:

```text
3_BattleZone/src/model_artifact.py
3_BattleZone/src/reporting.py
3_BattleZone/src/video.py
3_BattleZone/pipeline_battlezone.ipynb
3_BattleZone/tests/test_model_artifact.py
3_BattleZone/tests/test_reporting.py
3_BattleZone/tests/test_video.py
3_BattleZone/tests/test_notebook_hu011b.py
3_BattleZone/docs/hu011b_evidencia_implementacion.md
```

Cambios adicionales deben justificarse.

No añadir binarios falsos ni artefactos sintéticos al repositorio como si fueran el modelo final.

---

## 41. Validación Codex obligatoria

Ejecutar al menos:

```bash
python -m compileall -q 3_BattleZone/src 3_BattleZone/tests
PYTHONPATH=3_BattleZone python -m pytest 3_BattleZone/tests/test_model_artifact.py -q
PYTHONPATH=3_BattleZone python -m pytest 3_BattleZone/tests/test_reporting.py -q
PYTHONPATH=3_BattleZone python -m pytest 3_BattleZone/tests/test_video.py -q
PYTHONPATH=3_BattleZone python -m pytest 3_BattleZone/tests/test_notebook_hu011b.py -q
PYTHONPATH=3_BattleZone python -m pytest 3_BattleZone/tests -q
```

Criterio:

```text
0 failed
```

Además:

```bash
git diff --check
git diff --name-only origin/main...HEAD
```

---

## 42. STOP CONDITIONS

Codex debe detenerse si:

- necesita modificar `2_Assault/` para que BattleZone funcione;
- el modelo no puede cargarse sin estado de entrenamiento;
- el artefacto compacto continúa superando límites razonables por incluir estado innecesario;
- el video de entrenamiento no puede vincularse a un checkpoint intermedio real;
- los TensorBoard logs no contienen los tags mínimos necesarios;
- el modelo/video/reporting alteran el entrenamiento estable existente;
- la suite BattleZone presenta regresiones;
- el notebook solo funciona si el Drive privado del equipo está montado para cargar el modelo entregado.

---

## 43. Resultado esperado

Al cerrar HU011B, BattleZone debe poder demostrar esta cadena:

```text
HU011 real run
   ↓
checkpoint intermedio ─────────────→ training-process video
   ↓
final trained state
   ↓
compact inference export
   ↓
battlezone_dqn_model.pt
   ├──→ autonomous professor load/run
   ├──→ post-training exploitation video
   └──→ checksum + metadata

TensorBoard logs
   └──→ training figures

HU013 results (posterior)
   └──→ formal exploitation reward figure >=10 episodes
```

HU011B no busca demostrar que DQN sea óptimo. Busca asegurar que **el aprendizaje producido por HU011 pueda ser entregado, inspeccionado, reproducido y ejecutado por un tercero**, cumpliendo los artefactos técnicos exigidos por el reto.
