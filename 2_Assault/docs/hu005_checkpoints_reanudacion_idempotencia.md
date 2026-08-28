# HU005 — Checkpoints, reanudación e idempotencia

## 1. Identificación

- **ID:** HU005
- **Nombre:** Checkpoints, reanudación e idempotencia
- **Estado:** Lista para implementación
- **Dependencia previa:** HU004 — Ciclo de entrenamiento + Preflight `[COMPLETADA]`
- **Dependencias técnicas:** HU002/HU002B mantienen pendiente su validación formal en Colab, pero sus contratos locales de entorno/bootstrap continúan disponibles para el desarrollo controlado.
- **Habilita:** HU006 — Observabilidad con TensorBoard
- **Fuentes de verdad:**
  - `2_Assault/docs/implementacion.md`
  - `2_Assault/docs/arquitectura.md`
  - `2_Assault/docs/linemientos.md`
  - `2_Assault/docs/ficha_tecnica.md`
  - `2_Assault/docs/hu002_pipeline_reproducible_entorno.md`
  - `2_Assault/docs/hu002b_pipeline_ejecucion_local_github_colab.md`
  - `2_Assault/docs/hu003_nucleo_ddqn.md`
  - `2_Assault/docs/hu004_ciclo_entrenamiento.md`
  - `2_Assault/configs/ddqn_config.yaml`
  - `2_Assault/src/agent.py`
  - `2_Assault/src/replay_buffer.py`
  - `2_Assault/src/trainer.py`
  - `2_Assault/src/preflight.py`
  - `2_Assault/assault_ddqn.ipynb`
  - `enunciado_reto_1.txt`

---

## 2. Contexto y problema

HU004 dejó implementado un ciclo corto de entrenamiento DDQN controlado por `global_step` que integra:

- entorno `ALE/Assault-v5`;
- observaciones `(4, 84, 84)` en `uint8`;
- política epsilon-greedy;
- Replay Buffer uniforme;
- `learning_starts`;
- updates DDQN;
- sincronización periódica de Target Network;
- decay lineal de epsilon;
- métricas mínimas de entrenamiento;
- Preflight obligatorio antes de entrenar.

El siguiente riesgo del proyecto es **perder el progreso cuando termina una sesión de ejecución**, especialmente en Google Colab, donde `/content` es almacenamiento efímero y una sesión puede finalizar antes de completar un entrenamiento largo.

Actualmente un entrenamiento puede comenzar y modificar correctamente la Online Network, pero no existe todavía una forma completa y segura de responder a:

```text
¿En qué timestep iba?
¿Con qué epsilon debía continuar?
¿Qué optimizer state tenía?
¿Qué configuración produjo ese estado?
¿Qué Replay Buffer estaba utilizando?
¿Puedo continuar o debo empezar de cero?
¿Estoy sobrescribiendo un checkpoint válido?
```

HU005 debe convertir el entrenamiento corto de HU004 en un proceso **reanundable e idempotente**, sin implementar todavía observabilidad avanzada.

El flujo objetivo es:

```text
training session A
      ↓
checkpoint
      ↓
proceso/runtime termina
      ↓
nueva sesión
      ↓
selección explícita de checkpoint
      ↓
restore
      ↓
continuar desde global_step guardado
      ↓
training session B
```

---

## 3. Historia de usuario

> **Como** equipo que entrena un agente DDQN en sesiones potencialmente interrumpibles de Google Colab, **quiero** guardar y restaurar checkpoints completos o livianos de forma explícita e idempotente, **para** continuar el aprendizaje desde el timestep correcto sin perder silenciosamente progreso, configuración o estado del optimizer.

---

## 4. Objetivo verificable

Al finalizar HU005 el proyecto debe soportar explícitamente tres modos de ejecución:

```text
NEW
→ comienza desde global_step=0

RESUME_FULL
→ restaura agente + optimizer + progreso + Replay Buffer

RESUME_LIGHT
→ restaura agente + optimizer + progreso
→ Replay Buffer comienza vacío y se reconstruye
```

HU005 debe demostrar que:

1. un checkpoint guarda Online Network;
2. guarda Target Network;
3. guarda optimizer;
4. guarda `global_step`;
5. conserva información suficiente para reconstruir el epsilon correcto;
6. guarda la configuración asociada;
7. guarda métricas mínimas de continuidad;
8. guarda Replay Buffer en modo completo;
9. permite restaurar un checkpoint en un proceso/instancia nueva;
10. el trainer continúa desde el `global_step` restaurado;
11. no reinicia silenciosamente epsilon ni progreso;
12. `resume_full` recupera Replay Buffer;
13. `resume_light` deja explícitamente Replay Buffer vacío y exige reconstrucción antes de volver a actualizar;
14. una corrida `new` no sobrescribe accidentalmente un checkpoint previo;
15. un resume requiere seleccionar explícitamente el checkpoint;
16. no existe selección ambigua automática de “último checkpoint”;
17. el notebook deja claro si la ejecución es `new`, `resume_full` o `resume_light`;
18. las rutas de persistencia funcionan localmente y pueden apuntar a almacenamiento persistente en Colab;
19. los tests prueban guardar → destruir instancia → cargar → continuar;
20. no se implementan todavía TensorBoard, MLflow ni entrenamiento largo final.

Resultado esperado:

```text
train N steps
   ↓
save checkpoint @ N
   ↓
new Python objects
   ↓
load checkpoint
   ↓
resume from N
   ↓
train M more steps
   ↓
final global_step = N + M
```

---

## 5. Alcance

### 5.1 Estrategias obligatorias

HU005 debe soportar exactamente estas estrategias conceptuales:

#### `new`

Inicia una corrida nueva.

Condiciones:

- `global_step=0`;
- agente nuevo;
- optimizer nuevo;
- Replay Buffer vacío;
- no utiliza checkpoint existente;
- genera o recibe un `run_id` explícito;
- no debe sobrescribir silenciosamente artefactos de otra corrida.

#### `resume_full`

Restaura continuidad máxima.

Debe restaurar:

- Online Network;
- Target Network;
- optimizer;
- `global_step`;
- información necesaria para epsilon;
- configuración asociada al checkpoint;
- métricas mínimas acumuladas;
- Replay Buffer;
- cualquier estado adicional mínimo que sea realmente necesario para continuar correctamente.

#### `resume_light`

Restaura el agente y progreso, pero no el Replay Buffer.

Debe restaurar:

- Online Network;
- Target Network;
- optimizer;
- `global_step`;
- configuración;
- métricas mínimas;
- estado suficiente para continuar epsilon.

Debe iniciar:

```text
Replay Buffer size = 0
```

El trainer deberá reconstruir gradualmente el Replay Buffer antes de permitir nuevos updates.

No debe fingir que existe continuidad completa del buffer.

---

## 5.2 Checkpoint manager

Crear preferiblemente:

`2_Assault/src/checkpointing.py`

Responsabilidad principal:

- construir rutas seguras de checkpoints;
- serializar/deserializar estado de entrenamiento;
- validar compatibilidad básica;
- gestionar `new`, `resume_full` y `resume_light`;
- proteger contra sobrescritura accidental;
- devolver metadatos estructurados.

No convertir este módulo en un framework genérico de persistencia.

Interfaz conceptual permitida:

```python
manager = CheckpointManager(...)
manager.save(...)
state = manager.load(...)
```

Los nombres pueden adaptarse siempre que la responsabilidad se mantenga clara.

---

## 5.3 Contenido mínimo del checkpoint

Cada checkpoint debe incluir al menos:

```text
schema_version
run_id
created_at
checkpoint_step
git_commit
config
online_network
target_network
optimizer
global_step
epsilon_state / parámetros para reconstruir epsilon
training_metrics
resume_mode_capabilities
```

En `resume_full` debe incluir además:

```text
replay_buffer_state
```

No es necesario guardar objetos Gymnasium o el estado interno completo del emulador ALE en HU005.

El objetivo es continuar **el entrenamiento entre sesiones**, no pausar y reanudar en mitad exacta de un frame del mismo episodio.

Por diseño, una nueva sesión puede reiniciar el entorno/episodio mientras conserva el progreso de aprendizaje del agente.

---

## 5.4 Estado de epsilon

HU004 calcula epsilon de forma determinista usando:

- `epsilon_start`;
- `epsilon_final`;
- `epsilon_decay_steps`;
- `global_step`.

Por tanto HU005 no necesita guardar un scheduler complejo si epsilon puede reconstruirse exactamente desde:

```text
global_step + configuración
```

El checkpoint debe contener suficiente evidencia para reconstruirlo y validar que:

```python
epsilon_before_save == compute_epsilon(saved_global_step, ...)
```

al cargar.

No guardar información redundante si no aporta continuidad real.

---

## 5.5 Replay Buffer serializable

Extender `2_Assault/src/replay_buffer.py` únicamente con la capacidad mínima de exportar/importar su estado.

Puede utilizar una interfaz equivalente a:

```python
state_dict()
load_state_dict(...)
```

Debe conservar como mínimo:

- capacidad;
- state shape;
- tamaño actual;
- posición circular;
- estados válidos;
- next states válidos;
- acciones;
- rewards;
- dones;
- estado del RNG si es razonablemente simple y útil para reproducibilidad.

### Restricción de memoria

No serializar posiciones no utilizadas del buffer cuando esto multiplique innecesariamente el tamaño del checkpoint.

Debe evitarse copiar grandes bloques vacíos únicamente porque la capacidad máxima sea grande.

Los tests utilizarán capacidades pequeñas.

El dimensionamiento final del Replay Buffer para HU009 continúa fuera del alcance de HU005.

---

## 5.6 Integración con `DDQNAgent`

HU003 ya implementó `save/load` básico.

HU005 puede:

- reutilizar internamente `state_dict()` de redes/optimizer;
- ampliar el contrato de persistencia de forma compatible;
- mover la responsabilidad de checkpoint de entrenamiento completo a `checkpointing.py`.

No duplicar lógica de serialización innecesariamente entre `agent.py` y `checkpointing.py`.

El checkpoint completo pertenece al **estado de entrenamiento**, no únicamente al agente.

---

## 5.7 Integración con `Trainer`

Extender `2_Assault/src/trainer.py` para soportar continuidad.

Debe aceptar un `initial_global_step` o estado equivalente explícito.

Ejemplo conceptual:

```text
new:
initial_global_step = 0

resume:
initial_global_step = checkpoint.global_step
```

El trainer debe interpretar `total_timesteps` de forma inequívoca.

### Semántica obligatoria recomendada

`training.total_timesteps` representa el objetivo global total de la corrida.

Ejemplo:

```text
checkpoint global_step = 40
training.total_timesteps = 100

resume
→ ejecutar steps 41..100
→ finalizar global_step = 100
```

No debe ejecutar 100 pasos adicionales y terminar accidentalmente en 140 salvo que una API explícita de `additional_timesteps` sea utilizada y claramente nombrada.

Esta semántica debe estar probada.

---

## 5.8 `learning_starts` después de resume

### Resume completo

Si Replay Buffer restaurado contiene suficientes transiciones y el `global_step` ya superó `learning_starts`, los updates pueden continuar según las reglas existentes de `train_frequency`.

### Resume liviano

Aunque `global_step` sea alto, el Replay Buffer inicia vacío.

El trainer no debe actualizar hasta cumplir nuevamente:

```text
len(replay_buffer) >= batch_size
```

La condición existente del trainer debe proteger este caso.

No se debe reiniciar artificialmente `global_step` ni epsilon solo para reconstruir el buffer.

---

## 5.9 Métricas mínimas de continuidad

El checkpoint debe conservar suficientes métricas para demostrar continuidad, por ejemplo:

- número de episodios completados acumulados;
- recompensas de episodios recientes o acumuladas cuando sea razonable;
- longitudes de episodio relevantes;
- updates acumulados;
- última loss;
- número de Target syncs o información equivalente;
- tiempo de entrenamiento acumulado si ya puede mantenerse de forma simple.

HU005 no debe crear todavía un sistema avanzado de logging.

Las métricas pueden almacenarse mediante una estructura simple serializable.

---

## 5.10 `run_id`

Toda corrida reanudable debe tener un identificador explícito.

Ejemplo:

```text
assault_ddqn_exp_001
```

Reglas:

- `new` crea/recibe un nuevo `run_id`;
- `resume_full` y `resume_light` conservan el `run_id` del checkpoint;
- no generar un `run_id` diferente durante resume;
- checkpoint debe registrar `run_id`;
- rutas de artefactos deben poder agruparse por `run_id`.

No implementar todavía registro MLflow del `run_id`; eso pertenece a HU008.

---

## 5.11 Nombres y rutas de checkpoints

Usar nombres deterministas y legibles.

Ejemplo:

```text
checkpoints/
└── assault_ddqn_exp_001/
    ├── checkpoint_step_000040.pt
    ├── checkpoint_step_000080.pt
    └── checkpoint_step_000120.pt
```

Reglas:

- cada checkpoint debe incorporar el timestep;
- no utilizar únicamente `latest.pt` como fuente de verdad;
- puede existir un alias/metadata `latest` como comodidad futura, pero resume no debe elegirlo silenciosamente;
- si el path objetivo ya existe, `save()` debe fallar por defecto;
- sobrescribir debe requerir una opción explícita como `overwrite=True`;
- crear directorios con `exist_ok=True`.

---

## 5.12 Persistencia local y Colab

### Local

Por defecto puede utilizarse:

```text
2_Assault/checkpoints/
```

Los checkpoints no deben versionarse rutinariamente en GitHub.

### Google Colab

El proyecto debe permitir configurar una ruta persistente fuera de `/content`, preferiblemente Google Drive.

Ejemplo conceptual:

```text
/content/drive/MyDrive/reinforcement_learning_reto_1/checkpoints/
```

HU005 debe dejar el código preparado para recibir esa ruta.

No es obligatorio automatizar autenticación/montaje de Google Drive desde tests locales.

El notebook sí debe poder detectar/configurar la ruta elegida cuando el usuario ejecute manualmente en Colab.

No guardar checkpoints importantes únicamente en:

```text
/content/...
```

porque se perderían al finalizar la sesión.

---

## 5.13 Configuración central

Extender `2_Assault/configs/ddqn_config.yaml` con parámetros mínimos de checkpointing, por ejemplo:

```yaml
checkpointing:
  enabled: true
  interval_steps: <valor corto para HU005>
  directory: checkpoints
  mode: new
  run_id: assault_ddqn_exp_001
  resume_checkpoint: null
  save_replay_buffer: true
```

Los nombres pueden adaptarse.

Debe existir una semántica inequívoca para:

- modo;
- frecuencia;
- directorio;
- checkpoint seleccionado;
- si Replay Buffer será persistido.

No introducir configuración de TensorBoard ni MLflow.

---

## 5.14 Guardado periódico

HU005 debe integrar checkpointing con el flujo de entrenamiento de forma simple.

Puede implementarse mediante:

- callback pequeño específico;
- hook del trainer;
- función de checkpointing invocada cuando corresponda.

Evitar una arquitectura genérica de callbacks antes de HU006 si no aporta valor.

El checkpoint debe poder dispararse en:

```text
global_step % interval_steps == 0
```

cuando esté habilitado.

Para evitar scope creep, el trainer puede exponer un hook estrecho de persistencia en vez de implementar un sistema completo de eventos.

---

## 5.15 Idempotencia

HU005 debe aplicar estrictamente las reglas de `linemientos.md`:

- repetir una celda no elimina checkpoints válidos;
- `new` no reutiliza silenciosamente una corrida existente;
- resume no selecciona checkpoint ambiguo;
- no reiniciar `global_step` en resume;
- conservar configuración del checkpoint;
- no sobrescribir por defecto;
- rutas creadas de manera segura;
- errores explícitos ante incompatibilidades materiales.

### Configuración incompatible

Al cargar debe validarse al menos compatibilidad de:

- environment id;
- dimensiones de preprocessing;
- input channels;
- número de acciones;
- arquitectura/configuración crítica del agente;
- Replay Buffer shape cuando se usa resume completo.

Cambios claramente incompatibles deben abortar el resume.

No es necesario bloquear diferencias no materiales como `total_timesteps` cuando el objetivo sea extender la corrida.

---

## 5.16 Notebook

Actualizar `2_Assault/assault_ddqn.ipynb` únicamente como **orquestador**.

Debe incorporar un bloque explícito de ejecución:

```text
bootstrap
↓
config/runtime
↓
HU002 checks
↓
Preflight
↓
seleccionar execution mode
  ├── new
  ├── resume_full
  └── resume_light
↓
mostrar run_id + checkpoint elegido
↓
confirmar global_step inicial
↓
short training HU005
↓
checkpoint
↓
resumen de continuidad
```

El notebook debe mostrar claramente algo equivalente a:

```text
Execution mode: resume_full
Run ID: assault_ddqn_exp_001
Loaded checkpoint: .../checkpoint_step_000048.pt
Restored global_step: 48
Replay Buffer restored: True
Target total_timesteps: 80
```

No debe seleccionar un checkpoint arbitrariamente.

El usuario puede establecer el modo/configuración antes de `Run All`, pero después de esa selección el flujo debe estar automatizado.

---

## 6. Fuera de alcance

HU005 **no** debe implementar:

- TensorBoard;
- MLflow;
- métricas visuales;
- selección automática del mejor modelo;
- evaluación formal ≥10 episodios;
- video;
- entrenamiento largo/final;
- optimización de hiperparámetros;
- PER;
- Dueling DQN;
- Rainbow;
- Noisy Nets;
- n-step returns;
- reward clipping;
- servicios externos de almacenamiento distintos de una ruta de filesystem configurable;
- sincronización automática de Google Drive mediante APIs complejas;
- GitHub Actions para entrenamiento;
- automatización Codex → Colab.

TensorBoard pertenece a HU006.
El smoke E2E con checkpoints + TensorBoard pertenece a HU007.
MLflow pertenece a HU008.
El entrenamiento largo pertenece a HU009.

---

## 7. Decisiones y restricciones técnicas

### 7.1 Separación de responsabilidades

```text
environment.py
→ entorno/preprocessing

network.py
→ CNN

replay_buffer.py
→ memoria de experiencias + serialización mínima

agent.py
→ lógica DDQN

trainer.py
→ secuencia temporal y continuidad desde global_step

checkpointing.py
→ persistencia/restauración del estado de entrenamiento

preflight.py
→ gate previo

assault_ddqn.ipynb
→ orquestación/evidencia
```

---

### 7.2 Formato de checkpoint

Preferir un único artefacto PyTorch (`.pt`) con diccionario versionado para HU005.

Ejemplo conceptual:

```python
{
    "schema_version": 1,
    "run_id": ...,
    "global_step": ...,
    "config": ...,
    "agent": {...},
    "training_metrics": {...},
    "replay_buffer": {...} or None,
}
```

No usar pickle de objetos arbitrarios cuando sea evitable.

La carga debe usar prácticas compatibles con la versión de PyTorch definida en el proyecto.

---

### 7.3 Compatibilidad CPU/GPU

Un checkpoint guardado en GPU debe poder cargarse en CPU y viceversa mediante `map_location` o mecanismo equivalente.

Los tests CPU son obligatorios.

GPU se valida cuando esté disponible.

---

### 7.4 Persistencia atómica

Evitar dejar un checkpoint aparentemente válido si el proceso falla durante la escritura.

Preferir:

```text
write temporary file
↓
fsync/close cuando aplique
↓
rename/replace atómico al path final
```

Para HU005 basta una implementación simple y portable que reduzca el riesgo de archivos parciales.

No sobreingenierizar con bases de datos o locking distribuido.

---

### 7.5 Seguridad de configuración

No incluir secretos, tokens ni credenciales dentro del checkpoint o YAML.

Rutas persistentes son configuración; credenciales de Google/GitHub no lo son.

---

## 8. Plan de implementación / tareas

### T01 — Extender configuración

**Archivo:** `2_Assault/configs/ddqn_config.yaml`

Agregar sección de checkpointing.

**Resultado:** modo, intervalo, ruta, run_id y política de Replay Buffer quedan explícitos.

### T02 — Serializar Replay Buffer

**Archivo:** `2_Assault/src/replay_buffer.py`

Agregar export/import de estado.

**Resultado:** un buffer restaurado conserva tamaño, posición, contenido válido y contrato `uint8`.

### T03 — Crear checkpoint manager

**Archivo:** `2_Assault/src/checkpointing.py`

Implementar guardado/carga segura.

**Resultado:** checkpoint versionado con estado completo de entrenamiento.

### T04 — Validar compatibilidad

Implementar checks de configuración crítica antes de restaurar.

**Resultado:** resume incompatible falla explícitamente.

### T05 — Integrar global_step restaurado en Trainer

**Archivo:** `2_Assault/src/trainer.py`

Permitir iniciar desde timestep distinto de cero.

**Resultado:** el trainer finaliza exactamente en el objetivo global configurado.

### T06 — Integrar métricas acumuladas

Permitir restaurar/continuar métricas mínimas necesarias.

**Resultado:** resume no aparenta ser una corrida nueva.

### T07 — Implementar `new`

Validar creación limpia de una corrida nueva.

### T08 — Implementar `resume_full`

Restaurar agente, optimizer, global step, métricas y Replay Buffer.

### T09 — Implementar `resume_light`

Restaurar agente, optimizer, global step y métricas sin Replay Buffer.

### T10 — Guardado periódico

Integrar persistencia en intervalos configurados sin introducir framework de callbacks complejo.

### T11 — Idempotencia

Probar no sobrescritura, selección explícita de checkpoint y ejecución repetida segura.

### T12 — Integrar notebook

Agregar selección explícita de modo y resumen de continuidad.

### T13 — Tests

Crear tests focalizados de checkpointing/resume.

### T14 — Smoke real

Ejecutar Assault real: entrenar → guardar → recrear objetos → cargar → continuar.

### T15 — Documentar evidencia

Actualizar `2_Assault/docs/implementacion.md` únicamente con resultados reales.

---

## 9. Criterios de aceptación

### CA01 — Checkpoint mínimo

**Dado** un entrenamiento activo, **cuando** se guarda checkpoint, **entonces** contiene Online, Target, optimizer, `global_step`, configuración y métricas mínimas.

### CA02 — Run ID

**Dado** un checkpoint, **cuando** se inspecciona, **entonces** contiene el `run_id` correcto y resume conserva ese mismo identificador.

### CA03 — Epsilon reconstruible

**Dado** un checkpoint en timestep `N`, **cuando** se restaura, **entonces** epsilon puede reconstruirse de forma equivalente usando `N` y la configuración guardada.

### CA04 — Replay Buffer full

**Dado** `resume_full`, **cuando** se restaura, **entonces** Replay Buffer conserva tamaño, posición, shapes, dtypes y transiciones válidas.

### CA05 — Replay Buffer light

**Dado** `resume_light`, **cuando** se restaura, **entonces** Replay Buffer comienza vacío y esto queda explícitamente indicado.

### CA06 — Agent restore

**Dado** un checkpoint, **cuando** se carga en una nueva instancia, **entonces** Online/Target recuperan predicciones equivalentes.

### CA07 — Optimizer restore

**Dado** un optimizer con estado después de updates, **cuando** se restaura, **entonces** su estado necesario para continuar entrenamiento se conserva.

### CA08 — Global step

**Dado** checkpoint `global_step=N`, **cuando** se reanuda, **entonces** el primer nuevo step continúa desde `N` y no desde cero.

### CA09 — Objetivo global

**Dado** checkpoint `N` y `total_timesteps=T>N`, **cuando** se reanuda, **entonces** finaliza en `T`, no en `N+T`.

### CA10 — Resume full updates

**Dado** buffer restaurado suficiente y `global_step` posterior a `learning_starts`, **cuando** corresponde `train_frequency`, **entonces** los updates pueden continuar inmediatamente.

### CA11 — Resume light gate

**Dado** buffer vacío después de `resume_light`, **cuando** se reanuda, **entonces** no existen updates hasta volver a tener al menos `batch_size` transiciones.

### CA12 — No overwrite

**Dado** un checkpoint existente, **cuando** se intenta guardar sobre el mismo path sin autorización explícita, **entonces** la operación falla.

### CA13 — Explicit resume

**Dadas** múltiples opciones de checkpoint, **cuando** se solicita resume sin especificar una ruta inequívoca, **entonces** el sistema no elige una automáticamente.

### CA14 — New run isolation

**Dada** una corrida `new`, **cuando** ya existe otro `run_id`, **entonces** sus checkpoints no son sobrescritos ni reutilizados silenciosamente.

### CA15 — Config compatibility

**Dado** un checkpoint incompatible con dimensiones/acciones/entorno, **cuando** se intenta cargar, **entonces** falla con un error claro.

### CA16 — CPU map

**Dado** un checkpoint guardado desde un dispositivo disponible, **cuando** se carga en CPU, **entonces** no existen errores de device mismatch.

### CA17 — Atomic/safe save

**Dado** un guardado exitoso, **cuando** finaliza, **entonces** existe un archivo completo en el path final y no queda un temporal presentado como checkpoint válido.

### CA18 — Metrics continuity

**Dadas** métricas acumuladas antes del checkpoint, **cuando** se reanuda, **entonces** el resumen final refleja continuidad en lugar de reiniciar silenciosamente todos los contadores.

### CA19 — Notebook mode

**Dado** el notebook, **cuando** se ejecuta, **entonces** imprime modo, `run_id`, checkpoint seleccionado, `global_step` inicial y si Replay Buffer fue restaurado.

### CA20 — Local/Colab path

**Dado** local o Colab, **cuando** se configura una ruta de checkpoints válida, **entonces** checkpointing no depende de una ruta fija específica del PC.

### CA21 — Real Assault resume

**Dado** Assault real, **cuando** se ejecuta entrenamiento corto → save → recreación de objetos → load → resume, **entonces** la segunda fase continúa correctamente y produce loss finita cuando corresponda.

### CA22 — Sin scope creep

**Dado** el PR HU005, **cuando** se revisa, **entonces** no introduce TensorBoard, MLflow, evaluación formal, entrenamiento largo ni extensiones DDQN fuera de alcance.

---

## 10. Autovalidaciones obligatorias

### AV01 — Imports

```bash
python -c "... import CheckpointManager ..."
```

**Esperado:** imports limpios.

### AV02 — Suite completa

```bash
python -m pytest 2_Assault/tests -q
```

**Esperado:** tests previos + HU005 pasan.

### AV03 — Compile

```bash
python -m compileall -q 2_Assault/src
```

**Esperado:** PASS.

### AV04 — Replay Buffer roundtrip

Crear buffer pequeño, llenarlo, serializarlo y cargarlo en una instancia nueva.

**Esperado:** contenido/tamaño/posición equivalentes.

### AV05 — Checkpoint full roundtrip

Entrenar brevemente, guardar full, destruir objetos y restaurar.

**Esperado:** agente, optimizer, global step y buffer equivalentes.

### AV06 — Checkpoint light roundtrip

Guardar/cargar sin Replay Buffer.

**Esperado:** agente/progreso restaurados y buffer nuevo vacío.

### AV07 — No overwrite

Guardar dos veces el mismo path sin `overwrite`.

**Esperado:** segunda operación falla explícitamente.

### AV08 — Compatibility gate

Modificar una dimensión/configuración crítica y cargar.

**Esperado:** error claro de incompatibilidad.

### AV09 — Resume global step

Entrenar hasta `N`, guardar, cargar y continuar hasta `T`.

**Esperado:** `final_global_step == T`.

### AV10 — Resume epsilon

Comparar epsilon reconstruido después de load con epsilon esperado para el `global_step` guardado.

**Esperado:** equivalencia numérica.

### AV11 — Resume full immediate update

Con buffer suficiente y frecuencia cumplida, continuar.

**Esperado:** puede actualizar sin volver a llenar desde cero.

### AV12 — Resume light refill gate

Cargar light con buffer vacío.

**Esperado:** no update hasta `len(buffer) >= batch_size`.

### AV13 — Metrics continuity

Comparar contadores antes/después del resume.

**Esperado:** continuidad demostrable.

### AV14 — Safe path/run id

Crear dos `run_id`.

**Esperado:** rutas separadas y sin colisión.

### AV15 — Real Assault smoke

Ejecutar con entorno real:

```text
Preflight
→ train corto
→ save
→ recrear env/agent/buffer/trainer
→ load
→ resume
```

**Esperado:** ejecución completa sin error y loss finita cuando aplique.

### AV16 — Notebook local

Ejecutar las celdas HU005 localmente en orden con dependencias ya instaladas.

**Esperado:** modo explícito, checkpoint creado/cargado y continuidad demostrada.

### AV17 — Device

CPU obligatorio; GPU si disponible.

**Esperado:** load mediante `map_location` o equivalente sin mismatch.

### AV18 — Scope

Revisar diff.

**Esperado:** ausencia de TensorBoard/MLflow y entrenamiento largo.

---

## 11. Evidencias requeridas

El PR HU005 debe incluir o referenciar:

- salida completa de `pytest`;
- `compileall`;
- estructura del checkpoint;
- `schema_version`;
- `run_id`;
- checkpoint path;
- tamaño del checkpoint full y light;
- `global_step` antes de save;
- `global_step` después de load;
- objetivo global después de resume;
- epsilon reconstruido;
- Replay Buffer size antes/después en full;
- Replay Buffer size después de light;
- evidencia de optimizer restaurado;
- evidencia de no overwrite;
- prueba de configuración incompatible;
- métricas continuadas;
- resultado del smoke con Assault real;
- dispositivo utilizado;
- commit Git;
- confirmación explícita de ausencia de scope creep.

Si se prueba en Colab manualmente, agregar además:

- ruta persistente elegida;
- confirmación de que el checkpoint no reside únicamente en `/content`;
- dispositivo/GPU;
- SHA ejecutado.

No inventar evidencia Colab si no fue ejecutada.

---

## 12. Definition of Done

HU005 se considera terminada únicamente cuando:

- [ ] configuración de checkpointing está centralizada;
- [ ] existe un `run_id` explícito;
- [ ] `checkpointing.py` o equivalente está implementado;
- [ ] Replay Buffer soporta serialización/restauración;
- [ ] checkpoint guarda Online;
- [ ] checkpoint guarda Target;
- [ ] checkpoint guarda optimizer;
- [ ] checkpoint guarda `global_step`;
- [ ] checkpoint guarda configuración;
- [ ] epsilon puede reconstruirse correctamente;
- [ ] métricas mínimas se conservan;
- [ ] `new` funciona;
- [ ] `resume_full` funciona;
- [ ] `resume_light` funciona;
- [ ] full restaura Replay Buffer;
- [ ] light inicia Replay Buffer vacío;
- [ ] trainer continúa desde timestep restaurado;
- [ ] objetivo global de timesteps es inequívoco;
- [ ] resume light respeta batch gate;
- [ ] no overwrite por defecto funciona;
- [ ] resume requiere checkpoint explícito;
- [ ] compatibilidad de configuración se valida;
- [ ] carga CPU funciona;
- [ ] guardado seguro/atómico funciona;
- [ ] rutas por `run_id` no colisionan;
- [ ] notebook muestra modo y checkpoint;
- [ ] smoke real Assault save/load/resume pasa;
- [ ] tests previos continúan pasando;
- [ ] tests HU005 pasan;
- [ ] documentación/evidencia está actualizada;
- [ ] no existen errores bloqueantes conocidos;
- [ ] no se implementó TensorBoard, MLflow ni alcance futuro.

---

## 13. Riesgos y consideraciones

### 13.1 Tamaño del Replay Buffer

El Replay Buffer puede dominar el tamaño del checkpoint.

Mitigación:

- mantener `uint8`;
- guardar solo posiciones válidas;
- soportar `resume_light`;
- medir tamaño real de checkpoint antes de HU009.

### 13.2 Colab efímero

Guardar en `/content` no garantiza continuidad entre sesiones.

Mitigación:

- ruta configurable;
- recomendar Google Drive para ejecuciones persistentes;
- notebook debe mostrar el path final antes de entrenar.

### 13.3 Resume con configuración distinta

Cambiar arquitectura o preprocessing puede invalidar el checkpoint.

Mitigación:

- guardar config completa;
- validar campos críticos;
- fallar explícitamente.

### 13.4 Global step ambiguo

Un error frecuente sería tratar `total_timesteps` como pasos adicionales después de resume.

Mitigación:

- definirlo como objetivo global;
- tests `N → T`;
- APIs con nombres explícitos si existe override adicional.

### 13.5 Epsilon reiniciado

Si epsilon vuelve a `epsilon_start` después del resume, el comportamiento del agente cambia incorrectamente.

Mitigación:

- reconstruir epsilon desde `global_step` + config;
- test específico.

### 13.6 Optimizer state

Cargar solo pesos de redes no equivale a continuar entrenamiento.

Mitigación:

- persistir optimizer;
- probar su restauración después de updates reales.

### 13.7 Checkpoint parcial

Una interrupción durante escritura podría dejar un archivo corrupto.

Mitigación:

- escritura temporal;
- rename al final;
- nunca tratar temporales como checkpoints válidos.

### 13.8 Reanudación de episodio

HU005 no guarda el estado exacto del emulador ALE.

Después de resume se inicia un nuevo episodio, manteniendo el estado de aprendizaje del agente.

Esto es aceptable para el alcance y debe documentarse explícitamente.

### 13.9 Reproducibilidad absoluta

Restaurar redes/optimizer/buffer no garantiza bitwise determinism entre GPU/sesiones/ALE.

El objetivo es continuidad reproducible del entrenamiento, no identidad absoluta de cada frame futuro.

---

## 14. Resultado esperado y gate para HU006

HU006 solo debe comenzar cuando HU005 demuestre:

```text
HU004 short training
      ↓
checkpoint @ N
      ↓
new process/objects
      ↓
explicit resume
      ↓
Online/Target/optimizer restored
      ↓
global_step restored
      ↓
epsilon reconstructed
      ↓
Replay Buffer full OR explicit light mode
      ↓
continue until T
      ↓
final global_step = T
      ↓
checkpoint persistence PASS
      ↓
HU005 PASS
```

No es necesario demostrar todavía mejora significativa de recompensa ni ejecutar entrenamiento largo.

**Habilita:** HU006 — Observabilidad con TensorBoard.
