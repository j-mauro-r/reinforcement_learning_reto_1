# HU007 — Checkpoints, reanudación e idempotencia para BattleZone

## 1. Identificación

- **ID:** HU007
- **Nombre:** Checkpoints, reanudación e idempotencia para BattleZone
- **Estado:** Lista para implementación
- **Dependencias previas:** HU005 — Núcleo del agente DQN; HU006 — Ciclo de entrenamiento DQN
- **Habilita:** HU008 — Observabilidad con TensorBoard; HU009 — Smoke test end-to-end
- **Algoritmo vigente:** `DQN`
- **Fuentes de verdad:** `enunciado_reto_1.txt`, `3_BattleZone/docs/implementacion.md`, `3_BattleZone/docs/lineamientos.md`, `3_BattleZone/docs/arquitectura.md`, HU005, HU006 y `3_BattleZone/configs/battlezone_config.yaml`.

## 2. Contexto

HU006 dejó implementado un ciclo temporal DQN funcional con `global_step`, `episode_index`, epsilon schedule, Replay Buffer, updates y sincronización periódica de Target Network.

BattleZone tendrá entrenamientos largos y potencialmente fragmentados entre sesiones locales o Google Colab. Por tanto, antes de introducir observabilidad y entrenamientos largos, el proyecto debe garantizar que una ejecución pueda:

1. guardar el estado necesario para continuar;
2. restaurarlo de manera explícita y verificable;
3. continuar desde el timestep correcto sin reiniciar progreso silenciosamente;
4. distinguir entre una corrida nueva y una reanudación;
5. evitar seleccionar automáticamente checkpoints ambiguos;
6. evitar sobrescribir evidencia válida accidentalmente.

Esta HU implementa la base de recuperabilidad e idempotencia. No implementa TensorBoard, manifests completos ni entrenamiento largo.

## 3. Objetivo verificable

Implementar y validar una estrategia de checkpointing para DQN que permita ejecutar un entrenamiento corto, guardar su estado, reconstruir un agente/trainer compatibles y continuar el entrenamiento desde el `global_step` persistido.

La validación debe demostrar al menos:

```text
NEW RUN
  ↓
train N steps
  ↓
save checkpoint
  ↓
reconstruct process objects
  ↓
load checkpoint
  ↓
resume
  ↓
continue from N, not from 0
```

El resultado de HU007 es **continuidad funcional**, no aprendizaje ni performance.

## 4. Decisiones técnicas obligatorias

### DT01 — Responsabilidad de persistencia separada

La persistencia debe mantenerse desacoplada de `DQNAgent` y `DQNTrainer` tanto como sea razonable.

Crear preferiblemente:

```text
3_BattleZone/src/persistence.py
```

con responsabilidades de:

- construir payload de checkpoint;
- guardar checkpoint de forma segura;
- cargar checkpoint explícito;
- validar estructura/versión/compatibilidad;
- resolver rutas sin selección ambigua;
- restaurar Replay Buffer cuando corresponda.

No introducir un framework genérico innecesario.

### DT02 — Modos de checkpoint

HU007 debe soportar explícitamente dos modos:

#### Full checkpoint

Debe contener como mínimo:

- estado completo del `DQNAgent`:
  - Online Network;
  - Target Network;
  - optimizer;
  - gamma y metadatos estructurales ya definidos por HU005;
- estado del trainer/progreso:
  - `global_step`;
  - `episode_index`;
  - `episode_step` si se persiste de forma coherente;
  - `episode_reward` si se persiste de forma coherente;
- estado de exploración derivable o explícito;
- Replay Buffer;
- seed/base seed necesaria para continuidad;
- configuración asociada o snapshot mínimo suficiente para validación;
- versión del esquema de checkpoint.

#### Lightweight checkpoint

Debe contener como mínimo:

- agente;
- optimizer;
- progreso global;
- configuración/metadata necesaria;

pero puede omitir Replay Buffer deliberadamente.

Al reanudar un checkpoint liviano, el Replay Buffer se reconstruirá gradualmente y `learning_starts`/gates deberán seguir una política explícita y segura.

### DT03 — No serializar el entorno ALE

No intentar serializar directamente la instancia Gymnasium/ALE ni objetos de render.

El entorno debe reconstruirse mediante la factory HU003 usando configuración y seed apropiadas.

La HU no promete continuidad bit-a-bit del estado interno exacto de ALE entre procesos.

Debe distinguirse entre:

- **continuidad del entrenamiento DQN**: redes, optimizer, counters, replay cuando aplique;
- **continuidad exacta de un episodio ALE en curso**: fuera de alcance si no existe un contrato seguro y portable.

Por simplicidad y seguridad, el checkpoint periódico puede requerir guardarse en un límite de transición consistente y, si la reanudación no conserva estado ALE exacto, iniciar un nuevo episodio manteniendo `global_step` y `episode_index` coherentes.

### DT04 — Global step no se reinicia

Después de cargar un checkpoint con:

```text
global_step = N
```

el siguiente entrenamiento debe continuar desde `N`.

Está prohibido que `DQNTrainer.train()` vuelva silenciosamente a `0`.

HU007 puede adaptar HU006 para permitir estado inicial/restaurado de forma explícita.

### DT05 — Epsilon coherente con progreso

El epsilon después de resume debe calcularse a partir del `global_step` restaurado usando el mismo schedule versionado.

Ejemplo:

```text
checkpoint global_step = 500
resume → epsilon_schedule.value(500)
```

No reiniciar epsilon a `start`.

### DT06 — Replay Buffer full resume

El full checkpoint debe poder restaurar un Replay Buffer funcional y equivalente en:

- tamaño;
- capacidad;
- shape;
- dtype;
- posición/cursor de escritura o semántica equivalente;
- transiciones almacenadas.

No basta guardar solo `len(buffer)`.

Si el Replay Buffer actual no expone `state_dict/load_state_dict`, HU007 puede añadir el contrato mínimo necesario.

### DT07 — Replay Buffer lightweight resume

El lightweight resume debe comenzar con Replay vacío o reconstruido según una política explícita.

Mientras el Replay Buffer no alcance el mínimo necesario:

```text
len(replay) >= batch_size
```

no debe ejecutarse update.

La política debe evitar usar datos inexistentes o inventados.

### DT08 — Checkpoint explícito, no ambiguo

La API de carga debe recibir una ruta explícita.

Está prohibido en HU007 implementar comportamiento silencioso tipo:

```text
find latest checkpoint and resume automatically
```

si existen múltiples candidatos ambiguos.

Puede existir una función de listado/inspección de checkpoints, pero la selección final debe ser explícita.

### DT09 — Escritura segura

Guardar checkpoints debe minimizar riesgo de archivo parcialmente escrito.

Preferir patrón equivalente a:

```text
write temporary file
→ fsync/close cuando aplique
→ atomic replace/rename
```

No sobrescribir un checkpoint existente por defecto salvo que la llamada lo autorice explícitamente.

### DT10 — Versionado del esquema

Todo checkpoint debe incluir una versión de esquema, por ejemplo:

```text
schema_version: 1
```

Una versión no soportada debe fallar con error claro.

### DT11 — Compatibilidad

Antes de restaurar, validar al menos:

- algoritmo `DQN`;
- action_dim;
- state_shape;
- batch_size cuando corresponda;
- estructura/configuración crítica;
- modo de checkpoint;
- schema_version.

No cargar silenciosamente estados estructuralmente incompatibles.

### DT12 — Idempotencia básica

Reejecutar operaciones de checkpoint no debe destruir artefactos válidos.

Reglas:

- directorios con `exist_ok=True`;
- nombres/rutas deterministas o explícitos;
- no sobrescribir por defecto;
- carga nunca modifica el checkpoint fuente;
- NEW y RESUME deben ser acciones explícitas;
- no reiniciar progreso silenciosamente;
- no borrar checkpoints previos automáticamente.

## 5. Alcance funcional esperado

### 5.1 `src/persistence.py`

Contratos recomendados, ajustables si existe una alternativa más simple:

```python
@dataclass(frozen=True)
class CheckpointMetadata:
    schema_version: int
    checkpoint_mode: str
    algorithm: str
    global_step: int
    episode_index: int


def save_checkpoint(...): ...

def load_checkpoint(...): ...

def restore_training_state(...): ...
```

Puede usarse una clase `CheckpointManager` si reduce acoplamiento, pero no es obligatorio.

### 5.2 `src/replay_buffer.py`

Si es necesario, añadir:

```python
def state_dict(self) -> dict: ...
def load_state_dict(self, state: dict) -> None: ...
```

con validación estricta.

### 5.3 `src/trainer.py`

Adaptar el trainer para aceptar progreso inicial/restaurado.

Debe ser posible distinguir:

```text
NEW
RESUME_FULL
RESUME_LIGHTWEIGHT
```

sin convertir todavía el trainer en un orquestador de sesiones completo.

La lógica de checkpoint periódico puede integrarse mediante un callback pequeño o hook explícito si aporta claridad, pero no se debe adelantar `callbacks.py` de TensorBoard.

## 6. Configuración

Actualizar `battlezone_config.yaml` con una sección equivalente a:

```yaml
checkpointing:
  enabled: true
  baseline_note: "baseline de implementacion por validar"
  directory: "3_BattleZone/checkpoints"
  interval_steps: ...
  default_mode: "full"
  schema_version: 1
  allow_overwrite: false
```

Los valores son baseline de implementación, no hiperparámetros óptimos.

No introducir rutas absolutas locales.

## 7. Flujo NEW

El flujo NEW debe:

1. construir config;
2. crear entorno HU003;
3. crear `DQNAgent` nuevo;
4. crear Replay Buffer vacío;
5. iniciar `global_step=0`;
6. epsilon desde `schedule.value(0)`;
7. entrenar según HU006;
8. guardar checkpoint explícitamente cuando corresponda.

## 8. Flujo RESUME_FULL

Debe:

1. recibir ruta explícita;
2. cargar y validar checkpoint;
3. reconstruir entorno HU003;
4. reconstruir agente compatible;
5. restaurar Online/Target/optimizer;
6. restaurar Replay Buffer;
7. restaurar `global_step` y counters compatibles;
8. calcular epsilon desde el step restaurado;
9. continuar entrenamiento;
10. producir un nuevo checkpoint sin destruir el anterior por defecto.

## 9. Flujo RESUME_LIGHTWEIGHT

Debe:

1. recibir ruta explícita;
2. cargar agente/optimizer/progreso;
3. reconstruir entorno;
4. iniciar Replay Buffer vacío o bajo política documentada;
5. conservar `global_step`;
6. conservar epsilon según progreso;
7. no ejecutar updates hasta tener Replay suficiente;
8. continuar recolectando experiencia.

## 10. Semántica de episodios al reanudar

Como HU007 no serializa estado interno ALE, la reanudación entre procesos puede comenzar con `env.reset()`.

Política recomendada:

- conservar `global_step`;
- conservar número de episodios completados;
- no afirmar que se continúa exactamente el frame/episodio interrumpido;
- resetear `episode_step=0` y `episode_reward=0.0` al reconstruir entorno si el checkpoint fue tomado en medio de un episodio y no se conserva estado ALE;
- documentar este comportamiento explícitamente en metadata/evidencia.

No inventar continuidad exacta que el sistema no soporte.

## 11. Tests obligatorios

Crear preferiblemente:

```text
3_BattleZone/tests/test_persistence.py
```

y actualizar tests existentes cuando corresponda.

Cubrir como mínimo:

1. creación de directorio segura;
2. full checkpoint save;
3. lightweight checkpoint save;
4. load explícito;
5. archivo inexistente;
6. schema_version inválido;
7. algoritmo incompatible;
8. metadatos estructurales incompatibles;
9. overwrite bloqueado por defecto;
10. overwrite explícito permitido si se implementa;
11. full resume restaura Online;
12. full resume restaura Target;
13. full resume restaura optimizer;
14. full resume restaura Replay size/contenido;
15. full resume restaura `global_step`;
16. epsilon después de resume corresponde al step restaurado;
17. resume no reinicia `global_step`;
18. lightweight resume no restaura Replay;
19. lightweight resume respeta batch gate;
20. checkpoint fuente permanece sin modificación tras load;
21. múltiples checkpoints no provocan selección automática ambigua;
22. save → load → continue produce nuevos steps mayores al checkpoint;
23. no dependencia de `2_Assault/`;
24. no TensorBoard/MLflow.

## 12. Validación integrada obligatoria

Además de tests unitarios debe realizarse una validación corta real o controlada equivalente a:

```text
1. entrenar BattleZone hasta N steps;
2. guardar full checkpoint;
3. destruir/recrear agent + trainer;
4. cargar checkpoint;
5. comprobar global_step=N;
6. comprobar Replay restaurado;
7. continuar hasta M>N;
8. comprobar final_step=M;
9. guardar checkpoint posterior;
```

También validar lightweight resume:

```text
checkpoint N
→ recreate
→ load lightweight
→ global_step=N
→ replay vacío
→ recolectar nuevas transiciones
→ no update hasta batch gate
```

Usar valores pequeños para evitar costo computacional innecesario.

## 13. Evidencia requerida

Crear:

```text
3_BattleZone/docs/hu007_evidencia_implementacion.md
```

Debe registrar únicamente resultados reales:

- rama;
- SHA;
- archivos modificados;
- schema_version;
- modos soportados;
- configuración de checkpoint;
- ruta(s) de checkpoint de prueba;
- tamaño de archivos si se mide;
- step antes de save;
- step restaurado;
- step después de continuar;
- Replay size antes/después;
- epsilon antes/después de resume;
- tests ejecutados;
- resultados;
- CA/AV;
- limitaciones;
- política de episodio al reanudar.

No versionar checkpoints pesados producidos por pruebas salvo que sean artefactos mínimos deliberadamente necesarios. Preferir temporales en tests.

## 14. Criterios de aceptación

- **CA01:** HU006 está mergeada y el trainer vigente es la base de HU007.
- **CA02:** existe persistencia desacoplada y documentada.
- **CA03:** full checkpoint guarda/restaura agente y optimizer.
- **CA04:** full checkpoint guarda/restaura Replay Buffer funcional.
- **CA05:** lightweight checkpoint omite Replay explícitamente.
- **CA06:** `global_step` y progreso relevante se restauran sin reinicio silencioso.
- **CA07:** epsilon tras resume corresponde al `global_step` restaurado.
- **CA08:** schema_version y compatibilidad son validados.
- **CA09:** carga requiere ruta explícita; no hay selección automática ambigua.
- **CA10:** overwrite está protegido por defecto.
- **CA11:** entorno se reconstruye mediante HU003; ALE no se serializa directamente.
- **CA12:** política de episodio tras resume está documentada y no promete continuidad exacta inexistente.
- **CA13:** save → load → resume continúa a steps mayores y produce estado coherente.
- **CA14:** full y lightweight resume tienen tests específicos.
- **CA15:** suite BattleZone permanece en verde.
- **CA16:** no se introduce Assault, MLflow, TensorBoard, manifests completos, evaluación formal ni entrenamiento largo.

## 15. Autovalidaciones oficiales

- **AV01 Dependencias:** HU005/HU006 vigentes en `main`.
- **AV02 Configuración:** checkpointing centralizado y versionado.
- **AV03 Full save/load:** agente/optimizer restaurados.
- **AV04 Replay full:** contenido y tamaño restaurados.
- **AV05 Lightweight:** Replay omitido explícitamente.
- **AV06 Global step:** restaurado exactamente.
- **AV07 Epsilon continuity:** schedule usa step restaurado.
- **AV08 Schema:** versión soportada/invalidación clara.
- **AV09 Compatibility:** errores claros ante mismatch.
- **AV10 Explicit path:** sin auto-select ambiguo.
- **AV11 Safe write:** overwrite protegido/escritura segura.
- **AV12 ALE boundary:** entorno reconstruido, no serializado.
- **AV13 Resume full:** N → M con M>N.
- **AV14 Resume lightweight:** batch gate respetado después de restaurar.
- **AV15 Regression:** tests HU005/HU006/HU007 y suite BattleZone PASS.
- **AV16 Scope:** sin Assault/MLflow/TensorBoard/HU008+.

## 16. Definition of Done

HU007 puede cerrarse únicamente cuando:

- HU006 esté mergeada;
- full y lightweight checkpoint estén implementados;
- Replay Buffer full sea realmente restaurable;
- `global_step` no se reinicie;
- epsilon sea coherente tras resume;
- compatibilidad/schema/overwrite estén protegidos;
- save → load → resume haya sido validado con evidencia real o controlada;
- suite BattleZone pase;
- CA01–CA16 y AV01–AV16 estén PASS;
- evidencia HU007 esté versionada;
- no se adelante HU008+;
- PR HU007 sea auditado y mergeado.

## 17. Fuera de alcance

No implementar en HU007:

- TensorBoard;
- logging persistente de métricas de HU008;
- `run_manifest.json` completo de HU010;
- selección de mejor modelo;
- evaluación formal;
- video;
- entrenamiento completo/largo;
- tuning;
- PER/DDQN/REINFORCE;
- MLflow;
- almacenamiento específico de Google Drive como dependencia obligatoria;
- serialización exacta del estado interno ALE.

## 18. Riesgos

1. **Replay Buffer grande:** full checkpoints pueden crecer rápidamente. HU007 debe medir/observar tamaño en pruebas y mantener soporte lightweight.
2. **Episodio interrumpido:** sin serialización ALE exacta se reinicia episodio al reanudar; debe quedar explícito.
3. **Compatibilidad futura:** cambios de arquitectura/config pueden invalidar checkpoints; schema y validaciones reducen el riesgo.
4. **Archivos parciales:** escritura no atómica puede dejar checkpoints corruptos.
5. **GPU/CPU:** checkpoint debe poder restaurarse usando `map_location` o mecanismo equivalente razonable cuando el dispositivo cambia.

## 19. Resultado esperado para HU008

HU008 recibirá un entrenamiento DQN capaz de ejecutarse como NEW o RESUME de forma explícita, con checkpoints full/lightweight, progreso global preservado y recuperación validada. HU008 añadirá TensorBoard sin modificar la semántica de checkpointing.
