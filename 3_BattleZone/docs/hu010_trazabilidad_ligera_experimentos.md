# HU010 — Trazabilidad ligera de experimentos BattleZone

## 1. Propósito

Implementar la capa mínima de trazabilidad necesaria para que cada corrida relevante de BattleZone pueda asociarse de forma inequívoca a código, configuración, seed, runtime, hardware, TensorBoard, checkpoints y resultados persistidos, sin utilizar MLflow.

HU010 es el **último gate de infraestructura antes de HU011 — Entrenamiento completo**.

Debe permitir responder, para cualquier corrida relevante:

1. ¿Qué código exacto se ejecutó?
2. ¿Qué configuración exacta se utilizó?
3. ¿Qué seed y algoritmo se usaron?
4. ¿En qué runtime y hardware corrió?
5. ¿Cuándo inició y terminó?
6. ¿Desde qué `global_step` inició y en cuál terminó?
7. ¿Fue NEW, RESUME_FULL o RESUME_LIGHTWEIGHT?
8. ¿Qué checkpoint de entrada se utilizó, si aplica?
9. ¿Qué checkpoint/modelo de salida quedó asociado?
10. ¿Dónde están los logs TensorBoard?
11. ¿Qué artefactos pertenecen a la corrida?
12. ¿Cómo puede reanudarse sin seleccionar artefactos ambiguos?

HU010 **no debe ejecutar entrenamiento largo ni demostrar aprendizaje**. Debe preparar y validar la trazabilidad que HU011 utilizará durante entrenamiento largo en Colab GPU.

---

## 2. Fuentes de verdad

HU010 debe respetar:

- `enunciado_reto_1.txt`;
- `3_BattleZone/docs/implementacion.md`;
- `3_BattleZone/docs/lineamientos.md`;
- `3_BattleZone/docs/arquitectura.md`;
- `3_BattleZone/docs/hu006_ciclo_entrenamiento_dqn.md`;
- `3_BattleZone/docs/hu007_checkpoints_reanudacion_idempotencia.md`;
- `3_BattleZone/docs/hu008_observabilidad_tensorboard.md`;
- `3_BattleZone/docs/hu009_smoke_test_end_to_end.md`;
- `3_BattleZone/docs/hu009_evidencia_implementacion.md`;
- `3_BattleZone/configs/battlezone_config.yaml`;
- implementación vigente en `3_BattleZone/src/`.

Algoritmo vigente: **DQN clásico**.

BattleZone continúa siendo totalmente independiente de `2_Assault/`.

---

## 3. Dependencias obligatorias

HU010 solo puede cerrarse si están integradas y vigentes:

- HU006 — ciclo de entrenamiento;
- HU007 — checkpoints y resume;
- HU008 — TensorBoard;
- HU009 — smoke E2E real y controlado.

Si alguna dependencia está inconsistente, HU010 debe fallar explícitamente.

---

## 4. Principio arquitectónico

La trazabilidad será **MLOps-light**:

```text
Git/GitHub
   +
config versionada
   +
run_id
   +
run_manifest.json
   +
TensorBoard
   +
checkpoints
   +
resultados estructurados
```

No se implementará un servidor de tracking.

No se utilizará:

- MLflow;
- Weights & Biases;
- Neptune;
- base de datos de experimentos;
- servicio externo de metadata.

GitHub seguirá siendo la fuente de verdad de código y configuración.

---

## 5. Alcance

HU010 debe implementar únicamente la trazabilidad necesaria para HU011.

Artefactos esperados, sujetos a mantener la arquitectura mínima:

```text
3_BattleZone/
├── configs/
│   └── battlezone_config.yaml
├── src/
│   ├── persistence.py          # manifest y persistencia si encaja con responsabilidad actual
│   ├── utils.py                # run_id/runtime/git/hardware si son utilidades pequeñas
│   └── experiment.py           # opcional, solo si evita sobrecargar persistence.py/utils.py
├── tests/
│   └── test_experiment_tracking.py
└── docs/
    ├── hu010_trazabilidad_ligera_experimentos.md
    └── hu010_evidencia_implementacion.md
```

No crear `experiment.py` si `persistence.py` + `utils.py` permiten una solución clara y pequeña.

---

## 6. Fuera de alcance

HU010 no debe implementar:

- entrenamiento largo HU011;
- tuning HU012;
- evaluator formal HU013;
- evaluación de 10 episodios;
- video;
- selección de mejor modelo;
- comparación formal contra baseline;
- PER;
- DDQN;
- REINFORCE;
- MLflow/W&B/Neptune;
- CI/CD nuevo;
- automatización de Google Colab;
- sincronización automática con Google Drive;
- subida automática de artefactos a S3/GCS;
- dashboard de experimentos;
- modificación del preprocessing HU003;
- modificación de la CNN/DQN HU005;
- modificación del contrato de checkpoints HU007 salvo extensión explícitamente compatible de metadata si fuera estrictamente necesaria;
- dependencia con `2_Assault/`.

---

## 7. Identificador de corrida `run_id`

Cada corrida relevante debe tener un identificador único, legible y seguro para rutas.

Formato recomendado:

```text
battlezone-dqn-YYYYMMDD-HHMMSS-<short_git_sha>-<suffix>
```

Ejemplo:

```text
battlezone-dqn-20260902-154500-c7260cf-a1b2
```

Requisitos:

- generado una sola vez al iniciar una corrida NEW;
- estable durante resumes de la misma corrida lógica;
- no depender únicamente de timestamp si existe riesgo de colisión;
- portable como nombre de directorio;
- no contener espacios;
- no usar información sensible;
- debe poder inyectarse explícitamente al reanudar.

No seleccionar automáticamente un `run_id` existente por heurística.

---

## 8. Directorio canónico de resultados

Cada corrida debe usar:

```text
3_BattleZone/results/<run_id>/
```

Estructura mínima:

```text
results/<run_id>/
├── run_manifest.json
└── summaries/
```

Las rutas a logs/checkpoints pueden vivir fuera del directorio si ya existe una convención previa, pero deben quedar registradas en el manifest.

No duplicar checkpoints grandes únicamente para que queden dentro de `results/`.

---

## 9. Manifest obligatorio

Cada corrida relevante debe producir:

```text
results/<run_id>/run_manifest.json
```

El manifest será JSON UTF-8, legible y versionado por schema.

Debe contener como mínimo:

```json
{
  "schema_version": 1,
  "run_id": "...",
  "project": "BattleZone",
  "algorithm": "DQN",
  "status": "created|running|completed|interrupted|failed",
  "mode": "new|resume_full|resume_lightweight",
  "created_at_utc": "...",
  "updated_at_utc": "...",
  "git": {
    "commit": "...",
    "branch": "...",
    "dirty": false
  },
  "environment": {
    "env_id": "ALE/BattleZone-v5",
    "seed": 20260903
  },
  "config": {
    "path": "3_BattleZone/configs/battlezone_config.yaml",
    "sha256": "...",
    "snapshot": {}
  },
  "runtime": {
    "python": "...",
    "gymnasium": "...",
    "ale_py": "...",
    "torch": "...",
    "tensorboard": "..."
  },
  "hardware": {
    "device": "cpu|cuda|mps",
    "gpu_name": null,
    "cpu": "...",
    "ram_gb": null
  },
  "progress": {
    "start_global_step": 0,
    "end_global_step": 0,
    "episode_index": 0,
    "elapsed_seconds": 0.0
  },
  "artifacts": {
    "tensorboard_log_dir": null,
    "input_checkpoint": null,
    "output_checkpoint": null,
    "model_path": null,
    "evaluation_path": null
  },
  "resume": {
    "parent_checkpoint": null,
    "replay_restored": null
  },
  "notes": []
}
```

La estructura exacta puede ajustarse si conserva estos significados.

---

## 10. Versionado del schema

El manifest debe incluir:

```text
schema_version = 1
```

Reglas:

- validar el schema al leer;
- rechazar versiones no soportadas;
- no ignorar campos críticos inválidos;
- permitir agregar campos opcionales sin romper compatibilidad innecesariamente.

No construir un framework genérico de migraciones en HU010.

---

## 11. Snapshot de configuración

Cada manifest debe asociarse a la configuración exacta usada.

Debe registrar:

- ruta del archivo YAML;
- hash SHA-256 del contenido;
- snapshot serializable de la configuración efectiva.

El snapshot debe ser suficiente para auditar la corrida aun si el archivo YAML cambia posteriormente.

No guardar objetos Python no serializables.

No incluir secretos.

---

## 12. Git lineage

Registrar como mínimo:

- commit SHA completo;
- branch/ref si puede resolverse;
- estado `dirty` sí/no.

### Gate para HU011

Una corrida larga de HU011 no debe iniciarse silenciosamente con working tree sucio.

Política recomendada:

- HU010 puede permitir construir manifest con `dirty=true` para tests/smokes;
- HU011 deberá poder activar un gate estricto `require_clean_git=true`.

HU010 debe dejar preparada esa validación.

No modificar archivos Git ni limpiar working tree automáticamente.

---

## 13. Runtime y hardware

Registrar sin inventar valores:

- Python;
- Gymnasium;
- ALE-Py;
- PyTorch;
- TensorBoard;
- sistema operativo/plataforma;
- dispositivo seleccionado;
- CUDA disponible sí/no;
- GPU name cuando exista;
- CPU cuando sea razonable;
- RAM total si ya existe utilidad simple y confiable.

Si un dato no está disponible, usar `null`/`None` o equivalente explícito.

---

## 14. Estados del experimento

Estados mínimos:

```text
created
running
completed
interrupted
failed
```

Transición normal:

```text
created → running → completed
```

Si ocurre excepción:

```text
running → failed
```

Si una sesión termina de forma controlada antes del target final y requiere continuar posteriormente:

```text
running → interrupted
```

No marcar `completed` simplemente porque una sesión terminó si todavía no alcanzó el target lógico de la corrida.

---

## 15. Semántica NEW

Una corrida NEW debe:

1. recibir/generar `run_id`;
2. crear `results/<run_id>/` con `exist_ok=False` o protección equivalente;
3. capturar Git/config/runtime/hardware;
4. escribir manifest inicial `created`;
5. cambiar a `running` al iniciar entrenamiento;
6. registrar `start_global_step=0` o valor explícito;
7. registrar TensorBoard log dir;
8. actualizar progreso y artefactos al terminar la sesión.

No sobreescribir un `run_id` ya existente por defecto.

---

## 16. Semántica RESUME

RESUME_FULL y RESUME_LIGHTWEIGHT deben reutilizar el mismo `run_id` de la corrida lógica.

Al reanudar:

- cargar manifest explícito por `run_id`;
- validar que algoritmo, entorno y configuración crítica sean compatibles;
- registrar modo de resume;
- registrar checkpoint de entrada;
- registrar `start_global_step` restaurado;
- conservar historial previo del manifest;
- actualizar `end_global_step` al terminar la nueva sesión;
- registrar si Replay fue restaurado.

No crear silenciosamente un nuevo `run_id` para la continuación del mismo entrenamiento.

---

## 17. Historial de sesiones

Para preservar trazabilidad multi-sesión sin sobrecomplicar, el manifest debe incluir una lista pequeña equivalente a:

```json
"sessions": [
  {
    "session_index": 1,
    "mode": "new",
    "started_at_utc": "...",
    "ended_at_utc": "...",
    "start_global_step": 0,
    "end_global_step": 32,
    "elapsed_seconds": 12.3,
    "input_checkpoint": null,
    "output_checkpoint": "...",
    "tensorboard_log_dir": "...",
    "device": "cpu",
    "status": "interrupted"
  }
]
```

Cada resume agrega una sesión; no reemplaza la anterior.

El índice debe ser monotónico.

---

## 18. Escritura segura e idempotencia

`run_manifest.json` debe escribirse de forma segura.

Preferencia:

```text
write temp → flush/fsync → atomic replace
```

Puede reutilizar patrones ya existentes en `persistence.py`.

Reglas:

- no dejar JSON parcialmente escrito;
- no borrar manifest previo antes de completar escritura nueva;
- no crear dos corridas con el mismo `run_id` accidentalmente;
- reabrir un manifest existente requiere intención explícita;
- no seleccionar automáticamente el manifest “más reciente”.

---

## 19. Compatibilidad con checkpoints HU007

HU010 no debe romper el schema de checkpoint HU007.

El vínculo se hará preferiblemente desde manifest hacia checkpoint:

```text
manifest.artifacts.input_checkpoint
manifest.artifacts.output_checkpoint
```

Si se decide agregar `run_id` a metadata futura de checkpoint, debe ser:

- opcional;
- backward-compatible;
- cubierto por tests;
- no requerido para cargar checkpoints HU007 existentes.

La opción preferida en HU010 es **no modificar el schema HU007** salvo necesidad demostrada.

---

## 20. Compatibilidad con TensorBoard HU008

El manifest debe registrar explícitamente:

```text
artifacts.tensorboard_log_dir
```

NEW y RESUME de la misma corrida pueden continuar en el mismo directorio de TensorBoard si el caller lo decide explícitamente.

No seleccionar automáticamente el directorio de logs más reciente.

HU010 no cambia tags ni frecuencia de logging HU008.

---

## 21. Configuración centralizada HU010

Agregar una sección equivalente a:

```yaml
tracking:
  enabled: true
  results_dir: "3_BattleZone/results"
  manifest_filename: "run_manifest.json"
  schema_version: 1
  require_clean_git_for_long_run: true
```

Los nombres exactos pueden variar si permanecen claros.

No mezclar estos valores con `smoke:` ni con hiperparámetros DQN.

---

## 22. API mínima esperada

La implementación debe ofrecer una API reusable y pequeña, equivalente a:

```python
create_run_context(...)
load_run_manifest(...)
write_run_manifest(...)
start_session(...)
finish_session(...)
fail_session(...)
```

O una clase compacta:

```python
class ExperimentTracker:
    @classmethod
    def create_new(...): ...
    @classmethod
    def resume(...): ...
    def start_session(...): ...
    def finish_session(...): ...
    def fail_session(...): ...
```

No imponer la clase si funciones simples son suficientes.

Funciones/clases públicas deben usar docstrings estilo Google.

---

## 23. Integración con trainer

HU010 debe evitar contaminar `trainer.py` con lógica de archivos JSON.

Preferencia:

```text
orquestador/caller
  ├── crea tracking context
  ├── crea trainer
  ├── marca session running
  ├── trainer.train(...)
  └── actualiza manifest con TrainingSummary
```

Si se requiere callback mínimo, debe respetar separación de responsabilidades.

`DQNTrainer(..., tracker=None)` no debe convertirse en dependencia rígida salvo justificación fuerte.

---

## 24. Validación barata HU010

HU010 debe validarse con una corrida corta controlada, reutilizando infraestructura HU009.

Flujo mínimo:

```text
NEW 0 → N
manifest created/running/interrupted-or-completed
checkpoint asociado
TensorBoard asociado
↓
RESUME_FULL N → M
same run_id
new session appended
start_global_step=N
end_global_step=M
```

Puede usar FakeEnv para tests deterministas.

No es obligatorio repetir smoke ALE pesado si HU009 ya lo validó y HU010 no modifica entorno/trainer.

---

## 25. Gate de pre-entrenamiento HU011

HU010 debe entregar una función o validación reusable capaz de responder:

```text
READY_FOR_LONG_TRAINING = True|False
```

Debe ser `True` solo si, como mínimo:

- tracking habilitado;
- results dir resoluble/escribible;
- `run_id` válido;
- config cargada y hash calculable;
- Git commit resoluble;
- working tree limpio cuando `require_clean_git_for_long_run=true`;
- TensorBoard configurado;
- checkpointing configurado;
- manifest puede escribirse y releerse;
- algoritmo DQN;
- env_id correcto.

No debe validar todavía calidad de aprendizaje.

---

## 26. Test automatizado

Crear:

```text
3_BattleZone/tests/test_experiment_tracking.py
```

Debe cubrir como mínimo:

1. generación de `run_id` válido;
2. unicidad razonable;
3. creación de directorio de corrida;
4. rechazo de colisión por defecto;
5. manifest schema_version;
6. serialización JSON;
7. lectura del manifest;
8. rechazo de schema incompatible;
9. hash SHA-256 de config;
10. snapshot de config;
11. Git SHA capturado;
12. dirty flag capturado/controlado;
13. runtime capturado;
14. hardware capturado sin inventar valores;
15. NEW crea primera sesión;
16. RESUME conserva `run_id`;
17. RESUME agrega sesión sin borrar anterior;
18. monotonicidad de `session_index`;
19. continuidad `global_step` N→M;
20. input/output checkpoint registrados;
21. TensorBoard path registrado;
22. FULL registra `replay_restored=True`;
23. LIGHTWEIGHT registra `replay_restored=False`;
24. escritura atómica o comportamiento equivalente seguro;
25. excepción actualiza estado `failed` cuando corresponda;
26. `READY_FOR_LONG_TRAINING` PASS en contexto válido;
27. gate falla con dirty Git cuando modo estricto está activo;
28. gate falla si manifest no puede escribirse;
29. no MLflow;
30. no dependencia Assault.

---

## 27. Evidencia de implementación

Crear:

```text
3_BattleZone/docs/hu010_evidencia_implementacion.md
```

Debe registrar resultados reales, no inventados:

1. estado;
2. rama/commit;
3. archivos modificados;
4. schema del manifest;
5. ejemplo real de `run_id` de test/smoke;
6. ruta de manifest de prueba;
7. config hash;
8. Git SHA;
9. runtime/hardware;
10. NEW session;
11. RESUME session;
12. continuidad N→M;
13. checkpoint lineage;
14. TensorBoard lineage;
15. resultado `READY_FOR_LONG_TRAINING`;
16. tests;
17. CA01–CA18;
18. AV01–AV18;
19. scope;
20. limitaciones.

No versionar manifests temporales generados por tests.

---

## 28. Criterios de aceptación

### CA01 — HU009 integrada
HU009 está en `main` y su suite permanece verde.

### CA02 — Config tracking
Existe configuración centralizada de tracking separada de smoke/training.

### CA03 — run_id
Cada NEW relevante obtiene un `run_id` único y válido.

### CA04 — Directorio canónico
Cada corrida usa `results/<run_id>/` sin sobrescribir otra corrida.

### CA05 — Manifest
Se crea `run_manifest.json` válido y versionado.

### CA06 — Código/config
Manifest registra Git SHA y snapshot/hash de configuración.

### CA07 — Runtime/hardware
Manifest registra versiones y hardware disponible sin inventar datos.

### CA08 — Estado
Estados `created/running/completed/interrupted/failed` se manejan explícitamente.

### CA09 — NEW
NEW registra correctamente la primera sesión y progreso.

### CA10 — RESUME
RESUME reutiliza el mismo `run_id` y agrega sesión.

### CA11 — Continuidad
Resume conserva continuidad `global_step=N → M`.

### CA12 — Checkpoint lineage
Input/output checkpoints quedan asociados de forma explícita.

### CA13 — TensorBoard lineage
Log dir TensorBoard queda asociado explícitamente.

### CA14 — Idempotencia
No hay auto-overwrite ni selección implícita de latest run/manifest/checkpoint.

### CA15 — Escritura segura
Manifest no puede quedar parcialmente escrito en flujo normal.

### CA16 — Gate HU011
Existe validación reusable `READY_FOR_LONG_TRAINING` o equivalente.

### CA17 — Regresión
Suite BattleZone completa sigue verde.

### CA18 — Alcance
Sin MLflow, Assault, entrenamiento largo, tuning ni evaluación formal.

---

## 29. Auto-validaciones

### AV01
Confirmar PR/merge HU009 presente en `main`.

### AV02
Validar sección `tracking` en YAML.

### AV03
Generar dos run_id y confirmar formato/ausencia de colisión.

### AV04
Crear manifest en `tmp_path` y releerlo.

### AV05
Validar schema version.

### AV06
Validar SHA-256 de config contra contenido real.

### AV07
Validar Git SHA no vacío.

### AV08
Validar dirty flag sin modificar Git automáticamente.

### AV09
Validar runtime/hardware serializable.

### AV10
Ejecutar NEW controlado y registrar sesión 1.

### AV11
Guardar/asociar checkpoint explícito.

### AV12
Ejecutar RESUME_FULL y registrar sesión 2 con mismo run_id.

### AV13
Verificar session_index monotónico y continuidad N→M.

### AV14
Verificar TensorBoard log dir persistido.

### AV15
Validar LIGHTWEIGHT metadata con `replay_restored=False` mediante test controlado.

### AV16
Forzar error controlado y verificar estado `failed`.

### AV17
Validar gate READY_FOR_LONG_TRAINING en casos PASS/FAIL.

### AV18
Ejecutar suite BattleZone y scope checks completos.

---

## 30. Definition of Done

HU010 puede declararse completada únicamente si:

- HU009 está integrada en `main`;
- existe `run_id` reusable y testeado;
- existe `run_manifest.json` schema v1;
- configuración exacta se conserva vía snapshot + hash;
- Git SHA/dirty state se registra;
- runtime y hardware se registran;
- NEW y RESUME conservan el mismo `run_id`;
- el historial de sesiones se preserva;
- checkpoints y TensorBoard quedan vinculados;
- manifest se escribe de forma segura;
- no hay selección automática ambigua de latest run;
- existe gate de readiness para HU011;
- tests HU010 pasan;
- suite completa BattleZone permanece verde;
- existe evidencia `hu010_evidencia_implementacion.md`;
- no se introduce MLflow, Assault ni HU011+;
- PR auditado y mergeado.

Después de HU010 cerrada, el siguiente paso es:

> **HU011 — Entrenamiento completo DQN en Colab GPU con checkpoints, TensorBoard y trazabilidad persistente.**

---

## 31. Restricción final

HU010 debe terminar con infraestructura lista para entrenamiento largo, pero **no debe iniciar el entrenamiento largo**.

El criterio es:

```text
HU009 E2E PASS
   ↓
HU010 TRACEABILITY PASS
   ↓
READY_FOR_LONG_TRAINING = True
   ↓
HU011 puede iniciar entrenamiento completo
```
