# HU011 — Entrenamiento completo DQN de BattleZone

## 1. Propósito

Ejecutar por primera vez un **entrenamiento real, prolongado, punta a punta y recuperable** del agente DQN de `ALE/BattleZone-v5`, utilizando la infraestructura validada en HU003–HU010.

HU011 transforma el sistema ya validado técnicamente en una corrida de aprendizaje real y trazable, ejecutada preferiblemente en **Google Colab con GPU**, con observabilidad TensorBoard, checkpoints, reanudación y manifiesto de experimento.

El objetivo de HU011 no es demostrar todavía que la configuración sea óptima ni realizar la evaluación académica final. Su objetivo es producir una **corrida de referencia completa** que permita responder:

> ¿Puede el agente DQN entrenarse de forma sostenida, observable, recuperable y reproducible durante un presupuesto computacional real, dejando un modelo/checkpoint final y evidencia suficiente para decidir HU012?

HU011 debe finalizar con un DQN realmente entrenado bajo una configuración de referencia congelada, aun si su desempeño posterior demuestra que requiere optimización.

---

## 2. Fuente de verdad

HU011 debe respetar:

- `enunciado_reto_1.txt`;
- `3_BattleZone/docs/implementacion.md`;
- `3_BattleZone/docs/lineamientos.md`;
- `3_BattleZone/docs/arquitectura.md`;
- `3_BattleZone/docs/hu003_pipeline_reproducible_entorno.md`;
- `3_BattleZone/docs/hu005_nucleo_agente_dqn.md`;
- `3_BattleZone/docs/hu006_ciclo_entrenamiento_dqn.md`;
- `3_BattleZone/docs/hu007_checkpoints_reanudacion_idempotencia.md`;
- `3_BattleZone/docs/hu008_observabilidad_tensorboard.md`;
- `3_BattleZone/docs/hu009_smoke_test_end_to_end.md`;
- `3_BattleZone/docs/hu010_trazabilidad_ligera_experimentos.md`;
- evidencias HU009 y HU010;
- `3_BattleZone/configs/battlezone_config.yaml`.

Algoritmo vigente y obligatorio para HU011:

```text
DQN clásico
```

No implementar DDQN, PER ni REINFORCE dentro de HU011.

---

## 3. Dependencias obligatorias

HU011 solo puede iniciar cuando HU003–HU010 estén disponibles en `main`.

Antes de consumir cómputo intensivo debe comprobarse:

```text
READY_FOR_LONG_TRAINING = True
```

El gate debe ejecutarse sobre el checkout/configuración que realmente se utilizará para entrenar.

Si el gate falla, HU011 no debe iniciar entrenamiento largo.

---

## 4. Alcance funcional

HU011 debe implementar y ejecutar el flujo:

```text
checkout limpio
    ↓
configuración reference_v1 congelada
    ↓
READY_FOR_LONG_TRAINING
    ↓
run_id + run_manifest.json
    ↓
ALE/BattleZone-v5 real
    ↓
DQN + Replay Buffer + epsilon-greedy
    ↓
entrenamiento prolongado
    ↓
TensorBoard
    ↓
checkpoints periódicos
    ↓
resume si la sesión se interrumpe
    ↓
continuar hasta target_global_step
    ↓
checkpoint/modelo final
    ↓
manifest status=completed
    ↓
evidencia HU011
```

Debe existir una única corrida lógica identificada por el mismo `run_id`, aunque el entrenamiento se distribuya entre varias sesiones de Colab.

---

## 5. Fuera de alcance

HU011 NO debe implementar:

- optimización sistemática de hiperparámetros;
- búsqueda grid/random/bayesiana;
- múltiples configuraciones comparativas;
- PER;
- DDQN;
- REINFORCE;
- evaluación formal de al menos 10 episodios;
- comparación formal contra HU002;
- selección académica definitiva del mejor modelo;
- generación de video final;
- reporte final HU014;
- MLflow;
- W&B;
- Neptune;
- dependencia de `2_Assault/`;
- nueva infraestructura CI/CD.

La evaluación formal corresponde a HU013 y la optimización controlada a HU012.

---

## 6. Principio de configuración de referencia

Los valores actuales de `battlezone_config.yaml` fueron suficientes para validar implementación y smoke, pero no constituyen una corrida larga real.

HU011 debe introducir un perfil explícito y versionado:

```text
reference_v1
```

Este perfil es la **primera configuración de entrenamiento real**, no una configuración declarada como óptima.

Debe quedar congelado durante la corrida. Una vez iniciado un `run_id`, los parámetros críticos no se cambian silenciosamente al reanudar.

Cualquier cambio posterior con intención de mejorar desempeño pertenece a HU012.

---

## 7. Presupuesto inicial de entrenamiento

El perfil `reference_v1` debe utilizar como objetivo inicial:

```text
target_global_step = 1_000_000
```

Este valor representa el presupuesto de referencia de HU011.

No implica que un millón de timesteps garantice aprendizaje suficiente ni performance final.

Si una restricción objetiva de Colab/hardware impide completar el presupuesto, HU011 puede distribuirse entre múltiples sesiones usando resume, manteniendo exactamente el mismo `run_id` y target global.

No reducir el target silenciosamente para declarar la HU completada.

Si por una restricción excepcional se requiere revisar el presupuesto, debe documentarse como decisión explícita antes de continuar.

---

## 8. Perfil `reference_v1`

HU011 debe agregar una sección separada de la configuración smoke/validación. No debe reemplazar los valores HU009 usados para smoke.

Configuración objetivo inicial:

```yaml
long_training:
  enabled: true
  profile: "reference_v1"
  target_global_step: 1000000
  require_accelerator: true
  preferred_device: "cuda"

  dqn:
    batch_size: 32
    replay_buffer_capacity: 4096

  training:
    learning_starts: 1024
    train_frequency: 4
    target_sync_interval: 10000
    epsilon:
      start: 1.0
      end: 0.05
      decay_steps: 250000

  checkpointing:
    periodic_mode: "lightweight"
    interval_steps: 25000
    full_milestone_interval_steps: 250000

  tensorboard:
    scalar_log_interval_steps: 100
    flush_interval_steps: 5000
```

Los nombres exactos pueden adaptarse a la arquitectura existente, pero estos significados deben mantenerse.

Los valores son una configuración de referencia operativa para ejecutar HU011. No deben presentarse como hiperparámetros optimizados.

---

## 9. Restricción de memoria del Replay Buffer

La observación vigente es:

```text
(4, 128, 128, 3) uint8
```

El Replay Buffer actual mantiene arrays separados para `state` y `next_state`, por lo que aumentar su capacidad sin control puede consumir varios GB de RAM.

HU011 debe realizar un preflight de memoria antes de iniciar la corrida real.

Debe estimarse al menos:

```text
bytes_por_state
bytes_por_transition aproximados
RAM estimada del Replay a capacidad configurada
RAM total disponible
margen de seguridad
```

Para `reference_v1`, `replay_buffer_capacity=4096` es el máximo inicial esperado salvo evidencia medida que justifique otro valor antes de iniciar la corrida.

No rediseñar Replay Buffer en HU011 salvo que exista un bloqueo real que impida entrenar.

Si se requiere rediseño estructural de almacenamiento, debe documentarse explícitamente porque altera un componente estable.

---

## 10. Batch size

El perfil `reference_v1` utilizará:

```text
batch_size = 32
```

Este valor aplica solo al entrenamiento largo.

Los smoke tests HU009 deben mantener su configuración controlada y barata.

La implementación debe evitar que los valores de `long_training` modifiquen accidentalmente los contratos smoke existentes.

---

## 11. Learning starts

La corrida larga debe comenzar actualizaciones después de:

```text
learning_starts = 1024
```

Antes de este punto el agente debe poblar Replay Buffer y actuar según epsilon-greedy.

No iniciar optimización antes de tener experiencia suficiente para satisfacer el batch gate.

---

## 12. Exploración

La corrida `reference_v1` utiliza:

```text
epsilon_start = 1.0
epsilon_end = 0.05
epsilon_decay_steps = 250000
```

La programación depende exclusivamente de `global_step`.

Al reanudar:

```text
epsilon(resume) = schedule(restored_global_step)
```

No reiniciar epsilon a 1.0 después de una interrupción.

---

## 13. Target Network

La sincronización de Target Network para `reference_v1` será:

```text
target_sync_interval = 10000 global steps
```

Debe utilizar el mecanismo ya implementado en HU006.

No introducir soft-update ni Double-DQN.

---

## 14. Dispositivo de entrenamiento

La ejecución completa HU011 está diseñada para Colab GPU.

Preflight requerido:

```text
torch.cuda.is_available() == True
```

La corrida larga no debe iniciar accidentalmente en CPU cuando `require_accelerator=true`.

Si Colab entrega GPU diferente entre sesiones, registrar cada dispositivo en `sessions[]` del manifest.

No depender del modelo específico de GPU.

Para pruebas locales se permiten CPU/MPS, pero únicamente con presupuestos cortos de validación; no deben confundirse con la corrida HU011 completa.

---

## 15. Checkout Git limpio

Antes de crear/iniciar la corrida real:

```text
git.dirty = false
```

Debe ejecutarse el gate HU010 real, no una simulación.

El `git.commit` registrado será el commit exacto que produjo la corrida.

No limpiar automáticamente Git mediante reset/clean.

Si está dirty, detener y corregir el estado de forma explícita.

---

## 16. `run_id`

NEW debe generar un único `run_id` HU010.

Ejemplo conceptual:

```text
battlezone-dqn-YYYYMMDD-HHMMSS-<sha>-<suffix>
```

Todos los resumes posteriores de HU011 deben reutilizar exactamente ese `run_id`.

Prohibido:

- generar un nuevo run_id por sesión;
- seleccionar automáticamente latest run;
- iniciar desde un checkpoint sin manifest compatible.

---

## 17. Manifiesto

La corrida debe utilizar:

```text
results/<run_id>/run_manifest.json
```

Debe registrar las sesiones reales de entrenamiento.

Al iniciar NEW:

```text
status = running
mode = new
start_global_step = 0
```

Si Colab finaliza antes del target:

```text
status = interrupted
```

Al reanudar:

```text
mode = resume_full | resume_lightweight
same run_id
start_global_step = previous end_global_step
```

Solo cuando:

```text
end_global_step >= target_global_step
```

puede marcarse:

```text
status = completed
```

---

## 18. Persistencia fuera de `/content`

La evidencia principal de HU011 no puede depender exclusivamente del filesystem efímero de Colab.

Antes del entrenamiento real debe definirse un `persistent_root` explícito.

En Colab puede apuntar a Google Drive u otro almacenamiento persistente autorizado.

Debe persistir como mínimo:

```text
run_manifest.json
checkpoints seleccionados
TensorBoard logs
checkpoint/modelo final
resumen HU011
```

No es obligatorio versionar estos artefactos pesados en GitHub.

GitHub continúa siendo la fuente de verdad para código/configuración.

---

## 19. Checkpoints periódicos

Por el tamaño potencial del Replay Buffer, HU011 no debe guardar FULL checkpoints en cada intervalo corto.

Política `reference_v1`:

```text
cada 25_000 steps:
    LIGHTWEIGHT

cada 250_000 steps:
    FULL cuando sea viable
```

También debe existir un checkpoint al finalizar exitosamente.

Si FULL no es viable por RAM/almacenamiento, debe documentarse la razón medida y mantenerse recoverability mediante LIGHTWEIGHT.

No ocultar esa degradación.

La selección de checkpoint para resume siempre debe ser explícita.

---

## 20. Riesgo de FULL checkpoint

El método actual `ReplayBuffer.state_dict()` copia los arrays completos antes de serializar.

Eso puede aumentar temporalmente la memoria durante un FULL checkpoint.

HU011 debe medir/observar RAM antes de habilitar FULL periódico con `capacity=4096`.

Si el preflight muestra riesgo de OOM:

1. mantener checkpoints periódicos LIGHTWEIGHT;
2. registrar la limitación;
3. no forzar FULL hasta provocar fallo;
4. conservar al menos un mecanismo validado de resume.

No modificar HU007 silenciosamente para eludir el contrato.

---

## 21. Resume después de interrupción

HU011 debe demostrar en la corrida real que puede continuar después de al menos una separación de sesión lógica.

No es necesario provocar la caída de Colab.

Puede realizarse:

```text
sesión 1
→ checkpoint
→ cierre controlado
→ nueva sesión
→ restore explícito
→ continuar mismo run_id
```

Esto debe demostrar que la arquitectura funciona en el escenario real para el cual fue diseñada.

---

## 22. Política de resume preferida

Orden recomendado:

1. usar FULL si existe un FULL compatible y su carga es viable;
2. usar LIGHTWEIGHT explícitamente si FULL no existe o no es viable;
3. registrar `replay_restored` correctamente;
4. no seleccionar checkpoints automáticamente.

Después de LIGHTWEIGHT, los updates deben esperar hasta reconstruir Replay suficiente según el batch/learning gate.

---

## 23. TensorBoard

HU011 debe producir curvas reales durante todo el entrenamiento.

Tags mínimos existentes:

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

La corrida larga debe permitir inspeccionar:

- reward por episodio;
- reward media móvil;
- loss;
- Q-value medio;
- epsilon;
- Replay size;
- episode length;
- continuidad entre sesiones.

No añadir métricas sin valor diagnóstico claro.

---

## 24. Continuidad TensorBoard

NEW y RESUME pertenecientes al mismo `run_id` deben quedar vinculados al mismo linaje de logs.

Los `global_step` de TensorBoard no pueden reiniciarse a cero al reanudar.

Debe existir al menos un scalar con step posterior al checkpoint de la sesión anterior.

---

## 25. Monitoreo operativo

Durante la corrida debe registrarse periódicamente evidencia suficiente para diagnosticar estabilidad computacional:

- `global_step`;
- episodios completados;
- updates;
- epsilon;
- Replay size;
- loss reciente;
- reward media móvil cuando exista;
- tiempo transcurrido;
- throughput aproximado (`steps/second`) si puede obtenerse sin complejidad innecesaria;
- RAM disponible/usada si la utilidad ya existe;
- GPU/device.

No imprimir cada step.

---

## 26. Fail-fast

Detener explícitamente la corrida ante:

- loss NaN o Inf;
- Q-values NaN o Inf;
- incompatibilidad de checkpoint;
- manifest incompatible;
- `global_step` regresivo;
- GPU requerida pero no disponible;
- config crítica cambiada durante resume;
- checkpoint corrupto;
- error de escritura persistente que impida mantener recoverability.

No convertir fallos críticos en warnings silenciosos.

---

## 27. Interrupciones controladas

`KeyboardInterrupt` o cierre controlado debe intentar:

1. finalizar correctamente la sesión en el manifest;
2. marcar `interrupted`;
3. persistir un checkpoint LIGHTWEIGHT si es seguro;
4. flush/close TensorBoard;
5. cerrar environment.

Una excepción no controlada debe registrar `failed` cuando sea posible sin esconder la excepción original.

---

## 28. Idempotencia

Reejecutar celdas/orquestación no puede:

- sobrescribir un run existente por defecto;
- reiniciar `global_step` de un resume;
- generar un run_id nuevo sin intención;
- eliminar TensorBoard logs;
- sobrescribir checkpoints válidos;
- seleccionar automáticamente artefactos ambiguos.

NEW y RESUME deben ser modos explícitos.

---

## 29. Orquestación

HU011 puede agregar un módulo pequeño de orquestación si evita contaminar notebook y trainer, por ejemplo:

```text
src/training_run.py
```

Responsabilidad posible:

```text
tracker
+ trainer
+ checkpoint lifecycle
+ TensorBoard lifecycle
+ session lifecycle
```

No duplicar lógica de:

- `trainer.py`;
- `persistence.py`;
- `experiment.py`;
- `callbacks.py`;
- `environment.py`.

El trainer debe continuar desacoplado de detalles de filesystem/manifests.

---

## 30. Notebook de Colab

HU011 debe proporcionar una ruta clara y ejecutable desde notebook para realizar la corrida real.

Preferencia: evolucionar el notebook BattleZone existente como orquestador, sin copiar lógica reusable.

El notebook debe permitir explícitamente:

```text
MODE = "new"
```

o:

```text
MODE = "resume_full"
RUN_ID = "..."
CHECKPOINT_PATH = "..."
```

o:

```text
MODE = "resume_lightweight"
```

No usar auto-discovery de latest run/checkpoint.

---

## 31. Preflight Colab obligatorio

Antes del entrenamiento largo, el notebook/script debe mostrar y validar:

```text
Git commit
Git clean
config SHA256
profile = reference_v1
algorithm = DQN
env_id = ALE/BattleZone-v5
CUDA available
GPU name
RAM total
Replay RAM estimate
target_global_step
checkpoint policy
persistent_root
run_id / NEW mode
READY_FOR_LONG_TRAINING
```

La corrida solo inicia si todos los gates críticos pasan.

---

## 32. Separación local vs Colab

Local se utiliza para:

- tests;
- compileall;
- pruebas cortas de integración;
- revisión de configuración;
- validación del orquestador.

Colab GPU se utiliza para:

- ejecución HU011 real prolongada.

No considerar una corrida local corta como evidencia de entrenamiento completo.

---

## 33. Criterio de entrenamiento completado

Una corrida HU011 se considera computacionalmente completada cuando:

```text
manifest.status == completed
AND
manifest.progress.end_global_step >= 1_000_000
AND
final checkpoint/model exists
AND
TensorBoard contains long-run data
AND
0 unrecovered critical errors
```

Esto no implica aprobar performance del agente.

---

## 34. Resultado de aprendizaje esperado

HU011 debe producir datos suficientes para inspeccionar si existe señal de aprendizaje.

Se debe reportar de manera descriptiva:

- reward inicial vs tramo final;
- reward moving average;
- tendencia de loss;
- tendencia Q-values;
- epsilon final;
- episodios completados;
- proporción/registro de recompensas positivas si ya puede obtenerse de métricas existentes.

No utilizar estos datos como evaluación formal HU013.

No declarar “agente aprobado” solo por una curva ascendente.

---

## 35. Gate hacia HU012

Al finalizar HU011 debe emitirse una decisión:

```text
READY_FOR_HU012 = True | False
```

`True` significa:

- la corrida reference_v1 completó el presupuesto o quedó técnicamente interpretable;
- existen curvas y artefactos suficientes para formular hipótesis de optimización;
- no hay fallos estructurales pendientes.

No significa que tuning sea obligatorio. HU012 determinará si conviene modificar parámetros.

---

## 36. No evaluación formal

HU011 puede observar rewards del entrenamiento, pero no debe ejecutar el protocolo formal de ≥10 episodios independientes de HU013 para declarar performance.

No comparar formalmente contra el baseline HU002 dentro de HU011.

Puede mencionarse el baseline solo como contexto.

---

## 37. Artefactos esperados

Código/configuración versionada:

```text
3_BattleZone/
├── configs/
│   └── battlezone_config.yaml
├── src/
│   └── training_run.py           # solo si aporta separación real
├── tests/
│   └── test_full_training.py
├── docs/
│   ├── hu011_entrenamiento_completo_dqn.md
│   └── hu011_evidencia_implementacion.md
└── pipeline_battlezone.ipynb     # si se usa como orquestador Colab
```

Artefactos runtime no versionados rutinariamente:

```text
results/<run_id>/run_manifest.json
logs/<run_id>/...
checkpoints/<run_id>/...
models/<run_id>/...
```

---

## 38. Modelo/checkpoint final

Al alcanzar el target debe persistirse explícitamente un artefacto final identificable.

Puede ser:

```text
models/<run_id>/battlezone_dqn_final.pt
```

O un checkpoint final equivalente si contiene el estado necesario.

Debe registrarse su ruta en el manifest.

No seleccionar automáticamente “best model” en HU011.

---

## 39. Tests automatizados

Crear o extender pruebas para validar la orquestación sin ejecutar 1M timesteps.

Archivo esperado:

```text
3_BattleZone/tests/test_full_training.py
```

Debe utilizar overrides pequeños/tmp_path/fakes cuando corresponda.

Cobertura mínima:

1. carga profile `reference_v1`;
2. target > smoke target;
3. long profile no modifica smoke config;
4. GPU-required gate;
5. memory estimation;
6. NEW crea misma arquitectura HU010;
7. session lifecycle;
8. periodic checkpoint decision;
9. FULL milestone decision;
10. explicit resume path;
11. same run_id on resume;
12. TensorBoard step continuity;
13. completion only at target;
14. interruption status;
15. failure status;
16. cleanup logger/env;
17. no overwrite;
18. no latest auto-selection;
19. final artifact linkage;
20. no MLflow;
21. no Assault dependency;
22. DQN only.

---

## 40. Validación real previa al largo

Antes de lanzar 1M steps, ejecutar una validación real barata del nuevo orquestador con ALE real y un override pequeño, por ejemplo:

```text
128–512 global steps
```

Esta validación no sustituye HU009; solo comprueba que la nueva capa HU011 no rompió integración.

Debe usar un run temporal/no confundible con la corrida reference_v1 final.

---

## 41. Corrida real obligatoria

La HU no puede declararse completada únicamente por tests automatizados.

Debe existir evidencia de una ejecución real de `ALE/BattleZone-v5` con DQN usando el perfil `reference_v1` y Colab GPU.

Si la implementación del código está lista pero la corrida larga aún no se ha ejecutado, el estado debe ser:

```text
HU011 IMPLEMENTADA — ENTRENAMIENTO PENDIENTE
```

No marcar `[COMPLETADA]` hasta tener la evidencia real exigida.

---

## 42. Evidencia de implementación

Crear:

```text
3_BattleZone/docs/hu011_evidencia_implementacion.md
```

Debe separar claramente:

### Implementación

- branch/commit;
- config reference_v1;
- tests;
- preflight;
- orquestador;
- persistencia;
- checkpoint policy.

### Ejecución real

- run_id;
- commit Git;
- config SHA256;
- Colab runtime;
- GPU;
- RAM;
- fecha inicio/fin;
- número de sesiones;
- start/end global step por sesión;
- total global step;
- elapsed time;
- checkpoints;
- modo de resume;
- TensorBoard log path;
- final artifact path;
- episodios completados;
- reward summary descriptivo;
- loss/Q summary descriptivo;
- limitaciones/incidentes.

No inventar datos aún no ejecutados.

---

## 43. Criterios de aceptación

### CA01 — Dependencias
HU003–HU010 están disponibles desde `main` y se reutilizan sin duplicación.

### CA02 — DQN
La corrida usa exclusivamente DQN clásico.

### CA03 — Perfil
Existe `reference_v1` versionado y separado de smoke.

### CA04 — Presupuesto
`target_global_step=1_000_000` está explícito y congelado para la corrida.

### CA05 — Preflight
READY_FOR_LONG_TRAINING debe pasar sobre el entorno real antes de entrenar.

### CA06 — GPU
La corrida real usa CUDA cuando `require_accelerator=true`.

### CA07 — Memoria
Replay Buffer tiene estimación/preflight de RAM y no excede el presupuesto sin justificación.

### CA08 — NEW
Se crea run_id/manifest real y entrenamiento inicia desde 0.

### CA09 — TensorBoard
Curvas reales se escriben con global_step continuo.

### CA10 — Checkpoints
Existe política periódica LIGHTWEIGHT y milestones FULL cuando sean viables.

### CA11 — Persistencia
Artefactos críticos sobreviven al runtime efímero de Colab.

### CA12 — Resume
Se demuestra al menos una continuidad real entre sesiones con mismo run_id.

### CA13 — Epsilon
Exploración continúa desde global_step restaurado.

### CA14 — Replay
FULL restaura Replay; LIGHTWEIGHT registra reconstrucción según modo utilizado.

### CA15 — Completion
Manifest solo pasa a completed al alcanzar target.

### CA16 — Modelo final
Existe artefacto/checkpoint final y está referenciado en manifest.

### CA17 — Observabilidad
Existen métricas suficientes para analizar estabilidad y señal de aprendizaje.

### CA18 — Tests
Suite automatizada HU011 y regresión BattleZone están verdes.

### CA19 — Idempotencia
No hay sobrescritura/auto-latest/reinicio silencioso.

### CA20 — Scope
No se introduce tuning HU012, evaluación HU013, MLflow ni dependencia Assault.

### CA21 — Evidencia real
Existe evidencia documentada de la corrida real Colab.

### CA22 — Gate HU012
Se emite READY_FOR_HU012 con justificación técnica.

---

## 44. Auto-validaciones

### AV01
Confirmar que HU010 está mergeada en `main`.

### AV02
Confirmar checkout limpio y Git SHA real.

### AV03
Validar config profile `reference_v1` y que smoke permanece intacto.

### AV04
Validar DQN + BattleZone environment contract.

### AV05
Calcular Replay RAM estimada y registrar RAM disponible.

### AV06
Validar CUDA y GPU real para la corrida larga.

### AV07
Ejecutar READY_FOR_LONG_TRAINING real.

### AV08
Ejecutar integración corta ALE con nueva orquestación.

### AV09
Crear run_id y manifest NEW real.

### AV10
Validar TensorBoard con EventAccumulator durante integración corta.

### AV11
Validar checkpoint LIGHTWEIGHT periódico controlado.

### AV12
Validar FULL milestone controlado cuando sea viable.

### AV13
Validar resume explícito same run_id.

### AV14
Validar continuidad global_step/epsilon/TensorBoard.

### AV15
Validar interrupt/failure manifest state en tests.

### AV16
Ejecutar test HU011.

### AV17
Ejecutar regresión focal HU006–HU011.

### AV18
Ejecutar suite completa BattleZone.

### AV19
Ejecutar `git diff --check`.

### AV20
Confirmar ausencia de archivos Assault en el diff.

### AV21
Confirmar ausencia de MLflow/PER/DDQN/REINFORCE nuevos.

### AV22
Ejecutar corrida real reference_v1 en Colab GPU.

### AV23
Verificar manifest final y sesiones reales.

### AV24
Verificar checkpoint/modelo final existente.

### AV25
Verificar TensorBoard contiene datos hasta tramo final.

### AV26
Documentar métricas descriptivas de aprendizaje sin evaluación formal.

### AV27
Emitir READY_FOR_HU012.

---

## 45. Definition of Done

HU011 solo se considera `[COMPLETADA]` cuando:

- HU010 está integrada en main;
- profile `reference_v1` existe y está versionado;
- orquestación de entrenamiento largo está implementada;
- tests HU011 pasan;
- suite BattleZone completa pasa;
- preflight real pasa;
- corrida real Colab GPU fue ejecutada;
- misma corrida conserva run_id entre sesiones;
- checkpoints/resume reales fueron usados o demostrados en esa corrida;
- `global_step >= 1_000_000`;
- manifest queda `completed`;
- TensorBoard contiene curvas de la corrida;
- artefacto final existe;
- evidencia HU011 contiene resultados reales;
- no se ejecutó tuning formal;
- no se ejecutó evaluación formal HU013;
- no hay cambios Assault;
- no hay MLflow;
- PR permanece abierto hasta auditoría final;
- auditoría final aprueba integración.

Si solo está lista la implementación pero todavía falta ejecutar el entrenamiento largo:

```text
HU011 IMPLEMENTADA — ENTRENAMIENTO PENDIENTE
```

---

## 46. Estrategia Git/PR

Flujo obligatorio:

```text
main
  ↓
feature/battlezone-hu011-full-training
  ↓
documentación HU011
  ↓
PR DRAFT
  ↓
implementación Codex
  ↓
validaciones locales
  ↓
preflight Colab
  ↓
entrenamiento reference_v1
  ↓
evidencia real
  ↓
mark ready for review
  ↓
auditoría
  ↓
merge
```

El PR NO debe mergearse cuando solo contiene la definición documental.

Mientras no exista evidencia real de entrenamiento, debe permanecer DRAFT.

---

## 47. Regla de parada para Codex

Antes de implementar, Codex debe consultar el estado del PR HU011.

Si el PR esperado está `MERGED` o `CLOSED`:

```text
BLOCKED_PR_STATE
```

Y debe detenerse antes de crear commits técnicos.

No crear otro PR automáticamente salvo autorización explícita.

---

## 48. Resultado esperado

Al cerrar HU011 debe existir por primera vez en BattleZone:

```text
DQN realmente entrenado
+
1_000_000 global steps reference_v1
+
run_manifest completo
+
TensorBoard real
+
checkpoints/recoverability
+
modelo/checkpoint final
+
evidencia Colab GPU
```

HU012 utilizará esta corrida como base empírica para decidir si es necesario optimizar hiperparámetros o explorar DQN+PER bajo una hipótesis explícita.
