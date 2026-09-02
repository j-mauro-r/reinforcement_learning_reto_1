# HU009 — Smoke test end-to-end del agente BattleZone

## 1. Propósito

Validar de forma integrada, barata y reproducible que el sistema BattleZone construido en HU003–HU008 funciona de extremo a extremo antes de iniciar trazabilidad HU010, entrenamiento largo HU011, tuning HU012 o evaluación formal HU013.

HU009 actúa como **gate técnico obligatorio** previo a cualquier uso intensivo de cómputo.

Debe demostrar que, usando `ALE/BattleZone-v5` real y el pipeline actual del proyecto, el flujo completo puede:

1. cargar configuración;
2. crear entorno real mediante la factory HU003;
3. crear agente DQN HU005;
4. ejecutar entrenamiento HU006;
5. poblar Replay Buffer;
6. ejecutar al menos una actualización real del agente;
7. sincronizar Target Network;
8. generar TensorBoard HU008;
9. guardar checkpoint HU007;
10. restaurar checkpoint;
11. reanudar entrenamiento desde `global_step=N` hasta `M>N`;
12. producir evidencia estructurada de que todos los componentes anteriores funcionaron sin errores.

HU009 **no busca demostrar performance del agente**.

---

## 2. Fuentes de verdad

HU009 debe respetar:

- `enunciado_reto_1.txt`;
- `3_BattleZone/docs/implementacion.md`;
- `3_BattleZone/docs/lineamientos.md`;
- `3_BattleZone/docs/arquitectura.md`;
- `3_BattleZone/docs/hu003_pipeline_reproducible_entorno.md`;
- `3_BattleZone/docs/hu005_nucleo_agente_dqn.md`;
- `3_BattleZone/docs/hu006_ciclo_entrenamiento_dqn.md`;
- `3_BattleZone/docs/hu007_checkpoints_reanudacion_idempotencia.md`;
- `3_BattleZone/docs/hu008_observabilidad_tensorboard.md`;
- evidencias de implementación HU006–HU008;
- `3_BattleZone/configs/battlezone_config.yaml`.

La implementación debe utilizar únicamente el algoritmo BattleZone vigente: **DQN clásico**.

---

## 3. Dependencias obligatorias

HU009 solo puede implementarse si están disponibles en `main`:

- HU003 — entorno/preprocessing reproducible;
- HU005 — núcleo DQN;
- HU006 — trainer;
- HU007 — checkpoint/resume;
- HU008 — TensorBoard.

Si una dependencia falta o está inconsistente, HU009 debe fallar explícitamente en lugar de ocultar el problema.

---

## 4. Alcance

HU009 debe integrar componentes existentes.

Debe agregar únicamente la lógica mínima necesaria para ejecutar y validar un smoke test E2E.

Artefactos esperados:

```text
3_BattleZone/
├── configs/
│   └── battlezone_config.yaml
├── src/
│   └── smoke.py                 # solo si reduce duplicación
├── tests/
│   └── test_smoke.py
└── docs/
    ├── hu009_smoke_test_end_to_end.md
    └── hu009_evidencia_implementacion.md
```

`src/smoke.py` es opcional. Debe crearse solo si evita que `test_smoke.py` o el notebook acumulen lógica de orquestación repetida.

---

## 5. Fuera de alcance

HU009 no debe implementar:

- MLflow;
- W&B o Neptune;
- `run_manifest.json` definitivo;
- `run_id` definitivo HU010;
- evaluator formal;
- evaluación de 10 episodios;
- comparación formal contra baseline HU002;
- generación de video;
- selección de mejor modelo;
- entrenamiento largo;
- tuning de hiperparámetros;
- PER;
- DDQN;
- REINFORCE;
- infraestructura CI/CD nueva;
- dependencia hacia `2_Assault/`.

---

## 6. Principio de validación

El smoke test debe responder:

> ¿Podemos ejecutar el sistema completo con BattleZone real, producir aprendizaje técnico verificable, persistir estado y reanudar sin romper contratos?

No debe responder:

> ¿El agente ya aprendió a jugar bien?

---

## 7. Tipo de smoke test

Se requieren dos niveles:

### 7.1 Smoke controlado automatizado

Usará test doubles/fakes donde sea útil para validar contratos rápidamente.

Objetivo:

- detectar regresiones lógicas;
- asegurar comportamiento determinista de tests;
- ejecutarse dentro de la suite normal.

### 7.2 Smoke real integrado

Usará realmente:

```text
ALE/BattleZone-v5
```

mediante la factory HU003.

Objetivo:

- detectar problemas reales de integración que los fakes no pueden descubrir;
- comprobar dependencias ALE/Gymnasium/PyTorch/TensorBoard;
- comprobar shapes y dtypes reales;
- ejecutar update, checkpoint y resume reales.

Debe ser corto y barato.

---

## 8. Configuración smoke

Agregar configuración versionada mínima, preferiblemente:

```yaml
smoke:
  enabled: true
  total_timesteps_new: 32
  total_timesteps_resume: 48
  learning_starts: 8
  train_frequency: 4
  target_sync_interval: 16
  checkpoint_mode: full
```

Los valores pueden ajustarse si la implementación real requiere una cantidad ligeramente distinta para garantizar al menos un update.

Reglas:

- no modificar los hiperparámetros base de entrenamiento largo;
- los overrides smoke deben estar claramente separados;
- no dispersar constantes mágicas en tests/notebook.

---

## 9. Contrato de observación real

Antes de entrenar, el smoke debe verificar:

```text
shape = (4, 128, 128, 3)
dtype = uint8
action_dim = 18
```

según HU003.

También debe verificar:

- entorno creado por factory oficial BattleZone;
- `frameskip=4` efectivo una sola vez;
- sticky actions según configuración vigente;
- no render durante training smoke.

---

## 10. Flujo NEW obligatorio

El smoke real NEW debe:

1. cargar config;
2. construir entorno;
3. construir agente DQN;
4. construir TensorBoard logger;
5. construir trainer;
6. entrenar desde step `0` hasta `N`;
7. cerrar entorno/logger correctamente.

Durante esta fase debe verificarse:

- `global_step == N`;
- Replay Buffer contiene transiciones;
- ocurrió al menos un optimizer update;
- `last_loss` existe y es finito;
- `q_value_mean` fue emitido vía TensorBoard o evidencia equivalente;
- ocurrió al menos una sincronización de Target Network si `N` lo permite;
- epsilon disminuyó o evolucionó según schedule vigente;
- TensorBoard generó eventos válidos.

---

## 11. Checkpoint FULL obligatorio

Después del NEW:

- guardar checkpoint FULL explícito;
- verificar archivo existente;
- tamaño > 0;
- cargarlo mediante HU007;
- validar compatibilidad;
- restaurar agente + optimizer + Replay + progreso.

Debe demostrarse:

```text
restored_global_step == N
replay_restored == True
restored_replay_size > 0
```

No seleccionar automáticamente “latest checkpoint”.

---

## 12. Resume obligatorio

Crear de nuevo:

- entorno;
- agente;
- trainer;
- logger.

Restaurar checkpoint FULL y continuar hasta:

```text
M > N
```

Debe verificarse:

```text
start_global_step == N
final_global_step == M
```

Además:

- no reiniciar epsilon;
- no reiniciar Replay Buffer;
- update posteriores al resume son posibles;
- TensorBoard continúa con steps > N;
- no se promete continuidad frame-a-frame del episodio ALE interrumpido.

---

## 13. Checkpoint LIGHTWEIGHT

HU009 debe incluir al menos un test automatizado que verifique que el flujo E2E también soporta:

```text
RESUME_LIGHTWEIGHT
```

Debe demostrar:

- `global_step` restaurado;
- Replay vacío inmediatamente después de restore;
- `replay_restored=False`;
- entrenamiento puede continuar;
- updates esperan hasta reconstruir suficiente Replay;
- TensorBoard sigue usando el `global_step` restaurado.

No es obligatorio repetir una segunda corrida real ALE completa si la cobertura controlada es suficiente y HU008 ya validó el modo real.

---

## 14. TensorBoard gate

El smoke real debe comprobar con `EventAccumulator`:

- al menos un archivo `events.out.tfevents.*`;
- tamaño > 0;
- tags disponibles;
- al menos un scalar de `train/loss`;
- al menos un scalar de `train/q_value_mean`;
- scalars de `train/epsilon`;
- scalars de `train/replay_size`;
- `max_logged_step == M` o coherente con el intervalo configurado;
- existe al menos un step posterior a `N`.

Los tags de episodio solo serán obligatorios si efectivamente termina un episodio durante el smoke. Su ausencia no es fallo si la corrida corta no completa episodio.

---

## 15. Recursos y estabilidad

El smoke debe recoger evidencia básica de ejecución:

- dispositivo usado (`cpu`/`cuda`);
- RAM del proceso o aproximación razonable si ya existe utilidad disponible;
- GPU disponible sí/no;
- tiempo total NEW;
- tiempo total RESUME;
- tamaño checkpoint FULL;
- tamaño Replay al final de NEW y RESUME.

No implementar todavía el manifiesto persistente HU010.

Si no existe utilidad ligera de memoria, no crear infraestructura compleja solo para esta métrica.

---

## 16. Reproducibilidad

El smoke debe usar seed explícita.

Debe registrar en evidencia:

- seed;
- versión Python;
- Gymnasium;
- ALE-Py;
- PyTorch;
- TensorBoard;
- device.

No prometer determinismo absoluto debido a ALE/sticky actions/GPU.

---

## 17. Idempotencia

Reejecutar smoke no debe destruir evidencia válida.

Reglas:

- usar directorios temporales en tests automatizados;
- para validación manual/real, usar subdirectorio identificable de smoke;
- no sobreescribir checkpoint previo por defecto;
- no borrar logs previos;
- no seleccionar artefactos ambiguos automáticamente.

---

## 18. Test automatizado `test_smoke.py`

Crear:

```text
3_BattleZone/tests/test_smoke.py
```

Debe cubrir como mínimo:

1. carga de config smoke;
2. creación de agente/trainer/logger;
3. NEW corto controlado;
4. Replay > 0;
5. optimizer update > 0;
6. loss finita;
7. target sync > 0 cuando corresponda;
8. TensorBoard event file válido;
9. tags esenciales;
10. checkpoint FULL save;
11. checkpoint FULL load;
12. restore `global_step`;
13. restore Replay FULL;
14. resume FULL `N -> M`;
15. TensorBoard step continuity;
16. LIGHTWEIGHT restore;
17. LIGHTWEIGHT Replay vacío;
18. LIGHTWEIGHT batch gate;
19. `logger=None` no rompe trainer;
20. ausencia de MLflow;
21. ausencia de dependencia Assault.

---

## 19. Test real ALE

Debe existir al menos una prueba o script controlado que use la factory real HU003.

Puede ser:

- test marcado explícitamente como integración, o
- función/script smoke ejecutado manualmente y documentado.

Preferencia para evitar que la suite normal dependa de ROM/runtime pesado:

```text
unit/integration fake smoke → suite normal
real ALE smoke → comando explícito documentado
```

No ocultar si el entorno local no permite correr ALE real.

---

## 20. Resultado estructurado

Si se crea `src/smoke.py`, exponer un resultado pequeño equivalente a:

```python
@dataclass(frozen=True)
class SmokeResult:
    start_step: int
    checkpoint_step: int
    final_step: int
    updates_new: int
    updates_resume: int
    replay_size_new: int
    replay_size_resume: int
    last_loss: float | None
    checkpoint_path: str
    tensorboard_log_dir: str
```

No incluir manifiesto HU010.

---

## 21. Manejo de errores

El smoke debe fallar explícitamente si ocurre cualquiera de estos casos:

- observation shape incorrecto;
- dtype incorrecto;
- action_dim incorrecto;
- no hay updates cuando deberían existir;
- loss NaN/Inf;
- checkpoint no se crea;
- checkpoint incompatible;
- resume no continúa desde N;
- TensorBoard no produce eventos;
- TensorBoard reinicia steps tras resume;
- Replay FULL no se restaura;
- LIGHTWEIGHT restaura Replay accidentalmente.

No convertir errores técnicos en warnings silenciosos.

---

## 22. Criterios de aceptación

### CA01 — Dependencias
HU003–HU008 están disponibles y se utilizan sin duplicar lógica.

### CA02 — Entorno real
Smoke real crea `ALE/BattleZone-v5` mediante factory HU003.

### CA03 — Contrato observación
Shape/dtype/action_dim coinciden con HU003.

### CA04 — NEW
El entrenamiento corto NEW alcanza `N` sin excepción.

### CA05 — Replay
Replay contiene experiencia después de NEW.

### CA06 — Update real
Existe al menos un optimizer update con loss finita.

### CA07 — Target sync
Existe al menos una sincronización cuando la ventana smoke lo permite.

### CA08 — TensorBoard
Se producen eventos TensorBoard legibles.

### CA09 — FULL checkpoint
Checkpoint FULL se guarda y carga correctamente.

### CA10 — FULL restore
Se recuperan progreso y Replay.

### CA11 — Resume
El flujo continúa exactamente `N -> M`.

### CA12 — TensorBoard resume
Los steps continúan después de N.

### CA13 — LIGHTWEIGHT
Restore lightweight conserva progreso pero Replay inicia vacío.

### CA14 — Idempotencia
No hay auto-overwrite ni selección ambigua.

### CA15 — Regresión
La suite BattleZone completa sigue verde.

### CA16 — Alcance
No se introduce HU010+, MLflow, Assault ni algoritmos fuera de DQN.

---

## 23. Auto-validaciones

### AV01
Confirmar dependencias HU003–HU008 presentes.

### AV02
Verificar entorno real y preprocessing mediante factory oficial.

### AV03
Validar `(4,128,128,3)`, `uint8`, action space 18.

### AV04
NEW alcanza N.

### AV05
Replay size > 0.

### AV06
Updates > 0 y loss finita.

### AV07
Target sync count > 0 cuando aplique.

### AV08
EventAccumulator lee scalars esenciales.

### AV09
FULL checkpoint existe y tamaño > 0.

### AV10
FULL restore recupera `global_step=N` y Replay.

### AV11
RESUME termina en M>N.

### AV12
TensorBoard contiene steps > N.

### AV13
LIGHTWEIGHT restore deja Replay vacío.

### AV14
Batch gate lightweight evita update prematuro.

### AV15
Suite BattleZone PASS.

### AV16
`git diff --check` y scope checks PASS.

---

## 24. Evidencia obligatoria

Crear:

```text
3_BattleZone/docs/hu009_evidencia_implementacion.md
```

Debe incluir únicamente resultados reales:

- rama;
- commit;
- runtime;
- seed;
- device;
- config smoke;
- N y M;
- observation shape/dtype;
- action_dim;
- updates NEW;
- updates RESUME;
- target syncs;
- replay sizes;
- loss final;
- checkpoint path/size;
- restore FULL;
- restore LIGHTWEIGHT;
- TensorBoard event files/tags/steps;
- tiempos NEW/RESUME si fueron medidos;
- resultados tests;
- CA01–CA16;
- AV01–AV16;
- limitaciones;
- scope checks.

Si una métrica no se pudo medir, registrar `NO MEDIDO`.

---

## 25. Definition of Done

HU009 estará lista para auditoría cuando:

1. exista `test_smoke.py`;
2. suite automatizada smoke pase;
3. exista ejecución real corta con `ALE/BattleZone-v5`;
4. haya al menos un update real;
5. checkpoint FULL save/load/restore funcione;
6. resume continúe de N a M;
7. LIGHTWEIGHT esté cubierto por test;
8. TensorBoard sea legible y mantenga continuidad;
9. suite BattleZone completa pase;
10. evidencia HU009 esté documentada;
11. no haya cambios en `2_Assault/`;
12. no exista MLflow;
13. no se implemente HU010+;
14. PR quede abierto para auditoría, sin merge automático.

---

## 26. Decisión de cierre

La HU009 no se considera completada por simplemente ejecutar algunos steps.

Debe probar el contrato integrado:

```text
REAL ENV
  -> DQN
  -> TRAIN
  -> REPLAY
  -> UPDATE
  -> TARGET SYNC
  -> TENSORBOARD
  -> FULL CHECKPOINT
  -> RESTORE
  -> RESUME N -> M
```

Solo después de superar este gate podrá iniciarse HU010 — Trazabilidad ligera de experimentos.
