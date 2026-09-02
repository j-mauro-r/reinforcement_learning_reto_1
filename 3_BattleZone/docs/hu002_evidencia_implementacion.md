# Evidencia de implementacion - HU002 BattleZone

## 1. Identificacion

- HU: HU002 - Experimento 0 y baseline aleatorio de BattleZone
- Rama: `feature/battlezone-hu002-experimento-0`
- Notebook principal: `3_BattleZone/experimento_0_battlezone.ipynb`
- Artefacto de evidencia local: `3_BattleZone/data/baseline_random_battlezone_local.json`

## 2. Implementacion realizada

Se implemento un notebook reproducible para Experimento 0 que:

- instala dependencias en Colab cuando corresponde;
- registra runtime y hardware;
- crea `ALE/BattleZone-v5` con configuracion explicita;
- inspecciona contrato de observacion/acciones e `info`;
- ejecuta una politica 100% aleatoria usando solo `env.action_space.sample()`;
- inicializa la seed del entorno y la seed del action space por episodio (`env.reset(seed=s)` y `env.action_space.seed(s)`);
- corre al menos 10 episodios independientes con seeds explicitas;
- produce tabla por episodio con las metricas requeridas;
- calcula baseline agregado;
- genera visualizaciones minimas;
- incluye evidencia visual de radar y comparacion exploratoria grayscale/resize;
- incorpora una seccion `Conclusiones del Experimento 0`.

No se implemento entrenamiento, agente, replay buffer, checkpoints, TensorBoard ni MLflow.

## 3. Resumen empirico local (corrida real)

Configuracion usada:

- env_id: `ALE/BattleZone-v5`
- episodios: `10`
- base_seed: `20260830`
- mode: `1`
- difficulty: `0`
- obs_type: `rgb`
- frameskip: `4`
- repeat_action_probability: `0.25`
- action_space_seed_strategy: `seed action_space with episode seed before sampling`

Runtime observado:

- Python `3.8.10`
- Gymnasium `1.1.1`
- ALE-Py `0.10.1`
- NumPy `1.24.4`
- GPU disponible: `False`

Resultado agregado:

- reward mean/median/std/min/max: `3000.0 / 2000.0 / 3065.94 / 0.0 / 10000.0`
- steps mean/min/max: `1159.5 / 821 / 1960`
- terminated: `10`
- truncated: `0`
- reward density (positive/zero/negative): `0.1725% / 99.8275% / 0.0%`
- non-zero events promedio por episodio: `2.0`
- rewards unicos observados: `{0.0, 1000.0, 2000.0, 5000.0, 6000.0}`
- vidas iniciales observadas: `{5}`
- perdidas promedio por episodio: `5.0`
- vidas extra detectadas: `0`

Validacion de reproducibilidad de muestreo de acciones:

- seed usada para prueba: `BASE_SEED + 12345`
- pasos verificados: `32`
- resultado: `sequence_a == sequence_b` -> `True`

## 4. Autovalidaciones HU002

### AV01 - Ejecucion de imports

- Procedimiento: se ejecuto script local equivalente a las celdas de import.
- Resultado: imports correctos de `gymnasium`, `ale_py`, `numpy`, `pandas`, `matplotlib`, `psutil`, `PIL`.
- Estado: PASS.
- Evidencia: `3_BattleZone/data/baseline_random_battlezone_local.json` generado sin error de imports.

### AV02 - Creacion del entorno

- Procedimiento: `gym.make("ALE/BattleZone-v5", obs_type="rgb", frameskip=4, repeat_action_probability=0.25, mode=1, difficulty=0)` y `reset(seed=...)`.
- Resultado: entorno inicializo correctamente.
- Estado: PASS.
- Evidencia: `inspection.observation_space`, `inspection.reset_info` en el JSON.

### AV03 - Action space

- Procedimiento: inspeccion de `env.action_space` y `get_action_meanings()`.
- Resultado: `Discrete(18)` y 18 acciones observadas.
- Estado: PASS.
- Evidencia: `inspection.action_space`, `inspection.num_actions`, `inspection.action_meanings`.

### AV04 - Observacion

- Procedimiento: inspeccion de observacion de reset.
- Resultado: shape `(210,160,3)`, dtype `uint8`, min/max observado `[0,236]`.
- Estado: PASS.
- Evidencia: `inspection.initial_obs_shape`, `inspection.initial_obs_dtype`, `inspection.initial_obs_min`, `inspection.initial_obs_max`.

### AV05 - Reproducibilidad de secuencia de acciones

- Procedimiento: muestrear dos secuencias de 32 acciones con `env.action_space.sample()` usando la misma seed del action space.
- Resultado: ambas secuencias fueron identicas.
- Estado: PASS.
- Evidencia: `action_sampling_reproducibility.reproducible = true` en `3_BattleZone/data/baseline_random_battlezone_local.json`.

### AV06 - Interaccion corta

- Procedimiento: sonda aleatoria de hasta 120 steps (>=100) o terminacion.
- Resultado: loop funcional sin errores.
- Estado: PASS.
- Evidencia: bloque `probe` del JSON con samples y claves de info.

### AV07 - Baseline completo

- Procedimiento: ejecucion de 10 episodios con seeds `base_seed + episode_id`.
- Resultado: 10 episodios completados via `terminated or truncated`.
- Estado: PASS.
- Evidencia: `episode_records` con 10 filas; `termination.terminated_true_count=10`, `truncated_true_count=0`.

### AV08 - Integridad estadistica

- Procedimiento: calculo de resumen agregado desde tabla de episodios.
- Resultado: metricas consistentes con registros por episodio.
- Estado: PASS.
- Evidencia: bloque `aggregate` coherente con `episode_records`.

### AV09 - Densidad de reward

- Procedimiento: verificacion de suma de conteos positivo/cero/negativo contra total steps.
- Resultado: suma exacta.
- Estado: PASS.
- Evidencia: `consistency_checks.reward_classification_sum_equals_steps = true`.

### AV10 - Frecuencia de acciones

- Procedimiento: suma de conteos de 18 acciones y comparacion con total steps.
- Resultado: suma exacta.
- Estado: PASS.
- Evidencia: `consistency_checks.action_count_sum_equals_steps = true`.

### AV11 - Vidas

- Procedimiento: seguimiento de `info["lives"]` por episodio.
- Resultado: vidas disponibles en reset/step, perdidas coherentes hasta 0 en terminacion.
- Estado: PASS.
- Evidencia: `lives_start_values_observed=[5]`, `avg_lives_lost_per_episode=5.0`, `last_life_terminated_count=10`.

### AV12 - Visualizaciones

- Procedimiento: implementacion de celdas de graficas en notebook.
- Resultado: notebook incluye las 3 visualizaciones minimas requeridas.
- Estado: PASS (implementacion); ejecucion final depende del runtime.
- Evidencia: celdas de plotting en `3_BattleZone/experimento_0_battlezone.ipynb`.

### AV13 - Coherencia documental

- Procedimiento: contraste de resultados locales contra actualizacion de ficha tecnica.
- Resultado: se agrego seccion de evidencia empirica separada de hechos documentales y se verifico aislamiento respecto a `2_Assault/` en el diff de la rama.
- Estado: PASS.
- Evidencia: `3_BattleZone/docs/ficha_tecnica.md`, seccion 26, y diff de la rama sin cambios bajo `2_Assault/`.

### AV14 - Ejecucion Colab

- Procedimiento requerido:

1. abrir `3_BattleZone/experimento_0_battlezone.ipynb` en Colab limpio;
2. ejecutar todas las celdas en orden;
3. confirmar instalacion sin cambios manuales;
4. confirmar ejecucion de al menos 10 episodios;
5. confirmar generacion de tablas, metricas y graficas;
6. registrar versiones reales del runtime Colab;
7. actualizar AV14 a PASS solo despues de esa ejecucion.

- Resultado actual: pendiente de validacion real en Google Colab por parte del usuario.
- Estado: PENDING_COLAB_VALIDATION.
- Evidencia: no disponible aun; no se marca PASS hasta ejecutar el notebook en un runtime Colab limpio.

## 5. Estado de HU002

- HU002 IMPLEMENTADA con evidencia local reproducible.
- HU002 queda pendiente unicamente de validacion AV14 en Google Colab.
- HU002 no se marca como CERRADA ni COMPLETADA.
- AV14 (Colab runtime limpio) queda PENDING_COLAB_VALIDATION hasta que el usuario ejecute el notebook en Colab y registre las versiones reales del runtime.
