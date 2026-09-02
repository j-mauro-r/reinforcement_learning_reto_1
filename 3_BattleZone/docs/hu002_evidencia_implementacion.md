# Evidencia de implementación — HU002 BattleZone

## 1. Identificación

- **HU:** HU002 — Experimento 0 y baseline aleatorio de BattleZone
- **Rama:** `feature/battlezone-hu002-experimento-0`
- **PR:** #18
- **Notebook:** `3_BattleZone/experimento_0_battlezone.ipynb`
- **Artefacto local:** `3_BattleZone/data/baseline_random_battlezone_local.json`
- **Ficha técnica consolidada:** `3_BattleZone/docs/ficha_tecnica.md`

---

## 2. Implementación auditada

El PR implementa un Experimento 0 que:

- instala dependencias cuando detecta Google Colab;
- registra runtime y hardware;
- crea `ALE/BattleZone-v5` con configuración explícita;
- inspecciona observation/action spaces e `info`;
- usa una política 100 % aleatoria basada exclusivamente en `env.action_space.sample()`;
- inicializa seed del entorno y seed del action space por episodio;
- valida reproducibilidad del muestreo de acciones;
- ejecuta 10 episodios independientes;
- produce tabla por episodio;
- calcula métricas agregadas;
- calcula densidad de rewards y distribución de acciones;
- inspecciona vidas/terminación;
- genera visualizaciones mínimas;
- muestra frame original, radar y transformación exploratoria grayscale/resize.

No se implementaron DQN, DDQN, PER, REINFORCE, CNN, Replay Buffer, entrenamiento, checkpoints, TensorBoard ni MLflow.

No existen cambios bajo `2_Assault/` en el PR auditado.

---

## 3. Ejecución observada

La evidencia guardada actualmente corresponde a una ejecución **local en Windows**, no a Google Colab.

### Configuración

- `env_id`: `ALE/BattleZone-v5`
- episodios: `10`
- `base_seed`: `20260830`
- seeds por episodio: `20260831` a `20260840`
- mode: `1`
- difficulty: `0`
- `obs_type`: `rgb`
- `frameskip`: `4`
- `repeat_action_probability`: `0.25`
- política: `env.action_space.sample()`

### Runtime observado

- Python `3.8.10`
- Gymnasium `1.1.1`
- ALE-Py `0.10.1`
- NumPy `1.24.4`
- Windows 10
- RAM total `6.9 GB`
- GPU disponible: `False`

### Contrato observado

- observation space: `Box(0, 255, (210,160,3), uint8)`
- action space: `Discrete(18)`
- 18 action meanings confirmados
- observación inicial `(210,160,3)`, `uint8`
- rango observado inicial `[0,236]`

### `info`

En `reset()` se observaron:

- `lives`
- `episode_frame_number`
- `frame_number`
- `seeds`

En `step()` se observaron:

- `lives`
- `episode_frame_number`
- `frame_number`

---

## 4. Baseline aleatorio

### Recompensa

- media: `3000.0`
- mediana: `2000.0`
- desviación estándar poblacional: `3065.94`
- mínimo: `0.0`
- máximo: `10000.0`

Rewards por episodio:

`[2000, 0, 10000, 7000, 4000, 0, 1000, 1000, 2000, 3000]`

### Duración

- steps totales: `11595`
- media: `1159.5`
- mínimo: `821`
- máximo: `1960`

### Densidad de reward

- positivo: `0.1725 %`
- cero: `99.8275 %`
- negativo: `0.0 %`
- eventos no-cero promedio/episodio: `2.0`

Rewards únicos por step:

`{0.0, 1000.0, 2000.0, 5000.0, 6000.0}`

Frecuencias:

- `0 -> 11575`
- `1000 -> 17`
- `2000 -> 1`
- `5000 -> 1`
- `6000 -> 1`

### Vidas y terminación

- vidas iniciales: `5` en 10/10 episodios
- vidas finales: `0` en 10/10 episodios
- pérdidas promedio: `5.0`
- vidas extra observadas: `0`
- `terminated=True`: `10/10`
- `truncated=True`: `0/10`

### Reproducibilidad de acciones

Se ejecutó una prueba con 32 muestras de `env.action_space.sample()` utilizando la misma seed del action space dos veces.

- seed de prueba: `BASE_SEED + 12345`
- resultado: `sequence_a == sequence_b -> True`

Esto valida reproducibilidad del muestreo de acciones; no implica determinismo absoluto del episodio por la estocasticidad de ALE/sticky actions.

---

## 5. Autovalidaciones auditadas

| AV | Resultado | Estado auditado |
|---|---|---|
| AV01 Imports | Dependencias locales cargan | PASS |
| AV02 Entorno | `ALE/BattleZone-v5` inicializa | PASS |
| AV03 Action space | `Discrete(18)` | PASS |
| AV04 Observación | `(210,160,3)`, `uint8` | PASS |
| AV05 Reproducibilidad acciones | 32/32 secuencia reproducida | PASS |
| AV06 Interacción corta | Loop funcional | PASS |
| AV07 Baseline | 10 episodios completos | PASS |
| AV08 Estadísticas | Consistentes con registros | PASS |
| AV09 Densidad reward | Conteos suman total steps | PASS |
| AV10 Acciones | Conteos suman total steps | PASS |
| AV11 Vidas | Consistentes con `info` | PASS |
| AV12 Visualizaciones | Ejecutadas y guardadas en notebook | PASS |
| AV13 Coherencia documental | Revisada manualmente en auditoría PR #18 y ficha actualizada | PASS |
| AV14 Colab limpio | Sin evidencia Colab todavía | `PENDING_COLAB_VALIDATION` |

---

## 6. Hallazgos de auditoría

### H01 — Implementación funcional

El código satisface el propósito central de HU002: construir un baseline aleatorio reproducible y obtener evidencia empírica del entorno sin introducir entrenamiento ni heurísticas.

### H02 — Reward muy escaso

Solo `0.1725 %` de los steps produjo reward positivo. Esta evidencia debe alimentar HU004 al comparar eficiencia muestral de los algoritmos permitidos.

### H03 — Alta dispersión

La desviación estándar (`3065.94`) es ligeramente mayor que la media (`3000.0`). Diez episodios cumplen el mínimo definido, pero el baseline presenta alta variabilidad.

### H04 — Reward `6000`

El notebook observa un reward de `6000` en un step. No existe evidencia suficiente para mapearlo a un único objetivo de la tabla histórica del juego. La ficha técnica fue corregida para evitar presentar una equivalencia no demostrada.

### H05 — Ejecución guardada es local

Los outputs del notebook corresponden a Windows/Python 3.8.10. AV14 exige una ejecución independiente en Google Colab limpio.

### H06 — Conclusiones del notebook incompletas

La celda Markdown `Conclusiones del Experimento 0` existe, pero en la versión auditada permanece como plantilla con instrucciones de qué completar. No contiene todavía conclusiones derivadas de la ejecución guardada.

Esto incumple todavía el criterio/DoD que exige una sección de conclusiones completada.

---

## 7. Estado de HU002

Después de la auditoría del PR #18:

### Cumplido

- implementación del baseline;
- evidencia local real;
- reproducibilidad del action space;
- 10 episodios;
- métricas requeridas;
- visualizaciones;
- `info`, vidas y terminación;
- ficha técnica consolidada;
- AV01–AV13 auditadas como PASS;
- cero cambios/reutilización de `2_Assault/`.

### Pendiente antes de cierre formal

1. completar `Conclusiones del Experimento 0` dentro del notebook con los resultados reales;
2. ejecutar AV14 en Google Colab limpio y conservar evidencia real;
3. después de AV14, actualizar runtime Colab, evidencia y estado de HU002.

**Estado actual correcto:**

`HU002 IMPLEMENTADA — pendiente de conclusiones del notebook y AV14 en Google Colab.`

HU002 **no debe marcarse CERRADA/COMPLETADA todavía**.