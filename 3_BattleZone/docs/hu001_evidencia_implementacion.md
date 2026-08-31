# Evidencia de implementación — HU001 BattleZone

## 1. Identificación

- **HU:** HU001 — Caracterización técnica y ficha inicial de BattleZone
- **Rama:** `feature/battlezone-implementation-plan`
- **Estado de implementación:** IMPLEMENTADA
- **Documento DWP:** `3_BattleZone/docs/hu001_caracterizacion_tecnica_battlezone.md`
- **Entregable principal:** `3_BattleZone/docs/ficha_tecnica.md`

## 2. Resultado de implementación

La HU001 se implementó como una historia estrictamente documental y de caracterización. No se creó código de agente, entrenamiento, preprocessing, Replay Buffer, TensorBoard ni checkpoints.

La ficha técnica existente cubre:

- `ALE/BattleZone-v5`;
- action space `Discrete(18)` y las 18 acciones;
- observaciones RGB, grayscale y RAM;
- shapes, `uint8` y rango teórico de píxel;
- `frameskip=4`;
- `repeat_action_probability=0.25`;
- modes `[1,2,3]`, default `1`;
- difficulty `[0]`, default `0`;
- 5 vidas iniciales y posibles vidas adicionales;
- radar, vista principal, enemigos y obstáculos;
- separación entre score histórico y reward ALE;
- métrica principal de evaluación sobre al menos 10 episodios;
- baseline futuro mediante política aleatoria;
- riesgos de preprocessing, exploración y costo computacional;
- preguntas empíricas que deberá resolver HU002.

## 3. Versiones de runtime

HU001 no fija números de versión de Gymnasium o ALE-Py que no hayan sido ejecutados y verificados en un runtime limpio.

Por tanto:

- **Gymnasium:** pendiente de fijación/validación en la primera ejecución reproducible;
- **ALE-Py:** pendiente de fijación/validación en la primera ejecución reproducible.

Esta decisión cumple CA12 y evita documentar versiones inventadas. Las propiedades de `ALE/BattleZone-v5` se mantienen como hechos documentados independientemente de esa futura fijación de runtime.

## 4. Resultado de criterios de aceptación

| Criterio | Estado | Evidencia |
|---|---|---|
| CA01 Environment ID | PASS | `ficha_tecnica.md` identifica `ALE/BattleZone-v5` y `gymnasium.make(...)` |
| CA02 Observaciones | PASS | RGB, grayscale y RAM documentadas |
| CA03 Action space | PASS | `Discrete(18)` y 18/18 acciones documentadas |
| CA04 Dinámica temporal | PASS | `frameskip=4` y sticky actions `0.25` |
| CA05 Radar/percepción | PASS | riesgo de pérdida de radar por resize/cropping documentado |
| CA06 Recompensas | PASS | score histórico separado del reward ALE |
| CA07 Vidas/terminación | PASS | vidas documentadas; `info`, `terminated` y `truncated` delegados a HU002 |
| CA08 Métrica/baseline | PASS | recompensa promedio ≥10 episodios + baseline aleatorio |
| CA09 Pendientes diferenciados | PASS | sección explícita de información a validar empíricamente |
| CA10 Independencia Assault | PASS | no existe implementación ni dependencia desde `2_Assault/` |
| CA11 Coherencia documental | PASS | algoritmo sigue sin seleccionar; MLflow excluido; preprocessing abierto |
| CA12 Versiones | PASS | versiones concretas no verificadas se mantienen pendientes |

## 5. Autovalidaciones

### AV01 — Cobertura de HU001

**Resultado:** PASS.

Todos los elementos requeridos por HU001 están documentados o identificados explícitamente como pendientes de validación empírica.

### AV02 — Action space

**Resultado:** PASS.

La ficha contiene `Discrete(18)` y las 18 acciones documentadas para BattleZone.

### AV03 — Observaciones

**Resultado:** PASS.

Se documentan RGB `(210,160,3)`, grayscale `(210,160)` y RAM `(128,)`, con `uint8` y rango teórico `[0,255]`.

### AV04 — Configuración temporal

**Resultado:** PASS documental.

Se registran `frameskip=4`, `repeat_action_probability=0.25`, modes y difficulty. La confirmación del runtime concreto queda para la ejecución posterior.

### AV05 — Score vs reward

**Resultado:** PASS.

Los valores históricos de puntuación no se presentan como rewards ALE confirmados.

### AV06 — Preguntas para HU002

**Resultado:** PASS.

HU002 recibe preguntas concretas sobre baseline, rewards, duración, vidas, `info`, terminación, radar/preprocessing y score/reward.

### AV07 — Coherencia con lineamientos

**Resultado:** PASS.

- no se usa MLflow;
- no se reutiliza código Assault;
- no se selecciona algoritmo prematuramente;
- no se fija preprocessing definitivo;
- no se adelanta implementación de historias posteriores.

### AV08 — Rutas

**Resultado:** PASS con observación documental.

Los artefactos reales de HU001 están en `3_BattleZone/docs/`: `ficha_tecnica.md`, `arquitectura.md`, `implementacion.md`, `lineamientos.md` y el DWP de HU001. Cualquier referencia previa a `3_BattleZone/ficha_tecnica.md` debe interpretarse como ruta histórica/obsoleta y corregirse en la próxima edición del plan maestro para mantener una única ruta canónica.

## 6. Definition of Done

- [x] existe `3_BattleZone/docs/hu001_caracterizacion_tecnica_battlezone.md`;
- [x] existe `3_BattleZone/docs/ficha_tecnica.md`;
- [x] se documenta `ALE/BattleZone-v5`;
- [x] se documenta `Discrete(18)` y las 18 acciones;
- [x] se documentan RGB, grayscale y RAM;
- [x] se documentan `frameskip` y sticky actions;
- [x] se documentan modes, difficulty y vidas conocidas;
- [x] radar, enemigos, obstáculos y dinámica están descritos;
- [x] score histórico y reward ALE están diferenciados;
- [x] métrica principal y baseline futuro definidos;
- [x] riesgos y preguntas para HU002 documentados;
- [x] versiones no verificadas se mantienen explícitamente pendientes;
- [x] AV01–AV08 ejecutadas documentalmente;
- [x] no existe código ni dependencia reutilizada desde Assault;
- [x] no se implementó nada fuera del alcance de HU001;
- [x] no existen contradicciones materiales con la arquitectura o lineamientos técnicos;
- [x] evidencia de implementación disponible en esta rama.

## 7. Gate hacia HU002

**HU001 queda implementada y técnicamente habilita HU002 — Experimento 0 y baseline aleatorio.**

HU002 deberá convertir en evidencia empírica los elementos deliberadamente abiertos en HU001: versiones reales del runtime, contenido de `info`, distribución de rewards, duración de episodios, comportamiento de vidas, `terminated/truncated`, baseline aleatorio y adecuación visual del preprocessing.
