# HU011 — Entregable final de Assault y validación integral contra el enunciado

## 1. Identificación

- **ID:** HU011
- **Nombre:** Entregable final de Assault y validación integral contra el enunciado
- **Estado:** PENDIENTE
- **Dependencias previas:** HU001 `[COMPLETADA]`, HU009 `[COMPLETADA]`, HU009C `[COMPLETADA]`.
- **Dependencia no bloqueante:** HU008B conserva su estado independiente.
- **Fuente de verdad académica:** `enunciado_reto_1.txt`.
- **Fuente de verdad del plan:** `2_Assault/docs/implementacion.md`.
- **Fuentes técnicas:** `ficha_tecnica.md`, `arquitectura.md`, `configs/ddqn_config.yaml`, `assault_ddqn.ipynb` y artefactos de `assault_ddqn_full_001`.
- **Algoritmo:** DDQN.
- **Entorno:** `ALE/Assault-v5`.
- **Métrica principal:** recompensa raw promedio sobre al menos 10 partidas independientes.

## 2. Contexto y problema

Las HUs anteriores ya construyeron y validaron el agente, el entrenamiento full, el modelo compacto, TensorBoard, evaluación y video. HU011 consolida esas piezas como **entregable académico final de Assault** y aplica un gate explícito criterio por criterio contra el enunciado. Ningún criterio puede declararse cumplido sin evidencia verificable.

## 3. Historia de usuario

> **Como** equipo que debe entregar Assault del Reto 1, **quiero** consolidar notebook, modelo, video, métricas y reporte técnico en un entregable reproducible y auditable, **para** demostrar que cada requisito del enunciado está satisfecho antes de entregar.

## 4. Objetivo verificable

Al finalizar HU011 debe existir una versión final de `2_Assault/assault_ddqn.ipynb` y una matriz de cumplimiento que permitan a un evaluador:

1. ejecutar `Run All` en Colab sin editar código;
2. instalar dependencias;
3. entrenar desde cero si no hay artefactos o reutilizar el estado final válido;
4. identificar y justificar DDQN;
5. revisar entorno, preprocessing, arquitectura e hiperparámetros;
6. revisar versiones y hardware;
7. inspeccionar evidencia cuantitativa del entrenamiento;
8. inspeccionar recompensa durante entrenamiento y explotación;
9. verificar evaluación de al menos 10 partidas;
10. comparar con baseline aleatorio;
11. reproducir el video final;
12. verificar lineage modelo ↔ entrenamiento ↔ evaluación ↔ video;
13. leer comportamiento aprendido, limitaciones y conclusión;
14. observar todos los criterios obligatorios de Assault en `PASS`.

HU011 solo puede marcarse `[COMPLETADA]` cuando no exista ningún criterio obligatorio en `PENDIENTE`, `PARCIAL`, `NO VALIDADO` o `FAIL`.

## 5. Alcance

### 5.1 Notebook final

`2_Assault/assault_ddqn.ipynb` debe ser el notebook único de Assault, ejecutable en Colab, con dependencias versionadas, bootstrap GitHub, `ASSAULT_EXECUTION_MODE=auto`, ejecución ordenada y gate final.

### 5.2 Modelo entrenado

El modelo entregable debe ser el compacto de inferencia, con path, `project_run_id`, `source_checkpoint_step`, SHA-256, metadata de entorno/preprocessing, carga en agente nuevo e identidad compartida con evaluación y video.

### 5.3 Video

MP4 reproducible con evidencia del entrenamiento y gameplay real de Assault, `epsilon=0.0`, seed/reward/steps, sin entrenamiento durante captura y visible inline.

### 5.4 Reporte técnico dentro del notebook

Debe cubrir: problema y objetivo; selección/justificación DDQN; entorno/preprocessing; arquitectura; condiciones de ejecución; entrenamiento; evolución del aprendizaje; evaluación final; comparación baseline; comportamiento aprendido; limitaciones; conclusión; artefactos y trazabilidad.

### 5.5 Evaluación formal final

La medición principal se ejecuta sobre el **modelo compacto cargado desde disco**: `ALE/Assault-v5`, preprocessing contractual, >=10 episodios, `epsilon=0.0`, rewards raw, seeds explícitas, sin updates, estadísticas mean/median/std/min/max y sidecar JSON auditable.

### 5.6 Gráfica de explotación

Además de las tres figuras de entrenamiento HU009C, debe existir una figura explícita `episodio de evaluación vs reward` con línea de media. No cuenta como una cuarta figura de entrenamiento.

### 5.7 Matriz de cumplimiento

Crear una matriz con `ID | criterio | evidencia | método | estado`. Los criterios obligatorios de Assault solo pueden finalizar en `PASS`.

## 6. Fuera de alcance

No implementar algoritmo nuevo, HPO exhaustivo, reentrenamiento innecesario, cherry-picking, hardcodear resultados como fuente primaria, versionar binarios grandes ni declarar cumplido el requisito global de dos algoritmos solo con Assault.

## 7. Decisiones y restricciones

- Notebook = orquestador/reporte; lógica reutilizable en `src/`.
- Una única identidad de modelo final.
- Evaluación separada del entrenamiento.
- Datos derivados de artefactos/config cuando existan.
- `epsilon=0.0` en explotación final.
- Git sin `.pt`, `.mp4` ni event files.
- `GLOBAL_RETO_MULTI_ALGORITHM=PASS|PENDING` es informativo y global.

## 8. Tareas

### T01 Inventario de requisitos
Mapear cada requisito del enunciado a evidencia.

### T02 Consolidación del reporte
Actualizar narrativa final del notebook y eliminar textos obsoletos.

### T03 Evaluación final
Cargar modelo compacto, ejecutar >=10 episodios y persistir JSON.

### T04 Evidencia de explotación
Tabla por episodio + figura reward por episodio con media.

### T05 Baseline
Usar baseline HU001 comparable y calcular diferencia absoluta/relativa.

### T06 Lineage
Validar checkpoint → modelo SHA → evaluación → figura → video → notebook.

### T07 Video
Validar MP4, intro, gameplay, seed, epsilon, reward, steps e identidad.

### T08 Colab Run All
Validación final en runtime nuevo, sin editar variables.

### T09 Calidad
`compileall`, suite completa, `git diff --check`, JSON notebook e higiene Git.

### T10 Gate final
Todos los criterios obligatorios específicos de Assault deben estar en PASS.

## 9. Criterios de aceptación

- **CA01 Método permitido:** Assault usa DDQN y el reporte lo justifica.
- **CA02 Notebook Colab:** `Run All` termina sin edición manual.
- **CA03 Dependencias:** instalación desde fuente versionada.
- **CA04 Modelo:** modelo entrenado cargable, checksum y lineage.
- **CA05 Correspondencia:** modelo corresponde al entrenamiento documentado.
- **CA06 Video:** MP4 real, no vacío y visible.
- **CA07 Video entrenamiento:** evidencia breve basada en métricas reales.
- **CA08 Video comportamiento:** gameplay real del agente entrenado.
- **CA09 Justificación:** DDQN relacionado con acciones discretas, imágenes y mitigación de sobreestimación.
- **CA10 Hiperparámetros:** lr, gamma, batch, buffer, learning starts, frecuencias, epsilon y timesteps.
- **CA11 Versiones:** Python, Gymnasium, ALE-Py, PyTorch y librerías relevantes.
- **CA12 Hardware:** GPU/CUDA y condiciones reales de entrenamiento.
- **CA13 Tiempo:** tiempo real y fuente.
- **CA14 Evaluación:** >=10 partidas independientes desde modelo compacto.
- **CA15 Score:** mean, median, std, min, max.
- **CA16 Reward entrenamiento:** curva reward + media móvil.
- **CA17 Reward explotación:** figura reward por episodio final + media.
- **CA18 Baseline:** comparación cuantitativa real.
- **CA19 Comportamiento aprendido:** descripción observable relacionada con DDQN/config.
- **CA20 Conclusión:** soportada por evidencia y limitaciones.
- **CA21 Completitud:** notebook + modelo + video + reporte.
- **CA22 Organización:** notebook orquesta; lógica reusable en `src/`; tests pasan.
- **CA23 Método del curso:** Online/Target, replay uniforme, epsilon-greedy y target DDQN visibles.
- **CA24 Desempeño:** mean agente > mean baseline y gameplay coherente con aprendizaje.
- **CA25 Reporte formal:** comprensible sin documentación interna adicional.
- **CA26 Consistencia:** notebook, modelo, evaluación y video comparten identidad.
- **CA27 Requisito global:** `ASSAULT_METHOD_ALLOWED=PASS` y `GLOBAL_RETO_MULTI_ALGORITHM=PASS|PENDING` sin afirmaciones falsas.

## 10. Definition of Done

HU011 se marca `[COMPLETADA]` únicamente cuando CA01–CA26 = PASS; CA27 registra correctamente el estado global; suite/compileall/diff pasan; notebook es JSON válido; no hay binarios grandes versionados; matriz final no contiene FAIL/PARCIAL/NO VALIDADO; y `HU011_FINAL_DELIVERY_GATE=PASS`.

## 11. Autovalidaciones

- **AV01 Notebook:** `NOTEBOOK_FINAL_STRUCTURE_PASS=True`.
- **AV02 Calidad:** `python -m compileall -q 2_Assault/src`, `python -m pytest 2_Assault/tests -q`, `git diff --check`.
- **AV03 Higiene:** working tree limpio y sin binarios grandes versionados.
- **AV04 Colab:** `COLAB_RUN_ALL_FINAL_PASS=True`.
- **AV05 Modelo:** `FINAL_MODEL_PASS=True`.
- **AV06 Evaluación:** `FINAL_EVALUATION_PASS=True`.
- **AV07 Baseline:** `BASELINE_COMPARISON_PASS=True`.
- **AV08 Entrenamiento:** `TRAINING_EVIDENCE_PASS=True`.
- **AV09 Explotación:** `EXPLOITATION_REWARD_FIGURE_PASS=True`.
- **AV10 Video:** `FINAL_VIDEO_PASS=True`.
- **AV11 Lineage:** `ARTIFACT_LINEAGE_PASS=True`.
- **AV12 Reporte:** `TECHNICAL_REPORT_PASS=True`.
- **AV13 Matriz:** `HU011_FINAL_DELIVERY_GATE=PASS`.

## 12. Evidencias esperadas

Notebook ejecutado; SHA Git; requirements/versiones; runtime/hardware; config; checkpoint; modelo compacto/SHA; evaluación >=10 + JSON; baseline; comparación; 3 figuras entrenamiento; figura explotación; MP4 + metadata; tiempo entrenamiento; tests; matriz CA01–CA27.

## 13. Riesgos

Usar evaluación histórica equivocada, confundir training/eval reward, omitir gráfica de explotación, hardcodear cifras, perder lineage, depender de Drive personal, exagerar conclusiones o falsear el requisito global de dos algoritmos.

## 14. Gate final esperado

```text
ASSAULT_METHOD_ALLOWED=PASS
COLAB_NOTEBOOK_EXECUTABLE=PASS
DEPENDENCIES_PASS=PASS
MODEL_ARTIFACT_PASS=PASS
MODEL_TRAINING_CORRESPONDENCE=PASS
VIDEO_TRAINING_EVIDENCE=PASS
VIDEO_LEARNED_BEHAVIOR=PASS
ALGORITHM_JUSTIFICATION=PASS
HYPERPARAMETERS_REPORTED=PASS
LIBRARY_VERSIONS_REPORTED=PASS
HARDWARE_REPORTED=PASS
TRAINING_TIME_REPORTED=PASS
FINAL_EVALUATION_N_GE_10=PASS
FINAL_SCORE_REPORTED=PASS
TRAINING_REWARD_FIGURE=PASS
EXPLOITATION_REWARD_FIGURE=PASS
RANDOM_BASELINE_COMPARISON=PASS
LEARNED_BEHAVIOR_ANALYSIS=PASS
CONCLUSION=PASS
DELIVERY_COMPLETENESS=PASS
CODE_ORGANIZATION=PASS
COURSE_METHOD_EVIDENCE=PASS
AGENT_EFFECTIVE_VS_BASELINE=PASS
TECHNICAL_REPORT=PASS
ARTIFACT_LINEAGE=PASS
GLOBAL_RETO_MULTI_ALGORITHM=PASS|PENDING
HU011_FINAL_DELIVERY_GATE=PASS
```
