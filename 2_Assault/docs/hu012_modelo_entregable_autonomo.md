# HU012 — Modelo entregable autónomo y ejecución independiente del profesor

## 1. Identificación

- **ID:** HU012
- **Nombre:** Modelo entregable autónomo y ejecución independiente del profesor
- **Estado:** PENDIENTE
- **Dependencias previas:** HU011 — Entregable final de Assault y validación integral contra el enunciado.
- **Fuente de verdad académica:** `enunciado_reto_1.txt`.
- **Fuente de verdad del plan:** `2_Assault/docs/implementacion.md`.
- **Fuentes técnicas:** `2_Assault/assault_ddqn.ipynb`, `2_Assault/src/model_artifact.py`, `2_Assault/src/environment.py`, `2_Assault/src/evaluator.py`, `2_Assault/configs/ddqn_config.yaml`.
- **Algoritmo:** DDQN.
- **Entorno:** `ALE/Assault-v5`.

## 2. Contexto y problema

El entregable de Assault ya cuenta con entrenamiento full, checkpoint final, modelo compacto de inferencia, evaluación formal, TensorBoard, video y reporte técnico. Sin embargo, el flujo actual de carga del modelo compacto está orientado principalmente a los artefactos persistidos en Google Drive del equipo.

Para la entrega académica, el evaluador debe poder verificar la correcta ejecución y el comportamiento aprendido usando el modelo entregado, sin depender del Drive personal del equipo y sin necesidad de volver a entrenar el agente.

HU012 agrega una capa final de portabilidad en el notebook. El objetivo es que `2_Assault/assault_ddqn_model.pt` sea la primera fuente de carga cuando forme parte del paquete entregado y que el flujo histórico de Drive permanezca disponible como fallback para mantener compatibilidad y no romper ninguna funcionalidad existente.

Esta HU es estrictamente aditiva y de cierre. No debe modificar la lógica del agente, la arquitectura DDQN, el entrenamiento, los hiperparámetros, la evaluación formal existente, los checkpoints ni los artefactos ya validados.

## 3. Historia de usuario

> **Como** profesor o evaluador del Reto 1, **quiero** poder cargar y ejecutar directamente el modelo entrenado entregado junto al notebook, **para** verificar el comportamiento aprendido del agente Assault sin acceder al Google Drive personal del equipo ni volver a entrenarlo.

## 4. Objetivo verificable

Al finalizar HU012, `2_Assault/assault_ddqn.ipynb` debe incorporar una nueva sección final que permita:

1. localizar primero el modelo entregable `2_Assault/assault_ddqn_model.pt`;
2. usar como fallback la ruta histórica del modelo compacto bajo `BASE / models / PROJECT_RUN_ID / assault_ddqn_model.pt`;
3. cargar el modelo con la infraestructura existente de `load_inference_model`;
4. crear el entorno de evaluación usando la fábrica actual;
5. ejecutar el agente cargado sin entrenamiento ni actualización de pesos;
6. dejar una salida explícita que confirme qué ruta de modelo fue utilizada;
7. fallar con un mensaje claro si ninguna fuente de modelo existe;
8. conservar intacto el comportamiento de las secciones anteriores del notebook.

## 5. Alcance

### 5.1 Nueva sección 15 del notebook

Agregar al final de `2_Assault/assault_ddqn.ipynb` una sección claramente identificable, por ejemplo:

`## 15. Carga y ejecución independiente del modelo entrenado`

Esta sección debe ser conceptualmente independiente del entrenamiento y orientada al evaluador.

### 5.2 Orden de resolución del modelo

La resolución del archivo debe seguir este orden obligatorio:

1. **Modelo incluido en la entrega:**
   - `ASSAULT_DIR / "assault_ddqn_model.pt"`
2. **Fallback histórico existente:**
   - `BASE / "models" / PROJECT_RUN_ID / "assault_ddqn_model.pt"`
3. **Error controlado:**
   - `FileNotFoundError` o equivalente con mensaje que indique ambas rutas verificadas y cómo proporcionar el modelo.

No reemplazar ni eliminar la ruta actual de Drive. El nuevo comportamiento debe ser un fallback compatible, no una sustitución destructiva.

### 5.3 Carga del modelo

La sección debe reutilizar `load_inference_model` de `src/model_artifact.py`.

No duplicar la lógica de reconstrucción del agente ni implementar un segundo cargador de pesos.

Cuando la metadata disponible lo permita, conservar las validaciones existentes de integridad, arquitectura y `project_run_id`. Si alguna validación estricta depende de metadata generada en una sección previa del notebook, la sección 15 debe resolver esa dependencia de manera segura sin volver a entrenar.

### 5.4 Ejecución independiente

Con el agente cargado desde el modelo resuelto, la sección debe:

- crear un entorno de evaluación con `create_assault_env`;
- ejecutar al menos una partida o una evaluación corta verificable;
- usar `epsilon=0.0`;
- no llamar métodos de entrenamiento;
- no ejecutar `optimizer.step()`;
- no modificar replay buffer;
- no modificar Online/Target weights;
- cerrar correctamente el entorno al finalizar.

La evaluación formal de >=10 episodios definida en HU011 no debe eliminarse ni reemplazarse. Esta ejecución es una facilidad adicional de verificación para el evaluador.

### 5.5 Evidencia visible

La sección debe imprimir o mostrar como mínimo:

- ruta seleccionada del modelo;
- origen `DELIVERY` o `DRIVE_FALLBACK`;
- carga exitosa;
- dispositivo utilizado;
- `epsilon=0.0`;
- recompensa y cantidad de steps de la ejecución de verificación, o resumen equivalente;
- bandera final `ASSAULT_DELIVERY_MODEL_EXECUTION_PASS=True` cuando todas las validaciones de esta sección sean satisfactorias.

### 5.6 Modelo entregable

El artefacto que se entregue junto al notebook será:

`2_Assault/assault_ddqn_model.pt`

Debe corresponder al mismo modelo compacto validado en HU009C/HU011. No se debe entrenar ni generar un modelo distinto exclusivamente para HU012.

Si el archivo binario supera las políticas de versionamiento del repositorio o existe una decisión explícita de no versionar `.pt`, HU012 no obliga a almacenarlo en GitHub; sí obliga a que el paquete académico final lo incluya junto al notebook y a que la sección 15 sepa localizarlo en `ASSAULT_DIR`.

## 6. Fuera de alcance

No implementar:

- nuevo algoritmo;
- cambios a DDQN;
- cambios a `network.py`, `agent.py` o replay buffer salvo que sean estrictamente indispensables para corregir un defecto comprobado;
- reentrenamiento;
- optimización de hiperparámetros;
- cambios a los resultados oficiales de HU011;
- reemplazo de la evaluación formal >=10 episodios;
- eliminación de checkpoints o artefactos históricos;
- modificación de la estrategia de persistencia de Drive;
- cambios de preprocessing;
- dependencia nueva de servicios externos.

## 7. Decisiones y restricciones

- Cambio **aditivo y retrocompatible**.
- Notebook = orquestador/reporte; lógica reusable continúa en `src/`.
- Reutilizar `ASSAULT_DIR`, `BASE`, `PROJECT_RUN_ID`, `create_assault_env` y `load_inference_model` existentes.
- No hardcodear `/content/...` cuando ya existe una abstracción de ruta equivalente.
- El modelo local de entrega tiene prioridad sobre Drive.
- Drive es fallback, no requisito del profesor.
- La sección 15 no debe disparar entrenamiento bajo ninguna circunstancia.
- La ejecución de la sección 15 no debe alterar el agente ni los artefactos validados previamente.
- Ninguna modificación puede romper `Run All`, el gate HU011 ni los tests existentes.

## 8. Tareas

### T01 — Resolver modelo entregable

Implementar en el notebook la resolución ordenada:

`ASSAULT_DIR/assault_ddqn_model.pt` → `BASE/models/<PROJECT_RUN_ID>/assault_ddqn_model.pt` → error controlado.

### T02 — Cargar modelo sin duplicación

Reutilizar `load_inference_model` y las validaciones existentes. No implementar carga manual de `state_dict` en el notebook.

### T03 — Ejecutar verificación independiente

Crear un entorno de evaluación y ejecutar el agente cargado con `epsilon=0.0`, sin entrenamiento.

### T04 — Evidencia para el profesor

Mostrar claramente origen del modelo, ruta, estado de carga, dispositivo y resultado de la partida/evaluación corta.

### T05 — Protección contra regresiones

Agregar o actualizar tests focalizados del notebook para comprobar:

- existencia de la sección 15;
- prioridad de `ASSAULT_DIR / "assault_ddqn_model.pt"`;
- fallback a Drive;
- error explícito cuando no existe ninguno;
- uso de `load_inference_model`;
- uso de `epsilon=0.0`;
- ausencia de llamadas de entrenamiento dentro del flujo nuevo.

### T06 — Validaciones de regresión

Ejecutar como mínimo:

```bash
python -m compileall -q 2_Assault/src
PYTHONPATH=2_Assault python -m pytest 2_Assault/tests -q
git diff --check
```

Y validar el notebook como JSON válido.

### T07 — Validación Colab del caso de entrega

En un runtime de Colab, validar al menos el caso donde el modelo está disponible como `2_Assault/assault_ddqn_model.pt` y confirmar que la sección 15 lo selecciona antes que Drive y ejecuta el agente sin entrenar.

## 9. Criterios de aceptación

- **CA01 Sección final:** existe una sección 15 claramente identificada para cargar y ejecutar el modelo entregado.
- **CA02 Prioridad local:** si `ASSAULT_DIR/assault_ddqn_model.pt` existe, esa es la ruta seleccionada aunque también exista el modelo en Drive.
- **CA03 Fallback:** si el modelo local no existe y el modelo histórico de Drive sí existe, se usa el fallback sin alterar el flujo anterior.
- **CA04 Error controlado:** si no existe ninguna fuente, el mensaje identifica claramente que falta el modelo entrenado y las rutas verificadas.
- **CA05 Reutilización:** la carga usa `load_inference_model`; no existe una segunda implementación manual de carga del agente.
- **CA06 Sin entrenamiento:** la sección 15 no ejecuta entrenamiento, actualizaciones de pesos, optimizador ni replay buffer.
- **CA07 Explotación:** la ejecución de verificación usa `epsilon=0.0`.
- **CA08 Entorno contractual:** el entorno se crea usando la fábrica actual de Assault y su preprocessing existente.
- **CA09 Modelo correcto:** el archivo local entregable corresponde al mismo modelo compacto validado previamente; no se crea un modelo alternativo.
- **CA10 Independencia de Drive:** el evaluador puede ejecutar el modelo local sin acceso al Google Drive del equipo.
- **CA11 Retrocompatibilidad:** las secciones 1–14 mantienen su comportamiento y HU011 continúa pudiendo alcanzar `HU011_FINAL_DELIVERY_GATE=PASS`.
- **CA12 Evaluación formal intacta:** la evaluación >=10 episodios y los resultados oficiales de HU011 no son reemplazados ni recalculados como parte de esta mejora salvo ejecución explícita del flujo existente.
- **CA13 No regresión:** suite focalizada/completa de `2_Assault/tests`, `compileall` y `git diff --check` pasan.
- **CA14 Portabilidad:** funciona tanto cuando `ASSAULT_DIR` proviene de Colab como cuando apunta al checkout local.
- **CA15 Evidencia:** la sección imprime `ASSAULT_DELIVERY_MODEL_EXECUTION_PASS=True` únicamente después de una carga y ejecución satisfactorias.

## 10. Definition of Done

HU012 se marca `[COMPLETADA]` únicamente cuando CA01–CA15 están satisfechos y existe evidencia verificable de que:

- el modelo local de entrega tiene prioridad;
- el fallback de Drive continúa funcionando;
- el modelo puede cargarse y ejecutarse con `epsilon=0.0` sin entrenamiento;
- las secciones previas no fueron degradadas;
- los tests y validaciones de calidad pasan;
- la ejecución en Colab del caso de entrega produce `ASSAULT_DELIVERY_MODEL_EXECUTION_PASS=True`.

## 11. Autovalidaciones

- **AV01 Estructura:** `SECTION_15_DELIVERY_MODEL_PRESENT=True`.
- **AV02 Prioridad local:** `DELIVERY_MODEL_PRIORITY_PASS=True`.
- **AV03 Fallback Drive:** `DELIVERY_MODEL_DRIVE_FALLBACK_PASS=True`.
- **AV04 Carga:** `DELIVERY_MODEL_LOAD_PASS=True`.
- **AV05 Sin entrenamiento:** `DELIVERY_MODEL_NO_TRAINING_PASS=True`.
- **AV06 Ejecución:** `ASSAULT_DELIVERY_MODEL_EXECUTION_PASS=True`.
- **AV07 Calidad:** `compileall`, `pytest`, `git diff --check` y JSON notebook válidos.
- **AV08 Regresión HU011:** `HU011_FINAL_DELIVERY_GATE=PASS` permanece alcanzable sin cambios en sus criterios.

## 12. Evidencias esperadas

- diff focalizado de `2_Assault/assault_ddqn.ipynb`;
- tests nuevos o actualizados para la sección 15;
- salida mostrando `MODEL_SOURCE=DELIVERY` cuando existe `2_Assault/assault_ddqn_model.pt`;
- salida mostrando `MODEL_SOURCE=DRIVE_FALLBACK` cuando el local no existe y Drive sí;
- ejecución con `epsilon=0.0`;
- reward/steps de la partida o resumen equivalente;
- `ASSAULT_DELIVERY_MODEL_EXECUTION_PASS=True`;
- suite de tests aprobada;
- `git diff --check` sin errores bloqueantes;
- evidencia de que no se modificó la lógica DDQN ni se reentrenó el agente.

## 13. Riesgos y mitigaciones

### R01 — Romper el flujo histórico de Drive

**Mitigación:** mantener la ruta actual como fallback y agregar el modelo local únicamente como primera prioridad.

### R02 — Ejecutar entrenamiento accidentalmente

**Mitigación:** la sección 15 no debe invocar resolución AUTO/NEW/RESUME ni funciones de entrenamiento; solo carga y evaluación.

### R03 — Entregar un modelo distinto al evaluado

**Mitigación:** usar exactamente el modelo compacto de HU009C/HU011 y conservar sus metadatos/lineage.

### R04 — Duplicar lógica existente

**Mitigación:** reutilizar `load_inference_model`, `create_assault_env` y las abstracciones de rutas existentes.

### R05 — Introducir regresiones justo antes de la entrega

**Mitigación:** limitar el diff a la nueva sección, tests focalizados y documentación necesaria; prohibir refactors oportunistas y cambios no relacionados.

## 14. Gate final esperado

```text
SECTION_15_DELIVERY_MODEL_PRESENT=PASS
DELIVERY_MODEL_PRIORITY=PASS
DELIVERY_MODEL_DRIVE_FALLBACK=PASS
DELIVERY_MODEL_LOAD=PASS
DELIVERY_MODEL_NO_TRAINING=PASS
DELIVERY_MODEL_EPSILON_ZERO=PASS
DELIVERY_MODEL_ENVIRONMENT=PASS
DELIVERY_MODEL_PORTABILITY=PASS
HU011_REGRESSION=PASS
ASSAULT_DELIVERY_MODEL_EXECUTION_PASS=True
HU012_DELIVERY_MODEL_GATE=PASS
```
