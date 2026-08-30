# HU009C — Artefactos de entrega: modelo compacto, evidencias TensorBoard, video y reporte técnico

## 1. Identificación

- **ID:** HU009C
- **Nombre:** Artefactos de entrega del agente Assault DDQN
- **Estado:** PENDIENTE
- **Dependencia previa:** HU009 — Entrenamiento DDQN completo `[COMPLETADA]`.
- **Dependencia relacionada no bloqueante:** HU008B — automatización de reanudación multisesión. HU009C no requiere interrumpir ni repetir el entrenamiento full para generar los artefactos de entrega.
- **Habilita:** HU010 — Optimización controlada de hiperparámetros; además contribuye directamente a HU011 — Evaluación formal contra baseline y HU012 — Evidencias y entrega final.
- **Entorno objetivo:** Google Colab, con posibilidad de validar artefactos de inferencia en CPU cuando sea suficiente.
- **Fuente de verdad de código:** GitHub `main`/SHA explícito.
- **Fuente de verdad de resultados:** corrida full `assault_ddqn_full_001`, checkpoint final, TensorBoard, MLflow y notebook ejecutado.
- **Algoritmo:** DDQN con Replay Buffer uniforme; esta HU no cambia el algoritmo ni reentrena el agente.

---

## 2. Contexto y problema

HU009 produjo una corrida full trazable del agente DDQN para `ALE/Assault-v5` hasta `250000` timesteps y una evaluación de 10 episodios. El checkpoint final sirve para continuidad de entrenamiento porque conserva Online Network, Target Network, optimizer, `global_step` y Replay Buffer. Esa propiedad es útil operacionalmente, pero el checkpoint completo pesa aproximadamente **2.88 GB**, principalmente por el Replay Buffer de observaciones visuales, y no es el artefacto adecuado para entregar o ejecutar inferencia.

El reto académico exige, para cada problema, un notebook ejecutable en Colab, el **modelo entrenado**, un **video corto** que evidencie entrenamiento y comportamiento aprendido, y un reporte técnico con hiperparámetros, librerías/versiones, hardware, tiempo de entrenamiento, evaluación de al menos 10 episodios, gráficas de evolución de recompensa y análisis de resultados.

El proyecto ya registra señales de entrenamiento mediante TensorBoard (`episode/reward`, `episode/reward_mean`, `train/loss`, `train/q_mean`, `train/epsilon`, `train/learning_rate`) y resultados agregados mediante MLflow. HU009C debe convertir esa trazabilidad en **artefactos finales pequeños, verificables y presentables**, sin duplicar lógica DDQN ni gastar GPU repitiendo el entrenamiento full.

---

## 3. Historia de usuario

> **Como** equipo que debe entregar el agente DDQN de Assault, **quiero** transformar la corrida full ya validada en un modelo compacto de inferencia, un conjunto mínimo de gráficas útiles, un video reproducible del comportamiento del agente y un reporte técnico integrado al notebook, **para** cumplir los artefactos exigidos por el reto sin depender del checkpoint de entrenamiento de 2.88 GB ni repetir cómputo innecesario.

---

## 4. Objetivo verificable

Al finalizar HU009C debe existir un flujo reproducible que, partiendo del checkpoint final de una corrida full válida:

1. genere un artefacto compacto de inferencia que no contenga Replay Buffer ni estado innecesario para explotación;
2. pueda cargar ese artefacto en un proceso/runtime limpio y ejecutar el agente con `epsilon=0.0`;
3. conserve metadatos suficientes para demostrar de qué entrenamiento provino;
4. lea los eventos TensorBoard persistidos y produzca **máximo tres figuras** no redundantes que expliquen la evolución del aprendizaje;
5. genere un video corto reproducible del agente jugando Assault a partir del modelo compacto;
6. incluya en el mismo video evidencia breve del proceso de entrenamiento, sin volver a ejecutar 250000 timesteps;
7. complete dentro de `2_Assault/assault_ddqn.ipynb` el reporte técnico requerido por el enunciado;
8. mantenga separadas las responsabilidades de entrenamiento, evaluación, visualización y empaquetado;
9. deje tests y evidencia verificable de todos los artefactos producidos.

HU009C **no declara por sí sola una nueva mejora de desempeño**. La comparación formal y la decisión final contra baseline permanecen en HU011, aunque el reporte puede presentar los resultados ya observados y dejar claramente identificado qué medición corresponde a la corrida full.

---

## 5. Alcance

### 5.1 Modelo compacto de inferencia

Implementar una utilidad reutilizable, preferentemente en un módulo dedicado como:

```text
2_Assault/src/model_artifact.py
```

Debe permitir exportar desde el checkpoint full un artefacto de inferencia, por ejemplo:

```text
<BASE>/models/assault_ddqn_full_001/assault_ddqn_model.pt
```

El artefacto compacto debe contener únicamente lo necesario para reconstruir y ejecutar la política:

- `online_network.state_dict()`;
- arquitectura mínima necesaria (`input_channels=4`, `num_actions=7`);
- contrato de preprocesamiento necesario para inferencia;
- `environment.id` y parámetros que condicionan comparabilidad (`frame_skip`, `repeat_action_probability`, `full_action_space`);
- seed/config fingerprint cuando esté disponible;
- `project_run_id`;
- `source_checkpoint_step`;
- referencia/identidad del checkpoint fuente;
- Git SHA/ref ejecutado cuando esté disponible;
- versión de schema del artefacto.

No debe contener:

- Replay Buffer;
- optimizer state;
- histórico completo de métricas;
- Target Network, salvo que exista una justificación técnica explícita. Para inferencia greedy DDQN la política se obtiene de la Online Network;
- objetos dependientes de una sesión Colab activa.

Debe existir una API equivalente a:

```python
artifact = export_inference_model(...)
agent, metadata = load_inference_model(...)
```

La carga debe validar shapes y metadatos antes de ejecutar el agente.

### 5.2 Validación del modelo compacto

Después de exportarlo:

- comprobar que el archivo existe y es cargable;
- comprobar que es materialmente menor que el checkpoint full y establecer como guardrail `size < 100 MiB`, salvo justificación documentada;
- crear un agente nuevo y cargar únicamente el artefacto compacto;
- ejecutar al menos un episodio de smoke sin mutar pesos;
- comprobar que las acciones greedy del modelo compacto coinciden con las del modelo fuente para un conjunto fijo de observaciones de prueba o estados reproducibles;
- ejecutar la evaluación formal posterior desde el modelo compacto, no desde el objeto de entrenamiento que quedó en memoria.

El checkpoint full de aproximadamente 2.88 GB **se conserva** para resume/auditoría y no debe ser sobrescrito ni reemplazado por el modelo compacto.

### 5.3 Gráficas TensorBoard — máximo 3 figuras

El notebook debe leer los eventos TensorBoard de la corrida full persistida y construir exactamente las figuras que aporten evidencia académica. No duplicar información en figuras diferentes.

Máximo permitido: **3 figuras**.

#### Figura 1 — Recompensa durante entrenamiento

Debe incluir:

- `episode/reward`;
- media móvil de 10 episodios (`episode/reward_mean`) o cálculo equivalente si el tag no está disponible;
- eje X por `global_step`;
- identificación del `project_run_id` y target final.

**Pregunta que responde:** ¿la política obtuvo recompensas mayores/estables a medida que avanzó el entrenamiento?

#### Figura 2 — Pérdida DDQN

Debe incluir:

- `train/loss`;
- curva suavizada o media móvil claramente identificada, sin ocultar la señal original cuando sea razonable;
- eje X por `global_step`.

**Pregunta que responde:** ¿el proceso de optimización se mantuvo finito y cómo evolucionó el error TD/Huber durante el aprendizaje?

#### Figura 3 — Q-value medio y exploración

Combinar en una sola figura, con ejes claramente diferenciados:

- `train/q_mean`;
- `train/epsilon`;
- mismo eje X `global_step`;
- dos ejes Y o representación equivalente legible.

**Pregunta que responde:** ¿cómo evolucionaron las estimaciones de valor mientras disminuía la exploración?

No crear gráficas adicionales de `learning_rate` si permanece constante, ni repetir la recompensa en otra visualización salvo que el reporte justifique una excepción. La evaluación final de 10 episodios puede presentarse como tabla/resumen estadístico y **no necesita consumir una cuarta gráfica**.

La lógica de transformación/visualización debe ser reutilizable fuera del notebook cuando tenga sentido, por ejemplo en:

```text
2_Assault/src/reporting.py
```

El notebook conserva la responsabilidad de presentación final.

### 5.4 Video corto del agente y evidencia de entrenamiento

Generar un MP4 reproducible desde el modelo compacto. Preferir `render_mode="rgb_array"` y evitar capturas de pantalla manuales como mecanismo principal.

El video final debe evidenciar los dos elementos exigidos por el enunciado:

1. **proceso de entrenamiento:** una introducción breve con los datos reales de la corrida full y/o una animación/placa generada desde sus métricas (algoritmo DDQN, `250000` timesteps, evolución de recompensa, checkpoint/modelo fuente, tiempo de entrenamiento). No volver a ejecutar el entrenamiento full únicamente para grabarlo;
2. **comportamiento aprendido:** render de al menos una partida del agente cargado desde el modelo compacto, con `epsilon=0.0` y configuración de evaluación declarada.

El segmento de gameplay debe:

- usar `ALE/Assault-v5` mediante la fábrica del proyecto;
- conservar `frame_skip=4`, `repeat_action_probability=0.25`, `Discrete(7)` y preprocessing contractual;
- usar `render_mode="rgb_array"`;
- mostrar reward acumulada y, si es técnicamente simple, timestep/episodio como overlay;
- utilizar una seed explícita/documentada;
- no hacer `agent.update`, no tocar Replay Buffer y no modificar pesos;
- cerrar correctamente el entorno y writer de video.

Artefacto esperado, por ejemplo:

```text
<BASE>/videos/assault_ddqn_full_001/assault_ddqn_demo.mp4
```

El video debe ser corto y apto para entrega. Si se selecciona un episodio representativo en vez del primero de una evaluación predefinida, el criterio de selección debe quedar documentado para evitar cherry-picking silencioso.

### 5.5 Reporte técnico dentro del notebook

Completar `2_Assault/assault_ddqn.ipynb` con una estructura profesional que incluya, como mínimo:

1. **Problema y objetivo:** Assault y métrica de recompensa promedio.
2. **Selección del algoritmo:** DDQN y justificación frente a las alternativas permitidas por el reto.
3. **Entorno y preprocessing:** `ALE/Assault-v5`, RGB base, grayscale, resize `84x84`, stack de 4 frames, frameskip efectivo 4, stochasticity `repeat_action_probability=0.25`, 7 acciones.
4. **Arquitectura del agente:** CNN Atari-style, Online/Target Network, Replay Buffer uniforme y actualización DDQN.
5. **Hiperparámetros efectivos de la corrida full:** extraídos de la configuración efectiva/artefactos, no transcritos manualmente si pueden derivarse.
6. **Condiciones de ejecución:** Python, Gymnasium, ALE-Py, PyTorch, CUDA, GPU/VRAM, RAM y SHA ejecutado usando los artefactos reales de runtime/MLflow.
7. **Entrenamiento:** target global, episodios completados, updates, epsilon inicial/final, tamaño del Replay Buffer, checkpoints, tiempo de entrenamiento y estado de observabilidad.
8. **Resultados de entrenamiento:** las tres figuras definidas en 5.3 y análisis conciso de cada una.
9. **Evaluación:** resultados de al menos 10 episodios independientes con reward medio, mediana, desviación estándar, mínimo, máximo y epsilon de evaluación.
10. **Comparación con baseline:** política aleatoria de HU001 bajo protocolo comparable, indicando claramente las cifras observadas y la mejora relativa sin inventar resultados.
11. **Comportamiento aprendido:** descripción apoyada en métricas y en el video; evitar afirmar estrategias que no puedan observarse.
12. **Limitaciones:** presupuesto de entrenamiento, una seed principal, stochasticity de ALE, tamaño de muestra de evaluación y cualquier limitación de comparabilidad.
13. **Conclusión:** capacidad observada de DDQN para superar la política aleatoria y próximos pasos si aplica.
14. **Artefactos de entrega:** enlaces/rutas al modelo compacto, checkpoint full, video y referencias a TensorBoard/MLflow.

Los resultados ya ejecutados pueden incorporarse, pero el notebook debe obtener o validar programáticamente las cifras siempre que sea posible. No dejar números históricos hardcodeados como fuente primaria si existen JSON/MLflow/TensorBoard que los contienen.

---

## 6. Fuera de alcance

HU009C no debe:

- reentrenar automáticamente el agente durante 250000 timesteps;
- cambiar DDQN por otro algoritmo;
- implementar PER, Dueling DQN, Rainbow, n-step o Noisy Nets;
- realizar HPO;
- modificar hiperparámetros para buscar un mejor score;
- modificar el checkpoint full original;
- introducir Model Registry, serving, API o infraestructura cloud adicional;
- subir checkpoints de varios GB al repositorio Git;
- duplicar en el notebook lógica de agente, entorno, evaluación o procesamiento que pertenezca a `src/`;
- crear más de tres figuras de entrenamiento sin una decisión posterior explícita.

---

## 7. Decisiones y restricciones técnicas

1. **Notebook como orquestador/reporte.** La exportación/carga del modelo, preparación de métricas y grabación del video deben vivir preferentemente en módulos reutilizables de `src/`; el notebook invoca y presenta.
2. **No duplicar lógica de inferencia.** `DDQNAgent.select_action(..., epsilon=0.0)` y la fábrica `create_assault_env(...)` continúan siendo las fuentes de verdad.
3. **Modelo compacto != checkpoint de resume.** Son artefactos distintos y ambos deben coexistir.
4. **Carga segura y verificable.** El loader debe validar schema, arquitectura y metadata mínima antes de aplicar pesos.
5. **Evaluación sin entrenamiento.** El flujo de video/evaluación no puede llamar `update`, no debe crear gradientes y no puede mutar el Replay Buffer.
6. **Persistencia fuera de Git para binarios grandes.** Checkpoints, MP4, eventos TensorBoard y modelos binarios se almacenan en Google Drive/artefactos definidos por el proyecto; Git versiona código, notebook, documentación y metadatos livianos.
7. **Trazabilidad.** El modelo compacto debe apuntar al `project_run_id`, checkpoint/step y SHA fuente.
8. **Sin resultados inventados.** Si un artefacto de TensorBoard/MLflow no está disponible, el notebook debe fallar con mensaje claro o marcar evidencia pendiente; no fabricar curvas ni métricas.
9. **Reproducibilidad del video.** Seed, epsilon y configuración de entorno deben registrarse junto al video.
10. **Compatibilidad Colab.** Evitar codecs/dependencias exóticas; usar una solución MP4 soportada por Colab y declarar cualquier dependencia nueva en `requirements.txt`.
11. **SOLID/DRY pragmático.** Separar exportación de modelo, reporting y video solo si cada módulo tiene responsabilidad clara; evitar frameworks internos innecesarios.

---

## 8. Plan de implementación / tareas

### T01 — Contrato del artefacto compacto

- Definir schema/versionado y metadatos mínimos.
- Implementar export desde checkpoint final sin Replay Buffer/optimizer.
- Implementar loader que reconstruya un `DDQNAgent` compatible para inferencia.
- No depender de objetos vivos del notebook.

**Resultado:** modelo compacto exportable/cargable y trazable hasta el checkpoint fuente.

### T02 — Tests del modelo compacto

- Crear checkpoint sintético/pequeño compatible.
- Exportar artefacto compacto.
- Verificar ausencia de Replay Buffer y optimizer.
- Verificar shape/metadata.
- Cargar en agente nuevo.
- Comparar Q-values/acción greedy con el agente fuente sobre observaciones controladas.
- Verificar guardrail de tamaño mediante el artefacto real en Colab.

**Resultado:** equivalencia funcional de inferencia demostrada.

### T03 — Integración con la corrida full

- Resolver explícitamente checkpoint final de `assault_ddqn_full_001` o project id configurable.
- Exportar el modelo compacto a persistencia.
- Registrar checksum (por ejemplo SHA-256), tamaño y metadata.
- Validar que el checkpoint full permanece intacto.

**Resultado:** artefacto de entrega identificado inequívocamente.

### T04 — Reporting TensorBoard

- Cargar tags de la corrida full con `load_tensorboard_scalars(...)` o helper reutilizable.
- Construir las tres figuras definidas en 5.3.
- Manejar tags ausentes con errores accionables.
- Mantener global step monotónico y continuidad de sesiones si existiesen múltiples event files.

**Resultado:** máximo tres figuras, cada una respondiendo una pregunta distinta del proceso de aprendizaje.

### T05 — Video reproducible

- Crear entorno `eval` con `render_mode="rgb_array"`.
- Cargar **modelo compacto** en un agente nuevo.
- Ejecutar episodio con epsilon `0.0`.
- Capturar frames y generar MP4.
- Crear una intro/segmento breve con evidencia real del entrenamiento y unirla al gameplay.
- Registrar seed, reward, steps, modelo/checksum y configuración usados para el video.

**Resultado:** MP4 reproducible que evidencia entrenamiento y comportamiento aprendido.

### T06 — Reporte técnico del notebook

- Reorganizar markdown del notebook sin mover lógica reutilizable a celdas monolíticas.
- Poblar secciones desde config/runtime/training/evaluation artifacts.
- Insertar las tres figuras.
- Insertar o enlazar el video de forma compatible con Colab.
- Mostrar rutas/checksums del modelo compacto y checkpoint fuente.
- Incorporar análisis del baseline sin duplicar ejecución si HU001 ya provee evidencia comparable.

**Resultado:** notebook que funciona simultáneamente como orquestador reproducible y reporte académico.

### T07 — Validación desde runtime limpio

- Abrir runtime Colab limpio.
- Checkout de SHA conocido.
- Instalar requirements.
- Montar persistencia.
- Cargar **únicamente el modelo compacto** para inferencia.
- Ejecutar smoke y evaluación de 10 episodios o la evaluación formal definida por HU011.
- Abrir/validar las figuras y reproducir el MP4.

**Resultado:** artefactos utilizables sin depender del estado de memoria de la corrida original.

### T08 — Documentación y cierre

- Documentar rutas, tamaños, checksum, SHA fuente y evidencia.
- Actualizar `implementacion.md` solo con resultados realmente observados.
- No versionar binarios grandes por accidente.

---

## 9. Criterios de aceptación

### CA01 — Modelo compacto

**Given** un checkpoint full válido de HU009, **When** se exporta el artefacto de inferencia, **Then** el archivo contiene la Online Network y metadata contractual, no contiene Replay Buffer ni optimizer, y es menor a 100 MiB salvo excepción justificada.

### CA02 — Equivalencia de política

**Given** el agente fuente y un agente nuevo cargado desde el modelo compacto, **When** ambos reciben las mismas observaciones con `epsilon=0.0`, **Then** sus Q-values son equivalentes dentro de tolerancia numérica y seleccionan la misma acción greedy.

### CA03 — Carga limpia

**Given** un runtime/proceso sin el agente entrenado en memoria, **When** se carga el modelo compacto, **Then** puede ejecutarse al menos un episodio de Assault sin acceder al Replay Buffer del checkpoint full.

### CA04 — Trazabilidad

**Given** el modelo compacto, **When** se inspecciona su metadata, **Then** se puede identificar `project_run_id`, checkpoint/step fuente, arquitectura/preprocessing y SHA/ref cuando estén disponibles.

### CA05 — Figuras no redundantes

**Given** los eventos TensorBoard de la corrida full, **When** se genera el reporte, **Then** existen como máximo tres figuras: recompensa+media móvil, loss y q_mean+epsilon; no hay una cuarta figura que repita la misma evidencia.

### CA06 — Datos reales de TensorBoard

**Given** los event files persistidos, **When** se cargan las figuras, **Then** los puntos provienen de tags reales y conservan `global_step`; si falta un tag obligatorio, la generación falla explícitamente en lugar de inventar datos.

### CA07 — Video de comportamiento

**Given** el modelo compacto, **When** se genera el video, **Then** el gameplay usa `epsilon=0.0`, entorno contractual y seed documentada, sin actualizaciones de entrenamiento, y produce un MP4 reproducible.

### CA08 — Video evidencia entrenamiento + explotación

**Given** el requerimiento académico del reto, **When** se reproduce el video final, **Then** contiene una sección breve con evidencia real del entrenamiento y otra con el render del comportamiento aprendido.

### CA09 — Reporte técnico

**Given** el notebook final, **When** se revisa como reporte, **Then** contiene algoritmo/justificación, hiperparámetros, versiones, hardware, tiempo, entrenamiento, máximo tres figuras, evaluación de al menos 10 episodios, comparación con baseline, comportamiento aprendido, limitaciones, conclusión y artefactos.

### CA10 — Correspondencia modelo/evaluación

**Given** el score reportado para el agente entregable, **When** se inspecciona el flujo de evaluación, **Then** el modelo evaluado corresponde al artefacto compacto derivado del checkpoint full documentado y no a un agente diferente residente en memoria.

### CA11 — No regresión

**Given** las suites existentes, **When** se implementa HU009C, **Then** las pruebas HU002–HU009 continúan pasando y no se altera la semántica DDQN, preprocessing, checkpoint resume ni tracking.

---

## 10. Definition of Done

HU009C puede marcarse `[COMPLETADA]` únicamente cuando:

- [ ] existe módulo/función reutilizable de export y load del modelo compacto;
- [ ] el modelo compacto real fue generado desde el checkpoint full seleccionado;
- [ ] el modelo compacto no incluye Replay Buffer ni optimizer;
- [ ] tamaño real `< 100 MiB` o excepción justificada y documentada;
- [ ] checksum y metadata del modelo quedaron registrados;
- [ ] equivalencia de inferencia fuente vs compacto está probada;
- [ ] carga desde runtime limpio está validada;
- [ ] existen máximo tres figuras TensorBoard y cubren reward, loss, q_mean y epsilon;
- [ ] las figuras usan datos reales del entrenamiento full;
- [ ] existe MP4 reproducible del agente cargado desde el modelo compacto;
- [ ] el video evidencia tanto proceso de entrenamiento como comportamiento aprendido;
- [ ] el reporte técnico del notebook contiene todos los puntos exigidos por el enunciado;
- [ ] la evaluación mostrada usa al menos 10 episodios y `epsilon` explícito;
- [ ] modelo, evaluación, video y reporte apuntan a la misma corrida/modelo fuente;
- [ ] tests focales nuevos pasan;
- [ ] suite completa relevante pasa;
- [ ] `python -m compileall -q 2_Assault/src` pasa;
- [ ] notebook es JSON válido y ejecutable en Colab;
- [ ] no se versionaron checkpoints, event files, MP4 o binarios grandes accidentalmente;
- [ ] documentación y evidencia final quedaron actualizadas;
- [ ] PR limitado al alcance de HU009C y revisable.

---

## 11. Autovalidaciones obligatorias

### AV01 — Compilación

```bash
python -m compileall -q 2_Assault/src
```

**Éxito:** exit code 0.

### AV02 — Tests focales del modelo de inferencia

```bash
python -m pytest 2_Assault/tests/test_model_artifact.py -q
```

Debe validar export/load, schema, ausencia de Replay Buffer/optimizer, equivalencia de Q-values/acciones y errores de metadata incompatible.

### AV03 — Tests focales de reporting/video

```bash
python -m pytest 2_Assault/tests/test_reporting.py 2_Assault/tests/test_video.py -q
```

Los nombres pueden ajustarse si se decide una organización equivalente; los tests deben comprobar transformación de scalars, límite de tres figuras, manejo de tags faltantes y generación de video en una prueba corta.

### AV04 — Suite completa

```bash
python -m pytest 2_Assault/tests -q
```

**Éxito:** no introducir fallos nuevos; skips justificados por GPU/Colab son permitidos.

### AV05 — Integridad notebook

Procedimiento automático o semiautomático:

- parsear `2_Assault/assault_ddqn.ipynb` como JSON;
- comprobar que importa helpers de `src/` en vez de duplicar implementación;
- comprobar que contiene las secciones del reporte técnico;
- comprobar que no contiene más de tres figuras de entrenamiento planificadas;
- comprobar que no contiene IDs/checkpoints locales históricos como única fuente operacional.

### AV06 — Artefacto compacto real — Colab

Desde persistencia de HU009:

1. resolver checkpoint final;
2. exportar modelo compacto;
3. imprimir tamaño y SHA-256;
4. validar tamaño `<100 MiB`;
5. cargarlo en agente nuevo;
6. ejecutar smoke con `epsilon=0.0`.

**Validación Colab pendiente de ejecución por el usuario** hasta que exista salida real. Codex no debe inventar tamaño ni checksum.

### AV07 — Evaluación desde modelo compacto — Colab

Ejecutar al menos 10 episodios independientes usando el **modelo compacto cargado desde disco**, con recompensa raw y `epsilon=0.0` salvo decisión explícita distinta.

Registrar:

- lista de rewards;
- mean/median/std/min/max;
- episode lengths;
- seed/protocolo;
- checksum del modelo evaluado.

### AV08 — TensorBoard real

Cargar event files de la corrida full y comprobar presencia de:

```text
episode/reward
train/loss
train/q_mean
train/epsilon
```

Generar solo las tres figuras definidas. Validar que los steps cubren el entrenamiento observado y que no existen NaN/Inf en los scalars utilizados.

### AV09 — Video real

- generar el MP4 desde el modelo compacto;
- verificar archivo existente y tamaño > 0;
- reproducirlo en Colab;
- confirmar visualmente que se observa gameplay de Assault;
- validar metadata lateral: seed, reward, steps, epsilon y checksum/model id;
- comprobar que incluye evidencia breve del entrenamiento.

### AV10 — Consistencia de entrega

Crear un resumen verificable que confirme:

```text
checkpoint source
  -> model compact checksum
  -> evaluation 10 episodes
  -> TensorBoard figures
  -> video
  -> notebook report
```

Todas las flechas deben apuntar al mismo `project_run_id`/modelo fuente.

---

## 12. Evidencias esperadas

Al cerrar HU009C deben conservarse, como mínimo:

- path y tamaño del checkpoint full fuente;
- path, tamaño y SHA-256 del modelo compacto;
- metadata del modelo (`project_run_id`, source step, SHA/config fingerprint);
- resultado de equivalencia de inferencia;
- evidencia de carga desde runtime limpio;
- tres figuras generadas desde TensorBoard;
- tags y rango de `global_step` usados;
- path/metadata del MP4;
- reward/steps del episodio mostrado;
- evaluación de al menos 10 episodios desde el modelo compacto;
- resultados del baseline utilizados para comparación;
- runtime/hardware/versiones;
- tiempo de entrenamiento registrado por HU009;
- resultados de tests;
- SHA Git ejecutado;
- notebook ejecutado/revisado como reporte técnico.

Los binarios grandes permanecen fuera de Git salvo que exista una decisión explícita distinta.

---

## 13. Riesgos y consideraciones

### R01 — Confundir checkpoint con modelo de entrega

Mitigación: nombres/rutas distintos, schemas distintos y tests que rechacen Replay Buffer/optimizer en el artefacto compacto.

### R02 — Evaluar accidentalmente el agente residente en memoria

Mitigación: evaluación de aceptación desde un agente nuevo cargado del modelo compacto en runtime/proceso limpio.

### R03 — Curvas incompletas o múltiples event files

Mitigación: loader que agregue eventos por tag y ordene/deduzca por `global_step`, detectando conflictos en vez de silenciarlos.

### R04 — Gráficas decorativas o redundantes

Mitigación: cada una responde una pregunta explícita; máximo tres figuras y no graficar learning rate constante.

### R05 — Video no cumple el enunciado

Mitigación: una sola pieza final con breve evidencia del proceso de entrenamiento + gameplay del modelo aprendido.

### R06 — Cherry-picking del gameplay

Mitigación: seed/criterio de selección explícito y reward del episodio mostrado registrado.

### R07 — Codec no disponible en Colab

Mitigación: usar MP4 con dependencia común/portable y test de generación corto; declarar dependencia en `requirements.txt` si es necesaria.

### R08 — Artefactos demasiado grandes en Git

Mitigación: persistir binarios en Drive/almacenamiento de artefactos, extender `.gitignore` si es necesario y añadir test/check previo a commit.

### R09 — Reporte desacoplado de los artefactos reales

Mitigación: poblar cifras desde JSON/MLflow/TensorBoard/config cuando sea posible y añadir un bloque final de consistencia con IDs/checksums.

---

## 14. Flujo esperado

```text
checkpoint_step_250000.pt  (~2.88 GB, resume/auditoría)
            │
            ├── export_inference_model(...)
            ▼
assault_ddqn_model.pt       (<100 MiB, inferencia/entrega)
            │
            ├── load en agente nuevo
            ├── evaluación >=10 episodios
            └── gameplay rgb_array
                     │
                     ▼
             assault_ddqn_demo.mp4

TensorBoard full run
      │
      ├── reward + media móvil
      ├── loss
      └── q_mean + epsilon
              │
              ▼
       reporte técnico notebook
```

HU009C debe cerrar la brecha entre **"el agente ya fue entrenado"** y **"el agente está listo para ser entregado, evaluado y demostrado"**, sin convertir el notebook en un monolito ni repetir entrenamiento costoso.

---

## 15. Ejecuci?n por defecto desde Colab limpio

El notebook de entrega debe ser ejecutable con **Run All** desde Google Colab aun cuando el almacenamiento persistente est? vac?o.

La orquestaci?n por defecto usa:

```text
ASSAULT_EXECUTION_MODE=auto
```

Sem?ntica:

- si existe el checkpoint final esperado, resolver `AUTO_RESOLUTION=DELIVERY` y no volver a entrenar;
- si no existe checkpoint final pero hay una sesi?n parcial v?lida, resolver `AUTO_RESOLUTION=RESUME` mediante `prepare_training_session(...)`;
- si no existe manifest ni checkpoint, resolver `AUTO_RESOLUTION=NEW` mediante `prepare_training_session(...)` y entrenar desde `global_step=0`;
- `ASSAULT_EXECUTION_MODE=train` fuerza el flujo de entrenamiento existente;
- `ASSAULT_EXECUTION_MODE=delivery` exige checkpoint final existente y falla de forma clara si no est? disponible.

Caso de aceptaci?n principal pendiente de validaci?n real:

```text
Given:
- Colab limpio
- repositorio clonado desde GitHub
- Drive vac?o
- GPU disponible

When:
- el evaluador ejecuta Run All sin modificar variables

Then:
- instala dependencias
- monta Drive
- valida entorno
- inicia entrenamiento desde step 0
- entrena DDQN hasta el target full
- guarda checkpoints peri?dicos
- persiste TensorBoard y MLflow
- genera checkpoint final
- exporta modelo compacto
- eval?a >=10 episodios con epsilon=0
- genera tres figuras
- genera video
- muestra video inline
- presenta reporte final
```

Esta validaci?n real no debe declararse completada sin ejecutar el entrenamiento full en Colab.
