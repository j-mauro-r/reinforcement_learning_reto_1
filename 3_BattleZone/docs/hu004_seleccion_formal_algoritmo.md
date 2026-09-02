# HU004 — Selección formal del algoritmo para BattleZone

## 1. Identificación

- **ID:** HU004
- **Nombre:** Selección formal del algoritmo para BattleZone
- **Estado:** [COMPLETADA]
- **Dependencia previa:** HU003 — Pipeline reproducible del entorno `[COMPLETADA]`
- **Habilita:** HU005 — Núcleo del agente
- **Gate posterior:** HU005 debe implementar únicamente el algoritmo seleccionado y documentado por HU004.
- **Fuentes de verdad:**
  - `enunciado_reto_1.txt`;
  - `3_BattleZone/docs/ficha_tecnica.md`;
  - `3_BattleZone/docs/implementacion.md`;
  - `3_BattleZone/docs/lineamientos.md`;
  - `3_BattleZone/docs/arquitectura.md`;
  - `3_BattleZone/docs/hu002_experimento_0_baseline_aleatorio.md` y evidencia empírica equivalente de HU002;
  - `3_BattleZone/docs/hu003_pipeline_reproducible_entorno.md`;
  - `3_BattleZone/docs/hu003_evidencia_implementacion.md`;
  - `3_BattleZone/configs/battlezone_config.yaml`;
  - `3_BattleZone/docs/hu004_decision_algoritmo.md`.

---

## 2. Contexto y problema

HU001 caracterizó `ALE/BattleZone-v5`, HU002 construyó el baseline aleatorio y HU003 congeló el contrato perceptual del entorno. Por tanto, HU004 es el primer punto del proyecto en el que puede tomarse una decisión algorítmica sin mezclar simultáneamente cambios de entorno, preprocessing y agente.

Las evidencias disponibles condicionan la selección:

### Evidencia de HU002

- action space completo: `Discrete(18)`;
- recompensa aleatoria media: `3000`;
- mediana aleatoria: `2000`;
- desviación estándar poblacional aproximada: `3065.94`;
- rango observado: `0` a `10000`;
- duración media aproximada: `1159.5` steps por episodio;
- recompensa positiva en aproximadamente `0.1725 %` de los steps;
- reward `0` en aproximadamente `99.8275 %` de los steps;
- alta variabilidad entre episodios;
- sticky actions y dinámica estocástica del entorno.

### Contrato congelado por HU003

- entorno: `ALE/BattleZone-v5`;
- input final: `(4, 128, 128, 3)`;
- dtype: `uint8`;
- pipeline: `battlezone_rgb_128_stack4_no_crop`;
- RGB preservado;
- resize `128x128`;
- frame stack `4`;
- cropping desactivado para preservar radar;
- `frameskip=4` aplicado una sola vez;
- `repeat_action_probability=0.25`;
- action space `Discrete(18)`;
- reward sin clipping, normalización ni shaping;
- train y eval usan el mismo contrato perceptual;
- tamaño aproximado por estado: `196608` bytes (`0.1875 MB`).

Estas condiciones hacen que la decisión de algoritmo deba considerar simultáneamente:

- observaciones visuales de alta dimensionalidad;
- 18 acciones discretas;
- recompensa extremadamente escasa;
- alta varianza del retorno;
- necesidad de eficiencia muestral;
- costo de memoria de estados apilados;
- estabilidad del aprendizaje;
- complejidad de implementación y depuración;
- restricciones de tiempo, RAM/VRAM y duración de sesiones de Google Colab.

HU004 no debe escoger un algoritmo por preferencia previa, por haber sido utilizado en Assault ni por considerarlo una convención Atari. La decisión debe quedar respaldada por una matriz comparativa reproducible y una justificación técnica explícita basada en BattleZone.

---

## 3. Historia de usuario

> **Como** equipo responsable de desarrollar el agente de Reinforcement Learning para BattleZone, **quiero** comparar de forma estructurada los algoritmos permitidos por el reto y seleccionar uno a partir de las restricciones reales del entorno y del pipeline ya validado, **para** que HU005 implemente una arquitectura justificada, viable y coherente con el presupuesto computacional disponible.

---

## 4. Objetivo verificable

Al finalizar HU004 deberá existir una decisión técnica versionada que:

1. evalúe explícitamente DQN, DQN + PER, DDQN y REINFORCE;
2. use criterios derivados de HU001, HU002, HU003, arquitectura y lineamientos;
3. defina pesos y escala de puntuación antes de calcular el resultado final;
4. documente la evidencia que sustenta cada puntuación;
5. produzca una matriz comparativa auditable;
6. seleccione exactamente un algoritmo para HU005;
7. identifique los componentes técnicos que dicho algoritmo exigirá en HU005–HU010;
8. identifique los principales riesgos de la selección;
9. no implemente todavía el agente ni realice entrenamiento del modelo;
10. mantenga la independencia total respecto de `2_Assault/`.

La salida principal de HU004 será una **decisión de arquitectura**, no un resultado de entrenamiento.

---

## 5. Alcance

### 5.1 Algoritmos obligatorios a comparar

La matriz deberá incluir exactamente los cuatro algoritmos permitidos por el reto:

1. DQN;
2. DQN + Prioritized Experience Replay (PER);
3. DDQN;
4. REINFORCE.

No se incluirán como candidatos seleccionables algoritmos fuera del enunciado, por ejemplo PPO, A2C, SAC, Rainbow, C51, QR-DQN u otros.

Pueden mencionarse únicamente como contexto académico si fuera indispensable, pero no deben puntuar ni competir en la matriz.

### 5.2 Criterios mínimos obligatorios

La comparación debe incluir, como mínimo:

1. **Compatibilidad con action space discreto de 18 acciones.**
2. **Adecuación a observación visual/temporal** `(4,128,128,3)`.
3. **Eficiencia muestral.**
4. **Manejo esperado de reward extremadamente sparse.**
5. **Sensibilidad a alta varianza de retornos.**
6. **Estabilidad esperada del aprendizaje.**
7. **Costo de memoria.**
8. **Costo computacional por actualización.**
9. **Complejidad de implementación y validación.**
10. **Compatibilidad con checkpoints y resume.**
11. **Compatibilidad con entrenamiento por timesteps y sesiones fragmentadas de Colab.**
12. **Facilidad de observabilidad con TensorBoard.**
13. **Riesgo técnico para completar el reto dentro del presupuesto disponible.**
14. **Coherencia con el baseline y la duración observada de episodios.**

### 5.3 Criterios derivados directamente de BattleZone

La decisión debe explicar explícitamente cómo afectan al algoritmo:

- `99.8275 %` de steps con reward `0` en HU002;
- alta variabilidad del retorno aleatorio;
- episodios de alrededor de mil steps en promedio;
- 18 acciones;
- input visual RGB con cuatro frames;
- sticky actions `0.25`;
- necesidad de conservar radar y contexto temporal;
- costo de almacenar experiencia si el método utiliza Replay Buffer.

### 5.4 Matriz de decisión

Crear una matriz versionada con columnas mínimas:

| Campo | Descripción |
|---|---|
| Criterio | Dimensión evaluada |
| Peso | Importancia relativa |
| Evidencia BattleZone | Dato o restricción que origina el criterio |
| DQN | Puntuación |
| DQN+PER | Puntuación |
| DDQN | Puntuación |
| REINFORCE | Puntuación |
| Justificación | Razón de las puntuaciones |

La escala debe fijarse antes de puntuar. Recomendación:

```text
1 = muy desfavorable
2 = desfavorable
3 = aceptable
4 = favorable
5 = muy favorable
```

Los pesos deben sumar `100 %` o `1.0`.

No se permite alterar los pesos después de conocer el ganador sin documentar explícitamente la razón y recalcular toda la matriz.

### 5.5 Método de puntuación

Para cada algoritmo:

```text
score_total = Σ(peso_criterio × puntuación_criterio)
```

La implementación debe mostrar:

- puntuación bruta por criterio;
- puntuación ponderada;
- score total;
- ranking final.

Si dos alternativas quedan prácticamente empatadas, la decisión no debe resolverse de forma arbitraria. Debe aplicarse un desempate explícito basado, en este orden, en:

1. eficiencia muestral;
2. estabilidad esperada;
3. costo/viabilidad en Colab;
4. complejidad/riesgo de implementación.

### 5.6 Evidencia y trazabilidad de puntuaciones

Cada puntuación debe poder rastrearse a una de estas fuentes:

- evidencia HU001/HU002/HU003;
- arquitectura/lineamientos del proyecto;
- enunciado académico;
- material académico o fuente técnica autorizada cuando sea necesario para una propiedad específica del algoritmo.

No se deben presentar como hechos propiedades de un algoritmo que no estén respaldadas por alguna fuente del proyecto o referencia técnica identificable.

Las inferencias del equipo deben marcarse como **inferencia técnica** y no como dato empírico de BattleZone.

### 5.7 Sensibilidad de la decisión

La matriz debe incluir una comprobación sencilla de sensibilidad para evitar que la selección dependa de un único peso arbitrario.

Como mínimo:

- identificar los tres criterios de mayor peso;
- variar razonablemente uno de ellos;
- recalcular el ranking;
- documentar si el ganador cambia.

No se requiere optimización matemática ni simulación Monte Carlo.

### 5.8 Resultado de arquitectura para HU005

La decisión final debe indicar qué componentes serán necesarios a partir de HU005.

#### Si gana DQN

HU005 deberá considerar:

- CNN/Q-Network;
- Online Network;
- Target Network;
- Replay Buffer uniforme;
- epsilon-greedy;
- cálculo de targets DQN.

#### Si gana DQN + PER

Además de los elementos DQN:

- prioridades;
- muestreo no uniforme;
- importance-sampling weights;
- actualización de prioridades;
- parámetros `alpha`/`beta` posteriores.

#### Si gana DDQN

HU005 deberá considerar:

- Online Network;
- Target Network;
- Replay Buffer uniforme;
- epsilon-greedy;
- selección de acción mediante Online Network;
- evaluación de la acción mediante Target Network.

#### Si gana REINFORCE

HU005 deberá considerar:

- policy network;
- distribución sobre 18 acciones;
- trayectorias completas;
- retornos por episodio;
- optimizer;
- ausencia de Replay Buffer clásico y Target Network.

HU004 no implementará ninguno de estos componentes; solo definirá cuál conjunto corresponde a la decisión final.

### 5.9 Entregable de decisión

La implementación de HU004 deberá crear:

`3_BattleZone/docs/hu004_decision_algoritmo.md`

El documento debe incluir como mínimo:

1. problema de decisión;
2. evidencia BattleZone utilizada;
3. criterios;
4. pesos;
5. escala;
6. matriz completa;
7. cálculo de scores;
8. ranking;
9. sensibilidad;
10. algoritmo seleccionado;
11. justificación final;
12. implicaciones para HU005;
13. riesgos/limitaciones;
14. tabla de autovalidaciones de HU004.

### 5.10 Configuración

Una vez tomada la decisión, puede actualizarse `3_BattleZone/configs/battlezone_config.yaml` únicamente para registrar de forma declarativa el nombre del algoritmo seleccionado, si hacerlo no introduce todavía hiperparámetros ni rompe el contrato de HU003.

No deben añadirse en HU004 valores de learning rate, gamma, batch size, Replay Buffer, epsilon, target sync u otros hiperparámetros de implementación salvo que el documento de decisión los mencione únicamente como futuras variables de HU005/HU012.

---

## 6. Fuera de alcance

HU004 **no** debe:

- implementar `network.py`;
- implementar `agent.py`;
- implementar `replay_buffer.py`;
- implementar `trainer.py`;
- crear optimizer;
- ejecutar forward pass de la red;
- actualizar pesos;
- entrenar DQN, DQN+PER, DDQN o REINFORCE;
- ejecutar benchmarks de rendimiento que impliquen entrenamiento del agente;
- optimizar hiperparámetros;
- implementar checkpoints o resume;
- implementar TensorBoard;
- crear `run_manifest.json` de entrenamiento;
- realizar evaluación formal del agente;
- modificar el preprocessing aprobado en HU003;
- reducir el action space;
- aplicar reward clipping, normalization o shaping;
- modificar `2_Assault/`;
- importar o copiar código desde `2_Assault/`;
- utilizar MLflow.

Si durante HU004 se identifica una posible mejora del preprocessing, debe documentarse como riesgo o hipótesis futura; no debe reabrirse HU003 dentro del mismo PR.

---

## 7. Decisiones y restricciones técnicas

### 7.1 La decisión debe preceder al código del agente

No se permite crear componentes del agente para justificar posteriormente una elección ya implementada.

Primero se aprueba HU004. Después HU005 implementa exclusivamente el algoritmo ganador.

### 7.2 Independencia de Assault

La experiencia de Assault puede utilizarse como conocimiento metodológico, pero ninguna puntuación puede justificarse con la afirmación “se usó en Assault”. La selección debe sostenerse sobre BattleZone y las restricciones de este proyecto.

### 7.3 No modificar el contrato HU003

La comparación utiliza como entrada fija:

```text
shape = (4, 128, 128, 3)
dtype = uint8
action_space = Discrete(18)
frameskip = 4
sticky_actions = 0.25
reward_transform = none
```

Cambiar simultáneamente el algoritmo y el preprocessing invalidaría la comparabilidad de la decisión.

### 7.4 MLOps ligera

HU004 es una decisión versionada mediante Git/GitHub. No requiere MLflow, TensorBoard, checkpoints ni infraestructura experimental adicional.

### 7.5 SOLID/DRY

HU004 es principalmente documental. No deben crearse abstracciones de código para una matriz que puede resolverse de forma transparente con Markdown y, opcionalmente, una utilidad mínima de cálculo si aporta auditabilidad real.

### 7.6 Validar barato antes de entrenar caro

HU004 debe cerrarse sin consumir una sesión larga de GPU. La selección debe poder revisarse localmente mediante documentos, cálculos deterministas y evidencia ya disponible.

### 7.7 Separar evidencia de inferencia

El documento deberá etiquetar claramente:

- **dato empírico:** medido en HU002/HU003;
- **restricción del proyecto:** definida por enunciado/arquitectura/lineamientos;
- **propiedad del algoritmo:** sustentada por material técnico;
- **inferencia técnica:** conclusión razonada del equipo.

### 7.8 No forzar un ganador

El algoritmo con mayor score será la recomendación por defecto, pero cualquier excepción deberá:

1. documentarse;
2. explicar por qué la matriz no captura un riesgo material;
3. ser aprobada explícitamente antes de HU005.

---

## 8. Plan de implementación / tareas

### T01 — Validar gate de entrada

**Cambio:** confirmar que HU003 está `[COMPLETADA]` y que existe el contrato final del entorno.

**Resultado esperado:** HU004 parte de un estado perceptual estable.

---

### T02 — Consolidar evidencia de HU001–HU003

**Archivos fuente:** ficha técnica, evidencia HU002, evidencia HU003, configuración.

**Cambio:** construir una tabla de restricciones y datos que afecten la decisión.

**Resultado esperado:** todas las puntuaciones parten de evidencia explícita.

**Depende de:** T01.

---

### T03 — Definir criterios y pesos

**Archivo:** `3_BattleZone/docs/hu004_decision_algoritmo.md`.

**Cambio:** definir criterios, escala y pesos antes de puntuar algoritmos.

**Resultado esperado:** método de decisión reproducible y no ajustado al ganador.

**Depende de:** T02.

---

### T04 — Caracterizar los cuatro algoritmos

**Cambio:** documentar únicamente propiedades necesarias para los criterios de DQN, DQN+PER, DDQN y REINFORCE.

**Resultado esperado:** comparación suficiente sin convertir HU004 en una revisión bibliográfica general.

**Depende de:** T03.

---

### T05 — Construir matriz comparativa

**Cambio:** asignar puntuaciones justificadas y calcular scores ponderados.

**Resultado esperado:** ranking calculable y auditable.

**Depende de:** T04.

---

### T06 — Ejecutar análisis de sensibilidad

**Cambio:** variar de forma controlada al menos uno de los criterios de mayor peso y recalcular ranking.

**Resultado esperado:** conocer si la decisión es robusta o frágil ante pesos razonables.

**Depende de:** T05.

---

### T07 — Seleccionar algoritmo

**Cambio:** declarar exactamente un algoritmo ganador, justificarlo y documentar por qué los otros tres quedan descartados para la primera implementación.

**Resultado esperado:** HU005 recibe una decisión no ambigua.

**Depende de:** T06.

---

### T08 — Derivar implicaciones para HU005

**Cambio:** listar componentes obligatorios y componentes que no deben existir según el algoritmo seleccionado.

**Resultado esperado:** HU005 puede redactarse sin reinterpretar la arquitectura.

**Depende de:** T07.

---

### T09 — Actualizar configuración solo si aporta trazabilidad

**Archivo opcional:** `3_BattleZone/configs/battlezone_config.yaml`.

**Cambio:** registrar únicamente el nombre del algoritmo seleccionado, sin introducir hiperparámetros prematuros.

**Resultado esperado:** decisión declarativa disponible para siguientes HUs sin alterar HU003.

**Depende de:** T07.

---

### T10 — Ejecutar autovalidaciones y revisar alcance

**Cambio:** validar pesos, cálculos, ranking, documentación, independencia de Assault y ausencia de código de agente.

**Resultado esperado:** PR exclusivamente documental/de decisión, listo para revisión.

**Depende de:** T08-T09.

---

## 9. Criterios de aceptación

### CA01 — Gate HU003 satisfecho

**Dado** que HU004 depende del pipeline reproducible,  
**cuando** se inicia la historia,  
**entonces** HU003 figura `[COMPLETADA]` y su contrato final está documentado.

### CA02 — Cuatro algoritmos permitidos

**Dado** el enunciado del reto,  
**cuando** se revisa la matriz,  
**entonces** contiene DQN, DQN+PER, DDQN y REINFORCE y ningún algoritmo externo compite como candidato seleccionable.

### CA03 — Criterios derivados del proyecto

**Dado** HU001–HU003,  
**cuando** se revisan los criterios,  
**entonces** sparse reward, alta varianza, action space, input visual, costo de memoria y restricciones de Colab están representados explícitamente.

### CA04 — Pesos definidos antes del resultado

**Dado** el método de decisión,  
**cuando** se examinan pesos y escala,  
**entonces** están documentados, suman `100 %` o `1.0` y no dependen del algoritmo ganador.

### CA05 — Matriz auditable

**Dado** cada puntuación,  
**cuando** se solicita su origen,  
**entonces** existe una justificación basada en evidencia, restricción, propiedad técnica o inferencia marcada explícitamente.

### CA06 — Cálculo reproducible

**Dado** los pesos y puntuaciones,  
**cuando** se recalculan los scores,  
**entonces** se obtiene el mismo ranking documentado.

### CA07 — Sensibilidad evaluada

**Dado** el ranking inicial,  
**cuando** se modifica razonablemente un criterio de alto peso,  
**entonces** el documento muestra si el ganador permanece o cambia.

### CA08 — Selección única

**Dado** el ranking final,  
**cuando** HU004 concluye,  
**entonces** existe exactamente un algoritmo seleccionado para HU005.

### CA09 — Alternativas descartadas justificadamente

**Dado** los tres algoritmos no seleccionados,  
**cuando** se revisa la conclusión,  
**entonces** se explica por qué cada uno resulta menos conveniente para la primera implementación BattleZone.

### CA10 — Implicaciones HU005 explícitas

**Dado** el algoritmo seleccionado,  
**cuando** se prepara HU005,  
**entonces** puede determinarse qué red, memoria, target, exploración o trayectoria debe implementarse y qué componentes no aplican.

### CA11 — Sin entrenamiento

**Dado** el alcance de HU004,  
**cuando** se revisan los cambios,  
**entonces** no existe entrenamiento, actualización de pesos, Replay Buffer operativo ni componentes del agente.

### CA12 — HU003 permanece estable

**Dado** el contrato aprobado,  
**cuando** se revisa el PR,  
**entonces** no cambian preprocessing, frameskip, frame stack, action space ni reward transform.

### CA13 — Independencia de Assault

**Dado** la implementación de HU004,  
**cuando** se revisan archivos y referencias,  
**entonces** no se modifica ni importa `2_Assault/` y la elección no se fundamenta en reutilizar su código.

### CA14 — Sin MLflow

**Dado** los lineamientos de BattleZone,  
**cuando** se revisa la solución,  
**entonces** no se introduce MLflow ni infraestructura experimental innecesaria.

### CA15 — Decisión versionada

**Dado** HU004 cerrada,  
**cuando** se consulta `main`,  
**entonces** existe `hu004_decision_algoritmo.md` con matriz, ranking, sensibilidad, algoritmo ganador y evidencia de validación.

---

## 10. Autovalidaciones obligatorias

### AV01 — Estado de dependencia

**Procedimiento:** revisar HU003 e `implementacion.md`.

**PASS:** HU003 aparece `[COMPLETADA]`.

---

### AV02 — Candidatos permitidos

**Procedimiento:** inspeccionar la matriz de decisión.

**PASS:** exactamente DQN, DQN+PER, DDQN y REINFORCE como candidatos.

---

### AV03 — Suma de pesos

**Procedimiento:** sumar todos los pesos de la matriz.

**PASS:** total igual a `100 %` o `1.0`, tolerando únicamente error de redondeo explícito.

---

### AV04 — Escala válida

**Procedimiento:** revisar cada puntuación.

**PASS:** todos los valores usan la escala declarada y están dentro de rango.

---

### AV05 — Recalcular scores

**Procedimiento:** recalcular `Σ(peso × puntuación)` para los cuatro algoritmos.

**PASS:** scores y ranking coinciden con el documento.

---

### AV06 — Trazabilidad de evidencia

**Procedimiento:** seleccionar al menos una puntuación por criterio y verificar su fuente/justificación.

**PASS:** no existen puntuaciones críticas sin explicación.

---

### AV07 — Evidencia BattleZone

**Procedimiento:** verificar que la matriz incorpora explícitamente sparse reward, variabilidad, action space, contrato visual y restricciones computacionales.

**PASS:** dichos factores influyen de forma identificable en la decisión.

---

### AV08 — Sensibilidad

**Procedimiento:** modificar el peso de al menos uno de los tres criterios principales dentro de un rango razonable y recalcular.

**PASS:** resultado y efecto sobre ranking documentados correctamente.

---

### AV09 — Selección única

**Procedimiento:** inspeccionar la conclusión.

**PASS:** un único algoritmo está marcado como seleccionado.

---

### AV10 — Componentes posteriores coherentes

**Procedimiento:** comparar algoritmo ganador con la sección de implicaciones HU005.

**PASS:** los componentes requeridos corresponden al algoritmo y no incluyen piezas ajenas sin justificación.

---

### AV11 — Sin implementación del agente

**Procedimiento:** revisar diff del PR.

**PASS:** no se crean/modifican módulos de network, agent, replay buffer o trainer con lógica de aprendizaje.

---

### AV12 — Contrato HU003 intacto

**Procedimiento:** revisar diff de configuración/environment.

**PASS:** ninguna modificación cambia shape, RGB, resize, frame stack, frameskip, sticky actions, action space o reward passthrough.

---

### AV13 — Independencia de Assault

**Procedimiento:** revisar diff y buscar referencias a `2_Assault/`.

**PASS:** cero modificaciones/imports/copia de código Assault.

---

### AV14 — Ausencia de MLflow

**Procedimiento:** buscar nuevas referencias/imports/configuración de MLflow dentro del alcance HU004.

**PASS:** ninguna.

---

### AV15 — Alcance del PR

**Procedimiento:** comparar rama HU004 contra `main`.

**PASS:** cambios limitados a documentos de HU004 y, opcionalmente, registro declarativo del algoritmo seleccionado en configuración.

---

## 11. Definition of Done (DoD)

HU004 se considera `[COMPLETADA]` únicamente cuando:

- [ ] HU003 está cerrada y su contrato se conserva sin cambios;
- [ ] existe `3_BattleZone/docs/hu004_decision_algoritmo.md`;
- [ ] DQN, DQN+PER, DDQN y REINFORCE fueron comparados;
- [ ] criterios y pesos están documentados;
- [ ] pesos suman `100 %` o `1.0`;
- [ ] escala de puntuación está definida;
- [ ] cada criterio tiene relación explícita con el problema BattleZone o una restricción del proyecto;
- [ ] cada puntuación material posee justificación;
- [ ] scores ponderados fueron recalculados y verificados;
- [ ] existe ranking final;
- [ ] análisis de sensibilidad fue realizado;
- [ ] exactamente un algoritmo fue seleccionado;
- [ ] se justificó por qué los otros tres no fueron seleccionados;
- [ ] implicaciones concretas para HU005 están documentadas;
- [ ] no se implementó ni entrenó el agente;
- [ ] no se modificó el contrato perceptual de HU003;
- [ ] no se modificó ni reutilizó código de `2_Assault/`;
- [ ] no se introdujo MLflow;
- [ ] AV01–AV15 están en `PASS` o cualquier excepción aprobada está documentada;
- [ ] evidencia de la decisión está versionada;
- [ ] el PR está limitado al alcance de HU004 y listo para revisión.

---

## 12. Evidencias esperadas

La implementación deberá conservar como mínimo:

1. `3_BattleZone/docs/hu004_decision_algoritmo.md`;
2. tabla de evidencia BattleZone utilizada;
3. tabla de criterios, pesos y escala;
4. matriz DQN vs DQN+PER vs DDQN vs REINFORCE;
5. scores ponderados y ranking;
6. cálculo/revisión de suma de pesos;
7. análisis de sensibilidad;
8. algoritmo seleccionado y justificación;
9. razones de descarte de las otras tres alternativas;
10. implicaciones arquitectónicas para HU005;
11. tabla AV01–AV15 con PASS/FAIL;
12. diff del PR demostrando ausencia de código del agente, Assault y MLflow.

No se requiere checkpoint, modelo, TensorBoard, GPU ni métricas de entrenamiento para cerrar HU004.

---

## 13. Riesgos y mitigaciones

### R01 — Selección basada en preferencia previa

**Riesgo:** escoger un algoritmo porque el equipo ya lo conoce o porque fue usado en otro entorno.

**Mitigación:** matriz obligatoria basada en evidencia BattleZone y pesos definidos antes del resultado.

### R02 — Pesos manipulados para producir un ganador

**Riesgo:** ajustar criterios después de observar la puntuación.

**Mitigación:** documentar criterios/pesos antes de puntuar y ejecutar análisis de sensibilidad.

### R03 — Confundir evidencia empírica con expectativa teórica

**Riesgo:** afirmar que un algoritmo funcionará mejor en BattleZone sin haberlo entrenado.

**Mitigación:** etiquetar propiedades del algoritmo e inferencias técnicas; HU004 selecciona por adecuación esperada, no demuestra desempeño empírico.

### R04 — Penalizar incorrectamente métodos con mayor complejidad

**Riesgo:** que complejidad de implementación domine criterios más relevantes como eficiencia muestral o estabilidad.

**Mitigación:** pesos explícitos y sensibilidad sobre criterios de mayor impacto.

### R05 — Ignorar costo de Replay Buffer

**Riesgo:** métodos value-based pueden requerir memoria significativa con estados `(4,128,128,3)`.

**Mitigación:** costo de memoria debe formar parte explícita de la matriz y de las implicaciones para HU005.

### R06 — Ignorar sparse reward

**Riesgo:** una matriz genérica para Atari podría no reflejar que HU002 observó ~`99.83 %` de reward cero.

**Mitigación:** sparse reward es criterio obligatorio y debe influir explícitamente en las puntuaciones.

### R07 — Adelantar HU005

**Riesgo:** comenzar a programar redes/replay/training para “probar” candidatos.

**Mitigación:** AV11 bloqueante y revisión de diff.

### R08 — Reabrir HU003

**Riesgo:** modificar input o preprocessing para favorecer un algoritmo.

**Mitigación:** contrato HU003 fijo y AV12 bloqueante.

### R09 — Contaminación con Assault

**Riesgo:** reutilizar código o justificar la selección con el agente previo.

**Mitigación:** independencia explícita, AV13 y revisión de alcance.

### R10 — Decisión frágil

**Riesgo:** un cambio mínimo de pesos invierte el ganador.

**Mitigación:** análisis de sensibilidad obligatorio; si la decisión es frágil, documentar la incertidumbre y el criterio de desempate antes de cerrar HU004.

---

## 14. Resultado esperado para HU005

HU005 debe recibir una especificación no ambigua compuesta por:

- nombre del algoritmo seleccionado;
- razones principales de selección;
- riesgos principales;
- contrato de entrada heredado de HU003;
- action space `Discrete(18)`;
- lista de componentes obligatorios del agente;
- lista de componentes no aplicables;
- estrategia conceptual de exploración o muestreo según algoritmo;
- necesidad o ausencia de Replay Buffer;
- necesidad o ausencia de Target Network;
- necesidad o ausencia de trayectorias completas;
- criterios que HU005 deberá autovalidar antes de pasar a HU006.

HU005 no debe reconsiderar el algoritmo salvo que aparezca evidencia nueva que invalide materialmente una premisa de HU004. En ese caso deberá documentarse explícitamente la desviación antes de implementar una alternativa.

---

## 15. Resultado formal de HU004

La evidencia consolidada y auditable de la selección quedó registrada en:

- `3_BattleZone/docs/hu004_decision_algoritmo.md`

Resultado final de la matriz ponderada:

1. DDQN: `3.72`
2. DQN: `3.34`
3. DQN + PER: `3.30`
4. REINFORCE: `2.14`

Algoritmo seleccionado para HU005:

- `DDQN`

Sensibilidad:

- Escenario S1 (prioridad a eficiencia muestral): DDQN permanece primero.
- Escenario S2 (prioridad a simplicidad/costo): DDQN permanece primero.
- Conclusión: la selección es robusta dentro de los escenarios evaluados.

### Estado de criterios de aceptación

- CA01: PASS
- CA02: PASS
- CA03: PASS
- CA04: PASS
- CA05: PASS
- CA06: PASS
- CA07: PASS
- CA08: PASS
- CA09: PASS
- CA10: PASS
- CA11: PASS
- CA12: PASS
- CA13: PASS
- CA14: PASS
- CA15: PASS

### Estado de autovalidaciones

- AV01: PASS
- AV02: PASS
- AV03: PASS
- AV04: PASS
- AV05: PASS
- AV06: PASS
- AV07: PASS
- AV08: PASS
- AV09: PASS
- AV10: PASS
- AV11: PASS
- AV12: PASS
- AV13: PASS
- AV14: PASS
- AV15: PASS

### Definition of Done

La checklist de DoD de HU004 queda cumplida mediante la evidencia versionada de la sección 15 y el documento `3_BattleZone/docs/hu004_decision_algoritmo.md`.
