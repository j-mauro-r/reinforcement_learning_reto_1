# HU003 — Pipeline reproducible del entorno BattleZone

## 1. Identificación

- **ID:** HU003
- **Nombre:** Pipeline reproducible del entorno BattleZone
- **Estado:** `[COMPLETADA]`
- **Dependencia previa:** HU002 — Experimento 0 y baseline aleatorio `[COMPLETADA]`
- **Habilita:** HU004 — Selección formal del algoritmo
- **Gate posterior:** el contrato de observación y acciones debe quedar estable antes de implementar el agente en HU005.
- **Cierre:** PR #21 mergeado a `main`; merge commit `beffdacf0e3e5d8b656bfee3f11e88eacc3b7228`.
- **Fuentes de verdad:**
  - `enunciado_reto_1.txt`;
  - `3_BattleZone/docs/implementacion.md`;
  - `3_BattleZone/docs/lineamientos.md`;
  - `3_BattleZone/docs/arquitectura.md`;
  - `3_BattleZone/docs/ficha_tecnica.md`;
  - evidencias empíricas de HU002;
  - documentación oficial de Gymnasium y Arcade Learning Environment para `ALE/BattleZone-v5`.

---

## 2. Contexto y problema

HU001 caracterizó técnicamente `ALE/BattleZone-v5` y HU002 produjo un baseline aleatorio con evidencia empírica suficiente para diseñar el pipeline del entorno sin copiar supuestos de otros juegos Atari.

HU002 confirmó, entre otros aspectos:

- observación visual RGB de `210×160×3`, `uint8`;
- action space `Discrete(18)`;
- `frameskip=4` en el entorno base;
- `repeat_action_probability=0.25`;
- episodios completos con alta variabilidad de recompensa;
- recompensa extremadamente escasa: aproximadamente `99.83 %` de los steps tuvieron reward `0` en la muestra observada;
- dependencia de información visual pequeña y estratégica, especialmente el radar;
- necesidad de contexto temporal para interpretar movimiento, orientación, enemigos y proyectiles;
- riesgo de perder información mediante resize o cropping agresivo.

El proyecto necesita ahora una **única forma reproducible y reutilizable de crear BattleZone**, aplicar preprocessing, controlar seeds y producir observaciones con un contrato estable para entrenamiento y evaluación.

Esta HU no selecciona algoritmo ni entrena un agente. Su objetivo es estabilizar el entorno que posteriormente consumirán HU004–HU014.

El notebook de HU002 `3_BattleZone/experimento_0_battlezone.ipynb` se conserva únicamente como evidencia y conocimiento del proyecto. **HU003 no debe importarlo, copiar su lógica ni modificarlo.**

---

## 3. Historia de usuario

> **Como** equipo que desarrollará y evaluará el agente de Reinforcement Learning para BattleZone, **quiero** disponer de un pipeline único, reproducible y probado para crear y preprocesar `ALE/BattleZone-v5`, **para** garantizar que entrenamiento, evaluación y futuros experimentos utilicen exactamente el mismo contrato de observaciones, acciones y configuración temporal.

---

## 4. Objetivo verificable

Al finalizar HU003 deberá existir una implementación reproducible que permita, desde una configuración versionada:

1. crear `ALE/BattleZone-v5` mediante una única fábrica del entorno;
2. controlar explícitamente seed del entorno y del `action_space`;
3. aplicar el preprocessing seleccionado para BattleZone;
4. preservar información estratégica del radar y la escena principal;
5. aplicar `frameskip` efectivo exactamente una vez;
6. producir un contrato estable de observación con shape y dtype conocidos;
7. soportar creación diferenciada para `train` y `eval` sin divergencia de preprocessing;
8. validar `Discrete(18)` y las 18 acciones esperadas;
9. detectar hardware disponible sin hacer depender el entorno de GPU;
10. superar smoke tests focalizados localmente;
11. demostrar el flujo mediante un notebook nuevo, independiente de HU002.

La HU deberá concluir con una decisión explícita y versionada sobre el pipeline definitivo que utilizarán HU004 en adelante.

---

## 5. Alcance

### 5.1 Configuración centralizada

Crear una configuración versionada bajo:

`3_BattleZone/configs/`

Como mínimo deberá centralizar:

- `env_id`;
- seed base;
- `mode`;
- `difficulty`;
- `obs_type` del entorno base;
- `frameskip` del entorno base;
- `repeat_action_probability`;
- estrategia de preprocessing seleccionada;
- conversión RGB/grayscale cuando aplique;
- resize seleccionado;
- cropping, únicamente si queda justificado;
- `frame_stack` seleccionado;
- action space esperado;
- parámetros diferenciados de train/eval únicamente cuando exista una razón válida;
- render desactivado por defecto.

No deben existir constantes mágicas duplicadas entre notebook, tests y `src/`.

La configuración de HU003 debe limitarse al entorno y preprocessing. Los hiperparámetros propios del algoritmo se definirán después de HU004.

### 5.2 Fábrica única del entorno

Crear:

`3_BattleZone/src/environment.py`

Debe ser la única fuente de verdad para construir el entorno BattleZone utilizado por entrenamiento y evaluación.

Responsabilidades mínimas:

- validar configuración;
- crear `ALE/BattleZone-v5`;
- establecer mode/difficulty cuando corresponda;
- inicializar seeds;
- aplicar wrappers de preprocessing en un orden explícito;
- aplicar frame stacking cuando corresponda;
- validar contrato final de observación;
- validar action space;
- permitir `train` y `eval` mediante parámetros explícitos sin duplicar la lógica base;
- cerrar correctamente el entorno.

Las funciones públicas reutilizables deberán usar docstrings estilo Google.

### 5.3 Preprocessing visual basado en evidencia

HU003 deberá evaluar alternativas sobre BattleZone antes de fijar el pipeline definitivo.

Como mínimo se deberá comparar:

- RGB frente a grayscale;
- imagen original frente a al menos dos tamaños candidatos de resize;
- observación sin cropping frente a cropping únicamente si existe una hipótesis justificada;
- observación individual frente a una alternativa con contexto temporal mediante frame stacking.

`84×84`, grayscale y frame stack de 4 pueden utilizarse como **candidatos de comparación**, pero no deben adoptarse automáticamente por ser convenciones Atari ni por haber sido usados en otros proyectos.

La selección final deberá priorizar:

1. conservación del radar;
2. conservación de enemigos, proyectiles y obstáculos visibles;
3. conservación de contexto temporal suficiente;
4. reducción razonable de dimensionalidad;
5. costo de memoria y cómputo compatible con entrenamiento posterior en Colab.

### 5.4 Evidencia de preservación del radar

El notebook de HU003 deberá mostrar evidencia visual de:

- frame RGB original;
- región donde se observa el radar;
- salida de cada pipeline candidato relevante;
- salida del pipeline seleccionado.

No se aprobará un cropping que elimine el radar.

Si grayscale o un resize dificultan significativamente interpretar radar, enemigos u obstáculos, deberán descartarse o justificarse con evidencia adicional.

### 5.5 Contexto temporal y frame stacking

BattleZone requiere interpretar cambios entre frames. HU003 deberá establecer explícitamente el número de frames apilados que compondrán el estado final.

La decisión deberá:

- diferenciar `frameskip` de `frame_stack`;
- documentar el orden de wrappers;
- verificar shape final;
- demostrar que reset y step producen observaciones coherentes;
- evitar duplicación temporal accidental.

No debe confundirse frame stacking con repetir una acción.

### 5.6 Regla crítica de frameskip

`ALE/BattleZone-v5` ya utiliza `frameskip=4` según la configuración del proyecto.

HU003 deberá demostrar que ningún wrapper adicional aplica otro action repeat/frameskip de forma accidental.

El pipeline debe tener **un único lugar explícito** donde se configure el frameskip efectivo.

Un doble frameskip se considera defecto bloqueante porque alteraría dinámica, frecuencia de decisiones y comparabilidad con HU002.

### 5.7 Seeds y reproducibilidad

La fábrica deberá permitir seed explícita por creación/episodio.

Como mínimo:

```python
env.reset(seed=seed)
env.action_space.seed(seed)
```

La implementación deberá verificar reproducibilidad de aspectos controlables sin prometer determinismo absoluto debido a sticky actions, ALE y futuras ejecuciones GPU.

Train y eval deben poder utilizar conjuntos de seeds distintos y explícitos.

### 5.8 Separación train/eval

Debe existir una única fábrica configurable, no dos pipelines independientes.

Entrenamiento y evaluación deben compartir:

- environment ID;
- preprocessing;
- resize;
- canales;
- frame stacking;
- action space;
- frameskip.

Las diferencias permitidas deberán ser explícitas y justificadas, por ejemplo:

- seed;
- render/video;
- modo de evaluación cuando una HU posterior lo requiera.

La evaluación no puede cambiar silenciosamente el contrato de observación.

### 5.9 Action space

La fábrica debe verificar que el entorno expone:

`Discrete(18)`

y que, cuando `get_action_meanings()` esté disponible, se obtienen las 18 acciones esperadas.

HU003 **no reducirá el action space**. Cualquier decisión sobre reducción de acciones requeriría una justificación separada y no forma parte del alcance actual.

### 5.10 Dtype y eficiencia de memoria

Mantener observaciones como `uint8` mientras sea razonable dentro del pipeline del entorno.

La normalización/conversión a tensores deberá realizarse posteriormente en la capa apropiada del agente/red y no convertir permanentemente observaciones almacenables a `float32` sin necesidad.

El pipeline deberá reportar:

- shape final;
- dtype final;
- rango observado;
- tamaño aproximado de una observación;
- impacto aproximado del frame stacking sobre memoria por estado.

### 5.11 Detección de hardware

Crear utilidades mínimas bajo:

`3_BattleZone/src/utils.py`

solo cuando aporten valor transversal.

HU003 deberá poder reportar:

- Python/runtime;
- CPU;
- RAM;
- disponibilidad de CUDA/GPU cuando PyTorch esté disponible.

La creación y prueba del entorno debe funcionar en CPU. HU003 no requiere GPU para aprobarse.

### 5.12 Smoke tests focalizados

Crear tests bajo:

`3_BattleZone/tests/`

Como mínimo validar:

- carga de configuración;
- creación del entorno;
- seed válida;
- `Discrete(18)`;
- 18 action meanings cuando estén disponibles;
- shape y dtype del estado final;
- reset;
- varios steps aleatorios;
- `terminated`/`truncated` manejables;
- mismo contrato de observación para train/eval;
- ausencia de doble frameskip según la arquitectura implementada;
- cierre del entorno;
- independencia total de `2_Assault/`.

Las pruebas no deben entrenar un agente.

### 5.13 Notebook nuevo de HU003

Crear:

`3_BattleZone/pipeline_battlezone.ipynb`

Este notebook será un **orquestador y reporte de validación del pipeline**, no un repositorio de lógica duplicada.

Debe:

1. instalar/cargar dependencias necesarias;
2. importar la implementación desde `3_BattleZone/src/`;
3. cargar la configuración versionada;
4. mostrar runtime/hardware;
5. crear entorno mediante la fábrica única;
6. mostrar contrato raw;
7. comparar candidatos de preprocessing;
8. mostrar evidencia visual del radar;
9. mostrar contrato final seleccionado;
10. ejecutar un smoke test corto con acciones aleatorias;
11. presentar una tabla de autovalidaciones;
12. cerrar con la decisión de preprocessing de HU003.

**Restricción obligatoria:**

`3_BattleZone/experimento_0_battlezone.ipynb` no debe ser utilizado como base de código, importado, modificado ni copiado. Puede ser consultado únicamente como evidencia histórica de HU002.

### 5.14 Evidencia estructurada de la decisión

Crear un documento de evidencia, por ejemplo:

`3_BattleZone/docs/hu003_evidencia_implementacion.md`

Debe registrar:

- configuración evaluada;
- alternativas comparadas;
- decisión final de preprocessing;
- contrato final de observación;
- shape/dtype;
- frame stack;
- frameskip efectivo;
- action space;
- resultados de tests;
- resultados de autovalidaciones;
- limitaciones conocidas;
- implicaciones para HU004.

---

## 6. Fuera de alcance

HU003 **no** debe implementar ni seleccionar definitivamente:

- DQN;
- DQN + PER;
- DDQN;
- REINFORCE;
- CNN/Q-Network;
- policy network;
- Replay Buffer;
- Target Network;
- optimizer;
- ciclo de entrenamiento;
- epsilon-greedy u otra política de exploración del agente;
- checkpoints;
- resume;
- TensorBoard;
- MLflow;
- `run_manifest.json` de entrenamiento;
- entrenamiento corto o largo;
- optimización de hiperparámetros;
- evaluación formal del modelo.

HU003 tampoco debe:

- modificar `2_Assault/`;
- importar módulos de `2_Assault/`;
- copiar código de Assault;
- modificar `3_BattleZone/experimento_0_battlezone.ipynb`;
- reutilizar dicho notebook como módulo o fuente de funciones.

---

## 7. Decisiones y restricciones técnicas

### 7.1 Independencia de Assault

BattleZone debe tener implementación propia. Assault puede aportar conocimiento metodológico, pero no código.

### 7.2 Notebook como orquestador

La lógica reutilizable debe residir en `src/`. El notebook únicamente configura, invoca, muestra evidencia y documenta decisiones.

### 7.3 SOLID pragmático

- `environment.py` tendrá responsabilidad sobre creación/preprocessing del entorno;
- `utils.py` contendrá únicamente utilidades transversales pequeñas;
- no crear jerarquías de clases innecesarias;
- preferir composición de wrappers y funciones pequeñas;
- entrenamiento y evaluación futuros dependerán del contrato estable del entorno, no de detalles internos.

### 7.4 DRY

No duplicar:

- creación del entorno;
- preprocessing;
- seeds;
- carga de configuración;
- detección de dispositivo;
- validaciones de shape/action space.

### 7.5 Configuración antes que constantes

Las decisiones experimentales deben cambiarse mediante configuración cuando sea razonable, no editando múltiples archivos.

### 7.6 Reward sin transformación

HU003 no aplicará reward clipping, normalization ni shaping. La recompensa deberá mantenerse exactamente como la retorna ALE para preservar comparabilidad con HU002 y HU013.

### 7.7 Render

Render desactivado durante smoke tests normales salvo cuando se necesite evidencia visual. No mantener ventanas/render costoso durante ejecución estándar.

### 7.8 Reproducibilidad sin promesas absolutas

Se controlarán seeds y configuración, pero no se declarará determinismo absoluto cuando existan sticky actions o variabilidad legítima de ALE.

### 7.9 Git y alcance del PR

La implementación de HU003 deberá realizarse en una rama propia creada desde `main` actualizado y mediante un PR focalizado en `3_BattleZone/`.

El PR no debe modificar Assault ni mezclar trabajo de otras HUs.

---

## 8. Plan de implementación / tareas

### T01 — Crear rama y validar estado base

**Cambio:** crear rama propia de HU003 desde `main` actualizado.

**Resultado esperado:** HU003 parte del último estado aprobado del repositorio y no arrastra cambios no relacionados.

---

### T02 — Definir configuración del entorno

**Archivos:** `3_BattleZone/configs/`.

**Cambio:** centralizar parámetros del entorno y candidatos de preprocessing sin introducir hiperparámetros del algoritmo.

**Resultado esperado:** configuración versionada y sin constantes mágicas dispersas.

**Depende de:** T01.

---

### T03 — Implementar fábrica única

**Archivo:** `3_BattleZone/src/environment.py`.

**Cambio:** implementar creación, seeds, wrappers, validaciones y separación explícita train/eval.

**Resultado esperado:** cualquier consumidor futuro obtiene BattleZone mediante un único contrato.

**Depende de:** T02.

---

### T04 — Implementar utilidades transversales mínimas

**Archivo:** `3_BattleZone/src/utils.py`.

**Cambio:** añadir únicamente carga/configuración o detección de hardware si no corresponde a `environment.py`.

**Resultado esperado:** responsabilidades separadas sin sobrearquitectura.

**Depende de:** T02.

---

### T05 — Crear notebook nuevo de pipeline

**Archivo:** `3_BattleZone/pipeline_battlezone.ipynb`.

**Cambio:** crear notebook desde cero que importe `src/` y no reutilice código del Experimento 0.

**Resultado esperado:** notebook limpio, ejecutable y enfocado en HU003.

**Depende de:** T03-T04.

---

### T06 — Comparar preprocessing candidato

**Archivo:** notebook + configuración.

**Cambio:** generar evidencia RGB/grayscale, resize y temporalidad sobre frames reales de BattleZone.

**Resultado esperado:** las alternativas pueden compararse sin entrenar un agente.

**Depende de:** T05.

---

### T07 — Seleccionar y congelar contrato de observación

**Archivos:** configuración + `environment.py` + evidencia.

**Cambio:** fijar preprocessing final de HU003 y documentar shape, dtype, stack y frameskip.

**Resultado esperado:** HU004 y HU005 reciben un único contrato estable.

**Depende de:** T06.

---

### T08 — Implementar tests focalizados

**Archivos:** `3_BattleZone/tests/`.

**Cambio:** probar configuración, fábrica, seeds, action space, contrato, train/eval y varios steps.

**Resultado esperado:** errores estructurales pueden detectarse sin entrenamiento.

**Depende de:** T07.

---

### T09 — Ejecutar validación local barata

**Cambio:** ejecutar tests y notebook/smoke test localmente cuando el hardware disponible lo permita.

**Resultado esperado:** pipeline aprobado en CPU antes de cualquier uso innecesario de recursos Colab.

**Depende de:** T08.

---

### T10 — Consolidar evidencia y decisión

**Archivo:** `3_BattleZone/docs/hu003_evidencia_implementacion.md`.

**Cambio:** registrar resultados, autovalidaciones y contrato definitivo.

**Resultado esperado:** la decisión puede auditarse sin inspeccionar todo el código.

**Depende de:** T09.

---

### T11 — Validar alcance del PR

**Cambio:** revisar diff contra `main`.

**Resultado esperado:** cambios limitados a HU003/BattleZone, sin modificaciones en `2_Assault/` ni en el notebook HU002.

**Depende de:** T10.

---

## 9. Criterios de aceptación

### CA01 — Configuración centralizada

**Dado** el pipeline BattleZone,  
**cuando** se revisan sus parámetros de entorno y preprocessing,  
**entonces** existe una configuración versionada bajo `3_BattleZone/configs/` y no hay constantes críticas duplicadas en notebook y módulos.

### CA02 — Fábrica única

**Dado** un consumidor de BattleZone,  
**cuando** necesita crear un entorno,  
**entonces** utiliza la fábrica definida en `3_BattleZone/src/environment.py` tanto para train como para eval.

### CA03 — Entorno correcto

**Dado** la configuración aprobada,  
**cuando** se crea el entorno,  
**entonces** corresponde a `ALE/BattleZone-v5`, mode/difficulty configurados y sticky actions esperadas.

### CA04 — Action space estable

**Dado** el entorno creado,  
**cuando** se valida su action space,  
**entonces** es `Discrete(18)` y se conservan las 18 acciones.

### CA05 — Preprocessing justificado

**Dado** que HU002 evidenció riesgo de perder radar e información visual pequeña,  
**cuando** se selecciona grayscale/resize/cropping/frame stack,  
**entonces** la decisión está respaldada por comparación explícita sobre frames de BattleZone y no por herencia de otro Atari.

### CA06 — Radar preservado

**Dado** el preprocessing final,  
**cuando** se inspecciona su salida,  
**entonces** el radar continúa presente y razonablemente interpretable.

### CA07 — Contrato final explícito

**Dado** el pipeline seleccionado,  
**cuando** se ejecutan `reset()` y `step()`,  
**entonces** el shape, dtype y estructura de la observación coinciden con el contrato documentado.

### CA08 — Contexto temporal explícito

**Dado** BattleZone y su dependencia temporal,  
**cuando** se revisa el pipeline,  
**entonces** el valor de `frame_stack` está decidido, configurado y validado explícitamente.

### CA09 — Frameskip aplicado una sola vez

**Dado** `frameskip=4` del entorno base,  
**cuando** se inspecciona la cadena de wrappers,  
**entonces** no existe un segundo action repeat/frameskip accidental.

### CA10 — Train/eval equivalentes

**Dado** un entorno train y uno eval,  
**cuando** se comparan sus contratos de observación,  
**entonces** utilizan el mismo preprocessing, channels, resize, frame stack y action space.

### CA11 — Seeds explícitas

**Dado** una seed conocida,  
**cuando** se crea/reset el entorno,  
**entonces** se inicializan explícitamente entorno y `action_space` sin prometer determinismo absoluto.

### CA12 — Reward intacto

**Dado** un step del entorno,  
**cuando** pasa por el pipeline,  
**entonces** el reward no es clipped, normalizado ni shaped por HU003.

### CA13 — Smoke test funcional

**Dado** el pipeline final,  
**cuando** se ejecutan varios steps con acciones válidas,  
**entonces** no ocurren errores de shape, dtype, wrappers, action space o cierre del entorno.

### CA14 — Notebook independiente

**Dado** HU003,  
**cuando** se revisan sus notebooks,  
**entonces** existe `3_BattleZone/pipeline_battlezone.ipynb` creado para esta HU y `experimento_0_battlezone.ipynb` permanece sin modificaciones ni reutilización de código.

### CA15 — Independencia de Assault

**Dado** el PR de HU003,  
**cuando** se revisan imports y archivos modificados,  
**entonces** no existe dependencia, copia ni modificación de `2_Assault/`.

### CA16 — Sin selección prematura de algoritmo

**Dado** que HU004 es responsable de la selección formal,  
**cuando** se revisa HU003,  
**entonces** no se implementa ni se declara ganador DQN, DQN+PER, DDQN o REINFORCE.

### CA17 — Documentación de decisión

**Dado** HU003 terminada,  
**cuando** se consulta su evidencia,  
**entonces** puede conocerse exactamente el pipeline final y la razón de cada transformación relevante.

---

## 10. Autovalidaciones obligatorias

### AV01 — Carga de configuración

**Procedimiento:** cargar la configuración de HU003 desde un proceso limpio.

**PASS:** parámetros requeridos presentes, tipos válidos y sin dependencia del notebook.

---

### AV02 — Creación del entorno

**Procedimiento:** crear entorno mediante la fábrica única.

**PASS:** `ALE/BattleZone-v5` inicia correctamente y `reset()` retorna observación válida.

---

### AV03 — Action space

**Procedimiento:** inspeccionar `action_space` y `get_action_meanings()`.

**PASS:** `Discrete(18)` y 18 acciones esperadas.

---

### AV04 — Contrato de observación raw

**Procedimiento:** inspeccionar observación antes de preprocessing.

**PASS:** contrato raw coincide con lo documentado para BattleZone o cualquier discrepancia queda explicada.

---

### AV05 — Comparación de preprocessing

**Procedimiento:** generar y conservar evidencia de las alternativas evaluadas.

**PASS:** la decisión final no se basa únicamente en una convención Atari y queda justificada para BattleZone.

---

### AV06 — Radar

**Procedimiento:** inspeccionar visualmente original y salida final.

**PASS:** radar presente y utilizable; cropping no lo elimina.

---

### AV07 — Shape y dtype final

**Procedimiento:** ejecutar reset + varios steps y validar observaciones.

**PASS:** todas las observaciones respetan exactamente shape/dtype configurados.

---

### AV08 — Frame stack

**Procedimiento:** validar inicialización del stack y desplazamiento después de steps.

**PASS:** número de frames y dimensión resultante coinciden con configuración y contrato.

---

### AV09 — Frameskip único

**Procedimiento:** inspeccionar creación del entorno y cadena de wrappers; complementar con contadores de frame cuando estén disponibles.

**PASS:** no existe segundo wrapper que replique action repeat/frameskip sobre el `frameskip=4` configurado en ALE.

---

### AV10 — Train/eval contract parity

**Procedimiento:** crear ambos modos con seeds diferentes y comparar espacios/shape/dtype/wrappers relevantes.

**PASS:** contrato perceptual y action space equivalentes.

---

### AV11 — Seeds

**Procedimiento:** crear dos instancias controladas con misma configuración/seed y validar aspectos reproducibles, incluyendo seed de `action_space`.

**PASS:** controles de seed aplicados explícitamente y limitaciones de determinismo documentadas.

---

### AV12 — Reward passthrough

**Procedimiento:** comparar reward retornado por el entorno/wrapper frente al reward observado en la interfaz final del pipeline durante steps de prueba.

**PASS:** HU003 no altera reward.

---

### AV13 — Smoke tests

**Procedimiento:** ejecutar la suite focalizada de HU003.

**PASS:** todos los tests de HU003 pasan.

---

### AV14 — Notebook independiente

**Procedimiento:** revisar imports y diff del notebook.

**PASS:** `pipeline_battlezone.ipynb` usa `src/` y configuración versionada; no importa/copia/modifica `experimento_0_battlezone.ipynb`.

---

### AV15 — Independencia de Assault

**Procedimiento:** buscar referencias/imports a `2_Assault` y revisar diff del PR.

**PASS:** cero dependencias y cero cambios bajo `2_Assault/`.

---

### AV16 — Validación local barata

**Procedimiento:** ejecutar tests y smoke test en CPU local cuando el entorno local sea compatible.

**PASS:** pipeline funcional sin requerir GPU ni entrenamiento.

---

### AV17 — Alcance algorítmico

**Procedimiento:** revisar archivos creados/modificados.

**PASS:** no existen agente, CNN, Replay Buffer, optimizer, trainer ni selección formal de algoritmo introducidos por HU003.

---

## 11. Definition of Done (DoD)

HU003 se considera `[COMPLETADA]` únicamente cuando:

- [x] existe configuración centralizada bajo `3_BattleZone/configs/`;
- [x] existe fábrica única en `3_BattleZone/src/environment.py`;
- [x] utilidades transversales están separadas solo cuando aportan valor;
- [x] existe `3_BattleZone/pipeline_battlezone.ipynb` independiente;
- [x] se compararon alternativas de preprocessing sobre BattleZone;
- [x] se documentó la decisión final;
- [x] radar e información crítica permanecen preservados;
- [x] se fijó el contrato final de observación;
- [x] frame stack está explícitamente definido;
- [x] frameskip efectivo se aplica una única vez;
- [x] train y eval comparten el mismo contrato perceptual;
- [x] action space continúa siendo `Discrete(18)`;
- [x] rewards permanecen sin transformación;
- [x] seeds están implementadas explícitamente;
- [x] smoke tests focalizados pasan;
- [x] validación local barata fue ejecutada cuando fue viable;
- [x] existe evidencia versionada de implementación;
- [x] no se modificó ni reutilizó `experimento_0_battlezone.ipynb`;
- [x] no existen cambios/imports desde `2_Assault/`;
- [x] no se adelantó selección ni implementación del algoritmo de HU004/HU005;
- [x] todas las autovalidaciones aplicables están en `PASS` o cualquier excepción aprobada queda explícitamente documentada;
- [x] el PR fue focalizado, revisado y mergeado a `main`.

---

## 12. Evidencias esperadas

La implementación deberá conservar, como mínimo:

1. archivo de configuración de entorno/preprocessing;
2. código de fábrica del entorno;
3. tests focalizados;
4. notebook nuevo HU003 con outputs útiles de validación;
5. evidencia visual original vs preprocessing seleccionado;
6. tabla con contrato raw y contrato final;
7. tabla de autovalidaciones AV01–AV17;
8. documento `hu003_evidencia_implementacion.md`;
9. resultado de tests locales;
10. diff del PR demostrando alcance limitado a BattleZone/HU003.

---

## 13. Riesgos y mitigaciones

### R01 — Resize destruye información del radar

**Mitigación:** comparar visualmente candidatos y no aprobar transformaciones que vuelvan ilegible la información estratégica.

### R02 — Grayscale elimina señales útiles

**Mitigación:** comparar RGB/grayscale antes de seleccionar; conservar RGB si la reducción de canales implica pérdida relevante.

### R03 — Doble frameskip

**Mitigación:** una única fábrica, configuración explícita y AV09 bloqueante.

### R04 — Train/eval divergentes

**Mitigación:** compartir la misma fábrica y validar paridad de contrato mediante tests.

### R05 — Notebook concentra lógica

**Mitigación:** notebook solo orquesta; lógica reutilizable en `src/`.

### R06 — Sobrearquitectura temprana

**Mitigación:** funciones pequeñas, composición y módulos mínimos; no crear componentes del agente antes de HU005.

### R07 — Confundir reproducibilidad con determinismo absoluto

**Mitigación:** controlar seeds y documentar sticky actions/ALE como fuentes legítimas de variabilidad.

### R08 — Pipeline optimizado para un algoritmo no seleccionado

**Mitigación:** HU003 estabiliza percepción/entorno sin introducir decisiones exclusivas de DQN, PER, DDQN o REINFORCE.

### R09 — Contaminación con código Assault

**Mitigación:** AV15 y revisión explícita del diff del PR.

### R10 — Reutilización accidental del Experimento 0

**Mitigación:** notebook nuevo desde cero, imports únicamente desde módulos BattleZone de HU003 y revisión de diff/imports en AV14.

---

## 14. Resultado esperado para HU004

HU004 deberá recibir de HU003 un contrato estable y versionado que permita comparar los algoritmos permitidos sin que la comparación quede contaminada por cambios simultáneos de preprocessing.

Como mínimo HU004 debe conocer:

- shape final de observación;
- dtype;
- canales;
- resolución;
- frame stack;
- frameskip efectivo;
- action space `Discrete(18)`;
- sticky actions;
- costo aproximado por estado;
- pipeline idéntico para train/eval;
- evidencia de que radar e información relevante fueron preservados.

Con este gate cumplido, la selección algorítmica de HU004 podrá concentrarse en sparse rewards, alta variabilidad, eficiencia muestral, estabilidad y costo computacional sin reabrir decisiones básicas del entorno.

---

## 15. Evidencia de cierre

HU003 fue auditada después de su integración y se considera cerrada porque:

- PR #21 fue mergeado a `main`;
- merge commit: `beffdacf0e3e5d8b656bfee3f11e88eacc3b7228`;
- el pipeline final es `battlezone_rgb_128_stack4_no_crop`;
- la suite focalizada reportó `9 passed`;
- AV01–AV17 están documentadas como `PASS` en `hu003_evidencia_implementacion.md`;
- el diff de la HU se mantuvo bajo `3_BattleZone/`, sin dependencia o modificación de `2_Assault/`;
- no se seleccionó ni implementó algoritmo de agente dentro de HU003.
