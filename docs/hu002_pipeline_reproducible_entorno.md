# HU002 — Pipeline reproducible del entorno Assault

## 1. Identificación

- **ID:** HU002
- **Nombre:** Pipeline reproducible del entorno
- **Estado:** Lista para implementación
- **Dependencia previa:** HU001 — Experimento 0: EDA y baseline aleatorio
- **Habilita:** HU003 — Núcleo DDQN
- **Fuentes de verdad:**
  - `docs/implementacion.md`
  - `docs/arquitectura.md`
  - `2_Assault/ficha_tecnica.md`
  - `2_Assault/experimento_0_assault.ipynb`
  - `enunciado_reto_1.txt`

---

## 2. Contexto y problema

HU001 permitió caracterizar `ALE/Assault-v5`, validar el espacio de observaciones, las 7 acciones discretas, las vidas iniciales, las variables expuestas por `info`, el `frameskip` y la estocasticidad del entorno.

Antes de implementar la CNN, Replay Buffer o la lógica DDQN, el proyecto necesita una **única forma reproducible de crear y preprocesar Assault**. Si entrenamiento, evaluación y pruebas construyen el entorno de maneras diferentes, las métricas dejan de ser comparables y pueden aparecer errores difíciles de detectar, especialmente duplicación de `frameskip`, cambios de dimensiones o manejo inconsistente de seeds.

La arquitectura define que `src/environment.py` será el único responsable de crear y configurar el entorno y que `configs/ddqn_config.yaml` será la fuente única de configuración. Esta HU materializa esa decisión sin implementar todavía entrenamiento ni lógica DDQN.

---

## 3. Historia de usuario

> **Como** equipo que desarrolla el agente DDQN para Assault, **quiero** disponer de un pipeline único, reproducible y validado para crear y preprocesar `ALE/Assault-v5`, **para** asegurar que entrenamiento, evaluación y pruebas utilicen exactamente la misma representación del entorno y que los experimentos sean comparables.

---

## 4. Objetivo verificable

Al finalizar la HU debe existir un pipeline reutilizable que, a partir de una configuración explícita y una seed conocida:

1. cree correctamente `ALE/Assault-v5`;
2. aplique el preprocesamiento Atari definido por la arquitectura;
3. produzca observaciones con dimensiones y tipo de dato esperados;
4. mantenga un `frameskip` efectivo de 4 sin duplicación;
5. permita crear entornos diferenciados para entrenamiento y evaluación sin duplicar lógica;
6. exponga información básica de hardware y ejecución útil para Google Colab;
7. pueda ejecutar una secuencia corta de interacción sin errores.

---

## 5. Alcance

### 5.1 Configuración central

Crear `2_Assault/configs/ddqn_config.yaml` como fuente única de configuración inicial del proyecto.

Debe incluir, como mínimo:

```yaml
environment:
  id: ALE/Assault-v5
  obs_type: rgb
  frame_skip: 4
  repeat_action_probability: 0.25
  full_action_space: false

preprocessing:
  grayscale: true
  resize_height: 84
  resize_width: 84
  frame_stack: 4

reproducibility:
  seed: 42

evaluation:
  episodes: 10
```

La HU puede agregar parámetros técnicos estrictamente necesarios para construir el entorno, pero no debe incorporar aún hiperparámetros propios del aprendizaje DDQN salvo que se documenten como placeholders para historias posteriores.

### 5.2 Módulo del entorno

Crear `2_Assault/src/environment.py`.

Debe centralizar:

- registro de ALE cuando sea requerido por la versión instalada;
- creación del entorno base;
- aplicación del pipeline de preprocesamiento;
- seeds;
- validación del espacio de acciones;
- distinción entre modo entrenamiento y modo evaluación;
- exposición de metadatos útiles del entorno.

No debe contener lógica de red neuronal, Replay Buffer, optimización, TensorBoard, MLflow ni checkpoints.

### 5.3 Preprocesamiento Atari

El pipeline objetivo será:

```text
RGB 210×160×3
    ↓
grayscale
    ↓
resize 84×84
    ↓
frame stack de 4 observaciones
    ↓
estado temporal compacto
```

La normalización de píxeles no deberá introducirse de forma irreversible en el entorno si es más conveniente mantener `uint8` para memoria y normalizar posteriormente en la red. La decisión implementada debe quedar documentada en el código y en la autovalidación.

### 5.4 Manejo de `frameskip`

El `frameskip` efectivo debe ser **4 una sola vez**.

La implementación debe revisar cómo interactúan el entorno base y los wrappers seleccionados. Si un wrapper realiza el salto de frames, el entorno base debe configurarse para evitar aplicar un segundo `frameskip`.

Esta condición es obligatoria y debe quedar cubierta por una autovalidación explícita.

### 5.5 Reproducibilidad

El pipeline debe permitir proporcionar una seed y aplicarla, como mínimo, a:

- `env.reset(seed=...)`;
- espacio de acciones cuando aplique;
- NumPy u otras fuentes de aleatoriedad utilizadas dentro del módulo, si existieran.

No se debe prometer determinismo absoluto del juego porque `repeat_action_probability=0.25` introduce estocasticidad deliberada. La meta es **reproducibilidad controlada de la configuración**, no eliminar la estocasticidad de Assault.

### 5.6 Hardware y entorno de ejecución

Crear en `2_Assault/src/utils.py` únicamente las utilidades transversales necesarias en esta HU para reportar:

- versión de Python;
- versión de Gymnasium;
- versión de ALE-Py;
- CPU disponible;
- RAM disponible;
- disponibilidad de GPU;
- nombre de GPU y VRAM cuando puedan consultarse.

La detección de hardware no debe hacer que el entorno dependa de GPU: ALE debe poder crearse y ejecutarse aunque el entrenamiento posterior use GPU.

### 5.7 Notebook principal

Crear el esqueleto de `2_Assault/assault_ddqn.ipynb` únicamente hasta el punto necesario para validar HU002.

Debe contener secciones simples para:

1. instalación/importación de dependencias;
2. carga de configuración;
3. inspección de hardware;
4. creación del entorno mediante `src/environment.py`;
5. impresión de observación procesada y espacios;
6. ejecución de una interacción corta de validación.

El notebook debe consumir la lógica de `src/` y no duplicarla.

### 5.8 Pruebas

Crear o ampliar `2_Assault/tests/test_smoke.py` con pruebas de bajo costo para HU002.

Las pruebas deben validar, como mínimo:

- creación del entorno;
- espacio de acciones `Discrete(7)`;
- observación preprocesada con dimensiones esperadas;
- ejecución de `reset()` y varios `step()` sin errores;
- aceptación de seed;
- separación de construcción entre modo entrenamiento y evaluación.

---

## 6. Fuera de alcance

HU002 **no** debe implementar:

- CNN;
- Online Network;
- Target Network;
- Replay Buffer;
- epsilon-greedy;
- cálculo DDQN;
- ciclo de entrenamiento;
- optimizer;
- checkpoints;
- TensorBoard;
- MLflow;
- entrenamiento real;
- evaluación formal de 10 episodios;
- optimización de hiperparámetros;
- video final.

Estos elementos corresponden a historias posteriores según `docs/implementacion.md`.

---

## 7. Decisiones y restricciones técnicas

### 7.1 Fuente única de creación del entorno

Ningún notebook o módulo futuro debe construir `ALE/Assault-v5` directamente si puede consumir la función definida en `src/environment.py`.

### 7.2 Separación de responsabilidades

- `environment.py`: entorno y wrappers.
- `utils.py`: utilidades transversales pequeñas.
- `ddqn_config.yaml`: configuración.
- `assault_ddqn.ipynb`: orquestación y evidencia.

No crear clases o abstracciones si funciones pequeñas resuelven adecuadamente el problema.

### 7.3 SOLID y DRY

Aplicar principalmente:

- **SRP:** cada módulo tiene una responsabilidad clara.
- **DRY:** entrenamiento y evaluación deberán reutilizar la misma fábrica de entorno.
- **Dependency direction:** módulos posteriores dependen del pipeline del entorno, no al contrario.

### 7.4 Compatibilidad con Colab

La implementación debe ejecutar en Google Colab desde un runtime limpio instalando únicamente las dependencias necesarias.

No se deben introducir servicios externos, contenedores o infraestructura adicional en esta HU.

### 7.5 Documentación de código

Las funciones públicas reutilizables deben incluir docstrings estilo Google con:

- propósito;
- argumentos;
- retorno;
- excepciones relevantes cuando apliquen.

Los comentarios deben explicar decisiones no obvias, no repetir literalmente el código.

---

## 8. Plan de implementación / tareas

### T01 — Crear configuración central

**Archivo:** `2_Assault/configs/ddqn_config.yaml`

**Cambio:** definir entorno, preprocessing, seed y parámetros mínimos de evaluación.

**Resultado esperado:** toda la configuración necesaria para crear Assault queda centralizada.

---

### T02 — Implementar fábrica del entorno

**Archivo:** `2_Assault/src/environment.py`

**Cambio:** implementar funciones pequeñas para construir entorno base, aplicar preprocessing y validar configuración.

**Resultado esperado:** entrenamiento y evaluación podrán solicitar un entorno sin conocer detalles internos de Gymnasium/ALE.

**Depende de:** T01.

---

### T03 — Implementar preprocesamiento

**Archivo:** `2_Assault/src/environment.py`

**Cambio:** aplicar grayscale, resize 84×84 y frame stack de 4 según arquitectura.

**Resultado esperado:** estado procesado consistente y apto para futura CNN.

**Depende de:** T02.

---

### T04 — Validar `frameskip`

**Archivo:** `2_Assault/src/environment.py` y pruebas.

**Cambio:** asegurar que el salto efectivo de frames sea 4 una única vez.

**Resultado esperado:** no existe doble salto causado por combinación de configuración base y wrappers.

**Depende de:** T02/T03.

---

### T05 — Añadir utilidades de reproducibilidad y hardware

**Archivo:** `2_Assault/src/utils.py`

**Cambio:** implementar seed helper y detección mínima de hardware/versiones.

**Resultado esperado:** el notebook puede registrar condiciones de ejecución sin duplicar código.

---

### T06 — Crear esqueleto del notebook DDQN

**Archivo:** `2_Assault/assault_ddqn.ipynb`

**Cambio:** agregar únicamente las secciones necesarias para validar HU002 consumiendo configuración y módulos.

**Resultado esperado:** el notebook demuestra que el pipeline puede inicializarse desde Colab.

**Depende de:** T01–T05.

---

### T07 — Crear smoke tests de entorno

**Archivo:** `2_Assault/tests/test_smoke.py`

**Cambio:** pruebas de creación, formas, acciones, seed e interacción corta.

**Resultado esperado:** fallos básicos del pipeline se detectan sin iniciar entrenamiento.

**Depende de:** T02–T05.

---

### T08 — Ejecutar autovalidaciones

Ejecutar pruebas en un entorno limpio compatible con Colab y registrar evidencia.

**Resultado esperado:** todos los criterios de aceptación de HU002 quedan demostrados.

---

## 9. Criterios de aceptación

### CA01 — Configuración única

**Dado** el proyecto Assault,  
**cuando** se revisa la configuración del entorno,  
**entonces** environment ID, seed, preprocessing, `frameskip`, `repeat_action_probability` y frame stack provienen de `2_Assault/configs/ddqn_config.yaml` y no están duplicados innecesariamente en otros módulos.

### CA02 — Creación reproducible

**Dada** una seed explícita,  
**cuando** se crea y reinicia el entorno mediante `src/environment.py`,  
**entonces** la configuración utilizada queda controlada y reportada sin modificar la estocasticidad intencional de Assault.

### CA03 — Espacio de acciones

**Dado** el entorno procesado,  
**cuando** se consulta el espacio de acciones,  
**entonces** sigue siendo `Discrete(7)` y conserva las acciones mínimas de Assault.

### CA04 — Observación procesada

**Dado** un `reset()` del entorno,  
**cuando** se obtiene la observación procesada,  
**entonces** corresponde a 4 frames apilados de 84×84 en escala de grises, usando un tipo de dato consistente con la decisión de memoria documentada.

### CA05 — Frameskip único

**Dado** `frameskip=4` como configuración objetivo,  
**cuando** se inspecciona y prueba el pipeline,  
**entonces** el salto efectivo es 4 y no existe aplicación duplicada entre entorno base y wrappers.

### CA06 — Interacción funcional

**Dado** un entorno recién creado,  
**cuando** se ejecutan `reset()` y varios `step()` con acciones válidas,  
**entonces** no ocurren errores de dimensiones, tipos, wrappers o acciones.

### CA07 — Entrenamiento y evaluación reutilizan fábrica

**Dado** que existen modos de entrenamiento y evaluación,  
**cuando** se construye cualquiera de los dos,  
**entonces** ambos utilizan la misma fábrica de `environment.py` y solo difieren en parámetros explícitamente justificados.

### CA08 — Hardware observable

**Dado** el runtime actual,  
**cuando** se ejecutan las utilidades de hardware,  
**entonces** se reportan CPU, RAM, versiones y GPU/VRAM cuando estén disponibles, sin fallar si no existe GPU.

### CA09 — Notebook como orquestador

**Dado** `assault_ddqn.ipynb`,  
**cuando** se revisa su implementación,  
**entonces** consume `src/environment.py`, `src/utils.py` y `ddqn_config.yaml` sin duplicar la lógica del pipeline.

### CA10 — Simplicidad arquitectónica

**Dado** el alcance de HU002,  
**cuando** se revisa el PR,  
**entonces** no contiene componentes DDQN, entrenamiento, Replay Buffer, tracking avanzado ni abstracciones innecesarias.

---

## 10. Autovalidaciones obligatorias

### AV01 — Importación limpia

**Procedimiento:** importar `environment.py` y `utils.py` desde Python/Colab limpio después de instalar dependencias.

**Resultado esperado:** importación sin excepciones.

**Criterio de éxito:** PASS.

### AV02 — Validación de observación

**Procedimiento:** crear entorno, ejecutar `reset()` e imprimir forma y dtype.

**Resultado esperado:** 4 frames de 84×84 y dtype consistente con la implementación documentada.

**Criterio de éxito:** PASS si coincide exactamente con lo definido por el pipeline.

### AV03 — Validación del action space

**Procedimiento:** consultar `env.action_space` y nombres de acciones cuando estén disponibles.

**Resultado esperado:** `Discrete(7)` y acciones mínimas esperadas.

**Criterio de éxito:** PASS.

### AV04 — Interacción corta

**Procedimiento:** ejecutar al menos 100 steps con acciones aleatorias o hasta terminación del episodio.

**Resultado esperado:** ausencia de errores y observaciones siempre compatibles con el contrato del pipeline.

**Criterio de éxito:** PASS.

### AV05 — Verificación de frameskip

**Procedimiento:** instrumentar o inspeccionar los contadores `episode_frame_number`/`frame_number` reportados por ALE durante varios steps y contrastarlos con la configuración de wrappers.

**Resultado esperado:** evidencia de que cada decisión del agente corresponde al salto efectivo esperado de 4 frames, sin duplicación accidental.

**Criterio de éxito:** PASS.

### AV06 — Seed/configuración

**Procedimiento:** crear dos entornos independientes con la misma configuración y seed, registrar configuración y primeras observaciones/metadatos disponibles.

**Resultado esperado:** configuración idéntica y comportamiento compatible con la reproducibilidad posible bajo la estocasticidad documentada.

**Criterio de éxito:** PASS si no existen diferencias de configuración no explicadas.

### AV07 — Modo train/eval

**Procedimiento:** crear un entorno en modo entrenamiento y otro en modo evaluación.

**Resultado esperado:** ambos comparten el mismo preprocessing y contrato de observación/acciones.

**Criterio de éxito:** PASS.

### AV08 — Hardware

**Procedimiento:** ejecutar detección de hardware con y sin GPU cuando sea posible.

**Resultado esperado:** salida válida; la ausencia de GPU se reporta como condición y no como excepción fatal.

**Criterio de éxito:** PASS.

### AV09 — Notebook Colab

**Procedimiento:** abrir `assault_ddqn.ipynb` en runtime limpio de Google Colab y ejecutar todas las celdas implementadas para HU002 en orden.

**Resultado esperado:** instalación, configuración, detección de hardware, creación del entorno y prueba corta completan sin cambios manuales al código.

**Criterio de éxito:** PASS.

---

## 11. Evidencias requeridas

El PR de implementación debe incluir o referenciar evidencia suficiente de:

- salida de las pruebas automatizadas;
- forma y dtype de observación preprocesada;
- `Discrete(7)`;
- configuración efectiva de `frameskip`;
- salida de detección de hardware;
- ejecución corta sin errores;
- ejecución del notebook en Colab cuando corresponda.

Cuando una validación solo pueda completarse en Colab, puede quedar marcada como **pendiente de ejecución manual**, pero la HU no deberá considerarse terminada hasta obtener dicha evidencia.

---

## 12. Definition of Done / criterios de finalización

HU002 se considera terminada únicamente cuando:

- [ ] existe `2_Assault/configs/ddqn_config.yaml`;
- [ ] existe `2_Assault/src/environment.py`;
- [ ] existe o se amplía `2_Assault/src/utils.py` con utilidades mínimas de esta HU;
- [ ] existe el esqueleto de `2_Assault/assault_ddqn.ipynb` para validar el pipeline;
- [ ] existe `2_Assault/tests/test_smoke.py` con pruebas del entorno;
- [ ] `ALE/Assault-v5` se crea únicamente mediante la fábrica definida;
- [ ] el espacio de acciones sigue siendo `Discrete(7)`;
- [ ] el preprocessing produce 4 frames de 84×84 en escala de grises;
- [ ] el `frameskip` efectivo es 4 una sola vez;
- [ ] seeds y configuración quedan explícitos;
- [ ] entrenamiento y evaluación pueden crear entornos mediante la misma fábrica;
- [ ] la detección de hardware funciona con y sin GPU;
- [ ] las funciones públicas reutilizables tienen docstrings estilo Google;
- [ ] las autovalidaciones AV01–AV09 están ejecutadas y aprobadas;
- [ ] el notebook ejecuta la parte correspondiente a HU002 en Google Colab desde un runtime limpio;
- [ ] no se implementó lógica fuera del alcance de HU002;
- [ ] el PR es acotado, revisable y consistente con `docs/arquitectura.md`.

---

## 13. Resultado esperado y gate para HU003

Al cerrar HU002 debe existir un contrato estable del estado que recibirá la futura CNN:

```text
Assault RGB
   ↓
pipeline único y reproducible
   ↓
grayscale + resize 84×84 + stack 4
   ↓
estado consistente para DDQN
```

**Gate:** HU003 no debe comenzar si las dimensiones finales del estado, el espacio de acciones o el manejo de `frameskip` siguen ambiguos o si el pipeline no ejecuta correctamente en Google Colab.

Una vez aprobado HU002, HU003 podrá implementar la CNN, Online Network, Target Network y Replay Buffer contra un contrato de entorno estable y probado.
