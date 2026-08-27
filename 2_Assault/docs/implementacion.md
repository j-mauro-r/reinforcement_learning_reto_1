# Plan de implementación — Assault con DDQN

## 1. Objetivo

Definir el **orden obligatorio de implementación** para desarrollar, entrenar, evaluar y entregar el agente DDQN de `ALE/Assault-v5` siguiendo la arquitectura del proyecto, la ficha técnica y una filosofía de MLOps ligera.

Este documento funciona como mapa maestro de ejecución. Cada HU debe implementarse únicamente cuando las dependencias y validaciones de las HUs anteriores estén satisfechas, salvo que exista una decisión técnica documentada que justifique lo contrario.

Fuentes de verdad relacionadas:

- `docs/arquitectura.md`
- `2_Assault/ficha_tecnica.md`
- `enunciado_reto_1.txt`

---

## 2. Principios del plan

1. Validar primero lo barato antes de consumir GPU en entrenamientos largos.
2. Separar entorno, agente, entrenamiento, evaluación y observabilidad.
3. Mantener el notebook como orquestador y reporte, no como contenedor de toda la lógica.
4. Aplicar SOLID y DRY sin crear abstracciones innecesarias.
5. Toda ejecución relevante debe ser reproducible y trazable.
6. Los checkpoints deben permitir continuar entrenamiento entre sesiones de Google Colab.
7. TensorBoard se utilizará para observar el entrenamiento y MLflow para comparar experimentos.
8. La evaluación final debe estar separada del entrenamiento y ejecutarse sobre al menos 10 episodios independientes.
9. El baseline aleatorio y la recompensa promedio definidos en la ficha técnica serán la referencia principal de evaluación.
10. Ninguna HU se considera terminada únicamente porque el código exista: debe superar sus autovalidaciones y producir evidencia verificable.

---

## 3. Mapa de HUs en orden de implementación

```text
HU001  EDA + baseline aleatorio                      [COMPLETADA]
  ↓
HU002  Pipeline reproducible del entorno              [IMPLEMENTADA — VALIDACIÓN LOCAL COMPLETADA — AV09 COLAB PENDIENTE]
  ↓
HU003  Núcleo DDQN
  ↓
HU004  Ciclo de entrenamiento
  ↓
HU005  Checkpoints + reanudación + idempotencia
  ↓
HU006  Observabilidad con TensorBoard
  ↓
HU007  Smoke test end-to-end
  ↓
HU008  MLflow y trazabilidad de experimentos
  ↓
HU009  Entrenamiento DDQN completo
  ↓
HU010  Optimización controlada de hiperparámetros
  ↓
HU011  Evaluación formal contra baseline
  ↓
HU012  Evidencias y entrega final
```

La secuencia es deliberada: primero se construye y valida el sistema; después se consume cómputo en entrenamientos largos.

---

## 4. HUs

### HU001 — Experimento 0: EDA y baseline aleatorio

**Estado:** completada.

**Propósito:** caracterizar empíricamente Assault, validar observaciones, acciones, vidas, variables de `info`, comportamiento temporal y construir el baseline aleatorio que servirá como referencia de desempeño.

**Entregables principales:**

- `2_Assault/experimento_0_assault.ipynb`
- actualización de `2_Assault/ficha_tecnica.md`

**Habilita:** HU002.

---

### HU002 — Pipeline reproducible del entorno

**Estado:** implementada — validación local completada — AV09 Colab pendiente.

**Propósito:** construir la única fábrica/configuración de `ALE/Assault-v5` que será utilizada por entrenamiento y evaluación.

Debe implementar:

- configuración central en `configs/ddqn_config.yaml`;
- creación reproducible del entorno;
- seeds;
- preprocessing Atari;
- grayscale;
- resize objetivo;
- frame stacking;
- manejo correcto de `frameskip` sin duplicarlo;
- distinción entre entorno de entrenamiento y evaluación;
- detección básica de hardware de Colab.

**Resultado esperado:** para una misma configuración y seed, el pipeline crea observaciones con dimensiones y tipos esperados y puede ejecutar episodios sin errores.

**Entregables principales:**

- `2_Assault/configs/ddqn_config.yaml`
- `2_Assault/src/environment.py`
- `2_Assault/src/utils.py`
- `2_Assault/tests/test_smoke.py`
- `2_Assault/assault_ddqn.ipynb`
- `2_Assault/requirements.txt`

**Evidencia de autovalidación HU002:**

- Rama validada localmente: `feature/hu002-pipeline-reproducible-entorno`, PR #3.
- Entorno virtual limpio en Windows: `.venv` creado con `python -m venv .venv`.
- Instalación desde cero validada con `.venv\Scripts\python -m pip install -r 2_Assault\requirements.txt`.
- `python -m pytest 2_Assault\tests -q` -> `6 passed`.
- Comando real ejecutado en el venv limpio: `.venv\Scripts\python -m pytest 2_Assault\tests -q` -> `6 passed`.
- Observación procesada validada: shape `(4, 84, 84)`, dtype `uint8`.
- Espacio de acciones validado: `Discrete(7)` con `NOOP`, `FIRE`, `UP`, `RIGHT`, `LEFT`, `RIGHTFIRE`, `LEFTFIRE`.
- `frameskip` efectivo validado con contadores ALE: tras 100 `step()`, `episode_frame_number=400`, consistente con 4 frames por decisión y sin duplicación.
- Train/eval crean entornos mediante la misma fábrica y comparten contrato de observación/acciones.
- Reproducibilidad local validada: dos entornos independientes con seed `42` producen la misma observación inicial procesada y los mismos `info["seeds"]`.
- Detección de hardware ejecutada localmente sin requerir GPU:
  - Python `3.8.10`;
  - Windows `10.0.19044`;
  - Gymnasium `1.1.1`;
  - ALE-Py `0.10.1`;
  - CPU `AMD64 Family 23 Model 17 Stepping 0, AuthenticAMD`;
  - 8 CPUs lógicas, 4 físicas;
  - RAM total `6.9 GB`, disponible observada `0.95 GB`;
  - GPU no disponible en esta Lenovo (`gpu_available=false`).
- `2_Assault/assault_ddqn.ipynb` ejecutado localmente de principio a fin mediante ejecución automatizada equivalente en el mismo venv. Celdas de código ejecutadas: `[2, 4, 6, 8, 10, 11]`. Resultado: `HU002 validations passed`.
- AV09 queda solo como **prevalidada localmente**. La ejecución real en Google Colab sigue pendiente y es gate final antes de declarar HU002 como completamente cerrada.
- Correcciones realizadas en esta iteración: documentación de estado/evidencia; no se requirieron cambios de código ni notebook para que las validaciones locales pasaran.

**Habilita:** HU003.

---

### HU003 — Núcleo DDQN

**Propósito:** implementar los componentes propios del algoritmo seleccionado sin incorporar aún el ciclo completo de entrenamiento.

Debe implementar:

- CNN Q-Network;
- Online Network;
- Target Network;
- Replay Buffer uniforme;
- política epsilon-greedy;
- cálculo del target DDQN;
- actualización de la Online Network;
- sincronización de la Target Network;
- optimizer;
- interfaces básicas de `save` y `load` del agente.

**Restricción:** no implementar Prioritized Experience Replay, ya que el algoritmo seleccionado es DDQN con Experience Replay uniforme.

**Resultado esperado:** los componentes reciben batches sintéticos/reales con las dimensiones esperadas, producen Q-values para las 7 acciones y ejecutan al menos un paso de optimización sin errores.

**Habilita:** HU004.

---

### HU004 — Ciclo de entrenamiento

**Propósito:** integrar entorno, agente y Replay Buffer en un ciclo de entrenamiento controlado por timesteps.

Debe implementar:

- `reset` y `step` del entorno;
- selección epsilon-greedy;
- almacenamiento de transiciones;
- `learning_starts`;
- muestreo por batches;
- actualización DDQN;
- decay de epsilon;
- sincronización periódica de Target Network;
- registro de métricas básicas en memoria/log;
- control por `global_step`.

**Resultado esperado:** el sistema puede entrenar durante un número pequeño de timesteps y modificar los pesos de la Online Network de forma verificable.

**Habilita:** HU005.

---

### HU005 — Checkpoints, reanudación e idempotencia

**Propósito:** asegurar continuidad entre sesiones de Google Colab y evitar pérdida de progreso.

El checkpoint debe guardar como mínimo:

- Online Network;
- Target Network;
- optimizer;
- timestep global;
- estado/valor de epsilon o información suficiente para reconstruirlo;
- configuración del experimento;
- métricas mínimas de continuidad;
- Replay Buffer cuando se utilice modo de resume completo.

Debe soportar explícitamente:

1. entrenamiento nuevo;
2. resume completo;
3. resume liviano cuando el Replay Buffer no pueda persistirse.

**Resultado esperado:** entrenar → guardar → reiniciar proceso → cargar → continuar desde el timestep correcto sin reiniciar silenciosamente el entrenamiento.

**Habilita:** HU006.

---

### HU006 — Observabilidad con TensorBoard

**Propósito:** hacer observable el proceso de aprendizaje durante entrenamiento.

Registrar como mínimo:

- recompensa por episodio;
- recompensa media móvil;
- longitud del episodio;
- loss;
- epsilon;
- Q-value medio o equivalente útil;
- timestep global;
- learning rate si cambia.

**Resultado esperado:** una corrida corta genera logs válidos que TensorBoard puede visualizar y que permiten detectar si el agente está aprendiendo, divergiendo o dejó de explorar.

**Habilita:** HU007.

---

### HU007 — Smoke test end-to-end

**Propósito:** validar todo el pipeline antes de gastar recursos en un entrenamiento largo.

Debe ejecutar una corrida corta con GPU y verificar conjuntamente:

- creación del entorno;
- preprocessing;
- inferencia de la red;
- Replay Buffer;
- aprendizaje;
- actualización de Target Network;
- TensorBoard;
- checkpoint;
- restauración del checkpoint;
- continuidad del entrenamiento;
- evaluación corta del modelo resultante.

**Resultado esperado:** el pipeline completo funciona sin errores funcionales ni problemas evidentes de dimensiones, dispositivo, memoria o persistencia.

**Gate:** no iniciar HU009 si HU007 no está aprobada.

**Habilita:** HU008.

---

### HU008 — MLflow y trazabilidad de experimentos

**Propósito:** registrar de forma comparable las ejecuciones que sí importan para tomar decisiones.

Cada run relevante debe registrar como mínimo:

- algoritmo;
- hiperparámetros;
- configuración del entorno;
- preprocessing;
- seed;
- versiones principales;
- hardware;
- commit Git;
- timestep inicial y final;
- tiempo de entrenamiento;
- métricas de evaluación;
- referencia al checkpoint/modelo.

**Resultado esperado:** dos corridas pueden compararse en MLflow y es posible identificar exactamente qué código y configuración produjo cada resultado.

**Habilita:** HU009.

---

### HU009 — Entrenamiento DDQN completo

**Propósito:** ejecutar el primer entrenamiento largo del agente usando la arquitectura validada.

Debe:

- usar GPU de Colab;
- entrenar por timesteps;
- persistir checkpoints fuera del almacenamiento efímero cuando corresponda;
- permitir varias sesiones;
- conservar logs de TensorBoard;
- registrar el experimento en MLflow;
- guardar modelos candidatos;
- registrar tiempo acumulado de entrenamiento.

**Resultado esperado:** producir al menos un modelo DDQN entrenado y evaluable, con trazabilidad completa y evidencia de evolución del aprendizaje.

**Habilita:** HU010.

---

### HU010 — Optimización controlada de hiperparámetros

**Propósito:** mejorar el desempeño sin realizar una búsqueda exhaustiva costosa.

Solo deben modificarse parámetros con una hipótesis clara, por ejemplo:

- learning rate;
- gamma;
- batch size;
- tamaño de Replay Buffer;
- learning starts;
- epsilon decay;
- frecuencia de aprendizaje;
- frecuencia de sincronización Target Network.

Cada variante debe compararse con el experimento anterior usando MLflow y un protocolo de evaluación consistente.

**Resultado esperado:** seleccionar justificadamente el mejor modelo/configuración candidata para evaluación formal.

**Habilita:** HU011.

---

### HU011 — Evaluación formal contra baseline

**Propósito:** ejecutar la medición oficial del desempeño del agente.

Debe:

- cargar el modelo seleccionado;
- separar completamente evaluación de entrenamiento;
- ejecutar al menos 10 episodios independientes;
- utilizar recompensa real del entorno;
- desactivar exploración o utilizar epsilon de evaluación explícitamente documentado;
- calcular media, mediana, desviación estándar, mínimo y máximo;
- registrar duración/vidas cuando aporten al análisis;
- comparar contra el baseline aleatorio de la ficha técnica.

**Métrica principal:** recompensa promedio sobre al menos 10 episodios independientes.

**Criterio interno mínimo:** recompensa promedio del agente superior al baseline aleatorio bajo un protocolo comparable.

**Resultado esperado:** evidencia cuantitativa de que el agente aprendió un comportamiento superior a la política aleatoria.

**Habilita:** HU012.

---

### HU012 — Evidencias y entrega final

**Propósito:** consolidar todos los artefactos requeridos por el reto académico.

Debe producir/verificar:

- `assault_ddqn.ipynb` ejecutable en Google Colab;
- instalación explícita de dependencias;
- modelo entrenado correspondiente a la ejecución documentada;
- video del entrenamiento/comportamiento aprendido;
- hiperparámetros;
- versiones de librerías;
- hardware utilizado;
- tiempo de entrenamiento;
- gráficas de TensorBoard;
- evaluación sobre al menos 10 episodios;
- comparación contra baseline;
- análisis del comportamiento aprendido;
- conclusión técnica.

**Resultado esperado:** entrega reproducible, consistente y trazable entre notebook, modelo, métricas, video y código Git.

---

## 5. Reglas de transición entre HUs

Una HU posterior no debe utilizarse para ocultar una validación fallida de una HU anterior.

Antes de avanzar:

1. todos los criterios de aceptación de la HU deben estar satisfechos;
2. todas las autovalidaciones obligatorias deben ejecutarse correctamente;
3. la evidencia debe quedar disponible en el PR, notebook, logs o artefactos según corresponda;
4. los criterios de finalización deben estar completos;
5. cualquier desviación debe quedar documentada explícitamente;
6. el PR debe ser revisable y limitarse al alcance de la HU.

Para cambios de implementación se debe seguir el flujo:

```text
main
  ↓
feature/HU-xxx
  ↓
implementación
  ↓
autovalidaciones
  ↓
Pull Request
  ↓
revisión
  ↓
merge a main
```

---

## 6. Estándar obligatorio para construir cada HU

Cada nueva HU deberá redactarse como un **Deep Work Plan (DWP) ejecutable**, de forma que otro desarrollador o agente pueda implementarla sin tener que reinterpretar el objetivo.

Como mínimo debe contener las siguientes secciones.

### 6.1 Identificación

- ID y nombre de la HU.
- Estado.
- Dependencias previas.
- Archivos/documentos fuente de verdad.

### 6.2 Contexto y problema

Explicar:

- qué problema existe;
- por qué debe resolverse ahora;
- qué decisión o capacidad habilita la HU;
- qué información de `arquitectura.md`, `ficha_tecnica.md` o del enunciado condiciona la solución.

### 6.3 Historia de usuario

Formato recomendado:

> **Como** [actor], **quiero** [capacidad], **para** [resultado/valor].

### 6.4 Objetivo verificable

Definir el resultado técnico concreto que debe existir al finalizar. Debe ser observable y verificable, no una intención genérica.

### 6.5 Alcance

Listar explícitamente:

- componentes que deben crearse/modificarse;
- comportamientos requeridos;
- integraciones necesarias;
- datos/configuración involucrados.

### 6.6 Fuera de alcance

Indicar lo que **no** debe implementarse en la HU para evitar sobreingeniería, scope creep y duplicación con historias posteriores.

### 6.7 Decisiones y restricciones técnicas

Documentar únicamente las decisiones necesarias para implementar la historia, por ejemplo:

- interfaces esperadas;
- módulos responsables;
- reglas de arquitectura;
- compatibilidad con Colab/GPU;
- idempotencia;
- persistencia;
- principios SOLID/DRY aplicables;
- restricciones del algoritmo DDQN.

La HU no debe contradecir `docs/arquitectura.md`. Si requiere una excepción, esta debe justificarse explícitamente.

### 6.8 Plan de implementación / tareas

Dividir la implementación en tareas pequeñas y ordenadas.

Cada tarea debe indicar:

- qué cambia;
- dónde cambia;
- resultado esperado;
- dependencia con tareas anteriores cuando exista.

El plan debe permitir desarrollar la HU incrementalmente y revisar el PR con facilidad.

### 6.9 Criterios de aceptación

Los criterios deben ser objetivos y verificables. Preferir formato Given/When/Then:

```text
CA01
Dado ...
Cuando ...
Entonces ...
```

Deben cubrir como mínimo:

- comportamiento funcional;
- integración con componentes relacionados;
- casos relevantes de error o borde;
- reproducibilidad cuando aplique;
- restricciones explícitas de la arquitectura.

### 6.10 Criterios de finalización / Definition of Done

Debe existir una checklist explícita.

Ejemplo:

```text
- [ ] implementación completada;
- [ ] criterios de aceptación satisfechos;
- [ ] autovalidaciones ejecutadas;
- [ ] no existen errores conocidos bloqueantes;
- [ ] documentación/configuración actualizada cuando aplica;
- [ ] evidencia de ejecución disponible;
- [ ] PR limitado al alcance de la HU.
```

Los criterios de terminado deben describir el estado que permite declarar la HU realmente cerrada.

### 6.11 Autovalidaciones obligatorias

Toda HU debe especificar **cómo comprobar automáticamente o semiautomáticamente que la implementación funciona**.

Las autovalidaciones deben ser proporcionales al cambio e incluir solo las que aporten valor. Pueden incluir:

- imports de módulos;
- tests unitarios focalizados;
- smoke tests;
- validación de shapes/dtypes;
- ejecución de algunos steps del entorno;
- forward pass de la red;
- un paso real de optimización;
- validación save/load;
- comparación de estado antes/después de un checkpoint;
- verificación de logs TensorBoard;
- validación de parámetros/métricas en MLflow;
- evaluación corta del agente;
- ejecución end-to-end de una sección del notebook.

Cada autovalidación debe definir:

1. **comando o procedimiento**;
2. **resultado esperado**;
3. **criterio de éxito/fallo**.

No es suficiente escribir "probar que funciona".

Cuando una validación solo pueda ejecutarse en Google Colab/GPU, debe quedar marcada explícitamente como **validación Colab pendiente de ejecución por el usuario** y no debe sustituirse con resultados inventados.

### 6.12 Evidencias esperadas

La HU debe definir qué evidencia demuestra su éxito, por ejemplo:

- salida de test;
- tabla de métricas;
- screenshot o gráfica de TensorBoard;
- run de MLflow;
- checkpoint restaurado;
- notebook ejecutado;
- resultados de evaluación;
- archivo/modelo generado.

### 6.13 Riesgos y consideraciones

Registrar únicamente riesgos materiales para la HU, especialmente:

- consumo de RAM/VRAM;
- duración de Colab;
- pérdida de checkpoints;
- incompatibilidades de versiones;
- errores de shapes;
- duplicación de frameskip;
- desviaciones que puedan invalidar la comparación con el baseline.

---

## 7. Regla de cierre de una HU

Una HU se considera **implementada** cuando el código existe.

Una HU se considera **terminada** únicamente cuando:

```text
Implementación
    +
Criterios de aceptación
    +
Autovalidaciones exitosas
    +
Evidencia verificable
    +
Definition of Done completa
    =
HU CERRADA
```

Si una autovalidación obligatoria depende de Google Colab y todavía no fue ejecutada, la HU debe mantenerse como **implementada pendiente de validación**, no como completada.
