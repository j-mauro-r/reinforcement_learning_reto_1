# Plan de implementación E2E — BattleZone

## 1. Objetivo

Definir el orden obligatorio de implementación para desarrollar, entrenar, evaluar y entregar un agente de Reinforcement Learning para `ALE/BattleZone-v5`, cumpliendo las restricciones académicas del Reto 1 y aplicando una filosofía de MLOps ligera, reproducible y trazable.

El proyecto debe producir un agente capaz de demostrar un comportamiento lógico aprendido y maximizar la recompensa promedio sobre al menos 10 partidas independientes.

Algoritmos permitidos por el reto:

- DQN;
- DQN + Prioritized Experience Replay;
- DDQN;
- REINFORCE.

La selección definitiva del algoritmo para BattleZone se realizará después de completar el EDA y el baseline aleatorio.

---

## 2. Relación con el agente Assault

El trabajo realizado previamente para Assault se utilizará únicamente como **base de conocimiento y referencia metodológica**.

Se reutilizarán conceptos, prácticas y aprendizajes como:

- metodología incremental por HUs;
- Deep Work Plans (DWP);
- separación entre entorno, agente, entrenamiento, evaluación y observabilidad;
- configuración centralizada;
- smoke tests antes de entrenamientos largos;
- checkpoints y reanudación;
- TensorBoard;
- MLflow;
- evaluación formal contra baseline aleatorio;
- ejecución local antes de consumir GPU cuando sea viable;
- Google Colab como entorno principal de entrenamiento GPU;
- GitHub como fuente de verdad.

### Restricción obligatoria

**No se copiará, importará ni reutilizará código del agente Assault.**

BattleZone tendrá su propia implementación, configuración, tests, notebooks, módulos, artefactos y decisiones de arquitectura.

El proyecto Assault se considera únicamente conocimiento previo del equipo.

---

## 3. Principios del plan

1. Entender primero el entorno antes de seleccionar el algoritmo.
2. Construir un baseline aleatorio antes de entrenar.
3. Validar barato antes de entrenar caro.
4. Mantener el notebook como orquestador y reporte, no como repositorio de lógica duplicada.
5. Separar responsabilidades técnicas.
6. Mantener configuración, seeds y versiones explícitas.
7. Todo entrenamiento largo debe poder reanudarse.
8. TensorBoard se utilizará para observabilidad temporal.
9. MLflow se utilizará para trazabilidad y comparación de experimentos.
10. La evaluación final estará desacoplada del entrenamiento.
11. El agente final deberá evaluarse sobre al menos 10 episodios independientes.
12. El resultado debe compararse contra una política aleatoria bajo condiciones equivalentes.
13. No se optimizarán hiperparámetros sin una hipótesis explícita.
14. El código de BattleZone será independiente del código de Assault.

---

## 4. Mapa E2E de HUs

```text
HU001  Caracterización técnica y ficha inicial de BattleZone
  ↓
HU002  Experimento 0 y baseline aleatorio
  ↓
HU003  Pipeline reproducible del entorno
  ↓
HU004  Selección formal del algoritmo
  ↓
HU005  Núcleo del agente
  ↓
HU006  Ciclo de entrenamiento
  ↓
HU007  Checkpoints, reanudación e idempotencia
  ↓
HU008  Observabilidad con TensorBoard
  ↓
HU009  Smoke test end-to-end
  ↓
HU010  MLflow y trazabilidad de experimentos
  ↓
HU011  Entrenamiento completo
  ↓
HU012  Optimización controlada de hiperparámetros
  ↓
HU013  Evaluación formal contra baseline
  ↓
HU014  Reporte técnico, evidencias y entrega final
```

---

# 5. Historias de Usuario

## HU001 — Caracterización técnica y ficha inicial de BattleZone

**Propósito:** comprender formalmente el entorno antes de diseñar el agente.

Debe documentar como mínimo:

- environment ID;
- versión de Gymnasium/ALE-Py;
- observaciones disponibles;
- shape, dtype y rango de observaciones;
- action space mínimo y completo;
- significado de las acciones;
- modes y difficulties;
- `frameskip`;
- `repeat_action_probability`;
- vidas;
- terminación y truncation;
- variables expuestas mediante `info`;
- dinámica del radar;
- enemigos;
- obstáculos;
- recompensas y scoring cuando estén documentados;
- factores que aumentan la dificultad del aprendizaje;
- preguntas que deberán resolverse empíricamente.

**Entregable principal:**

`3_BattleZone/ficha_tecnica.md`

**Gate:** no comenzar diseño algorítmico definitivo antes de completar esta caracterización.

---

## HU002 — Experimento 0 y baseline aleatorio

**Propósito:** medir empíricamente BattleZone mediante una política completamente aleatoria.

Debe ejecutar al menos 10 episodios independientes y registrar, cuando estén disponibles:

- recompensa por episodio;
- media, mediana, desviación estándar, mínimo y máximo;
- duración por episodio;
- lives;
- pérdidas de vida;
- `terminated` y `truncated`;
- densidad de recompensas positivas, cero y negativas;
- frecuencia de cada acción;
- comportamiento observado del radar;
- información adicional expuesta por ALE.

Debe incluir visualizaciones mínimas de recompensa, duración y frecuencia de acciones.

**Entregables:**

- `3_BattleZone/experimento_0_battlezone.ipynb`;
- actualización de `ficha_tecnica.md`.

**Resultado:** baseline cuantitativo para evaluar posteriormente al agente entrenado.

---

## HU003 — Pipeline reproducible del entorno

**Propósito:** crear una única forma reproducible de inicializar y preprocesar BattleZone.

Debe incluir:

- configuración centralizada;
- seeds;
- fábrica única del entorno;
- preprocessing visual;
- grayscale si se valida como apropiado;
- resize;
- frame stacking;
- tratamiento correcto de `frameskip`;
- separación train/eval;
- validación del action space;
- detección de hardware;
- smoke tests del entorno.

El preprocessing definitivo deberá determinarse a partir de HU001 y HU002, no copiarse automáticamente de Assault.

**Entregables previstos:**

- `configs/`;
- `src/environment.py`;
- `src/utils.py`;
- tests focalizados;
- esqueleto del notebook principal.

**Gate:** el contrato de estado y acciones debe quedar estable antes de HU005.

---

## HU004 — Selección formal del algoritmo

**Propósito:** elegir justificadamente uno de los algoritmos permitidos por el reto.

La comparación deberá considerar como mínimo:

- tamaño del action space;
- dimensionalidad visual;
- densidad y variabilidad de recompensas;
- costo computacional;
- estabilidad esperada;
- eficiencia muestral;
- complejidad de implementación;
- duración de los episodios;
- baseline aleatorio;
- posibilidad de completar entrenamiento dentro de las restricciones de Colab.

Algoritmos candidatos:

- DQN;
- DQN + PER;
- DDQN;
- REINFORCE.

La decisión debe quedar documentada mediante una matriz comparativa y una justificación técnica.

**Entregable:** decisión técnica versionada dentro de la documentación de BattleZone.

---

## HU005 — Núcleo del agente

**Propósito:** implementar los componentes propios del algoritmo seleccionado sin integrar todavía el ciclo completo de entrenamiento.

Para algoritmos value-based, incluir según corresponda:

- CNN/Q-Network;
- Online Network;
- Target Network cuando aplique;
- Replay Buffer;
- Prioritized Experience Replay únicamente si el algoritmo seleccionado lo exige;
- política epsilon-greedy;
- cálculo de targets;
- optimizer;
- actualización de pesos;
- interfaces básicas save/load.

Para REINFORCE, la HU deberá adaptarse a su arquitectura policy-based sin introducir componentes de DQN innecesarios.

**Resultado:** forward pass válido y al menos una actualización del agente ejecutada sobre datos controlados.

---

## HU006 — Ciclo de entrenamiento

**Propósito:** integrar entorno, agente y almacenamiento de experiencia en un flujo controlado.

Debe implementar según el algoritmo:

- `reset` y `step`;
- selección de acciones;
- exploración;
- almacenamiento de transiciones o trayectorias;
- inicio del aprendizaje;
- actualizaciones del modelo;
- actualización de Target Network cuando corresponda;
- control por timestep o episodio según el algoritmo;
- métricas básicas de entrenamiento;
- manejo explícito de `terminated` y `truncated`.

**Resultado:** entrenamiento corto funcional con modificación verificable de parámetros.

---

## HU007 — Checkpoints, reanudación e idempotencia

**Propósito:** evitar pérdida de entrenamiento ante desconexiones o reinicios de Colab.

Cada checkpoint debe guardar la información necesaria para continuar de forma consistente.

Según el algoritmo, esto puede incluir:

- redes;
- optimizer;
- timestep/episodio global;
- epsilon o estado de exploración;
- Replay Buffer cuando sea viable;
- configuración;
- métricas acumuladas;
- estado adicional requerido para reconstruir el entrenamiento.

Debe soportar explícitamente:

1. nueva corrida;
2. resume completo;
3. resume liviano cuando sea necesario.

**Resultado:** guardar → cerrar proceso → restaurar → continuar sin reinicio silencioso.

---

## HU008 — Observabilidad con TensorBoard

**Propósito:** hacer visible la evolución del aprendizaje.

Registrar como mínimo, cuando aplique:

- recompensa por episodio;
- recompensa media móvil;
- longitud del episodio;
- loss;
- exploración/epsilon;
- Q-value medio o métrica equivalente;
- timestep global;
- learning rate;
- métricas adicionales que ayuden a detectar estancamiento o inestabilidad.

**Resultado:** una corrida corta produce logs válidos y gráficas interpretables.

---

## HU009 — Smoke test end-to-end

**Propósito:** validar el sistema completo antes del entrenamiento largo.

Debe verificar conjuntamente:

- entorno;
- preprocessing;
- action space;
- inferencia;
- entrenamiento;
- almacenamiento de experiencias;
- checkpoints;
- restore;
- TensorBoard;
- uso correcto de CPU/GPU;
- evaluación corta;
- ausencia de errores de shapes, dispositivos o memoria.

Debe ejecutarse primero localmente cuando sea viable y después en Colab GPU.

**Gate:** HU011 no puede iniciar si HU009 no está aprobada.

---

## HU010 — MLflow y trazabilidad de experimentos

**Propósito:** permitir comparar y reproducir las corridas relevantes.

Cada experimento debe registrar como mínimo:

### Parámetros

- algoritmo;
- environment ID;
- preprocessing;
- hiperparámetros;
- seed;
- versiones;
- hardware;
- commit Git;
- run ID.

### Métricas

- timestep/episodio inicial y final;
- tiempo de entrenamiento;
- recompensa de evaluación;
- desviación estándar;
- mínimo y máximo;
- mejor resultado relevante.

### Artefactos

- configuración;
- resumen de evaluación;
- referencia al checkpoint/modelo;
- gráficas finales cuando aporten valor.

**Resultado:** dos experimentos pueden compararse y asociarse a código y configuración específicos.

---

## HU011 — Entrenamiento completo

**Propósito:** ejecutar el primer entrenamiento largo del agente BattleZone.

Debe:

- usar GPU de Colab;
- producir checkpoints periódicos;
- soportar múltiples sesiones;
- conservar TensorBoard;
- registrar MLflow;
- persistir artefactos importantes;
- registrar tiempo acumulado de entrenamiento;
- producir al menos un modelo candidato evaluable.

**Resultado:** agente entrenado con trazabilidad completa y evidencia observable de aprendizaje o estancamiento.

---

## HU012 — Optimización controlada de hiperparámetros

**Propósito:** mejorar el desempeño utilizando experimentos pequeños y comparables.

Cada cambio debe partir de una hipótesis.

Dependiendo del algoritmo, podrán evaluarse parámetros como:

- learning rate;
- gamma;
- batch size;
- Replay Buffer;
- alpha/beta de PER;
- learning starts;
- epsilon decay;
- frecuencia de actualización;
- Target Network sync;
- arquitectura CNN;
- duración total del entrenamiento.

Cada variante debe registrar:

- valor anterior;
- valor nuevo;
- hipótesis;
- resultado esperado;
- run MLflow;
- resultado observado.

**Resultado:** selección justificada de la mejor configuración candidata.

---

## HU013 — Evaluación formal contra baseline

**Propósito:** medir oficialmente el desempeño del modelo seleccionado.

Debe:

- cargar explícitamente el modelo final;
- utilizar el mismo pipeline de observaciones;
- ejecutar al menos 10 episodios independientes;
- usar recompensa real del entorno;
- desactivar exploración deliberada o documentar el comportamiento de evaluación;
- calcular media, mediana, desviación estándar, mínimo y máximo;
- registrar duración y vidas cuando sean útiles;
- comparar directamente contra el baseline aleatorio de HU002.

Además de las métricas, debe revisarse cualitativamente si existe comportamiento lógico aprendido.

**Criterio interno mínimo:** el agente debe mostrar evidencia cuantitativa y/o conductual consistente de aprendizaje frente a la política aleatoria.

---

## HU014 — Reporte técnico, evidencias y entrega final

**Propósito:** consolidar todo el trabajo en los artefactos requeridos por el reto.

Debe producir/verificar:

- notebook ejecutable de principio a fin en Google Colab;
- instalación explícita de dependencias;
- modelo entrenado correspondiente al experimento documentado;
- video del comportamiento aprendido;
- evidencia del proceso de entrenamiento;
- configuración final;
- hiperparámetros;
- versiones de librerías;
- hardware utilizado;
- tiempo de entrenamiento;
- gráficas de TensorBoard;
- información de MLflow;
- evaluación formal de al menos 10 episodios;
- comparación contra baseline aleatorio;
- análisis del comportamiento aprendido;
- limitaciones;
- conclusiones.

### Estructura mínima del reporte dentro del notebook

1. Descripción del problema.
2. Caracterización de BattleZone.
3. Baseline aleatorio.
4. Algoritmos considerados.
5. Justificación del algoritmo seleccionado.
6. Arquitectura del agente.
7. Pipeline de observaciones.
8. Hiperparámetros.
9. Condiciones de ejecución.
10. Hardware y versiones.
11. Estrategia de entrenamiento.
12. Checkpoints y reanudación.
13. TensorBoard y evolución del aprendizaje.
14. Experimentos y comparaciones MLflow.
15. Evaluación final ≥10 episodios.
16. Comparación con baseline.
17. Análisis del comportamiento aprendido.
18. Limitaciones y amenazas a la validez.
19. Conclusiones.
20. Instrucciones de reproducción.

**Resultado:** entrega consistente entre código, notebook, modelo, métricas, video y reporte.

---

# 6. Reglas de transición entre HUs

Una HU posterior no debe utilizarse para ocultar fallos de una HU anterior.

Antes de avanzar:

1. criterios de aceptación satisfechos;
2. autovalidaciones ejecutadas;
3. evidencia disponible;
4. Definition of Done completada;
5. desviaciones documentadas;
6. PR limitado al alcance de la HU;
7. no introducir código de Assault en BattleZone.

Flujo esperado:

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

# 7. Estándar obligatorio para crear cada HU — Metodología DWP

Cada nueva HU de BattleZone deberá redactarse como un **Deep Work Plan (DWP) ejecutable**, suficientemente preciso para que otro desarrollador o agente pueda implementarla sin reinterpretar el objetivo.

## 7.1 Identificación

Debe incluir:

- ID;
- nombre;
- estado;
- dependencia previa;
- HU que habilita;
- archivos/documentos fuente de verdad.

---

## 7.2 Contexto y problema

Explicar claramente:

- qué problema existe;
- por qué debe resolverse ahora;
- qué riesgo evita;
- qué capacidad habilita;
- qué decisiones previas condicionan la solución.

La sección debe distinguir hechos conocidos de supuestos aún pendientes de validar.

---

## 7.3 Historia de usuario

Formato recomendado:

> **Como** [actor], **quiero** [capacidad], **para** [resultado o valor].

Debe describir valor técnico o académico, no únicamente una tarea de programación.

---

## 7.4 Objetivo verificable

Definir el resultado técnico concreto que debe existir al finalizar.

Debe ser observable y comprobable.

Evitar formulaciones ambiguas como:

- “mejorar el código”;
- “hacer que funcione”;
- “optimizar el agente”.

---

## 7.5 Alcance

Listar explícitamente:

- archivos que deben crearse o modificarse;
- componentes involucrados;
- comportamientos requeridos;
- configuraciones;
- integraciones;
- datos o artefactos afectados.

---

## 7.6 Fuera de alcance

Definir qué **no** debe implementarse.

Su objetivo es prevenir:

- scope creep;
- sobreingeniería;
- implementación anticipada de HUs posteriores;
- duplicación;
- cambios accidentales en otros agentes del reto.

Toda HU de BattleZone debe recordar, cuando sea relevante, que no debe modificar archivos de Assault.

---

## 7.7 Decisiones y restricciones técnicas

Documentar las decisiones necesarias para implementar la HU, por ejemplo:

- módulos responsables;
- interfaces;
- separación de responsabilidades;
- configuración centralizada;
- compatibilidad local/Colab;
- GPU;
- idempotencia;
- persistencia;
- reproducibilidad;
- SOLID;
- DRY;
- restricciones del algoritmo seleccionado;
- restricciones académicas del enunciado.

No deben crearse abstracciones sin una necesidad concreta.

---

## 7.8 Plan de implementación / tareas

Dividir la HU en tareas pequeñas y ordenadas.

Cada tarea debe indicar:

- identificador;
- archivo o componente afectado;
- cambio esperado;
- resultado esperado;
- dependencias con tareas anteriores.

Formato sugerido:

```text
T01 — Nombre de la tarea
Archivo: ...
Cambio: ...
Resultado esperado: ...
Depende de: ...
```

---

## 7.9 Criterios de aceptación

Deben ser objetivos y verificables.

Se recomienda Given/When/Then:

```text
CA01
Dado ...
Cuando ...
Entonces ...
```

Deben cubrir según corresponda:

- comportamiento funcional;
- integración;
- errores y casos borde;
- reproducibilidad;
- shapes y dtypes;
- persistencia;
- restricciones arquitectónicas;
- compatibilidad local/Colab;
- condiciones específicas del algoritmo.

---

## 7.10 Autovalidaciones obligatorias

Cada HU debe definir cómo demostrar que funciona.

Cada autovalidación debe especificar:

1. identificador;
2. comando o procedimiento;
3. resultado esperado;
4. criterio de PASS/FAIL;
5. ambiente donde debe ejecutarse.

Ejemplos:

- imports;
- pytest;
- shapes/dtypes;
- ejecución de steps;
- forward pass;
- actualización de pesos;
- save/load;
- checkpoint/resume;
- TensorBoard;
- MLflow;
- evaluación corta;
- ejecución E2E;
- validación local;
- validación Colab GPU.

Cuando una prueba únicamente pueda ejecutarse en Colab, debe marcarse explícitamente como:

**Validación Colab pendiente de ejecución por el usuario.**

No deben inventarse resultados.

---

## 7.11 Evidencias esperadas

Definir qué evidencia demuestra el cumplimiento de la HU.

Puede incluir:

- salida de tests;
- logs;
- métricas;
- tablas;
- screenshots;
- TensorBoard;
- MLflow;
- checkpoint restaurado;
- notebook ejecutado;
- modelo generado;
- resultados de evaluación.

---

## 7.12 Definition of Done

Toda HU debe incluir una checklist explícita.

Formato base:

```text
- [ ] implementación completada;
- [ ] criterios de aceptación satisfechos;
- [ ] autovalidaciones ejecutadas;
- [ ] evidencia disponible;
- [ ] documentación/configuración actualizada;
- [ ] no existen errores conocidos bloqueantes;
- [ ] no se modificaron componentes fuera del alcance;
- [ ] PR limitado al alcance de la HU;
- [ ] validaciones Colab ejecutadas cuando apliquen.
```

Una HU no está cerrada únicamente porque exista código.

---

## 7.13 Riesgos y consideraciones

Registrar únicamente riesgos materiales, como:

- RAM/VRAM;
- duración de Colab;
- pérdida de checkpoints;
- incompatibilidad de versiones;
- bugs de shapes;
- action space incorrecto;
- `frameskip` duplicado;
- preprocessing inconsistente;
- diferencias train/eval;
- desviaciones frente al baseline;
- experimentos no comparables;
- contaminación accidental entre BattleZone y Assault.

---

## 7.14 Resultado esperado y gate

Cada HU debe finalizar indicando:

- qué capacidad nueva existe;
- qué evidencia demuestra su funcionamiento;
- qué condición debe cumplirse antes de iniciar la HU siguiente.

---

# 8. Regla de cierre de una HU

Una HU se considera **implementada** cuando el código o artefacto existe.

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

Si una autovalidación obligatoria depende de Colab/GPU y aún no fue ejecutada, la HU debe permanecer como:

**IMPLEMENTADA — PENDIENTE DE VALIDACIÓN**

---

# 9. Criterio global de éxito del proyecto BattleZone

El proyecto se considerará técnicamente completo cuando sea posible:

1. partir desde `main`;
2. ejecutar el notebook en un runtime limpio de Google Colab;
3. reproducir la configuración del entorno;
4. iniciar o reanudar el entrenamiento;
5. observar la evolución mediante TensorBoard;
6. identificar la corrida en MLflow;
7. recuperar checkpoints después de interrupciones;
8. evaluar el modelo sobre al menos 10 episodios independientes;
9. compararlo con el baseline aleatorio;
10. demostrar comportamiento aprendido razonablemente lógico;
11. generar modelo, métricas, video y reporte;
12. relacionar todos los artefactos con una configuración y commit Git;
13. mantener completa independencia de código entre BattleZone y Assault.

La prioridad no es construir una plataforma MLOps empresarial. La prioridad es producir un agente BattleZone reproducible, observable, recuperable, evaluable y defendible académicamente.