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

Se reutilizarán aprendizajes y prácticas como:

- metodología incremental por HUs;
- Deep Work Plans (DWP);
- separación entre entorno, agente, entrenamiento, evaluación y observabilidad;
- configuración centralizada;
- smoke tests antes de entrenamientos largos;
- checkpoints y reanudación;
- TensorBoard;
- evaluación formal contra baseline aleatorio;
- ejecución local antes de consumir GPU cuando sea viable;
- Google Colab como entorno principal de entrenamiento GPU;
- GitHub como fuente de verdad.

### Restricción obligatoria

**No se copiará, importará ni reutilizará código del agente Assault.**

BattleZone tendrá su propia implementación, configuración, tests, notebooks, módulos, artefactos y decisiones de arquitectura.

El proyecto Assault se considera únicamente conocimiento previo del equipo.

---

## 3. Decisiones técnicas transversales

1. Entender el entorno antes de seleccionar algoritmo.
2. Construir baseline aleatorio antes de entrenar.
3. Validar barato antes de entrenar caro.
4. Mantener el notebook como orquestador y reporte.
5. Separar responsabilidades técnicas.
6. Mantener configuración, seeds y versiones explícitas.
7. Todo entrenamiento largo debe poder reanudarse.
8. TensorBoard será la herramienta principal de observabilidad.
9. **BattleZone no utilizará MLflow.**
10. La trazabilidad se resolverá con Git/GitHub, configuración versionada, `run_id`, manifiesto de ejecución, checkpoints, TensorBoard y archivos estructurados de resultados.
11. La evaluación final estará desacoplada del entrenamiento.
12. El agente final deberá evaluarse sobre al menos 10 episodios independientes.
13. El resultado se comparará contra una política aleatoria bajo condiciones equivalentes.
14. No se optimizarán hiperparámetros sin una hipótesis explícita.
15. El código de BattleZone será independiente del código de Assault.

Los lineamientos técnicos detallados se encuentran en `3_BattleZone/docs/lineamientos.md`.

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
HU010  Trazabilidad ligera de experimentos
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

**Entregable principal:** `3_BattleZone/ficha_tecnica.md`.

**Gate:** no comenzar diseño algorítmico definitivo antes de completar esta caracterización.

---

## HU002 — Experimento 0 y baseline aleatorio

**Propósito:** medir empíricamente BattleZone mediante una política completamente aleatoria.

Debe ejecutar al menos 10 episodios independientes y registrar, cuando estén disponibles:

- recompensa por episodio;
- media, mediana, desviación estándar, mínimo y máximo;
- duración por episodio;
- vidas y pérdidas de vida;
- `terminated` y `truncated`;
- densidad de recompensas positivas, cero y negativas;
- frecuencia de cada acción;
- comportamiento observado del radar;
- información adicional expuesta por ALE.

Debe incluir visualizaciones mínimas de recompensa, duración y frecuencia de acciones.

**Entregables:**

- `3_BattleZone/experimento_0_battlezone.ipynb`;
- actualización de `3_BattleZone/ficha_tecnica.md`.

**Resultado:** baseline cuantitativo reutilizable en HU013.

---

## HU003 — Pipeline reproducible del entorno

**Propósito:** crear una única forma reproducible de inicializar y preprocesar BattleZone.

Debe incluir:

- configuración centralizada;
- seeds;
- fábrica única del entorno;
- preprocessing visual;
- grayscale si HU001/HU002 demuestran que conserva información suficiente;
- resize;
- frame stacking;
- tratamiento correcto de `frameskip`;
- separación train/eval;
- validación del action space;
- detección de hardware;
- smoke tests del entorno.

El preprocessing definitivo deberá determinarse a partir de BattleZone, no copiarse automáticamente de Assault.

**Entregables previstos:**

- `3_BattleZone/configs/`;
- `3_BattleZone/src/environment.py`;
- `3_BattleZone/src/utils.py`;
- tests focalizados;
- esqueleto del notebook principal.

**Gate:** el contrato de estado y acciones debe quedar estable antes de HU005.

---

## HU004 — Selección formal del algoritmo

**Propósito:** elegir justificadamente uno de los algoritmos permitidos por el reto.

La comparación deberá considerar:

- tamaño del action space;
- dimensionalidad visual;
- densidad y variabilidad de recompensas;
- costo computacional;
- estabilidad esperada;
- eficiencia muestral;
- complejidad de implementación;
- duración de episodios;
- baseline aleatorio;
- restricciones de Colab.

Algoritmos candidatos:

- DQN;
- DQN + PER;
- DDQN;
- REINFORCE.

**Entregable:** decisión técnica versionada con matriz comparativa y justificación.

**Gate:** HU005 debe implementar únicamente el algoritmo seleccionado.

---

## HU005 — Núcleo del agente

**Propósito:** implementar los componentes propios del algoritmo seleccionado sin integrar todavía el ciclo completo de entrenamiento.

Para algoritmos value-based, incluir según corresponda:

- CNN/Q-Network;
- Online Network;
- Target Network cuando aplique;
- Replay Buffer;
- Prioritized Experience Replay únicamente si el algoritmo seleccionado es DQN + PER;
- política epsilon-greedy;
- cálculo de targets;
- optimizer;
- actualización de pesos;
- interfaces básicas save/load.

Para REINFORCE, adaptar la HU a una policy network y trayectorias completas sin introducir componentes DQN innecesarios.

**Resultado:** forward pass válido y al menos una actualización del agente sobre datos controlados.

---

## HU006 — Ciclo de entrenamiento

**Propósito:** integrar entorno, agente y experiencia en un flujo controlado.

Debe implementar según el algoritmo:

- `reset` y `step`;
- selección de acciones;
- exploración;
- almacenamiento de transiciones o trayectorias;
- inicio del aprendizaje;
- actualizaciones del modelo;
- actualización de Target Network cuando corresponda;
- control por timestep o episodio;
- métricas básicas;
- manejo explícito de `terminated` y `truncated`.

**Resultado:** entrenamiento corto funcional con modificación verificable de parámetros.

---

## HU007 — Checkpoints, reanudación e idempotencia

**Propósito:** evitar pérdida de entrenamiento ante desconexiones o reinicios de Colab.

Cada checkpoint debe guardar el estado necesario para continuar consistentemente.

Puede incluir, según el algoritmo:

- redes;
- optimizer;
- timestep/episodio global;
- epsilon o estado de exploración;
- Replay Buffer cuando sea viable;
- configuración;
- métricas acumuladas;
- estado adicional necesario.

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
- timestep/episodio global;
- learning rate;
- métricas adicionales justificadas para detectar estancamiento o inestabilidad.

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

Debe ejecutarse primero localmente cuando sea viable y luego en Colab GPU.

**Gate:** HU011 no puede iniciar si HU009 no está aprobada.

---

## HU010 — Trazabilidad ligera de experimentos

**Propósito:** permitir reproducir y comparar corridas sin utilizar MLflow.

Cada corrida relevante deberá poseer un `run_id` único y producir un manifiesto estructurado, por ejemplo:

`3_BattleZone/results/<run_id>/run_manifest.json`

El manifiesto debe registrar como mínimo:

### Identidad

- `run_id`;
- algoritmo;
- commit Git;
- fecha/hora;
- seed.

### Configuración

- environment ID;
- preprocessing;
- hiperparámetros;
- versiones de librerías;
- hardware.

### Ejecución

- timestep/episodio inicial y final;
- nuevo entrenamiento o resume;
- checkpoint de entrada cuando aplique;
- checkpoint/modelo de salida;
- tiempo acumulado.

### Resultados

- ruta de logs TensorBoard;
- métricas de evaluación cuando existan;
- resumen de resultados estructurado.

Debe poder generarse una tabla comparativa de corridas desde estos resultados sin depender de un servicio externo.

**Resultado:** dos experimentos pueden asociarse inequívocamente a código, configuración, logs, checkpoints y métricas.

---

## HU011 — Entrenamiento completo

**Propósito:** ejecutar el primer entrenamiento largo del agente BattleZone.

Debe:

- usar GPU de Colab;
- producir checkpoints periódicos;
- soportar múltiples sesiones;
- conservar TensorBoard;
- conservar `run_manifest` y resultados estructurados;
- persistir artefactos importantes;
- registrar tiempo acumulado;
- producir al menos un modelo candidato evaluable.

**Resultado:** agente entrenado con trazabilidad completa y evidencia observable de aprendizaje o estancamiento.

---

## HU012 — Optimización controlada de hiperparámetros

**Propósito:** mejorar el desempeño utilizando experimentos pequeños y comparables.

Cada cambio debe partir de una hipótesis.

Dependiendo del algoritmo, podrán evaluarse:

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

- `run_id`;
- valor anterior;
- valor nuevo;
- hipótesis;
- resultado esperado;
- resultado observado;
- comparación contra corrida anterior.

**Resultado:** selección justificada de la mejor configuración candidata.

---

## HU013 — Evaluación formal contra baseline

**Propósito:** medir oficialmente el desempeño del modelo seleccionado.

Debe:

- cargar explícitamente el modelo final;
- utilizar el mismo pipeline de observaciones;
- ejecutar al menos 10 episodios independientes;
- usar recompensa real del entorno;
- desactivar exploración deliberada o documentar la política de evaluación;
- calcular media, mediana, desviación estándar, mínimo y máximo;
- registrar duración y vidas cuando aporten al análisis;
- comparar directamente contra el baseline aleatorio de HU002;
- producir evidencia cualitativa del comportamiento aprendido.

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
- manifiestos y tabla comparativa de experimentos;
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
14. Trazabilidad y comparación de experimentos.
15. Evaluación final ≥10 episodios.
16. Comparación con baseline.
17. Análisis del comportamiento aprendido.
18. Limitaciones y amenazas a la validez.
19. Conclusiones.

**Resultado:** entrega reproducible, consistente entre notebook, modelo, métricas, video y código.

---

# 6. Reglas de transición entre HUs

Una HU posterior no debe utilizarse para ocultar una validación fallida de una HU anterior.

Antes de avanzar:

1. todos los criterios de aceptación de la HU deben estar satisfechos;
2. las autovalidaciones obligatorias deben ejecutarse correctamente;
3. la evidencia debe quedar disponible en PR, notebook, logs o artefactos;
4. Definition of Done debe estar completa;
5. cualquier desviación debe documentarse explícitamente;
6. el PR debe limitarse al alcance de la HU.

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

# 7. Estándar obligatorio para crear cada HU según DWP

Cada HU debe redactarse como un **Deep Work Plan (DWP) ejecutable**, de forma que otro desarrollador o agente pueda implementarla sin reinterpretar el objetivo.

## 7.1 Identificación

Debe incluir:

- ID y nombre de la HU;
- estado;
- dependencia previa;
- HUs que habilita;
- archivos/documentos fuente de verdad.

## 7.2 Contexto y problema

Explicar:

- qué problema existe;
- por qué debe resolverse ahora;
- qué capacidad habilita;
- qué decisiones anteriores condicionan la solución.

## 7.3 Historia de usuario

Formato recomendado:

> **Como** [actor], **quiero** [capacidad], **para** [resultado/valor].

## 7.4 Objetivo verificable

Definir un resultado técnico concreto, observable y verificable.

Evitar objetivos ambiguos como “mejorar el agente” o “hacer que funcione”.

## 7.5 Alcance

Listar explícitamente:

- componentes que deben crearse/modificarse;
- comportamientos requeridos;
- integraciones;
- configuración/datos implicados.

## 7.6 Fuera de alcance

Indicar qué **no** debe implementarse para prevenir scope creep, sobreingeniería y adelanto de historias posteriores.

## 7.7 Decisiones y restricciones técnicas

Documentar únicamente lo necesario para implementar la HU:

- interfaces esperadas;
- módulos responsables;
- reglas de arquitectura;
- compatibilidad local/Colab/GPU;
- idempotencia;
- persistencia;
- SOLID/DRY;
- restricciones específicas del algoritmo.

Debe respetar `3_BattleZone/docs/lineamientos.md`.

## 7.8 Plan de implementación / tareas

Dividir el trabajo en tareas pequeñas y ordenadas.

Cada tarea debe indicar:

- qué cambia;
- dónde cambia;
- resultado esperado;
- dependencias con tareas anteriores.

## 7.9 Criterios de aceptación

Preferir formato Given/When/Then:

```text
CA01
Dado ...
Cuando ...
Entonces ...
```

Deben cubrir:

- comportamiento funcional;
- integración;
- errores/casos borde relevantes;
- reproducibilidad cuando aplique;
- restricciones arquitectónicas.

## 7.10 Definition of Done

Checklist mínima:

```text
- [ ] implementación completada;
- [ ] criterios de aceptación satisfechos;
- [ ] autovalidaciones ejecutadas;
- [ ] no existen bloqueantes conocidos;
- [ ] documentación/configuración actualizada cuando aplica;
- [ ] evidencia disponible;
- [ ] PR limitado al alcance de la HU.
```

## 7.11 Autovalidaciones obligatorias

Cada HU debe especificar cómo demostrar que funciona.

Pueden incluir:

- imports;
- tests unitarios focalizados;
- smoke tests;
- shapes/dtypes;
- steps del entorno;
- forward pass;
- actualización real de pesos;
- Replay Buffer/PER;
- Target Network sync;
- save/load;
- checkpoint/resume;
- TensorBoard;
- `run_manifest`;
- evaluación corta;
- ejecución E2E;
- validación local;
- validación Colab GPU.

Cada autovalidación debe definir:

1. comando o procedimiento;
2. resultado esperado;
3. criterio PASS/FAIL.

Si una validación solo puede ejecutarse en Colab/GPU, debe marcarse explícitamente como **validación Colab pendiente de ejecución por el usuario** hasta obtener evidencia real.

## 7.12 Evidencias esperadas

La HU debe definir qué evidencia demuestra su éxito, por ejemplo:

- salida de tests;
- métricas;
- tabla;
- screenshot o gráfica TensorBoard;
- manifiesto de ejecución;
- checkpoint restaurado;
- notebook ejecutado;
- modelo generado;
- resultados de evaluación.

## 7.13 Riesgos y consideraciones

Registrar únicamente riesgos materiales, especialmente:

- RAM/VRAM;
- duración de sesión Colab;
- pérdida de checkpoints;
- versiones incompatibles;
- errores de shapes;
- duplicación de frameskip;
- degradación de información visual por preprocessing;
- cambios que invaliden comparación con baseline.

---

# 8. Regla de cierre de una HU

Una HU se considera **implementada** cuando el código existe.

Una HU se considera **cerrada** únicamente cuando:

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

Si una autovalidación obligatoria depende de Colab y todavía no fue ejecutada, la HU debe mantenerse como **implementada pendiente de validación**.

---

# 9. Regla de precedencia documental

Ante contradicciones:

1. `enunciado_reto_1.txt` — restricciones académicas;
2. `3_BattleZone/ficha_tecnica.md` — decisiones específicas del entorno;
3. `3_BattleZone/docs/implementacion.md` — secuencia de implementación;
4. `3_BattleZone/docs/lineamientos.md` — políticas técnicas transversales;
5. HU/DWP puntual — alcance específico.

Ninguna HU puede invalidar silenciosamente una restricción de nivel superior.