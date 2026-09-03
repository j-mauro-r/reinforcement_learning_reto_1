# Plan de implementación E2E — BattleZone

## 1. Objetivo

Definir el orden obligatorio de implementación para desarrollar, entrenar, evaluar y entregar un agente de Reinforcement Learning para `ALE/BattleZone-v5`, cumpliendo las restricciones académicas del Reto 1 y aplicando una filosofía de MLOps ligera, reproducible y trazable.

El proyecto debe producir un agente capaz de demostrar un comportamiento lógico aprendido y maximizar la recompensa promedio sobre al menos 10 partidas independientes.

Algoritmos permitidos por el reto:

- DQN;
- DQN + Prioritized Experience Replay;
- DDQN;
- REINFORCE.

Restricción global del reto: deben utilizarse al menos dos métodos distintos a lo largo de los tres problemas. Como DDQN ya fue utilizado en LunarLander y Assault, BattleZone utilizará `DQN` como algoritmo final seleccionado en HU004.

---

## 2. Relación con el agente Assault

El trabajo realizado previamente para Assault se utilizará únicamente como base de conocimiento y referencia metodológica.

No se copiará, importará ni reutilizará código del agente Assault.

BattleZone tendrá su propia implementación, configuración, tests, notebooks, módulos, artefactos y decisiones de arquitectura.

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
9. BattleZone no utilizará MLflow.
10. La evaluación final estará desacoplada del entrenamiento.
11. El agente final deberá evaluarse sobre al menos 10 episodios independientes.
12. El resultado se comparará contra una política aleatoria bajo condiciones equivalentes.
13. No se optimizarán hiperparámetros sin una hipótesis explícita.
14. El código BattleZone será independiente de Assault.
15. Algoritmo final BattleZone: `DQN`.
16. El modelo académico entregable deberá poder cargarse y ejecutarse de forma autónoma sin Replay Buffer ni optimizer.
17. La evidencia visual de entrenamiento y comportamiento post-entrenamiento debe poder regenerarse desde artefactos trazables del mismo `run_id`.

---

## 4. Mapa E2E de HUs

```text
HU001  Caracterización técnica y ficha inicial de BattleZone
  ↓
HU002  Experimento 0 y baseline aleatorio - [COMPLETADA]
  ↓
HU003  Pipeline reproducible del entorno - [COMPLETADA]
  ↓
HU004  Selección formal del algoritmo - [COMPLETADA — DQN]
  ↓
HU005  Núcleo del agente DQN
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
HU011B Entregables técnicos: modelo, gráficas y videos
  ↓
HU012  Optimización controlada de hiperparámetros
  ↓
HU013  Evaluación formal contra baseline
  ↓
HU014  Reporte técnico, evidencias y entrega final
```

---

## 5. Historias de Usuario

### HU001 — Caracterización técnica y ficha inicial de BattleZone
Propósito: comprender formalmente el entorno antes de diseñar el agente.

### HU002 — Experimento 0 y baseline aleatorio
Propósito: medir empíricamente BattleZone mediante una política completamente aleatoria.

### HU003 — Pipeline reproducible del entorno
Propósito: crear una única forma reproducible de inicializar y preprocesar BattleZone.
Estado: `[COMPLETADA]`.

### HU004 — Selección formal del algoritmo
Propósito: elegir justificadamente uno de los algoritmos permitidos por el reto.

Resultado final revisado: `DQN`.

La matriz técnica original se conserva:

- DDQN: `3.72` — mejor score técnico, no elegible por restricción global del reto.
- DQN: `3.34` — mejor alternativa elegible y algoritmo seleccionado.
- DQN + PER: `3.30` — alternativa de contingencia.
- REINFORCE: `2.14`.

Entregable: `3_BattleZone/docs/hu004_decision_algoritmo.md`.

Gate: HU005 debe implementar únicamente `DQN`.

### HU005 — Núcleo del agente DQN
Propósito: implementar los componentes propios de DQN sin integrar todavía el ciclo completo de entrenamiento.

Debe incluir:

- CNN/Q-Network;
- Online Network;
- Target Network;
- Replay Buffer uniforme;
- política epsilon-greedy;
- targets DQN usando `max(Q_target(next_state))`;
- optimizer;
- actualización de pesos;
- interfaces básicas save/load.

No debe implementar lógica DDQN de selección Online + evaluación Target.

Resultado: forward pass válido y al menos una actualización sobre datos controlados.

### HU006 — Ciclo de entrenamiento
Integrar entorno, agente, exploración, replay y actualizaciones DQN en un flujo controlado.

### HU007 — Checkpoints, reanudación e idempotencia
Guardar/restaurar redes, optimizer, progreso y demás estado necesario.

### HU008 — Observabilidad con TensorBoard
Registrar reward, loss, epsilon, Q-values y métricas operativas.

### HU009 — Smoke test end-to-end
Validar sistema completo antes del entrenamiento largo.

### HU010 — Trazabilidad ligera de experimentos
Versionar configuración, run_id, manifiestos y resultados sin MLflow.

### HU011 — Entrenamiento completo
Ejecutar entrenamiento largo en Colab GPU con checkpoints y trazabilidad.

### HU011B — Entregables técnicos: modelo, gráficas y videos
Propósito: convertir la corrida real de HU011 en artefactos académicos verificables por terceros antes de optimización/evaluación formal.

Debe incluir:

- `battlezone_dqn_model.pt` compacto y cargable sin estado de entrenamiento;
- checksum y metadata de linaje;
- carga autónoma por el profesor sin depender obligatoriamente del Drive del equipo;
- gráficas de entrenamiento reconstruidas desde TensorBoard;
- API de gráfica de rewards de explotación preparada para resultados HU013;
- video MP4 del proceso de entrenamiento usando un checkpoint intermedio real;
- video MP4 post-entrenamiento generado desde el modelo entregable con `epsilon=0.0`;
- integración y visualización de estos artefactos en el notebook;
- `HU011B_DELIVERY_GATE=PASS` como condición de cierre.

Fuente de verdad: `3_BattleZone/docs/hu011b_entregables_tecnicos_modelo_graficas_video.md`.

La generación de videos/sanity de HU011B no sustituye la evaluación formal de al menos 10 episodios de HU013.

### HU012 — Optimización controlada de hiperparámetros
Evaluar cambios bajo hipótesis explícitas. Si DQN muestra baja eficiencia muestral, DQN + PER puede evaluarse como alternativa formal.

### HU013 — Evaluación formal contra baseline
Evaluar al menos 10 episodios y comparar contra HU002. Los rewards formales deben alimentar la gráfica final de explotación preparada en HU011B.

### HU014 — Reporte técnico, evidencias y entrega final
Consolidar notebook, modelo, videos, gráficas, métricas, hardware, configuración, evaluación formal y conclusiones.

---

## 6. Estado actual

- HU001: completada según documentación del proyecto.
- HU002: `[COMPLETADA]`.
- HU003: `[COMPLETADA]`.
- HU004: `[COMPLETADA — decisión revisada a DQN]`.
- HU005: requiere alineación de su implementación a `DQN` antes de poder cerrarse.
- HU011B: `[DEFINIDA — PENDIENTE DE IMPLEMENTACIÓN]`.