# HU004 — Selección formal del algoritmo para BattleZone

## 1. Identificación

- **ID:** HU004
- **Nombre:** Selección formal del algoritmo para BattleZone
- **Estado:** [COMPLETADA — decisión revisada]
- **Dependencia previa:** HU003 — Pipeline reproducible del entorno `[COMPLETADA]`
- **Habilita:** HU005 — Núcleo del agente
- **Algoritmo final seleccionado:** `DQN`
- **Gate posterior:** HU005 debe implementar únicamente DQN.

## 2. Corrección de la decisión

La primera versión de HU004 seleccionó DDQN porque obtuvo el mejor score técnico de la matriz comparativa (`3.72`). Posteriormente se identificó una restricción global del enunciado que no había sido incorporada: el reto exige utilizar al menos dos métodos distintos entre los tres problemas.

El equipo ya utiliza DDQN en LunarLander y Assault. Por tanto, DDQN deja de ser elegible para BattleZone aunque continúe siendo el mejor candidato técnico de la matriz original.

Esta corrección es exclusivamente documental y no altera HU001-HU003 ni los resultados empíricos previamente obtenidos.

## 3. Matriz técnica conservada

| Algoritmo | Score |
|---|---:|
| DDQN | 3.72 |
| DQN | 3.34 |
| DQN + PER | 3.30 |
| REINFORCE | 2.14 |

La matriz original permanece válida como análisis técnico.

## 4. Regla de elegibilidad global

Para la decisión final se añade la siguiente restricción:

> El algoritmo seleccionado para BattleZone debe contribuir al cumplimiento del requisito global de utilizar al menos dos métodos distintos en el reto.

Aplicación:

- DDQN: no elegible para BattleZone por uso previo en LunarLander y Assault.
- DQN: elegible.
- DQN + PER: elegible.
- REINFORCE: elegible.

## 5. Resultado final

**Algoritmo seleccionado para BattleZone: DQN.**

DQN es la alternativa elegible con mejor score técnico (`3.34`).

DQN + PER queda como alternativa de contingencia si durante entrenamiento DQN evidencia baja eficiencia muestral.

## 6. Implicaciones para HU005

HU005 debe implementar:

- CNN/Q-Network;
- Online Network;
- Target Network;
- Replay Buffer uniforme;
- epsilon-greedy;
- target DQN clásico usando `max(Q_target(next_state))`;
- optimizer;
- actualización de pesos;
- save/load básico.

HU005 no debe implementar lógica DDQN de selección Online + evaluación Target.

## 7. Alcance de esta corrección

Incluido:

- actualización de la decisión de HU004;
- incorporación de la restricción global del reto;
- cambio de algoritmo final DDQN → DQN;
- actualización de implicaciones para HU005;
- actualización del mapa de implementación.

Fuera de alcance:

- código del agente;
- tests de agente;
- entrenamiento;
- preprocessing;
- HU003;
- `2_Assault/`;
- MLflow.

## 8. Criterios de aceptación revisados

- CA01: matriz original conservada — PASS.
- CA02: restricción global del enunciado incorporada — PASS.
- CA03: DDQN identificado como no elegible — PASS.
- CA04: DQN seleccionado como mejor alternativa elegible — PASS.
- CA05: implicaciones HU005 actualizadas a DQN — PASS.
- CA06: sin cambios de código o entrenamiento — PASS.
- CA07: contrato HU003 intacto — PASS.
- CA08: independencia de Assault intacta — PASS.

## 9. Evidencia oficial

La decisión final y su trazabilidad se encuentran en:

`3_BattleZone/docs/hu004_decision_algoritmo.md`

## 10. Cierre

HU004 continúa `[COMPLETADA]`, pero con una decisión revisada y trazable:

**DDQN → descartado por restricción global del reto.**

**DQN → algoritmo final seleccionado para BattleZone.**