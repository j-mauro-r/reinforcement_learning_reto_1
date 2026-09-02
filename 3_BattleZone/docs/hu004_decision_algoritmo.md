# Evidencia de implementación — HU004 Selección formal del algoritmo (BattleZone)

## 1. Objetivo

Seleccionar formalmente un único algoritmo para HU005 usando evidencia disponible del proyecto (HU001-HU003), restricciones del enunciado y propiedades teóricas de los algoritmos permitidos, sin implementar agente ni entrenamiento en HU004.

## 2. Fuentes consultadas

- `enunciado_reto_1.txt`
- `3_BattleZone/docs/ficha_tecnica.md`
- `3_BattleZone/docs/implementacion.md`
- `3_BattleZone/docs/lineamientos.md`
- `3_BattleZone/docs/arquitectura.md`
- `3_BattleZone/docs/hu002_evidencia_implementacion.md`
- `3_BattleZone/docs/hu003_evidencia_implementacion.md`
- `3_BattleZone/configs/battlezone_config.yaml`

## 3. Evidencia BattleZone conservada

- Entorno: `ALE/BattleZone-v5`.
- Action space: `Discrete(18)`.
- Baseline aleatorio (10 episodios): media `3000.0`, mediana `2000.0`, desviación estándar `3065.94`, mínimo `0.0`, máximo `10000.0`.
- Reward positivo aproximado: `0.1725 %`; reward cero: `99.8275 %`.
- Contrato HU003: `(4,128,128,3)` `uint8`, RGB, sin crop, `frame_stack=4`, `frameskip=4`, sticky actions `0.25`, reward sin transformación.

## 4. Restricción global del reto incorporada

El enunciado exige utilizar **al menos dos métodos distintos** a lo largo de los tres problemas del reto. DDQN ya fue usado por el equipo en LunarLander y Assault. Por tanto, usar también DDQN en BattleZone impediría cumplir el requisito global mínimo de diversidad algorítmica.

Esta restricción no invalida la matriz original de HU004. Se incorpora como una **condición de elegibilidad transversal** que no se había considerado en la primera decisión.

## 5. Matriz técnica original

| Algoritmo | Score técnico |
|---|---:|
| DDQN | 3.72 |
| DQN | 3.34 |
| DQN + PER | 3.30 |
| REINFORCE | 2.14 |

Ranking técnico original:

1. DDQN — `3.72`
2. DQN — `3.34`
3. DQN + PER — `3.30`
4. REINFORCE — `2.14`

## 6. Elegibilidad revisada

| Algoritmo | Score | Elegible | Motivo |
|---|---:|---|---|
| DDQN | 3.72 | No | Ya utilizado en LunarLander y Assault; no aporta el segundo método requerido por el reto. |
| DQN | 3.34 | Sí | Segundo mejor score técnico y cumple la diversidad requerida. |
| DQN + PER | 3.30 | Sí | Elegible, pero con mayor complejidad/costo que DQN. |
| REINFORCE | 2.14 | Sí | Elegible, pero con peor adecuación esperada a reward sparse y episodios largos. |

## 7. Decisión final revisada

### Algoritmo seleccionado: `DQN`

DQN pasa a ser la mejor alternativa **elegible** para BattleZone.

La decisión cambia por una restricción global del reto, no porque DDQN haya dejado de ser técnicamente competitivo.

## 8. Razones principales

1. Segundo mejor score técnico de la matriz original (`3.34`).
2. Cumple el requisito de usar al menos dos métodos distintos en el reto.
3. Compatible con observación visual y `Discrete(18)`.
4. Conserva Replay Buffer uniforme, relevante ante reward sparse.
5. Menor complejidad que DQN + PER.
6. Compatible con checkpoints/resume y sesiones fragmentadas de Colab.

## 9. Sensibilidad reinterpretada

Se conservan los escenarios originales como evidencia histórica:

- Matriz base: DQN es el mejor algoritmo elegible.
- Escenario simplicidad/costo: DQN sigue siendo el mejor elegible (`3.46`).
- Escenario eficiencia muestral: DQN + PER supera a DQN (`3.44` vs `3.12`), por lo que queda como alternativa de contingencia si DQN muestra baja eficiencia muestral.

## 10. Implicaciones para HU005

HU005 debe implementar **DQN**, no DDQN.

Componentes requeridos:

- CNN/Q-Network;
- Online Network;
- Target Network;
- Replay Buffer uniforme;
- epsilon-greedy;
- target DQN clásico: `reward + gamma * (1-done) * max_a Q_target(next_state,a)`;
- optimizer y actualización de pesos;
- save/load básico.

No debe implementar:

- selección Online + evaluación Target propia de DDQN;
- PER en la primera iteración;
- REINFORCE;
- MLflow.

## 11. Riesgos y contingencias

- DQN puede presentar sobreestimación de Q-values frente a DDQN; debe observarse durante entrenamiento.
- Si la eficiencia muestral es insuficiente, DQN + PER es la alternativa prioritaria.
- HU004 no demuestra performance; la selección debe validarse empíricamente en HUs posteriores.

## 12. Trazabilidad de la corrección

La decisión original DDQN se conserva como antecedente técnico. La corrección cambia únicamente la **decisión final elegible** a DQN por una restricción académica global omitida inicialmente.

No se modifican:

- HU003;
- preprocessing;
- código del agente;
- tests;
- `2_Assault/`;
- resultados empíricos previos.

## 13. Autovalidaciones HU004 revisadas

| AV | Estado | Evidencia |
|---|---|---|
| AV01 Dependencias | PASS | HU003 completada y contrato vigente. |
| AV02 Candidatos permitidos | PASS | DQN, DQN+PER, DDQN y REINFORCE. |
| AV03 Suma de pesos | PASS | Matriz original conserva `100 %`. |
| AV04 Escala válida | PASS | Escala original 1–5 conservada. |
| AV05 Scores | PASS | Scores originales conservados sin alteración. |
| AV06 Trazabilidad | PASS | Evidencia BattleZone y restricción global diferenciadas. |
| AV07 Evidencia BattleZone | PASS | Sparse reward, varianza, Discrete(18), contrato visual y Colab considerados. |
| AV08 Sensibilidad | PASS | Escenarios originales preservados y reinterpretados bajo elegibilidad. |
| AV09 Selección única | PASS | Selección final única: DQN. |
| AV10 Componentes posteriores | PASS | HU005 debe implementar DQN. |
| AV11 Sin implementación agente | PASS | Corrección exclusivamente documental. |
| AV12 Contrato HU003 intacto | PASS | Sin cambios. |
| AV13 Independencia Assault | PASS | Sin cambios/imports desde `2_Assault/`. |
| AV14 Sin MLflow | PASS | No se introduce MLflow. |
| AV15 Alcance PR | PASS | Solo documentación HU004/plan de implementación. |

## 14. Conclusión

**Decisión final HU004: DQN.**

DDQN conserva el mejor score técnico original, pero queda descartado por la restricción global del reto. DQN es la mejor alternativa elegible y permite cumplir el requisito de utilizar al menos dos métodos distintos en los tres ejercicios.