# Evidencia de implementación — HU004 Selección formal del algoritmo (BattleZone)

## 1. Objetivo

Seleccionar formalmente un único algoritmo para HU005 usando evidencia disponible del proyecto (HU001-HU003), restricciones del enunciado y propiedades teóricas de los algoritmos permitidos, sin implementar agente ni entrenamiento en HU004.

## 2. Fuentes consultadas

Fuentes leídas y utilizadas:

- `enunciado_reto_1.txt`
- `3_BattleZone/docs/ficha_tecnica.md`
- `3_BattleZone/docs/implementacion.md`
- `3_BattleZone/docs/lineamientos.md`
- `3_BattleZone/docs/arquitectura.md`
- `3_BattleZone/docs/hu001_caracterizacion_tecnica_battlezone.md`
- `3_BattleZone/docs/hu001_evidencia_implementacion.md`
- `3_BattleZone/docs/hu002_experimento_0_baseline_aleatorio.md`
- `3_BattleZone/docs/hu002_evidencia_implementacion.md`
- `3_BattleZone/docs/hu003_pipeline_reproducible_entorno.md`
- `3_BattleZone/docs/hu003_evidencia_implementacion.md`
- `3_BattleZone/configs/battlezone_config.yaml`

## 3. Evidencia HU001-HU003 utilizada

### 3.1 Evidencia empírica del proyecto

- Entorno y acción: `ALE/BattleZone-v5`, `Discrete(18)`.
  Fuente: `3_BattleZone/docs/ficha_tecnica.md`, `3_BattleZone/docs/hu002_evidencia_implementacion.md`.
- Baseline aleatorio (10 episodios):
  - media `3000.0`
  - mediana `2000.0`
  - desviación estándar `3065.94`
  - mínimo `0.0`
  - máximo `10000.0`
  - steps promedio `1159.5`
  - reward positivo `0.1725 %`
  - reward cero `99.8275 %`
  Fuente: `3_BattleZone/docs/hu002_evidencia_implementacion.md`, `3_BattleZone/docs/ficha_tecnica.md`.
- Contrato de HU003:
  - observación final `(4, 128, 128, 3)` `uint8`
  - pipeline RGB, sin crop, `frame_stack=4`
  - `frameskip=4` aplicado una sola vez
  - `repeat_action_probability=0.25`
  - reward transform `none`
  - tamaño aproximado por estado `0.1875 MB`
  Fuente: `3_BattleZone/docs/hu003_evidencia_implementacion.md`, `3_BattleZone/configs/battlezone_config.yaml`.

### 3.2 Decisiones técnicas previas aprobadas

- BattleZone no usa MLflow.
  Fuente: `3_BattleZone/docs/lineamientos.md`, `3_BattleZone/docs/implementacion.md`.
- HU004 es una HU de decisión, no de entrenamiento.
  Fuente: `3_BattleZone/docs/implementacion.md`, `3_BattleZone/docs/hu004_seleccion_formal_algoritmo.md`.
- HU005 debe implementar solo el algoritmo elegido en HU004.
  Fuente: `3_BattleZone/docs/implementacion.md`, `3_BattleZone/docs/hu004_seleccion_formal_algoritmo.md`.

## 4. Clasificación de afirmaciones (anti-alucinación)

| Afirmación | Tipo | Fuente | Verificación |
|---|---|---|---|
| `Discrete(18)` | Evidencia empírica | `3_BattleZone/docs/ficha_tecnica.md`, `3_BattleZone/docs/hu002_evidencia_implementacion.md` | PASS |
| Baseline random media `3000.0` | Evidencia empírica | `3_BattleZone/docs/hu002_evidencia_implementacion.md` | PASS |
| Reward positivo `0.1725 %` | Evidencia empírica | `3_BattleZone/docs/hu002_evidencia_implementacion.md` | PASS |
| Estado final `(4,128,128,3)` `uint8` | Decisión técnica aprobada | `3_BattleZone/docs/hu003_evidencia_implementacion.md`, `3_BattleZone/configs/battlezone_config.yaml` | PASS |
| DDQN reduce sobreestimación respecto a DQN | Propiedad teórica | Definición algorítmica de DDQN usada en curso | PASS |
| DQN+PER prioriza transiciones con mayor señal de aprendizaje | Propiedad teórica | Definición algorítmica de PER usada en curso | PASS |
| REINFORCE podría sufrir alta varianza con episodios largos y reward sparse | Inferencia/Hipótesis | Razonamiento técnico sobre HU002 + propiedad del método | PASS (como hipótesis) |
| Consumo exacto de VRAM por algoritmo en Colab | No disponible | No hay medición HU001-HU003 | N/A |

## 5. Criterios de selección

Escala única usada para todos los criterios:

- `1 = muy desfavorable`
- `2 = desfavorable`
- `3 = aceptable`
- `4 = favorable`
- `5 = muy favorable`

Criterios (no redundantes):

1. Eficiencia muestral con reward sparse.
2. Estabilidad de aprendizaje y control de sobreestimación/varianza.
3. Costo computacional esperado por actualización.
4. Costo de memoria esperado (RAM/VRAM) para entrenamiento.
5. Complejidad de implementación y riesgo de errores en HU005-HU006.
6. Compatibilidad con contrato BattleZone (visual + `Discrete(18)`).
7. Compatibilidad con checkpoint/resume para sesiones Colab.
8. Facilidad de observabilidad operativa con TensorBoard.
9. Adecuación a episodios largos y alta variabilidad observada.

## 6. Pesos y justificación

| Criterio | Peso (%) | Justificación del peso | Tipo de evidencia dominante |
|---|---:|---|---|
| Eficiencia muestral con reward sparse | 20 | HU002 observó reward positivo muy escaso (`0.1725 %`) | Evidencia empírica |
| Estabilidad y sobreestimación/varianza | 16 | Baseline con alta varianza; estabilidad condiciona viabilidad | Evidencia + teoría |
| Costo computacional por actualización | 10 | Colab tiene presupuesto temporal limitado | Restricción proyecto |
| Costo de memoria esperado | 10 | Estado visual apilado y posible replay buffer | Evidencia + inferencia |
| Complejidad de implementación/riesgo | 12 | HU005-HU006 deben cerrar sin sobre-ingeniería | Restricción proyecto |
| Compatibilidad con contrato visual + 18 acciones | 10 | Contrato HU003 ya está fijado | Decisión aprobada |
| Checkpoint/resume en sesiones fragmentadas | 8 | Enunciado exige continuidad entre sesiones | Enunciado |
| Observabilidad con TensorBoard | 6 | Métricas entrenables y depuración operativa | Lineamientos |
| Adecuación a episodios largos y varianza | 8 | HU002 mostró episodios largos y dispersión alta | Evidencia empírica |

Suma de pesos: `100 %`.

## 7. Matriz comparativa completa

| Criterio | Peso | DQN | DQN+PER | DDQN | REINFORCE | Justificación resumida |
|---|---:|---:|---:|---:|---:|---|
| Eficiencia muestral con reward sparse | 20 | 2 | 4 | 3 | 1 | PER mejora muestreo de eventos raros; REINFORCE usa trayectorias completas con señal tardía |
| Estabilidad y sobreestimación/varianza | 16 | 2 | 3 | 4 | 1 | DDQN desacopla selección/evaluación; REINFORCE tiene varianza alta esperada |
| Costo computacional por actualización | 10 | 4 | 2 | 3 | 2 | PER añade sobrecosto; DDQN costo intermedio por doble red |
| Costo de memoria esperado | 10 | 3 | 2 | 3 | 4 | Métodos con replay exigen buffer; PER añade estructuras de prioridad |
| Complejidad implementación/riesgo HU005 | 12 | 4 | 2 | 3 | 3 | DQN más simple; PER aumenta complejidad; DDQN intermedio |
| Compatibilidad con contrato visual + 18 acciones | 10 | 5 | 5 | 5 | 4 | Los cuatro aplican; value-based encaja de forma directa en acción discreta |
| Checkpoint/resume en Colab | 8 | 5 | 4 | 5 | 2 | Value-based guarda estado incremental por timestep con mayor naturalidad |
| Observabilidad con TensorBoard | 6 | 5 | 4 | 5 | 3 | Value-based facilita loss/q-metrics directas |
| Adecuación a episodios largos y varianza | 8 | 3 | 4 | 4 | 1 | Replay ayuda reutilización; REINFORCE acumula varianza en retornos largos |

## 8. Fórmula

`score_total = Σ((peso_i / 100) * puntuación_i)`

## 9. Cálculo reproducible

| Algoritmo | Suma ponderada | Score final (0-5) |
|---|---:|---:|
| DQN | 334 | 3.34 |
| DQN + PER | 330 | 3.30 |
| DDQN | 372 | 3.72 |
| REINFORCE | 214 | 2.14 |

Donde “Suma ponderada” es `Σ(peso * score)` y `Score final = suma / 100`.

## 10. Ranking

1. `DDQN` — `3.72`
2. `DQN` — `3.34`
3. `DQN + PER` — `3.30`
4. `REINFORCE` — `2.14`

## 11. Análisis cualitativo

- `DDQN` queda primero por mejor equilibrio entre estabilidad teórica, costo razonable y compatibilidad con restricciones de Colab.
- `DQN` queda segundo por simplicidad y viabilidad, pero con mayor riesgo teórico de sobreestimación.
- `DQN + PER` mejora eficiencia muestral esperada en reward sparse, pero su complejidad/costo adicional reduce su prioridad como primera implementación.
- `REINFORCE` queda último por riesgo esperado de varianza alta con episodios largos y señal de reward escasa.

## 12. Sensibilidad de la decisión

### Escenario S1: Prioridad a eficiencia muestral

Pesos alternativos (renormalizados, suman 100):

- eficiencia 30, estabilidad 16, costo cómputo 8, memoria 9, complejidad 8, compatibilidad 10, resume 7, observabilidad 4, episodios largos 8.

Resultados:

- DQN: `3.12`
- DQN+PER: `3.44`
- DDQN: `3.66`
- REINFORCE: `1.96`

### Escenario S2: Prioridad a simplicidad y costo

Pesos alternativos (suman 100):

- eficiencia 14, estabilidad 14, costo cómputo 16, memoria 14, complejidad 18, compatibilidad 10, resume 6, observabilidad 4, episodios largos 4.

Resultados:

- DQN: `3.46`
- DQN+PER: `3.00`
- DDQN: `3.58`
- REINFORCE: `2.38`

### Conclusión de sensibilidad

La selección es robusta dentro de los escenarios evaluados: el ganador se mantiene (`DDQN`).

## 13. Algoritmo seleccionado

`DDQN`.

## 14. Razones principales de selección

1. Mejor balance ponderado en matriz base.
2. Ventaja teórica relevante en control de sobreestimación frente a DQN.
3. Costo y complejidad menores que DQN+PER para una primera implementación BattleZone.
4. Mejor adecuación esperada que REINFORCE dadas recompensas extremadamente sparse y episodios largos observados.

## 15. Riesgos

- Hipótesis aún no validadas por entrenamiento real: que la ventaja teórica de DDQN se traduzca en mejora práctica en BattleZone.
- Riesgo de presión de memoria al definir tamaño de replay buffer en HU005-HU006.
- Posible necesidad futura de PER si la eficiencia muestral observada en entrenamiento resulta insuficiente.

## 16. Implicaciones para HU005

### Componentes que HU005 deberá implementar

- Q-Network (CNN para entrada visual).
- Online Network y Target Network.
- Replay Buffer uniforme.
- Política epsilon-greedy.
- Target DDQN (selección con Online, evaluación con Target).

### Componentes que HU005 no necesita implementar

- Prioritized Experience Replay (PER) en la primera implementación.
- Elementos de policy-gradient tipo REINFORCE.
- MLflow.

### Hipótesis por validar en HU005/HU006/HU009

- Hipótesis: DDQN logrará aprendizaje más estable que DQN bajo este contrato BattleZone.
- Hipótesis: Replay uniforme será suficiente en primera iteración sin PER.
- Hipótesis: costos de RAM/VRAM en Colab serán viables con configuración inicial.

## 17. Autovalidaciones de HU004

| AV | Estado | Evidencia |
|---|---|---|
| AV01 Cuatro algoritmos incluidos | PASS | Matriz incluye DQN, DQN+PER, DDQN, REINFORCE |
| AV02 Ningún algoritmo permitido omitido | PASS | Se cubren exactamente los cuatro del enunciado |
| AV03 Pesos suman 100 % | PASS | Tabla de pesos suma 100 |
| AV04 Fórmula documentada | PASS | Sección de fórmula |
| AV05 Scores recalculables | PASS | Tabla de cálculo reproducible |
| AV06 Ranking matemáticamente consistente | PASS | Orden consistente con scores |
| AV07 Sensibilidad calculada | PASS | Escenarios S1 y S2 |
| AV08 Datos empíricos con fuente | PASS | Secciones 3 y 4 con trazabilidad |
| AV09 Teoría vs inferencia diferenciadas | PASS | Sección 4 |
| AV10 Sin resultados inventados de entrenamiento | PASS | No se reportan entrenamientos |
| AV11 Sin cambios en `2_Assault/` | PASS | Alcance documental en `3_BattleZone/` |
| AV12 Sin imports desde `2_Assault/` en cambios HU004 | PASS | Sin código nuevo de agente |
| AV13 Sin implementación de agente/entrenamiento | PASS | Cambios documentales |
| AV14 Sin MLflow | PASS | Sin referencias nuevas de MLflow |
| AV15 Preprocessing HU003 intacto | PASS | No hay cambios de contrato |
| AV16 HU005 restringida a algoritmo seleccionado | PASS | Sección 16 |
| AV17 Diff limitado a HU004 | PASS | Cambios acotados a docs HU004 y plan |

## 18. Limitaciones

- HU004 no ejecuta entrenamiento; por diseño, los scores son una decisión de arquitectura y no una prueba empírica de performance.
- No existe evidencia empírica disponible en HU001-HU003 para cuantificar consumo exacto de VRAM por algoritmo en Colab.
- No existe evidencia empírica disponible en HU001-HU003 para comparar throughput real entre DQN, DQN+PER, DDQN y REINFORCE en BattleZone.

## 19. Datos no disponibles que no pudieron verificarse

- Consumo exacto de RAM/VRAM por algoritmo durante entrenamiento real en Colab.
- Tiempo por actualización y tiempo total por target de timesteps para cada algoritmo candidato.
- Diferencias empíricas reales de recompensa final entre candidatos en BattleZone.

Estado de esos puntos: `PENDIENTE — por validar en HU005/HU006/HU009/HU011`.
