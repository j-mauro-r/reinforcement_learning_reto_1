# Ficha técnica del entorno BattleZone

## 1. Objetivo

Documentar las características de `ALE/BattleZone-v5` relevantes para diseñar, implementar, entrenar y evaluar el agente de Reinforcement Learning del Reto 1, integrando en un único documento:

1. hechos soportados por documentación oficial;
2. restricciones académicas del reto;
3. evidencia empírica obtenida en HU002 — Experimento 0 y baseline aleatorio;
4. implicaciones y preguntas abiertas para HU003 y HU004.

BattleZone se implementará de forma completamente independiente de Assault. El trabajo previo de Assault se utiliza únicamente como referencia metodológica y base de conocimiento. **No se copiará, importará ni reutilizará código desde `2_Assault/`.**

---

## 2. Fuentes

### Fuente académica principal

- `enunciado_reto_1.txt`.

### Fuentes técnicas principales

- Arcade Learning Environment — BattleZone: https://ale.farama.org/environments/battle_zone/
- Arcade Learning Environment — Atari Environments: https://ale.farama.org/environments/
- Manual Atari 2600 BattleZone publicado por AtariAge: https://www.atariage.com/manual_html_page.php?SoftwareID=859&SystemID=2600&itemTypeID=HTMLMANUAL

### Evidencia interna

- `3_BattleZone/experimento_0_battlezone.ipynb`.
- `3_BattleZone/data/baseline_random_battlezone_local.json`.
- `3_BattleZone/docs/hu002_evidencia_implementacion.md`.

### Documentación interna del proyecto

- `3_BattleZone/docs/implementacion.md`.
- `3_BattleZone/docs/lineamientos.md`.
- `3_BattleZone/docs/arquitectura.md`.
- `3_BattleZone/docs/hu001_caracterizacion_tecnica_battlezone.md`.
- `3_BattleZone/docs/hu002_experimento_0_baseline_aleatorio.md`.

---

## 3. Restricciones académicas

El Reto 1 permite únicamente:

- DQN;
- DQN + Prioritized Experience Replay;
- DDQN;
- REINFORCE.

La selección formal del algoritmo corresponde a HU004. HU002 no selecciona algoritmo ni implementa entrenamiento.

Para BattleZone no se exige alcanzar desempeño humano u óptimo. Debe demostrarse comportamiento lógico aprendido y no predominantemente aleatorio.

La evaluación formal final deberá ejecutar al menos **10 episodios independientes** y comparar el agente entrenado contra el baseline aleatorio definido en HU002.

---

## 4. Descripción del problema

BattleZone es un juego Atari en primera persona en el que el agente controla un tanque. El objetivo es destruir vehículos enemigos, sobrevivir y maximizar la recompensa acumulada.

El problema combina:

- navegación y orientación espacial;
- adquisición de blancos;
- disparo y evasión;
- radar;
- obstáculos;
- dependencia temporal;
- observación visual de alta dimensionalidad;
- un espacio de acciones considerablemente mayor que otros entornos Atari del reto.

El radar y la vista principal se encuentran en la misma observación visual, por lo que el preprocessing debe conservar información útil de ambas regiones.

---

## 5. Contrato oficial de `ALE/BattleZone-v5`

| Característica | Valor |
|---|---|
| Familia | Atari / Arcade Learning Environment |
| Environment ID | `ALE/BattleZone-v5` |
| Creación | `gymnasium.make("ALE/BattleZone-v5")` |
| Action space | `Discrete(18)` |
| Observación por defecto | RGB |
| Observation space | `Box(0, 255, (210, 160, 3), uint8)` |
| `frameskip` | `4` |
| `repeat_action_probability` | `0.25` |
| Modes disponibles | `[1, 2, 3]` |
| Mode por defecto | `1` |
| Difficulties disponibles | `[0]` |
| Difficulty por defecto | `0` |
| Vidas iniciales documentadas | `5` |

ALE documenta además la posibilidad de obtener vidas adicionales durante una partida según la puntuación alcanzada.

---

## 6. Espacio de acciones

BattleZone utiliza las 18 acciones Atari completas. `full_action_space=True` no amplía el espacio para este juego.

| Índice | Acción |
|---:|---|
| 0 | `NOOP` |
| 1 | `FIRE` |
| 2 | `UP` |
| 3 | `RIGHT` |
| 4 | `LEFT` |
| 5 | `DOWN` |
| 6 | `UPRIGHT` |
| 7 | `UPLEFT` |
| 8 | `DOWNRIGHT` |
| 9 | `DOWNLEFT` |
| 10 | `UPFIRE` |
| 11 | `RIGHTFIRE` |
| 12 | `LEFTFIRE` |
| 13 | `DOWNFIRE` |
| 14 | `UPRIGHTFIRE` |
| 15 | `UPLEFTFIRE` |
| 16 | `DOWNRIGHTFIRE` |
| 17 | `DOWNLEFTFIRE` |

### Implicación para RL

`Discrete(18)` aumenta:

- el costo de exploración;
- el número de Q-values a estimar en métodos value-based;
- la posibilidad de ejecutar acciones poco útiles durante exploración;
- la dificultad de descubrir secuencias coordinadas de movimiento y disparo.

No se reducirá artificialmente el action space antes de una decisión técnica explícita y sustentada.

---

## 7. Espacio de observaciones

ALE soporta:

| `obs_type` | Espacio |
|---|---|
| `rgb` | `Box(0,255,(210,160,3),uint8)` |
| `grayscale` | `Box(0,255,(210,160),uint8)` |
| `ram` | `Box(0,255,(128,),uint8)` |

El proyecto parte de observaciones visuales RGB, coherente con el enunciado y con la necesidad de interpretar radar, enemigos y obstáculos.

El preprocessing definitivo permanece abierto para HU003. Deben validarse específicamente:

- RGB vs grayscale;
- resolución objetivo;
- frame stacking;
- normalización;
- cropping;
- conservación del radar y de objetos pequeños.

---

## 8. Dinámica temporal y estocasticidad

Para `ALE/BattleZone-v5`:

- `frameskip=4`;
- `repeat_action_probability=0.25`.

Esto implica que una decisión del agente corresponde normalmente a cuatro frames internos y que existe un 25 % de probabilidad de repetición de la acción anterior.

Consecuencias:

1. el entorno no debe tratarse como completamente determinista;
2. el `frameskip` efectivo debe aplicarse una sola vez;
3. baseline, entrenamiento y evaluación deben mantener configuraciones equivalentes;
4. una única imagen no contiene por sí sola toda la información de movimiento;
5. HU003 deberá evaluar frame stacking u otra representación temporal permitida por la arquitectura definida.

---

## 9. Radar y percepción

BattleZone incorpora un radar en la parte superior de la pantalla. Esta señal puede informar sobre amenazas fuera del campo visual frontal.

La observación contiene por tanto dos regiones complementarias:

- vista principal en primera persona;
- radar.

### Evidencia HU002

El notebook muestra explícitamente:

- frame RGB original;
- región superior correspondiente al radar;
- transformación exploratoria a grayscale y resize `84×84`.

Esta visualización es diagnóstica. **HU002 no demuestra que grayscale `84×84` sea suficiente**, ni autoriza recortar el radar.

HU003 debe evitar decisiones de resize/cropping que destruyan una señal estratégica pequeña.

---

## 10. Movimiento, enemigos y obstáculos

El tanque puede avanzar, retroceder, girar, disparar y combinar movimiento con fuego mediante las acciones disponibles.

La documentación del juego describe objetivos como:

- tanque estándar;
- fighter/aerial fighter o misil según la documentación histórica;
- supertank;
- flying saucer.

Los obstáculos pueden afectar movimiento, línea de visión y posicionamiento relativo.

Una política útil debería aprender patrones relacionados con orientación, adquisición de blancos, disparo, movimiento y evasión. Estos comportamientos se evaluarán cualitativamente además de la recompensa final.

---

## 11. Scoring histórico vs. reward ALE

El manual Atari 2600 reporta:

| Objetivo | Puntaje histórico documentado |
|---|---:|
| Tank | 1.000 |
| Fighter | 2.000 |
| Supertank | 3.000 |
| Saucer | 5.000 |

Estos valores son referencias históricas del videojuego y **no deben asumirse automáticamente como una relación uno-a-uno con cada reward retornado por ALE**.

HU002 observó rewards por step de `1000`, `2000`, `5000` y `6000`, además de `0`. El valor `6000` no corresponde directamente a un único objetivo de la tabla histórica; HU002 no identificó el evento causal exacto. Por tanto, la equivalencia evento del juego ↔ reward ALE permanece parcialmente abierta.

---

# 12. Evidencia empírica HU002 — Experimento 0

## 12.1 Naturaleza de la corrida

La evidencia guardada en el notebook corresponde a una ejecución **local en Windows**, no a Google Colab.

La política fue estrictamente aleatoria mediante:

`env.action_space.sample()`

No se aplicó:

- clipping de reward;
- normalización;
- reward shaping;
- heurísticas;
- preferencia por FIRE;
- reducción del action space.

## 12.2 Configuración

| Parámetro | Valor |
|---|---|
| Environment ID | `ALE/BattleZone-v5` |
| Episodios | `10` |
| Seed base | `20260830` |
| Seeds por episodio | `20260831` a `20260840` |
| Mode | `1` |
| Difficulty | `0` |
| `obs_type` | `rgb` |
| `frameskip` | `4` |
| `repeat_action_probability` | `0.25` |
| Política | Aleatoria |

Para cada episodio se sembraron tanto el entorno como el action space. El notebook incluye además una prueba independiente que reprodujo correctamente una secuencia de 32 acciones usando la misma seed del action space.

---

## 13. Runtime observado en HU002

| Componente | Valor observado |
|---|---|
| Python | `3.8.10` |
| Gymnasium | `1.1.1` |
| ALE-Py | `0.10.1` |
| NumPy | `1.24.4` |
| Plataforma | `Windows-10-10.0.19044-SP0` |
| CPU | `AMD64 Family 23 Model 17 Stepping 0, AuthenticAMD` |
| RAM total | `6.9 GB` |
| GPU | No disponible |

Estas son las versiones **realmente observadas en la corrida local**. Las versiones efectivas del runtime Colab deben registrarse cuando se ejecute AV14.

---

## 14. Contrato empírico observado

HU002 confirmó:

- `observation_space = Box(0, 255, (210, 160, 3), uint8)`;
- `action_space = Discrete(18)`;
- 18 action meanings esperados;
- observación inicial `(210,160,3)`, dtype `uint8`;
- rango observado en el frame inicial `[0,236]`.

No se encontró contradicción material con la caracterización oficial de HU001.

---

## 15. `info` observado

### En `reset()`

Se observaron:

- `lives`;
- `episode_frame_number`;
- `frame_number`;
- `seeds`.

### Durante `step()`

En pasos normales, rewards no-cero, cambios de vida y terminación se observaron:

- `lives`;
- `episode_frame_number`;
- `frame_number`.

No se observaron otras claves relevantes en esta corrida.

Los contadores de frames son candidatos útiles para observabilidad y validación temporal en HU003.

---

## 16. Baseline aleatorio por episodio

| Episodio | Seed | Reward | Steps | Terminated | Truncated | Vidas inicio | Vidas fin |
|---:|---:|---:|---:|---|---|---:|---:|
| 1 | 20260831 | 2000 | 821 | Sí | No | 5 | 0 |
| 2 | 20260832 | 0 | 1079 | Sí | No | 5 | 0 |
| 3 | 20260833 | 10000 | 1038 | Sí | No | 5 | 0 |
| 4 | 20260834 | 7000 | 946 | Sí | No | 5 | 0 |
| 5 | 20260835 | 4000 | 866 | Sí | No | 5 | 0 |
| 6 | 20260836 | 0 | 1960 | Sí | No | 5 | 0 |
| 7 | 20260837 | 1000 | 868 | Sí | No | 5 | 0 |
| 8 | 20260838 | 1000 | 1277 | Sí | No | 5 | 0 |
| 9 | 20260839 | 2000 | 1156 | Sí | No | 5 | 0 |
| 10 | 20260840 | 3000 | 1584 | Sí | No | 5 | 0 |

---

## 17. Métricas agregadas del baseline

### Recompensa

| Métrica | Valor |
|---|---:|
| Media | `3000.0` |
| Mediana | `2000.0` |
| Desviación estándar poblacional | `3065.94` |
| Mínimo | `0.0` |
| Máximo | `10000.0` |

La desviación estándar es ligeramente mayor que la media, por lo que este baseline de 10 episodios presenta alta dispersión.

### Duración

| Métrica | Valor |
|---|---:|
| Steps totales | `11595` |
| Steps promedio/episodio | `1159.5` |
| Mínimo | `821` |
| Máximo | `1960` |

### Terminación

- `terminated=True`: `10/10` episodios.
- `truncated=True`: `0/10` episodios.

En los 10 episodios observados, la terminación ocurrió con `lives_end == 0`.

---

## 18. Densidad de reward

Sobre `11595` steps:

| Clase | Proporción |
|---|---:|
| Reward positivo | `0.1725 %` |
| Reward cero | `99.8275 %` |
| Reward negativo | `0.0 %` |

Eventos de reward no-cero promedio por episodio: `2.0`.

Frecuencia de rewards por step:

| Reward | Frecuencia |
|---:|---:|
| `0` | `11575` |
| `1000` | `17` |
| `2000` | `1` |
| `5000` | `1` |
| `6000` | `1` |

### Lectura técnica

La recompensa es **muy escasa** en esta muestra: aproximadamente 1,7 steps de cada 1000 produjeron reward positivo.

Este resultado es relevante para HU004 porque la eficiencia muestral y la capacidad de aprender de experiencias poco frecuentes deben formar parte de la comparación entre algoritmos permitidos.

---

## 19. Vidas y finalización

HU002 resolvió empíricamente varios puntos abiertos de HU001:

- vidas iniciales observadas: siempre `5`;
- pérdidas promedio por episodio: `5.0`;
- vidas extra detectadas: `0` en esta muestra;
- los 10 episodios terminaron con `lives_end = 0`;
- no se observó `truncated=True`.

Estos resultados describen la muestra HU002 y no demuestran que vidas extra o truncation sean imposibles en otras condiciones.

---

## 20. Distribución de acciones del baseline

El total de acciones contabilizadas coincide con los `11595` steps.

| Acción | Conteo | Frecuencia aprox. |
|---:|---:|---:|
| 0 | 624 | 5.38 % |
| 1 | 629 | 5.42 % |
| 2 | 628 | 5.42 % |
| 3 | 633 | 5.46 % |
| 4 | 626 | 5.40 % |
| 5 | 608 | 5.24 % |
| 6 | 691 | 5.96 % |
| 7 | 665 | 5.74 % |
| 8 | 651 | 5.61 % |
| 9 | 632 | 5.45 % |
| 10 | 655 | 5.65 % |
| 11 | 673 | 5.80 % |
| 12 | 603 | 5.20 % |
| 13 | 622 | 5.36 % |
| 14 | 662 | 5.71 % |
| 15 | 664 | 5.73 % |
| 16 | 653 | 5.63 % |
| 17 | 676 | 5.83 % |

La distribución observada es compatible con el muestreo uniforme esperado de `env.action_space.sample()`; no existe evidencia de una preferencia manual introducida por la implementación.

---

## 21. Preguntas de HU001 resueltas por HU002

| Pregunta | Resultado HU002 |
|---|---|
| Reward promedio de política aleatoria | `3000.0` |
| Dispersión | Alta; std `3065.94`, rango `0–10000` |
| Densidad de reward | `0.1725 %` positivo; `99.8275 %` cero |
| Rewards negativos | No observados |
| Duración típica | Media `1159.5` steps; rango `821–1960` |
| Vidas en `info` | Sí; 5 iniciales, 0 al terminar en 10/10 episodios |
| Claves de `info` | `lives`, `episode_frame_number`, `frame_number`, y `seeds` en reset |
| Terminación | 10/10 `terminated`; 0/10 `truncated` |
| Action space | `Discrete(18)` confirmado |
| Runtime local | Python 3.8.10 / Gymnasium 1.1.1 / ALE-Py 0.10.1 |
| Equivalencia exacta score/reward | Solo parcialmente caracterizada; no demostrada evento por evento |

---

## 22. Preguntas todavía abiertas

HU003/HU004 deberán resolver o considerar:

1. ¿RGB o grayscale conserva mejor la información necesaria del radar y objetos pequeños?
2. ¿`84×84` conserva suficiente información o se necesita una resolución mayor?
3. ¿Qué frame stack proporciona contexto temporal suficiente?
4. ¿Debe evitarse completamente cropping?
5. ¿Cuál es el consumo real de RAM/VRAM con el preprocessing y buffer seleccionados?
6. ¿Cuál es el throughput de interacción/entrenamiento en Colab GPU?
7. ¿Qué algoritmo permitido ofrece mejor compromiso entre eficiencia muestral, estabilidad y costo?
8. ¿Qué explica exactamente los rewards por step agregados como `6000`?
9. ¿Las versiones de runtime en Colab serán iguales o deberán fijarse explícitamente en HU003?

---

## 23. Implicaciones para HU003 — Pipeline reproducible

La evidencia recomienda que HU003:

- conserve una fábrica única para `ALE/BattleZone-v5`;
- mantenga `frameskip=4` aplicado una sola vez;
- preserve `repeat_action_probability=0.25`;
- mantenga action space completo mientras no exista decisión contraria justificada;
- pruebe explícitamente la legibilidad del radar tras preprocessing;
- pruebe RGB vs grayscale y resolución antes de fijarlas;
- considere contexto temporal mediante frame stack;
- registre las versiones efectivas de Colab;
- utilice `info`/contadores de frames para validar el contrato temporal cuando sea útil;
- mantenga train/eval bajo la misma fábrica y preprocessing.

HU002 no fija el preprocessing definitivo.

---

## 24. Implicaciones para HU004 — Selección de algoritmo

La evidencia disponible introduce cuatro factores importantes:

1. **Reward muy escaso:** solo `0.1725 %` de steps tuvo reward positivo.
2. **Alta dispersión:** reward medio `3000`, std `3065.94` y máximo `10000`.
3. **18 acciones:** aumenta el costo de exploración.
4. **Observación visual/temporal:** el agente debe aprender desde píxeles y contexto temporal.

Estos hallazgos deben utilizarse en la matriz comparativa DQN vs DQN+PER vs DDQN vs REINFORCE, pero **no constituyen por sí solos la selección del algoritmo**.

---

## 25. Riesgos técnicos actualizados

| Riesgo | Impacto | Evidencia / mitigación |
|---|---|---|
| Radar pierde legibilidad | Alto | HU002 confirma que es una región pequeña; validar preprocessing en HU003 |
| Reward muy escaso | Alto | 0.1725 % positivo; considerar eficiencia muestral en HU004 |
| Baseline con alta varianza | Alto | std > media; usar evaluación multi-episodio |
| 18 acciones dificultan exploración | Alto | action space confirmado empíricamente |
| Doble frameskip | Alto | fábrica única y test explícito en HU003 |
| Diferencias train/eval | Alto | misma fábrica/preprocessing |
| Sesiones Colab interrumpidas | Alto | checkpoints/resume en HUs posteriores |
| Replay Buffer consume RAM | Alto si aplica | mantener observaciones compactas y perfilar memoria |
| Versiones Colab diferentes | Medio/alto | registrar y fijar versiones en pipeline reproducible |
| Confundir score histórico/reward | Medio | no mapear eventos sin evidencia |

---

## 26. Protocolo de comparación futura

Baseline y agente entrenado deberán conservar condiciones comparables:

- mismo environment ID;
- mismo mode/difficulty;
- mismo `frameskip`;
- mismo `repeat_action_probability`;
- rewards reales del entorno en evaluación;
- al menos 10 episodios independientes;
- reporte de media, mediana, desviación estándar, mínimo y máximo.

Además se registrarán duración, vidas y evidencia cualitativa cuando aporten valor.

El baseline de HU002 que servirá de referencia inicial es:

**reward promedio = `3000.0` en 10 episodios aleatorios locales.**

---

## 27. Estado de HU002 después de auditoría del PR #18

### Implementado y validado con evidencia local

- notebook del Experimento 0;
- política estrictamente aleatoria;
- seed del entorno y del action space;
- prueba de reproducibilidad de muestreo de acciones;
- 10 episodios completos;
- estadísticas agregadas;
- densidad de rewards;
- frecuencia de acciones;
- vidas y terminación;
- `info`;
- visualizaciones de recompensa, duración y acciones;
- inspección visual del radar;
- actualización de esta ficha técnica;
- independencia de `2_Assault/`.

### Pendiente para cierre formal

1. **AV14 — ejecución completa en Google Colab limpio.** La evidencia guardada actualmente corresponde a ejecución local.
2. **Conclusiones del Experimento 0 dentro del notebook.** La sección existe, pero en el notebook auditado permanece como plantilla con instrucciones para completarla, no como conclusiones derivadas de los resultados ejecutados.

Por lo anterior, el estado correcto es:

**HU002 IMPLEMENTADA — pendiente de completar las conclusiones del notebook y de AV14 en Google Colab.**

No debe marcarse como CERRADA/COMPLETADA hasta resolver ambos puntos y actualizar la evidencia.

---

## 28. Implicaciones MLOps

BattleZone no utilizará MLflow.

Las fases posteriores utilizarán, según `lineamientos.md`:

- Git/GitHub como fuente de verdad;
- configuración versionada;
- `run_id` y `run_manifest.json`;
- TensorBoard para entrenamiento;
- checkpoints y resume;
- resultados persistidos.

HU002 es un baseline exploratorio y no utiliza TensorBoard, checkpoints ni `run_manifest`.