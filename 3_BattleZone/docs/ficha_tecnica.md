# Ficha técnica del entorno BattleZone

## 1. Objetivo

Documentar las características de `ALE/BattleZone-v5` relevantes para diseñar, implementar, entrenar y evaluar un agente de Reinforcement Learning dentro del Reto 1.

Esta ficha constituye la caracterización inicial del entorno y debe evolucionar con evidencia empírica obtenida en HU002 — Experimento 0 y baseline aleatorio.

La información se clasifica en:

1. **Hechos documentados:** provenientes de Arcade Learning Environment (ALE) y documentación del juego.
2. **Decisiones iniciales del proyecto:** acuerdos técnicos sujetos a validación mediante las HUs.
3. **Información pendiente:** datos que deben medirse empíricamente antes de convertirse en verdad del proyecto.

BattleZone se implementará de forma completamente independiente de Assault. El trabajo previo de Assault se utiliza únicamente como referencia metodológica y base de conocimiento. No se copiará, importará ni reutilizará código desde `2_Assault/`.

---

## 2. Fuentes

### Fuente académica principal

- `enunciado_reto_1.txt`.

### Fuentes técnicas principales

- Arcade Learning Environment — BattleZone: https://ale.farama.org/environments/battle_zone/
- Arcade Learning Environment — Atari Environments: https://ale.farama.org/environments/
- Manual Atari 2600 BattleZone publicado por AtariAge: https://www.atariage.com/manual_html_page.php?SoftwareID=859&SystemID=2600&itemTypeID=HTMLMANUAL

### Documentación interna del proyecto

- `3_BattleZone/docs/implementacion.md`.
- `3_BattleZone/docs/lineamientos.md`.
- `3_BattleZone/docs/arquitectura.md`.

---

## 3. Descripción del problema

BattleZone es un videojuego Atari en primera persona donde el agente controla un tanque dentro de un entorno que simula profundidad tridimensional mediante gráficos vectoriales.

El objetivo es destruir vehículos enemigos, sobrevivir a sus ataques y maximizar la recompensa acumulada.

El juego combina simultáneamente:

- navegación;
- orientación espacial;
- adquisición de blancos;
- disparo;
- evasión;
- uso de radar;
- interpretación temporal del movimiento;
- interacción con obstáculos.

El agente no recibe directamente coordenadas semánticas de enemigos, obstáculos o radar. Bajo la configuración visual del reto deberá aprender estas relaciones a partir de píxeles.

---

## 4. Restricciones académicas

El Reto 1 permite únicamente los siguientes métodos:

- DQN;
- DQN + Prioritized Experience Replay;
- DDQN;
- REINFORCE.

La selección formal del algoritmo se realizará en HU004 después de completar:

- HU001 — caracterización técnica;
- HU002 — Experimento 0 y baseline aleatorio;
- HU003 — pipeline reproducible del entorno.

Para BattleZone no se exige alcanzar un puntaje humano u óptimo. El entregable debe demostrar que el agente aprendió un comportamiento lógico y no predominantemente aleatorio.

La evaluación formal deberá utilizar al menos **10 partidas independientes**.

---

## 5. Identificación oficial del entorno

| Característica | Valor documentado |
|---|---|
| Familia | Atari / Arcade Learning Environment |
| Environment ID | `ALE/BattleZone-v5` |
| Creación | `gymnasium.make("ALE/BattleZone-v5")` |
| Action space | `Discrete(18)` |
| Observación por defecto | RGB |
| Observation space | `Box(0, 255, (210, 160, 3), uint8)` |
| `frameskip` v5 | `4` |
| `repeat_action_probability` v5 | `0.25` |
| `full_action_space` | no cambia BattleZone: el juego utiliza las 18 acciones |
| Modes disponibles | `[1, 2, 3]` |
| Mode por defecto | `1` |
| Difficulties disponibles | `[0]` |
| Difficulty por defecto | `0` |
| Vidas iniciales documentadas por ALE | `5` |

ALE indica además que el jugador puede obtener hasta dos vidas adicionales al alcanzar suficiente puntuación.

---

## 6. Espacio de acciones

BattleZone utiliza las **18 acciones Atari completas**. A diferencia de muchos juegos ALE, habilitar `full_action_space=True` no aumenta el espacio de acciones.

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

El espacio de acciones es considerablemente mayor que el de Assault y aumenta:

- el número de Q-values a estimar en algoritmos value-based;
- la complejidad de exploración;
- la probabilidad de acciones poco útiles durante una política aleatoria;
- el tiempo necesario para descubrir combinaciones de movimiento y disparo coherentes.

No se reducirá artificialmente el action space antes de HU002/HU004 sin evidencia empírica y justificación técnica.

---

## 7. Espacio de observaciones

ALE soporta tres tipos de observación para BattleZone.

| `obs_type` | Espacio |
|---|---|
| `rgb` | `Box(0,255,(210,160,3),uint8)` |
| `grayscale` | `Box(0,255,(210,160),uint8)` |
| `ram` | `Box(0,255,(128,),uint8)` |

La configuración por defecto `ALE/BattleZone-v5` utiliza RGB.

### Decisión inicial del proyecto

El proyecto partirá de observaciones visuales, coherente con el enunciado y con la naturaleza perceptual del problema.

El pipeline definitivo de preprocessing no se fijará copiando el utilizado en Assault. HU001-HU003 deberán validar específicamente para BattleZone:

- escala de grises vs RGB;
- resolución objetivo;
- frame stacking;
- normalización;
- pérdida potencial de información relevante del radar y de blancos pequeños.

### Hipótesis inicial

Una representación compacta basada en resize y frame stack es razonable para controlar memoria y costo computacional, pero debe comprobarse que conserva señales útiles del radar y del campo visual.

---

## 8. Dinámica temporal y estocasticidad

Para `ALE/BattleZone-v5`:

- `frameskip=4`;
- `repeat_action_probability=0.25`.

Esto significa que cada decisión del agente corresponde normalmente a cuatro frames del emulador y existe una probabilidad del 25 % de repetir la acción anterior en lugar de ejecutar la acción recién seleccionada.

### Implicaciones

1. El entorno no debe tratarse como determinista desde la perspectiva del agente.
2. El `frameskip` efectivo debe aplicarse **una sola vez**.
3. Entrenamiento, baseline y evaluación deben mantener configuraciones equivalentes.
4. Una única imagen no captura velocidad ni dirección de movimiento.
5. Frame stacking u otro mecanismo temporal debe evaluarse antes de fijar la arquitectura de entrada.

---

## 9. Radar y percepción parcial

BattleZone incorpora un radar en la parte superior de la pantalla que permite localizar amenazas que pueden no estar directamente visibles en la vista frontal.

Esto hace que la observación visual contenga dos fuentes complementarias de información:

1. **vista principal en primera persona**, útil para apuntar, percibir enemigos y obstáculos;
2. **radar**, útil para orientación espacial y detección de amenazas fuera del campo visual directo.

### Implicación de diseño

El preprocessing debe evitar destruir información pequeña o de alto contraste del radar.

Reducir excesivamente la resolución o recortar la zona superior podría eliminar una señal estratégica crítica.

Cualquier cropping deberá considerarse una decisión experimental explícita y no un default heredado de otros juegos Atari.

---

## 10. Movimiento y control

El tanque puede:

- avanzar;
- retroceder;
- girar a izquierda o derecha;
- desplazarse en combinaciones diagonales/arcadas;
- disparar;
- combinar movimiento y disparo.

Las combinaciones movimiento+fuego explican buena parte del action space de 18 acciones.

### Implicación para la política

Una política útil debe aprender al menos parte de los siguientes comportamientos:

- orientar el tanque hacia enemigos;
- disparar cuando existe oportunidad de impacto;
- moverse o girar ante amenazas;
- evitar permanecer inmóvil de forma sistemática;
- utilizar combinaciones de movimiento y fuego cuando sean ventajosas.

El reto académico no exige perfección, pero estos patrones permiten evaluar cualitativamente si emerge comportamiento lógico.

---

## 11. Enemigos y objetivos

La documentación del juego describe distintos tipos de objetivos/enemigos, entre ellos:

- tanque estándar;
- fighter/aerial fighter o misil según la versión/documentación;
- supertank;
- flying saucer.

El manual Atari 2600 reporta la siguiente tabla de puntuación:

| Objetivo | Puntaje documentado |
|---|---:|
| Tank | 1,000 |
| Fighter | 2,000 |
| Supertank | 3,000 |
| Saucer | 5,000 |

ALE únicamente documenta de forma general que se reciben puntos por destruir enemigos.

### Regla de evidencia

La equivalencia exacta entre estos valores del manual y los rewards observados mediante la API Gymnasium/ALE deberá validarse empíricamente en HU002 antes de usar esta tabla como verdad cuantitativa del entrenamiento.

---

## 12. Obstáculos

El escenario contiene objetos geométricos que pueden influir en:

- movimiento;
- línea de visión;
- trayectoria de disparos;
- evasión;
- posicionamiento relativo frente a enemigos.

Desde RL, los obstáculos agregan una capa de navegación que no puede resolverse únicamente mediante una política de "disparar siempre".

HU002 deberá observar si la política aleatoria presenta bloqueos, giros prolongados, colisiones o patrones que permitan definir métricas adicionales útiles.

---

## 13. Recompensas

La página oficial de ALE indica que el jugador recibe puntos por destruir enemigos, pero no expone en su ficha la tabla exacta de rewards retornados por cada evento.

### Métrica oficial del proyecto

La métrica principal será:

**Recompensa promedio obtenida por el agente en al menos 10 episodios independientes de evaluación.**

### Baseline

La referencia mínima será una política completamente aleatoria ejecutada bajo el mismo entorno y protocolo.

### Reward clipping

Si el algoritmo seleccionado utiliza reward clipping durante entrenamiento, la evaluación formal deberá calcularse con la recompensa real del entorno para mantener comparabilidad con el baseline.

Cualquier clipping debe quedar documentado como transformación de entrenamiento y nunca confundirse con la métrica oficial.

---

## 14. Vidas y finalización

ALE documenta:

- 5 vidas iniciales;
- posibilidad de obtener hasta 2 vidas extra con suficiente puntuación.

El juego termina cuando se agotan las vidas.

### Pendiente de validación empírica

HU002 deberá verificar:

- valor de `info["lives"]` en `reset()` y durante `step()`;
- relación exacta entre pérdida de la última vida y `terminated=True`;
- existencia y causas de `truncated=True`;
- duración típica de episodios;
- cambios de recompensa alrededor de pérdida de vida;
- comportamiento al conseguir vidas extra.

No se utilizará pérdida de vida como terminación artificial durante el entrenamiento salvo decisión explícita posterior.

---

## 15. `info` y observabilidad del entorno

ALE suele exponer metadatos útiles como vidas y contadores de frames, pero la ficha oficial de BattleZone no enumera exhaustivamente el contenido real de `info` para la versión instalada.

HU002 deberá inspeccionar directamente `info` en:

- `reset()`;
- pasos normales;
- eventos con recompensa;
- pérdida de vida;
- terminación.

La documentación interna solo incorporará como hechos aquellas claves verificadas en ejecución.

---

## 16. Modes y difficulty

La documentación oficial indica:

- modes disponibles: `[1, 2, 3]`;
- mode default: `1`;
- difficulties disponibles: `[0]`;
- difficulty default: `0`.

El proyecto utilizará inicialmente los valores default para evitar introducir una variable adicional durante el baseline.

Cualquier cambio de mode constituye un experimento diferente y debe quedar versionado en configuración y manifiesto de ejecución.

---

## 17. Principales dificultades de aprendizaje

### 17.1 Observación visual de alta dimensionalidad

Cada frame RGB contiene 100,800 valores (`210×160×3`). Una red debe aprender características útiles sin supervisión directa.

### 17.2 Espacio de acciones grande

`Discrete(18)` incrementa el costo de exploración y la dificultad de estimar acciones útiles.

### 17.3 Dependencia temporal

Posición instantánea no describe velocidad, trayectoria o dirección relativa. El agente necesita contexto temporal.

### 17.4 Información distribuida en la pantalla

Radar y vista principal contienen información complementaria separada espacialmente.

### 17.5 Perspectiva en primera persona

El agente debe inferir orientación y profundidad aproximada desde píxeles.

### 17.6 Equilibrio ataque-defensa

El reward proviene del combate, pero sobrevivir permite acceder a más oportunidades de recompensa futura.

### 17.7 Sticky actions

La acción ejecutada puede diferir de la intención actual de la política.

### 17.8 Recompensa potencialmente escasa

La densidad real de eventos con reward debe determinarse en HU002. Si resulta baja, la eficiencia muestral será un criterio importante en HU004.

---

## 18. Información para selección de algoritmo

La caracterización inicial permite afirmar:

1. El espacio de acciones es discreto.
2. Existen 18 acciones.
3. El estado visual es de alta dimensionalidad.
4. El entorno requiere razonamiento temporal.
5. El entrenamiento será computacionalmente costoso.
6. El reto permite DQN, DQN+PER, DDQN y REINFORCE.

No se seleccionará definitivamente el algoritmo hasta contar con las métricas del Experimento 0.

### Variables que HU004 deberá ponderar

- densidad de recompensa;
- dispersión del baseline;
- longitud de episodios;
- frecuencia de acciones efectivamente útiles;
- tamaño de Replay Buffer viable;
- estabilidad del aprendizaje;
- costo GPU;
- riesgo de sobreestimación de Q-values;
- beneficio potencial de priorizar experiencias raras;
- costo de implementación dentro del tiempo académico.

---

## 19. Baseline y protocolo de evaluación

HU002 construirá una política aleatoria sobre al menos 10 episodios.

El baseline deberá registrar como mínimo:

### Recompensa

- promedio;
- mediana;
- desviación estándar;
- mínimo;
- máximo.

### Duración

- steps por episodio;
- frames cuando estén disponibles.

### Recompensa por step

- proporción de rewards positivos;
- rewards cero;
- rewards negativos si existen;
- cantidad media de eventos con reward por episodio.

### Vidas

- iniciales;
- pérdidas;
- vidas extra si ocurren;
- relación con terminación.

### Acciones

- frecuencia absoluta y relativa de las 18 acciones.

### Criterio de comparación

Baseline y agente entrenado deberán utilizar:

- mismo environment ID;
- mismo mode/difficulty;
- mismo `frameskip`;
- mismo `repeat_action_probability`;
- mismo pipeline de observaciones para la evaluación del agente;
- mismo número mínimo de episodios;
- rewards reales del entorno.

---

## 20. Evidencia cualitativa de aprendizaje

Debido a que el enunciado no fija un puntaje absoluto mínimo para BattleZone, además de la recompensa deberán observarse comportamientos como:

- reducción de acciones claramente erráticas;
- uso intencional de FIRE o acciones combinadas con FIRE;
- orientación hacia objetivos;
- cambios de dirección asociados a amenazas;
- mayor supervivencia si se observa de forma consistente;
- secuencias de acción más estructuradas que el baseline aleatorio.

Estas observaciones complementan, pero no sustituyen, la evaluación cuantitativa.

---

## 21. Preprocessing: decisiones abiertas

La arquitectura deberá soportar experimentación controlada con preprocessing sin acoplarlo a la red.

HU003 deberá resolver:

1. ¿RGB o grayscale?
2. ¿84×84 conserva información suficiente del radar?
3. ¿Se requiere una resolución mayor?
4. ¿Frame stack de 4 es suficiente?
5. ¿Conviene mantener píxeles `uint8` en memoria y normalizar al convertir a tensor?
6. ¿Debe evitarse cropping para conservar el radar?
7. ¿Cómo asegurar `frameskip` efectivo de 4 una sola vez?

Hasta entonces, ninguna combinación concreta se considera definitiva.

---

## 22. Riesgos técnicos iniciales

| Riesgo | Impacto | Mitigación prevista |
|---|---|---|
| Radar pierde legibilidad tras resize | Alto | validar visualmente y mediante smoke test |
| 18 acciones dificultan exploración | Alto | analizar acción/reward en HU002 y selección algorítmica en HU004 |
| Entrenamiento largo excede sesión Colab | Alto | checkpoints y resume obligatorios |
| Replay Buffer consume demasiada RAM | Alto | `uint8`, capacidad configurable, profiling |
| Doble frameskip | Alto | fábrica única + test explícito |
| Diferencias train/eval | Alto | misma fábrica y preprocessing |
| Corridas no reproducibles | Medio/alto | seed, commit, config, run manifest |
| Optimización sin evidencia | Medio | cambios por hipótesis y comparación controlada |
| Sobreingeniería | Medio | SOLID/DRY pragmáticos y DWP |

---

## 23. Preguntas abiertas para HU002

1. ¿Cuál es la recompensa promedio real de una política aleatoria?
2. ¿Qué tan alta es su dispersión?
3. ¿Qué porcentaje de steps entrega reward distinto de cero?
4. ¿Existen rewards negativos?
5. ¿Cuánto dura un episodio típico?
6. ¿Qué acciones aparecen asociadas a eventos positivos por azar?
7. ¿Cómo se comportan las vidas en `info`?
8. ¿Qué claves adicionales entrega `info`?
9. ¿Cómo se relacionan `terminated` y `truncated` con la dinámica real?
10. ¿El radar sigue siendo interpretable después de posibles resize/grayscale?
11. ¿Qué modes/configuración efectiva confirma la instalación seleccionada?
12. ¿Los valores de scoring documentados por el manual coinciden con los rewards ALE observados?

---

## 24. Estado de la ficha

### Confirmado por documentación oficial

- `ALE/BattleZone-v5`;
- `Discrete(18)`;
- significado de las 18 acciones;
- observación RGB `(210,160,3)` `uint8`;
- observaciones alternativas grayscale y RAM;
- `frameskip=4` en v5;
- `repeat_action_probability=0.25`;
- modes `[1,2,3]`, default `1`;
- difficulty `[0]`, default `0`;
- 5 vidas iniciales;
- posibilidad de obtener vidas adicionales;
- existencia de radar;
- recompensa asociada a destruir enemigos.

### Requiere validación empírica

- distribución de reward;
- equivalencia exacta score/reward;
- duración de episodios;
- comportamiento real de `info`;
- pérdida de vidas y terminación;
- preprocessing definitivo;
- resolución adecuada;
- densidad de recompensas;
- baseline aleatorio;
- consumo real de RAM/VRAM;
- throughput de interacción y entrenamiento.

---

## 25. Implicaciones MLOps iniciales

Toda corrida relevante deberá registrar:

- environment ID;
- mode/difficulty;
- `obs_type`;
- action space;
- `frameskip`;
- sticky actions;
- preprocessing;
- seed;
- algoritmo;
- hiperparámetros;
- versiones;
- hardware;
- commit Git;
- `run_id`;
- TensorBoard log path;
- checkpoint/modelo asociado;
- métricas de evaluación.

BattleZone no utilizará MLflow. La trazabilidad seguirá los lineamientos de `3_BattleZone/docs/lineamientos.md` mediante Git/GitHub, configuración versionada, `run_manifest.json`, TensorBoard, checkpoints y resultados persistidos.

---

## 26. Evidencia empírica observada (HU002 - Experimento 0 local)

Esta sección resume una corrida empírica del Experimento 0 ejecutada localmente con política estrictamente aleatoria (`env.action_space.sample()`) y **sin** clipping/shaping/normalización de reward.

### 26.1 Configuración de la corrida

- Environment ID: `ALE/BattleZone-v5`
- Episodios: `10`
- Seed base: `20260830` (seed por episodio: `base_seed + episode_id`)
- Mode: `1`
- Difficulty: `0`
- `obs_type`: `rgb`
- `frameskip`: `4`
- `repeat_action_probability`: `0.25`

### 26.2 Runtime observado

- Python: `3.8.10`
- Gymnasium: `1.1.1`
- ALE-Py: `0.10.1`
- NumPy: `1.24.4`
- Plataforma: `Windows-10-10.0.19044-SP0`
- CPU: `AMD64 Family 23 Model 17 Stepping 0, AuthenticAMD`
- RAM total: `6.9 GB`
- GPU disponible para la corrida: `False`

### 26.3 Contrato observado del entorno

- `observation_space`: `Box(0, 255, (210, 160, 3), uint8)`
- `action_space`: `Discrete(18)`
- Action meanings: 18 acciones Atari completas (`NOOP` ... `DOWNLEFTFIRE`)
- Observación inicial: shape `(210, 160, 3)`, dtype `uint8`, rango observado min/max `[0, 236]`

### 26.4 `info` observado

Claves observadas en `reset()`:

- `lives`
- `episode_frame_number`
- `frame_number`
- `seeds`

Claves observadas en `step()` (incluyendo eventos de reward no-cero, cambios de vida y terminación):

- `lives`
- `episode_frame_number`
- `frame_number`

En esta corrida no aparecieron claves adicionales fuera de las listadas.

### 26.5 Baseline aleatorio (10 episodios)

Métricas agregadas observadas:

- Recompensa media: `1300.0`
- Recompensa mediana: `1000.0`
- Desviación estándar: `1187.43`
- Recompensa mínima: `0.0`
- Recompensa máxima: `4000.0`
- Steps promedio por episodio: `1096.6`
- Steps min/max: `687 / 1450`
- Episodios `terminated=True`: `10`
- Episodios `truncated=True`: `0`

Densidad global de reward por step:

- Positive: `0.1094%`
- Zero: `99.8906%`
- Negative: `0.0%`
- Eventos no-cero promedio por episodio: `1.2`

Rewards observados por step:

- Valores únicos: `{0.0, 1000.0, 2000.0}`
- Frecuencias: `0.0 -> 10954`, `1000.0 -> 11`, `2000.0 -> 1`

Vidas y terminación:

- Vidas iniciales observadas: `{5}`
- Pérdidas promedio por episodio: `5.0`
- Vidas extra detectadas: `0`
- Episodios terminados con `lives_end == 0`: `10`

### 26.6 Lectura técnica de la evidencia

- En esta corrida, la recompensa fue muy escasa (casi todos los steps con reward cero).
- No se observaron rewards negativos.
- Se observaron valores de reward compatibles con el scoring histórico (1000 y 2000), pero esta evidencia no prueba por sí sola equivalencia completa para todos los objetivos/eventos.
- La alta proporción de reward cero y la varianza inter-episodio apoyan evaluar cuidadosamente eficiencia muestral y estabilidad en HU004.
- La presencia consistente de `lives`, `frame_number` y `episode_frame_number` en `info` es útil para observabilidad de HU003+.

### 26.7 Alcance y límites de esta evidencia

- Estos resultados corresponden a una corrida local específica de baseline aleatorio y no reemplazan futuras validaciones en Colab.
- La decisión de preprocessing definitivo (grayscale/RGB, resize, frame stack, posibles recortes) permanece abierta para HU003 y debe basarse en evidencia visual adicional del radar.