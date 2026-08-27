# Ficha técnica del entorno Assault

## 1. Objetivo

Documentar las características del entorno `ALE/Assault-v5` relevantes para diseñar, entrenar y evaluar un agente de aprendizaje por refuerzo profundo dentro del Reto 1.

Esta ficha constituye la fase inicial de exploración del entorno (EDA para Reinforcement Learning). Separa la información conocida por documentación oficial de aquella obtenida empíricamente mediante la primera ejecución del Experimento 0.

## 2. Fuentes

- Documentación oficial de Arcade Learning Environment (ALE): https://ale.farama.org/environments/assault/
- Enunciado del Reto 1 incluido en este repositorio.
- `2_Assault/experimento_0_assault.ipynb`, primera ejecución del Experimento 0.

## 3. Descripción del problema

Assault es un videojuego Atari en el que el agente controla un vehículo ubicado en la parte inferior de la pantalla. El vehículo puede desplazarse lateralmente y disparar. Una nave principal enemiga circula en la parte superior y despliega continuamente drones; el objetivo del agente es destruir enemigos y evitar sus ataques.

Desde la perspectiva de RL, el agente debe aprender simultáneamente comportamiento ofensivo (disparar y posicionarse para destruir enemigos) y defensivo (evitar amenazas), a partir exclusivamente de observaciones visuales si utilizamos la configuración RGB estándar.

## 4. Identificación del entorno

| Característica | Valor |
|---|---|
| Entorno recomendado | `ALE/Assault-v5` |
| Creación | `gymnasium.make("ALE/Assault-v5")` |
| Familia | Atari / Arcade Learning Environment |
| Espacio de acciones | `Discrete(7)` |
| Observación por defecto | RGB |
| Forma de observación | `(210, 160, 3)` |
| Tipo de dato | `uint8` |
| Rango teórico de píxel | `[0, 255]` |
| Rango observado en el frame inicial del Experimento 0 | `0` a `214` |
| Frameskip de `ALE/Assault-v5` | `4` |
| Repeat action probability | `0.25` |
| Mode disponible / default | `0` / `0` |
| Difficulty disponible / default | `0` / `0` |
| Vidas iniciales observadas | `4` |

### Configuración validada en la primera ejecución

La primera ejecución del Experimento 0 confirmó empíricamente:

- Python `3.8.10`;
- Gymnasium `1.1.1`;
- ALE-Py `0.10.1`;
- observación RGB de forma `(210, 160, 3)` y tipo `uint8`;
- espacio de acciones `Discrete(7)`;
- `frameskip=4`;
- `repeat_action_probability=0.25`;
- seed base `42`.

## 5. Espacio de acciones

El espacio mínimo de acciones contiene siete acciones discretas:

| Índice | Acción |
|---:|---|
| 0 | `NOOP` |
| 1 | `FIRE` |
| 2 | `UP` |
| 3 | `RIGHT` |
| 4 | `LEFT` |
| 5 | `RIGHTFIRE` |
| 6 | `LEFTFIRE` |

La primera ejecución del Experimento 0 confirmó que estos siete significados son los retornados por `env.unwrapped.get_action_meanings()`.

ALE permite habilitar las 18 acciones posibles del Atari 2600 mediante `full_action_space=True`. Para este proyecto se recomienda inicialmente conservar el conjunto mínimo de 7 acciones, ya que contiene las acciones útiles identificadas por ALE y reduce innecesariamente la dimensionalidad del problema.

## 6. Espacio de observaciones

ALE ofrece tres representaciones posibles:

| `obs_type` | Espacio |
|---|---|
| `rgb` | `Box(0, 255, (210, 160, 3), uint8)` |
| `grayscale` | `Box(0, 255, (210, 160), uint8)` |
| `ram` | `Box(0, 255, (128,), uint8)` |

`ALE/Assault-v5` utiliza RGB por defecto.

La primera ejecución confirmó que una observación RGB real tiene:

- `shape=(210, 160, 3)`;
- `dtype=uint8`;
- valor mínimo observado `0`;
- valor máximo observado `214` en el frame inicial inspeccionado.

El rango `0-214` corresponde únicamente al frame observado durante la ejecución y no reemplaza el rango teórico `0-255` definido por el espacio de observación.

Para el reto conviene partir de imágenes, coherente con la dificultad descrita en el enunciado. Sin embargo, entregar directamente imágenes RGB de 210x160x3 a una red neuronal sería computacionalmente costoso. En la etapa de diseño deberán evaluarse preprocesamientos estándar de Atari, particularmente reducción de resolución, escala de grises y apilamiento de frames.

El apilamiento temporal es especialmente relevante porque una sola imagen no permite inferir directamente velocidad ni dirección de movimiento de enemigos, proyectiles o de la nave.

## 7. Dinámica temporal y estocasticidad

En `ALE/Assault-v5`:

- `frameskip=4`: una acción seleccionada se aplica durante cuatro frames del emulador.
- `repeat_action_probability=0.25`: existe un 25 % de probabilidad de repetir la acción previa en lugar de ejecutar exactamente la nueva acción solicitada.

Esto convierte el entorno en estocástico y evita asumir una correspondencia completamente determinista entre acción y siguiente estado.

La configuración debe mantenerse fija y registrada para asegurar comparabilidad y reproducibilidad entre experimentos.

## 8. Información expuesta por `info`

La primera ejecución del Experimento 0 permitió descubrir información adicional que no estaba detallada en la ficha original.

En `reset()` se observó:

```python
{
    'lives': 4,
    'episode_frame_number': 0,
    'frame_number': 0,
    'seeds': (3444837047, 2669555309)
}
```

Esto confirma que ALE expone directamente información útil para observabilidad del entorno:

| Variable | Interpretación práctica |
|---|---|
| `lives` | permite registrar y detectar pérdidas de vida sin inferirlas desde la imagen |
| `episode_frame_number` | permite seguir el avance temporal dentro del episodio |
| `frame_number` | permite observar el contador global de frames reportado por ALE |
| `seeds` | expone semillas internas utilizadas por ALE |

Estas variables pueden reutilizarse posteriormente para diagnóstico, evaluación y callbacks de entrenamiento, sin necesidad de reconstruir esta información a partir de observaciones visuales.

## 9. Recompensas

La página oficial de ALE consultada no documenta la tabla exacta de recompensas ni los valores otorgados por cada tipo de enemigo o evento.

El enunciado del reto establece que el objetivo es destruir la mayor cantidad posible de enemigos y maximizar la recompensa acumulada.

El Experimento 0 fue diseñado para medir empíricamente la distribución de recompensas mediante una política aleatoria. La primera ejecución ya permitió validar la instrumentación necesaria para registrar recompensas por step y por episodio, pero las métricas agregadas completas deben mantenerse como evidencia del notebook antes de incorporarlas como valores definitivos en esta ficha.

No se deben asumir valores específicos sin evidencia explícita de la ejecución.

## 10. Terminación del episodio y vidas

La primera ejecución resolvió parcialmente una de las incógnitas originales:

- `lives` está disponible en `info`;
- el valor inicial observado fue de **4 vidas**;
- `episode_frame_number` y `frame_number` están disponibles para estudiar duración y dinámica temporal.

Todavía deben consolidarse a partir de los resultados completos del Experimento 0:

- comportamiento exacto de las pérdidas de vida durante los episodios;
- relación entre pérdida de la última vida y `terminated`;
- presencia o ausencia de `truncated`;
- duración típica de una partida;
- cantidad media de pérdidas de vida por episodio.

## 11. Dificultades de aprendizaje identificadas

### 11.1 Observación de alta dimensionalidad

Cada estado RGB contiene 210 x 160 x 3 valores. El agente deberá extraer automáticamente características visuales relevantes.

### 11.2 Dependencia temporal

Una imagen individual muestra posiciones, pero no describe completamente movimiento. Para decidir si debe esquivar o disparar es necesario recuperar información temporal.

### 11.3 Múltiples objetivos simultáneos

El agente debe considerar su posición, enemigos, proyectiles y amenazas simultáneamente.

### 11.4 Equilibrio ataque-defensa

Disparar constantemente puede no ser suficiente: el agente debe aprender cuándo atacar y cuándo desplazarse para sobrevivir.

### 11.5 Entorno estocástico

La probabilidad de repetición de acción de 0.25 introduce variabilidad incluso ante políticas idénticas.

### 11.6 Costo computacional

El aprendizaje desde imágenes requiere una red convolucional y un volumen elevado de interacciones. El enunciado recomienda explícitamente GPU y Google Colab.

## 12. Información disponible para seleccionar algoritmo

La documentación oficial y la primera ejecución permiten establecer varias restricciones importantes del diseño:

1. Las acciones son discretas y pocas (`Discrete(7)`).
2. El estado es visual y de alta dimensionalidad.
3. Existe dependencia temporal entre frames.
4. El entorno es estocástico.
5. ALE expone vidas y contadores de frames, mejorando la observabilidad durante entrenamiento y evaluación.
6. El entrenamiento será costoso y requerirá reutilización eficiente de experiencias.

Estas propiedades favorecen preliminarmente métodos value-based compatibles con acciones discretas, como DQN, DQN + Prioritized Experience Replay o DDQN, todos permitidos por el enunciado.

La selección final del algoritmo debe apoyarse también en las métricas agregadas del baseline aleatorio, especialmente dispersión y densidad de recompensas.

## 13. Baseline y métricas del proyecto

### 13.1 Métrica principal exigida por el enunciado

Para Assault, el enunciado no define un puntaje absoluto mínimo para considerar el entorno resuelto. El criterio obligatorio consiste en evaluar el agente en **al menos 10 partidas independientes** y reportar el **puntaje o recompensa promedio** obtenido.

Por lo tanto, la métrica principal del proyecto será:

**Recompensa promedio del agente sobre al menos 10 episodios independientes de evaluación.**

Esta será la métrica utilizada para comparar versiones del agente y para reportar el desempeño final del reto.

### 13.2 Baseline oficial del proyecto

El baseline será una **política completamente aleatoria** ejecutada sobre el mismo entorno y bajo un protocolo de evaluación equivalente al utilizado para el agente entrenado.

El baseline debe calcularse sobre al menos 10 episodios independientes y registrar como mínimo:

- recompensa promedio;
- mediana;
- desviación estándar;
- recompensa mínima;
- recompensa máxima.

El resultado del agente entrenado se comparará contra este baseline para evidenciar que aprendió un comportamiento superior a seleccionar acciones aleatoriamente.

### 13.3 Criterio interno de éxito

Dado que el enunciado no define un umbral numérico absoluto para Assault, el proyecto no establecerá artificialmente uno.

El criterio interno mínimo será:

**La recompensa promedio del agente entrenado debe ser superior a la recompensa promedio del baseline aleatorio, evaluando ambos con el mismo protocolo.**

Además, el comportamiento observado en el video debe ser coherente con una política aprendida y no con acciones predominantemente aleatorias.

### 13.4 Métricas secundarias de evaluación

Estas métricas no reemplazan la métrica principal, pero ayudan a interpretar la calidad y estabilidad del agente:

| Métrica | Propósito |
|---|---|
| Mediana de recompensa | reducir el efecto de episodios excepcionalmente altos o bajos |
| Desviación estándar | medir estabilidad entre episodios |
| Recompensa mínima y máxima | identificar variabilidad y casos extremos |
| Steps o frames por episodio | observar supervivencia y duración de las partidas |
| Pérdidas de vida por episodio | analizar comportamiento defensivo |
| Porcentaje de steps con recompensa positiva | medir densidad de eventos exitosos |
| Tiempo total de entrenamiento | cumplir el análisis requerido por el reporte técnico |

### 13.5 Protocolo de comparación

Para que las comparaciones sean válidas, baseline y agente entrenado deberán evaluarse manteniendo constante:

- `ALE/Assault-v5`;
- configuración del entorno;
- espacio de acciones;
- `frameskip`;
- `repeat_action_probability`;
- número mínimo de episodios de evaluación;
- política sin exploración adicional durante la evaluación del agente entrenado, salvo que el algoritmo requiera explícitamente otro comportamiento y se documente.

Los resultados de entrenamiento y evaluación deben mantenerse separados. La recompensa observada durante entrenamiento sirve para analizar aprendizaje; la recompensa promedio de los episodios independientes de evaluación será la métrica utilizada para reportar el desempeño final.

## 14. Estado del EDA

### Información ya cubierta

- identificador del entorno;
- espacio de observaciones;
- tipos posibles de observación;
- espacio y significado de las acciones;
- `frameskip`;
- probabilidad de repetición de acción;
- modos y dificultades disponibles;
- descripción general de la dinámica del juego;
- forma y tipo real de una observación RGB;
- rango de píxeles observado en el frame inicial;
- vidas iniciales observadas: `4`;
- claves disponibles en `info`: `lives`, `episode_frame_number`, `frame_number`, `seeds`;
- versiones principales utilizadas en la primera ejecución;
- métrica principal del proyecto;
- baseline de comparación;
- métricas secundarias y protocolo de evaluación.

### Información pendiente de consolidar con el Experimento 0

1. valor cuantitativo final del baseline aleatorio: media, mediana, desviación, mínimo y máximo;
2. duración media y rango de episodios en steps y frames;
3. comportamiento de pérdidas de vida durante los episodios;
4. densidad de recompensas positivas, cero y negativas;
5. relación entre `terminated`, `truncated` y pérdida de vidas;
6. dimensiones después del preprocesamiento definitivo;
7. consumo real de RAM/VRAM y velocidad de interacción durante el futuro entrenamiento en Colab con GPU.

## 15. Implicaciones MLOps iniciales

Desde esta fase deben registrarse como configuración versionada:

- environment ID;
- versión de Gymnasium/ALE;
- tipo de observación;
- action space utilizado;
- seed(s);
- wrappers/preprocesamiento;
- frameskip;
- repeat action probability;
- hardware de ejecución;
- variables de observabilidad disponibles en `info`;
- protocolo de evaluación;
- baseline utilizado;
- métricas principales y secundarias.

Esto permitirá que los experimentos posteriores en Colab, TensorBoard y MLflow sean comparables y reproducibles.
