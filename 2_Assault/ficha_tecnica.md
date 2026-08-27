# Ficha técnica del entorno Assault

## 1. Objetivo

Documentar las características del entorno `ALE/Assault-v5` relevantes para diseñar, entrenar y evaluar un agente de aprendizaje por refuerzo profundo dentro del Reto 1.

Esta ficha constituye la fase inicial de exploración del entorno (EDA para Reinforcement Learning). Separa la información conocida por documentación oficial de aquella que debe obtenerse empíricamente ejecutando el entorno.

## 2. Fuentes

- Documentación oficial de Arcade Learning Environment (ALE): https://ale.farama.org/environments/assault/
- Enunciado del Reto 1 incluido en este repositorio.

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
| Rango de píxel | `[0, 255]` |
| Frameskip de `ALE/Assault-v5` | `4` |
| Repeat action probability | `0.25` |
| Mode disponible / default | `0` / `0` |
| Difficulty disponible / default | `0` / `0` |

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

ALE permite habilitar las 18 acciones posibles del Atari 2600 mediante `full_action_space=True`. Para este proyecto se recomienda inicialmente conservar el conjunto mínimo de 7 acciones, ya que contiene las acciones útiles identificadas por ALE y reduce innecesariamente la dimensionalidad del problema.

## 6. Espacio de observaciones

ALE ofrece tres representaciones posibles:

| `obs_type` | Espacio |
|---|---|
| `rgb` | `Box(0, 255, (210, 160, 3), uint8)` |
| `grayscale` | `Box(0, 255, (210, 160), uint8)` |
| `ram` | `Box(0, 255, (128,), uint8)` |

`ALE/Assault-v5` utiliza RGB por defecto.

Para el reto conviene partir de imágenes, coherente con la dificultad descrita en el enunciado. Sin embargo, entregar directamente imágenes RGB de 210x160x3 a una red neuronal sería computacionalmente costoso. En la etapa de diseño deberán evaluarse preprocesamientos estándar de Atari, particularmente reducción de resolución, escala de grises y apilamiento de frames.

El apilamiento temporal es especialmente relevante porque una sola imagen no permite inferir directamente velocidad ni dirección de movimiento de enemigos, proyectiles o de la nave.

## 7. Dinámica temporal y estocasticidad

En `ALE/Assault-v5`:

- `frameskip=4`: una acción seleccionada se aplica durante cuatro frames del emulador.
- `repeat_action_probability=0.25`: existe un 25 % de probabilidad de repetir la acción previa en lugar de ejecutar exactamente la nueva acción solicitada.

Esto convierte el entorno en estocástico y evita asumir una correspondencia completamente determinista entre acción y siguiente estado.

La configuración debe mantenerse fija y registrada para asegurar comparabilidad y reproducibilidad entre experimentos.

## 8. Recompensas

La página oficial de ALE consultada no documenta la tabla exacta de recompensas ni los valores otorgados por cada tipo de enemigo o evento.

El enunciado del reto establece que el objetivo es destruir la mayor cantidad posible de enemigos y maximizar la recompensa acumulada.

Por lo tanto, la distribución real de recompensas debe medirse empíricamente durante la exploración del entorno. No se deben asumir valores específicos sin evidencia de ejecución o documentación adicional.

## 9. Terminación del episodio y vidas

La página específica de Assault de ALE no proporciona suficiente detalle para documentar con precisión:

- número de vidas iniciales;
- eventos exactos que provocan pérdida de vida;
- condición exacta de terminación (`terminated`);
- posibles truncamientos (`truncated`);
- duración típica de una partida.

Estos elementos deben medirse ejecutando el entorno antes del entrenamiento definitivo.

## 10. Dificultades de aprendizaje identificadas

### 10.1 Observación de alta dimensionalidad

Cada estado RGB contiene 210 x 160 x 3 valores. El agente deberá extraer automáticamente características visuales relevantes.

### 10.2 Dependencia temporal

Una imagen individual muestra posiciones, pero no describe completamente movimiento. Para decidir si debe esquivar o disparar es necesario recuperar información temporal.

### 10.3 Múltiples objetivos simultáneos

El agente debe considerar su posición, enemigos, proyectiles y amenazas simultáneamente.

### 10.4 Equilibrio ataque-defensa

Disparar constantemente puede no ser suficiente: el agente debe aprender cuándo atacar y cuándo desplazarse para sobrevivir.

### 10.5 Entorno estocástico

La probabilidad de repetición de acción de 0.25 introduce variabilidad incluso ante políticas idénticas.

### 10.6 Costo computacional

El aprendizaje desde imágenes requiere una red convolucional y un volumen elevado de interacciones. El enunciado recomienda explícitamente GPU y Google Colab.

## 11. Información suficiente para seleccionar algoritmo

La documentación oficial sí proporciona información suficiente para establecer varias restricciones importantes del diseño:

1. Las acciones son discretas y pocas (`Discrete(7)`).
2. El estado es visual y de alta dimensionalidad.
3. Existe dependencia temporal entre frames.
4. El entorno es estocástico.
5. El entrenamiento será costoso y requerirá reutilización eficiente de experiencias.

Estas propiedades favorecen preliminarmente métodos value-based compatibles con acciones discretas, como DQN, DQN + Prioritized Experience Replay o DDQN, todos permitidos por el enunciado.

La selección final del algoritmo debe realizarse después de completar el EDA empírico y construir el baseline aleatorio.

## 12. ¿Es suficiente la documentación oficial para completar el EDA?

**No completamente.** La documentación de ALE es suficiente para caracterizar la interfaz formal del entorno y diseñar la exploración, pero no sustituye una ejecución empírica.

### Información ya cubierta

- identificador del entorno;
- espacio de observaciones;
- tipos posibles de observación;
- espacio y significado de las acciones;
- `frameskip`;
- probabilidad de repetición de acción;
- modos y dificultades disponibles;
- descripción general de la dinámica del juego.

### Información pendiente de EDA empírico

Antes del entrenamiento deben medirse como mínimo:

1. recompensa media, desviación y distribución de una política aleatoria;
2. recompensa mínima y máxima observada;
3. duración de episodios en steps y frames;
4. número y comportamiento de las vidas;
5. frecuencia de recompensas positivas y presencia de recompensas nulas/negativas;
6. causas observadas de `terminated` y `truncated`;
7. comportamiento real del espacio de acciones mediante episodios de prueba;
8. dimensiones luego del preprocesamiento seleccionado;
9. consumo aproximado de RAM/VRAM y velocidad de interacción en Colab;
10. baseline sobre al menos 10 episodios, alineado con el criterio de evaluación del reto.

## 13. Próximo paso recomendado

Crear un notebook o módulo de exploración reproducible que ejecute una política completamente aleatoria sobre un número controlado de episodios y registre:

- seed;
- recompensa por episodio;
- longitud del episodio;
- acciones ejecutadas;
- recompensas por step;
- vidas reportadas por el entorno;
- `terminated` y `truncated`;
- estadísticos descriptivos;
- algunos frames de referencia.

El resultado constituirá el **baseline experimental** contra el cual comparar posteriormente el agente entrenado y permitirá cerrar formalmente la etapa de EDA antes de seleccionar el algoritmo definitivo.

## 14. Implicaciones MLOps iniciales

Desde esta fase deben registrarse como configuración versionada:

- environment ID;
- versión de Gymnasium/ALE;
- tipo de observación;
- action space utilizado;
- seed(s);
- wrappers/preprocesamiento;
- frameskip;
- repeat action probability;
- hardware de ejecución.

Esto permitirá que los experimentos posteriores en Colab, TensorBoard y MLflow sean comparables y reproducibles.
