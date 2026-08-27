# HU001 — Experimento 0: Exploración empírica y baseline aleatorio de Assault

## 1. Contexto

La ficha técnica de `ALE/Assault-v5` documenta la interfaz formal del entorno, pero identifica información que debe obtenerse empíricamente antes de seleccionar definitivamente el algoritmo de aprendizaje por refuerzo. Entre los datos faltantes se encuentran la distribución real de recompensas, duración de episodios, comportamiento de vidas, frecuencia de recompensas, causas de terminación y baseline de una política aleatoria.

Este experimento corresponde a la fase inicial de exploración del entorno dentro del flujo MLOps del proyecto. Su propósito no es entrenar un agente, sino medir y documentar cómo se comporta Assault bajo una política completamente aleatoria.

## 2. Historia de usuario

**Como** equipo que desarrolla el agente de Reinforcement Learning para Assault,  
**quiero** ejecutar un notebook sencillo de exploración sobre `ALE/Assault-v5` usando una política aleatoria,  
**para** completar la información faltante del EDA, construir un baseline cuantitativo y obtener evidencia suficiente para orientar la selección entre DQN, DQN + Prioritized Experience Replay y DDQN.

## 3. Objetivo

Construir y ejecutar un notebook reproducible y de baja complejidad que observe el comportamiento real del entorno y produzca un resumen cuantitativo y cualitativo de las variables relevantes para el diseño posterior del agente.

## 4. Principios de implementación

El experimento debe priorizar simplicidad y observabilidad.

- No implementar entrenamiento.
- No implementar redes neuronales.
- No implementar Replay Buffer.
- No incorporar MLflow ni TensorBoard en esta HU salvo que aporten valor directo a la exploración.
- Evitar clases o abstracciones innecesarias.
- Preferir funciones pequeñas y código Python fácil de leer.
- Mantener parámetros principales agrupados en una única sección de configuración.
- Fijar y registrar seeds cuando sea posible.
- El notebook debe poder ejecutarse de principio a fin en Google Colab.
- El código debe documentarse con comentarios mínimos y docstrings estilo Google cuando se definan funciones reutilizables.

## 5. Alcance funcional

### 5.1 Configuración del entorno

El notebook debe:

1. instalar únicamente las dependencias necesarias para ejecutar `ALE/Assault-v5`;
2. mostrar las versiones principales utilizadas;
3. crear el entorno con una configuración explícita;
4. registrar como mínimo:
   - environment ID;
   - `obs_type`;
   - espacio de observaciones;
   - espacio de acciones;
   - nombres de las acciones disponibles;
   - `frameskip`;
   - `repeat_action_probability` cuando pueda consultarse;
   - seed(s) utilizadas;
   - CPU/GPU disponible y memoria relevante reportada por Colab.

### 5.2 Inspección inicial

Antes de ejecutar múltiples episodios, el notebook debe mostrar o imprimir:

- forma (`shape`) de una observación;
- tipo de dato;
- valores mínimo y máximo observados en los píxeles;
- contenido del diccionario `info` retornado por `reset()` y por algunos `step()`;
- un frame inicial del juego;
- cualquier variable adicional relevante que aparezca en `info` y no estuviera documentada previamente.

El objetivo de esta sección es detectar información adicional disponible en tiempo de ejecución.

### 5.3 Política aleatoria

Ejecutar una política completamente aleatoria durante **al menos 10 episodios independientes**.

En cada step se seleccionará una acción usando el espacio de acciones del entorno.

Por cada episodio se debe registrar como mínimo:

- número de episodio;
- seed, si aplica;
- recompensa acumulada;
- cantidad de steps;
- `terminated`;
- `truncated`;
- número inicial y final de vidas si el entorno las reporta;
- cantidad de cambios de vida detectados;
- cantidad de steps con recompensa positiva;
- cantidad de steps con recompensa igual a cero;
- cantidad de steps con recompensa negativa;
- recompensa máxima y mínima observada en un step;
- frecuencia de uso de cada acción.

### 5.4 Exploración de información no conocida

Durante la ejecución se debe inspeccionar explícitamente si el entorno expone información adicional útil, por ejemplo:

- vidas;
- número de frame;
- puntuación interna;
- condiciones de terminación;
- variables adicionales del diccionario `info`;
- cambios relevantes en observaciones o recompensas alrededor de una pérdida de vida;
- cualquier comportamiento no anticipado en la ficha técnica.

No es necesario comprender internamente todas las variables encontradas. Deben registrarse y describirse de forma breve si parecen relevantes para el diseño del agente.

## 6. Métricas de salida

Al finalizar los episodios, el notebook debe calcular como mínimo:

### Recompensa

- media;
- mediana;
- desviación estándar;
- mínimo;
- máximo.

### Duración

- steps promedio por episodio;
- mínimo;
- máximo.

### Densidad de recompensa

- porcentaje de steps con recompensa positiva;
- porcentaje con recompensa cero;
- porcentaje con recompensa negativa;
- cantidad promedio de eventos con recompensa positiva por episodio.

### Acciones

- frecuencia absoluta y relativa de cada acción durante el experimento.

### Vidas y terminación

Cuando la información esté disponible:

- vidas iniciales observadas;
- pérdidas de vida promedio por episodio;
- relación observada entre pérdida de vidas y `terminated`;
- cantidad de episodios finalizados por `terminated`;
- cantidad de episodios finalizados por `truncated`.

## 7. Visualizaciones mínimas

Mantener las visualizaciones simples. Incluir únicamente las que ayuden a interpretar el entorno:

1. recompensa total por episodio;
2. longitud de cada episodio;
3. distribución/frecuencia de las acciones;
4. opcionalmente, distribución de recompensas por step si aporta información.

No se requiere construir dashboards ni visualizaciones avanzadas.

## 8. Entregable final

El entregable de la HU será un notebook de Google Colab versionado en el repositorio, recomendado como:

`2_Assault/experimento_0_assault.ipynb`

El notebook debe contener al final una sección titulada **"Conclusiones del Experimento 0"** que produzca un resumen directamente utilizable para actualizar la ficha técnica y tomar decisiones de diseño.

El resumen debe incluir como mínimo:

| Aspecto | Resultado esperado |
|---|---|
| Baseline aleatorio | recompensa promedio y dispersión sobre ≥10 episodios |
| Rango observado | recompensa mínima y máxima por episodio |
| Duración | steps promedio y rango |
| Densidad de recompensa | proporción de steps positivos, cero y negativos |
| Vidas | comportamiento observado, si ALE lo expone |
| Terminación | evidencia sobre `terminated` y `truncated` |
| Acciones | frecuencia observada de las 7 acciones |
| Información nueva | variables o comportamientos descubiertos durante ejecución |
| Implicación para RL | interpretación breve de cómo los resultados afectan selección de algoritmo y diseño del agente |

El notebook debe finalizar indicando explícitamente qué preguntas de la ficha técnica quedaron respondidas y cuáles siguen abiertas.

## 9. Criterios de aceptación

### CA01 — Ejecución reproducible

**Dado** un runtime limpio de Google Colab,  
**cuando** se ejecutan todas las celdas en orden,  
**entonces** el notebook instala las dependencias necesarias y ejecuta el experimento sin modificaciones manuales al código.

### CA02 — Entorno validado

**Dado** `ALE/Assault-v5`,  
**cuando** se inicializa el entorno,  
**entonces** el notebook evidencia observaciones, espacio de acciones, nombres de acciones y configuración utilizada.

### CA03 — Inspección de información disponible

**Dado** que ALE puede retornar información adicional mediante `info`,  
**cuando** se ejecutan `reset()` y `step()`,  
**entonces** el notebook inspecciona y documenta las claves disponibles sin asumir previamente su contenido.

### CA04 — Baseline aleatorio

**Dado** el espacio de acciones de Assault,  
**cuando** se ejecuta la política aleatoria,  
**entonces** se completan al menos 10 episodios independientes y se almacena un registro por episodio.

### CA05 — Métricas del entorno

**Dado** el conjunto de episodios ejecutados,  
**cuando** finaliza el experimento,  
**entonces** se calculan las métricas de recompensa, duración, densidad de recompensas, frecuencia de acciones y terminación definidas en esta HU.

### CA06 — Vidas

**Dado** que la ficha técnica no confirma el comportamiento de las vidas,  
**cuando** el entorno exponga dicha información,  
**entonces** el notebook registra vidas iniciales, cambios detectados y relación observable con la finalización del episodio.

Si ALE no expone esta información mediante la interfaz utilizada, el notebook debe dejar evidencia explícita de esta limitación en lugar de inventar o inferir el dato.

### CA07 — Descubrimientos no previstos

**Dado** que el objetivo también es identificar información desconocida,  
**cuando** durante la ejecución aparezcan variables o comportamientos adicionales relevantes,  
**entonces** se documentan en la sección final del notebook.

### CA08 — Resultado interpretable

**Dado** que el experimento alimentará la selección del algoritmo,  
**cuando** finalice el notebook,  
**entonces** existe una sección de conclusiones que distingue claramente:

- hechos observados;
- métricas calculadas;
- información todavía desconocida;
- implicaciones preliminares para el agente.

### CA09 — Simplicidad

**Dado** que esta HU es de exploración,  
**cuando** se revise la implementación,  
**entonces** no contiene arquitectura, patrones, clases o dependencias que no sean necesarias para ejecutar y medir el entorno.

## 10. Definition of Done / Criterios de terminado

La HU se considera terminada únicamente cuando:

- [ ] existe `2_Assault/experimento_0_assault.ipynb` en el repositorio;
- [ ] el notebook ejecuta completamente en Google Colab desde un runtime limpio;
- [ ] las dependencias y versiones relevantes quedan registradas;
- [ ] la configuración utilizada para `ALE/Assault-v5` queda explícita;
- [ ] se ejecutan al menos 10 episodios con política aleatoria;
- [ ] existe una tabla con resultados por episodio;
- [ ] se calculan los estadísticos definidos en la HU;
- [ ] se inspecciona el contenido de `info` y cualquier información adicional del entorno;
- [ ] se registra el comportamiento de vidas o se documenta que no fue posible obtenerlo;
- [ ] se registran `terminated` y `truncated`;
- [ ] existen las visualizaciones mínimas definidas;
- [ ] se documentan hallazgos no previstos;
- [ ] existe una sección final de conclusiones;
- [ ] las conclusiones identifican qué información faltante de `ficha_tecnica.md` quedó resuelta;
- [ ] se obtiene un baseline cuantitativo reutilizable para comparar posteriormente el agente entrenado;
- [ ] no se realiza entrenamiento de ningún algoritmo de RL dentro de esta HU.

## 11. Fuera de alcance

Esta HU no incluye:

- selección definitiva del algoritmo;
- entrenamiento DQN, DDQN, DQN + PER o REINFORCE;
- diseño de CNN;
- optimización de hiperparámetros;
- Replay Buffer;
- checkpoints de entrenamiento;
- TensorBoard;
- MLflow;
- evaluación de un agente entrenado;
- optimización avanzada de hardware.

Estos elementos serán abordados en historias posteriores una vez cerrado el EDA empírico.

## 12. Dependencias

- `2_Assault/ficha_tecnica.md`.
- Enunciado del Reto 1.
- Documentación oficial de ALE para Assault.
- Google Colab.

## 13. Decisión habilitada por esta HU

Una vez terminado el Experimento 0, el equipo deberá poder responder con evidencia empírica preguntas como:

1. ¿Qué tan dispersa es la recompensa de Assault bajo comportamiento aleatorio?
2. ¿Las recompensas son frecuentes o relativamente escasas?
3. ¿Qué tan largos son los episodios?
4. ¿Cómo se comportan las vidas y la terminación del juego?
5. ¿Existe información adicional útil entregada por ALE durante la ejecución?
6. ¿Los resultados refuerzan la elección preliminar de DDQN o sugieren que la priorización de experiencias de DQN + PER puede aportar mayor valor?

La selección formal del algoritmo debe realizarse en una decisión posterior utilizando conjuntamente la ficha técnica y los resultados obtenidos por este experimento.
