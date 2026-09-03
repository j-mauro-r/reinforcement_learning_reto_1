# HU011C — Evaluación académica de explotación y cierre de evidencia

## 1. Propósito

Completar únicamente los cinco bloqueadores académicos pendientes de BattleZone exigidos por el enunciado del Reto 1:

1. ejecutar al menos 10 partidas de evaluación con el agente entrenado;
2. calcular y reportar el puntaje promedio de esas partidas;
3. generar la gráfica de recompensa durante explotación;
4. dejar disponible la evidencia necesaria para que el estudiante describa el comportamiento aprendido;
5. dejar preparada la sección para que el estudiante redacte las conclusiones finales.

Esta HU prioriza el cumplimiento académico sobre infraestructura adicional. No debe introducir complejidad que no sea necesaria para satisfacer el enunciado.

---

## 2. Fuente de verdad

La única fuente de verdad funcional para esta HU es `enunciado_reto_1.txt`.

Para BattleZone, el enunciado exige que el agente:

- utilice uno de los algoritmos permitidos;
- aprenda una política que maximice la recompensa promedio;
- sea evaluado en al menos 10 partidas;
- demuestre algún comportamiento lógico en lugar de actuar de forma muy aleatoria;
- presente evidencia cuantitativa del desempeño;
- incluya una gráfica de recompensa durante explotación;
- incluya una descripción del comportamiento aprendido;
- incluya conclusiones basadas en los resultados.

La comparación con una política aleatoria es opcional y no constituye criterio obligatorio de cierre para esta HU.

---

## 3. Políticas obligatorias de implementación

1. **Simplicidad primero.** Implementar únicamente lo necesario para cumplir el enunciado.
2. **SOLID y DRY.** Reutilizar la carga del modelo, creación del entorno y utilidades existentes. No duplicar lógica.
3. **Sin reentrenamiento.** HU011C consume el modelo ya entrenado.
4. **Sin tuning.** No modificar hiperparámetros ni buscar un mejor modelo.
5. **Sin infraestructura adicional.** No agregar MLflow, W&B, nuevas capas de persistencia, pipelines, servicios ni abstracciones innecesarias.
6. **Sin métricas no requeridas como condición de cierre.** La métrica académica obligatoria es el promedio de recompensa sobre al menos 10 episodios.
7. **El análisis cualitativo lo redacta el estudiante.** El código solo debe producir evidencia objetiva.
8. **Las conclusiones las redacta el estudiante.** No generar conclusiones automáticas ni texto interpretativo mediante código o IA.
9. **No modificar DQN.** Esta HU evalúa el agente existente; no cambia su aprendizaje.
10. **No depender de `2_Assault/`.** Puede consultarse como referencia, pero BattleZone debe ejecutar con sus propios módulos.

---

## 4. Alcance

HU011C debe:

- cargar `battlezone_dqn_model.pt` sin reentrenar;
- ejecutar un mínimo de 10 episodios independientes de evaluación;
- ejecutar el agente en explotación, sin exploración aleatoria (`epsilon=0.0`);
- registrar al menos `episode`, `seed`, `reward` y `steps` por episodio;
- calcular el promedio de recompensa de los episodios evaluados;
- generar una gráfica simple de recompensa por episodio de explotación;
- mostrar los resultados dentro de `pipeline_battlezone.ipynb`;
- dejar una sección Markdown claramente identificada para que el estudiante escriba el análisis del comportamiento aprendido;
- dejar una sección Markdown claramente identificada para que el estudiante escriba las conclusiones finales.

---

## 5. Fuera de alcance

HU011C NO debe:

- volver a entrenar el agente;
- cambiar arquitectura de red;
- modificar rewards;
- modificar preprocessing;
- cambiar hiperparámetros;
- ejecutar tuning;
- seleccionar automáticamente otro checkpoint;
- exigir superar el baseline aleatorio como criterio de cierre;
- calcular métricas estadísticas avanzadas como requisito obligatorio;
- crear dashboards;
- crear nuevos sistemas de tracking;
- crear nuevos formatos de artefactos si los existentes son suficientes;
- redactar automáticamente análisis o conclusiones;
- declarar que BattleZone está "resuelto" si la evidencia no lo soporta.

---

## 6. Flujo mínimo esperado

```text
battlezone_dqn_model.pt
        ↓
cargar modelo existente
        ↓
crear entorno BattleZone en modo evaluación
        ↓
ejecutar >= 10 episodios con epsilon=0.0
        ↓
registrar reward y steps por episodio
        ↓
calcular reward promedio
        ↓
graficar reward por episodio
        ↓
mostrar tabla + promedio + gráfica en notebook
        ↓
estudiante redacta análisis del comportamiento
        ↓
estudiante redacta conclusiones
```

---

## 7. Evaluación de episodios

### 7.1 Número de episodios

El valor por defecto debe ser:

```python
EVALUATION_EPISODES = 10
```

Puede ejecutarse un número mayor, pero nunca menor para la evidencia final.

### 7.2 Política de evaluación

El agente debe evaluarse en explotación:

```python
epsilon = 0.0
```

La finalidad es observar el comportamiento aprendido por el modelo, no su política exploratoria de entrenamiento.

### 7.3 Independencia de episodios

Cada episodio debe comenzar con un `reset()` nuevo y usar una seed explícita distinta.

No se requiere construir un framework adicional de gestión de seeds. Una lista simple y visible en el notebook es suficiente.

Ejemplo:

```python
EVALUATION_SEEDS = [20262001 + i for i in range(EVALUATION_EPISODES)]
```

### 7.4 Datos mínimos por episodio

Registrar únicamente lo necesario:

| Campo | Obligatorio | Uso |
|---|---:|---|
| `episode` | Sí | Identificar la partida |
| `seed` | Sí | Trazabilidad básica |
| `reward` | Sí | Métrica académica |
| `steps` | Sí | Contexto de duración |

No agregar métricas adicionales salvo que ya existan y puedan reutilizarse sin aumentar complejidad.

---

## 8. Puntaje promedio

Al terminar los episodios debe calcularse directamente:

```python
average_reward = sum(rewards) / len(rewards)
```

El notebook debe mostrar de forma visible:

```text
EVALUATION_EPISODES: 10
AVERAGE_REWARD: <valor_real>
```

El promedio debe provenir exclusivamente de los episodios de evaluación final ejecutados con el modelo entrenado.

No usar como sustituto:

- reward de entrenamiento;
- reward medio de TensorBoard;
- reward del video;
- sanity episode de HU011B;
- una sola partida.

---

## 9. Gráfica de explotación

Generar una única gráfica simple y legible con:

- eje X: episodio de evaluación;
- eje Y: recompensa total del episodio;
- los 10 o más episodios evaluados;
- una línea horizontal o anotación con el promedio, si puede hacerse sin complejidad adicional.

Título sugerido:

```text
BattleZone — Recompensa durante explotación
```

La gráfica debe aparecer dentro del notebook final.

No es necesario crear dashboards, múltiples visualizaciones ni herramientas adicionales.

---

## 10. Evidencia para análisis del comportamiento aprendido

El enunciado exige describir el comportamiento aprendido y demostrar que el agente presenta alguna conducta lógica en lugar de actuar de forma muy aleatoria.

El código NO debe escribir esta interpretación.

HU011C debe únicamente dejar al estudiante evidencia objetiva suficiente para analizarla, usando como mínimo:

- los resultados de los episodios de explotación;
- la gráfica de explotación;
- el video post-entrenamiento ya generado en HU011B;
- el comportamiento observable durante las partidas de evaluación.

En el notebook debe existir una sección Markdown vacía o con instrucciones breves:

```markdown
## Análisis del comportamiento aprendido — completar por el estudiante

Describir aquí, a partir del video y de los episodios de evaluación, qué comportamientos lógicos se observan en el agente y cómo se relacionan con el DQN y los hiperparámetros utilizados.
```

No generar automáticamente afirmaciones como "el agente aprendió a apuntar", "aprendió a evadir" o equivalentes si no han sido observadas y redactadas por el estudiante.

---

## 11. Conclusiones

El enunciado exige una conclusión general basada en los resultados obtenidos.

HU011C no debe redactarla automáticamente.

El notebook debe incluir al final:

```markdown
## Conclusiones — completar por el estudiante

Redactar aquí la conclusión general basada en los resultados de entrenamiento y evaluación del agente BattleZone.
```

La HU se considera técnicamente implementada cuando la evidencia cuantitativa está disponible y estas secciones están listas para ser completadas manualmente.

La entrega académica final solo estará completa cuando el estudiante haya escrito ambos apartados.

---

## 12. Implementación técnica mínima recomendada

Antes de crear código nuevo, revisar si la evaluación puede implementarse reutilizando directamente:

- carga del modelo existente;
- `create_battlezone_env(...)`;
- `select_action(..., epsilon=0.0)`;
- utilidades de reporting existentes.

Si la lógica no existe, implementar una función pequeña y con una sola responsabilidad, por ejemplo:

```python
def evaluate_agent(agent, env_factory, seeds):
    ...
```

La función debe limitarse a ejecutar episodios y devolver resultados estructurados.

La generación de la gráfica puede permanecer en el notebook o reutilizar la utilidad de reporting existente si ello reduce duplicación.

**No crear una nueva arquitectura de evaluación si una función simple resuelve el requisito.**

---

## 13. Integración en `pipeline_battlezone.ipynb`

Agregar una sección final claramente identificada:

```text
HU011C — Evaluación académica de explotación
```

Orden mínimo de celdas:

1. cargar el modelo entrenado;
2. definir `EVALUATION_EPISODES >= 10` y seeds;
3. ejecutar evaluación;
4. mostrar tabla de resultados;
5. mostrar promedio de recompensa;
6. mostrar gráfica de explotación;
7. sección Markdown de análisis del comportamiento para completar por el estudiante;
8. sección Markdown de conclusiones para completar por el estudiante.

La evaluación no debe disparar entrenamiento largo ni modificar el modelo.

---

## 14. Criterios de aceptación

### CA01 — Evaluación mínima

Se ejecutan **al menos 10 episodios completos** con el agente entrenado.

### CA02 — Explotación

Todos los episodios finales se ejecutan con `epsilon=0.0` o mecanismo equivalente de política greedy.

### CA03 — Resultados por episodio

El notebook muestra por cada episodio al menos:

```text
episode
seed
reward
steps
```

### CA04 — Promedio

El notebook calcula y muestra el **puntaje promedio** usando los rewards de los episodios de evaluación.

### CA05 — Gráfica de explotación

El notebook contiene una gráfica de recompensa por episodio correspondiente a la misma evaluación final.

### CA06 — Modelo existente

La evaluación usa el modelo BattleZone entrenado existente y no ejecuta reentrenamiento.

### CA07 — Evidencia para análisis

La entrega deja visibles resultados, gráfica y video suficientes para que el estudiante pueda redactar el análisis del comportamiento aprendido.

### CA08 — Análisis manual

Existe una sección dentro del notebook para el análisis del comportamiento aprendido y su contenido final es responsabilidad del estudiante.

### CA09 — Conclusiones manuales

Existe una sección dentro del notebook para conclusiones y su contenido final es responsabilidad del estudiante.

### CA10 — Simplicidad

No se introducen dependencias, servicios, trackers, abstracciones o métricas que no sean necesarias para cumplir esta HU.

### CA11 — Regresión

La implementación no modifica el entrenamiento DQN ni rompe la carga/inferencia existente del modelo.

---

## 15. Definition of Done

HU011C puede marcarse **IMPLEMENTADA** cuando:

- [ ] existen al menos 10 episodios de evaluación reales;
- [ ] todos fueron ejecutados en explotación con el modelo entrenado;
- [ ] existe tabla con reward por episodio;
- [ ] existe puntaje promedio calculado sobre esos episodios;
- [ ] existe gráfica de recompensa durante explotación;
- [ ] la evidencia aparece dentro de `pipeline_battlezone.ipynb`;
- [ ] existe la sección para análisis manual del comportamiento;
- [ ] existe la sección para conclusiones manuales;
- [ ] no se ejecutó nuevo entrenamiento;
- [ ] no se agregó complejidad fuera del alcance.

La **entrega académica de BattleZone** solo puede considerarse completa cuando, además de lo anterior:

- [ ] el estudiante haya redactado el análisis del comportamiento aprendido;
- [ ] el estudiante haya redactado las conclusiones basadas en los resultados.

---

## 16. Resultado esperado

Al finalizar HU011C, el notebook debe permitir responder directamente las preguntas académicas pendientes:

```text
¿Cuántas partidas se evaluaron?            >= 10
¿Cuál fue el puntaje promedio?             <valor real>
¿Cómo evolucionó la recompensa al explotar? <gráfica visible>
¿Qué comportamiento aprendió el agente?    <análisis del estudiante>
¿Qué concluye el estudiante?               <conclusión del estudiante>
```

No se requiere ningún desarrollo adicional para cerrar esta HU salvo lo necesario para producir estas evidencias.