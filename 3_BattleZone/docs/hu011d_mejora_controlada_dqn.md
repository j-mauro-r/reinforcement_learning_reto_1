# HU011D — Mejora controlada del aprendizaje DQN

## 1. Propósito

Mejorar el comportamiento del agente BattleZone manteniendo **DQN clásico** y modificando únicamente tres parámetros con evidencia directa de que pueden estar limitando el aprendizaje actual.

La implementación debe ser deliberadamente simple, compatible con el código existente y orientada a producir un nuevo entrenamiento comparable con la corrida actual.

HU011D no cambia el algoritmo, no introduce DQN+PER y no crea nueva infraestructura.

---

## 2. Evidencia que motiva la HU

La corrida actual de 1.000.000 de pasos muestra:

- recompensa de entrenamiento muy variable, con episodios altos aislados pero sin tendencia creciente sostenida del promedio;
- Q-value medio creciente y luego estable, sin una mejora equivalente en recompensa;
- epsilon llegando al mínimo aproximadamente en 250.000 pasos, dejando gran parte del entrenamiento con exploración muy baja;
- evaluación greedy de 10 episodios con recompensa idéntica de 3.000 en todos los casos;
- comportamiento visual observado sin una estrategia sólida y claramente aprendida.

La hipótesis de HU011D es que DQN sí está actualizando sus estimaciones, pero puede estar convergiendo demasiado pronto a una política limitada por poca diversidad de Replay y exploración insuficiente.

---

## 3. Objetivo funcional

Entrenar una nueva versión del mismo agente DQN aplicando exclusivamente estas tres mejoras:

1. aumentar la capacidad del Replay Buffer;
2. prolongar la exploración epsilon-greedy;
3. reducir el learning rate para estabilizar las actualizaciones.

Después del entrenamiento se debe reutilizar el flujo existente de generación de artefactos y evaluación de 10 episodios.

No se deben agregar métricas, componentes o abstracciones que no sean necesarias para aplicar y validar estas tres modificaciones.

---

## 4. Políticas obligatorias

1. **Simplicidad primero.** Resolver la HU con cambios mínimos sobre configuración y código existente.
2. **SOLID y DRY de forma pragmática.** Reutilizar `DQNAgent`, `DQNTrainer`, Replay Buffer, `training_run.py`, TensorBoard, checkpoints, delivery y evaluación existentes.
3. **No duplicar el pipeline.** No crear otro trainer, otro agente, otro notebook ni otro sistema de evaluación.
4. **No cambiar el algoritmo.** El algoritmo sigue siendo DQN clásico.
5. **No implementar PER en esta HU.** DQN+PER queda como alternativa posterior solo si esta mejora controlada no produce evidencia suficiente.
6. **No cambiar preprocessing.** Mantener RGB 128×128, stack 4 y el contrato actual de observaciones para evitar cambios de arquitectura y riesgo de regresión.
7. **No cambiar arquitectura de red.** Mantener la CNN y dimensiones actuales.
8. **No cambiar reward, gamma, batch size, train frequency ni target sync salvo que sea imprescindible para corregir un error.**
9. **No crear nuevos sistemas de tracking, persistencia o reporting.** Reutilizar los existentes.
10. **No agregar intervención manual adicional al notebook.** La ejecución del profesor debe conservar el flujo actual de principio a fin.
11. **No eliminar compatibilidad con checkpoints/modelos existentes.** Los artefactos anteriores deben seguir siendo cargables con el código compatible existente cuando corresponda.
12. **No redactar automáticamente análisis académico ni conclusiones.** Esa interpretación seguirá siendo responsabilidad del estudiante.

---

## 5. Mejora 1 — Replay Buffer más amplio

### Problema observado

La corrida larga actual usa:

```yaml
replay_buffer_capacity: 4096
```

Para un entrenamiento de 1.000.000 de pasos, este buffer conserva una fracción muy pequeña de la experiencia y favorece que el aprendizaje dependa en exceso de transiciones recientes.

### Cambio propuesto

Usar:

```yaml
replay_buffer_capacity: 16384
```

El cambio multiplica por cuatro la diversidad de experiencia sin modificar la implementación del Replay Buffer ni introducir PER.

### Restricción de memoria

El código existente ya estima memoria y protege los checkpoints FULL. HU011D debe reutilizar ese mecanismo.

No se debe crear otro memory manager ni otra política de checkpoint.

Si el preflight existente determina que la memoria disponible no permite el entrenamiento de forma segura, debe fallar claramente antes del entrenamiento, igual que en el flujo actual.

No reducir resolución, cambiar a grayscale ni modificar la arquitectura como parte de esta HU.

---

## 6. Mejora 2 — Exploración más prolongada

### Problema observado

La configuración actual reduce epsilon de 1.0 a 0.05 en:

```yaml
decay_steps: 250000
```

Por tanto, aproximadamente el 75 % del entrenamiento largo ocurre con epsilon cercano al mínimo.

La evidencia visual y la evaluación idéntica de 3.000 puntos sugieren una posible convergencia temprana hacia una política determinista limitada.

### Cambio propuesto

Usar:

```yaml
epsilon:
  start: 1.0
  end: 0.05
  decay_steps: 750000
```

Esto mantiene exploración significativa durante una mayor parte del entrenamiento sin cambiar la estrategia epsilon-greedy existente.

No crear nuevos schedules ni clases de exploración.

---

## 7. Mejora 3 — Learning rate más conservador

### Problema observado

La corrida muestra loss con picos importantes y recompensa media sin crecimiento sostenido.

La configuración base usa:

```yaml
learning_rate: 0.00025
```

### Cambio propuesto

Usar:

```yaml
learning_rate: 0.0001
```

La intención es reducir la magnitud de las actualizaciones y favorecer una convergencia más estable.

No cambiar optimizer, loss, gamma ni arquitectura de red.

---

## 8. Implementación técnica mínima

La solución debe aprovechar la estructura existente.

### 8.1 Configuración

Actualizar el perfil de entrenamiento largo con los valores HU011D:

```yaml
long_training:
  profile: "improved_v2"
  dqn:
    batch_size: 32
    replay_buffer_capacity: 16384
    learning_rate: 0.0001
  training:
    learning_starts: 1024
    train_frequency: 4
    target_sync_interval: 10000
    epsilon:
      start: 1.0
      end: 0.05
      decay_steps: 750000
```

Los demás valores del entrenamiento largo deben conservarse.

### 8.2 Resolver del perfil

Actualmente `resolve_long_training_config(...)` ya aplica overrides de Replay y entrenamiento.

Solo debe extenderse lo mínimo necesario para:

- aceptar `improved_v2` como perfil soportado;
- aplicar `long_training.dqn.learning_rate` a `effective["dqn"]["learning_rate"]`.

No crear un framework genérico de perfiles ni nuevas clases de configuración.

La lógica esperada es equivalente conceptualmente a:

```python
SUPPORTED_LONG_TRAINING_PROFILES = {"reference_v1", "improved_v2"}
```

y una asignación adicional del learning rate.

`reference_v1` debe seguir siendo reconocido por el código para no romper compatibilidad con evidencia/configuraciones anteriores.

### 8.3 Notebook

No crear nuevas secciones de orquestación.

El notebook debe seguir usando el mismo flujo existente:

```text
bootstrap
→ preflight
→ entrenamiento DQN
→ checkpoints / TensorBoard
→ modelo final
→ delivery existente
→ evaluación greedy de 10 episodios
```

HU011D no debe exigir que el profesor copie run IDs, edite rutas o ejecute celdas manualmente entre etapas.

---

## 9. Qué NO modificar

No modificar salvo corrección indispensable:

- `DQNAgent`;
- `DQNTrainer`;
- implementación del Replay Buffer;
- CNN;
- preprocessing RGB 128×128 stack 4;
- action space;
- reward;
- gamma;
- Smooth L1 loss;
- Adam;
- batch size 32 del long run;
- train frequency 4;
- target sync 10.000;
- checkpoints existentes;
- TensorBoard existente;
- modelo entregable;
- generación de videos;
- evaluación HU011C;
- lógica de 10 episodios greedy.

La HU debe cambiar parámetros, no rediseñar el proyecto.

---

## 10. Validación barata antes de Colab

Antes de ejecutar otro entrenamiento largo:

1. cargar configuración HU011D;
2. validar que `resolve_long_training_config(...)` produce exactamente:
   - replay = 16.384;
   - learning rate = 0.0001;
   - epsilon decay = 750.000;
3. ejecutar tests focales de configuración/training;
4. ejecutar smoke test corto existente;
5. verificar preflight de memoria;
6. verificar que el notebook sigue siendo ejecutable sin nuevos pasos manuales.

No ejecutar entrenamiento largo durante tests automáticos.

---

## 11. Entrenamiento real

Una vez superadas las validaciones baratas, ejecutar desde cero una nueva corrida de:

```text
1.000.000 global steps
```

con el perfil `improved_v2`.

No reutilizar pesos del modelo anterior para esta comparación principal.

La corrida debe producir los mismos tipos de artefactos ya implementados:

- TensorBoard;
- checkpoints;
- modelo final;
- videos;
- gráfica de entrenamiento;
- evaluación greedy de 10 episodios;
- gráfica de explotación.

No crear artefactos adicionales para HU011D.

---

## 12. Criterios de comparación

HU011D debe permitir comparar el nuevo entrenamiento con la corrida DQN actual usando evidencia ya existente.

La decisión se debe apoyar principalmente en:

1. reward promedio de los 10 episodios greedy;
2. variabilidad de rewards entre episodios;
3. evolución del reward medio durante entrenamiento;
4. observación cualitativa del video post-entrenamiento.

No se requieren tests estadísticos avanzados.

El baseline actual de referencia académica es:

```text
DQN actual: 10 episodios × reward 3000
average_reward = 3000
```

La política aleatoria histórica tuvo promedio 3.000, por lo que el nuevo agente debe aportar evidencia más convincente que la corrida actual para justificar mantener DQN como solución final.

---

## 13. Criterios de aceptación

### CA01 — Cambios mínimos

La implementación modifica únicamente lo necesario para soportar los tres parámetros HU011D.

### CA02 — DQN intacto

El agente sigue siendo DQN clásico y no incorpora PER, DDQN ni otra variante.

### CA03 — Replay

El perfil `improved_v2` usa Replay Buffer de 16.384 transiciones.

### CA04 — Exploración

El perfil `improved_v2` usa epsilon 1.0 → 0.05 con decay de 750.000 pasos.

### CA05 — Learning rate

El perfil `improved_v2` usa learning rate 0.0001.

### CA06 — Compatibilidad

`reference_v1` continúa siendo aceptado por el resolver y no se rompen los tests/contracts existentes sin necesidad.

### CA07 — Ejecución del profesor

No se agregan pasos manuales, dependencias nuevas ni configuraciones externas que impidan ejecutar el notebook en Google Colab.

### CA08 — Smoke/regresión

Las validaciones cortas existentes siguen funcionando.

### CA09 — Entrenamiento real

La nueva corrida alcanza 1.000.000 global steps con artefactos estándar del proyecto.

### CA10 — Evaluación

El modelo nuevo se evalúa con el flujo HU011C existente sobre 10 episodios greedy (`epsilon=0.0`).

### CA11 — Sin análisis automático

El código no redacta conclusiones sobre si el agente aprendió mejor; solo deja la evidencia para análisis del estudiante.

---

## 14. Definition of Done

HU011D puede marcarse **IMPLEMENTADA — LISTA PARA ENTRENAMIENTO REAL** cuando:

- [ ] el perfil `improved_v2` está configurado;
- [ ] replay = 16.384;
- [ ] epsilon decay = 750.000;
- [ ] learning rate = 0.0001;
- [ ] `reference_v1` sigue siendo compatible;
- [ ] no se modificó algoritmo, red ni preprocessing;
- [ ] tests focales pasan;
- [ ] smoke test pasa;
- [ ] preflight de memoria pasa en el runtime objetivo;
- [ ] el notebook conserva ejecución de principio a fin sin intervención adicional.

HU011D puede marcarse **COMPLETADA** únicamente cuando:

- [ ] se ejecutó la nueva corrida completa de 1.000.000 pasos;
- [ ] se generó el modelo nuevo;
- [ ] se generaron las gráficas existentes;
- [ ] se generaron los videos existentes;
- [ ] se ejecutaron 10 episodios greedy;
- [ ] quedó disponible el nuevo `AVERAGE_REWARD` para comparación;
- [ ] el estudiante dispone de evidencia suficiente para decidir si continuar con DQN o pasar a DQN+PER.

---

## 15. Decisión posterior

HU011D no decide automáticamente migrar a DQN+PER.

Después de la corrida `improved_v2`:

- si la recompensa y el comportamiento muestran una mejora clara frente al DQN actual, continuar con DQN;
- si el agente sigue alrededor del baseline aleatorio, mantiene recompensas prácticamente idénticas o continúa mostrando una política visualmente limitada, evaluar DQN+PER como siguiente hipótesis.

La decisión final debe basarse en evidencia y no en agregar complejidad antes de necesitarla.
