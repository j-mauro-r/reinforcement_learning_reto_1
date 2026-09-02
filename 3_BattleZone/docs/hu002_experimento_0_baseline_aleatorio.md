# HU002 — Experimento 0 y baseline aleatorio de BattleZone

## 1. Identificación

- **ID:** HU002
- **Nombre:** Experimento 0 y baseline aleatorio de BattleZone
- **Estado:** IMPLEMENTADA - pendiente únicamente de validación AV14 en Google Colab
- **Dependencia previa:** HU001 — Caracterización técnica y ficha inicial de BattleZone
- **Habilita:** HU003 — Pipeline reproducible del entorno
- **Fuentes de verdad:**
  - `enunciado_reto_1.txt`
  - `3_BattleZone/docs/ficha_tecnica.md`
  - `3_BattleZone/docs/arquitectura.md`
  - `3_BattleZone/docs/implementacion.md`
  - `3_BattleZone/docs/lineamientos.md`
  - documentación oficial de Arcade Learning Environment para BattleZone

---

## 2. Contexto y problema

HU001 dejó documentado el contrato inicial de `ALE/BattleZone-v5`: observación visual RGB `210×160×3`, `Discrete(18)`, `frameskip=4`, `repeat_action_probability=0.25`, modes `[1,2,3]`, difficulty `[0]`, radar, enemigos, obstáculos y 5 vidas iniciales documentadas.

Sin embargo, varias decisiones necesarias para continuar siguen dependiendo de evidencia empírica. Antes de diseñar el preprocessing definitivo o seleccionar formalmente el algoritmo, el proyecto necesita observar cómo se comporta BattleZone bajo una política completamente aleatoria.

HU002 debe responder con datos reales preguntas sobre:

- recompensa y su dispersión;
- duración de episodios;
- densidad de rewards;
- vidas y terminación;
- contenido real de `info`;
- frecuencia de uso de las 18 acciones;
- relación observable entre eventos de recompensa y acciones;
- comportamiento del radar y legibilidad visual;
- versiones efectivas del runtime.

Esta HU es exploratoria. No debe entrenar ningún agente.

---

## 3. Historia de usuario

> **Como** equipo que desarrolla el agente de Reinforcement Learning para BattleZone, **quiero** ejecutar un Experimento 0 reproducible con política aleatoria, **para** obtener un baseline cuantitativo y caracterizar empíricamente el entorno antes de definir preprocessing y seleccionar algoritmo.

---

## 4. Objetivo verificable

Al finalizar HU002 debe existir un notebook reproducible que:

1. ejecute `ALE/BattleZone-v5` desde un runtime limpio;
2. registre versiones reales de Python, Gymnasium y ALE-Py;
3. inspeccione observaciones, action space e `info`;
4. ejecute **al menos 10 episodios independientes** con política aleatoria;
5. produzca una tabla por episodio;
6. calcule el baseline cuantitativo requerido;
7. genere visualizaciones mínimas;
8. actualice `3_BattleZone/docs/ficha_tecnica.md` con hallazgos empíricos confirmados;
9. deje explícitas las implicaciones para HU003 y HU004.

---

## 5. Alcance

### 5.1 Notebook del Experimento 0

Crear:

`3_BattleZone/experimento_0_battlezone.ipynb`

El notebook debe ser ejecutable de principio a fin en Google Colab y, cuando sea viable, también localmente.

Debe priorizar simplicidad, legibilidad y bajo costo computacional.

### 5.2 Configuración explícita del experimento

Agrupar en una sola sección del notebook, como mínimo:

- `ENV_ID = "ALE/BattleZone-v5"`;
- número de episodios, mínimo `10`;
- seed base;
- mode default `1`;
- difficulty default `0`;
- `obs_type="rgb"`;
- `frameskip=4`;
- `repeat_action_probability=0.25`;
- política = aleatoria;
- render desactivado durante el baseline normal.

No crear todavía `battlezone_config.yaml`; la configuración productiva corresponde a HU003.

### 5.3 Registro del runtime

Registrar como mínimo:

- Python;
- Gymnasium;
- ALE-Py;
- NumPy;
- sistema operativo/runtime;
- CPU;
- RAM disponible;
- disponibilidad de GPU.

La GPU puede reportarse, pero HU002 no debe depender de ella para una política aleatoria.

### 5.4 Inspección inicial del entorno

Antes de ejecutar los episodios, el notebook debe evidenciar:

- `env.observation_space`;
- `env.action_space`;
- `env.unwrapped.get_action_meanings()` cuando esté disponible;
- shape de observación;
- dtype;
- rango mínimo/máximo observado en al menos un frame;
- frame inicial visible;
- contenido de `info` en `reset()`;
- contenido de `info` en algunos `step()`;
- mode/difficulty efectivos cuando puedan consultarse;
- contadores de frame cuando estén disponibles.

El notebook debe comprobar empíricamente que existen 18 acciones y registrar cualquier discrepancia frente a la ficha técnica.

### 5.5 Política aleatoria

Cada acción se seleccionará exclusivamente mediante el action space del entorno, sin heurísticas ni conocimiento del juego.

Ejemplo conceptual:

```python
action = env.action_space.sample()
```

No se permite introducir reglas como "disparar más", "evitar NOOP" o priorizar acciones con FIRE porque eso invalidaría el baseline como política completamente aleatoria.

### 5.6 Episodios

Ejecutar al menos **10 episodios independientes**.

Para cada episodio registrar como mínimo:

- `episode_id`;
- seed utilizada;
- recompensa acumulada;
- steps;
- frames cuando puedan calcularse;
- `terminated`;
- `truncated`;
- vidas iniciales;
- vidas finales;
- cantidad de pérdidas de vida detectadas;
- vidas extra detectadas, si ocurren;
- steps con reward positivo;
- steps con reward cero;
- steps con reward negativo;
- reward máximo por step;
- reward mínimo por step;
- cantidad de eventos con reward distinto de cero;
- frecuencia de cada una de las 18 acciones.

Si una métrica no puede obtenerse con la API utilizada, debe registrarse explícitamente como no disponible en lugar de inferirse.

### 5.7 Inspección de `info`

Consolidar las claves observadas en `info` durante:

- `reset()`;
- pasos normales;
- cambios de vida;
- reward distinto de cero;
- terminación.

Cualquier clave nueva relevante debe describirse brevemente y trasladarse a `ficha_tecnica.md` solo si fue observada realmente.

### 5.8 Recompensas y scoring

El notebook debe identificar empíricamente los valores de reward observados por step.

Debe comparar estos valores con el scoring histórico documentado en la ficha, pero **no asumir equivalencia**.

Registrar:

- valores únicos de rewards observados;
- frecuencia de cada reward;
- reward acumulado por episodio;
- si existen rewards negativos;
- si el reward parece escaso o frecuente.

No modificar ni normalizar rewards durante HU002.

### 5.9 Radar y observación visual

HU002 no define todavía el preprocessing definitivo, pero debe generar evidencia que ayude a HU003.

Debe mostrar como mínimo:

- frame RGB original;
- región del radar claramente visible;
- una comparación exploratoria no vinculante entre imagen original y una versión grayscale/resize candidata, si aporta evidencia visual.

Esta comparación no debe convertirse automáticamente en decisión de preprocessing.

Registrar observaciones sobre:

- legibilidad del radar;
- tamaño aparente de enemigos/objetos;
- riesgo de perder información con resize agresivo;
- necesidad aparente de contexto temporal.

### 5.10 Métricas agregadas del baseline

Calcular como mínimo:

#### Recompensa

- media;
- mediana;
- desviación estándar;
- mínimo;
- máximo.

#### Duración

- media de steps;
- mínimo;
- máximo.

#### Densidad de reward

- porcentaje global de steps con reward positivo;
- porcentaje con reward cero;
- porcentaje con reward negativo;
- eventos no-cero promedio por episodio.

#### Vidas

Cuando estén disponibles:

- vidas iniciales observadas;
- pérdidas promedio por episodio;
- vidas extra observadas;
- relación entre última vida y `terminated`.

#### Terminación

- cantidad de episodios con `terminated=True`;
- cantidad de episodios con `truncated=True`.

#### Acciones

- frecuencia absoluta de cada acción;
- frecuencia relativa de cada acción.

### 5.11 Visualizaciones mínimas

Incluir únicamente visualizaciones útiles:

1. recompensa acumulada por episodio;
2. longitud de episodio en steps;
3. frecuencia relativa de las 18 acciones;
4. opcional: distribución/frecuencia de rewards por step si mejora la interpretación.

No construir dashboards ni visualizaciones avanzadas.

### 5.12 Conclusiones del Experimento 0

El notebook debe terminar con una sección Markdown titulada:

`Conclusiones del Experimento 0`

Debe distinguir claramente:

- hechos observados;
- métricas calculadas;
- información todavía desconocida;
- implicaciones para HU003 — preprocessing/pipeline;
- implicaciones para HU004 — selección de algoritmo.

### 5.13 Actualización de ficha técnica

Actualizar `3_BattleZone/docs/ficha_tecnica.md` únicamente con datos empíricos reales obtenidos durante la ejecución.

No reemplazar hechos documentados por resultados de una única corrida sin explicar su naturaleza.

Separar:

- documentación oficial;
- evidencia observada en Experimento 0.

---

## 6. Fuera de alcance

HU002 **no** debe implementar:

- DQN;
- DQN + PER;
- DDQN;
- REINFORCE;
- CNN;
- Replay Buffer;
- Target Network;
- policy network;
- optimizer;
- entrenamiento;
- checkpointing;
- TensorBoard;
- `run_manifest.json`;
- MLflow;
- configuración productiva YAML;
- pipeline definitivo de preprocessing;
- selección formal del algoritmo;
- evaluación de un agente entrenado.

No debe modificar ni reutilizar código bajo `2_Assault/`.

---

## 7. Decisiones y restricciones técnicas

### 7.1 Simplicidad primero

HU002 es una exploración del entorno. Se prefieren funciones pequeñas y código directo dentro del notebook antes que una arquitectura modular prematura.

### 7.2 Independencia total de Assault

El conocimiento previo puede orientar qué medir, pero no se copiará, importará ni reutilizará código, notebooks, helpers ni módulos de `2_Assault/`.

### 7.3 Sin MLflow

BattleZone no utilizará MLflow en ninguna fase.

### 7.4 Sin TensorBoard en HU002

TensorBoard está reservado para observabilidad del entrenamiento en HU008. El baseline aleatorio debe mantenerse simple.

### 7.5 Seeds

Usar una seed base explícita y derivar seeds de episodio de manera reproducible, por ejemplo:

```text
seed_episode = base_seed + episode_id
```

Registrar las seeds utilizadas.

No prometer determinismo absoluto debido a sticky actions y comportamiento interno de ALE.

### 7.6 Rewards reales

No aplicar clipping, normalización ni shaping.

Los valores registrados deben ser exactamente los retornados por ALE.

### 7.7 Terminación

El loop debe finalizar con:

```text
terminated OR truncated
```

No utilizar pérdida de una vida individual como fin artificial de episodio.

### 7.8 Frameskip

HU002 utiliza la configuración oficial de `ALE/BattleZone-v5` y no debe añadir wrappers que introduzcan un segundo frameskip.

Cuando `info` exponga contadores adecuados, registrar evidencia del comportamiento temporal observado.

### 7.9 Idempotencia

Reejecutar el notebook no debe depender de artefactos previos ni destruir resultados externos.

Como HU exploratoria, los resultados pueden regenerarse íntegramente desde las seeds y configuración documentadas.

### 7.10 Documentación

Funciones reutilizables definidas en el notebook deben utilizar docstrings estilo Google cuando su complejidad lo justifique.

No añadir comentarios que solo repitan el código.

---

## 8. Plan de implementación / tareas

### T01 — Preparar notebook y dependencias

**Archivo:** `3_BattleZone/experimento_0_battlezone.ipynb`

**Cambio:** crear notebook ejecutable en Colab con instalación mínima de dependencias y sección de configuración.

**Resultado esperado:** runtime limpio puede preparar BattleZone sin intervención manual adicional.

---

### T02 — Registrar runtime y hardware

**Archivo:** notebook.

**Cambio:** mostrar versiones, CPU, RAM y GPU disponible.

**Resultado esperado:** quedan fijadas las versiones reales para futuras HUs.

**Depende de:** T01.

---

### T03 — Inspeccionar contrato real del entorno

**Archivo:** notebook.

**Cambio:** crear el entorno, ejecutar `reset()`, mostrar frame y registrar observation/action spaces, acciones e `info`.

**Resultado esperado:** confirmar o identificar diferencias respecto a HU001.

**Depende de:** T01.

---

### T04 — Implementar colector simple del baseline

**Archivo:** notebook.

**Cambio:** implementar funciones pequeñas para ejecutar un episodio aleatorio y registrar métricas.

**Resultado esperado:** un episodio produce un registro estructurado completo sin entrenamiento.

**Depende de:** T03.

---

### T05 — Ejecutar al menos 10 episodios

**Archivo:** notebook.

**Cambio:** ejecutar la política aleatoria con seeds explícitas y consolidar tabla por episodio.

**Resultado esperado:** dataset del baseline disponible en memoria/notebook.

**Depende de:** T04.

---

### T06 — Calcular métricas agregadas

**Archivo:** notebook.

**Cambio:** calcular recompensa, duración, densidad de reward, terminación, vidas y acciones.

**Resultado esperado:** baseline cuantitativo reproducible.

**Depende de:** T05.

---

### T07 — Crear visualizaciones mínimas

**Archivo:** notebook.

**Cambio:** recompensa por episodio, duración y distribución de acciones.

**Resultado esperado:** interpretación rápida del baseline.

**Depende de:** T05.

---

### T08 — Inspeccionar radar y preprocessing candidato

**Archivo:** notebook.

**Cambio:** incluir evidencia visual suficiente para evaluar si grayscale/resize podrían degradar radar u objetos pequeños.

**Resultado esperado:** insumos explícitos para HU003 sin fijar todavía el pipeline.

**Depende de:** T03.

---

### T09 — Redactar conclusiones

**Archivo:** notebook.

**Cambio:** completar `Conclusiones del Experimento 0` separando hechos, métricas, incógnitas e implicaciones.

**Resultado esperado:** HU003 y HU004 reciben evidencia utilizable.

**Depende de:** T06–T08.

---

### T10 — Actualizar ficha técnica

**Archivo:** `3_BattleZone/docs/ficha_tecnica.md`

**Cambio:** incorporar solo hallazgos empíricos confirmados y versiones reales del runtime.

**Resultado esperado:** ficha técnica evoluciona de caracterización documental a caracterización parcialmente empírica.

**Depende de:** T09.

---

## 9. Criterios de aceptación

### CA01 — Notebook reproducible

**Dado** un runtime limpio de Google Colab,  
**cuando** se ejecutan las celdas en orden,  
**entonces** las dependencias se instalan y el Experimento 0 puede ejecutarse sin modificar manualmente el código.

### CA02 — Runtime registrado

**Dado** el entorno de ejecución,  
**cuando** comienza el experimento,  
**entonces** quedan registradas versiones reales de Python, Gymnasium, ALE-Py y las condiciones básicas de hardware.

### CA03 — Contrato ALE verificado

**Dado** `ALE/BattleZone-v5`,  
**cuando** se ejecuta `reset()`,  
**entonces** el notebook evidencia observation space, action space, shape, dtype, acciones disponibles e `info` real.

### CA04 — Política estrictamente aleatoria

**Dado** el action space,  
**cuando** se selecciona cada acción,  
**entonces** no se usa ninguna heurística ni preferencia manual sobre acciones específicas.

### CA05 — Baseline ≥10 episodios

**Dado** el Experimento 0,  
**cuando** finaliza su ejecución,  
**entonces** existen resultados de al menos 10 episodios independientes con seeds registradas.

### CA06 — Tabla por episodio

**Dado** cada episodio ejecutado,  
**cuando** se consolidan resultados,  
**entonces** existe una fila con recompensa, duración, terminación, vidas cuando estén disponibles, densidad de reward y frecuencia de acciones.

### CA07 — Métricas agregadas

**Dado** el conjunto de episodios,  
**cuando** se calcula el baseline,  
**entonces** existen media, mediana, desviación estándar, mínimo y máximo de recompensa, además de métricas de duración y densidad de reward.

### CA08 — Vidas y terminación

**Dado** que `info` puede exponer vidas,  
**cuando** se ejecutan episodios,  
**entonces** el notebook registra el comportamiento observado o documenta explícitamente que la información no está disponible.

### CA09 — `info` explorado

**Dado** `reset()` y `step()`,  
**cuando** aparecen claves de `info`,  
**entonces** se consolidan y describen aquellas relevantes para observabilidad futura.

### CA10 — Rewards reales

**Dado** el entorno,  
**cuando** se registran rewards,  
**entonces** no existe clipping, normalización ni shaping y se conservan los valores reales retornados por ALE.

### CA11 — Visualizaciones mínimas

**Dado** el baseline,  
**cuando** termina el análisis,  
**entonces** existen gráficas de recompensa por episodio, duración y frecuencia de acciones.

### CA12 — Radar inspeccionado

**Dado** que el radar es información estratégica,  
**cuando** se inspeccionan frames,  
**entonces** el notebook contiene evidencia visual y una observación explícita sobre riesgos de grayscale/resize/cropping.

### CA13 — Ficha técnica actualizada

**Dado** el resultado empírico,  
**cuando** termina HU002,  
**entonces** `3_BattleZone/docs/ficha_tecnica.md` incorpora los hallazgos confirmados sin mezclar evidencia observada con supuestos.

### CA14 — Independencia de Assault

**Dado** el proyecto BattleZone,  
**cuando** se revisan cambios de HU002,  
**entonces** no existen imports, copias ni modificaciones de archivos bajo `2_Assault/`.

### CA15 — Sin scope creep

**Dado** que HU002 es exploratoria,  
**cuando** se revisa el PR,  
**entonces** no contiene agente, red neuronal, Replay Buffer, entrenamiento, checkpoints, TensorBoard ni selección formal del algoritmo.

---

## 10. Autovalidaciones obligatorias

### AV01 — Ejecución de imports

**Procedimiento:** ejecutar las celdas de instalación/importación en runtime limpio.

**Resultado esperado:** imports sin excepciones.

**PASS:** todas las dependencias mínimas cargan correctamente.

---

### AV02 — Creación del entorno

**Procedimiento:** crear `ALE/BattleZone-v5` con la configuración explícita y ejecutar `reset()`.

**Resultado esperado:** observación válida e `info` retornado.

**PASS:** entorno inicializa sin errores.

---

### AV03 — Action space

**Procedimiento:** consultar `env.action_space` y action meanings.

**Resultado esperado:** `Discrete(18)` y 18 acciones.

**PASS:** contrato coincide o discrepancia queda documentada como hallazgo crítico.

---

### AV04 — Observación

**Procedimiento:** inspeccionar observación de `reset()`.

**Resultado esperado:** shape, dtype y rango observado quedan registrados.

**PASS:** datos disponibles y consistentes con el environment ID o discrepancia documentada.

---

### AV05 — Interacción corta

**Procedimiento:** ejecutar al menos 100 steps aleatorios o hasta terminación.

**Resultado esperado:** ausencia de errores y captura de rewards/`info`.

**PASS:** loop funcional.

---

### AV06 — Baseline completo

**Procedimiento:** ejecutar al menos 10 episodios aleatorios.

**Resultado esperado:** tabla con ≥10 filas y seeds distintas/registradas.

**PASS:** todos los episodios finalizan válidamente mediante `terminated` o `truncated`.

---

### AV07 — Integridad estadística

**Procedimiento:** recalcular métricas agregadas desde la tabla de episodios.

**Resultado esperado:** media, mediana, desviación, mínimo y máximo corresponden a los datos almacenados.

**PASS:** sin inconsistencias entre tabla y resumen.

---

### AV08 — Densidad de reward

**Procedimiento:** verificar que conteos positivos/cero/negativos sumen el total de steps observados.

**Resultado esperado:** suma exacta.

**PASS:** no existen steps sin clasificación de reward.

---

### AV09 — Frecuencia de acciones

**Procedimiento:** sumar los conteos de las 18 acciones.

**Resultado esperado:** total igual al número total de steps del baseline.

**PASS:** no existen acciones sin contabilizar.

---

### AV10 — Vidas

**Procedimiento:** inspeccionar cambios de `lives` cuando la clave esté disponible.

**Resultado esperado:** pérdidas/ganancias coherentes con los valores observados.

**PASS:** métrica consistente o limitación explícitamente documentada.

---

### AV11 — Visualizaciones

**Procedimiento:** ejecutar las celdas de gráficos.

**Resultado esperado:** tres visualizaciones mínimas se generan sin errores.

**PASS:** recompensa, duración y acciones visibles.

---

### AV12 — Coherencia documental

**Procedimiento:** contrastar conclusiones y actualización de `ficha_tecnica.md` con los datos reales del notebook.

**Resultado esperado:** ninguna afirmación empírica aparece sin evidencia en la ejecución.

**PASS:** trazabilidad notebook → ficha técnica verificable.

---

### AV13 — Independencia de Assault

**Procedimiento:** revisar el diff de HU002.

**Resultado esperado:** ningún archivo bajo `2_Assault/` modificado y ningún import hacia esa ruta.

**PASS:** aislamiento completo.

---

### AV14 — Ejecución Colab

**Procedimiento:** abrir `experimento_0_battlezone.ipynb` en un runtime limpio de Google Colab y ejecutar todas las celdas en orden.

**Resultado esperado:** instalación, inspección y baseline ≥10 episodios finalizan sin cambios manuales al código.

**Criterio en esta fase:** validación **PENDING_COLAB_VALIDATION**. No marcar AV14 como PASS hasta ejecutar el notebook en un runtime limpio de Google Colab y registrar evidencia real.

Instrucciones mínimas para validar AV14:

1. abrir `3_BattleZone/experimento_0_battlezone.ipynb` en Colab limpio;
2. ejecutar todas las celdas en orden;
3. confirmar instalación sin cambios manuales;
4. confirmar ejecución de ≥10 episodios;
5. confirmar generación de tablas, métricas y gráficas;
6. registrar versiones reales del runtime Colab;
7. actualizar evidencia AV14 a PASS solo después de esa ejecución.

---

## 11. Evidencias esperadas

La HU debe producir o referenciar:

- `3_BattleZone/experimento_0_battlezone.ipynb`;
- versiones reales del runtime;
- captura/frame inicial;
- observation/action spaces;
- action meanings;
- claves de `info` observadas;
- tabla de ≥10 episodios;
- tabla/resumen del baseline;
- visualización de recompensa;
- visualización de duración;
- visualización de acciones;
- evidencia visual del radar;
- sección `Conclusiones del Experimento 0`;
- `3_BattleZone/docs/ficha_tecnica.md` actualizada;
- resultado de AV01–AV14.

---

## 12. Riesgos y consideraciones

### R01 — Episodios muy largos

**Riesgo:** una política aleatoria puede generar episodios costosos.

**Mitigación:** medir duración real antes de aumentar el número de episodios; no introducir truncation artificial salvo necesidad técnica documentada porque afectaría el baseline.

### R02 — Reward extremadamente disperso

**Riesgo:** 10 episodios pueden mostrar alta varianza.

**Mitigación:** cumplir mínimo académico y registrar mediana/desviación; si es barato, aumentar episodios sin cambiar protocolo.

### R03 — `info` incompleto

**Riesgo:** algunas métricas de vidas/frames pueden no estar disponibles.

**Mitigación:** documentar la limitación; no inferir desde píxeles en esta HU.

### R04 — Radar degradado por transformación exploratoria

**Riesgo:** sacar conclusiones prematuras de una sola resolución.

**Mitigación:** cualquier resize/grayscale en HU002 es exclusivamente diagnóstico y no configura HU003 automáticamente.

### R05 — Confundir score histórico con reward ALE

**Riesgo:** interpretar valores de Atari como reward exacto del entorno.

**Mitigación:** registrar rewards reales y mantener separación documental.

### R06 — Dependencias Colab cambiantes

**Riesgo:** versiones instaladas por defecto cambian y afectan reproducibilidad.

**Mitigación:** registrar versiones reales y, una vez validadas, utilizarlas como base para fijar dependencias en HU003.

### R07 — Scope creep

**Riesgo:** comenzar preprocessing definitivo o selección de algoritmo dentro del EDA.

**Mitigación:** HU002 solo produce evidencia; decisiones formales corresponden a HU003/HU004.

---

## 13. Definition of Done

HU002 se considera implementada, pendiente únicamente de AV14 en Google Colab, cuando:

- [ ] existe `3_BattleZone/docs/hu002_experimento_0_baseline_aleatorio.md`;
- [ ] existe `3_BattleZone/experimento_0_battlezone.ipynb`;
- [ ] el notebook instala dependencias desde runtime limpio;
- [ ] versiones de Python, Gymnasium y ALE-Py quedan registradas;
- [ ] observation space, action space y action meanings quedan evidenciados;
- [ ] se inspecciona `info` real;
- [ ] se muestra al menos un frame original;
- [ ] se ejecutan ≥10 episodios independientes con política estrictamente aleatoria;
- [ ] cada episodio tiene un registro estructurado;
- [ ] se calculan media, mediana, desviación estándar, mínimo y máximo de recompensa;
- [ ] se calculan métricas de duración;
- [ ] se calcula densidad de rewards;
- [ ] se calcula frecuencia absoluta/relativa de las 18 acciones;
- [ ] vidas y terminación se documentan cuando la API lo permita;
- [ ] existen las visualizaciones mínimas;
- [ ] el radar se inspecciona visualmente;
- [ ] existe `Conclusiones del Experimento 0`;
- [ ] `3_BattleZone/docs/ficha_tecnica.md` está actualizada con hallazgos reales;
- [ ] AV01–AV13 están ejecutadas y aprobadas con evidencia local reproducible;
- [ ] AV14 queda explícitamente marcada como PENDING_COLAB_VALIDATION con instrucciones mínimas de ejecución posterior en Colab;
- [ ] no se utiliza MLflow;
- [ ] no se utiliza TensorBoard;
- [ ] no se implementa entrenamiento ni lógica de agente;
- [ ] no se modifica ni reutiliza código de `2_Assault/`;
- [ ] el cambio se limita al alcance de HU002.

Nota de fase: la validación integral en Colab se mantiene diferida para evitar consumo innecesario de capacidad durante esta etapa exploratoria.

---

## 14. Resultado esperado y gate para HU003

Al cerrar HU002 debe existir el siguiente flujo de evidencia:

```text
HU001 — Caracterización documental
              +
HU002 — Evidencia empírica
              ↓
Baseline aleatorio cuantitativo
              +
Contrato real del entorno
              +
Riesgos visuales observados
              ↓
HU003 — Pipeline reproducible del entorno
```

**Gate para HU003:** no debe fijarse el preprocessing definitivo sin disponer de evidencia de HU002 sobre el frame real, radar, duración, rewards, action space y versiones efectivas del runtime.

HU002 no selecciona el algoritmo. Sus resultados alimentan HU003 y posteriormente HU004.
