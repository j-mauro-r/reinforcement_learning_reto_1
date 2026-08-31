# HU001 — Caracterización técnica y ficha inicial de BattleZone

## 1. Identificación

- **ID:** HU001
- **Nombre:** Caracterización técnica y ficha inicial de BattleZone
- **Estado:** Lista para ejecución
- **Dependencia previa:** Ninguna dentro del flujo BattleZone
- **Habilita:** HU002 — Experimento 0 y baseline aleatorio
- **Fuentes de verdad:**
  - `enunciado_reto_1.txt`
  - `3_BattleZone/docs/implementacion.md`
  - `3_BattleZone/docs/lineamientos.md`
  - `3_BattleZone/docs/arquitectura.md`
  - documentación oficial de Arcade Learning Environment para BattleZone

---

## 2. Contexto y problema

BattleZone es el problema de dificultad alta del Reto 1 y combina observaciones visuales de alta dimensionalidad, perspectiva en primera persona, radar, obstáculos, enemigos con diferentes dinámicas y un espacio discreto de 18 acciones.

Antes de construir un baseline, definir preprocessing o seleccionar algoritmo, el proyecto necesita una caracterización técnica explícita del entorno que distinga:

1. hechos soportados por documentación oficial;
2. decisiones iniciales del proyecto;
3. hipótesis de diseño;
4. información que solo puede confirmarse empíricamente.

Sin esta separación existe riesgo de heredar supuestos de otros entornos Atari, diseñar un preprocessing que destruya información del radar o seleccionar un algoritmo antes de comprender correctamente la dinámica del problema.

El conocimiento adquirido en Assault puede orientar la metodología, pero **no se copiará, importará ni reutilizará código de `2_Assault/`**.

---

## 3. Historia de usuario

> **Como** equipo que desarrollará el agente de Reinforcement Learning para BattleZone, **quiero** disponer de una ficha técnica verificable del entorno, **para** conocer sus observaciones, acciones, dinámica, recompensas, restricciones y preguntas abiertas antes de ejecutar experimentos o diseñar el agente.

---

## 4. Objetivo verificable

Al finalizar HU001 debe existir `3_BattleZone/docs/ficha_tecnica.md` con una caracterización suficiente para que HU002 pueda construir el Experimento 0 sin reinterpretar el entorno.

La ficha deberá:

- identificar `ALE/BattleZone-v5`;
- documentar observaciones y acciones;
- documentar configuración temporal y estocasticidad;
- describir radar, enemigos, obstáculos, vidas y dinámica de juego;
- establecer la métrica principal y el baseline futuro;
- distinguir hechos documentados de información pendiente de validación;
- identificar riesgos para preprocessing y aprendizaje;
- declarar explícitamente las preguntas que HU002 deberá responder.

---

## 5. Alcance

### 5.1 Identificación del entorno

Documentar como mínimo:

- familia Atari/ALE;
- environment ID;
- forma de creación mediante Gymnasium;
- versiones de Gymnasium y ALE-Py que el proyecto adoptará como referencia una vez fijadas para ejecución;
- mode y difficulty disponibles/default;
- `frameskip`;
- `repeat_action_probability`;
- vidas iniciales documentadas.

### 5.2 Observaciones

Documentar:

- `obs_type` disponibles;
- shape, dtype y rango teórico;
- observación RGB por defecto;
- alternativas grayscale y RAM;
- implicaciones del radar y de la vista principal;
- necesidad potencial de información temporal.

No se fijará todavía el preprocessing definitivo.

### 5.3 Action space

Documentar:

- `Discrete(18)`;
- significado de cada una de las 18 acciones;
- diferencia entre acciones simples y combinadas con FIRE;
- implicaciones para exploración y estimación de políticas/Q-values.

No se reducirá artificialmente el action space en HU001.

### 5.4 Dinámica del juego

Describir:

- control del tanque;
- navegación y orientación;
- radar;
- enemigos conocidos;
- obstáculos;
- ataque y evasión;
- dependencia temporal;
- estocasticidad introducida por sticky actions.

### 5.5 Recompensas y scoring

Documentar únicamente valores soportados por fuentes externas claramente identificadas.

Debe diferenciarse entre:

- scoring histórico del videojuego;
- reward entregado por ALE/Gymnasium.

La equivalencia entre ambos **no debe asumirse** y será validada en HU002.

### 5.6 Vidas y terminación

Documentar lo soportado por ALE sobre vidas iniciales y posibles vidas extra.

Identificar como pendiente de HU002:

- comportamiento real de `info["lives"]`;
- relación entre última vida y `terminated`;
- aparición y causas de `truncated`;
- duración real de episodios.

### 5.7 Métrica y baseline del proyecto

Definir como métrica principal:

**recompensa promedio sobre al menos 10 episodios independientes de evaluación.**

Definir como referencia inicial:

**política completamente aleatoria bajo configuración equivalente**, a construir en HU002.

### 5.8 Riesgos iniciales

La ficha debe identificar como mínimo:

- pérdida de información del radar por resize/cropping;
- alto costo de exploración por 18 acciones;
- dependencia temporal;
- costo de entrenamiento visual;
- posible escasez de recompensas;
- riesgos de doble `frameskip`;
- diferencias accidentales entre train/eval;
- necesidad de checkpoints en fases posteriores.

---

## 6. Fuera de alcance

HU001 **no** debe:

- ejecutar el baseline aleatorio;
- entrenar ningún agente;
- seleccionar definitivamente DQN, DQN+PER, DDQN o REINFORCE;
- implementar CNN, Replay Buffer, policy network o trainer;
- fijar hiperparámetros de aprendizaje;
- fijar preprocessing definitivo;
- crear TensorBoard;
- crear checkpoints;
- implementar trazabilidad de runs;
- inferir valores empíricos sin ejecutar el entorno;
- modificar o reutilizar código de Assault.

Estos elementos corresponden a HUs posteriores.

---

## 7. Decisiones y restricciones técnicas

### 7.1 Fuente de evidencia

Los datos técnicos del entorno deben provenir prioritariamente de:

1. documentación oficial ALE/Gymnasium;
2. enunciado del reto;
3. documentación histórica del videojuego cuando sea útil para entender dinámica o scoring.

Cualquier dato no confirmado deberá etiquetarse como hipótesis o pendiente de validación.

### 7.2 Independencia de Assault

Assault se utiliza únicamente como referencia metodológica. Ningún archivo o módulo de `2_Assault/` podrá convertirse en dependencia de HU001 ni de BattleZone.

### 7.3 No seleccionar algoritmo prematuramente

HU001 puede identificar implicaciones para los algoritmos permitidos, pero la selección formal ocurre en HU004 después del baseline y del pipeline reproducible.

### 7.4 Preprocessing abierto

No se asumirá automáticamente grayscale, `84×84`, frame stack de 4 ni cropping por haber funcionado en otros Atari.

La ficha debe explicar qué información estratégica podría perderse, especialmente en el radar.

### 7.5 Versiones

HU001 debe distinguir entre:

- propiedades estables documentadas del environment ID;
- versiones concretas de Gymnasium/ALE-Py que finalmente se instalarán.

Si las versiones aún no han sido ejecutadas en un runtime limpio, deben quedar marcadas como **pendientes de fijación/validación** y no inventarse.

---

## 8. Plan de implementación / tareas

### T01 — Consolidar restricciones académicas

**Archivo:** `3_BattleZone/docs/ficha_tecnica.md`

**Cambio:** documentar algoritmos permitidos, dificultad del problema, necesidad de GPU, evaluación ≥10 partidas y criterio cualitativo de comportamiento lógico.

**Resultado esperado:** la caracterización no contradice `enunciado_reto_1.txt`.

---

### T02 — Caracterizar la interfaz ALE

**Archivo:** `3_BattleZone/docs/ficha_tecnica.md`

**Cambio:** registrar environment ID, observaciones, acciones, modes/difficulty, `frameskip`, sticky actions y vidas documentadas.

**Resultado esperado:** contrato inicial del entorno claramente identificado.

**Depende de:** T01.

---

### T03 — Caracterizar dinámica del juego

**Archivo:** `3_BattleZone/docs/ficha_tecnica.md`

**Cambio:** documentar radar, navegación, enemigos, obstáculos, combate y dependencia temporal.

**Resultado esperado:** explicar por qué BattleZone constituye un problema perceptual y de control más complejo.

---

### T04 — Documentar reward y scoring

**Archivo:** `3_BattleZone/docs/ficha_tecnica.md`

**Cambio:** separar scoring histórico de reward ALE y declarar explícitamente qué debe verificarse empíricamente.

**Resultado esperado:** evitar usar valores no validados como función de recompensa real.

---

### T05 — Definir métrica, baseline y preguntas abiertas

**Archivo:** `3_BattleZone/docs/ficha_tecnica.md`

**Cambio:** formalizar recompensa promedio ≥10 episodios como métrica principal, política aleatoria como baseline y preguntas para HU002.

**Resultado esperado:** HU002 recibe un conjunto concreto de mediciones pendientes.

---

### T06 — Registrar riesgos y decisiones abiertas

**Archivo:** `3_BattleZone/docs/ficha_tecnica.md`

**Cambio:** documentar riesgos del action space, radar, preprocessing, temporalidad y cómputo.

**Resultado esperado:** HU003 y HU004 conocen qué decisiones necesitan evidencia experimental.

---

### T07 — Validar coherencia documental

**Archivos:**

- `3_BattleZone/docs/ficha_tecnica.md`;
- `3_BattleZone/docs/arquitectura.md`;
- `3_BattleZone/docs/implementacion.md`;
- `3_BattleZone/docs/lineamientos.md`.

**Cambio:** revisar rutas, precedencia documental, terminología y ausencia de contradicciones.

**Resultado esperado:** HU001 no introduce decisiones incompatibles con el plan E2E ni con los lineamientos.

---

## 9. Criterios de aceptación

### CA01 — Environment ID

**Dado** el proyecto BattleZone,  
**cuando** se consulta la ficha técnica,  
**entonces** identifica explícitamente `ALE/BattleZone-v5` y su forma de creación con Gymnasium.

### CA02 — Observaciones

**Dado** el entorno documentado,  
**cuando** se revisa el espacio de observaciones,  
**entonces** se describen RGB, grayscale y RAM con shapes, dtype y rango documentados.

### CA03 — Action space completo

**Dado** BattleZone,  
**cuando** se consulta el espacio de acciones,  
**entonces** se documenta `Discrete(18)` y las 18 acciones sin reducirlas artificialmente.

### CA04 — Dinámica temporal

**Dado** `ALE/BattleZone-v5`,  
**cuando** se revisa la ficha,  
**entonces** `frameskip=4` y `repeat_action_probability=0.25` aparecen documentados y se explica su impacto.

### CA05 — Radar y percepción

**Dado** que BattleZone utiliza radar y vista principal,  
**cuando** se analizan las decisiones de preprocessing,  
**entonces** la ficha identifica explícitamente el riesgo de perder información estratégica mediante resize o cropping.

### CA06 — Recompensas

**Dado** que existe scoring histórico y reward ALE,  
**cuando** se consulta la ficha,  
**entonces** ambos conceptos están separados y cualquier equivalencia se marca como pendiente de HU002.

### CA07 — Vidas y terminación

**Dado** lo documentado por ALE,  
**cuando** se revisa la ficha,  
**entonces** se registran vidas conocidas y se identifican como preguntas empíricas `info`, `terminated`, `truncated` y duración.

### CA08 — Métrica y baseline

**Dado** el enunciado del reto,  
**cuando** se revisa la estrategia de evaluación,  
**entonces** la recompensa promedio sobre ≥10 episodios es la métrica principal y la política aleatoria queda definida como baseline futuro.

### CA09 — Información pendiente diferenciada

**Dado** que HU001 es documental,  
**cuando** un dato requiere ejecución,  
**entonces** no se presenta como hecho y queda explícitamente asignado a validación empírica posterior.

### CA10 — Independencia de Assault

**Dado** el conocimiento previo de Assault,  
**cuando** se revisa HU001,  
**entonces** no existe dependencia, importación ni reutilización de código desde `2_Assault/`.

### CA11 — Coherencia documental

**Dado** el conjunto de documentos BattleZone,  
**cuando** se comparan ficha técnica, arquitectura, implementación y lineamientos,  
**entonces** no existen contradicciones materiales sobre algoritmo, MLflow, evaluación, preprocessing o trazabilidad.

### CA12 — Versiones tratadas correctamente

**Dado** que las versiones concretas del runtime pueden no haberse ejecutado todavía,  
**cuando** se documentan Gymnasium y ALE-Py,  
**entonces** solo se fijan valores efectivamente validados; de lo contrario quedan marcados como pendientes.

---

## 10. Autovalidaciones obligatorias

### AV01 — Cobertura de campos de HU001

**Procedimiento:** revisar `ficha_tecnica.md` contra la lista requerida en `implementacion.md`.

**Resultado esperado:** todos los campos obligatorios están presentes o explícitamente marcados como pendientes de validación empírica.

**PASS:** no existe ningún requisito omitido silenciosamente.

---

### AV02 — Validación de action space

**Procedimiento:** contrastar la tabla de acciones de la ficha con la documentación oficial ALE de BattleZone.

**Resultado esperado:** `Discrete(18)` y correspondencia completa de índices/nombres.

**PASS:** 18/18 acciones coinciden.

---

### AV03 — Validación de observaciones

**Procedimiento:** contrastar shapes, dtype y tipos de observación con ALE.

**Resultado esperado:** RGB `(210,160,3)`, grayscale `(210,160)` y RAM `(128,)`, `uint8`, rango teórico `[0,255]`.

**PASS:** no hay diferencias no explicadas.

---

### AV04 — Configuración temporal

**Procedimiento:** contrastar `frameskip`, sticky actions, modes y difficulty con la documentación oficial.

**Resultado esperado:** los valores documentados coinciden con `ALE/BattleZone-v5`.

**PASS:** configuración correcta o discrepancia documentada como pendiente de ejecución.

---

### AV05 — Separación score/reward

**Procedimiento:** revisar la sección de recompensas.

**Resultado esperado:** ningún valor histórico del manual se presenta como reward ALE confirmado.

**PASS:** diferencia explícita y validación asignada a HU002.

---

### AV06 — Preguntas para HU002

**Procedimiento:** revisar la sección de preguntas abiertas.

**Resultado esperado:** incluye como mínimo baseline, densidad de reward, duración, vidas, `info`, terminación/truncation, preprocessing/radar y equivalencia score/reward.

**PASS:** HU002 puede diseñarse directamente a partir de estas preguntas.

---

### AV07 — Coherencia con lineamientos

**Procedimiento:** revisar HU001 contra `lineamientos.md`.

**Resultado esperado:** no usa MLflow, no reutiliza código Assault, no adelanta implementación y mantiene trazabilidad/documentación consistente.

**PASS:** sin contradicciones materiales.

---

### AV08 — Validación de rutas

**Procedimiento:** comprobar que las referencias documentales usan la estructura vigente.

**Resultado esperado:** la ficha técnica y arquitectura se referencian bajo `3_BattleZone/docs/`.

**PASS:** no existen rutas internas obsoletas relevantes para HU001.

---

## 11. Evidencias esperadas

HU001 debe dejar como evidencia:

- `3_BattleZone/docs/ficha_tecnica.md`;
- tabla completa de las 18 acciones;
- tabla de observaciones;
- configuración oficial documentada;
- descripción del radar, enemigos y obstáculos;
- sección de reward/scoring con fuentes y limitaciones;
- riesgos técnicos iniciales;
- lista explícita de preguntas abiertas para HU002;
- resultado documentado de AV01–AV08.

No requiere notebook, GPU, TensorBoard ni ejecución de entrenamiento.

---

## 12. Riesgos y consideraciones

### R01 — Confundir documentación histórica con API ALE

**Riesgo:** asumir que puntuación Atari equivale exactamente al reward retornado por Gymnasium.

**Mitigación:** mantener ambos conceptos separados hasta HU002.

### R02 — Heredar preprocessing de Assault

**Riesgo:** perder información del radar o blancos pequeños.

**Mitigación:** dejar preprocessing abierto y validarlo específicamente para BattleZone.

### R03 — Seleccionar algoritmo demasiado pronto

**Riesgo:** elegir DDQN/DQN+PER/REINFORCE sin evidencia de densidad y dispersión de rewards.

**Mitigación:** reservar selección formal para HU004.

### R04 — Fijar versiones no ejecutadas

**Riesgo:** documentar versiones que luego no coincidan con Colab/local.

**Mitigación:** fijar únicamente versiones verificadas; mantener las demás como pendientes.

### R05 — Scope creep

**Riesgo:** convertir la caracterización en implementación.

**Mitigación:** HU001 permanece documental y no crea lógica del agente.

---

## 13. Definition of Done

HU001 se considera cerrada únicamente cuando:

- [ ] existe `3_BattleZone/docs/hu001_caracterizacion_tecnica_battlezone.md`;
- [ ] existe `3_BattleZone/docs/ficha_tecnica.md`;
- [ ] la ficha documenta `ALE/BattleZone-v5`;
- [ ] action space `Discrete(18)` y las 18 acciones están documentados;
- [ ] observaciones RGB, grayscale y RAM están documentadas;
- [ ] `frameskip` y sticky actions están documentados;
- [ ] modes, difficulty y vidas conocidas están documentados;
- [ ] radar, enemigos, obstáculos y dinámica están descritos;
- [ ] score histórico y reward ALE están claramente diferenciados;
- [ ] la métrica principal y el baseline futuro están definidos;
- [ ] riesgos técnicos y preguntas abiertas para HU002 están documentados;
- [ ] versiones concretas de Gymnasium/ALE-Py están fijadas solo si fueron verificadas o marcadas explícitamente como pendientes;
- [ ] AV01–AV08 están ejecutadas y aprobadas;
- [ ] no existe código ni dependencia reutilizada desde Assault;
- [ ] no se implementó nada fuera del alcance de HU001;
- [ ] no existen contradicciones materiales con `arquitectura.md`, `implementacion.md` o `lineamientos.md`;
- [ ] la evidencia requerida está disponible en la rama/PR.

---

## 14. Resultado esperado y gate para HU002

Al cerrar HU001 debe existir una separación clara:

```text
Hechos documentados de BattleZone
        +
Restricciones del Reto 1
        +
Riesgos y decisiones abiertas
        ↓
Ficha técnica inicial verificable
        ↓
Preguntas empíricas concretas
        ↓
HU002 — Experimento 0 y baseline aleatorio
```

**Gate:** HU002 puede comenzar cuando la caracterización inicial sea suficiente para ejecutar el entorno sin ambigüedades conceptuales sobre observaciones, acciones, configuración temporal, métrica o baseline.

Los datos que por naturaleza requieren ejecución —baseline, contenido real de `info`, distribución de rewards, duración de episodios y comportamiento de terminación— **no bloquean el cierre de HU001** siempre que estén correctamente identificados como objetivos explícitos de HU002.