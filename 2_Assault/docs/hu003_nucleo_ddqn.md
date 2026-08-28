# HU003 — Núcleo DDQN

## 1. Identificación

- **ID:** HU003
- **Nombre:** Núcleo DDQN
- **Estado:** Lista para implementación
- **Dependencia previa:** HU002B — Pipeline de ejecución Local → GitHub → Colab, cerrada temporalmente para permitir avance controlado
- **Habilita:** HU004 — Ciclo de entrenamiento
- **Fuentes de verdad:**
  - `2_Assault/docs/implementacion.md`
  - `2_Assault/docs/arquitectura.md`
  - `2_Assault/docs/linemientos.md`
  - `2_Assault/docs/ficha_tecnica.md`
  - `2_Assault/docs/hu002_pipeline_reproducible_entorno.md`
  - `2_Assault/docs/hu002b_pipeline_ejecucion_local_github_colab.md`
  - `2_Assault/configs/ddqn_config.yaml`
  - `2_Assault/src/environment.py`
  - `2_Assault/requirements.txt`
  - `enunciado_reto_1.txt`

---

## 2. Contexto y problema

HU002 construyó el pipeline reproducible de `ALE/Assault-v5` y dejó un contrato de observación procesada `(4, 84, 84)` en `uint8`, con espacio de acciones `Discrete(7)` y `frameskip=4` aplicado una sola vez.

HU002B añadió el flujo de ejecución Local → GitHub → Colab. Su validación remota puede mantenerse temporalmente pendiente, pero no debe alterarse el contrato técnico ya validado localmente.

El siguiente paso es implementar el **núcleo algorítmico DDQN** sin introducir todavía el ciclo completo de interacción con el entorno.

La arquitectura del proyecto separa responsabilidades:

```text
state (4,84,84)
      ↓
QNetwork
      ↓
7 Q-values

Replay Buffer
      ↓
batch de transiciones
      ↓
DDQN Agent
 ├─ Online Network
 ├─ Target Network
 ├─ epsilon-greedy
 ├─ DDQN target
 ├─ optimizer
 └─ sync target
```

HU003 debe demostrar que estas piezas funcionan correctamente de forma aislada antes de integrarlas con episodios completos en HU004.

---

## 3. Historia de usuario

> **Como** equipo que desarrolla el agente para Assault, **quiero** disponer de una implementación modular y validada del núcleo DDQN, **para** poder integrar posteriormente un ciclo de entrenamiento confiable sin mezclar errores de red, Replay Buffer o actualización del agente con errores del entorno.

---

## 4. Objetivo verificable

Al finalizar HU003 debe existir un núcleo DDQN capaz de:

1. recibir estados `(4, 84, 84)`;
2. convertir `uint8` a tensor `float32` normalizado fuera del entorno;
3. producir exactamente 7 Q-values por estado;
4. mantener Online Network y Target Network independientes;
5. inicializar Target Network con los mismos pesos que Online Network;
6. seleccionar acciones epsilon-greedy válidas;
7. almacenar y muestrear experiencias con Replay Buffer uniforme;
8. calcular targets DDQN separando selección y evaluación de la siguiente acción;
9. ejecutar al menos un paso de optimización de Online Network;
10. demostrar que los pesos de Online cambian después de optimizar;
11. demostrar que Target no cambia durante la optimización normal;
12. sincronizar Target explícitamente desde Online;
13. guardar y cargar el estado básico del agente;
14. funcionar en CPU y, cuando exista, GPU mediante un dispositivo configurable.

Resultado esperado:

```text
batch válido
   ↓
Online + Target
   ↓
DDQN update
   ↓
loss finita
   ↓
pesos Online modificados
   ↓
Target solo cambia con sync explícito
```

---

## 5. Alcance

### 5.1 Dependencias

Agregar PyTorch a `2_Assault/requirements.txt` si todavía no está definido explícitamente.

La versión debe ser compatible con Google Colab y CPU local. Evitar dependencias adicionales si PyTorch cubre la necesidad.

### 5.2 Configuración central

Extender `2_Assault/configs/ddqn_config.yaml` únicamente con los parámetros necesarios para HU003, como mínimo:

```yaml
agent:
  gamma: <valor>
  learning_rate: <valor>
  epsilon_start: <valor>
  epsilon_final: <valor>

replay_buffer:
  capacity: <valor>
  batch_size: <valor>

network:
  input_channels: 4
  num_actions: 7
```

Los valores definitivos deben quedar centralizados en YAML y no repetirse como constantes mágicas.

Parámetros propios del trainer como `learning_starts`, `train_frequency`, `target_update_frequency` o `total_timesteps` pueden reservarse para HU004 cuando no sean necesarios para probar HU003.

### 5.3 `src/network.py`

Crear una Q-Network CNN compatible con entrada `(batch, 4, 84, 84)`.

Responsabilidades:

- aceptar tensores con canales primero;
- normalizar píxeles a `[0,1]` si recibe datos `uint8` o delegar esta normalización a un helper único claramente definido;
- extraer características mediante convoluciones;
- producir salida `(batch, 7)`;
- no seleccionar acciones;
- no calcular loss;
- no ejecutar optimizer;
- no conocer Replay Buffer.

La arquitectura puede seguir el patrón CNN clásico de Atari/DQN siempre que se mantenga simple y documentada.

### 5.4 `src/replay_buffer.py`

Implementar Experience Replay **uniforme**.

Cada transición deberá almacenar como mínimo:

```text
state
 action
 reward
 next_state
 done
```

Requisitos:

- capacidad fija;
- comportamiento FIFO/circular cuando se llena;
- muestreo aleatorio uniforme sin reemplazo dentro de un batch, salvo justificación explícita;
- evitar convertir todo el buffer a `float32` para no multiplicar innecesariamente el uso de RAM;
- almacenar estados visuales preferiblemente como `uint8`;
- devolver batches con shapes y dtypes predecibles;
- no implementar prioridades.

### 5.5 `src/agent.py`

Implementar la lógica propia de DDQN.

Debe contener:

- Online Network;
- Target Network;
- optimizer;
- dispositivo (`cpu`/`cuda`);
- `select_action(...)` epsilon-greedy;
- conversión/preparación de batch;
- cálculo del target DDQN;
- `update(...)` o método equivalente;
- `sync_target_network()`;
- interfaces básicas de `save(...)` y `load(...)`.

### 5.6 Semántica DDQN obligatoria

La siguiente acción del estado siguiente debe ser **seleccionada por Online Network** y su valor debe ser **evaluado por Target Network**.

Conceptualmente:

```text
next_state
   ↓
Online Network
   ↓
argmax action
   ↓
Target Network
   ↓
Q-value de esa acción
   ↓
DDQN target
```

No utilizar `max(Target(next_state))` como sustituto, porque eso correspondería al target clásico de DQN y eliminaría la característica principal de DDQN.

### 5.7 Terminales

Las transiciones con `done=True` no deben bootstrappear el valor del siguiente estado.

El contrato debe poder ampliarse posteriormente para diferenciar `terminated` y `truncated` si HU004 lo necesita, evitando por ahora complejidad innecesaria.

### 5.8 Save/load básico

HU003 debe demostrar serialización mínima de:

- Online Network;
- Target Network;
- optimizer;
- configuración mínima necesaria del agente cuando corresponda.

La persistencia completa para reanudar sesiones, Replay Buffer y timestep global pertenece a HU005.

### 5.9 Tests

Crear tests focalizados para network, Replay Buffer y agent.

Preferir archivos como:

```text
2_Assault/tests/test_network.py
2_Assault/tests/test_replay_buffer.py
2_Assault/tests/test_agent.py
```

o una agrupación menor si evita duplicación y mantiene claridad.

---

## 6. Fuera de alcance

HU003 **no** debe implementar:

- ciclo completo `env.step()` de entrenamiento;
- trainer por timesteps;
- `learning_starts` operativo sobre episodios;
- decay temporal completo de epsilon;
- frecuencia automática de entrenamiento;
- frecuencia automática de sincronización Target;
- checkpoints completos/resume;
- persistencia del Replay Buffer;
- TensorBoard;
- MLflow;
- evaluación formal;
- video;
- búsqueda de hiperparámetros;
- Prioritized Experience Replay;
- Dueling DQN;
- Noisy Nets;
- n-step returns;
- Rainbow;
- otras extensiones no definidas en el reto.

HU003 debe limitarse al **núcleo DDQN verificable**.

---

## 7. Decisiones y restricciones técnicas

### 7.1 Framework

Usar **PyTorch** para las redes y optimización.

### 7.2 Entrada de la red

Contrato proveniente de HU002:

```text
shape: (4, 84, 84)
dtype: uint8
```

Para batches:

```text
(batch_size, 4, 84, 84)
```

La normalización a `[0,1]` debe ocurrir en la capa de red/agente o preparación de batch, nunca modificando el contrato del entorno.

### 7.3 Salida

Para Assault:

```text
num_actions = 7
```

La red debe producir:

```text
(batch_size, 7)
```

### 7.4 Target Network

- inicializar con copia exacta de Online Network;
- no recibir gradientes durante el update normal;
- mantenerse en modo apropiado de inferencia cuando aplique;
- cambiar únicamente mediante sincronización explícita en HU003.

### 7.5 Optimizer

Usar un optimizer estándar compatible con DDQN, preferiblemente Adam si no existe decisión previa distinta.

El learning rate debe provenir del YAML.

### 7.6 Loss

Usar una pérdida apropiada para regresión de Q-values. Huber/Smooth L1 es preferible por robustez y uso habitual en DQN/DDQN; si se utiliza MSE debe documentarse la razón.

### 7.7 Replay Buffer y memoria

Los estados visuales no deben almacenarse como `float32` por defecto.

Ejemplo conceptual:

```text
uint8 state       ~ 4x menos memoria que float32
```

La conversión a tensor normalizado debe hacerse al muestrear/preparar batches.

### 7.8 Epsilon-greedy

`select_action` debe aceptar epsilon como parámetro explícito o usar una configuración clara.

Comportamiento verificable:

- `epsilon=0`: selección greedy;
- `epsilon=1`: acción aleatoria válida;
- `0<epsilon<1`: mezcla estocástica.

El scheduling temporal de epsilon pertenece a HU004.

### 7.9 Seeds

Reutilizar la estrategia de reproducibilidad del proyecto para NumPy/Python/PyTorch cuando corresponda.

No prometer determinismo absoluto en GPU si PyTorch/ALE no lo garantizan.

### 7.10 SOLID/DRY

- `network.py` conoce arquitectura neuronal, no reglas de entrenamiento;
- `replay_buffer.py` conoce almacenamiento, no redes;
- `agent.py` coordina redes y optimizer, no episodios;
- `trainer.py` permanece fuera de HU003;
- evitar helpers genéricos que solo se usan una vez;
- evitar duplicar conversión de estados entre varios módulos.

### 7.11 Docstrings

Funciones y clases reutilizables deben usar docstrings estilo Google según `linemientos.md`.

---

## 8. Plan de implementación / tareas

### T01 — Extender dependencias

**Archivo:** `2_Assault/requirements.txt`

Agregar PyTorch de forma compatible con local/Colab.

**Resultado:** `import torch` funciona en el entorno del proyecto.

### T02 — Extender configuración DDQN

**Archivo:** `2_Assault/configs/ddqn_config.yaml`

Agregar parámetros requeridos por network, replay buffer y agent.

**Resultado:** todos los valores relevantes de HU003 provienen del YAML.

### T03 — Implementar QNetwork

**Archivo:** `2_Assault/src/network.py`

Implementar CNN con salida de 7 Q-values.

**Resultado:** forward pass correcto para batches sintéticos y observaciones reales procesadas.

### T04 — Implementar ReplayBuffer

**Archivo:** `2_Assault/src/replay_buffer.py`

Implementar buffer uniforme de capacidad fija.

**Resultado:** append y sample producen batches consistentes.

### T05 — Implementar inicialización DDQNAgent

**Archivo:** `2_Assault/src/agent.py`

Crear Online, Target, optimizer y selección de dispositivo.

**Resultado:** Online y Target empiezan con pesos idénticos, pero son objetos independientes.

### T06 — Implementar epsilon-greedy

Implementar selección de acción válida para `epsilon ∈ [0,1]`.

**Resultado:** greedy y random funcionan bajo casos controlados.

### T07 — Implementar target DDQN

Implementar selección con Online y evaluación con Target.

**Resultado:** test específico diferencia el cálculo DDQN del target DQN clásico.

### T08 — Implementar update

Preparar batch, calcular predicción, target, loss, backward y optimizer step.

**Resultado:** loss finita y pesos Online cambian.

### T09 — Implementar sincronización Target

Implementar copia explícita Online → Target.

**Resultado:** Target permanece estable antes de sync y coincide con Online después.

### T10 — Implementar save/load básico

Guardar/restaurar estado mínimo del agente.

**Resultado:** predicciones antes de guardar y después de cargar coinciden para el mismo estado.

### T11 — Agregar tests unitarios/focalizados

Cubrir network, buffer y agent.

### T12 — Smoke con observación real

Crear un entorno mediante `src/environment.py`, obtener una observación procesada y ejecutar inferencia/selección de acción sin entrenamiento largo.

### T13 — Actualizar documentación de implementación

Solo cuando existan resultados reales, registrar evidencia y estado de HU003 en `2_Assault/docs/implementacion.md`.

---

## 9. Criterios de aceptación

### CA01 — QNetwork

**Dado** un batch de estados `(N,4,84,84)`, **cuando** pasa por QNetwork, **entonces** la salida tiene shape `(N,7)` y valores finitos.

### CA02 — Compatibilidad uint8

**Dado** un estado `uint8` del entorno, **cuando** es preparado para inferencia, **entonces** se convierte correctamente a `float32` normalizado sin modificar el contrato del entorno.

### CA03 — Independencia de redes

**Dado** un agente recién inicializado, **cuando** se comparan Online y Target, **entonces** sus pesos tienen los mismos valores pero no comparten referencias de parámetros.

### CA04 — Replay Buffer uniforme

**Dadas** transiciones válidas, **cuando** se agregan y muestrean, **entonces** el buffer respeta capacidad, shapes, dtypes y muestreo uniforme sin prioridades.

### CA05 — Buffer circular

**Dada** una capacidad `C`, **cuando** se insertan más de `C` transiciones, **entonces** el tamaño permanece en `C` y las experiencias antiguas son reemplazadas de forma controlada.

### CA06 — Acción greedy

**Dado** `epsilon=0`, **cuando** se selecciona acción, **entonces** corresponde al `argmax` de Online Network.

### CA07 — Acción aleatoria

**Dado** `epsilon=1`, **cuando** se seleccionan acciones, **entonces** todas pertenecen al rango `[0,6]` y no se fuerza el `argmax` greedy.

### CA08 — Target DDQN

**Dado** un batch controlado donde Online y Target prefieren acciones diferentes, **cuando** se calcula el target, **entonces** la acción siguiente se selecciona con Online y su valor se toma de Target.

### CA09 — Terminal

**Dada** una transición terminal, **cuando** se calcula target, **entonces** no se agrega valor bootstrap del siguiente estado.

### CA10 — Optimización

**Dado** un batch válido, **cuando** se ejecuta `update`, **entonces** retorna una loss finita y al menos un parámetro de Online Network cambia.

### CA11 — Target estable

**Dado** un `update` normal, **cuando** finaliza, **entonces** Target Network no cambia automáticamente.

### CA12 — Sync

**Dado** que Online y Target tienen pesos diferentes, **cuando** se ejecuta `sync_target_network`, **entonces** sus parámetros vuelven a coincidir.

### CA13 — Save/load

**Dado** un agente guardado, **cuando** se carga en una instancia compatible, **entonces** restaura Online, Target y optimizer y conserva predicciones equivalentes para un estado fijo.

### CA14 — CPU/GPU

**Dado** un dispositivo disponible, **cuando** se inicializa el agente, **entonces** las redes y batches utilizan el dispositivo configurado sin mezclar CPU/GPU.

### CA15 — Integración con HU002

**Dada** una observación real producida por `create_assault_env`, **cuando** se entrega al agente, **entonces** puede generar una acción válida sin adaptar manualmente shape/dtype fuera del contrato definido.

### CA16 — Sin scope creep

**Dado** el PR HU003, **cuando** se revisa, **entonces** no contiene trainer completo, TensorBoard, MLflow, checkpoint/resume completo, PER ni extensiones de DQN no aprobadas.

---

## 10. Autovalidaciones obligatorias

### AV01 — Imports

**Procedimiento:**

```bash
python -c "from 2_Assault.src.network import QNetwork"
```

Adaptar el comando a la estructura real de imports del proyecto si el nombre de paquete impide usar `2_Assault` directamente.

**Esperado:** imports limpios sin errores.

### AV02 — Suite de tests

**Procedimiento:**

```bash
python -m pytest 2_Assault/tests -q
```

**Esperado:** todos los tests previos + HU003 pasan.

### AV03 — Forward pass

**Procedimiento:** batch sintético `uint8` de shape `(2,4,84,84)`.

**Esperado:** salida `(2,7)`, dtype float y valores finitos.

### AV04 — Observación real

**Procedimiento:** crear Assault mediante fábrica HU002, obtener estado y pasarlo por QNetwork.

**Esperado:** salida `(1,7)` sin manipulación ad hoc del entorno.

### AV05 — Replay Buffer

**Procedimiento:** llenar por encima de su capacidad reducida de prueba y muestrear batch.

**Esperado:** capacidad respetada, shapes/dtypes correctos y ausencia de prioridades.

### AV06 — Epsilon-greedy

**Procedimiento:** probar epsilon `0.0` y `1.0` con red controlada.

**Esperado:** greedy correcto y acciones aleatorias siempre válidas.

### AV07 — DDQN target

**Procedimiento:** usar redes controladas/mockeadas con rankings distintos entre Online y Target.

**Esperado:** selección Online + evaluación Target demostrable.

### AV08 — Update real

**Procedimiento:** ejecutar un optimizer step sobre batch válido.

**Esperado:** loss finita, gradientes válidos, Online cambia y Target permanece igual.

### AV09 — Sync Target

**Procedimiento:** modificar Online mediante update y luego sincronizar.

**Esperado:** parámetros Target == Online después del sync.

### AV10 — Save/load

**Procedimiento:** guardar agente en archivo temporal, cargar en una nueva instancia y comparar outputs.

**Esperado:** outputs equivalentes dentro de tolerancia numérica y optimizer restaurado.

### AV11 — Device

**Procedimiento:** ejecutar en CPU obligatoriamente y GPU si está disponible durante validación Colab futura.

**Esperado:** no existen errores de device mismatch.

La validación GPU no bloquea HU003 si no hay GPU local; debe registrarse como validación Colab futura, sin inventar resultados.

---

## 11. Evidencias requeridas

El PR HU003 debe incluir o referenciar:

- salida completa de `pytest`;
- versión de PyTorch utilizada;
- configuración añadida al YAML;
- shape de entrada y salida de QNetwork;
- evidencia de observación real `(4,84,84)` → 7 Q-values;
- prueba de Replay Buffer;
- prueba específica DDQN selection/evaluation;
- loss de un update real;
- evidencia de cambio de pesos Online;
- evidencia de Target estable antes de sync;
- evidencia de igualdad Online/Target después de sync;
- prueba save/load;
- dispositivo utilizado;
- commit Git;
- confirmación explícita de ausencia de PER y scope creep.

No se requiere entrenamiento prolongado ni evidencia de mejora de reward en HU003.

---

## 12. Definition of Done

HU003 se considera terminada únicamente cuando:

- [ ] PyTorch está definido en dependencias del proyecto;
- [ ] configuración necesaria está centralizada en YAML;
- [ ] `network.py` implementado;
- [ ] `replay_buffer.py` implementado;
- [ ] `agent.py` implementado;
- [ ] QNetwork produce 7 Q-values;
- [ ] Online y Target se inicializan correctamente;
- [ ] Replay Buffer uniforme funciona;
- [ ] epsilon-greedy funciona;
- [ ] target DDQN está probado explícitamente;
- [ ] terminales no bootstrappean;
- [ ] un update real produce loss finita;
- [ ] Online cambia durante update;
- [ ] Target no cambia durante update normal;
- [ ] sincronización Target funciona;
- [ ] save/load básico funciona;
- [ ] observación real HU002 es consumida correctamente;
- [ ] tests previos continúan pasando;
- [ ] nuevos tests HU003 pasan;
- [ ] no existen errores bloqueantes conocidos;
- [ ] documentación/evidencia está actualizada;
- [ ] PR está limitado a HU003;
- [ ] no se implementó PER ni extensiones fuera del alcance.

---

## 13. Riesgos y consideraciones

### 13.1 Uso de memoria

Replay Buffer visual puede consumir varios GB si se dimensiona agresivamente. Mantener estados como `uint8` y evitar copias innecesarias.

Los tests deben utilizar capacidades pequeñas; no asignar el buffer final completo solo para probar HU003.

### 13.2 Shape de tensores

La convención del proyecto es channels-first `(4,84,84)`. Evitar transposiciones silenciosas o múltiples formatos internos.

### 13.3 Device mismatch

Todos los tensores usados en el cálculo de loss deben moverse de forma consistente al dispositivo del agente.

### 13.4 Gradientes sobre Target

El cálculo del target debe hacerse sin construir gradientes sobre Target Network.

### 13.5 Diferencia DQN vs DDQN

Un error crítico sería usar Target Network tanto para seleccionar como para evaluar la siguiente acción. Debe existir un test diseñado específicamente para detectar este error.

### 13.6 Save/load incompleto

HU003 solo requiere persistencia básica del agente. No anticipar el esquema completo de resume de HU005.

### 13.7 HU002B temporalmente pendiente

El avance a HU003 se realiza por decisión explícita del proyecto. La validación Colab pendiente de HU002/HU002B no debe reinterpretarse como aprobada ni borrarse de la documentación.

---

## 14. Resultado esperado y gate para HU004

HU004 solo debe comenzar cuando HU003 demuestre:

```text
HU002 observation
      ↓
QNetwork → 7 Q-values
      ↓
Replay Buffer → valid batch
      ↓
Online selects next action
      ↓
Target evaluates it
      ↓
finite DDQN loss
      ↓
Online weights updated
      ↓
Target stable
      ↓
explicit sync works
      ↓
HU003 PASS
```

No es necesario demostrar todavía que el agente mejora su recompensa en Assault. Ese comportamiento comienza a evaluarse una vez exista el ciclo de entrenamiento de HU004 y los smoke tests posteriores.