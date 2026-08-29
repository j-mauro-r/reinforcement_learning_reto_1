# HU009 — Entrenamiento DDQN completo

## 1. Identificación

- **ID:** HU009
- **Nombre:** Entrenamiento DDQN completo
- **Estado:** PENDIENTE
- **Dependencia previa:** HU008 — MLflow y trazabilidad de experimentos.
- **Dependencia operacional:** HU008B — automatización de arranque y reanudación de experimentos.
- **Habilita:** HU010 — Optimización controlada de hiperparámetros.
- **Entorno objetivo:** Google Colab GPU.
- **Persistencia:** Google Drive para checkpoints, TensorBoard, MLflow y estado de experimentos.
- **Fuente de verdad de código:** GitHub mediante ref/commit explícito.
- **Algoritmo:** DDQN con Replay Buffer uniforme. No usar PER ni cambiar de algoritmo dentro de HU009.

---

## 2. Contexto

Las HUs anteriores construyeron y validaron de forma incremental:

```text
entorno reproducible
        ↓
DDQN
        ↓
Trainer
        ↓
checkpoints + resume
        ↓
TensorBoard
        ↓
smoke E2E
        ↓
MLflow
        ↓
automatización new/resume
        ↓
HU009: primer entrenamiento DDQN completo
```

HU009 es el primer punto del proyecto donde se permite consumir cómputo de forma prolongada para obtener un agente entrenado. Su objetivo no es todavía optimizar hiperparámetros ni realizar la comparación formal final contra el baseline. Es producir **un primer modelo DDQN completo, reproducible y trazable**, usando exactamente la arquitectura ya validada.

TensorBoard continúa siendo la herramienta de observación durante el entrenamiento; MLflow conserva identidad, configuración, métricas agregadas y artefactos; los checkpoints permiten continuidad entre runtimes; GitHub determina el código ejecutado.

---

## 3. Historia de usuario

> **Como** equipo que debe entrenar un agente DDQN para `ALE/Assault-v5`, **quiero** ejecutar el primer entrenamiento completo sobre GPU usando el pipeline reproducible, reanudable y trazable construido en las HUs anteriores, **para** obtener un modelo candidato que pueda analizarse posteriormente en HU010 y compararse formalmente contra el baseline en HU011.

---

## 4. Objetivo verificable

HU009 debe demostrar que un experimento DDQN de duración sustancial puede:

1. iniciarse desde un runtime limpio de Colab;
2. ejecutar el SHA esperado de GitHub;
3. usar GPU;
4. entrenar con la configuración declarada;
5. persistir checkpoints en Google Drive;
6. registrar TensorBoard de forma continua;
7. registrar el experimento en MLflow;
8. sobrevivir a la destrucción del runtime;
9. reanudarse automáticamente desde otro runtime usando HU008B;
10. conservar el mismo `project_run_id` y `mlflow_run_id`;
11. restaurar pesos, optimizer, global step y Replay Buffer mediante `resume_full`;
12. continuar hasta el target global configurado;
13. generar un checkpoint final válido;
14. ejecutar una evaluación técnica posterior sin modificar el agente;
15. dejar evidencia suficiente para decidir si procede HU010.

HU009 **no exige todavía superar formalmente el baseline**. Esa comparación pertenece a HU011. Sin embargo, el entrenamiento debe producir métricas suficientemente informativas para detectar si el agente está aprendiendo o si existe una falla técnica evidente.

---

## 5. Gate G0 — cierre integrado de HU008B

Por decisión del proyecto, HU008B no requiere una ejecución separada antes de HU009. **La primera prueba de HU009 será también la prueba de cierre de HU008B.**

Antes de iniciar entrenamiento prolongado, ejecutar una validación barata en dos runtimes Colab independientes utilizando el flujo automático.

### Runtime A

Entrada manual permitida:

```text
project_run_id = hu009_preflight_001
target_timesteps = 48
requested_mode = auto
```

Resultado esperado:

```text
SESSION_BOOTSTRAP_READY=True
tracking_mode=new
tracking_session_id=session_001
mlflow_run_id=<generado automáticamente>
checkpoint_input=None
initial_global_step=0
final_global_step=48
MLFLOW_TRACKING_PASS=True
manifest_updated=True
```

Eliminar completamente el runtime después de confirmar el checkpoint y el manifest.

### Runtime B limpio

Entrada manual permitida:

```text
project_run_id = hu009_preflight_001
target_timesteps = 64
requested_mode = auto
```

El usuario **no debe introducir manualmente**:

- `mlflow_run_id`;
- `tracking_session_id`;
- checkpoint path;
- `tracking_mode=resume`.

Resultado esperado:

```text
SESSION_BOOTSTRAP_READY=True
tracking_mode=resume
tracking_session_id=session_002
checkpoint_input_loaded=True
restored_expected_step=48
restored_global_step=48
replay_buffer_restored=True
initial_global_step=48
final_global_step=64
MULTISESSION_CHECKPOINT_RESUME_PASS=True
MLFLOW_TRACKING_PASS=True
manifest_updated=True
```

### Regla de cierre HU008B

Si ambos runtimes pasan todos los gates anteriores:

```text
HU008B → [COMPLETADA]
```

Actualizar `2_Assault/docs/implementacion.md` con la evidencia real antes de continuar al entrenamiento prolongado de HU009.

Si este gate falla, **HU009 debe detenerse**. No iniciar entrenamiento largo hasta corregir HU008B y repetir G0.

---

## 6. Configuración base del entrenamiento

La configuración efectiva debe partir de:

```text
2_Assault/configs/ddqn_config.yaml
```

Contrato existente que no debe cambiar silenciosamente:

```yaml
environment:
  id: ALE/Assault-v5
  obs_type: rgb
  frame_skip: 4
  repeat_action_probability: 0.25

preprocessing:
  grayscale: true
  resize_height: 84
  resize_width: 84
  frame_stack: 4
  dtype: uint8

reproducibility:
  seed: 42

network:
  input_channels: 4
  num_actions: 7

agent:
  gamma: 0.99
  learning_rate: 0.0001
  epsilon_start: 1.0
  epsilon_final: 0.01
```

Los valores actuales de smoke/test (`total_timesteps=48`, Replay Buffer pequeño, intervalos cortos, etc.) **no representan la configuración productiva de HU009**. HU009 debe introducir una configuración explícita para entrenamiento completo sin destruir los perfiles de smoke existentes.

Preferir una separación clara, por ejemplo:

```yaml
training_profiles:
  smoke:
    ...
  full:
    ...
```

O una solución equivalente que mantenga una única fuente de verdad y evite editar manualmente valores cada vez que se ejecuta el notebook.

---

## 7. Presupuesto de entrenamiento

El target de HU009 debe ser explícito, trazable y configurable.

No hardcodear el presupuesto directamente dentro del Trainer o del notebook.

Debe existir un valor equivalente a:

```text
FULL_TRAINING_TARGET_TIMESTEPS
```

El valor definitivo debe quedar documentado antes de iniciar la corrida prolongada, junto con la justificación de costo/tiempo observada en Colab.

La arquitectura debe permitir dividir el target global en varias sesiones sin cambiar la identidad lógica del experimento:

```text
project_run_id = assault_ddqn_full_001

session_001: 0 → N1
session_002: N1 → N2
session_003: N2 → target
```

Cada sesión debe continuar desde el último checkpoint válido mediante HU008B.

---

## 8. Política de checkpoints

Para HU009 los checkpoints dejan de ser únicamente una evidencia de smoke y pasan a ser un mecanismo operacional.

Requisitos:

- persistencia obligatoria en Google Drive;
- `save_replay_buffer=true` para permitir `resume_full`;
- frecuencia configurable razonable para no perder una porción significativa de entrenamiento si Colab desconecta el runtime;
- checkpoint al finalizar normalmente cada sesión;
- checkpoint final inequívoco al alcanzar el target global;
- nombre asociado al mismo `project_run_id`;
- no seleccionar checkpoints por heurística global;
- HU008B debe resolver automáticamente el checkpoint declarado por el manifest;
- validar físicamente el checkpoint antes de actualizar el manifest.

No reducir o cambiar Replay Buffer para hacer compatible un checkpoint existente.

---

## 9. Replay Buffer y memoria

La capacidad utilizada en HU009 debe definirse para entrenamiento real y no heredarse accidentalmente del perfil de smoke.

Antes de lanzar la corrida prolongada:

1. estimar memoria RAM requerida por el Replay Buffer con observaciones `(4, 84, 84)` `uint8`;
2. verificar RAM disponible del runtime Colab;
3. mantener margen suficiente para entorno, Python, PyTorch, MLflow y estructuras auxiliares;
4. documentar la capacidad seleccionada;
5. incluir dicha capacidad en el fingerprint de HU008B;
6. mantenerla inmutable durante todas las sesiones del mismo `project_run_id`.

No realizar cambios silenciosos de capacidad entre sesiones.

---

## 10. Política de exploración

HU009 debe utilizar el contrato de epsilon definido por la configuración.

Registrar como mínimo:

- `epsilon_start`;
- `epsilon_final`;
- `epsilon_decay_steps`;
- epsilon observado durante entrenamiento.

La política de exploración debe estar definida respecto del **global step**, de modo que una reanudación continúe la curva de epsilon y no reinicie la exploración desde cero.

Cambiar la política de exploración durante la misma corrida invalida la continuidad del experimento.

---

## 11. TensorBoard

Durante HU009 TensorBoard debe permitir observar como mínimo:

```text
train/epsilon
train/loss
train/q_mean
train/learning_rate
```

Y, cuando el Trainer produzca episodios completos:

```text
episode/reward
rolling reward
longitud del episodio
```

Requisitos:

- logs persistentes entre runtimes;
- global steps monotónicos después de resume;
- logs anteriores preservados;
- ninguna segunda sesión debe reiniciar las curvas en step 0;
- documentar cualquier señal de inestabilidad: loss no finita, Q-values explosivos, recompensa degradándose de forma anómala, etc.

TensorBoard se usa para diagnóstico intra-run; no reemplaza MLflow.

---

## 12. MLflow

Todo entrenamiento completo debe quedar asociado a un único experimento lógico y un único MLflow run mientras represente continuidad del mismo entrenamiento.

Identidad esperada:

```text
project_run_id = assault_ddqn_full_001
mlflow_run_id = <uno por entrenamiento lógico>
tracking_session_id = session_001, session_002, ...
```

Cada sesión debe registrar:

- Git SHA/ref;
- runtime y GPU;
- configuración efectiva;
- target de sesión;
- initial/final global step;
- checkpoint input/output;
- resumen de entrenamiento;
- evaluación técnica disponible;
- artefactos de sesión;
- fingerprint de configuración.

No crear un nuevo `mlflow_run_id` únicamente porque Colab creó un runtime nuevo.

---

## 13. Entrenamiento por sesiones

El notebook debe permitir operar el entrenamiento largo únicamente con intención de alto nivel, por ejemplo:

```text
ASSAULT_PROJECT_RUN_ID=assault_ddqn_full_001
ASSAULT_TARGET_TIMESTEPS=<target global de esta sesión>
ASSAULT_REQUESTED_MODE=auto
```

HU008B debe resolver automáticamente:

- `new` o `resume`;
- `mlflow_run_id`;
- `tracking_session_id`;
- checkpoint de entrada;
- restored expected step;
- rutas persistentes.

Antes de cada sesión debe imprimirse `SESSION_BOOTSTRAP_READY=True`.

Si el bootstrap detecta inconsistencia, el entrenamiento debe abortar antes de modificar el experimento.

---

## 14. Evaluación técnica durante HU009

HU009 puede ejecutar evaluaciones técnicas cortas para comprobar integridad del agente y observar tendencia, pero estas **no sustituyen HU011**.

La evaluación debe:

- utilizar el entorno de evaluación común;
- usar `epsilon=0.0` salvo decisión documentada;
- no actualizar pesos;
- no modificar optimizer;
- no modificar Replay Buffer;
- no alterar global step de entrenamiento;
- registrar raw reward;
- quedar asociada al checkpoint evaluado.

No declarar éxito del reto únicamente por una evaluación corta de HU009.

---

## 15. Protección del cómputo

Antes de cualquier sesión prolongada deben pasar gates baratos.

### Preflight obligatorio

Confirmar:

```text
READY_FOR_TRAINING=True
runtime=Google Colab
device=cuda
observation=(4,84,84) uint8
action_space=Discrete(7)
config fingerprint válido
session bootstrap válido
tracking store accesible
checkpoint root accesible
TensorBoard root accesible
```

Si GPU no está disponible, no iniciar la sesión prolongada salvo decisión explícita documentada.

### Abort conditions

Abortar ante:

- NaN/Inf en loss;
- checkpoint incompatible;
- Replay Buffer incompatible;
- MLflow identity mismatch;
- manifest inconsistente;
- target global no mayor al restored step;
- pérdida del código SHA esperado;
- resume sin restauración real;
- error de persistencia en Drive;
- error crítico de memoria.

---

## 16. Idempotencia

Reejecutar accidentalmente una celda no debe:

- iniciar otro entrenamiento sobre la misma sesión ya registrada;
- sobrescribir artefactos de una sesión cerrada;
- crear un nuevo MLflow run para el mismo resume;
- volver a step 0;
- actualizar el manifest con estado incompleto.

La protección de colisiones implementada en HU008/HU008B se mantiene obligatoria.

---

## 17. Artefactos esperados

HU009 debe producir al menos:

```text
Google Drive/
└── reinforcement_learning_reto_1/
    ├── checkpoints/
    │   └── assault_ddqn_full_001/
    │       └── checkpoint_step_<TARGET>.pt
    ├── tensorboard/
    │   └── ...
    ├── mlruns/
    │   └── ...
    └── experiments/
        └── assault_ddqn_full_001/
            └── experiment_state.json
```

Y en MLflow:

```text
sessions/session_001/...
sessions/session_002/...
...
```

El repositorio no debe versionar checkpoints binarios, `mlruns`, TensorBoard logs ni artefactos de Drive.

---

## 18. Tests y autovalidaciones obligatorias

Antes de ejecutar entrenamiento prolongado:

1. `python -m compileall -q 2_Assault/src`;
2. tests focales HU008B;
3. suite completa `2_Assault/tests`;
4. `git diff --check`;
5. validación estática del notebook;
6. Gate G0 HU009/HU008B en dos runtimes Colab.

Durante/después del entrenamiento validar programáticamente:

7. `initial_global_step` correcto;
8. `final_global_step` igual al target de la sesión;
9. checkpoints existentes y no vacíos;
10. global step monotónico entre sesiones;
11. misma identidad `project_run_id`;
12. mismo `mlflow_run_id` en resumes;
13. `tracking_session_id` único por sesión;
14. Replay Buffer restaurado en `resume_full`;
15. TensorBoard contiene steps posteriores al resume;
16. artefactos MLflow de sesiones anteriores siguen existiendo;
17. manifest apunta al último checkpoint confirmado;
18. manifest no se modifica tras una sesión fallida;
19. evaluación no altera estado de entrenamiento;
20. checkpoint final puede cargarse en un runtime limpio.

---

## 19. Criterios de aceptación

HU009 se considera completada únicamente si:

- [ ] Gate G0 pasó en dos runtimes independientes y HU008B quedó `[COMPLETADA]`;
- [ ] existe una configuración explícita para entrenamiento completo separada del smoke;
- [ ] el presupuesto global del entrenamiento está documentado;
- [ ] la capacidad del Replay Buffer fue dimensionada y documentada;
- [ ] la corrida usa GPU;
- [ ] Git SHA/ref ejecutado está registrado;
- [ ] el entrenamiento utiliza DDQN sin cambiar de algoritmo;
- [ ] TensorBoard registra métricas durante toda la corrida;
- [ ] MLflow registra el experimento y todas sus sesiones;
- [ ] los checkpoints persisten en Drive;
- [ ] al menos una reanudación real entre runtimes independientes fue validada;
- [ ] el mismo experimento conserva el mismo `mlflow_run_id` durante resume;
- [ ] pesos, optimizer, global step y Replay Buffer se restauran correctamente;
- [ ] el global step alcanza el target configurado;
- [ ] existe checkpoint final cargable;
- [ ] existe evaluación técnica del checkpoint final;
- [ ] no existen NaN/Inf ni errores técnicos que invaliden la corrida;
- [ ] todas las suites/tests relevantes permanecen en verde;
- [ ] la evidencia queda documentada en `implementacion.md`;
- [ ] el resultado habilita análisis/optimización en HU010.

---

## 20. Definition of Done

HU009 puede marcarse `[COMPLETADA]` cuando exista evidencia reproducible de:

```text
GitHub SHA conocido
        ↓
Colab GPU
        ↓
SESSION_BOOTSTRAP_READY=True
        ↓
DDQN full training
        ↓
TensorBoard + checkpoints + MLflow
        ↓
runtime independiente / resume real
        ↓
mismo project_run_id + mlflow_run_id
        ↓
checkpoint final válido
        ↓
evaluación técnica
        ↓
HU009_COMPLETED=True
```

No avanzar a HU010 si la corrida terminó por error técnico, si el checkpoint final no puede cargarse o si la trazabilidad multisesión quedó inconsistente.

---

## 21. Evidencia obligatoria

Documentar al cierre:

- branch y commit SHA ejecutados;
- `project_run_id`;
- `mlflow_run_id`;
- lista de `tracking_session_id`;
- GPU utilizada;
- configuración efectiva;
- fingerprint;
- target total;
- steps inicial/final por sesión;
- duración por sesión y total;
- checkpoints input/output;
- tamaño de checkpoints;
- Replay Buffer capacity y evidencia de restore;
- TensorBoard event files/tags relevantes;
- métricas de entrenamiento disponibles;
- evaluación técnica final: episodios, epsilon, rewards, mean reward;
- evidencia de consulta MLflow;
- estado final del manifest;
- resultados de tests;
- cualquier incidente/reanudación ocurrida.

---

## 22. Fuera de alcance

HU009 no debe:

- implementar PER;
- cambiar DDQN por otro algoritmo;
- realizar búsqueda de hiperparámetros;
- declarar el mejor modelo entre múltiples configuraciones;
- ejecutar todavía la comparación formal final contra baseline;
- modificar la definición de la métrica final;
- introducir serving/deployment;
- agregar infraestructura MLOps remota innecesaria.

Estos temas corresponden a HU010, HU011 o trabajos posteriores.

---

## 23. Habilita

Una HU009 completada habilita:

```text
HU010 — Optimización controlada de hiperparámetros
```

HU010 debe partir del primer entrenamiento completo y de las señales observadas en TensorBoard/MLflow, evitando optimización ciega o cambios simultáneos no trazables.
