# Arquitectura del proyecto — BattleZone

## 1. Objetivo

Definir la arquitectura técnica y de trabajo para desarrollar, entrenar, reanudar, evaluar y entregar un agente de Reinforcement Learning para `ALE/BattleZone-v5` de forma reproducible, observable, recuperable y coherente con:

- `3_BattleZone/docs/ficha_tecnica.md`;
- `3_BattleZone/docs/implementacion.md`;
- `3_BattleZone/docs/lineamientos.md`;
- `enunciado_reto_1.txt`.

BattleZone se implementará como solución independiente. Assault se utiliza únicamente como base de conocimiento y experiencia metodológica. **No se copiará, importará ni reutilizará código desde `2_Assault/`.**

BattleZone no utilizará MLflow.

---

## 2. Principios arquitectónicos

1. **Independencia total de Assault.** Ningún módulo de BattleZone dependerá de `2_Assault/`.
2. **Separación de responsabilidades.** Entorno, preprocessing, agente, entrenamiento, evaluación, persistencia y observabilidad deben estar desacoplados.
3. **Notebook como orquestador y reporte.** La lógica reusable reside en `src/`.
4. **Configuración centralizada.** Los parámetros que afectan resultados deben vivir en archivos versionados.
5. **Trazabilidad ligera.** Git/GitHub + configuración + `run_id` + `run_manifest.json` + TensorBoard + checkpoints + resultados estructurados.
6. **Idempotencia.** Reejecutar una etapa no debe destruir evidencia válida ni reiniciar silenciosamente un entrenamiento.
7. **Recuperabilidad.** Todo entrenamiento largo debe soportar checkpoints y resume.
8. **Validar barato antes de entrenar caro.** Smoke tests obligatorios antes de sesiones largas de GPU.
9. **Arquitectura adaptable al algoritmo.** HU004 seleccionará entre DQN, DQN+PER, DDQN o REINFORCE.
10. **No fijar preprocessing por analogía.** BattleZone requiere validar específicamente la conservación del radar y señales pequeñas.

---

## 3. Restricciones del entorno que condicionan la arquitectura

La documentación oficial de ALE establece para `ALE/BattleZone-v5`:

- observación RGB `210×160×3`, `uint8`;
- action space `Discrete(18)`;
- `frameskip=4`;
- `repeat_action_probability=0.25`;
- 5 vidas iniciales;
- radar integrado en la observación;
- perspectiva en primera persona;
- múltiples tipos de enemigos y obstáculos.

Estas propiedades obligan a diseñar una solución capaz de:

- procesar imágenes;
- conservar información espacial y temporal;
- producir decisiones entre 18 acciones;
- soportar entrenamiento costoso en GPU;
- reanudar sesiones interrumpidas;
- diferenciar entrenamiento de evaluación;
- conservar evidencia suficiente para el reporte final.

---

## 4. Arquitectura lógica general

La arquitectura debe soportar cualquiera de los algoritmos permitidos sin imponer componentes innecesarios antes de HU004.

```text
ALE/BattleZone-v5
        │
        ▼
Environment Factory
        │
        ▼
Preprocessing Pipeline
        │
        ▼
Estado visual/temporal
        │
        ▼
┌─────────────────────────────┐
│          Agent              │
│                             │
│  Network / Policy           │
│  Exploration strategy       │
│  Learning logic             │
│  Replay Buffer (si aplica)  │
│  Target Network (si aplica) │
└──────────────┬──────────────┘
               │
               ▼
            Trainer
               │
       ┌───────┼────────┐
       ▼       ▼        ▼
 Checkpoints TensorBoard Results
       │                │
       └───────┬────────┘
               ▼
        run_manifest.json
               │
               ▼
            Evaluator
               │
               ▼
      Reporte + Video + Modelo
```

---

## 5. Estructura esperada del proyecto

```text
3_BattleZone/
├── battlezone_agent.ipynb
│
├── configs/
│   └── battlezone_config.yaml
│
├── src/
│   ├── environment.py
│   ├── preprocessing.py          # solo si aporta claridad frente a environment.py
│   ├── network.py
│   ├── replay_buffer.py          # si aplica al algoritmo
│   ├── agent.py
│   ├── trainer.py
│   ├── evaluator.py
│   ├── callbacks.py
│   ├── persistence.py            # si checkpointing requiere responsabilidad propia
│   └── utils.py
│
├── tests/
│   ├── test_environment.py
│   ├── test_agent.py
│   └── test_smoke.py
│
├── checkpoints/
├── models/
├── logs/
├── results/
│   └── <run_id>/
│       ├── run_manifest.json
│       ├── evaluation.json
│       └── summaries/
├── videos/
│
└── docs/
    ├── ficha_tecnica.md
    ├── arquitectura.md
    ├── implementacion.md
    └── lineamientos.md
```

La estructura debe mantenerse mínima. No se crearán módulos vacíos ni capas sin responsabilidad clara.

---

## 6. Responsabilidad de cada componente

### 6.1 `battlezone_agent.ipynb`

Será el entregable principal y reporte académico.

Debe orquestar:

1. instalación de dependencias;
2. carga/actualización del repositorio;
3. carga de configuración;
4. identificación de hardware;
5. creación del entorno;
6. inicialización o restauración del agente;
7. entrenamiento;
8. visualización de TensorBoard;
9. evaluación formal;
10. generación de video;
11. presentación del reporte técnico.

No debe duplicar lógica de `src/`.

---

### 6.2 `configs/battlezone_config.yaml`

Será la fuente única de configuración experimental.

Debe incluir según el algoritmo:

#### Entorno

- environment ID;
- mode;
- difficulty;
- `obs_type`;
- `frameskip`;
- `repeat_action_probability`;
- seed;
- action space.

#### Preprocessing

- grayscale/RGB;
- resize;
- frame stack;
- normalización;
- clipping de reward si aplica;
- cualquier cropping explícitamente validado.

#### Entrenamiento

- algoritmo;
- total timesteps/episodes;
- learning rate;
- gamma;
- batch size;
- Replay Buffer;
- parámetros PER cuando aplique;
- epsilon/exploración;
- learning starts;
- target sync cuando aplique;
- checkpoint interval;
- logging interval.

#### Evaluación

- episodios;
- seed strategy;
- exploración de evaluación;
- video settings.

No deben existir constantes mágicas dispersas.

---

### 6.3 `src/environment.py`

Será la única fuente para crear `ALE/BattleZone-v5`.

Responsabilidades:

- registrar ALE cuando corresponda;
- construir entorno base;
- aplicar mode/difficulty;
- controlar seeds;
- aplicar wrappers;
- garantizar `frameskip` efectivo una sola vez;
- exponer train/eval mediante una fábrica común;
- validar observation/action spaces;
- permitir render solo cuando corresponda.

Ningún otro módulo debe llamar directamente a `gymnasium.make("ALE/BattleZone-v5")` salvo tests muy controlados de contrato.

---

### 6.4 `src/preprocessing.py`

Solo se creará si HU003 demuestra que separar preprocessing reduce acoplamiento.

Posibles responsabilidades:

- grayscale/RGB;
- resize;
- frame stack;
- validación de shape/dtype;
- transformación a tensor cuando corresponda.

No debe contener lógica del agente.

---

### 6.5 `src/network.py`

Contendrá únicamente la arquitectura neuronal seleccionada.

Para value-based:

- CNN/Q-Network;
- salida de 18 Q-values.

Para REINFORCE:

- policy network;
- salida de distribución de probabilidad sobre 18 acciones.

No debe incluir ciclo de entrenamiento ni interacción con el entorno.

---

### 6.6 `src/replay_buffer.py`

Solo existirá si el algoritmo lo requiere.

Para DQN/DDQN:

- Experience Replay uniforme.

Para DQN+PER:

- prioridades;
- muestreo ponderado;
- importance-sampling weights;
- actualización de prioridades.

Para REINFORCE no debe existir Replay Buffer persistente de transiciones si no aporta valor al algoritmo.

---

### 6.7 `src/agent.py`

Encapsulará la lógica del algoritmo seleccionado.

Interfaz mínima esperada:

- `select_action(...)`;
- `learn(...)` o equivalente;
- `save_state(...)`;
- `load_state(...)`;
- acceso controlado a métricas necesarias.

El trainer no debe depender de detalles internos de la red.

---

### 6.8 `src/trainer.py`

Responsable del ciclo de aprendizaje.

Debe gestionar:

- reset;
- selección de acción;
- `env.step()`;
- almacenamiento de experiencia/trayectoria;
- disparo de actualizaciones del agente;
- control global de timestep/episodio;
- manejo explícito de `terminated` y `truncated`;
- callbacks de checkpoint/logging;
- continuidad al reanudar.

No debe generar reportes ni gráficos directamente.

---

### 6.9 `src/evaluator.py`

Debe estar totalmente desacoplado del entrenamiento.

Responsabilidades:

- cargar modelo explícito;
- crear entorno mediante la misma fábrica;
- ejecutar ≥10 episodios;
- usar reward real del entorno;
- calcular media, mediana, desviación estándar, mínimo y máximo;
- registrar duración, vidas y métricas secundarias útiles;
- producir resultados estructurados;
- soportar render/video.

La evaluación no debe modificar pesos ni Replay Buffer.

---

### 6.10 `src/callbacks.py`

Centralizará tareas periódicas sin contaminar `trainer.py`:

- checkpoints;
- TensorBoard;
- medición de tiempo;
- persistencia de métricas;
- guardado del mejor modelo si la política de selección lo justifica.

No debe contener lógica del algoritmo.

---

### 6.11 `src/persistence.py`

Se creará únicamente si checkpointing y manifests justifican una responsabilidad separada.

Posibles funciones:

- guardar/restaurar checkpoint;
- validar compatibilidad de configuración;
- escribir `run_manifest.json`;
- resolver rutas persistentes;
- evitar sobrescrituras ambiguas.

---

### 6.12 `src/utils.py`

Solo utilidades transversales pequeñas:

- seeds;
- hardware;
- versiones;
- commit Git;
- `run_id`;
- creación segura de directorios;
- utilidades de tiempo.

No debe convertirse en contenedor genérico de lógica de dominio.

---

## 7. Arquitectura del estado visual

BattleZone requiere especial cuidado porque el radar forma parte de la imagen.

Pipeline conceptual inicial:

```text
RGB 210×160×3
        │
        ▼
Validación de información visual
        │
        ├── RGB o grayscale
        ├── resolución objetivo
        └── conservación del radar
        │
        ▼
Frame stack / contexto temporal
        │
        ▼
Tensor de entrada
        │
        ▼
CNN / Policy Network
```

### Reglas

- No recortar la zona del radar sin evidencia.
- No fijar 84×84 automáticamente por haber funcionado en otro entorno.
- Mantener `uint8` mientras sea viable para ahorrar RAM.
- Normalizar al convertir a tensor si esa decisión reduce memoria.
- Train y eval deben usar exactamente el mismo preprocessing.

---

## 8. Arquitectura adaptable al algoritmo

HU004 definirá el algoritmo. La arquitectura debe evitar asumir DDQN por defecto.

### 8.1 Si se selecciona DQN

```text
State → Online Q-Network → 18 Q-values
                  ▲
                  │
            Replay Buffer
```

Requiere:

- Q-network;
- target network;
- replay uniforme;
- epsilon-greedy.

### 8.2 Si se selecciona DQN + PER

Añade:

- prioridades por transición;
- sampling no uniforme;
- importance sampling;
- actualización de prioridades.

### 8.3 Si se selecciona DDQN

```text
Online Network → selecciona argmax
Target Network → evalúa esa acción
```

Requiere Replay uniforme salvo decisión distinta prohibida por alcance.

### 8.4 Si se selecciona REINFORCE

```text
State → Policy Network → π(a|s) sobre 18 acciones
                         │
                         ▼
                      Sample
```

Requiere trayectorias/retornos y no necesita Target Network ni Replay Buffer clásico.

---

## 9. Gestión de checkpoints

Los checkpoints son obligatorios para entrenamientos largos.

Cada checkpoint debe guardar el estado mínimo suficiente según algoritmo.

### Value-based

- Online Network;
- Target Network;
- optimizer;
- global timestep;
- epsilon/estado de exploración;
- configuración;
- métricas de continuidad;
- Replay Buffer cuando se utilice resume completo.

### REINFORCE

- policy network;
- optimizer;
- episodio/timestep global;
- configuración;
- estado necesario para continuidad;
- métricas acumuladas.

### Modos soportados

1. **New run**.
2. **Resume completo**.
3. **Resume liviano**.

No se debe seleccionar automáticamente un checkpoint ambiguo.

---

## 10. Persistencia y almacenamiento

### GitHub

Versionar:

- código;
- configuración;
- documentación;
- HUs;
- notebooks;
- resultados finales pequeños.

No versionar rutinariamente:

- Replay Buffers;
- checkpoints intermedios grandes;
- logs temporales;
- caches;
- videos pesados salvo necesidad académica explícita.

### Google Colab / almacenamiento persistente

Los entrenamientos largos no deben depender exclusivamente de `/content`.

Persistir externamente:

- checkpoints relevantes;
- modelo final;
- TensorBoard logs necesarios;
- manifests;
- evaluación final;
- video final.

---

## 11. TensorBoard

TensorBoard será la herramienta principal de observabilidad del entrenamiento.

Registrar, cuando aplique:

- reward por episodio;
- reward media móvil;
- episode length;
- loss;
- epsilon/exploración;
- Q-value medio o entropía/policy loss según algoritmo;
- timestep global;
- learning rate;
- throughput si aporta valor;
- métricas de memoria solo cuando ayuden a diagnóstico.

TensorBoard no reemplaza la evaluación formal.

---

## 12. Trazabilidad sin MLflow

Cada corrida relevante deberá crear:

```text
results/<run_id>/run_manifest.json
```

El manifest contendrá:

- `run_id`;
- algoritmo;
- commit Git;
- seed;
- archivo de configuración;
- parámetros efectivos;
- environment ID;
- mode/difficulty;
- preprocessing;
- versiones;
- hardware;
- timestamp;
- timestep/episodio inicial y final;
- tiempo acumulado;
- checkpoint/modelo;
- TensorBoard log path;
- resultado de evaluación cuando exista.

### Comparación de experimentos

Se realizará mediante:

- manifests;
- archivos JSON/CSV estructurados;
- tablas generadas en notebook;
- curvas de TensorBoard;
- commit/configuración asociados.

---

## 13. Idempotencia

Toda etapa debe ser segura al repetirse.

Reglas:

- `mkdir(..., exist_ok=True)`;
- no sobrescribir checkpoints válidos accidentalmente;
- no reiniciar `global_step` durante resume;
- distinguir explícitamente new/resume;
- usar `run_id` único;
- asociar artefactos al run;
- no eliminar resultados anteriores por reejecutar una celda;
- validar compatibilidad entre checkpoint y configuración.

---

## 14. Entrenamiento local vs Colab

### Local

Usar para:

- imports;
- pruebas unitarias;
- validación de configuración;
- entorno;
- preprocessing;
- forward pass;
- Replay Buffer;
- una actualización de pesos;
- checkpoint/save/load básico.

### Google Colab GPU

Usar para:

- smoke E2E GPU;
- profiling de VRAM;
- entrenamientos largos;
- optimización;
- evaluación final cuando render/video lo requiera.

La GPU acelera la red neuronal, pero no elimina los costos CPU de emulación ALE y preprocessing.

---

## 15. Estrategia de pruebas

No se busca una suite empresarial, sino detectar errores antes de consumir GPU.

### Environment tests

- creación;
- action space `Discrete(18)`;
- reset/step;
- shape/dtype;
- frameskip efectivo;
- train/eval equivalentes;
- seed/configuración.

### Network/agent tests

- forward pass;
- 18 outputs;
- batch shapes;
- actualización de pesos;
- device correcto.

### Replay tests

Cuando aplique:

- add;
- sample;
- shapes;
- prioridades y weights para PER.

### Persistence tests

- save;
- load;
- checkpoint;
- global step;
- config compatibility;
- resume.

### Smoke E2E

- interacción;
- entrenamiento corto;
- TensorBoard;
- checkpoint;
- restore;
- evaluación corta.

---

## 16. Flujo E2E

```text
HU001  Ficha técnica
   ↓
HU002  Baseline aleatorio
   ↓
HU003  Pipeline reproducible
   ↓
HU004  Selección algoritmo
   ↓
HU005  Núcleo agente
   ↓
HU006  Training loop
   ↓
HU007  Checkpoints / resume
   ↓
HU008  TensorBoard
   ↓
HU009  Smoke E2E
   ↓
HU010  Trazabilidad ligera
   ↓
HU011  Entrenamiento largo
   ↓
HU012  Optimización controlada
   ↓
HU013  Evaluación ≥10 episodios
   ↓
HU014  Reporte + video + modelo
```

---

## 17. Arquitectura de evaluación

La evaluación debe ser un flujo independiente:

```text
Modelo explícito
      │
      ▼
Environment Factory (eval)
      │
      ▼
Mismo preprocessing
      │
      ▼
Policy sin exploración deliberada
      │
      ▼
≥10 episodios
      │
      ▼
Resultados estructurados
      │
      ├── reward mean/median/std
      ├── min/max
      ├── episode length
      ├── lives
      └── evidencia conductual
      │
      ▼
Comparación vs baseline HU002
```

La recompensa de entrenamiento no sustituye la evaluación formal.

---

## 18. Arquitectura del reporte final

El notebook final deberá integrar evidencia generada por la arquitectura, no recrearla manualmente.

Debe contener:

1. problema;
2. caracterización del entorno;
3. baseline;
4. selección algorítmica;
5. arquitectura;
6. preprocessing;
7. hiperparámetros;
8. hardware/versiones;
9. estrategia de entrenamiento;
10. checkpoints/resume;
11. TensorBoard;
12. experimentos y optimización;
13. evaluación ≥10 episodios;
14. comparación con baseline;
15. análisis de comportamiento;
16. limitaciones;
17. conclusiones;
18. video y modelo final.

---

## 19. Riesgos arquitectónicos

| Riesgo | Mitigación |
|---|---|
| Radar degradado por preprocessing | validar visualmente y por experimento antes de fijar resolución |
| 18 acciones ralentizan exploración | baseline + análisis de acción/reward + HU004 |
| Replay Buffer excesivo | `uint8`, capacidad configurable, profiling |
| Doble frameskip | fábrica única + test explícito |
| Checkpoint incompleto | contrato por algoritmo + test resume |
| Divergencia train/eval | misma fábrica + config compartida |
| Pérdida de sesión Colab | persistencia externa + checkpoints |
| Resultados no trazables | run manifest + Git + config + TensorBoard |
| Arquitectura demasiado rígida antes de HU004 | módulos condicionales según algoritmo |
| Sobreingeniería | SOLID/DRY pragmáticos y DWP |

---

## 20. Reglas arquitectónicas obligatorias

1. `environment.py` será la única fábrica del entorno.
2. El notebook no duplicará lógica reusable.
3. Configuración centralizada y versionada.
4. Train y eval separados.
5. `Discrete(18)` debe respetarse salvo decisión posterior explícita y justificada.
6. Preprocessing no debe destruir el radar sin evidencia.
7. `frameskip` efectivo debe ser 4 una sola vez.
8. Checkpoints obligatorios antes de entrenamientos largos.
9. Todo resume debe conservar progreso global.
10. TensorBoard obligatorio en corridas relevantes.
11. No se utilizará MLflow.
12. Cada run relevante debe tener `run_manifest.json`.
13. El algoritmo debe pertenecer a la lista permitida por el reto.
14. No se reutilizará código de Assault.
15. Smoke test E2E aprobado antes de entrenamiento largo.
16. Evaluación formal sobre ≥10 episodios.
17. Baseline y agente deben compararse bajo condiciones equivalentes.
18. Toda modificación de hiperparámetros debe responder a una hipótesis.
19. Las funciones públicas deben usar documentación estilo Google.
20. Artefactos pesados no deben contaminar GitHub.

---

## 21. Criterio de éxito de la arquitectura

La arquitectura será adecuada si permite que un integrante del equipo pueda, desde una rama limpia:

1. instalar dependencias;
2. cargar configuración;
3. crear `ALE/BattleZone-v5` de forma reproducible;
4. ejecutar tests básicos;
5. inicializar el algoritmo seleccionado;
6. entrenar localmente en modo smoke;
7. continuar en Colab GPU;
8. guardar y restaurar checkpoints;
9. observar TensorBoard;
10. asociar cada corrida a commit/configuración/run manifest;
11. evaluar ≥10 episodios;
12. comparar contra baseline;
13. generar modelo, video y reporte final;
14. completar todo el flujo sin depender de código de Assault.

La prioridad es construir un agente **independiente, reproducible, observable, recuperable y defendible técnicamente**, usando únicamente la complejidad necesaria para cumplir correctamente el reto.