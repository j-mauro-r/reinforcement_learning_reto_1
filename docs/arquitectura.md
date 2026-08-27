# Arquitectura del proyecto — Assault con DDQN

## 1. Objetivo

Definir la arquitectura técnica y de trabajo que debe seguir el proyecto para desarrollar, entrenar, reanudar, evaluar y entregar un agente **Double Deep Q-Network (DDQN)** para `ALE/Assault-v5` de forma reproducible, simple y trazable.

La arquitectura aplica una filosofía **MLOps ligera**: se incorporan únicamente prácticas que aportan valor directo al reto académico y al entrenamiento en Google Colab, evitando infraestructura innecesaria.

Los principios rectores son:

- Google Colab como entorno principal de ejecución y entrenamiento con GPU.
- GitHub como fuente de verdad para código, notebooks, configuración y documentación.
- MLflow para registrar experimentos, hiperparámetros, métricas y artefactos relevantes.
- TensorBoard para observar la evolución del entrenamiento.
- Checkpoints para permitir continuidad entre sesiones de Colab.
- Código modular, simple, documentado y reutilizable.
- Principios SOLID y DRY aplicados sin sobreingeniería.
- Ejecuciones idempotentes: volver a ejecutar una etapa no debe destruir resultados válidos ni obligar a reiniciar el entrenamiento.

---

## 2. Restricciones de diseño

La arquitectura parte de las condiciones ya documentadas para Assault:

- entorno: `ALE/Assault-v5`;
- observaciones visuales RGB de alta dimensionalidad;
- espacio de acciones discreto de 7 acciones;
- dinámica temporal rápida;
- `frameskip` efectivo de 4;
- `repeat_action_probability=0.25`;
- entrenamiento costoso que requiere GPU;
- ejecución potencialmente distribuida en varias sesiones de Google Colab;
- evaluación final mediante recompensa promedio sobre al menos 10 episodios independientes;
- comparación contra el baseline aleatorio del proyecto;
- obligación de entregar notebook ejecutable, modelo entrenado, video y reporte técnico.

El algoritmo seleccionado para Assault es **DDQN**.

---

## 3. Arquitectura lógica

```text
Observación RGB
    │
    ▼
Preprocesamiento Atari
(grayscale + resize + frame stack)
    │
    ▼
Estado compacto
    │
    ▼
┌───────────────────────────────┐
│            DDQN               │
│                               │
│  Online Q-Network ─────────┐  │
│       │                     │  │
│       │ selecciona acción   │  │
│       ▼                     │  │
│   ε-greedy policy           │  │
│                             │  │
│  Target Q-Network ◄─────────┘  │
│       │ evalúa target          │
└───────┼───────────────────────┘
        │
        ↕
   Replay Buffer
        │
        ▼
   Optimización GPU
        │
        ├── TensorBoard
        ├── MLflow
        └── Checkpoints
```

DDQN debe mantener dos redes:

1. **Online Network**: selecciona acciones y se actualiza mediante gradiente.
2. **Target Network**: evalúa los targets y se sincroniza periódicamente desde la Online Network.

La separación entre selección y evaluación de la acción reduce la sobreestimación de valores Q característica de DQN.

---

## 4. Estructura del proyecto

```text
reinforcement_learning_reto_1/
│
├── docs/
│   └── arquitectura.md
│
├── enunciado_reto_1.txt
│
└── 2_Assault/
    │
    ├── ficha_tecnica.md
    ├── experimento_0_assault.ipynb
    ├── assault_ddqn.ipynb
    │
    ├── configs/
    │   └── ddqn_config.yaml
    │
    ├── src/
    │   ├── environment.py
    │   ├── network.py
    │   ├── replay_buffer.py
    │   ├── agent.py
    │   ├── trainer.py
    │   ├── evaluator.py
    │   ├── callbacks.py
    │   ├── tracking.py
    │   └── utils.py
    │
    ├── tests/
    │   └── test_smoke.py
    │
    ├── checkpoints/
    ├── models/
    ├── logs/
    ├── videos/
    │
    └── docs/
        └── hu001_experimento_0.md
```

Las carpetas `checkpoints/`, `logs/` y otros artefactos pesados o temporales no deben versionarse de forma rutinaria en GitHub. El modelo final sí deberá conservarse como artefacto de entrega; si su tamaño impide almacenarlo razonablemente en GitHub, deberá mantenerse en almacenamiento persistente y documentarse su ubicación.

---

## 5. Responsabilidad de cada componente

### 5.1 `assault_ddqn.ipynb`

Es el **entregable principal y orquestador del proyecto**.

Debe permitir ejecutar de principio a fin:

1. instalación de dependencias;
2. clonación o actualización del repositorio en Colab;
3. lectura de configuración;
4. identificación del hardware;
5. creación del entorno;
6. inicialización o restauración del agente;
7. entrenamiento;
8. visualización de TensorBoard;
9. evaluación final;
10. creación del video;
11. presentación del reporte técnico.

El notebook no debe contener implementaciones duplicadas del agente, Replay Buffer o red neuronal. Debe consumir los módulos de `src/`.

Esto mantiene el notebook legible y enfocado en explicar el experimento.

### 5.2 `configs/ddqn_config.yaml`

Será la **fuente única de configuración** para entrenamiento.

Debe contener como mínimo:

- environment ID;
- seed;
- dimensiones del preprocesamiento;
- número de frames apilados;
- tamaño del Replay Buffer;
- batch size;
- learning rate;
- gamma;
- epsilon inicial, final y estrategia de decay;
- frecuencia de aprendizaje;
- frecuencia de actualización de Target Network;
- learning starts;
- total timesteps objetivo;
- frecuencia de checkpoints;
- frecuencia de logging;
- número de episodios de evaluación.

Los hiperparámetros no deben quedar dispersos en diferentes archivos.

### 5.3 `src/environment.py`

Único responsable de crear y configurar Assault.

Debe:

- crear `ALE/Assault-v5`;
- aplicar el mismo pipeline de preprocesamiento de manera reproducible;
- controlar seeds;
- permitir distinguir entrenamiento de evaluación;
- evitar duplicar el `frameskip`.

El `frameskip` efectivo debe ser 4 **una sola vez**. Si el wrapper de preprocesamiento implementa el frame skipping, el entorno base debe configurarse de forma que no vuelva a aplicarlo.

Preprocesamiento recomendado:

```text
RGB 210×160×3
     ↓
grayscale
     ↓
resize 84×84
     ↓
normalización en la red / tensor
     ↓
stack de 4 frames
```

El frame stack permite inferir movimiento y dirección a partir de varias observaciones consecutivas.

### 5.4 `src/network.py`

Contendrá únicamente la arquitectura de la CNN utilizada como Q-Network.

Responsabilidades:

- transformar el estado visual apilado en características;
- generar un Q-value para cada una de las 7 acciones;
- mantener una arquitectura compatible con GPU;
- evitar lógica de entrenamiento dentro de la red.

### 5.5 `src/replay_buffer.py`

Responsable exclusivamente de almacenar y muestrear experiencias:

```text
(state, action, reward, next_state, done)
```

Para DDQN se utilizará **Experience Replay uniforme**. No se implementará Prioritized Experience Replay, porque constituiría un algoritmo diferente al DDQN seleccionado.

### 5.6 `src/agent.py`

Implementará la lógica propia del DDQN.

Debe contener:

- Online Network;
- Target Network;
- optimizer;
- selección de acciones ε-greedy;
- cálculo del target DDQN;
- actualización de la Online Network;
- sincronización de Target Network;
- save/load del estado del agente.

No debe administrar episodios completos ni visualizaciones.

### 5.7 `src/trainer.py`

Responsable del ciclo de entrenamiento.

Flujo:

```text
reset
  ↓
seleccionar acción
  ↓
env.step()
  ↓
guardar transición
  ↓
muestrear Replay Buffer
  ↓
actualizar DDQN
  ↓
registrar métricas
  ↓
checkpoint cuando corresponda
  ↓
continuar
```

Debe trabajar principalmente por **timesteps**, no depender de episodios para poder guardar progreso y reanudar de forma predecible.

### 5.8 `src/evaluator.py`

La evaluación debe estar completamente separada del entrenamiento.

Debe:

- cargar el modelo seleccionado;
- desactivar exploración ε-greedy o usar epsilon de evaluación definido explícitamente;
- ejecutar al menos 10 episodios independientes;
- usar recompensa real del entorno;
- registrar recompensa por episodio;
- calcular media, mediana, desviación estándar, mínimo y máximo;
- comparar el resultado con el baseline aleatorio.

La **recompensa promedio sobre ≥10 episodios** es la métrica principal del proyecto.

Si durante entrenamiento se usa clipping de rewards, la evaluación final debe conservar la recompensa real del entorno para que la métrica sea comparable con el baseline y con el criterio del reto.

### 5.9 `src/callbacks.py`

Centralizará observabilidad y persistencia periódica.

Debe implementar de manera simple:

- checkpoints;
- métricas para TensorBoard;
- medición del tiempo de entrenamiento;
- guardado del mejor modelo cuando exista evidencia suficiente para hacerlo.

No debe contener lógica DDQN.

### 5.10 `src/tracking.py`

Encapsulará el uso de MLflow.

Como mínimo cada experimento deberá registrar:

- nombre/id del experimento;
- algoritmo = DDQN;
- hiperparámetros;
- seed;
- versiones principales de librerías;
- características de GPU/CPU;
- commit de Git utilizado;
- timestep inicial y final;
- recompensa de evaluación;
- tiempo de entrenamiento;
- ruta o referencia del modelo/checkpoint.

MLflow no sustituye TensorBoard: tienen objetivos diferentes.

- **TensorBoard:** observación detallada de las curvas durante entrenamiento.
- **MLflow:** comparación y trazabilidad entre experimentos.

### 5.11 `src/utils.py`

Solo debe contener funciones transversales pequeñas, por ejemplo:

- configuración de seeds;
- identificación de hardware;
- creación segura de directorios;
- identificación del commit de Git;
- utilidades de tiempo.

No debe convertirse en un archivo genérico donde se acumule lógica de dominio.

---

## 6. Persistencia y checkpoints

Este punto es crítico debido a las limitaciones de sesión de Google Colab.

Cada checkpoint debe permitir **continuar realmente el entrenamiento**, no únicamente cargar una red para inferencia.

Debe guardar como mínimo:

- pesos de Online Network;
- pesos de Target Network;
- estado del optimizer;
- timestep global;
- epsilon o estado necesario para reconstruirlo;
- configuración del experimento;
- seed/configuración reproducible;
- métricas mínimas acumuladas necesarias para continuar el seguimiento.

El Replay Buffer puede convertirse en el elemento de mayor tamaño. Su persistencia deberá evaluarse frente al costo de almacenamiento. La arquitectura debe permitir dos modos documentados:

1. **Resume completo:** restaura agente, optimizer y Replay Buffer.
2. **Resume liviano:** restaura agente/optimizer y reconstruye gradualmente el Replay Buffer.

Para entrenamientos largos, el resume completo es preferible cuando el almacenamiento disponible lo permita.

Los checkpoints persistentes deben almacenarse fuera del almacenamiento efímero de `/content`, por ejemplo en Google Drive montado desde Colab.

---

## 7. Idempotencia

La ejecución del notebook debe ser segura al repetirse.

Reglas:

- crear directorios con `exist_ok=True`;
- no sobrescribir checkpoints válidos accidentalmente;
- detectar explícitamente si se desea iniciar un entrenamiento nuevo o reanudar uno existente;
- usar nombres de ejecución únicos;
- asociar artefactos a un `run_id`;
- registrar el timestep desde el cual se reanudó;
- evitar instalar o descargar recursos repetidamente cuando no sea necesario;
- mantener entrenamiento y evaluación como etapas independientes.

Nunca se debe reanudar automáticamente desde un checkpoint ambiguo. El notebook debe mostrar claramente cuál checkpoint será utilizado.

---

## 8. Gestión de experimentos

Cada entrenamiento debe identificarse mediante un nombre como:

```text
assault_ddqn_exp_001
assault_ddqn_exp_002
...
```

Un experimento representa una combinación concreta de:

- código Git;
- configuración;
- seed;
- hiperparámetros;
- entorno;
- preprocesamiento.

El principio fundamental es:

> Si una variable que puede modificar el aprendizaje cambia, debe quedar registrada.

Esto permitirá explicar por qué un modelo obtuvo un resultado diferente a otro.

---

## 9. TensorBoard

Durante entrenamiento deben registrarse como mínimo:

- recompensa por episodio;
- recompensa media móvil;
- longitud del episodio;
- loss;
- epsilon;
- Q-value medio o métrica equivalente útil;
- timestep global;
- learning rate si cambia durante entrenamiento.

TensorBoard debe permitir detectar:

- ausencia de aprendizaje;
- inestabilidad;
- divergencia de loss;
- colapso de exploración;
- mejoras reales de recompensa.

No se deben crear métricas únicamente por disponibilidad; cada métrica debe ayudar a entender el entrenamiento.

---

## 10. MLflow

MLflow deberá utilizarse como registro de experimentos, no como infraestructura compleja de despliegue.

Cada ejecución deberá guardar:

### Parámetros

- algoritmo;
- hiperparámetros;
- configuración del entorno;
- preprocesamiento;
- seed;
- hardware;
- versiones.

### Métricas

- recompensa media de evaluación;
- recompensa mínima y máxima;
- desviación estándar;
- tiempo de entrenamiento;
- timestep final;
- mejor recompensa observada durante entrenamiento.

### Artefactos relevantes

- configuración utilizada;
- modelo final o referencia al mismo;
- resumen de evaluación;
- gráficas o resultados finales cuando aporten valor.

---

## 11. Estrategia GitHub

GitHub será la fuente de verdad del proyecto.

Flujo recomendado:

```text
main
  ↓
feature/HU-xxx
  ↓
implementación
  ↓
validación local/Colab
  ↓
Pull Request
  ↓
revisión
  ↓
merge a main
```

No deben desarrollarse cambios relevantes directamente sobre `main` cuando se trate de historias de implementación.

Los notebooks con resultados finales deben conservar sus outputs cuando estos formen parte de la evidencia académica, pero código, configuración y resultados deben mantenerse diferenciados.

---

## 12. Pruebas mínimas

No se busca construir una suite empresarial de pruebas.

`tests/test_smoke.py` debe validar al menos:

1. el entorno puede crearse;
2. el preprocesamiento produce la forma esperada;
3. la red acepta un batch de estados;
4. la red retorna exactamente 7 Q-values por estado;
5. el Replay Buffer puede almacenar y muestrear transiciones;
6. el agente puede realizar un paso de aprendizaje sin error;
7. un checkpoint puede guardarse y cargarse.

Estas pruebas buscan detectar errores de integración antes de consumir horas de GPU.

---

## 13. Optimización del uso de Google Colab

La arquitectura debe priorizar eficiencia práctica:

- utilizar GPU para forward/backpropagation de la CNN;
- evitar render durante entrenamiento normal;
- renderizar únicamente evaluación/video;
- usar observaciones reducidas `84×84` en escala de grises;
- usar frame stack para conservar información temporal;
- mantener Replay Buffer en CPU RAM y transferir únicamente batches a GPU;
- evitar copias innecesarias de tensores;
- guardar checkpoints periódicamente;
- registrar hardware disponible al inicio de cada sesión;
- permitir reanudar entrenamiento sin repetir timesteps ya completados.

Antes de un entrenamiento largo debe ejecutarse un **smoke training** corto que valide todo el pipeline.

---

## 14. Flujo MLOps ligero del proyecto

```text
1. Entender entorno
        ↓
2. Baseline + métricas
        ↓
3. Seleccionar algoritmo: DDQN
        ↓
4. Versionar configuración
        ↓
5. Smoke test
        ↓
6. Entrenar en Colab GPU
        ↓
7. TensorBoard + MLflow
        ↓
8. Checkpoint
        ↓
   ¿sesión termina?
      ↙        ↘
    Sí          No
     ↓           ↓
  reanudar   continuar
      ↘        ↙
        ↓
9. Evaluar ≥10 episodios
        ↓
10. Comparar vs baseline
        ↓
11. Seleccionar modelo final
        ↓
12. Generar video + reporte
```

---

## 15. Reglas arquitectónicas obligatorias

Para mantener el proyecto consistente:

1. El notebook **orquesta**, no duplica lógica de `src/`.
2. `environment.py` es la única fuente de creación/configuración del entorno.
3. Los hiperparámetros viven en un único archivo de configuración.
4. Entrenamiento y evaluación son procesos separados.
5. La recompensa oficial se mide sin exploración deliberada y sobre ≥10 episodios.
6. El baseline y el agente deben evaluarse bajo condiciones equivalentes.
7. El `frameskip` efectivo debe aplicarse una sola vez.
8. DDQN usa Replay Buffer uniforme; PER queda fuera de esta implementación.
9. Cada experimento debe poder relacionarse con una configuración y un commit de Git.
10. Todo entrenamiento largo debe tener checkpoints.
11. Los checkpoints deben permitir reanudar entrenamiento, no solo inferencia.
12. TensorBoard se utiliza para diagnóstico temporal del entrenamiento.
13. MLflow se utiliza para trazabilidad y comparación de experimentos.
14. El código debe usar docstrings estilo Google en funciones y clases públicas/reutilizables.
15. Se aplicarán SOLID y DRY cuando reduzcan acoplamiento o duplicación; no se crearán abstracciones sin una necesidad concreta.
16. Antes de consumir una sesión larga de GPU se debe superar un smoke test del pipeline.
17. Artefactos temporales y pesados no deben contaminar el repositorio.

---

## 16. Criterio de éxito de la arquitectura

La arquitectura se considera exitosa si permite que cualquier integrante del equipo pueda, partiendo de `main`:

1. abrir `assault_ddqn.ipynb` en Google Colab;
2. instalar las dependencias necesarias;
3. reproducir la configuración del entorno;
4. iniciar o reanudar un entrenamiento DDQN;
5. observar su evolución en TensorBoard;
6. identificar el experimento en MLflow;
7. recuperar un checkpoint después de una interrupción;
8. evaluar el modelo en al menos 10 episodios;
9. comparar objetivamente el resultado con el baseline aleatorio;
10. generar el modelo, las métricas, el video y la evidencia requeridos para la entrega.

La prioridad no es construir una plataforma MLOps completa. La prioridad es lograr un **agente reproducible, medible, recuperable y defendible técnicamente**, utilizando la mínima arquitectura necesaria para cumplir correctamente el reto.
