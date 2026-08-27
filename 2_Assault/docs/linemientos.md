# Lineamientos técnicos de implementación — Assault con DDQN

## 1. Objetivo

Definir las políticas, estándares técnicos y prácticas obligatorias que deben regir la implementación del agente **Double Deep Q-Network (DDQN)** para `ALE/Assault-v5`.

Estos lineamientos complementan:

- `docs/arquitectura.md`;
- `docs/implementacion.md`;
- `2_Assault/ficha_tecnica.md`;
- las HUs del proyecto;
- el enunciado del Reto 1.

El objetivo es asegurar una implementación reproducible, mantenible, observable y adecuada para un proyecto académico de Reinforcement Learning ejecutado principalmente en Google Colab.

La filosofía general será **MLOps ligera**: aplicar únicamente prácticas que agreguen trazabilidad, reproducibilidad, mantenibilidad y capacidad de reanudar entrenamientos, evitando sobreingeniería.

---

## 2. Principios rectores

Toda implementación deberá cumplir los siguientes principios:

1. **Simplicidad primero.** La solución debe ser tan simple como sea posible sin sacrificar reproducibilidad o calidad técnica.
2. **Reproducibilidad.** Una corrida debe poder reconstruirse a partir del código, configuración, seed y versiones utilizadas.
3. **Trazabilidad.** Todo experimento relevante debe dejar evidencia de qué configuración y qué código produjo sus resultados.
4. **Separación de responsabilidades.** Entorno, agente, entrenamiento, evaluación, logging y tracking no deben mezclarse innecesariamente.
5. **Idempotencia.** Repetir una etapa no debe destruir resultados válidos ni reiniciar silenciosamente un entrenamiento existente.
6. **Observabilidad.** El entrenamiento debe producir métricas suficientes para detectar aprendizaje, estancamiento o divergencia.
7. **Uso eficiente de recursos.** Antes de ejecutar entrenamientos largos, se deben realizar validaciones pequeñas y smoke tests.
8. **Evidencia antes de optimización.** Los cambios de hiperparámetros deben responder a una hipótesis verificable.
9. **No duplicación.** Una misma lógica no debe mantenerse en notebook y módulos Python simultáneamente.
10. **Compatibilidad con Google Colab.** Todo el flujo principal debe poder ejecutarse en Colab desde un entorno limpio.

---

## 3. Metodología MLOps del proyecto

El proyecto seguirá un ciclo MLOps adaptado:

```text
Definir problema
   ↓
Caracterizar entorno
   ↓
Definir baseline y métricas
   ↓
Diseñar arquitectura
   ↓
Implementar incrementalmente
   ↓
Validar con smoke test
   ↓
Registrar experimento
   ↓
Entrenar
   ↓
Evaluar
   ↓
Comparar
   ↓
Documentar y entregar
```

Herramientas principales:

- **GitHub:** fuente de verdad del código, documentación, configuración y notebooks.
- **Google Colab:** entorno principal de entrenamiento con GPU.
- **TensorBoard:** observabilidad detallada del entrenamiento.
- **MLflow:** trazabilidad y comparación entre experimentos.
- **Checkpoints:** continuidad entre sesiones de Colab.

---

## 4. Estándares de arquitectura de código

La implementación deberá respetar la arquitectura definida en `docs/arquitectura.md`.

Estructura esperada:

```text
2_Assault/
├── assault_ddqn.ipynb
├── ficha_tecnica.md
├── configs/
│   └── ddqn_config.yaml
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
├── tests/
├── checkpoints/
├── models/
├── logs/
├── videos/
└── docs/
```

Reglas:

- `assault_ddqn.ipynb` será **orquestador y reporte**, no repositorio de lógica duplicada.
- La lógica reutilizable deberá estar en `src/`.
- Los hiperparámetros deberán centralizarse en `configs/ddqn_config.yaml`.
- No se deben crear módulos nuevos sin una responsabilidad clara.

---

## 5. Principios SOLID

SOLID debe aplicarse de forma pragmática.

### 5.1 Single Responsibility Principle

Cada módulo debe tener una responsabilidad principal:

- `environment.py`: creación y preprocessing del entorno;
- `network.py`: CNN/Q-Network;
- `replay_buffer.py`: almacenamiento y muestreo de experiencias;
- `agent.py`: lógica DDQN;
- `trainer.py`: ciclo de entrenamiento;
- `evaluator.py`: evaluación independiente;
- `callbacks.py`: checkpoints y observabilidad periódica;
- `tracking.py`: MLflow;
- `utils.py`: utilidades transversales pequeñas.

### 5.2 Open/Closed Principle

Las configuraciones deben modificarse preferiblemente mediante parámetros o YAML antes que editando lógica de negocio.

### 5.3 Liskov Substitution Principle

Cuando existan abstracciones compartidas, sus implementaciones deberán mantener contratos compatibles. No se crearán jerarquías de clases innecesarias.

### 5.4 Interface Segregation Principle

No se deben crear interfaces grandes. Cada componente deberá exponer únicamente operaciones necesarias.

### 5.5 Dependency Inversion Principle

El trainer no debe depender de detalles internos de la red. Debe interactuar con el agente mediante funciones o métodos claramente definidos.

---

## 6. Principio DRY

Se debe evitar duplicar:

- creación del entorno;
- preprocessing;
- carga de configuración;
- selección de dispositivo;
- lógica de evaluación;
- cálculo de métricas;
- rutas de artefactos;
- save/load de checkpoints.

Si una misma lógica aparece en dos lugares, deberá evaluarse si corresponde extraerla a una función o módulo reutilizable.

No se debe aplicar DRY creando abstracciones complejas para código trivial o de una sola ocurrencia.

---

## 7. Documentación de código estilo Google

Las funciones y clases reutilizables deberán usar docstrings estilo Google.

Ejemplo:

```python
def create_environment(config: dict, seed: int):
    """Creates a configured Assault environment.

    Args:
        config: Environment and preprocessing configuration.
        seed: Seed used to initialize the environment.

    Returns:
        Configured Gymnasium environment.

    Raises:
        ValueError: If the environment configuration is invalid.
    """
```

Reglas:

- documentar propósito, argumentos, retornos y errores relevantes;
- evitar comentarios que solo repitan el código;
- explicar decisiones no obvias;
- nombres de variables y funciones deben ser descriptivos;
- código y nombres técnicos preferiblemente en inglés;
- documentación académica y Markdown pueden mantenerse en español.

---

## 8. Configuración centralizada

`configs/ddqn_config.yaml` será la fuente única para parámetros de entrenamiento y entorno.

Debe incluir, según corresponda:

- environment ID;
- seed;
- preprocessing;
- tamaño de imagen;
- frame stack;
- Replay Buffer;
- batch size;
- learning rate;
- gamma;
- epsilon inicial y final;
- estrategia de epsilon decay;
- learning starts;
- frecuencia de aprendizaje;
- frecuencia de sincronización de Target Network;
- total timesteps;
- checkpoint interval;
- logging interval;
- número de episodios de evaluación.

Los valores no deben quedar dispersos como constantes mágicas en múltiples archivos.

---

## 9. Reproducibilidad

Cada experimento relevante debe registrar como mínimo:

- seed;
- commit Git;
- configuración YAML;
- versiones de Python, Gymnasium, ALE-Py, PyTorch y librerías principales;
- hardware utilizado;
- identificador del experimento;
- timestamp o run ID.

Cuando una operación no sea completamente determinista por naturaleza de GPU/ALE, debe documentarse la limitación.

---

## 10. Idempotencia

El código debe soportar ejecuciones repetidas de forma segura.

Reglas obligatorias:

- crear directorios con `exist_ok=True`;
- no sobrescribir checkpoints válidos accidentalmente;
- diferenciar claramente `new run` y `resume`;
- no seleccionar automáticamente un checkpoint ambiguo;
- usar `run_id` o nombres únicos para experimentos;
- mantener entrenamiento y evaluación desacoplados;
- no reiniciar `global_step` al reanudar;
- conservar la configuración asociada a cada checkpoint.

Una celda de Colab ejecutada nuevamente no debe eliminar evidencia previa salvo instrucción explícita.

---

## 11. Checkpoints

El proyecto debe soportar entrenamientos distribuidos en varias sesiones de Colab.

Cada checkpoint deberá guardar como mínimo:

- Online Network;
- Target Network;
- optimizer;
- `global_step`;
- epsilon o estado necesario para reconstruirlo;
- configuración;
- métricas necesarias para continuar trazabilidad.

Cuando sea viable, también:

- Replay Buffer;
- estado adicional de entrenamiento necesario para resume completo.

Se deben soportar dos estrategias:

### Resume completo

Restaura agente, optimizer, progreso y Replay Buffer.

### Resume liviano

Restaura agente y optimizer pero reconstruye el Replay Buffer mediante nuevas interacciones.

Los checkpoints importantes deben persistirse fuera de `/content`, preferiblemente en Google Drive.

---

## 12. TensorBoard

TensorBoard será obligatorio durante los entrenamientos relevantes.

Métricas mínimas:

- recompensa por episodio;
- recompensa media móvil;
- longitud del episodio;
- loss;
- epsilon;
- Q-value medio o métrica equivalente;
- global timestep;
- learning rate si cambia.

Objetivos de TensorBoard:

- detectar si existe aprendizaje;
- identificar inestabilidad;
- observar comportamiento de epsilon;
- detectar divergencia de loss;
- comparar fases del entrenamiento;
- producir evidencia gráfica para el reporte académico.

No se deben registrar métricas sin utilidad analítica clara.

---

## 13. MLflow

MLflow será utilizado para trazabilidad de experimentos, no como plataforma de despliegue.

Cada run relevante debe registrar:

### Parámetros

- algoritmo = DDQN;
- configuración del entorno;
- preprocessing;
- hiperparámetros;
- seed;
- versiones;
- hardware;
- commit Git.

### Métricas

- timestep inicial y final;
- tiempo de entrenamiento;
- recompensa de evaluación;
- desviación estándar;
- mínimo y máximo;
- mejor recompensa relevante observada.

### Artefactos

- configuración;
- modelo/checkpoint o referencia;
- resumen de evaluación;
- gráficas finales relevantes.

TensorBoard y MLflow no se reemplazan entre sí:

- TensorBoard analiza la evolución interna del entrenamiento.
- MLflow permite comparar y reproducir experimentos.

---

## 14. Google Colab y optimización de hardware

El código debe diseñarse para aprovechar la GPU de Colab sin depender de un tipo específico de GPU.

### Reglas

- detectar automáticamente `cuda` cuando esté disponible;
- no asumir nombre/modelo específico de GPU;
- mover tensores y modelos al dispositivo de forma explícita;
- evitar conversiones CPU ↔ GPU innecesarias;
- mantener observaciones como `uint8` mientras sea posible antes de convertir a tensor/float;
- evitar almacenar frames duplicados innecesariamente;
- usar minibatches razonables para memoria disponible;
- monitorear RAM y VRAM durante smoke tests;
- liberar entornos y recursos cuando finalicen experimentos;
- persistir artefactos importantes antes de terminar una sesión.

Antes de aumentar `batch_size`, Replay Buffer o arquitectura de red se debe validar su impacto en memoria.

No se debe diseñar el entrenamiento asumiendo sesiones de Colab ilimitadas.

---

## 15. Estrategia de eficiencia computacional

Se seguirá el principio:

> Primero validar barato; después entrenar caro.

Orden esperado:

1. pruebas unitarias simples;
2. validación de dimensiones;
3. inferencia con batch pequeño;
4. uno o pocos pasos de optimización;
5. entrenamiento corto;
6. smoke test end-to-end;
7. entrenamiento completo.

Un entrenamiento largo no deberá utilizarse para descubrir errores básicos de forma, dispositivo, Replay Buffer o persistencia.

---

## 16. Replay Buffer

Para DDQN se utilizará **Experience Replay uniforme**.

No se implementará Prioritized Experience Replay dentro del agente DDQN seleccionado.

Reglas:

- evitar duplicación innecesaria de observaciones;
- dimensionar capacidad considerando RAM disponible;
- muestrear batches aleatorios;
- documentar estrategia de persistencia;
- validar que `state`, `action`, `reward`, `next_state` y `done` mantienen tipos y dimensiones consistentes.

---

## 17. Preprocessing del entorno

Pipeline objetivo:

```text
RGB 210×160×3
    ↓
grayscale
    ↓
resize 84×84
    ↓
frame stack = 4
    ↓
CNN
```

Reglas:

- el `frameskip` efectivo debe ser 4 una sola vez;
- no aplicar frame skipping simultáneamente en entorno base y wrapper;
- entrenamiento y evaluación deben compartir el mismo preprocessing;
- cualquier cambio del preprocessing implica un nuevo experimento/configuración.

---

## 18. Separación entrenamiento / evaluación

La evaluación debe estar desacoplada del entrenamiento.

Durante evaluación:

- cargar un modelo explícito;
- utilizar el mismo entorno/preprocessing;
- utilizar recompensa real del entorno;
- desactivar exploración o usar epsilon explícitamente documentado;
- ejecutar al menos 10 episodios independientes para evaluación formal;
- calcular media, mediana, desviación estándar, mínimo y máximo;
- comparar contra el baseline aleatorio.

La recompensa de entrenamiento no sustituye la métrica formal de evaluación.

---

## 19. GitHub y control de versiones

GitHub será la fuente de verdad.

Flujo obligatorio para HUs de implementación:

```text
main
  ↓
feature/HU-xxx
  ↓
implementación
  ↓
autovalidaciones
  ↓
Pull Request
  ↓
revisión
  ↓
merge
```

Reglas:

- no desarrollar HUs directamente sobre `main`;
- commits deben tener propósito claro;
- cada PR debe corresponder principalmente a una HU;
- no incluir checkpoints, logs o archivos pesados innecesarios en Git;
- notebooks finales pueden conservar outputs cuando constituyan evidencia académica;
- código reusable debe residir en `.py`, no duplicado dentro del notebook.

---

## 20. Pruebas y autovalidaciones

Cada HU debe incluir autovalidaciones alineadas con `docs/implementacion.md`.

Como mínimo se deben verificar según el alcance:

- imports;
- carga de configuración;
- dimensiones de observaciones;
- dimensiones de Q-values;
- Replay Buffer;
- actualización de pesos;
- sincronización de Target Network;
- save/load;
- checkpoint/resume;
- logs de TensorBoard;
- registro MLflow;
- ejecución en GPU cuando aplique.

Una HU no se considera terminada únicamente porque el código compile.

---

## 21. Manejo de errores

El código debe fallar de manera explícita y entendible.

Preferencias:

- validar configuración al inicio;
- verificar existencia de checkpoints antes de cargarlos;
- verificar compatibilidad entre checkpoint y configuración;
- mostrar dispositivo utilizado;
- generar errores claros ante dimensiones inesperadas;
- evitar `except Exception: pass`;
- no ocultar errores de entrenamiento.

---

## 22. Logging

Los logs deben ser suficientes para entender una ejecución sin saturar la salida.

Cada corrida relevante debe informar al menos:

- run ID;
- device;
- seed;
- configuración principal;
- timestep inicial;
- si inicia o reanuda;
- checkpoint utilizado;
- progreso periódico;
- resultado final.

No imprimir información por cada step salvo durante debugging controlado.

---

## 23. Nomenclatura

Convenciones recomendadas:

### Experimentos

```text
assault_ddqn_exp_001
assault_ddqn_exp_002
```

### Ramas

```text
feature/HU-002-pipeline-entorno
feature/HU-003-nucleo-ddqn
```

### Checkpoints

```text
assault_ddqn_exp_001_step_100000.pt
```

### Modelos finales

```text
assault_ddqn_best.pt
```

Los nombres deben permitir identificar claramente el propósito del artefacto.

---

## 24. Calidad del notebook

`assault_ddqn.ipynb` debe ser ejecutable de principio a fin en Google Colab.

Debe contener:

1. contexto del problema;
2. instalación de dependencias;
3. información de versiones y hardware;
4. configuración;
5. selección y justificación de DDQN;
6. entrenamiento;
7. TensorBoard;
8. MLflow;
9. evaluación;
10. métricas;
11. video/evidencia;
12. análisis de resultados;
13. conclusiones.

El notebook debe consumir módulos de `src/` y evitar duplicar su lógica.

---

## 25. Política de cambios de hiperparámetros

No se deben cambiar varios hiperparámetros simultáneamente sin una hipótesis que permita interpretar el resultado.

Cada ajuste debe registrar:

- valor anterior;
- valor nuevo;
- hipótesis;
- resultado esperado;
- experimento asociado;
- comparación con la corrida anterior.

La optimización deberá ser controlada, no una búsqueda exhaustiva innecesaria.

---

## 26. Política de artefactos

### Versionar normalmente

- código;
- configuración;
- documentación;
- HUs;
- notebooks;
- resultados finales pequeños.

### No versionar rutinariamente

- Replay Buffers;
- checkpoints intermedios pesados;
- logs temporales;
- artefactos duplicados;
- caches.

Los artefactos necesarios para la entrega deben tener una ubicación persistente y documentada.

---

## 27. Seguridad ante pérdida de sesión

Antes y durante entrenamientos largos:

- checkpoints periódicos;
- artefactos persistidos fuera de `/content`;
- logs guardados;
- configuración asociada al run;
- `global_step` persistido;
- posibilidad de resume validada previamente.

No iniciar un entrenamiento largo si la capacidad de reanudar todavía no ha sido probada.

---

## 28. Criterio general de calidad

Una implementación será considerada técnicamente adecuada cuando:

- cumple la HU correspondiente;
- respeta `docs/arquitectura.md`;
- respeta estos lineamientos;
- supera sus autovalidaciones;
- es reproducible;
- no duplica lógica innecesariamente;
- puede ejecutarse en Colab;
- produce evidencia observable;
- permite continuar el entrenamiento sin perder progreso;
- mantiene trazabilidad mediante GitHub y MLflow;
- proporciona métricas mediante TensorBoard;
- utiliza recursos de hardware de forma razonable;
- mantiene el código comprensible y documentado.

---

## 29. Regla de precedencia

Si existe contradicción entre documentos, se deberá resolver antes de implementar.

Orden de referencia recomendado:

1. `enunciado_reto_1.txt` para restricciones académicas obligatorias;
2. `2_Assault/ficha_tecnica.md` para decisiones específicas del entorno y métricas;
3. `docs/arquitectura.md` para arquitectura técnica;
4. `docs/implementacion.md` para secuencia y estándar de HUs;
5. `2_Assault/docs/linemientos.md` para políticas técnicas transversales;
6. HU correspondiente para alcance puntual de implementación.

Una HU no puede invalidar silenciosamente una restricción de nivel superior.
