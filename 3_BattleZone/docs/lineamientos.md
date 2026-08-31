# Lineamientos técnicos de implementación — BattleZone

## 1. Objetivo

Definir las políticas técnicas, buenas prácticas y criterios obligatorios para desarrollar el agente de Reinforcement Learning de `ALE/BattleZone-v5` dentro del Reto 1.

Estos lineamientos complementan `3_BattleZone/docs/implementacion.md` y deben aplicarse a todas las HUs de BattleZone.

BattleZone se desarrollará como una implementación independiente. El trabajo previo de Assault se utilizará únicamente como base de conocimiento y experiencia metodológica. **No se copiará, importará ni reutilizará código de Assault.**

El proyecto aplicará una filosofía de **MLOps ligera**, priorizando reproducibilidad, trazabilidad, recuperación ante fallos, observabilidad y simplicidad.

**Decisión explícita:** BattleZone **no utilizará MLflow**. La trazabilidad se resolverá mediante Git/GitHub, configuración versionada, manifiestos de ejecución, TensorBoard, checkpoints y resultados de evaluación persistidos.

---

## 2. Principios rectores

1. **Simplicidad primero.** No crear infraestructura o abstracciones sin una necesidad concreta.
2. **Reproducibilidad.** Toda corrida relevante debe poder asociarse a código, configuración, seed, versiones y hardware.
3. **Trazabilidad sin MLflow.** Cada experimento debe quedar identificado mediante `run_id`, commit Git, configuración y resumen de resultados persistido.
4. **Separación de responsabilidades.** Entorno, agente, entrenamiento, evaluación, persistencia y observabilidad deben mantenerse desacoplados.
5. **Idempotencia.** Reejecutar una etapa no debe destruir resultados válidos ni reiniciar silenciosamente un entrenamiento.
6. **Observabilidad.** TensorBoard será la herramienta principal para analizar la evolución del entrenamiento.
7. **Recuperabilidad.** Todo entrenamiento largo debe poder continuar desde checkpoints.
8. **Validar barato antes de entrenar caro.** Ninguna corrida larga debe usarse para descubrir errores de integración básicos.
9. **Evidencia antes de optimización.** Los cambios de hiperparámetros deben responder a una hipótesis explícita.
10. **BattleZone independiente.** Ningún módulo de `2_Assault/` puede convertirse en dependencia del agente BattleZone.

---

## 3. Restricciones académicas

El agente debe respetar el enunciado del Reto 1.

Los únicos algoritmos permitidos son:

- DQN;
- DQN + Prioritized Experience Replay;
- DDQN;
- REINFORCE.

La selección definitiva se realizará después de HU001 y HU002.

La evaluación final deberá ejecutarse sobre **al menos 10 partidas independientes** y debe proporcionar evidencia de comportamiento aprendido y no predominantemente aleatorio.

---

## 4. Arquitectura de código

La arquitectura concreta se definirá progresivamente mediante las HUs, pero deberá respetar como mínimo la siguiente separación conceptual:

```text
3_BattleZone/
├── battlezone_agent.ipynb
├── ficha_tecnica.md
├── configs/
├── src/
│   ├── environment.py
│   ├── network.py
│   ├── replay_buffer.py        # cuando aplique
│   ├── agent.py
│   ├── trainer.py
│   ├── evaluator.py
│   ├── callbacks.py
│   └── utils.py
├── tests/
├── checkpoints/
├── models/
├── logs/
├── results/
├── videos/
└── docs/
    ├── implementacion.md
    └── lineamientos.md
```

La estructura definitiva puede variar según el algoritmo elegido, evitando crear módulos que no aporten valor.

### Reglas

- El notebook será **orquestador y reporte**, no repositorio de lógica duplicada.
- La lógica reutilizable debe vivir en `src/`.
- La configuración debe estar centralizada en `configs/`.
- Entrenamiento y evaluación deben estar separados.
- Ningún archivo de BattleZone debe importar código desde `2_Assault/`.

---

## 5. SOLID aplicado de forma pragmática

SOLID se utilizará para reducir acoplamiento y facilitar pruebas, no para construir una arquitectura empresarial innecesaria.

### 5.1 Single Responsibility Principle

Cada módulo debe tener una responsabilidad principal:

- `environment.py`: creación y preprocessing del entorno;
- `network.py`: arquitectura neuronal;
- `replay_buffer.py`: almacenamiento y muestreo de experiencia cuando aplique;
- `agent.py`: lógica específica del algoritmo;
- `trainer.py`: ciclo de entrenamiento;
- `evaluator.py`: evaluación independiente;
- `callbacks.py`: checkpoints y observabilidad periódica;
- `utils.py`: utilidades transversales pequeñas.

### 5.2 Open/Closed Principle

Los cambios experimentales deben realizarse preferiblemente mediante configuración antes que modificando lógica estable.

### 5.3 Liskov Substitution Principle

Si se crean abstracciones compartidas, sus implementaciones deberán respetar los mismos contratos. No se crearán jerarquías de clases si funciones o composición son suficientes.

### 5.4 Interface Segregation Principle

Las interfaces deberán ser pequeñas y enfocadas. Un componente no debe depender de operaciones que no utiliza.

### 5.5 Dependency Inversion Principle

Los componentes de alto nivel, especialmente entrenamiento y evaluación, deben depender de contratos estables del agente y del entorno, no de detalles internos de la red neuronal.

---

## 6. DRY

Debe evitarse duplicar lógica relacionada con:

- creación del entorno;
- preprocessing;
- seeds;
- configuración;
- selección de dispositivo;
- evaluación;
- cálculo de métricas;
- rutas de artefactos;
- save/load;
- checkpoints;
- generación de identificadores de corrida.

No se debe aplicar DRY creando abstracciones complejas para código trivial o que aparece una sola vez.

---

## 7. Documentación estilo Google

Las funciones y clases públicas/reutilizables deben usar docstrings estilo Google.

Ejemplo:

```python
def create_environment(config: dict, seed: int):
    """Creates a configured BattleZone environment.

    Args:
        config: Environment and preprocessing configuration.
        seed: Seed used to initialize the environment.

    Returns:
        Configured Gymnasium environment.

    Raises:
        ValueError: If the environment configuration is invalid.
    """
```

### Reglas

- documentar propósito, argumentos, retornos y errores relevantes;
- explicar decisiones no obvias;
- evitar comentarios que repitan literalmente el código;
- nombres técnicos y código preferiblemente en inglés;
- documentación académica y Markdown pueden mantenerse en español.

---

## 8. Configuración centralizada

Los parámetros que afecten aprendizaje, entorno o evaluación deben centralizarse en archivos de configuración versionados.

Según el algoritmo seleccionado, incluirán:

- environment ID;
- seed;
- preprocessing;
- tamaño de imagen;
- frame stack;
- action space seleccionado;
- Replay Buffer y PER cuando aplique;
- batch size;
- learning rate;
- gamma;
- estrategia de exploración;
- learning starts;
- frecuencia de aprendizaje;
- Target Network sync cuando aplique;
- total timesteps/episodes;
- checkpoint interval;
- logging interval;
- episodios de evaluación.

No deben existir constantes mágicas dispersas entre notebook y módulos.

---

## 9. Reproducibilidad

Cada corrida relevante debe registrar como mínimo:

- `run_id`;
- algoritmo;
- seed;
- commit Git;
- archivo/configuración utilizada;
- versiones de Python, Gymnasium, ALE-Py, PyTorch y dependencias principales;
- hardware utilizado;
- fecha/hora de inicio;
- timestep/episodio inicial;
- timestep/episodio final;
- tiempo de entrenamiento;
- checkpoint/modelo asociado.

No se debe prometer determinismo absoluto cuando ALE, GPU o sticky actions introduzcan fuentes legítimas de variabilidad.

---

## 10. Trazabilidad de experimentos sin MLflow

BattleZone no utilizará MLflow.

Cada experimento relevante deberá producir un manifiesto persistido, por ejemplo:

```text
results/<run_id>/run_manifest.json
```

El manifiesto debe contener, como mínimo:

- `run_id`;
- commit Git;
- algoritmo;
- seed;
- configuración;
- versiones;
- hardware;
- timestep/episodio inicial y final;
- tiempo de entrenamiento;
- rutas de TensorBoard;
- checkpoint/modelo;
- métricas de evaluación cuando existan.

Las comparaciones entre experimentos se realizarán mediante archivos de resultados estructurados, tablas generadas en el notebook y curvas de TensorBoard.

GitHub será la fuente de verdad del código y configuración.

---

## 11. Idempotencia

El proyecto debe soportar ejecuciones repetidas de forma segura.

Reglas obligatorias:

- crear directorios con `exist_ok=True`;
- usar identificadores únicos de corrida;
- diferenciar explícitamente `new run` y `resume`;
- nunca seleccionar automáticamente un checkpoint ambiguo;
- no sobrescribir modelos/checkpoints válidos sin intención explícita;
- no reiniciar `global_step` o contador equivalente al reanudar;
- conservar la configuración asociada al checkpoint;
- mantener entrenamiento y evaluación desacoplados;
- permitir reejecutar celdas del notebook sin eliminar evidencia previa.

---

## 12. Checkpoints

Los checkpoints son obligatorios para entrenamientos largos debido a las limitaciones de sesiones de Google Colab.

Cada checkpoint debe guardar todo el estado necesario para continuar de forma consistente según el algoritmo.

Para algoritmos value-based puede incluir:

- Online Network;
- Target Network cuando aplique;
- optimizer;
- timestep global;
- epsilon/estado de exploración;
- Replay Buffer cuando sea viable;
- configuración;
- métricas de continuidad.

Para REINFORCE deberá conservar el estado necesario de la policy network, optimizer y progreso del entrenamiento.

Se deben soportar:

### Resume completo

Restaura el máximo estado disponible, incluido Replay Buffer cuando corresponda y sea viable.

### Resume liviano

Restaura modelo, optimizer y progreso, reconstruyendo gradualmente la experiencia necesaria.

Antes de iniciar un entrenamiento largo, el flujo save → load → resume debe haber sido validado mediante smoke test.

---

## 13. TensorBoard

TensorBoard será obligatorio durante entrenamientos relevantes.

Registrar como mínimo, cuando aplique:

- recompensa por episodio;
- recompensa media móvil;
- longitud del episodio;
- loss;
- epsilon o métrica equivalente de exploración;
- Q-value medio para métodos value-based;
- timestep/episodio global;
- learning rate si cambia;
- métricas adicionales justificadas para diagnosticar aprendizaje.

TensorBoard debe permitir detectar:

- ausencia de aprendizaje;
- inestabilidad;
- divergencia de loss;
- colapso prematuro de exploración;
- mejora o estancamiento de recompensa.

No registrar métricas únicamente porque estén disponibles.

---

## 14. Preprocessing del entorno

El preprocessing definitivo no debe copiarse automáticamente de Assault.

HU001 y HU002 deberán validar qué pipeline conserva suficiente información visual de BattleZone, especialmente radar, enemigos, proyectiles y obstáculos.

Cualquier decisión sobre:

- grayscale;
- resize;
- frame stack;
- frame skip;
- cropping;
- reward clipping;

debe quedar explícitamente justificada y versionada.

### Regla crítica de frameskip

El `frameskip` efectivo debe aplicarse una sola vez. Se debe validar la interacción entre el entorno base y los wrappers elegidos para evitar duplicación accidental.

Entrenamiento y evaluación deben utilizar el mismo contrato de observación.

---

## 15. Eficiencia de memoria y GPU

- Mantener observaciones como `uint8` mientras sea razonable antes de convertirlas a tensores.
- Mantener Replay Buffer en CPU RAM salvo justificación contraria.
- Transferir a GPU únicamente batches necesarios.
- Evitar conversiones CPU ↔ GPU innecesarias.
- No renderizar durante entrenamiento normal.
- Renderizar únicamente para inspección, evaluación o video.
- Monitorear RAM/VRAM durante smoke tests.
- No aumentar batch size, Replay Buffer o CNN sin validar impacto en memoria.
- Detectar automáticamente CUDA sin depender del modelo específico de GPU.

---

## 16. Estrategia de validación computacional

Orden obligatorio recomendado:

1. imports;
2. carga de configuración;
3. creación del entorno;
4. validación de shapes/dtypes;
5. forward pass;
6. Replay Buffer/trajectory handling;
7. una actualización real del agente;
8. save/load;
9. checkpoint/resume;
10. TensorBoard;
11. smoke test E2E corto;
12. prueba Colab GPU;
13. entrenamiento completo.

Principio:

> Primero validar barato; después entrenar caro.

---

## 17. Entrenamiento y evaluación separados

La evaluación formal no debe reutilizar lógica exploratoria del entrenamiento de manera implícita.

Debe:

- cargar explícitamente el modelo seleccionado;
- usar el mismo entorno/preprocessing;
- usar recompensa real del entorno;
- desactivar exploración deliberada o documentar exactamente la política de evaluación;
- ejecutar al menos 10 episodios independientes;
- calcular media, mediana, desviación estándar, mínimo y máximo;
- comparar contra el baseline aleatorio definido en HU002;
- registrar evidencia cualitativa del comportamiento aprendido.

La recompensa observada durante entrenamiento no sustituye la evaluación formal.

---

## 18. Política de hiperparámetros

No modificar múltiples hiperparámetros simultáneamente sin una hipótesis que permita interpretar el resultado.

Cada experimento de optimización debe registrar:

- `run_id`;
- valor anterior;
- valor nuevo;
- hipótesis;
- resultado esperado;
- resultado observado;
- comparación contra corrida anterior;
- decisión posterior.

La optimización será controlada y limitada por el presupuesto computacional disponible.

---

## 19. GitHub y control de versiones

GitHub será la fuente de verdad del proyecto.

Flujo esperado:

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
- commits con propósito claro;
- cada PR debe limitarse principalmente al alcance de una HU;
- no versionar caches, logs temporales ni checkpoints intermedios pesados;
- el notebook final puede conservar outputs cuando constituyan evidencia académica;
- ninguna rama de BattleZone debe modificar archivos de Assault salvo una decisión explícita y separada del proyecto.

---

## 20. Pruebas y autovalidaciones

Cada HU debe incluir autovalidaciones proporcionales a su alcance.

Podrán incluir:

- imports;
- carga de configuración;
- dimensiones de observaciones;
- action space;
- forward pass;
- Replay Buffer/PER;
- actualización de pesos;
- Target Network sync;
- save/load;
- checkpoint/resume;
- TensorBoard;
- generación del manifiesto de corrida;
- evaluación corta;
- ejecución E2E;
- validación local;
- validación Colab GPU.

Una HU no se considera terminada únicamente porque el código compile o exista.

---

## 21. Manejo de errores

Preferencias obligatorias:

- validar configuración al inicio;
- verificar existencia de checkpoints antes de cargarlos;
- verificar compatibilidad entre checkpoint y configuración;
- mostrar explícitamente el dispositivo utilizado;
- producir errores claros ante dimensiones inesperadas;
- evitar `except Exception: pass`;
- no ocultar errores de entrenamiento;
- fallar temprano ante configuraciones incompatibles.

---

## 22. Logging

Cada corrida relevante debe informar como mínimo:

- `run_id`;
- algoritmo;
- device;
- seed;
- configuración principal;
- timestep/episodio inicial;
- si inicia o reanuda;
- checkpoint utilizado;
- progreso periódico;
- resultado final.

No imprimir información por cada step salvo debugging controlado.

---

## 23. Convenciones de nomenclatura

Ejemplos:

### Runs

```text
battlezone_exp_001
battlezone_exp_002
```

### Ramas

```text
feature/HU-001-battlezone-eda
feature/HU-005-agent-core
```

### Checkpoints

```text
battlezone_exp_001_step_100000.pt
```

### Modelo final

```text
battlezone_best.pt
```

La nomenclatura deberá permitir relacionar fácilmente modelos, logs, resultados y configuración.

---

## 24. Política de artefactos

### Versionar normalmente

- código;
- configuración;
- documentación;
- HUs/DWP;
- notebooks;
- resultados finales pequeños;
- manifiestos de ejecución;
- tablas/resúmenes necesarios para reproducibilidad.

### No versionar rutinariamente

- Replay Buffers;
- checkpoints intermedios pesados;
- logs temporales;
- caches;
- videos temporales;
- artefactos duplicados.

Los artefactos requeridos para la entrega deben tener una ubicación persistente y documentada.

---

## 25. Seguridad ante pérdida de sesión

Antes de cualquier entrenamiento largo:

- estrategia de checkpoints validada;
- ruta persistente definida;
- configuración asociada al run persistida;
- TensorBoard configurado;
- `run_manifest` creado;
- global step/episodio persistible;
- mecanismo de resume probado.

No iniciar un entrenamiento largo si todavía no se ha demostrado que puede reanudarse.

---

## 26. Calidad del notebook final

El notebook deberá ejecutar de principio a fin en Google Colab y funcionar como reporte técnico.

Debe incluir, como mínimo:

1. contexto del problema;
2. caracterización de BattleZone;
3. baseline aleatorio;
4. selección y justificación del algoritmo;
5. dependencias y versiones;
6. hardware;
7. configuración;
8. arquitectura del agente;
9. estrategia de entrenamiento;
10. checkpoints/reanudación;
11. TensorBoard;
12. trazabilidad de experimentos mediante Git/configuración/manifiestos;
13. evaluación formal ≥10 episodios;
14. comparación contra baseline;
15. video/evidencias;
16. análisis del comportamiento aprendido;
17. limitaciones;
18. conclusiones.

El notebook debe consumir módulos de `src/` y no duplicar su lógica.

---

## 27. Criterio general de calidad

Una implementación de BattleZone será técnicamente adecuada cuando:

- cumple el enunciado académico;
- respeta `implementacion.md` y estos lineamientos;
- no reutiliza código de Assault;
- supera sus autovalidaciones;
- es reproducible;
- puede ejecutarse en Colab;
- produce TensorBoard interpretable;
- puede reanudar entrenamientos;
- mantiene trazabilidad sin MLflow;
- separa entrenamiento y evaluación;
- utiliza recursos de hardware razonablemente;
- conserva código comprensible y documentado.

---

## 28. Regla de precedencia

Si existe contradicción entre documentos, debe resolverse antes de implementar.

Orden recomendado:

1. `enunciado_reto_1.txt` — restricciones académicas;
2. `3_BattleZone/ficha_tecnica.md` — decisiones y características específicas del entorno;
3. `3_BattleZone/docs/implementacion.md` — secuencia y alcance de HUs;
4. `3_BattleZone/docs/lineamientos.md` — políticas técnicas transversales;
5. HU/DWP correspondiente — alcance puntual.

Una HU no puede invalidar silenciosamente una restricción de nivel superior.