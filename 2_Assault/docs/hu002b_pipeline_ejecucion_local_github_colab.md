# HU002B — Pipeline de ejecución Local → GitHub → Colab

## 1. Identificación

- **ID:** HU002B
- **Nombre:** Pipeline de ejecución Local → GitHub → Colab
- **Estado:** Lista para implementación
- **Dependencia previa:** HU002 — Pipeline reproducible del entorno, con validación local completada
- **Habilita:** HU003 — Núcleo DDQN
- **Gate relacionado:** completar AV09 pendiente de HU002
- **Fuentes de verdad:**
  - `2_Assault/docs/implementacion.md`
  - `2_Assault/docs/arquitectura.md`
  - `2_Assault/docs/linemientos.md`
  - `2_Assault/docs/ficha_tecnica.md`
  - `2_Assault/docs/hu002_pipeline_reproducible_entorno.md`
  - `2_Assault/assault_ddqn.ipynb`
  - `enunciado_reto_1.txt`

---

## 2. Contexto y problema

HU002 implementó y validó localmente el pipeline reproducible de `ALE/Assault-v5`. Las pruebas locales demostraron que entorno, preprocessing, seeds, action space, `frameskip` y notebook funcionan correctamente en Windows.

Sin embargo, un notebook abierto desde VS Code puede estar conectado a un kernel remoto de Google Colab. En ese escenario, el archivo `.ipynb` pertenece a la copia local, mientras que el código Python se ejecuta en un filesystem remoto y efímero donde el repositorio no existe automáticamente.

Además, para reducir intervención manual y permitir que un agente como Codex pueda completar mayor parte de la validación, el proyecto debe contemplar un canal de **ejecución remota automatizable** hacia el runtime Colab mediante CLI oficial o mecanismo equivalente. Este canal es complementario al notebook y no sustituye la ejecución manual cuando no esté disponible.

El proyecto necesita formalizar el flujo:

```text
VS Code / desarrollo local
        ↓
validaciones locales
        ↓
Git commit + push
        ↓
GitHub = fuente de verdad
        ↓
Google Colab = copia de ejecución
        ↓
CLI / canal remoto automatizable (si disponible)
        ↓
mismo código / commit identificado
        ↓
notebook + smoke remoto reproducible
```

La solución debe mantener la filosofía MLOps ligera: GitHub conserva código/configuración, VS Code es el entorno de desarrollo y Colab funciona como runner de cómputo.

---

## 3. Historia de usuario

> **Como** equipo que desarrolla y entrena el agente DDQN de Assault, **quiero** que el proyecto prepare y valide automáticamente una copia de ejecución en Google Colab a partir de una rama o commit publicado en GitHub, **para** asegurar que cada ejecución remota utilice exactamente una versión identificable y reproducible del código y pueda ser validada con la mínima intervención manual posible.

---

## 4. Objetivo verificable

Al finalizar HU002B debe ser posible:

1. desarrollar y validar cambios localmente;
2. publicar dichos cambios en GitHub;
3. abrir `assault_ddqn.ipynb` desde VS Code o Colab;
4. conectar un runtime limpio de Google Colab;
5. verificar si existe acceso remoto automatizable al runtime Colab;
6. ejecutar bootstrap antes de importar `src`;
7. clonar o sincronizar el repositorio bajo `/content`;
8. seleccionar explícitamente rama o commit;
9. imprimir el commit SHA exacto utilizado;
10. instalar dependencias desde la copia versionada;
11. importar código desde la copia remota;
12. ejecutar las validaciones HU002 sin cambios manuales;
13. ejecutar smoke remoto mediante CLI o mecanismo equivalente cuando el acceso esté disponible;
14. recuperar logs/resultados de la ejecución remota;
15. repetir el bootstrap de forma segura dentro del mismo runtime;
16. disponer de un fallback manual documentado cuando la automatización remota no esté disponible.

Resultado esperado:

```text
Local → GitHub → Colab → mismo commit → ejecución reproducible
                         ↓
                  automatizable si hay CLI
                         ↓
                 fallback manual si no
```

---

## 5. Alcance

### 5.1 Desarrollo local

El desarrollo continúa realizándose en VS Code sobre la copia local del repositorio. Antes de publicar cambios relevantes se deben ejecutar las pruebas correspondientes, hacer commit y publicar la rama.

Colab no será el entorno primario de edición.

### 5.2 GitHub como fuente de verdad

Google Colab deberá obtener el código desde:

`https://github.com/j-mauro-r/reinforcement_learning_reto_1`

La copia bajo `/content` será una **copia de ejecución**. No debe crear commits, merges ni pushes automáticamente.

### 5.3 Canal remoto automatizable hacia Colab

Antes de asumir que Codex u otro agente puede ejecutar el runtime remoto, HU002B debe verificarlo explícitamente.

El mecanismo preferido será una CLI oficial de Google Colab o un mecanismo equivalente que permita:

- enviar comandos al runtime remoto;
- ejecutar Python o shell;
- recuperar stdout/stderr;
- consultar disponibilidad de GPU;
- ejecutar smoke tests del proyecto.

La HU **no debe asumir** que seleccionar un kernel Colab dentro de VS Code implica automáticamente que Codex tenga acceso a dicho kernel.

Prueba mínima conceptual:

```bash
colab exec "import torch; print(torch.cuda.is_available())"
```

El comando exacto podrá variar según la CLI oficialmente disponible en el momento de implementación.

Si el canal remoto no puede configurarse o no es accesible para el agente, la implementación debe continuar con un **fallback manual explícito**: el usuario ejecuta el notebook/smoke en Colab y entrega la evidencia requerida.

### 5.4 Bootstrap de Colab

`2_Assault/assault_ddqn.ipynb` deberá incorporar al inicio una sección de bootstrap ejecutable antes de cualquier import del proyecto.

#### Runtime limpio

```text
git clone
↓
seleccionar ref
↓
identificar commit
↓
instalar dependencias
↓
configurar working directory/imports
```

#### Runtime ya utilizado

```text
validar copia
↓
git fetch
↓
actualización controlada
↓
resolver ref/commit
↓
verificar commit
```

No se deben generar merges automáticos.

### 5.5 Rama y commit

El bootstrap deberá soportar:

- **modo desarrollo:** rama explícita;
- **modo formal/reproducible:** commit SHA explícito.

Aunque se utilice una rama, siempre debe resolverse e imprimirse el SHA real ejecutado.

### 5.6 Idempotencia

Reejecutar el bootstrap no debe:

- clonar múltiples copias;
- destruir artefactos válidos;
- crear merges;
- cambiar silenciosamente de rama;
- utilizar un commit distinto sin informarlo.

Si cambia el commit después de haber importado módulos `src.*`, la corrida formal debe bloquearse o exigir reinicio explícito del runtime/kernel.

### 5.7 Working directory e imports

Después del bootstrap deberán quedar explícitos:

- `PROJECT_ROOT`;
- `ASSAULT_DIR`;
- ref solicitada;
- commit SHA;
- ubicación desde la cual se importan módulos.

En Colab se debe comprobar que `src.environment` proviene de `/content/...` y no de la copia local de VS Code.

### 5.8 Dependencias

La instalación deberá consumir únicamente:

`2_Assault/requirements.txt`

correspondiente al mismo commit que se ejecuta.

### 5.9 Compatibilidad local / Colab

El mismo notebook debe soportar:

**Local**
- usar repo existente;
- no clonar;
- resolver SHA local;
- conservar validaciones HU002.

**Colab**
- crear o sincronizar copia bajo `/content`;
- resolver SHA remoto;
- importar desde esa copia.

### 5.10 Validación HU002 en Colab

HU002B deberá ejecutar desde runtime limpio las validaciones HU002:

- `(4, 84, 84)`;
- `uint8`;
- `Discrete(7)`;
- acciones esperadas;
- `frameskip=4` una sola vez;
- train/eval;
- interacción corta;
- detección de hardware.

Esta ejecución completa AV09 de HU002.

### 5.11 Evidencia de ejecución remota

Cuando el canal automatizado esté disponible, debe registrarse como mínimo:

- comando ejecutado;
- ref/commit SHA;
- versión Python remota;
- disponibilidad de GPU;
- resultado de smoke test;
- stdout/stderr relevante.

La ausencia de CLI no debe provocar resultados inventados: activa el fallback manual.

---

## 6. Fuera de alcance

HU002B **no** debe implementar:

- CNN;
- Online/Target Network;
- Replay Buffer;
- lógica DDQN;
- entrenamiento real;
- checkpoints de modelo;
- TensorBoard avanzado;
- servidor MLflow remoto;
- almacenamiento remoto MLflow;
- GitHub Actions para entrenamiento GPU;
- despliegue productivo;
- automatización CI/CD empresarial;
- entrenamientos automáticos al hacer push;
- infraestructura propia para sustituir servicios oficiales de Colab.

MLflow remoto y trazabilidad avanzada permanecen bajo **HU008**.

---

## 7. Decisiones y restricciones técnicas

### 7.1 Rol de las plataformas

```text
VS Code local
  desarrollo + pruebas
        ↓
GitHub
  fuente de verdad
        ↓
Google Colab
  runner de cómputo
        ↓
CLI remota opcional
  automatización de smoke/validación
```

### 7.2 Colab es desechable

La copia `/content/reinforcement_learning_reto_1` es efímera. La pérdida de `/content` no constituye pérdida de código porque GitHub es la fuente de verdad.

### 7.3 Sincronización segura

Preferir:

- `git clone`;
- `git fetch`;
- checkout/ref explícita;
- fast-forward cuando corresponda.

Evitar:

- merges automáticos;
- `git pull` ambiguo;
- push desde Colab;
- continuar con cambios locales desconocidos dentro de la copia remota.

### 7.4 Pin por commit

Para evidencia académica o experimentos formales debe preferirse commit SHA inmutable. Las ramas pueden usarse en desarrollo, pero el SHA resuelto debe registrarse siempre.

### 7.5 Imports obsoletos

Python mantiene módulos importados en memoria. Para una corrida formal:

```text
cambio de commit
↓
runtime ya importó src
↓
reiniciar runtime
↓
bootstrap
↓
imports
↓
ejecución
```

No se dependerá de `importlib.reload` como estrategia formal de reproducibilidad.

### 7.6 Repositorio público

El repositorio es público, por lo que esta HU no necesita almacenar PAT ni secretos en el notebook.

### 7.7 MLOps ligera

HU002B es un **pipeline reproducible de ejecución de experimentos**, no un CI/CD productivo. HU008 añadirá MLflow sobre esta base.

### 7.8 Automatización remota como capacidad opcional

La CLI remota es una mejora operativa, no una dependencia arquitectónica obligatoria para que el proyecto pueda ejecutarse. Por eso:

- si la CLI funciona, Codex puede realizar más validaciones end-to-end;
- si la CLI no funciona, el usuario ejecuta el gate remoto manualmente;
- en ambos casos, GitHub sigue siendo la fuente de verdad y Colab el runner.

---

## 8. Plan de implementación / tareas

### T00 — Validar acceso remoto automatizable a Colab

**Objetivo:** comprobar si el agente puede ejecutar comandos contra un runtime Colab previamente activado.

**Resultado esperado:** se obtiene una salida remota verificable, por ejemplo versión Python y disponibilidad de CUDA.

**Fallback:** si no existe acceso automatizable, documentar la limitación y activar ejecución manual del usuario para las tareas remotas.

### T01 — Definir contrato del bootstrap

**Archivo:** `2_Assault/assault_ddqn.ipynb`

Definir en una única sección:

- repository URL;
- ruta remota;
- branch/ref;
- commit opcional;
- modo local/Colab.

### T02 — Detectar runtime

Detectar de forma simple ejecución local o Google Colab.

### T03 — Implementar bootstrap Colab

Implementar clone/fetch, selección de ref, resolución de commit y working directory.

### T04 — Controlar rama/commit

Permitir rama explícita y commit pin. Registrar siempre SHA final.

### T05 — Instalar dependencias

Instalar `2_Assault/requirements.txt` de la copia sincronizada.

### T06 — Configurar imports

Configurar rutas después del bootstrap y verificar la ubicación real de `src.environment`.

### T07 — Proteger contra stale imports

Detectar cambio de commit con módulos ya importados y bloquear/solicitar reinicio.

### T08 — Mantener compatibilidad local

Asegurar que local use el repo existente y continúe pasando HU002.

### T09 — Validar local

Ejecutar:

```bash
python -m pytest 2_Assault/tests -q
```

y el notebook o ejecución automatizada equivalente.

### T10 — Ejecutar smoke remoto automatizado cuando sea posible

Usar el canal remoto validado en T00 para:

1. identificar runtime;
2. consultar Python/GPU;
3. ejecutar bootstrap;
4. verificar SHA;
5. ejecutar smoke HU002;
6. recuperar logs.

Si T00 falla, esta tarea se ejecuta manualmente por el usuario mediante el notebook.

### T11 — Validar Colab limpio

Ejecutar desde runtime nuevo:

1. bootstrap;
2. dependencias;
3. SHA;
4. imports;
5. validaciones HU002;
6. interacción corta;
7. registro de evidencia.

La ejecución puede ser automatizada si T00/T10 funcionan o manual si aplica fallback.

### T12 — Actualizar documentación

Solo con evidencia real:

- HU002 → `[COMPLETADA]` si AV09 pasa;
- HU002B → `[COMPLETADA]` si todos sus criterios pasan.

---

## 9. Criterios de aceptación

### CA01 — Separación de responsabilidades

**Dado** el flujo del proyecto, **cuando** se revisa HU002B, **entonces** VS Code se usa para desarrollo, GitHub como fuente de verdad y Colab como runner.

### CA02 — Detección de acceso remoto

**Dado** un runtime Colab activo, **cuando** se prueba el canal CLI/remoto desde el entorno del agente, **entonces** se obtiene una respuesta verificable o se declara explícitamente que debe usarse fallback manual.

### CA03 — Bootstrap limpio

**Dado** un runtime Colab sin repositorio, **cuando** se ejecuta bootstrap, **entonces** el proyecto se clona automáticamente bajo una ruta conocida.

### CA04 — Idempotencia

**Dado** que la copia ya existe, **cuando** se reejecuta bootstrap, **entonces** no se crea otra copia ni un merge automático.

### CA05 — Ref explícita

**Dada** una rama o commit configurado, **cuando** se prepara la copia, **entonces** el SHA real ejecutado queda visible.

### CA06 — GitHub como fuente de verdad

**Dado** código publicado, **cuando** Colab ejecuta, **entonces** importa módulos desde la copia obtenida de GitHub.

### CA07 — Dependencias reproducibles

**Dado** runtime limpio, **cuando** se prepara el proyecto, **entonces** las dependencias provienen de `2_Assault/requirements.txt`.

### CA08 — Imports verificables

**Dado** Colab, **cuando** se importa `src.environment`, **entonces** se demuestra que su ruta pertenece a `/content/reinforcement_learning_reto_1/2_Assault/src/...`.

### CA09 — Protección stale imports

**Dado** un módulo ya importado, **cuando** cambia el commit, **entonces** el flujo no continúa silenciosamente mezclando versiones.

### CA10 — Compatibilidad local

**Dado** ejecución local, **cuando** corre bootstrap, **entonces** utiliza repo local y HU002 sigue funcionando.

### CA11 — Ejecución remota automatizable o fallback

**Dado** un runtime Colab, **cuando** existe canal remoto compatible, **entonces** el smoke puede ejecutarse y recuperar sus logs sin intervención manual; si no existe, el flujo documenta y utiliza fallback manual sin bloquear el proyecto.

### CA12 — AV09 HU002

**Dado** runtime limpio Colab, **cuando** se ejecuta `assault_ddqn.ipynb` en orden, **entonces** HU002 termina sin modificaciones manuales y queda evidencia verificable.

### CA13 — Sin scope creep

**Dado** el PR, **cuando** se revisa, **entonces** no incluye DDQN, Replay Buffer, entrenamiento ni infraestructura MLflow remota.

---

## 10. Autovalidaciones obligatorias

### AV00 — Conectividad remota Colab

**Procedimiento:** ejecutar mediante CLI/mecanismo remoto un comando mínimo que reporte Python y, si existe PyTorch, disponibilidad de CUDA.

**Esperado:** salida remota recuperable.

**Éxito:** PASS automatizado si existe acceso; FALLBACK DOCUMENTADO si el mecanismo no está disponible.

### AV01 — Tests locales

**Procedimiento:** `python -m pytest 2_Assault/tests -q`.

**Esperado:** todos los tests pasan.

### AV02 — Notebook local

**Procedimiento:** ejecutar las secciones implementadas del notebook localmente.

**Esperado:** detecta local, usa repo local y completa HU002.

### AV03 — Clone Colab limpio

**Procedimiento:** iniciar runtime limpio y ejecutar bootstrap.

**Esperado:** repo clonado, ruta/ref/SHA visibles.

### AV04 — Idempotencia Colab

**Procedimiento:** reejecutar bootstrap.

**Esperado:** misma copia, sin segundo clone ni merge y mismo SHA si la ref no cambió.

### AV05 — Commit pinning

**Procedimiento:** configurar un SHA publicado.

**Esperado:** `git rev-parse HEAD` coincide exactamente.

### AV06 — Origen de imports

**Procedimiento:** inspeccionar ubicación de `src.environment`.

**Esperado:** módulo bajo `/content/reinforcement_learning_reto_1/2_Assault/src/`.

### AV07 — Stale imports

**Procedimiento:** probar controladamente cambio de commit después de importar `src`.

**Esperado:** bloqueo o exigencia explícita de reinicio.

### AV08 — Smoke remoto automatizado

**Procedimiento:** cuando AV00 sea PASS, ejecutar mediante CLI/remoto el smoke HU002 y recuperar stdout/stderr.

**Esperado:** validaciones HU002 exitosas y evidencia asociada al SHA ejecutado.

**Fallback:** si AV00 no es automatizable, esta autovalidación se sustituye por AV09 ejecutada manualmente por el usuario.

### AV09 — HU002 en Colab

**Procedimiento:** ejecutar las celdas HU002 completas desde runtime limpio, automatizada o manualmente.

**Esperado:** shape, dtype, acciones, frameskip, train/eval, interacción y hardware válidos.

**Resultado adicional:** completa AV09 original de HU002.

### AV10 — Sin escritura Git desde Colab

**Procedimiento:** revisar bootstrap y ejecución.

**Esperado:** no hay commits, merges ni pushes automáticos.

### AV11 — Reproducibilidad documental

**Procedimiento:** registrar ref y SHA ejecutados, canal usado (CLI/manual) y resultado.

**Esperado:** otra persona puede identificar exactamente qué código se utilizó y cómo se validó.

---

## 11. Evidencias requeridas

El PR debe incluir o referenciar:

- resultado de tests locales;
- ejecución local del notebook;
- resultado de AV00 indicando si existe canal remoto automatizable;
- comando remoto utilizado cuando aplique;
- salida de bootstrap Colab limpio;
- ruta remota;
- rama/ref;
- commit SHA;
- origen real de imports;
- versión Python remota;
- disponibilidad de GPU cuando aplique;
- logs del smoke remoto cuando AV00 sea PASS;
- validaciones HU002 en Colab;
- evidencia del fallback manual cuando la automatización no esté disponible;
- evidencia de idempotencia;
- evidencia de protección ante stale imports.

No deben declararse exitosas validaciones Colab que no hayan sido ejecutadas realmente.

---

## 12. Definition of Done

HU002B se considera terminada únicamente cuando:

- [ ] T00 identifica si existe acceso remoto automatizable a Colab;
- [ ] bootstrap local/Colab implementado;
- [ ] runtime limpio Colab clona el repo automáticamente;
- [ ] reejecutar bootstrap es idempotente;
- [ ] rama/ref y SHA quedan visibles;
- [ ] commit pinning funciona;
- [ ] dependencias se instalan desde el mismo commit;
- [ ] imports se realizan desde la copia correcta;
- [ ] existe protección contra stale imports;
- [ ] ejecución local continúa funcionando;
- [ ] tests locales pasan;
- [ ] smoke remoto se ejecuta vía CLI cuando AV00 sea PASS, o se usa fallback manual documentado;
- [ ] notebook ejecuta HU002 en runtime limpio Colab;
- [ ] AV09 original de HU002 queda completada;
- [ ] no existen escrituras Git automáticas desde Colab;
- [ ] no se implementan componentes de HU003+;
- [ ] MLflow remoto permanece fuera de alcance;
- [ ] documentación de estado/evidencia actualizada;
- [ ] PR limitado al alcance de HU002B;
- [ ] evidencia verificable disponible.

---

## 13. Riesgos y consideraciones

### Kernel remoto vs archivo local

El notebook local y el filesystem Colab son contextos distintos. No se debe asumir que archivos locales existen remotamente.

### Acceso del agente al runtime

Que VS Code esté conectado a Colab no garantiza que Codex u otro agente tenga acceso al mismo kernel. La conectividad debe verificarse explícitamente mediante AV00.

### Evolución de herramientas Colab

La CLI o mecanismo de ejecución remota puede cambiar. La implementación debe depender de interfaces oficiales disponibles y mantener fallback manual.

### Stale imports

Actualizar Git no actualiza módulos ya cargados en memoria. Las corridas formales deben reiniciar runtime cuando cambie el commit después de imports.

### Dependencias

Cambios en `requirements.txt` pueden requerir reinicio de runtime. Debe quedar visible cuando una instalación modifica librerías ya importadas.

### Branch mutable

Una rama puede avanzar entre ejecuciones. Para experimentos formales utilizar SHA.

### Runtime efímero

La eliminación de `/content` es normal. Código y configuración viven en GitHub; checkpoints persistentes corresponden a HUs posteriores.

---

## 14. Resultado esperado y gate para HU003

Al cerrar HU002B debe estar probado:

```text
VS Code
   ↓
validación local
   ↓
GitHub
   ↓
branch / commit explícito
   ↓
Colab limpio
   ↓
bootstrap idempotente
   ↓
SHA verificado
   ↓
CLI smoke automático, si disponible
   │
   └── fallback manual, si no
   ↓
HU002 validada en Colab
```

**Gate:** HU003 puede iniciar únicamente cuando:

1. HU002B esté cerrada;
2. AV09 original de HU002 esté ejecutada y aprobada;
3. el contrato del entorno `(4, 84, 84) uint8 / Discrete(7) / frameskip=4` siga estable;
4. exista evidencia del SHA ejecutado en Colab.

MLflow remoto permanece en HU008. Así se evita introducir infraestructura MLOps avanzada antes de estabilizar el pipeline básico de ejecución.
