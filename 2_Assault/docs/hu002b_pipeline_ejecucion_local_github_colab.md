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
mismo código / commit identificado
        ↓
notebook reproducible
```

La solución debe mantener la filosofía MLOps ligera: GitHub conserva código/configuración, VS Code es el entorno de desarrollo y Colab funciona como runner de cómputo.

---

## 3. Historia de usuario

> **Como** equipo que desarrolla y entrena el agente DDQN de Assault, **quiero** que el notebook prepare automáticamente una copia de ejecución en Google Colab a partir de una rama o commit publicado en GitHub, **para** asegurar que cada ejecución remota utilice exactamente una versión identificable y reproducible del código.

---

## 4. Objetivo verificable

Al finalizar HU002B debe ser posible:

1. desarrollar y validar cambios localmente;
2. publicar dichos cambios en GitHub;
3. abrir `assault_ddqn.ipynb` desde VS Code o Colab;
4. conectar un runtime limpio de Google Colab;
5. ejecutar bootstrap antes de importar `src`;
6. clonar o sincronizar el repositorio bajo `/content`;
7. seleccionar explícitamente rama o commit;
8. imprimir el commit SHA exacto utilizado;
9. instalar dependencias desde la copia versionada;
10. importar código desde la copia remota;
11. ejecutar las validaciones HU002 sin cambios manuales;
12. repetir el bootstrap de forma segura dentro del mismo runtime.

Resultado esperado:

```text
Local → GitHub → Colab → mismo commit → ejecución reproducible
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

### 5.3 Bootstrap de Colab

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

### 5.4 Rama y commit

El bootstrap deberá soportar:

- **modo desarrollo:** rama explícita;
- **modo formal/reproducible:** commit SHA explícito.

Aunque se utilice una rama, siempre debe resolverse e imprimirse el SHA real ejecutado.

### 5.5 Idempotencia

Reejecutar el bootstrap no debe:

- clonar múltiples copias;
- destruir artefactos válidos;
- crear merges;
- cambiar silenciosamente de rama;
- utilizar un commit distinto sin informarlo.

Si cambia el commit después de haber importado módulos `src.*`, la corrida formal debe bloquearse o exigir reinicio explícito del runtime/kernel.

### 5.6 Working directory e imports

Después del bootstrap deberán quedar explícitos:

- `PROJECT_ROOT`;
- `ASSAULT_DIR`;
- ref solicitada;
- commit SHA;
- ubicación desde la cual se importan módulos.

En Colab se debe comprobar que `src.environment` proviene de `/content/...` y no de la copia local de VS Code.

### 5.7 Dependencias

La instalación deberá consumir únicamente:

`2_Assault/requirements.txt`

correspondiente al mismo commit que se ejecuta.

### 5.8 Compatibilidad local / Colab

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

### 5.9 Validación HU002 en Colab

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
- entrenamientos automáticos al hacer push.

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
```

HU002B implementa exclusivamente esta cadena.

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

---

## 8. Plan de implementación / tareas

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

### T10 — Validar Colab limpio

Ejecutar desde runtime nuevo:

1. bootstrap;
2. dependencias;
3. SHA;
4. imports;
5. validaciones HU002;
6. interacción corta;
7. registro de evidencia.

### T11 — Actualizar documentación

Solo con evidencia real:

- HU002 → `[COMPLETADA]` si AV09 pasa;
- HU002B → `[COMPLETADA]` si todos sus criterios pasan.

---

## 9. Criterios de aceptación

### CA01 — Separación de responsabilidades

**Dado** el flujo del proyecto, **cuando** se revisa HU002B, **entonces** VS Code se usa para desarrollo, GitHub como fuente de verdad y Colab como runner.

### CA02 — Bootstrap limpio

**Dado** un runtime Colab sin repositorio, **cuando** se ejecuta bootstrap, **entonces** el proyecto se clona automáticamente bajo una ruta conocida.

### CA03 — Idempotencia

**Dado** que la copia ya existe, **cuando** se reejecuta bootstrap, **entonces** no se crea otra copia ni un merge automático.

### CA04 — Ref explícita

**Dada** una rama o commit configurado, **cuando** se prepara la copia, **entonces** el SHA real ejecutado queda visible.

### CA05 — GitHub como fuente de verdad

**Dado** código publicado, **cuando** Colab ejecuta, **entonces** importa módulos desde la copia obtenida de GitHub.

### CA06 — Dependencias reproducibles

**Dado** runtime limpio, **cuando** se prepara el proyecto, **entonces** las dependencias provienen de `2_Assault/requirements.txt`.

### CA07 — Imports verificables

**Dado** Colab, **cuando** se importa `src.environment`, **entonces** se demuestra que su ruta pertenece a `/content/reinforcement_learning_reto_1/2_Assault/src/...`.

### CA08 — Protección stale imports

**Dado** un módulo ya importado, **cuando** cambia el commit, **entonces** el flujo no continúa silenciosamente mezclando versiones.

### CA09 — Compatibilidad local

**Dado** ejecución local, **cuando** corre bootstrap, **entonces** utiliza repo local y HU002 sigue funcionando.

### CA10 — AV09 HU002

**Dado** runtime limpio Colab, **cuando** se ejecuta `assault_ddqn.ipynb` en orden, **entonces** HU002 termina sin modificaciones manuales y queda evidencia verificable.

### CA11 — Sin scope creep

**Dado** el PR, **cuando** se revisa, **entonces** no incluye DDQN, Replay Buffer, entrenamiento ni infraestructura MLflow remota.

---

## 10. Autovalidaciones obligatorias

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

### AV08 — HU002 en Colab

**Procedimiento:** ejecutar las celdas HU002 completas desde runtime limpio.

**Esperado:** shape, dtype, acciones, frameskip, train/eval, interacción y hardware válidos.

**Resultado adicional:** completa AV09 de HU002.

### AV09 — Sin escritura Git desde Colab

**Procedimiento:** revisar bootstrap y ejecución.

**Esperado:** no hay commits, merges ni pushes automáticos.

### AV10 — Reproducibilidad documental

**Procedimiento:** registrar ref y SHA ejecutados.

**Esperado:** otra persona puede identificar exactamente qué código se utilizó.

---

## 11. Evidencias requeridas

El PR debe incluir o referenciar:

- resultado de tests locales;
- ejecución local del notebook;
- salida de bootstrap Colab limpio;
- ruta remota;
- rama/ref;
- commit SHA;
- origen real de imports;
- validaciones HU002 en Colab;
- evidencia `frameskip`;
- hardware Colab;
- reejecución idempotente;
- confirmación de ausencia de merges/push desde Colab.

No deben declararse exitosas validaciones Colab que no hayan sido ejecutadas realmente.

---

## 12. Definition of Done

HU002B se considera terminada únicamente cuando:

- [ ] bootstrap local/Colab implementado;
- [ ] runtime limpio Colab clona el repo automáticamente;
- [ ] reejecución no duplica la copia;
- [ ] no se producen merges automáticos;
- [ ] rama/ref es explícita;
- [ ] SHA queda registrado;
- [ ] se puede fijar un commit concreto;
- [ ] dependencias provienen de `2_Assault/requirements.txt`;
- [ ] working directory/imports se configuran después del bootstrap;
- [ ] origen de imports verificado;
- [ ] protección contra stale imports validada;
- [ ] ejecución local continúa funcionando;
- [ ] tests locales pasan;
- [ ] notebook ejecuta desde runtime limpio de Colab;
- [ ] HU002 pasa en Colab;
- [ ] AV09 de HU002 queda completada;
- [ ] `implementacion.md` refleja estados reales;
- [ ] no se implementó MLflow remoto;
- [ ] no se implementó lógica DDQN;
- [ ] PR limitado al alcance de HU002B;
- [ ] evidencia verificable disponible.

---

## 13. Riesgos y consideraciones

### Kernel remoto vs archivo local

El notebook local y el filesystem Colab son contextos distintos. No se debe asumir que archivos locales existen remotamente.

### Cambios de commit durante una sesión

Cambiar branch/commit con módulos ya importados puede generar mezcla de versiones. Para una corrida formal se debe reiniciar runtime y volver a ejecutar bootstrap.

### Dependencias

El `requirements.txt` utilizado debe corresponder al mismo commit ejecutado.

### Runtime efímero

La eliminación de `/content` es normal. Código y configuración viven en GitHub; checkpoints persistentes corresponden a HUs posteriores.

---

## 14. Resultado esperado y gate para HU003

Al cerrar HU002B debe estar probado:

```text
VS Code
  ↓
desarrollo
  ↓
tests locales
  ↓
commit + push
  ↓
GitHub
  ↓
runtime limpio Colab
  ↓
clone/fetch
  ↓
ref + SHA verificado
  ↓
dependencias
  ↓
imports desde /content
  ↓
HU002 validations PASS
```

**Gate:** HU003 no debe comenzar hasta que HU002B reproduzca exitosamente en Google Colab el pipeline HU002 ya validado localmente.

---

## 15. Relación con HU008

HU002B prepara información que HU008 reutilizará:

- commit SHA;
- ref;
- hardware;
- configuración;
- runtime.

HU008 será responsable de conectar esos datos con MLflow y definir tracking/artefactos persistentes.

```text
HU002B
GitHub → Colab reproducible
        ↓
HU008
Colab → MLflow trazable
```

Así se evita introducir infraestructura MLOps avanzada antes de estabilizar el pipeline básico de ejecución.
