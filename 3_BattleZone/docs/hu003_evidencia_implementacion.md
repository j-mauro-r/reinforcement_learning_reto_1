# Evidencia de implementación - HU003 BattleZone

## 1. Identificación

- HU: HU003 - Pipeline reproducible del entorno BattleZone
- Rama: `feature/battlezone-hu003-pipeline-entorno`
- Notebook: `3_BattleZone/pipeline_battlezone.ipynb`
- Configuración: `3_BattleZone/configs/battlezone_config.yaml`
- Factory única: `3_BattleZone/src/environment.py`
- Tests: `3_BattleZone/tests/test_environment.py`

## 2. Implementación realizada

Se implementó un pipeline único y reusable para crear `ALE/BattleZone-v5` desde configuración versionada.

Archivos principales:

- `3_BattleZone/configs/battlezone_config.yaml`: centraliza entorno, seed, mode, difficulty, obs_type, frameskip, sticky actions, render, preprocessing, candidatos y contrato esperado.
- `3_BattleZone/src/environment.py`: factory única `create_battlezone_env`, wrappers de preprocessing y frame stack, validaciones de contrato, seed del entorno/action space y verificación de frameskip.
- `3_BattleZone/src/utils.py`: utilidades transversales de runtime/hardware y commit Git.
- `3_BattleZone/tests/test_environment.py`: suite focalizada de contrato y smoke tests.
- `3_BattleZone/pipeline_battlezone.ipynb`: notebook nuevo de HU003, orquestador/reporte.

No se implementó agente, entrenamiento, selección algorítmica, CNN, Replay Buffer, optimizer, checkpoints, TensorBoard ni MLflow.

## 3. Configuración final seleccionada

- Environment ID: `ALE/BattleZone-v5`
- Seed base: `20260903`
- Mode: `1`
- Difficulty: `0`
- `obs_type`: `rgb`
- `frameskip`: `4`
- `repeat_action_probability`: `0.25`
- Action space esperado: `Discrete(18)`
- Render default: `null`
- Pipeline: `battlezone_rgb_128_stack4_no_crop`
- Color: RGB
- Resize: `128x128`
- Cropping: desactivado
- Frame stack: `4`
- Dtype final: `uint8`
- Normalización: desactivada
- Reward transform: `none`

## 4. Alternativas de preprocessing evaluadas

| Candidato | Color | Resize | Cropping | Frame stack | Decisión |
|---|---|---:|---|---:|---|
| `rgb_84_no_crop_stack4` | RGB | `84x84` | No | 4 | Descartado como contrato final: reduce más memoria, pero achica radar y objetos pequeños. |
| `grayscale_84_no_crop_stack4` | Grayscale | `84x84` | No | 4 | Descartado: elimina color y conserva menos señal visual para radar/escena. |
| `rgb_128_no_crop_stack4` | RGB | `128x128` | No | 4 | Seleccionado: conserva color, radar y más detalle con reducción razonable de dimensionalidad. |

No se seleccionó cropping porque el radar está en la región superior y HU002 lo identificó como señal estratégica. El notebook muestra frame RGB original, región del radar y salidas de candidatos.

## 5. Contrato final de observación

- Raw: `(210, 160, 3)` `uint8`
- Final: `(4, 128, 128, 3)` `uint8`
- Rango del espacio final: `[0, 255]`
- Memoria aproximada por estado: `196608` bytes (`0.1875 MB`)
- Orden del pipeline:
  1. `gym.make("ALE/BattleZone-v5", frameskip=4, repeat_action_probability=0.25, obs_type="rgb", mode=1, difficulty=0)`;
  2. seed explícita del action space;
  3. resize RGB sin cropping;
  4. frame stack de 4 observaciones.

El frame stack aporta contexto temporal pero no repite acciones ni modifica rewards.

## 6. Frameskip efectivo

La fábrica configura `frameskip=4` únicamente en ALE. No se añadió wrapper de action repeat/frameskip.

Validación local observada:

- wrapper chain: `FrameStackObservation -> BattleZonePreprocessObservation -> OrderEnforcing -> PassiveEnvChecker -> AtariEnv`
- frame numbers: `[0, 4, 8, 12]`
- frame deltas: `[4, 4, 4]`
- resultado: PASS, sin doble frameskip.

## 7. Train vs eval

`create_battlezone_env(config, mode="train")` y `create_battlezone_env(config, mode="eval")` usan la misma factory y comparten:

- env_id;
- mode/difficulty;
- frameskip;
- sticky actions;
- action space;
- color mode;
- resize;
- cropping;
- frame stack;
- dtype;
- reward transform.

Las diferencias permitidas quedan limitadas a seed y render/video cuando se solicite explícitamente.

## 8. Tests ejecutados

Comando:

```powershell
python -m pytest 3_BattleZone/tests -q
```

Resultado:

```text
9 passed in 4.43s
```

Validación adicional del notebook:

- `python -m json.tool 3_BattleZone/pipeline_battlezone.ipynb`: PASS.
- Runner local de celdas de código en orden: PASS.
- `jupyter run 3_BattleZone/pipeline_battlezone.ipynb`: no usable en este entorno porque intenta ejecutar el JSON como script y falla con `NameError: null`; se documenta como limitación de herramienta local, no del contenido del notebook.

## 9. Autovalidaciones AV01-AV17

| AV | Resultado | Evidencia |
|---|---|---|
| AV01 Carga de configuración | PASS | `load_config()` y `validate_config()` cargan YAML sin notebook. |
| AV02 Creación del entorno | PASS | `create_battlezone_env()` crea BattleZone y `reset()` retorna observación válida. |
| AV03 Action space | PASS | `Discrete(18)` y 18 action meanings esperados. |
| AV04 Contrato raw | PASS | Raw `(210,160,3)` `uint8`. |
| AV05 Comparación preprocessing | PASS | Notebook compara RGB/grayscale y `84x84`/`128x128`. |
| AV06 Radar | PASS | Cropping desactivado; notebook muestra región superior/radar. |
| AV07 Shape/dtype final | PASS | Final `(4,128,128,3)` `uint8`. |
| AV08 Frame stack | PASS | Reset llena stack; step desplaza frames. |
| AV09 Frameskip único | PASS | Contadores ALE avanzan de 4 en 4; sin wrapper repeat adicional. |
| AV10 Train/eval parity | PASS | Tests comparan contrato perceptual train/eval. |
| AV11 Seeds | PASS | `reset(seed=seed)` y `action_space.seed(seed)` aplicados explícitamente. |
| AV12 Reward passthrough | PASS | Test compara rewards raw vs pipeline con mismas acciones/seeds. |
| AV13 Smoke tests | PASS | Suite focalizada local completa en CPU. |
| AV14 Notebook independiente | PASS | `pipeline_battlezone.ipynb` es nuevo y usa `src/` + config. |
| AV15 Independencia de Assault | PASS | Sin cambios ni dependencias bajo `2_Assault/`. |
| AV16 Validación local barata | PASS | Tests y smoke local ejecutados sin GPU. |
| AV17 Alcance algorítmico | PASS | No se añadió agente, entrenamiento ni selección de algoritmo. |

## 10. Limitaciones y riesgos pendientes

- La decisión RGB `128x128` se basa en inspección visual y contrato técnico local; futuras pruebas de entrenamiento deberán perfilar RAM/VRAM y throughput.
- El pipeline conserva `uint8`; la normalización a tensor queda para la capa del agente cuando HU004/HU005 definan algoritmo.
- `frame_stack=4` estabiliza un contrato temporal razonable, pero HU004/HU005 podrán medir si el costo computacional resultante es aceptable.
- No se ejecutó validación Colab completa en esta HU local; el notebook incluye instalación mínima para runtime limpio.
- HU002 permanece como evidencia histórica; no se modificó ni se reutilizó como implementación de HU003.

## 11. Implicaciones para HU004

HU004 puede comparar algoritmos permitidos usando un contrato estable:

- entrada visual: `(4,128,128,3)` `uint8`;
- action space: `Discrete(18)`;
- rewards reales sin transformación;
- `frameskip=4` una sola vez;
- sticky actions `0.25`;
- train/eval con mismo preprocessing.

La selección algorítmica deberá concentrarse en sparse rewards, alta variabilidad, 18 acciones, costo de memoria del estado visual apilado y estabilidad de aprendizaje.
