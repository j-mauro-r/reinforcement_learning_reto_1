# HU011 — Matriz de cumplimiento del entregable Assault

Esta matriz es la contraparte documental del gate programático incluido en `assault_ddqn.ipynb`. Durante implementación, `IMPLEMENTADO` significa que existe código/evidencia preparada; **no equivale a PASS final**. El estado final se obtiene únicamente de una ejecución completa del notebook en Google Colab y de las autovalidaciones de HU011.

| CA | Criterio | Evidencia implementada | Validación final requerida | Estado actual |
|---|---|---|---|---|
| CA01 | Método permitido | `DDQNAgent`, Online/Target, reporte | inspección + notebook | IMPLEMENTADO |
| CA02 | Notebook Colab ejecutable | `assault_ddqn.ipynb`, bootstrap `auto` | Run All Colab | PENDIENTE VALIDACIÓN |
| CA03 | Dependencias | `requirements.txt` | instalación runtime limpio | IMPLEMENTADO |
| CA04 | Modelo entrenado | `model_artifact.py`, SHA/metadata | export/load real | IMPLEMENTADO |
| CA05 | Modelo corresponde al entrenamiento | checkpoint/model lineage | gate `lineage_pass` | IMPLEMENTADO |
| CA06 | Video MP4 | `video.py`, display inline | reproducción real | IMPLEMENTADO |
| CA07 | Video evidencia entrenamiento | intro con metadata real | inspección MP4 | IMPLEMENTADO |
| CA08 | Video comportamiento aprendido | gameplay `rgb_array`, eps=0 | inspección MP4 | IMPLEMENTADO |
| CA09 | Justificación DDQN | reporte técnico notebook | revisión académica | IMPLEMENTADO |
| CA10 | Hiperparámetros | config perfil `full` | salida notebook | IMPLEMENTADO |
| CA11 | Versiones | `get_runtime_info()` | salida Colab | IMPLEMENTADO |
| CA12 | Hardware | runtime + evidencia HU009 | salida/reporte | IMPLEMENTADO |
| CA13 | Tiempo entrenamiento | evidencia HU009/checkpoint | reporte | IMPLEMENTADO |
| CA14 | >=10 partidas | evaluación compacta N>=10 | ejecución real | IMPLEMENTADO |
| CA15 | Score y estadísticas | `EvaluationSummary` | ejecución real | IMPLEMENTADO |
| CA16 | Reward entrenamiento | TensorBoard reward + media | event files reales | IMPLEMENTADO |
| CA17 | Reward explotación | `plot_exploitation_rewards` | rewards AV06 | IMPLEMENTADO |
| CA18 | Baseline aleatorio | `data/baseline_random_assault.json` | comparación real | IMPLEMENTADO |
| CA19 | Comportamiento aprendido | video + sección reporte | revisión evidencia | IMPLEMENTADO |
| CA20 | Conclusión | sección conclusión condicionada a evidencia | revisión final | IMPLEMENTADO |
| CA21 | Completitud | notebook/modelo/video/reporte | gate final | IMPLEMENTADO |
| CA22 | Organización | lógica en `src/`, notebook orquesta | suite + auditoría | IMPLEMENTADO |
| CA23 | Método del curso | DDQN explícito | tests/inspección | IMPLEMENTADO |
| CA24 | Desempeño efectivo | agente vs baseline + video | evaluación final | IMPLEMENTADO |
| CA25 | Reporte formal | 13 secciones obligatorias | auditoría notebook | IMPLEMENTADO |
| CA26 | Consistencia integral | SHA/model id/checkpoint | `ARTIFACT_LINEAGE` | IMPLEMENTADO |
| CA27 | >=2 métodos en Reto completo | flag global explícito | consolidación otros problemas | PENDING GLOBAL |

## Gate obligatorio

HU011 solo se cierra cuando la ejecución final produce:

```text
HU011_FINAL_DELIVERY_GATE=PASS
ENTREGABLE_ASSAULT_LISTO=True
```

y adicionalmente pasan:

```bash
python -m compileall -q 2_Assault/src
python -m pytest 2_Assault/tests -q
git diff --check
```

El requisito global CA27 puede permanecer `PENDING` durante el cierre específico de Assault, pero no puede convertirse en PASS hasta comprobar al menos dos métodos permitidos en el Reto 1 completo.
