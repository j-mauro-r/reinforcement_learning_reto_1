# HU011 — Evidencia de implementación

## Estado

**[IMPLEMENTADA — VALIDACIÓN LOCAL/CI Y COLAB FINAL PENDIENTES]**

La implementación existe en `feature/hu011-entregable-final-assault`. Este estado **no equivale a HU cerrada**: la HU solo puede pasar a `[COMPLETADA]` después de ejecutar todas las autovalidaciones y obtener `HU011_FINAL_DELIVERY_GATE=PASS` en Google Colab.

## Componentes implementados

- `2_Assault/assault_ddqn.ipynb`: notebook final/orquestador y reporte.
- `2_Assault/src/hu011_delivery.py`: baseline, evaluación auditable, figura de explotación, comparación y gate CA01–CA26.
- `2_Assault/src/evaluator.py`: seeds explícitas por episodio para la evaluación formal.
- `2_Assault/src/model_artifact.py`: conserva evidencia de entrenamiento del checkpoint dentro de metadata compacta, sin Replay Buffer/optimizer.
- `2_Assault/data/baseline_random_assault.json`: baseline real HU001 versionado.
- `2_Assault/data/full_training_summary_assault.json`: snapshot liviano de evidencia HU009.
- `2_Assault/docs/hu011_matriz_cumplimiento.md`: mapa criterio → evidencia → validación.
- `2_Assault/tests/test_hu011_delivery.py`.
- `2_Assault/tests/test_evaluator_hu011.py`.
- `2_Assault/tests/test_notebook_hu011.py`.
- regresiones HU009C del notebook ajustadas al entregable final.
- `requirements.txt`: `matplotlib` declarado explícitamente.

## Mejoras respecto de HU009C

1. La evaluación formal se ejecuta desde el modelo compacto cargado desde disco.
2. Se fuerzan al menos 10 episodios, `epsilon=0.0` y seeds únicas explícitas.
3. Se persiste `final_compact_evaluation.json` con protocolo, rewards, lengths, estadísticas, model SHA y Git SHA.
4. Se agrega la gráfica obligatoria de recompensa durante explotación.
5. El baseline HU001 se reutiliza programáticamente y se compara con el agente.
6. El gate CA01–CA26 impide declarar listo el entregable Assault si un requisito obligatorio falla.
7. CA27 mantiene explícito el requisito global de >=2 algoritmos sin convertirlo artificialmente en PASS.

## Validaciones todavía obligatorias

```bash
python -m compileall -q 2_Assault/src
python -m pytest 2_Assault/tests -q
git diff --check
```

Después, en Google Colab sobre el HEAD del PR:

```text
Run All
HU011_FINAL_DELIVERY_GATE=PASS
ENTREGABLE_ASSAULT_LISTO=True
```

Hasta observar estas evidencias, HU011 permanece implementada pero no cerrada.
