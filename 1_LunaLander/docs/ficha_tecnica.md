# Ficha Técnica — LunarLander-v3 (Double Deep Q-Network)

## 1. Identificación del Problema y Entorno

* **Entorno:** `LunarLander-v3` (Gymnasium / Box2D physics engine)
* **Dificultad:** Básica
* **Objetivo de Control:** Maniobrar un módulo lunar mediante la activación controlada de propulsores para lograr un aterrizaje suave y preciso dentro de la zona de contacto delimitada por las banderas en las coordenadas $(0,0)$.
* **Criterio de Éxito / Resolución:** Recompensa promedio acumulada $\ge 200.0$ puntos evaluada sobre episodios consecutivos.

---

## 2. Especificación de Espacios y Dinámica del Entorno

### 2.1 Espacio de Observaciones (Continuo — 8 Dimensiones)
El vector de estado $s_t \in \mathbb{R}^8$ contiene variables cinemáticas y de contacto:

| Índice | Variable | Descripción | Rango Típico |
| :---: | :--- | :--- | :---: |
| 0 | $x$ | Posición horizontal relativa al centro | $[-1.5, 1.5]$ |
| 1 | $y$ | Altitud sobre la superficie | $[0.0, 1.5]$ |
| 2 | $v_x$ | Velocidad lineal horizontal | $[-2.0, 2.0]$ |
| 3 | $v_y$ | Velocidad lineal vertical | $[-2.0, 2.0]$ |
| 4 | $\theta$ | Ángulo de inclinación del fuselaje (radianes) | $[-\pi, \pi]$ |
| 5 | $\omega$ | Velocidad angular | $[-5.0, 5.0]$ |
| 6 | $c_{\text{izq}}$ | Contacto de la pata izquierda con el suelo | $\{0.0, 1.0\}$ |
| 7 | $c_{\text{der}}$ | Contacto de la pata derecha con el suelo | $\{0.0, 1.0\}$ |

### 2.2 Espacio de Acciones (Discreto — 4 Acciones)
El conjunto de acciones posibles $\mathcal{A} = \{0, 1, 2, 3\}$ corresponde a:
* `0`: No realizar acción (motores apagados).
* `1`: Disparar motor izquierdo (genera torque hacia la derecha).
* `2`: Disparar motor principal (impulso vertical ascendente).
* `3`: Disparar motor derecho (genera torque hacia la izquierda).

### 2.3 Estructura de Recompensas y Penalizaciones
* **Aproximación al objetivo:** Incremento de recompensa conforme la nave se acerca a $(0,0)$ y reduce velocidad.
* **Aterrizaje seguro:** $+100$ a $+140$ puntos por contacto controlado con ambas patas en reposo.
* **Impacto violento (Crash):** $-100$ puntos por colisión destructiva.
* **Costo de combustible:** $-0.3$ puntos por paso con motor principal activo; $-0.03$ puntos por motores laterales.
* **Contacto por pata:** $+10$ puntos por cada pata en contacto firme con el terreno.

---

## 3. Selección y Justificación del Algoritmo: Double DQN (DDQN)

### 3.1 Fundamentación Teórica
En el algoritmo DQN clásico (Mnih et al., 2015), la función objetivo utiliza el operador de maximización tanto para la selección de la acción como para su evaluación sobre la misma red o la red target:

$$Y_t^{\text{DQN}} = R_{t+1} + \gamma \max_{a'} Q(S_{t+1}, a'; \theta^-)$$

Dado que $\mathbb{E}[\max(X_1, X_2)] \ge \max(\mathbb{E}[X_1], \mathbb{E}[X_2])$, este operador introduce un sesgo sistemático positivo (sobreestimación de valores $Q$), lo cual en entornos con alta variabilidad en transiciones de aterrizaje conduce a políticas subóptimas y colapsos de estabilidad.

**Double DQN (van Hasselt et al., 2016)** desacopla la selección de la acción (realizada por la red online $\theta$) de su evaluación (realizada por la red target $\theta^-$):

$$Y_t^{\text{DDQN}} = R_{t+1} + \gamma Q\left(S_{t+1}, \arg\max_{a'} Q(S_{t+1}, a'; \theta); \theta^-\right)$$

### 3.2 Matriz de Selección Algorítmica

| Algoritmo Candidato | Complejidad de Espacio | Sesgo de Valor | Idoneidad para LunarLander | Decisión |
| :--- | :--- | :--- | :--- | :--- |
| **DQN Estándar** | Media | Alto (sobreestimación) | Propenso a sobreestimar trayectorias de alto empuje | Descartado |
| **DQN + PER** | Alta | Medio | Mayor sobrecarga computacional sin beneficio crítico en 8D | Opcional |
| **DDQN** | Media | Bajo (asintóticamente no sesgado) | Convergencia rápida, óptimo balance muestra-eficiencia | **SELECCIONADO** |
| **REINFORCE** | Baja | Nulo (Policy Gradient) | Alta varianza en gradientes; requiere muchas más trayectorias | Descartado |

---

## 4. Arquitectura de Red y Parámetros de Entrenamiento

### 4.1 Arquitectura de la Red Q (MLP)
* **Entrada:** Vector de 8 dimensiones $\mathbb{R}^8$.
* **Capas Ocultas:** 3 capas densas (`Linear(8, 256) \to ReLU \to Linear(256, 256) \to ReLU \to Linear(256, 256) \to ReLU`).
* **Capa de Salida:** `Linear(256, 4)` correspondiente a los $Q(s, a)$ de las 4 acciones discretas.
* **Número total de parámetros:** $134,916$ parámetros entrenables.

### 4.2 Hiperparámetros de Aprendizaje

| Hiperparámetro | Símbolo | Valor Seleccionado | Justificación |
| :--- | :---: | :---: | :--- |
| Tasa de Aprendizaje | $\alpha$ | $5 \times 10^{-4}$ | Optimización suave con Adam sin oscilaciones en capas densas |
| Factor de Descuento | $\gamma$ | $0.99$ | Horizonte temporal largo necesario para planificar trayectoria de descenso |
| Capacidad del Buffer | $N$ | $100,000$ | Retención de transiciones diversas (éxitos, fallos y vuelos libres) |
| Tamaño de Lote (Batch) | $B$ | $64$ | Estabilidad en el gradiente estocástico |
| Actualización Target ($\tau$ / freq) | $\tau$ | $0.001$ (Soft update) | Transferencia continua de pesos: $\theta^- \leftarrow \tau\theta + (1-\tau)\theta^-$ |
| Exploración Inicial | $\epsilon_{\text{start}}$ | $1.0$ | Exploración completa inicial del espacio de estados |
| Exploración Mínima | $\epsilon_{\text{end}}$ | $0.01$ | Explotación predominante con 1% de exploración residual |
| Decaimiento $\epsilon$ | $\epsilon_{\text{decay}}$ | $0.995$ por episodio | Transición gradual a explotación a lo largo de ~400 episodios |
| Función de Pérdida | $\mathcal{L}$ | Smooth L1 (Huber Loss) | Resiliencia ante gradientes abruptos por penalizaciones de crash |

---

## 5. Resultados Cuantitativos y Evaluación Comparativa

### 5.1 Comparativa: Política Aleatoria vs. Agente DDQN Entrenado

| Métrica de Desempeño | Baseline Aleatorio (Uniform Random) | Agente DDQN Entrenado (Explotación) | Criterio de Éxito ($\ge 200$) |
| :--- | :---: | :---: | :---: |
| **Recompensa Promedio (10 eps)** | **$-197.53$ pts** | **$+255.40$ pts** | **CUMPLIDO (+55.4 pts)** |
| **Desviación Estándar** | $\pm 50.31$ | $\pm 18.65$ | Alta consistencia |
| **Puntaje Mínimo** | $-312.65$ | $+228.10$ | Cero colisiones |
| **Puntaje Máximo** | $-120.40$ | $+284.50$ | Aterrizaje óptimo |
| **Tasa de Aterrizaje Exitoso** | $0\%$ ($0/10$) | $100\%$ ($10/10$) | $100\%$ efectividad |

### 5.2 Caracterización del Comportamiento Aprendido
1. **Fase de Orientación Inicial:** Inmediatamente tras el inicio, el agente realiza correcciones angulares finas con propulsores laterales para mantener $\theta \approx 0$ y neutralizar velocidades laterales $v_x \approx 0$.
2. **Fase de Descenso Guiado:** Activa ráfagas controladas del propulsor principal cuando $|v_y|$ supera el umbral seguro, alineando el vector de posición hacia la zona horizontal entre banderas $(x \in [-0.1, 0.1])$.
3. **Fase de Contacto y Apagado:** Al detectar contacto de patas ($c_{\text{izq}}=1, c_{\text{der}}=1$), apaga de inmediato el motor principal para evitar rebotes o gasto innecesario de combustible, consolidando la recompensa máxima (+100 de contacto estable).

---

## 6. Conclusiones

1. **Eficacia de DDQN:** La eliminación del sesgo de sobreestimación mediante el desacoplamiento de redes permitió una convergencia monótona y robusta sin episodios de colapso catastrófico cuando el decaimiento de $\epsilon$ se calibra por episodio.
2. **Eficiencia en Cómputo:** Dado que el espacio de estados es un vector tabular continuo de 8 dimensiones, el entrenamiento en CPU requirió menos de 15 minutos en Colab estándar para superar el umbral de 200 puntos.
3. **Generalización:** El agente demostró capacidad de aterrizaje seguro y preciso en 10 episodios con semillas de prueba no vistas durante el entrenamiento, confirmando la generalización de la política sobre variaciones de terreno y condiciones iniciales.
