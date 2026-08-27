## Veredicto

El repositorio es ya una plataforma de investigación sustancial, no un scaffold vacío. Tiene instrumentación real de Qwen, intervenciones reproducibles,
controles sintéticos, auditoría causal y varios resultados negativos útiles. Pero todavía no satisface sus criterios de completitud: no hay workspace
aceptado, circuito Qwen validado ni ventaja demostrada del Intervention-JEPA sobre aproximaciones más simples.

De los dos papers, “Scaling Inherently Interpretable Language Models” es el que más puede hacer avanzar la investigación ahora, sobre todo el Track B y la
definición de circuitos bajo redundancia. LeVLJEPA tiene valor secundario para Track A, como encoder/control de representaciones densas, pero no resuelve
el cuello de botella causal actual.

## Estado actual del repositorio

La rama es crct-stage0-001, en 3f58ec5, sincronizada con origin/crct-stage0-001 y siete commits por delante de main.

No hay modificaciones tracked, pero sí doce entradas sin seguimiento:

- ocho parches de adjudicación/bridge;
- dos JSON de auditoría;
- un directorio de métricas;
- una especificación de despliegue.

Esto importa porque dos JSON locales son recogidos por la auditoría de reproducibilidad sin tener el esquema ni los sidecars de provenance exigidos.

La base implementada incluye:

- datasets Tier-0 deterministas;
- JEPA pequeño y reproducción/adaptadores de modelos publicados;
- hooks, intervenciones y activation stores;
- probes, sparse dictionaries, Jacobian lenses y circuit graphs;
- planificación;
- mock transformer y Qwen HF real;
- datasets reales de intervenciones Qwen;
- baselines y Intervention-JEPA.

La situación general se resume bien como: infraestructura amplia y operativa, evidencia científica predominantemente negativa o de nivel 1–3, sin mecanismo
de workspace aceptado. Véase el README.md:5.

### Verificación ejecutada

La suite rápida produjo:

- 284 passed
- 1 failed
- 20 warnings

El fallo está en test\_reproducibility\_audit.py: las dos métricas locales sin seguimiento carecen de experiment\_id, status o evidence\_level, además de sus
archivos .provenance.json.

Los tests focalizados recientes sí pasan:

- Qwen bridge/competence: 21 passed
- CRCT científico: 6 passed

Ruff no está limpio: encuentra cinco problemas —cuatro imports no usados y una variable no usada— en código tracked.

Por tanto, el estado correcto es:

- rama remota sincronizada;
- subsistemas recientes cubiertos por tests focalizados;
- suite completa y auditoría de reproducibilidad actualmente bloqueadas;
- lint con cinco errores;
- documentación principal retrasada respecto a la última ejecución del 18 de agosto.

## Últimos commits

 Commit     Función
━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
 3f58ec5    Corrige la sintaxis del entrypoint de competence recovery
─────────  ─────────────────────────────────────────────────────────────────────────
 fe153b0    Prerregistra la recuperación de competencia Qwen limitada a calibración
─────────  ─────────────────────────────────────────────────────────────────────────
 638ccfc    Hace compatibles los configs V3 con el loader del repositorio
─────────  ─────────────────────────────────────────────────────────────────────────
 accb68d    Prerregistra Qwen Binding V3 con contrato de tokenización revisado
─────────  ─────────────────────────────────────────────────────────────────────────
 d6e50b8    Corrige lectura HDF5 y telemetría Phase-0
─────────  ─────────────────────────────────────────────────────────────────────────
 e834f0e    Cierra CRCT Stage-0 y autoriza el bridge Qwen
─────────  ─────────────────────────────────────────────────────────────────────────
 b70442b    Añade la auditoría adversarial de circuitos causal-residuales

Son milestones coherentes, pero los documentos globales —README, RESULTS, ROADMAP, TODO— no incorporan plenamente la competence recovery posterior. Eso
incumple temporalmente el criterio de sincronización documental del propio repositorio.

## Resultados científicos recientes

### CRCT Stage-0

El control sintético básico recupera perfectamente los componentes plantados y obtiene errores de replay muy bajos. Sirve como positive control, aunque la
especificidad aleatoria original estaba contaminada por selección sobre evaluación.

### CRCT-HARD-002

El resultado científico congelado es negativo:

- potencia residual posterior a T2: 0.0547, 0.1290, 0.0420;
- fidelidad funcional IID/OOD superior a 0.992;
- recall de nodos: 0.8, 0.4, 0.6;
- recuperación de aristas QK: perfecta;
- especificidad emparejada: p = 1/257.

Dos seeds no superaron la potencia residual preregistrada y otro no alcanzó el recall de nodos. Más importante: un circuito funcional mínimo puede
reproducir más del 99 % del efecto y aun omitir nodos plantados redundantes o cancelatorios.

Eso falsifica una identificación ingenua entre:

> máxima fidelidad funcional = recuperación del circuito verdadero.

Además, el MLP que predice directamente el delta superó al modelo residual en todos los seeds IID y OOD. No hay justificación empírica para privilegiar
todavía la descomposición diferencial más residual aprendido. Véase la docs/CRCT\_STAGE0\_HARD002\_RESULT\_2026-08-17.md:1.

### Qwen Binding V3 Phase-0

La instrumentación de Qwen3-0.6B funciona correctamente:

- replay de capas 0 y 21: error cero;
- reconstrucción de o\_proj: error cero;
- error de descomposición por cabezas: 5.72e-06.

Pero el modelo no superó el gate de competencia de 0.90; las respuestas full-vocabulary fueron prácticamente cero. El resultado correcto es
INELIGIBLE\_TASK\_PHASE0, no un fracaso de CRCT. Véase la docs/QWEN\_BINDING\_ALGEBRA\_V3\_B0\_ADJUDICATION\_2026-08-18.md:1.

### Competence recovery

La ejecución más reciente encontró una plantilla qwen\_chat\_prefill\_v1 con:

- clean: 0.9375;
- direct-permuted: 0.9792.

Es una señal útil, pero sólo se abrió calibración. Train, validation, test y paraphrases permanecieron protegidos. Por ello es una selección de prompt de
desarrollo, no confirmación independiente ni rescate retroactivo de V3. Véase el artifacts/reports/qwen\_competence\_recovery/QWEN-BINDING-COMPETENCE-
RECOVERY-001\_20260818\_101736/summary.md:1.

## Paper 2607.00784 — LeVLJEPA

LeVLJEPA ([https://arxiv.org/abs/2607.00784](https://arxiv.org/abs/2607.00784)) aprende representaciones visión-lenguaje mediante predicción cruzada, stop-gradient y regularización SIGReg,
sin negativos ni momentum teacher. Hay código oficial ([https://github.com/MLO-lab/LeVLJEPA](https://github.com/MLO-lab/LeVLJEPA)) y un checkpoint ViT-B
([https://huggingface.co/lukaskuhn/LeVLJEPA-ViT-B-DataComp-200k](https://huggingface.co/lukaskuhn/LeVLJEPA-ViT-B-DataComp-200k)).

Su hallazgo interesante para este proyecto es que obtiene mejores representaciones densas y segmentación, aunque peores resultados zero-shot globales que
algunos objetivos contrastivos. Eso sugiere que el objetivo puede conservar información espacial/local valiosa para MiniPush, C-JEPA y análisis de
interacciones entre objetos.

Pero su encaje central es limitado:

- no es action-conditioned;
- no modela dinámica temporal;
- no estudia planificación;
- no hace intervención mecanística ni reconstrucción de circuitos;
- su entrenamiento principal usa decenas de millones de pares, fuera de la escala razonable del repositorio;
- aplicar el encoder preentrenado a imágenes sintéticas de 32×32 introduce un fuerte cambio de dominio y confusión por reescalado.

Uso recomendado: un experimento factorial pequeño en MiniPush:

1\. encoder actual frente a LeVLJEPA congelado;
2\. con y sin SIGReg;
3\. mismo predictor action-conditioned;
4\. medir rank efectivo, localización de objetos, probes, patching causal, especificidad y efecto en planificación.

Esto permitiría distinguir si mejora sólo la disponibilidad de información o también su uso causal. No lo convertiría ahora en la prioridad principal.

## Paper 2608.07594 — Steerling

“Scaling Inherently Interpretable Language Models” ([https://arxiv.org/abs/2608.07594](https://arxiv.org/abs/2608.07594)) introduce Steerling, un modelo de 8.4B con representaciones separadas
en componentes conocidos, desconocidos y residuales. Publica código ([https://github.com/guidelabs/steerling](https://github.com/guidelabs/steerling)), modelo
([https://huggingface.co/guidelabs/steerling-8b](https://huggingface.co/guidelabs/steerling-8b)) y documentación del proyecto
([https://www.guidelabs.ai/papers/scaling-inherently-interpretable-language-models/](https://www.guidelabs.ai/papers/scaling-inherently-interpretable-language-models/)).

Es mucho más relevante porque ataca directamente problemas que HARD-002 acaba de revelar:

- circuitos con componentes redundantes o cancelatorios;
- diferencia entre representación semántica, residuo y efecto funcional;
- intervención sobre conceptos;
- soporte de entrenamiento para que una intervención sea válida;
- contribución explícita de componentes a logits.

Pero debe incorporarse críticamente. En evaluación, si el residual se define como:

[
\epsilon = h-k-u,
]

entonces (k+u+\epsilon=h) por construcción. La reconstrucción exacta no demuestra que k y u sean el mecanismo causal correcto. Además, sumar contribuciones
absolutas puede ocultar cancelaciones. El paper muestra steering real, pero no ofrece todavía los controles de necesidad, suficiencia, especificidad y
generalización que exige este repositorio.

La ejecución completa de Steerling también queda fuera del perfil local actual: el release indica alrededor de 18 GB de VRAM y desaconseja 16 GB o menos.

### Cómo sí aprovecharlo ahora

Priorizaría cuatro incorporaciones resource-independent:

1\. Circuitos por grupos o coaliciones. Extender HARD-002 con nodos conocidos, desconocidos, residuales, redundantes y cancelatorios. Evaluar equivalencia
   funcional y conjuntos mínimos, no sólo recuperación exacta de nodos.

2\. Positive control interpretable-by-design. Crear un transformer diminuto con concept bottleneck plantado. CRCT y el meta-modelo deberían recuperar grupos
   causales y detectar cuándo múltiples circuitos son funcionalmente equivalentes.

3\. Descomposición del target del Intervention-JEPA. Predecir por separado efecto conocido, efecto desconocido y residual, pero con presupuestos residuales,
   controles de leakage y replay directo sobre Qwen.

4\. Validez de intervención. Registrar si operación, magnitud, sitio y combinación estaban dentro del soporte de entrenamiento. El propio paper muestra que
   el steering fue pobre hasta que se entrenó explícitamente para ello.

## Prioridad propuesta

1\. Cerrar el gate de competencia Qwen con un split confirmatorio intacto y la plantilla seleccionada congelada.
2\. Corregir hygiene, provenance, Ruff y sincronización documental.
3\. Adoptar de Steerling la evaluación por coaliciones y el concept-bottleneck sintético.
4\. Repetir HARD-002 con métricas de equivalencia funcional, minimalidad, necesidad y controles de cancelación.
5\. Sólo después ejecutar el factorial pequeño LeVLJEPA/SIGReg en MiniPush.
6\. Dejar la ejecución de Steerling-8B como BLOCKED\_RESOURCE hasta disponer de hardware adecuado.

En suma: Steerling puede cambiar de forma material cómo el repositorio define y audita un circuito; LeVLJEPA puede mejorar una representación visual, pero
no aborda aún la principal incertidumbre causal. No realicé cambios en el repositorio.