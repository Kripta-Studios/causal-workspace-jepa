# Qwen Binding Algebra V3 — B0 adjudication (2026-08-18)

Registered outcome: **`INELIGIBLE_TASK_PHASE0`**.

This disposition is frozen for `LLM-QWEN-BINDING-ALGEBRA-003` /
`QWEN-BINDING-ALGEBRA-CR-V2`. It must not be upgraded by later prompt work.

## What executed

`CRCT-QWEN-BRIDGE-002_20260818_013731` reached real model execution on the
allowed `calibration`, `train`, and `validation` splits. Protected `test` and
`paraphrase` remained unopened. B1 did not execute because B0 failed.

The integrity controls passed:

- strict answer-token contract: pass;
- treatment-token audit: pass, zero failures across 1,584 cases;
- exact layer-0 replay: max endpoint error `0.0`;
- exact observed layer-21 downstream replay: max endpoint error `0.0`;
- linear `o_proj` reconstruction: `0.0`;
- summed head-output decomposition error: `5.7220458984375e-06`.

The competence gates failed decisively:

| split | clean full-vocab | clean candidate-only | direct-permuted full-vocab | direct candidate-only |
|---|---:|---:|---:|---:|
| calibration | 0.0000 | 0.2500 | 0.0000 | 0.4375 |
| train | 0.0000 | 0.421875 | 0.0000 | 0.401042 |
| validation | 0.0000 | 0.3750 | 0.001042 | 0.384375 |

The preregistered competence floor was 0.90 full-vocabulary accuracy on every
allowed split. Therefore the binding task, under the V3 prompt rendering, is
not an eligible substrate for B1/CRCT analysis.

## Interpretation boundary

This is **not** evidence that causal-residual analysis fails in Qwen. It is
evidence that the V3 behavioural task/prompt is incompetent for this model
under the frozen endpoint.

The replay and decomposition controls being exact make an implementation
failure of the intervention stack unlikely as the explanation for B0.

## Prospective continuation

Prompt competence recovery is development work and may use only the already
opened calibration split. It must not inspect train, validation, test, or
paraphrase under alternative prompt renderers.

A fixed prompt roster and fixed selection rule are preregistered in
`configs/experiments/qwen_competence_recovery_v1.json`. If no candidate reaches
the original 0.90 clean and direct-permuted full-vocabulary floors on
calibration, the recovery attempt closes negative. If one does, its exact
renderer and hashes must be frozen in a later milestone before a **new**
confirmation split is created or executed.
