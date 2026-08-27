# QWEN-BINDING-COMPETENCE-RECOVERY-001 adjudication

Registered outcome: **`COMPETENCE_RECOVERY_PROMPT_SELECTED`** (calibration-only development).

This is **not** confirmation. It **does not** rescue `LLM-QWEN-BINDING-ALGEBRA-003` / V3, which remains **`INELIGIBLE_TASK_PHASE0`**.

## Artifact verified 2026-08-27

Local run directory (gitignored reports tree):

`artifacts/reports/qwen_competence_recovery/QWEN-BINDING-COMPETENCE-RECOVERY-001_20260818_101736/`

Verified fields:

- suite status `COMPETENCE_RECOVERY_PROMPT_SELECTED`
- selected renderer `qwen_chat_prefill_v1`
- clean full-vocabulary accuracy `0.9375` (16 calibration rows)
- direct-permuted full-vocabulary accuracy `0.9791666666666666` (48 rows)
- candidate-only metrics diagnostic only
- `model_forward_splits_executed`: `["calibration"]`
- `protected_splits_executed`: `[]`
- `test_executed` / `train_executed` / `validation_executed`: false
- ACCESS_LEDGER.jsonl records only calibration tokenizer/model forwards
- model `Qwen/Qwen3-0.6B` revision `c1899de289a04d12100db370d81485cdf75e47ca`
- runtime torch `2.10.0+cu128`, CUDA

## Frozen renderer

`qwen_chat_prefill_v1`: Qwen chat template, non-thinking, assistant prefill `Answer:`.

Other roster members were ineligible on calibration:

| variant | clean full | direct full |
|---|---:|---:|
| legacy_v3_control | 0.0000 | 0.0000 |
| explicit_plain_v1 | 0.1875 | 0.3542 |
| qwen_chat_prefill_fewshot_v1 | 0.0000 | 0.0000 |
| qwen_chat_prefill_v1 | 0.9375 | 0.9792 |

No threshold was changed after observing these numbers. The 0.90 floor is the original competence gate.

## What this authorizes

Only: freeze this renderer and write a **new** prospective confirmation protocol with fresh unused examples.

It does not authorize validation, test, paraphrase, B1, CRCT on Qwen, or any V3 status change.
