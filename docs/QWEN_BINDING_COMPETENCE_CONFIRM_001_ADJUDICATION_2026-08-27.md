# QWEN-BINDING-COMPETENCE-CONFIRM-001 adjudication

Registered outcome: **`COMPETENCE_CONFIRMATION_PASSED`**.

This does **not** rescue `LLM-QWEN-BINDING-ALGEBRA-003` / V3, which remains
**`INELIGIBLE_TASK_PHASE0`**. It is competence evidence (Availability) for the
frozen renderer on a fresh confirmation split. It is not a circuit, CRCT,
workspace, or B1 result.

## Protocol identity

- Config: `configs/experiments/qwen_competence_confirm_v1.json`
- Renderer: `qwen_chat_prefill_v1` (frozen before this run)
- Model: Qwen3-0.6B revision `c1899de289a04d12100db370d81485cdf75e47ca`
- Split: `confirmation`, seed `701`, 32 episodes
- Keys: maple, quartz, ridge, frost
- Values: teal, ivory, coral, peach
- Forbidden forwards: train, validation, test, paraphrase, calibration

Preregistration commit: `1b1a3fc` (and ancestors). Execution used a later clean
worktree; thresholds were not changed.

## Outcome

| treatment | n | full-vocab accuracy | candidate-only (diagnostic) |
|---|---:|---:|---:|
| clean | 32 | **1.0000** | 1.0000 |
| direct-permuted | 96 | **0.9896** | 1.0000 |

Both primary gates remain `>= 0.90`. One direct-permuted row emitted ` frost`
(a table key) instead of a value; that is a residual error, not a reason to
retune.

ACCESS_LEDGER records only `confirmation` stages. `protected_splits_executed`
is empty.

## What this authorizes

Drafting a **new** prospective mechanistic protocol
(`LLM-QWEN-BINDING-ALGEBRA-004` / `CRCT-QWEN-BRIDGE-003`) that still must not
open V3's test/paraphrase and must not rewrite V3's ineligible status.

It does not authorize treating Qwen as a validated circuit substrate yet.
