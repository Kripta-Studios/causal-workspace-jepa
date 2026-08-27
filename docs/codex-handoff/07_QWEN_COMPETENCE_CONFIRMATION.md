# Qwen Competence Confirmation Protocol Guidance

This file is guidance for writing a new prospective protocol. It is **not itself a preregistration**.

## Facts to preserve

- V3 B0 is frozen as `INELIGIBLE_TASK_PHASE0`.
- A later calibration-only development run reportedly selected
  `qwen_chat_prefill_v1` with clean `0.9375` and direct-permuted `0.9792`.
- That development result must be verified from its artifact before formal adjudication.

## Confirmation objective

Test one question only:

> Does the already-selected frozen renderer make the frozen Qwen model competently solve the same
> binding lookup behavior on genuinely fresh, never-used examples?

This is a competence confirmation, not yet a circuit claim.

## Freeze before execution

Record and commit:
- experiment ID;
- parent development artifact hash;
- renderer ID and exact renderer source hash;
- model ID and exact revision;
- tokenizer revision/hash;
- answer-token contract;
- generator code hash;
- fresh split seed;
- number of examples;
- clean/direct-permuted treatment construction;
- threshold `0.90`;
- access policy;
- output schema;
- stop/failure rules.

## Data boundary

Do not reuse calibration/train/validation examples that influenced renderer selection or protocol
design. Create a new named confirmation split.

Do not open `test` or `paraphrase`.

## Metrics

Primary:
- clean full-vocabulary accuracy;
- direct-permuted full-vocabulary accuracy.

Diagnostics:
- candidate-only accuracy;
- expected-answer logit margin;
- tokenization failure counts.

Eligibility requires both primary accuracies >= `0.90`.

## After outcome

- pass: permits drafting a **new** prospective mechanistic phase;
- fail: close the selected prompt/task path negative;
- infrastructure failure: fix infrastructure and rerun only if the protocol allows an exact rerun
  without outcome-dependent changes.

Never search additional prompts on the confirmation split.
