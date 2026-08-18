# Qwen Binding Competence Recovery V1

Experiment: `QWEN-BINDING-COMPETENCE-RECOVERY-001`

This is a **development-only calibration experiment**. It is not Phase-0 B0
confirmation and cannot rescue the registered V3 negative result.

## Why this exists

V3 passed token, treatment, replay, and output-decomposition integrity checks
but failed behavioural competence. A plausible confound is prompt format:
Qwen3 is normally used through its tokenizer chat template, whereas V3 used a
plain completion prompt.

The only legitimate next question is therefore:

> Can a prompt format selected entirely on the already-opened calibration
> split make the frozen 0.6B model perform the same lookup task competently?

## Hard access boundary

Model forwards are allowed on `calibration` only.

The runner must not generate or materialize:

- train episodes;
- validation episodes;
- test episodes;
- paraphrase episodes.

A ledger records every forward scope. Any non-calibration model-forward entry
is an infrastructure/protocol failure.

## Frozen prompt roster

1. `legacy_v3_control` — exact V3 plain prompt.
2. `explicit_plain_v1` — explicit single-value lookup instruction in a plain
   completion prompt.
3. `qwen_chat_prefill_v1` — Qwen tokenizer chat template, non-thinking mode,
   with a partial assistant message `Answer:` so the next token remains the
   same strict spaced single-token endpoint.
4. `qwen_chat_prefill_fewshot_v1` — the same chat/prefill contract plus one
   fixed synthetic demonstration. The demonstration is part of the frozen
   renderer, not selected from model outputs.

Unavailable renderer APIs are recorded as unavailable; they are never replaced
by an unregistered prompt.

## Frozen selection rule

For each variant compute on calibration:

- clean full-vocabulary accuracy;
- directly permuted full-vocabulary accuracy;
- candidate-only accuracy (diagnostic only);
- expected-answer logit margin against the best wrong table candidate.

A variant is eligible iff both full-vocabulary accuracies are at least `0.90`.

Among eligible variants select, in order:

1. highest `min(clean, direct)` full-vocabulary accuracy;
2. highest mean of clean/direct full-vocabulary accuracy;
3. lowest preregistered priority integer.

No threshold changes are allowed after observing results.

## Outcomes

- `COMPETENCE_RECOVERY_PROMPT_SELECTED`
- `COMPETENCE_RECOVERY_FAILED`
- `AVAILABILITY_BLOCKED`
- `PROTOCOL_GUARD_BLOCKED`
- `INFRASTRUCTURE_FAILURE`

Even `COMPETENCE_RECOVERY_PROMPT_SELECTED` does **not** authorize validation,
B1, B2, test, or paraphrase. Upload and independently adjudicate the ZIP first.
