# LLM-QWEN-BINDING-ALGEBRA-004 protocol

Status: `PREREGISTERED_NOT_RUN`.
`execution_authorized: false`.

This is a **new** prospective mechanistic phase. It does not mutate or rescue
`LLM-QWEN-BINDING-ALGEBRA-003` / V3 (`INELIGIBLE_TASK_PHASE0`).

Parent competence: `QWEN-BINDING-COMPETENCE-CONFIRM-001` passed on split
`confirmation` with frozen renderer `qwen_chat_prefill_v1`. That pass authorizes
**drafting this protocol only**. It does not authorize B1, CRCT, protected
splits, or a circuit claim.

## Question

On a **fresh** `mechanism_dev` split, with the already-frozen chat-prefill
renderer and never-used tokens, does Qwen3-0.6B remain full-vocabulary competent
(clean and direct-permuted `>= 0.90`) so that a later coalition-aware residual
CRCT phase can be preregistered?

Phase B0 here is competence + replay integrity on `mechanism_dev` only.
Phase B1/CRCT is **not** authorized by this document.

## Freeze

- Renderer: `qwen_chat_prefill_v1` only. No prompt search.
- Model: `Qwen/Qwen3-0.6B` revision `c1899de289a04d12100db370d81485cdf75e47ca`.
- Split: `mechanism_dev`, seed `901`, 32 episodes.
- Keys: cedar, slate, ember, linen.
- Values: navy, olive, pearl, moss.
- Tokenizer contract: strict spaced single-token; IDs frozen in
  `configs/experiments/qwen_binding_algebra_v4.json` after tokenizer-only
  verification on 2026-08-27. No substitutions after a model forward.
- Tokens are disjoint from V3 pools and from confirmation pools.
- Forbidden forwards: train, validation, test, paraphrase, calibration,
  confirmation.
- Candidate-only accuracy is diagnostic only.
- Direct-delta remains a mandatory baseline for any later residual claim
  (`ijepa_target_policy.py`). HARD-002 stays `NEGATIVE_RESULT`.

## Access

There is no `test` or `paraphrase` split in this protocol. Do not inherit V3's
protected splits. Do not reuse confirmation rows for circuit fitting.

## Failure

If B0 fails, status is `INELIGIBLE_TASK_PHASE0` for **004 only**. V3 remains
ineligible. Do not retune 0.90. Do not search prompts.

## Command (not to be run until a later commit sets execution_authorized)

No runner is authorized yet. A future commit must add an explicit runner and
set `execution_authorized: true` after this protocol is on origin.
