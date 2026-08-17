# Qwen Binding Algebra V3 tokenizer-only amendment protocol

## Purpose

V3 repairs only the lexical eligibility defect discovered in V2 before the first competence
forward.  It is **not** a threshold amendment and does not rescue or reinterpret a model result.

The source experiments remain frozen:

- `LLM-QWEN-BINDING-ALGEBRA-002`
- `QWEN-BINDING-ALGEBRA-CR-V1`

The amended descendants are generated deterministically as:

- `LLM-QWEN-BINDING-ALGEBRA-003`
- `QWEN-BINDING-ALGEBRA-CR-V2`

## Two-stage freeze

### Stage A — tokenizer-only resolution

Run `scripts/prepare_qwen_binding_algebra_v3_token_amendment.py` while the exact pinned tokenizer
is available locally.  The preparer imports no causal language model class and performs no model
forward.  It first reproduces the exact 23-source-failure roster in the committed amendment spec.
If even one failure differs, resolution aborts.

For each failed slot, resolution selects the first unused item in a committed ordered semantic
roster that satisfies all of:

- calibration: lowercase ASCII color word;
- train/validation/test: ASCII TitleCase city name;
- globally casefold-unique against every original key/value and every earlier replacement;
- `tokenizer(" " + value, add_special_tokens=False)` returns exactly one id;
- decoding that id returns exactly `" " + value`.

Every already-valid value is preserved.  The preparer writes:

- `configs/experiments/qwen_binding_algebra_v3.yaml`
- `configs/experiments/qwen_binding_algebra_cr_v2.yaml`
- `configs/experiments/qwen_binding_algebra_v3_token_contract.json`

The contract records the tokenizer-vocabulary hash, every replacement, every resolved token id,
source/config hashes, generated-config hashes, and a self hash.

### Stage B — commit/push before Qwen

The three generated artifacts, amendment code, Bridge-002 code, tests, and protocol documents must
be committed and pushed.  Bridge-002 fails closed unless the exact worktree is clean and the HEAD
is present under `origin/*`.

Only after that boundary may `qwen_binding_algebra_phase0_v2.py` load Qwen weights.

## Pre-model token contract

Phase0-v2 loads the tokenizer first, verifies the frozen contract and the generated config hashes,
recomputes the tokenizer-vocabulary hash, verifies every resolved value/token id, and performs the
allowed-split treatment-token audit.  A failure returns `TOKEN_CONTRACT_BLOCKED_PRE_MODEL`; model
weights are not loaded.

Protected **prompts/episodes** are not materialized by Phase0-v2.  The frozen protected value strings
may appear in the tokenizer-only contract because lexical eligibility is configuration metadata,
not a model outcome.

## Forward telemetry

The access ledger distinguishes:

- `SPLIT_MATERIALIZED_NO_FORWARD`
- `TOKEN_CONTRACT_VERIFIED_PRE_MODEL`
- `MODEL_LOAD_STARTED` / `MODEL_LOAD_COMPLETE`
- `B0_MODEL_FORWARD_EXECUTION_STARTED`
- `B0_MODEL_FORWARD_EXECUTION_COMPLETE`

A split is recorded as model-forward-completed only after B0 returns successfully.  Declared scope
is not reported as executed work.

## Scientific invariants

The amendment does not change:

- Qwen model or revision;
- split counts or seeds;
- key pools;
- S4 permutation convention or action partition;
- prompt templates;
- treatment and capture sites;
- B0 competence/replay thresholds;
- B1 interaction/quadratic thresholds;
- derivative policy (exact JVP/HVP only; no finite-difference rescue);
- B2/B3/B4 authorization;
- test/paraphrase protection.

If B0 passes, B1 is interpreted under exactly the same numerical gates as Bridge-001.  If B0 fails,
V3 is ineligible.  If B1 fails, the locally-differential negative is preserved.  No threshold is
retuned from the new outcome.
