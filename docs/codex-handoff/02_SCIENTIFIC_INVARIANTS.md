# Scientific Invariants

These rules are non-negotiable.

## Historical results are immutable

- `CRCT-STAGE0-HARD-002` remains `NEGATIVE_RESULT`.
- Qwen Binding V3 remains `INELIGIBLE_TASK_PHASE0`.
- Calibration-only competence recovery cannot retroactively rescue V3.
- Never lower a threshold after observing the result it governs.
- Never rename a negative result into a positive milestone by changing ontology after the fact.

## Data-access integrity

- Do not touch `test` or `paraphrase` unless a new prospective protocol explicitly authorizes it.
- Prompt selection may use only the already-opened calibration data described by its protocol.
- A prompt chosen on calibration must be frozen before a fresh confirmation split is generated.
- Confirmation examples must not have been used to select renderer, thresholds, layer, metric,
  architecture, seed, or stopping rule.
- Maintain a machine-readable forward-access ledger for Qwen model forwards.

## Provenance integrity

- Do not fabricate provenance fields to satisfy an audit.
- Commit preregistration/config/source-hash material before outcome access where required.
- Preserve exact model/tokenizer revisions and source hashes.
- Keep raw logs and aggregate adjudication separate.
- A missing artifact is not permission to reconstruct a favorable result from memory.

## Baseline integrity

For learned intervention-effect prediction, always retain strong baselines:
- JVP/first order;
- T2/quadratic;
- relinearized JVP when applicable;
- direct-delta capacity-matched predictor;
- differential + learned residual.

A residual model is not privileged by architecture. It must earn eligibility empirically.

## Circuit-claim integrity

Report separately:
- graph recovery;
- functional sufficiency;
- minimality;
- necessity;
- redundancy/cancellation coverage;
- specificity;
- faithfulness;
- held-out generalization;
- equivalence classes.

A high replay score does not prove recovery of the unique “true graph” when multiple circuits are
functionally substitutable.

## Workspace claim boundary

Do not reopen a workspace claim until a real model has a validated mechanism with prospective
necessity, sufficiency, specificity, faithfulness, and held-out generalization.
