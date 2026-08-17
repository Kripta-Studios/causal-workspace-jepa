# CRCT → Qwen Bridge V1 protocol

Status: `PREREGISTERED_IMPLEMENTED_NOT_EXECUTED`.

Experiment: `CRCT-QWEN-BRIDGE-001`. Registered 2026-08-17.

## Purpose

This milestone closes synthetic CRCT Stage-0 without tuning HARD-002 to pass and creates a
fail-closed bridge to real Qwen computation. It has three deliberately separated components:

1. **Circuit ontology v3, read-only.** Re-read the already opened HARD-002 bundle and separate
   literal graph recovery from epsilon-functional sufficiency. This never changes HARD-002's
   registered `NEGATIVE_RESULT`; necessity, redundancy-group coverage, cancellation-group
   coverage, and circuit-equivalence remain `NOT_MEASURED_PROSPECTIVELY` unless a future protocol
   measures them before outcome access.
2. **Capital-patch CRCT development audit.** Reuse only the already disclosed capital-patch HDF5
   shard. Compare exact JVP and quadratic transport by endpoint and compare equal-family centered
   SVD-ridge direct-delta versus quadratic-residual prediction. The test split is already opened;
   every result is development evidence only and cannot become fresh confirmation. No new Qwen
   forward is performed in this component.
3. **Binding-algebra Phase-0.** Execute only B0 competence/replay and, conditional on B0, B1 exact
   directional JVP/HVP nonlinearity on `calibration/train/validation`. `test` and `paraphrase` are
   not materialized. B2/B3/B4 predictors are not trained.

## Non-negotiable access boundary

The authoritative parent remains `LLM-QWEN-BINDING-ALGEBRA-002` and its existing causal-residual
extension. This bridge does not edit either preregistration. The bridge itself is the new, scoped
execution-authorization milestone requested on 2026-08-17: it authorizes only B0/B1 on
`calibration/train/validation`; the parents' `execution_authorized=false` fields remain byte-for-byte
unchanged and confer no inherited authorization. Before any Qwen forward it requires:

- the frozen base commit to be an ancestor of the execution commit;
- exact Git blob identities for the parent config, causal-residual config, and pure protocol;
- no tracked worktree/index changes;
- the exact execution commit to be visible in an `origin/*` remote-tracking ref;
- the bridge/config/evaluator/tests to be committed exactly at that commit.

There is intentionally no CLI argument that can authorize `test` or `paraphrase`. Protected
execution requires a later, separately reviewed authorization after Phase-0 and any Phase-1/B2--B4
freeze.

## Substrate-readiness audit

The bundle also records which mechanistic substrates are actually executable on the frozen
Qwen3-0.6B target. Native residual/module states are the reference substrate and the pre-`o_proj`
head slices are readiness-only. Exact post-RoPE/GQA QK routing, the published HVP
Screen-Flag-Fix reliability procedure, Qwen-Scope SAEs, cross-layer transcoders, and Sparse Weight
Decomposition remain deferred until an exact compatible artifact/implementation and native-model
validation path are frozen. Missing sparse artifacts are never replaced by ad-hoc learned features.

## Phase B0: competence and execution integrity

B0 checks the pinned local Qwen3-0.6B tokenizer/model, single-token value contract, full-vocabulary
clean competence, full-vocabulary direct-permuted competence, exact layer-0 replay, observed
layer-21 full-state downstream replay, and exact linear decomposition of the layer-21 attention
`o_proj` input into head slices. The head decomposition is only a readiness check. It does **not**
reconstruct post-RoPE/GQA QK scores and therefore permits no QK-routing mechanism claim.

If any B0 gate fails, status is `INELIGIBLE_TASK_PHASE0` and B1 is not executed. Candidate-only
accuracy is stored as diagnosis but cannot substitute for the preregistered full-vocabulary gate.

## Phase B1: exact differential/nonlinearity audit

Only after B0 passes, validation query-changing held-out actions are evaluated from one clean
origin. The endpoint concatenates final-query residual states from blocks 14/18/21/24/27 and the
four role-reindexed candidate logits. Exact forward-mode JVP and nested JVP/HVP produce T1/T2; no finite-difference fallback is
allowed. Direct finite composition is compared with the sum of exact local JVPs for the primitive
transposition chords from the same clean origin. Finite primitive targets on validation are never
executed because the parent causal-residual preregistration restricts those targets to train.

The frozen parent eligibility boundary is conjunctive: composition interaction power >= 0.10 and
quadratic NMSE >= 0.10. Failure yields `COMPLETED_NEGATIVE_LOCALLY_DIFFERENTIAL`; a pass yields
`PHASE0_B1_ELIGIBLE_FOR_LATER_B2`, **not** a circuit/mechanism result and not authorization for B2
or protected evaluation. Derivative unavailability is reported separately and is never silently
replaced by finite differences.

## Logs and bundle

The suite records Git/source hashes, effective config, environment, CUDA/driver information,
resource guard, tests, HARD-002 ontology audit, capital development audit, Phase-0 plan, every
allowed-split access event, Phase-0 stdout/stderr, aggregate status, and a SHA-256 manifest. A ZIP
is created on success, scientific negative, availability block, derivative block, or unexpected
infrastructure failure.

## Evidence boundary

No outcome from this milestone establishes a native Qwen circuit, J-space/global workspace, JEPA
workspace, cross-model mechanism, novelty, or SOTA result. Capital is development-only. Phase-0 is
an eligibility/nonlinearity audit. The next scientific decision is made only after independent
review of the frozen ZIP.
