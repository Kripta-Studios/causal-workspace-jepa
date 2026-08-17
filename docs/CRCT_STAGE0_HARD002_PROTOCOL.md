# CRCT-STAGE0-HARD-002 — frozen-discovery causal circuit benchmark

Status before primary execution: `PREREGISTERED_SYNTHETIC_FALSIFICATION`.

## Scientific question

Can Causal-Residual Circuit Tracing recover a planted finite nonlinear mechanism when the candidate
set contains realistic failure modes that made CRCT-STAGE0-001 too easy: causally active but
intervention-irrelevant decoys, action-sensitive cancelling pairs, redundant paths, true
cancellation, QK-like bilinear routing, OOD interventions, and coordinate gauges?

A pass is synthetic method validation only. A failure is a retained negative result. No threshold is
changed after opening the primary seeds.

## Phase isolation

1. Generate train and validation only.
2. Compute the strong differential baseline and validation residual.
3. Discover a signed residual-reconstruction circuit on validation only.
4. Construct matched random controls using validation activation RMS and finite-effect energy.
5. Freeze and SHA-256 hash the complete selected set, ranking, control sets, and validation scores.
6. Only then generate IID and OOD confirmation samples.
7. Evaluate the frozen circuit and frozen controls without re-selection.

Primary HARD-002 seeds are `1009, 2027, 4093`. They are separate from the CRCT-001 seeds and are
not used by the patch-development smoke tests. The process preflight uses seed `20260817` and is not
a scientific acceptance run.

## Planted hard cases

- **True nonlinear node routes:** tanh/cubic, sinusoidal/cubic, redundant, and imbalanced cancelling
  pathways whose higher-order effect is not fully captured by Taylor-2.
- **QK-like routing edges:** a bilinear query-key gate multiplies a value channel before an outgoing
  read; three edges are planted targets.
- **State-only active decoys:** they change the model output under direct manipulation but have zero
  action-mediated delta, distinguishing generic causal activity from mediation of the intervention.
- **Action-sensitive null pairs:** each member has a non-zero finite patch effect while each pair
  cancels exactly in the natural computation.
- **High-variance nuisance:** large activations with zero outgoing contribution.
- **Inactive coordinates inside mechanism families:** similar local statistics without the action
  pathway.

## Discovery and confirmation metrics

The primary selector is greedy **signed residual reconstruction**, not magnitude ranking. At each
validation step it adds the candidate that most reduces residual reconstruction MSE and stops when
its incremental recovery falls below the frozen threshold. Candidate truth labels are evaluation
metadata and are not consulted by the selector.

Confirmation records:

- node precision/recall;
- QK-like edge precision/recall;
- circuit sufficiency/recovery;
- complement/necessity and completeness diagnostics;
- decoy rejection;
- IID and OOD confirmation;
- frozen matched-control distribution and plus-one empirical p-value;
- function-preserving gauge invariance;
- all selected candidates, rankings, controls, hashes, and dataset hashes.

## Frozen gates

A seed is `HARD_VALIDATED` only if all conditions hold:

- confirmation was generated after the plan freeze;
- residual power after Taylor-2 is at least `0.08` of finite-effect energy;
- IID circuit recovery is at least `0.75`;
- OOD circuit recovery is at least `0.55`;
- node precision >= `0.70` and recall >= `0.60`;
- at least two of the three planted QK-like edges are recovered with edge precision >= `2/3`;
- frozen matched-control plus-one p <= `0.05`;
- selected recovery exceeds the frozen-control p95 by >= `0.15`;
- gauge function error <= `1e-5` and exact causal rank Spearman is effectively `1.0`;
- at least `75%` of decoys are rejected.

The aggregate is `HARD_VALIDATED` only if all three primary seeds pass. Any scientific gate failure
produces `NEGATIVE_RESULT`; crashes/timeouts/missing metrics produce `INFRASTRUCTURE_FAILURE`.

## Deliberately non-gating comparators

### HVP Screen-Flag-Fix

First-order attribution is screened against finite effects; unreliable candidates are flagged and a
second-order directional correction is recorded. HVP/T2 is **not required to improve the ranking**.
If it worsens, exact finite patching remains the confirmation standard. This mirrors the motivation
of Zhang & Wang, *When Attribution Patching Lies: Diagnosis and a Second-Order Correction*
(arXiv:2606.09899, 2026) without claiming their benchmark was reproduced.

### Equal-capacity learners

A residual student predicts only `finite - Taylor2`; a direct-delta student of exactly the same
architecture and parameter count predicts the full effect. IID and OOD NMSE are recorded. The
residual model is not required to win: an OOD loss is scientifically informative and cannot be
rescued by changing the gate.

### QK-like edge recovery

The routing case is motivated by the mechanistic need to explain query-key interactions rather than
only residual-stream nodes, following Anthropic's 2025 QK attribution work. HARD-002 is a synthetic
analogue only, not an implementation-equivalent reproduction.

## Relation to the August 2026 frontier

The next real-model milestone, conditional on HARD-002 surviving, should compare multiple sparse
substrates rather than assume one representation is canonical: Qwen-Scope SAE directions
(arXiv:2605.11887), cross-layer-transcoder/attribution-graph style features, and Sparse Weight
Decomposition (arXiv:2608.03913). Circuit claims must be validated on the original model with
necessity, sufficiency, faithfulness, and behavioral endpoints. Workspace/J-space remains a
secondary hypothesis requiring the stronger report/control/broadcast/selective-necessity criteria;
a low-dimensional or decodable subspace is insufficient.

## Resource boundary

HARD-002 is synthetic and may use the local RTX 5070 Ti. It does not execute protected Qwen or
EB-JEPA experiments, does not access their protected test outcomes, and does not authorize any
real-model circuit claim.
