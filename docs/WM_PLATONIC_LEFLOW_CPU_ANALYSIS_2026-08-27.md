# CPU Track A scientific analysis — 2026-08-27

This document interprets already-adjudicated confirmation artifacts. It does
**not** retune gates, overwrite frozen metrics, open stitching, or execute
paper-scale Platonic/LeFlow work.

| ID | Frozen status | Evidence |
|---|---|---|
| `WM-PLATONIC-MKNN-001` | `TRANSITION_NEIGHBORHOOD_ALIGNMENT_PASSED` | Availability |
| `WM-PLATONIC-MKNN-001-UNTRAINED-POSTHOC` | `POSTHOC_DIAGNOSTIC` | None; not a gate |
| `WM-LEFLOW-AMORTIZE-001` | `NEGATIVE_RESULT` | None |

Parent protocol freeze: `7392ab5`. T1 retained run: `01f93ab`. T2 run:
`f1ac460`. Untrained-predictor diagnostic code: `e29aa41`. Untrained
artifact provenance commit: `e29aa41` (clean tree).

Confirmation splits only. Downloads: none. `test` / `paraphrase`: closed.

## T1 — WM-PLATONIC-MKNN-001

Question: if two ridge TinyJEPAs learn the same PointMass2D
action-conditioned dynamics from the same trajectories, through different
frozen linear maps `R^4 → R^16`, do predicted-transition neighborhoods
align under m-kNN?

TinyJEPA has no early/middle/late MLP. Named points are
`encoder.latent`, `predictor.input`, and `predictor.latent`. The encoder is
the frozen identity, so `encoder.latent = predictor.input`. The only
distinct geometries are encoder/input versus one-step predictor output.

k=5, n_eval=128, chance = 5/127 ≈ 0.03937, frozen floor = 2×chance ≈ 0.07874.

### Frozen confirmation metrics

| seed | encoder A,B | predictor A,B | Δ (pred−enc) | shuffle | random-map | probe A,B | open-loop MSE B |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 131 | 0.890625 | 0.879688 | −0.0109 | 0.696875 | 0.048438 | 0.889063 | 2.48e-8 |
| 137 | 0.912500 | 0.915625 | +0.0031 | 0.803125 | 0.015625 | 0.928125 | 1.38e16 |
| 139 | 0.926563 | 0.921875 | −0.0047 | 0.737500 | 0.043750 | 0.915625 | 2.46e-12 |

Random-map encoder m-kNN is near chance (0.039 / 0.044 / 0.044). Observation
map SHA-256 values are identical across seeds. Paired confirmation actions
are identical for A and B.

The frozen gate is a conjunction: predictor A,B must beat shuffle,
random-map, and 2×chance. All three seeds pass. That is the registered
outcome. It is **not** the same claim as “transition learning increased
alignment beyond encoder geometry.”

### Post-hoc untrained-predictor diagnostic

Not a gate. Same confirmation rows, frozen maps, identity encoder; predictor
weights resampled from `N(0, 1/sqrt(fan_in))` with seeds `confirmation+10000`
and `+10001`. Artifact:
`artifacts/metrics/wm_platonic_mknn_v1.untrained_posthoc.json`.

| seed | trained A,B (sanity) | untrained A,B | trained A vs untrained A |
|---:|---:|---:|---:|
| 131 | 0.879688 | 0.698438 | 0.293750 |
| 137 | 0.915625 | 0.539063 | 0.310938 |
| 139 | 0.921875 | 0.689063 | 0.434375 |

Trained A,B in the post-hoc file matches the frozen parent to printed
precision. Untrained cross-model overlap is well above chance and below
trained A,B. Trained versus untrained on map A is far from 1, so the
random weights are not a copy of the fitted predictor.

### Q1. Did transition learning increase cross-model alignment beyond encoder, random, and shuffle?

Frozen confirmatory comparisons only (parent JSON):

**Relative to the frozen random-map control: yes.** Random-map predictor
m-kNN is ~0.02–0.05. Shared physical states are required for the high
scores. Random-map is a shared-index **noise** control, not an untrained
weight control.

**Relative to encoder/input geometry: no material increase.** Predictor−encoder
Δ is −0.011 / +0.003 / −0.005. Encoder A,B is already 0.89–0.93, above the
chance floor in every seed (`encoder_geometry_already_above_chance_floor:
true`). The predictor mostly **preserves** neighborhoods that the two frozen
linear maps of the same 4-d state already share.

**Relative to shuffled-action: the conjunction passes, but shuffle is
not a near-chance null.** Shuffle predictor m-kNN is 0.70–0.80. Action
identity is not the main source of neighborhood overlap.

Post-hoc only (does not change the parent gate):

**Relative to untrained predictors: trained A,B is higher than untrained
A,B.** Random linear predictors scramble some of the encoder-shared
geometry; fitting restores it toward encoder levels. That does not show a
new shared coordinate system created by dynamics learning.

### Q2. Layer-wise localization, competence association, pre-training presence

- **Monotonic across layers:** not testable. There is one linear predictor,
  not a deep stack. Do not invent intermediate layers.
- **Localized:** alignment is already present at `encoder.latent` /
  `predictor.input`. `predictor.latent` is statistically the same band.
- **Associated with transition competence:** n=3. Seed 137 has the
  **highest** predictor A,B (0.916) and an **exploded** open-loop multi-step
  latent MSE (~1.38e16). Seeds 131 and 139 have near-zero open-loop MSE
  and slightly lower or similar overlap. Open-loop MSE is an unroll from
  t=0, not one-step residual. There is no evidence that better unroll
  competence produces more m-kNN.
- **Already present before training:** yes, as encoder geometry of the
  frozen maps. Untrained predictors still sit well above chance (0.54–0.70).

### Q3. Does this justify stitching?

**No.** Stitching would test whether a **learned adapter** can port a
planner across observation maps. T1 shows that two identity-encoded linear
views of the same PointMass state already share k-NN structure, that
fitting a linear predictor does not add a distinct shared geometry, and
that shuffle remains high. That is the wrong substrate on which to spend a
stitching experiment: a positive stitch would be unsurprising and would
not license Platonic-WM claims. `WM-PLATONIC-STITCH-001` stays closed.

### Q4. T1 × CRCT — what is still missing for equivalent action→Δ-state circuits?

T1 is Availability of neighborhood overlap. Missing for a CRCT-style
equivalence claim:

- interventional ablations of candidate circuits, not geometry scores;
- necessity, sufficiency, redundancy, and cancellation controls;
- matched non-causal controls at the same capacity;
- a real (non-tautological) gauge;
- functional metrics (effect recovery), not k-NN overlap;
- HARD-002 remains `NEGATIVE_RESULT` on a different, harder planted
  substrate and is not relabeled by T1.

## T2 — WM-LEFLOW-AMORTIZE-001

Registered outcome: **`NEGATIVE_RESULT`**. Primary gate: H=5 latent-flow
N=64 versus random shooting N=64; success slack **and** amortized
wall-clock strictly less than shooting. Success saturated at 1.0; clock
clause failed on every seed.

This is a latent interpolator + ridge inverse dynamics + frozen-WM rerank.
It is **not** LeFlow.

Frozen JSON does **not** store final goal distance or peak memory.
Failure rate is `1 − success`. Those omissions are reported, not filled
with post-hoc numbers.

### Comparison table (confirmation, all success = 1.0, failure = 0.0)

WM forwards/replan = `mean_candidates_evaluated` for arms that rollout the
frozen WM (shooting, CEM, latent-flow). Action-flow records
`candidates_evaluated = 1` but does **not** WM-rerank (`predicted_cost` is
NaN in source, not stored in the frozen JSON); WM **rerank** forwards = 0.
Action-flow still calls `model.encode` twice per replan (start and goal);
that is not a search rollout. Inverse-dynamics forwards for latent-flow are
**structural** `N × H` from `amortized_latent.py`, not a measured JSON
field. Action-flow is one ridge map to a full action sequence (planner
forwards = 1, ID = 0). Shooting/CEM: planner network forwards = 0, ID = 0.

**H=5 (primary)**

| Planner | Success | Fail | WM fwd/replan | Planner fwd | ID fwd | wall-clock 151 | 157 | 163 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| random shooting N=64 | 1.0 | 0 | 64 | 0 | 0 | 1.22e-4 | 1.58e-4 | 1.26e-4 |
| CEM 16×4 | 1.0 | 0 | 64 | 0 | 0 | 3.96e-4 | 4.59e-4 | 3.88e-4 |
| latent-flow N=1 | 1.0 | 0 | 1 | 1 | 5 | 1.16e-4 | 1.36e-4 | 1.91e-4 |
| latent-flow N=64 (primary) | 1.0 | 0 | 64 | 64 | 320 | 2.60e-4 | 2.85e-4 | 3.79e-4 |
| action-flow N=1 | 1.0 | 0 | 0 | 1 | 0 | 1.93e-5 | 1.52e-5 | 2.98e-5 |

Mean H=5 wall-clock (s): shooting 1.35e-4; CEM 4.14e-4; latent N=1 1.48e-4;
latent N=64 3.08e-4; action-flow 2.15e-5.

**H=10 (diagnostic, not OOD)**

| Planner | Success | Fail | WM fwd | ID fwd | wall-clock 151 | 157 | 163 |
|---|---:|---:|---:|---:|---:|---:|---:|
| random shooting N=64 | 1.0 | 0 | 64 | 0 | 1.43e-4 | 1.68e-4 | 2.74e-4 |
| CEM 16×4 | 1.0 | 0 | 64 | 0 | 5.03e-4 | 5.85e-4 | 6.24e-4 |
| latent-flow N=1 | 1.0 | 0 | 1 | 10 | 1.80e-4 | 2.00e-4 | 2.16e-4 |
| latent-flow N=64 | 1.0 | 0 | 64 | 640 | 5.14e-4 | 5.44e-4 | 5.01e-4 |
| action-flow N=1 | 1.0 | 0 | 0 | 0 | 2.08e-5 | 1.55e-5 | 1.48e-5 |

Mean H=10 wall-clock (s): shooting 1.95e-4; CEM 5.71e-4; latent N=1 1.99e-4;
latent N=64 5.20e-4; action-flow 1.70e-5.

World-model fingerprints matched before/after planner comparison within each
seed (151: `5480688b…`; 157: `0a56bd22…`; 163: `7f35a4af…`).

Protocol caveat, recorded not patched: shooting/CEM receive 2-d goal
position; latent-flow interpolates using full 4-d `state[H]`.

### Inverse dynamics (holdout MSE, not a planning gate)

| seed | no Δz | with Δz |
|---:|---:|---:|
| 151 | 2.13e-10 | 1.18e-11 |
| 157 | 2.31e-10 | 1.17e-11 |
| 163 | 2.26e-8 | 4.18e-11 |

Δz is slightly better. Both arms are numerically near-zero on this linear
ridge substrate. That is an inductive-bias diagnostic, not workspace
evidence, and it does not change the primary clock failure.

### Q5. How much iterative online search did amortization replace?

**On the primary arm: none that reduced cost.** N=64 latent-flow still
evaluates 64 frozen-WM rollouts per replan, the same budget as shooting,
plus 320 inverse-dynamics decodes at H=5. The “amortized” candidate is a
perturbed interpolator that still **reranks** with the world model.

N=1 replaces 64 WM rollouts with 1, but is not the primary arm and is not
uniformly faster than shooting (seed 163 is slower). Action-flow replaces
WM search entirely (0 WM forwards) and is fastest; it is a direct action
regressor, not the gated method.

### Q6. Success-versus-compute Pareto relative to CEM

Every arm is at success 1.0. The Pareto front on **success** is a single
point. On **compute**, action-flow is cheapest, then shooting, then latent
N=1 (mixed vs shooting), then latent N=64, then CEM. CEM is the slowest
matched-64-rollout method because of Python iteration overhead, not
because it plans worse. Do not report a speedup without a success
degradation: there is no success degradation to report, and the primary
amortized arm is **slower** than shooting.

### Q7. Did WM reranking materially improve the amortized candidate?

N=1 and N=64 both have success 1.0. Reranking cannot improve success on a
ceiling. It **increases** wall-clock (mean 1.48e-4 → 3.08e-4 at H=5) and
WM forwards (1 → 64). No material quality gain is observable.

### Q8. Did explicit Δz improve inverse dynamics?

Slightly, on holdout MSE, at magnitudes that are already machine-small.
No planning-success contrast exists (ceiling). Do not promote Δz to a
latent-as-planner claim.

### Q9. How severe was H=5 → H=10 degradation?

**Success: none** (still 1.0). Wall-clock rises, especially CEM and latent
N=64. The protocol’s preregistered collapse warning did not trigger. That
is in-distribution PointMass holdout, not OOD robustness.

### Q10. What is the next justified experiment?

**Not** stitching, **not** Rectified Flow, **not** stochastic/generative
amortization on this substrate, **not** CRCT cross-model mechanism
comparison, **not** Qwen 004, **not** IBD-002.

This PointMass setup cannot fail planners on success. The scientifically
honest next step, if this track continues, is a **new prospective**
planning-competence protocol on a substrate where random shooting is
**not** at ceiling (harder PointMass, obstacles, or MiniPush), with the
same frozen-WM matching rules, clocks, and N∈{1,64}, H∈{5,10} recorded
**before** outcomes. Until that protocol exists, stop. Do not execute it
in this session.

## Unchanged frozen inventory

| ID | Status |
|---|---|
| `CRCT-STAGE0-HARD-002` | `NEGATIVE_RESULT` |
| `LLM-QWEN-BINDING-ALGEBRA-003` / V3 | `INELIGIBLE_TASK_PHASE0` |
| `QWEN-BINDING-COMPETENCE-CONFIRM-001` | `COMPETENCE_CONFIRMATION_PASSED` |
| `LLM-QWEN-BINDING-ALGEBRA-004` | `PREREGISTERED_NOT_RUN`, `execution_authorized: false` |
| `CRCT-QWEN-BRIDGE-003` | `PREREGISTERED_NOT_RUN` |
| `CRCT-COALITION-IBD-001` confirmation | smoke; not real-gauge confirmation |
| `CRCT-COALITION-IBD-002` | `PREREGISTERED_NOT_RUN` |
| `WM-PLATONIC-STITCH-001` | not opened |
