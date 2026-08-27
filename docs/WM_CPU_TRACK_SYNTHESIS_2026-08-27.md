# CPU world-model track synthesis — 2026-08-27

Status: **consolidation**. This document does not retune gates, does not
rerun experiments, and is not new scientific evidence.

No model forwards, downloads, or newly opened splits were performed to
write it. Facts below are taken from already-committed artifacts and
adjudications.

| ID | Recorded status | Evidence |
|---|---|---|
| `WM-PLATONIC-MKNN-001` | `TRANSITION_NEIGHBORHOOD_ALIGNMENT_PASSED` (frozen) | Availability |
| `WM-PLATONIC-MKNN-001-UNTRAINED-POSTHOC` | `POSTHOC_DIAGNOSTIC` (frozen) | None |
| `WM-LEFLOW-AMORTIZE-001` | `NEGATIVE_RESULT` (frozen) | None |
| `WM-AMORTIZED-PLANNING-MINIPUSH-002` | `UNINFORMATIVE_SUBSTRATE` (frozen) | None |
| `WM-AMORTIZED-PLANNING-REACHABLE-003` | `DRAFT_NOT_PREREGISTERED` (not a freeze) | not an experiment |

`WM-PLATONIC-STITCH-001` remains not opened. HARD-002 remains
`NEGATIVE_RESULT`. V3 remains ineligible. Qwen 004 remains unauthorized.

---

## OBSERVED

### T1 — geometry on two frozen linear views of PointMass

Encoder A,B m-kNN: 0.891 / 0.913 / 0.927. Trained predictor A,B: 0.880 /
0.916 / 0.922. Predictor−encoder Δ ≈ 0. Shuffle: 0.697 / 0.803 / 0.738.
Random-map: 0.048 / 0.016 / 0.044 (near chance 0.039). Chance floor 0.079.

Post-hoc untrained A,B (not a gate): 0.698 / 0.539 / 0.689.

Seed **137** open-loop multi-step latent MSE exploded (`≈1.38e16`) while
one-step predictor m-kNN stayed high. That unroll is not the T1 gate and
is not evidence of multi-step dynamics competence.

### T2 — PointMass amortized planning

Every planner, including random shooting N=64, success 1.0 at H=5 and
H=10. Primary latent N=64 slower than shooting. Gate failed on the clock
clause. Not LeFlow.

### MiniPush-002 — contact planning

Qualification passed (shooting 0.0, not a ceiling; WM full-state RMSE
low). Confirmation: CEM success **equals** shooting on every seed (0 /
0.042 / 0.083). Almost all failures `horizon_insufficient`. Status
`UNINFORMATIVE_SUBSTRATE`, not an amortizer pass or fail.

---

## INFERENCE

### Q1 — strongest T1 claim

Separate three layers:

**A. Already present (encoder).** Two frozen Gaussian maps of the **same**
4-d PointMass state, passed through identity encoders, already share
k-NN structure at 0.89–0.93. That is expected of two full-rank linear
views of one low-dimensional manifold. It is not a discovery of shared
latent physics.

**B. Preserved/recovered by fitting.** Trained predictor m-kNN stays in
the same band as the encoder (Δ ≈ 0). Post-hoc untrained predictors sit
lower (0.54–0.70) but still above chance: random linear maps scramble
some inherited neighborhoods; ridge fitting restores them toward encoder
levels. That is preservation, not creation of a new coordinate system.

**C. Attributable to correct action-conditioned dynamics.** The frozen
gate is passed because predictor A,B beats shuffle, random-map, and
2×chance. Random-map shows that **shared physical state** is required
(noise maps sit at chance). Shuffle remains **high** (0.70–0.80), so
action identity is not the main source of overlap. Probe-action scores
are close to shuffled-probe scores. Therefore the confirmatory pass does
**not** isolate correct action-conditioned dynamics as the cause of
alignment.

**Strongest supportable claim:** under this tiny linear setup, two
identity-encoded linear observations of the same PointMass trajectories
have highly overlapping neighborhoods; a fitted linear predictor
preserves that overlap above a noise-map null. It also beats shuffled-
action (margins ≈ 0.18 / 0.11 / 0.18), but shuffle itself remains high,
so action identity is not isolated as the cause of alignment. Evidence
level remains Availability. One-step neighborhood overlap is not
multi-step competence (seed 137 unroll).

**Not established:** stitching, shared causal circuits, Platonic physics,
workspace, architecture-universal convergence. Stitching would test
adapter-mediated portability. T1’s encoder already aligns, so a positive
stitch would be unsurprising and uninformative. Circuits require
intervention, necessity, sufficiency, redundancy, cancellation, and
specificity — none of which m-kNN supplies. HARD-002 stays negative and
is a different substrate.

### Q2 — two planning failure modes

T2 is an **easy substrate**: shooting ≈ CEM ≈ every amortizer ≈ perfect.
Quality cannot discriminate. The registered fail is compute (N=64 rerank
not faster than shooting) on a saturated success axis.

MiniPush-002 is an **uninformative hard substrate**: shooting ≈ CEM ≈
every amortizer ≈ poor, with the same success on each seed. Quality
still cannot discriminate, now because almost no query is reachable in
H. Search usefulness was never established (`s_cem − s_shoot = 0`).

Neither answers: *can amortization reproduce a real search advantage more
cheaply?* T2 has no advantage to reproduce (everyone already succeeds).
MiniPush-002 has no advantage to reproduce (search never beats shooting).
The MiniPush-002 result must not be relabeled as `NEGATIVE_RESULT` of
the amortizer, must not rescue T2, and must not justify Rectified Flow.

Low full-state RMSE on MiniPush-002 did not imply a planning-competent
world model: static object coordinates dominate one-step error when
contact is rare.

### Q3 — missing regime (Goldilocks)

Needed, symbolically, **before** asking about amortization:

```text
Success_shooting  <  Success_CEM
Success_CEM       well above floor
Success_shooting  well below ceiling
WM error          on the task-relevant dynamics, not static coordinates
Goals             reachable within H by an independent witness
```

Then, and only then:

```text
Can amortized N=1 approach Success_CEM with fewer online WM rollouts?
```

Numerical thresholds for that successor are **not** taken from these
confirmation outcomes. They belong in a future freeze, drafted separately
as `DRAFT_NOT_PREREGISTERED`.

---

## NOT ESTABLISHED

- Platonic shared coordinates or F=ma recovery
- Stitching justification
- LeFlow / Rectified Flow / elimination of search or replanning
- Workspace
- Cross-model CRCT equivalence
- Rescue of T2, V3, or HARD-002
- OOD robustness from held-out episodes

---

## NEXT HYPOTHESIS (prospective only)

If a MiniPush (or similar contact) query set is built from **witnessed**
windows `state_t → state_{t+H}` that contain object motion, with witness
actions sealed from planners, then CEM may beat shooting on development.
If a frozen search-usefulness gate and a contact-stratified WM gate both
pass, confirmation could test deterministic amortization versus CEM.

That hypothesis is **not** preregistered here. Draft:
`docs/research/WM_AMORTIZED_PLANNING_REACHABLE_003_DRAFT.md`.

Status of that draft: `DRAFT_NOT_PREREGISTERED`,
`execution_authorized: false`. **STOP** until an explicit authorization
commit freezes a protocol.

### What would justify deterministic amortization

On a qualified substrate: N=1 (or a frozen primary arm) non-inferior to
CEM on success within a pre-outcome slack **and** strictly cheaper in
frozen online compute (WM rollouts and/or wall-clock), with shooting
still worse than CEM.

### What would justify generative / Rectified Flow

All four, jointly, after a frozen informative regime:

1. CEM > shooting on the frozen search-usefulness rule.
2. Deterministic amortization materially worse than CEM (outside slack).
3. Failures are not explained by WM incompetence, unreachable goals,
   inverse-dynamics collapse, short horizon, or unequal goal information.
4. Documented multimodal feasible paths to the same goal, such that a
   mean interpolator would average between modes.

None of that is observed now.

### What would justify Platonic stitching

Two task-competent observation maps of the same dynamics that do **not**
already share ~0.9 encoder m-kNN; a frozen test that transition learning
increases alignment beyond those maps and beyond shuffle/untrained
nulls; only then adapter stitching as a separate ID. T1 is the wrong
substrate.

### What would justify cross-model CRCT

Not `z_A ≈ z_B`. Interventional evidence of epsilon-sufficient,
substitutable action→Δ-state mechanisms: necessity, sufficiency,
redundancy, cancellation, specificity, equivalence classes. HARD-002
remains `NEGATIVE_RESULT` and is not reinterpreted as a template pass.
