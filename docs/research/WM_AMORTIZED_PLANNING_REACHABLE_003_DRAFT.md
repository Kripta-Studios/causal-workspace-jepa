# WM-AMORTIZED-PLANNING-REACHABLE-003 — DRAFT ONLY

```text
STATUS:                 DRAFT_NOT_PREREGISTERED
EXECUTION_AUTHORIZED:   false
CONFIRMATION:           CLOSED
DEVELOPMENT:            NOT TO BE RUN UNDER THIS DOCUMENT
TRAINING:               NOT TO BE RUN UNDER THIS DOCUMENT
DOWNLOADS:              none
```

This file is **not** a protocol freeze. It must not be executed, must not
be copied into `configs/experiments/` as a runnable config, and must not
be treated as scientific evidence. A later authorization commit would
have to freeze a **new** protocol (possibly with different numbers) before
any split is opened.

Parent synthesis: `docs/WM_CPU_TRACK_SYNTHESIS_2026-08-27.md`.
Does not mutate T2 or MiniPush-002.

This is **not** LeFlow and **not** Rectified Flow.

---

## Defects this draft is meant to repair

MiniPush-002 sampled start-at-t0 versus a constructed “object already at
goal” state. Typical Manhattan approach exceeded H=5/10, so almost every
query was `horizon_insufficient`. Qualification only forbade a shooting
**ceiling** and a trivial **all-success** pattern, so an all-fail / CEM=
shooting substrate still opened confirmation.

T2 used in-distribution `state[H]` goals on PointMass and hit a quality
ceiling.

The missing regime is witnessed-reachable goals on a contact task, with
a development gate that **requires search to beat shooting**.

---

## Proposed scientific question (for a future freeze)

Given MiniPush windows that are **reachable in H by a sealed witness
action sequence**, can CEM beat random shooting on development, and if
so, can a deterministic latent interpolator approach that CEM success at
lower online cost?

---

## Proposed query construction (outcome-independent)

Generate MiniPush trajectories as in the existing generator (vector
state, not pixels; not the LeVLJEPA factorial).

For a frozen horizon H ∈ {5, 10}, extract **disjoint** windows:

```text
start = state_t
goal_observation = state_{t+H}     # full 6-d, same for every planner
witness = a_t, …, a_{t+H-1}
```

**Inclusion rule (environment event, not planner outcome):** keep the
window only if object xy **changes at least once** in `(t, t+H]`. That
selects contact-containing reachable segments without looking at CEM or
shooting success.

**Witness leakage protections (to be tested in a future implementation):**

- Planners receive only `start` and `goal_observation`.
- `witness` is hashed (SHA-256 of the action array) and stored for audit;
  the raw witness is not passed into planner functions.
- A future semantic test must fail if a planner kwargs dict contains
  `witness` or equals the witness action sequence by construction.
- Train / development / confirmation windows come from those trajectory
  splits only; no window crosses a split boundary.
- The witness is not a training target for the amortizer (no expert BC
  on confirmation windows). Action-flow, if retained, is trained only on
  **train** windows and still evaluated on constructed `goal_observation`
  without seeing the witness.

H=5 and H=10 populations are built separately with the same rule. They
are in-distribution holdouts of windowed episodes, **not** OOD worlds.

Do not use planner performance to drop or keep individual queries.

---

## Proposed identical goal information

All planners encode the same `goal_observation` with the same frozen
world-model encoder:

```text
z_goal = encoder(goal_observation)
cost   = || z_predicted_terminal − z_goal ||^2
```

No 2-d privileged slice. Success on the environment is a frozen object-
and/or full-state distance to `goal_observation`, identical across arms.

---

## Proposed world-model competence (development, before confirmation)

Do **not** gate on pooled 6-d RMSE. Static object coordinates can pass
that gate while contact is unmodeled.

Draft metrics, all on **development contact windows only**:

1. **Moving-object one-step RMSE:** RMSE of predicted vs true object xy
   restricted to transitions where the true object moved.
2. **Static-object one-step RMSE:** same, restricted to no-motion
   object transitions (should be near zero if the model learned “no
   contact ⇒ no object motion”).
3. **Contact event sign:** among true object-motion steps, fraction of
   predicted object deltas pointing into the same cardinal quadrant as
   the truth (or cosine > 0).
4. **H-step object unroll RMSE:** open-loop object xy error at horizon H
   on development windows.

Draft fail-closed rule (conventions for a future freeze, **not** taken
from MiniPush-002 confirmation rates):

- (1) < 2.0 pixels — one grid step is 1 pixel; twice that is a pre-outcome
  “still on the neighboring cell” tolerance for a linear CPU model.
- (2) < 0.5 pixels — no-contact object should not drift.
- (3) > 0.60 — better than chance on four quadrants (0.25) by a wide
  pre-outcome margin, without using confirmation.
- (4) recorded; collapse relative to (1) is diagnostic, not a retune
  trigger unless a freeze later promotes it.

If any of (1)–(3) fail: `WORLD_MODEL_INCOMPETENT`. Confirmation stays
closed. Do not enlarge the model after seeing the fail unless a finite
architecture ladder was frozen in advance (this draft specifies **no**
such ladder).

---

## Proposed search-usefulness qualification (development only)

Primary horizon for qualification: **H=5** (same role as MiniPush-002
primary H; H=10 remains diagnostic).

Let `I_cem,i`, `I_shoot,i` be per-query success indicators on the **same**
development windows. Define

```text
Δ_search = mean_i (I_cem,i − I_shoot,i)
```

Use a paired nonparametric bootstrap (B = 2000, frozen seed in a future
config) to obtain a 95% percentile interval for `Δ_search`.

Draft n: **48** development queries (and 48 confirmation queries in a
future freeze). Rationale: MiniPush-002 used 16/24, which gives a
binomial SE ≈ 0.1 near p=0.5 and cannot resolve a modest search effect.
n=48 is a pre-outcome power choice (SE ≈ 0.07 at p=0.5), not a fit to
observed MiniPush-002 confirmation rates.

Draft pass (all required):

1. Shooting not at ceiling: `s_shoot ≤ 0.80`.
   Rationale: leaves at least one-fifth of the query mass for search to
   improve; 0.80 is a design headroom convention, not MiniPush-002’s 0.90
   ceiling test inverted from confirmation.
2. CEM not at floor: `s_cem ≥ 0.25`.
   Rationale: 12/48 successes; below that, “search helps” is a handful of
   lucky windows.
3. Search effect: **lower** 95% bootstrap CI of `Δ_search` **> 0.125**.
   Rationale: 0.125 = 6/48 = 1 extra CEM success per 8 queries. That is
   larger than a single discordant pair (1/48 ≈ 0.021) and does not use
   MiniPush-002’s confirmation Δ = 0. It is a domain-meaningful minimum
   useful effect plus a CI so a point estimate of 0.13 with a CI crossing
   0 does not pass.

If any clause fails: `UNINFORMATIVE_SUBSTRATE`. Confirmation stays
closed. **STOP.** This draft does **not** include a difficulty ladder.
One qualification attempt after a future freeze; no adaptive H/N/spawn
search.

---

## Proposed planners (future implementation; not now)

Same frozen WM, starts, goals, H, action bounds, quantization:

1. Random shooting (frozen N, e.g. 64)
2. CEM (existing implementation; frozen population × iterations with
   **matched WM-rollout budget** to shooting, latent-goal cost)
3. Deterministic latent interpolator N=1
4. Latent interpolator N=64 + WM rerank
5. Direct action amortizer N=1 if it can be trained on **train windows
   only** without witness leakage; otherwise omit with a written reason
   in the freeze

No Rectified Flow in this ID.

Primary comparison, if confirmation ever opens: latent N=1 vs CEM on
H=5, competence slack and compute (WM rollouts + wall-clock) frozen
before outcomes — numbers **not** set here from MiniPush-002 clocks.

---

## Proposed metrics to persist

Quality: success, failure, object L2, terminal latent goal distance,
failure kinds (including `witness_inconsistent` if env rollout of the
**evaluated** plan diverges — still not exposing witness to the planner).

Compute: wall-clock, WM rollouts, planner forwards, ID forwards,
candidates, CEM iterations.

Model: contact-stratified RMSEs, ID Δz diagnostic, WM fingerprint.

---

## Decision tree (same logic as MiniPush-002, for a future freeze)

- CASE A: search useful and N=1 approaches CEM cheaper → future
  stochastic amortization **may** be discussed; not auto-run.
- CASE B: search useful, N=1 much worse, **and** multimodal path
  evidence → Rectified Flow **may** be justified as a later ID.
- CASE C: shooting ≈ CEM → uninformative; no RF.
- CASE D: WM competence fail → stop.
- CASE E: action-flow dominates latent N=1 → latent trajectories not
  needed on that task.

---

## Explicit non-actions

Do not run this draft. Do not freeze it by accident. Do not execute
Qwen 004, IBD-002, stitching, DINO-WM, or LeWM. Do not treat this file
as `PREREGISTERED_NOT_RUN` (that status is for frozen, authorized-or-
explicitly-unauthorized protocols). This is weaker: **not preregistered**.
