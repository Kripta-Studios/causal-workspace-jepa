# WM-LEFLOW-AMORTIZE-001 protocol

Status: `PREREGISTERED_NOT_RUN`.

This is a CPU-scale **amortized planning control** on one frozen tiny JEPA.
It is **not** a LeFlow paper reproduction, not a DINO-WM/LeWM download, not
planner transfer `A→B`, and not a claim that amortization eliminates
replanning.

Parent plan: `docs/research/PLATONIC_LEFLOW_INTEGRATION_PLAN.md`.
Config: `configs/experiments/wm_leflow_amortize_v1.json`.

Authorized only after `WM-PLATONIC-MKNN-001` has a committed adjudication
with empty integrity blockers. A **negative** MKNN result is not a blocker.

## Scientific question

Can a small amortized latent planner reduce **per-replan iterative search
cost** while preserving planning success under the **same frozen** world
model?

Amortizing optimization inside a replan ≠ eliminating replanning ≠
eliminating search universally.

## Evidence level if the primary gate passes

`Availability` — planning competence of a CPU interpolator-plus-inverse-dynamics
reranker versus random shooting on frozen TinyJEPA / PointMass2D.

A pass does **not** establish: LeFlow reproduction, workspace, latent space
as a planner, CEM obsolete, OOD generalization, or causal-circuit
equivalence.

A fail is `NEGATIVE_RESULT` and is not retuned.

## Frozen substrate

| field | frozen value |
|---|---|
| environment | PointMass2D |
| observations | native 4-d state (no dual maps, no stitching) |
| world model | TinyJEPA; encoder random projection `4→16`; predictor ridge `1e-4` |
| WM frozen during planner comparison | required; fingerprint must match |
| trajectories | 96 |
| steps | 16 |
| split | train 56 / development 16 / confirmation 24 |
| split seed | 229 |
| environment base seed | 233; dataset seed = `233 + confirmation_seed` |
| confirmation seeds | `151, 157, 163` |
| n_tasks | all 24 confirmation trajectories |
| start | confirmation `state[0]` |
| goal | true position `state[H, :2]` (in-distribution holdout, **not OOD**) |
| primary horizon | 5 |
| diagnostic horizon | 10 (collapse is an allowed, preregistered failure mode) |
| shooting candidates | 64 |
| CEM | 16 candidates × 4 iterations (64 world-model rollouts) |
| amortized N | `{1, 64}` |
| N=1 noise | 0 (deterministic interpolator) |
| N=64 noise std | 0.05 (perturbed interpolator, **not** rectified flow) |
| inverse-dynamics ridge | `1e-4` |
| resource | `cpu_vps`; no downloads |

The candidate is a **deterministic/perturbed latent interpolator** plus
capacity-matched ridge inverse dynamics plus frozen-WM rollout reranking.
Do not call it LeFlow unless the paper method is actually implemented.

## Frozen ablations (all run; only H=5 latent N=64 is primary)

1. **latent-flow**: interpolate `z_0 → z_goal`, inverse-dynamics to actions,
   rerank with frozen WM.
2. **action-flow**: ridge map `(z_0, z_goal) → action sequence` (N=1).
3. **deterministic vs perturbed**: N=1 noise 0 vs N=64 noise 0.05.
4. **N=1 vs N=64** world-model rollout reranking.
5. **H ∈ {5, 10}**. Longer horizon may collapse; that is not a retune trigger
   and is not OOD.
6. Wall-clock is recorded separately from success and is used in the primary
   gate as specified below.

## Inverse-dynamics H-WM-02 control

Same ridge capacity (`[z_t, z_{t+1}, pad]` vs `[z_t, z_{t+1}, Δz]`):

- A: input `(z_t, z_{t+1})` with **zeros** in the Δz slot at train **and**
  decode time.
- B: input `(z_t, z_{t+1}, z_{t+1}-z_t)`.

Same split, same ridge, same evaluation. Report holdout action MSE for both.
This is a **diagnostic**, not a pass/fail gate. A Δz improvement is not proof
that the latent space is a planner or workspace.

Primary amortized arm uses **B** (explicit Δz).

## Frozen primary gate (pre-outcome)

On confirmation seeds `{151,157,163}`, horizon **H=5**, matched start/goal
population:

- Success for a task: true PointMass closed-loop position MSE `< 0.15`.
- Let `s_amort` = mean success of latent-flow N=64.
- Let `s_shoot` = mean success of random shooting N=64.
- Let `t_amort`, `t_shoot` be mean wall-clock seconds per task.

Pass (`AMORTIZED_PLANNING_COMPETENCE_PASSED`) iff **every** confirmation seed
has:

1. `s_amort >= s_shoot - 0.05`
2. `t_amort < t_shoot`

Rationale, frozen before outcomes: 0.15 MSE is a pre-outcome PointMass
tolerance (~0.39 position RMSE) independent of planner outputs; the 0.05
success slack is a pre-outcome “no worse than shooting” convention; the
wall-clock clause is the cost-reduction half of the scientific question.
H=10 is diagnostic only.

Fail-closed: `NEGATIVE_RESULT`. CEM is reported, not part of the primary
conjunction. Do not change H, N, noise, or the MSE floor after seeing
results.

## Interpretation constraints

Do not claim amortization eliminates replanning. Do not claim CEM is
obsolete. Do not treat H=5 vs H=10 as OOD. Do not use this result to relabel
HARD-002 or to open stitching/transfer.

## Artifacts

- `artifacts/metrics/wm_leflow_amortize_v1.json`
- `artifacts/metrics/wm_leflow_amortize_v1.provenance.json`

## Stopping rule

One confirmation pass over the three frozen seeds after MKNN adjudication.
No extra seeds or horizon search.
