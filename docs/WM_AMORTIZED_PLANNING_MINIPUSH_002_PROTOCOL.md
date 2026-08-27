# WM-AMORTIZED-PLANNING-MINIPUSH-002 protocol

Status: `PREREGISTERED_NOT_RUN`.

This is a **new** CPU-scale amortized-planning experiment on MiniPush contact
dynamics. It does **not** overwrite `WM-LEFLOW-AMORTIZE-001`. It is **not** a
LeFlow or Rectified Flow reproduction, **not** `WM-LEVLJEPA-MINIPUSH-FACTORIAL-001`,
and not permission to open stitching or Qwen 004.

Parent analysis: `docs/WM_PLATONIC_LEFLOW_CPU_ANALYSIS_2026-08-27.md`.
Config: `configs/experiments/wm_amortized_minipush_v1.json`.

## Scientific question

Can a deterministic amortized latent interpolator reduce **online** planning
cost relative to CEM while preserving planning competence when the task is
**not** a PointMass success ceiling?

## Evidence level if the primary gate passes

`Availability` — MiniPush closed-loop planning competence of a CPU
interpolator-plus-inverse-dynamics candidate versus CEM under one frozen
TinyJEPA.

A pass does **not** establish: LeFlow, Rectified Flow, workspace, causal
latents, Platonic physics, elimination of replanning, OOD robustness, or
CEM obsolescence.

Fail-closed labels are below. They are not retuned after outcomes.

## Why MiniPush, and what is reused

T2 PointMass saturated at success 1.0 for every planner, including random
shooting. MiniPush moves an object only on **contact**. Linear latent
interpolation from start to “object already at goal” need not be
dynamically realizable.

Reused: `generate_minipush` dynamics, TinyJEPA, `planning/cem.py`,
`planning/amortized_latent.py`. Not reused as a result: T2 metrics, T2
seeds, T2 goal-slicing.

Pixels (32×32) are **not** the observation. This experiment does not
download or compare LeVLJEPA encoders.

## Frozen substrate

| field | frozen value |
|---|---|
| environment | MiniPush, resolution 32 |
| observation | 6-d **state vector** `[agent_x, agent_y, object_x, object_y, goal_x, goal_y]` |
| action space | four cardinals `{(0,-1),(0,1),(-1,0),(1,0)}` |
| planner sampling | continuous `[-1, 1]^2`, then **quantize** to nearest cardinal |
| world model | TinyJEPA ridge; random-projection encoder `6→16`; ridge `1e-4` |
| WM training | one closed-form fit on train split; no early stopping |
| WM frozen during planner comparison | required; fingerprint match |
| trajectories | 96 |
| steps | 16 |
| split | train 56 / development 16 / confirmation 24 |
| split seed | 277 |
| environment base seed | 281; dataset seed = `281 + run_seed` |
| qualification seed | **241** (train + development only) |
| confirmation seeds | `251, 257, 263` |
| n_tasks development | 16 (all development trajectories) |
| n_tasks confirmation | 24 (all confirmation trajectories) |
| start | `state[0]` of the query trajectory |
| goal observation | constructed, identical for every planner (below) |
| primary horizon | 5 |
| diagnostic horizon | 10 (not OOD) |
| shooting candidates | 64 |
| CEM | 16 candidates × 4 iterations, elite fraction 0.25 |
| amortized N | `{1, 64}` |
| N=1 noise | 0 |
| N=64 noise std | 0.05 |
| ID ridge | `1e-4` |
| resource | `cpu_vps`; no downloads |

### Identical goal information (T2 fix, prospective only)

Every planner receives the **same** 6-d `goal_observation`:

```text
[agent_x0, agent_y0, goal_x, goal_y, goal_x, goal_y]
```

copied from the query’s start agent and the episode goal. Then:

```text
z_goal = encoder(goal_observation)
```

using the **same frozen** world-model encoder.

Planning cost for shooting, CEM, latent N=1, latent N=64, and action-flow
rerank (when a WM score is computed):

```text
cost = || z_predicted_terminal − z_goal ||^2
```

No planner receives a 2-d slice, privileged full-state future, or a
different encoder. Action-flow N=1 does not WM-rerank; it still **encodes**
the same `goal_observation`.

Closed-loop **success** is environment-true object distance, not this
latent cost:

```text
d_obj = || object_xy_H − goal_xy ||_2
success iff d_obj < 1.5
```

The 1.5 pixel threshold is a pre-outcome MiniPush geometry convention
(integer grid; 3×3 drawn squares). It is not estimated from planner
outputs.

Final goal distance stored in artifacts: `d_obj` (mean over tasks, and
per-task arrays).

### Action execution

After any planner emits a length-H action sequence, every action is
quantized to the nearest cardinal **before** world-model scoring (MiniPush
wrappers) and **before** environment rollout. Bounds `[-1, 1]` apply before
quantization.

## Qualification (development, seed 241)

Run **before** any confirmation seed. Access only `train` and
`development` of seed 241. Do not load confirmation rows of seed 241.

World-model competence (development one-step, decoded state vs true next
state):

1. mean RMSE over the 6 state dims `< 5.0`
2. mean RMSE over object xy `< 4.0`

Substrate informativeness (H=5, development queries):

3. random-shooting N=64 success **strictly less than** 0.90
4. it is **not** the case that every listed planner has success `>= 0.95`

If (1) or (2) fail: status `WORLD_MODEL_INCOMPETENT`. Do **not** open
confirmation. Do not refit a larger model.

If (3) or (4) fail: status `UNINFORMATIVE_SUBSTRATE`. Do **not** open
confirmation. Do not invent a harder MiniPush variant.

If all four pass: `QUALIFICATION_PASSED`. Confirmation seeds may run.

No difficulty ladder is authorized. The finite set of MiniPush parameters
above is the only substrate.

## Planners (all use the same WM, starts, goals, H, bounds)

A. Random shooting N=64 — existing `random_shooting_plan` with latent-goal
   cost and MiniPush quantization.
B. CEM 16×4 — existing `iterative_cem_plan`, same cost and quantization.
C. Latent interpolator N=1 (deterministic) and N=64 (noise 0.05), inverse
   dynamics arm B (explicit Δz), frozen-WM rerank for both N.
D. Direct action-flow N=1 — ridge `(z_start, z_goal) → action sequence`.
   Train targets are dataset actions `[:H]` on the train split, with
   `z_goal` from the **constructed** goal observation (not expert push
   demonstrations). This control is intentionally misspecified relative to
   optimal pushing; it is retained because T2 showed a cheap action
   regressor and must not be dropped silently.

Inverse-dynamics diagnostic (not a pass gate): capacity-matched A
`(z_t, z_{t+1}, zeros)` vs B `(z_t, z_{t+1}, Δz)`. Report holdout action
MSE on the evaluation split of that run (development during qualification;
confirmation during confirmation).

## Primary gate (confirmation, frozen)

Primary horizon **H=5**. Primary amortized arm: **`latent_flow_n1`**.
Primary search baseline: **`iterative_cem`**.

Rationale, frozen before outcomes: T2 showed that N=64 WM rerank cannot
reduce world-model forward count against a 64-rollout searcher. The
compute question for a **deterministic** amortizer is N=1 versus CEM.
N=64 remains a frozen secondary arm (Q3). Success slack 0.10 is a
pre-outcome non-inferiority convention for a discrete contact task; it is
not estimated from MiniPush planner outputs.

On every confirmation seed `{251, 257, 263}`:

Let `s_*` be mean success and `t_*` mean wall-clock seconds per replan.

1. Search is useful: `s_cem >= s_shooting + 0.05`
2. Competence: `s_n1 >= s_cem - 0.10`
3. Compute: `t_n1 < t_cem` **and** mean WM rollout forwards of N=1 `<` CEM

If any seed violates (1), or qualification-equivalent shooting ceiling
appears on confirmation: `UNINFORMATIVE_SUBSTRATE`.

If (1) holds on all seeds but (2) or (3) fails on any seed:
`NEGATIVE_RESULT`.

If all seeds satisfy (1)–(3): `AMORTIZED_PLANNING_COMPETENCE_PASSED`.

Do not change H, N, slack, 1.5, or CEM/shooting budgets after outcomes.

## Secondary analyses (predeclared; not gates)

Q1. Does CEM outperform random shooting? (also used in the primary
    conjunction as clause 1)
Q2. Does N=1 approach CEM competence at lower online cost? (primary)
Q3. Does N=64 WM reranking improve success or mean `d_obj` enough to
    justify 64 WM rollouts versus N=1?
Q4. Is N=1 a Pareto point vs shooting and CEM (success vs WM forwards and
    wall-clock)?
Q5. Does action-flow N=1 outperform latent N=1 on success or `d_obj`?
Q6. Does explicit Δz reduce holdout action MSE vs the zero-padded arm?
Q7. H=5 → H=10 change in success and mean `d_obj` (not OOD)
Q8. Failure kinds, objective only:
    - `success`
    - `horizon_insufficient` if `manhattan(object,goal)+manhattan(agent,object)-1 > H` and fail
    - `no_object_motion` if object xy unchanged and fail
    - `goal_miss` otherwise on fail

## Artifact schema (must persist; fail closed if missing)

Per arm, per horizon, per seed:

- `success_rate`, `failure_rate` (= 1 − success_rate)
- `mean_goal_distance` (object L2)
- `mean_terminal_latent_goal_distance`
- `mean_wall_clock_s`, `sum_wall_clock_s`
- `mean_wm_rollout_forwards`, `mean_planner_forwards`, `mean_id_forwards`
- `mean_candidates_evaluated`, `mean_cem_iterations`
- `n_tasks`
- `cpu_peak_rss_bytes` or `null` if the host cannot report it

Model block: WM one-step RMSEs, ID MSEs, fingerprint before/after.

Missing required fields in the confirmation JSON is an integrity failure,
not a license to reconstruct from logs.

## Decision tree (frozen)

- CASE A: CEM beats shooting and N=1 approaches CEM at lower cost → a
  **future** stochastic/generative amortization protocol **may** be
  justified. Do not execute it here.
- CASE B: CEM beats shooting but N=1 is substantially worse → analyze Q8.
  Rectified Flow is justified **only** if failures are plausibly from
  averaging multimodal trajectories. Do not implement it here.
- CASE C: shooting near ceiling or CEM not better than shooting →
  uninformative. No Rectified Flow.
- CASE D: WM competence gate fails → `WORLD_MODEL_INCOMPETENT`; planner
  comparison not attributable.
- CASE E: action-flow dominates latent N=1 on success and `d_obj` →
  evidence against needing latent trajectories on this task.

## Claim boundary

`not_leflow`, `not_rectified_flow`, `not_workspace`, `not_ood`,
`does_not_eliminate_replanning`, `does_not_make_cem_obsolete`,
`does_not_relabel_hard002`, `does_not_open_stitching`,
`does_not_execute_levljepa_factorial`.

## Stopping rule

One qualification pass on seed 241. If it fails, stop. If it passes, one
confirmation pass over `{251, 257, 263}`. No extra seeds, H, N, or
threshold search.

## Forbidden seeds

HARD-002 `1009/2027/4093`; IBD `11/13/17/811/823/829`; factorial
`21/23/29/941/947/953`; Qwen `701/901`; T1 `131/137/139`; T2
`151/157/163`. Qualification seed 241 must not appear in confirmation.

## Splits

`test` and `paraphrase` must not exist or be accessed. Development is not
an adjudicating split. Confirmation is opened only after committed
qualification success.
