# WM-LEFLOW-AMORTIZE-001 adjudication

Registered outcome: **`NEGATIVE_RESULT`**.

Evidence level: **`None`**. The primary gate failed, so Availability is not
earned. JSON originally written with unconditional `Availability` was
schema-corrected to `None` without a new model forward or any gate change.

## Protocol identity

- Protocol: `docs/WM_LEFLOW_AMORTIZE_001_PROTOCOL.md`
- Config: `configs/experiments/wm_leflow_amortize_v1.json`
- Preregistration commit: `7392ab5`
- Execution commit: `f1ac460` (clean tree; MKNN already adjudicated)
- Confirmation seeds: `151 / 157 / 163`
- Primary arm: latent interpolator N=64 vs random shooting N=64 at **H=5**
- Success: closed-loop position MSE `< 0.15`
- Cost clause: amortized mean wall-clock **strictly less** than shooting
- Splits accessed: `train`, `confirmation`
- Downloads: none
- World model fingerprint matched before/after planner comparison

## Primary outcome

All three confirmation seeds have **success 1.0** on both primary amortized
N=64 and shooting N=64, so the success-slack clause is vacuously met. The
**wall-clock clause fails** in every seed: N=64 latent rerank is slower than
N=64 shooting.

| seed | amortized success | shooting success | amortized s | shooting s | seed gate |
|---:|---:|---:|---:|---:|---|
| 151 | 1.0 | 1.0 | 2.60e-4 | 1.22e-4 | fail (slower) |
| 157 | 1.0 | 1.0 | 2.85e-4 | 1.58e-4 | fail (slower) |
| 163 | 1.0 | 1.0 | 3.79e-4 | 1.26e-4 | fail (slower) |

This is a **negative result**. It is not rewritten as a partial pass because
N=1 was sometimes competitive, and it is not retuned by dropping the
wall-clock half of the frozen conjunction.

The success axis is saturated: every arm, including random shooting and
action-flow N=1, has success 1.0 at H=5 and H=10 under the frozen 0.15
floor. The experiment therefore cannot separate planner quality on success.
The fail is only that N=64 latent rerank is slower than N=64 shooting at
sub-millisecond CPU cost. That cost measurement is Python overhead on a
matched 64-rollout budget, not a demonstration that amortization is worse
at planning.

Protocol/code caveat, recorded not patched: shooting/CEM receive the frozen
2-d goal position, while latent-flow interpolates using the full 4-d
`state[H]` as `goal_observation`. That extra velocity channel did not decide
the gate (success ceiling + slower clock). The mismatch is not repaired by
a post-outcome rerun.

## Diagnostics (not gates)

- H=10 success is also 1.0 on this in-distribution PointMass holdout; the
  preregistered collapse warning did **not** trigger. That is not OOD and
  not a reason to claim long-horizon robustness.
- Inverse-dynamics holdout MSE with explicit Δz is slightly lower than the
  capacity-matched zero-padded arm (e.g. seed 151: `1.18e-11` vs `2.13e-10`).
  That is an architectural-prior diagnostic, not proof that the latent space
  is a planner.
- Action-flow N=1 is fastest and also at success 1.0; it is not the primary
  arm.
- Every listed arm saturates success, so this substrate cannot separate
  planners on the success axis under the frozen 0.15 floor.

## What this does not authorize

- Calling the candidate LeFlow
- Claiming CEM is obsolete
- Claiming amortization eliminates replanning
- Opening stitching or `WM-LEFLOW-TRANSFER-001`
- Relabeling HARD-002, V3, IBD-001, or Qwen confirmation
- Lowering the wall-clock clause or the 0.15 success floor after seeing
  outcomes
