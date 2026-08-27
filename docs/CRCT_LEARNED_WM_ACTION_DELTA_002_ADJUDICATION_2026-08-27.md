# CRCT-LEARNED-WM-ACTION-DELTA-002 — adjudication (2026-08-27)

Registered outcome: **`INCONCLUSIVE`**.
Evidence level: **None**.
Confirmation: **CLOSED** (not opened).
CRCT search: development only, after all seeds were competent. Not a pass.

Parent 001 remains `MODEL_INCOMPETENT` (not a mechanistic negative).
IBD-003 remains `MECHANISM_RECOVERY_PASSED` (synthetic IBD).
HARD-002 remains `NEGATIVE_RESULT`. IBD-002 was not executed.

Freeze (before 002 training): `3649dd9ade01e214a0b7ba7897c60685b098743a`.
Selected competence rung: **800** (first competent rung; rung 2000 **not run**).
Development seeds `59, 71, 73`. Confirmation seeds `1031, 1033, 1039` **not trained**.

This is a **supervised residual MLP**, not a JEPA-objective result.

## Ladder competence (full-state NMSE `<= 0.05` on all four Δ channels)

Rung 200 eval NMSE; CRCT not run:

| Seed | Δx | Δy | Δvx | Δvy | train loss final |
|---:|---:|---:|---:|---:|---:|
| 59 | 0.677 | 0.394 | 0.0208 | 0.00708 | 7.03e-5 |
| 71 | 0.940 | 0.821 | 0.00346 | 0.0166 | 1.10e-4 |
| 73 | 0.509 | 0.777 | 0.0231 | 0.0163 | 1.03e-4 |

Rung 800 eval (development split). All seeds **competent**. CRCT ran.

| Seed | var Δx | MSE Δx | NMSE Δx | var Δy | MSE Δy | NMSE Δy | var Δvx | MSE Δvx | NMSE Δvx | var Δvy | MSE Δvy | NMSE Δvy |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 59 | 2.11e-4 | 6.03e-6 | 0.0278 | 1.88e-4 | 4.47e-6 | 0.0223 | 3.07e-3 | 1.02e-5 | 0.00334 | 2.97e-3 | 8.59e-6 | 0.00289 |
| 71 | 2.00e-4 | 9.70e-7 | 0.00470 | 1.97e-4 | 2.67e-6 | 0.0135 | 3.46e-3 | 5.64e-6 | 0.00163 | 3.19e-3 | 5.21e-6 | 0.00164 |
| 73 | 2.93e-4 | 9.42e-6 | 0.0322 | 2.16e-4 | 3.21e-6 | 0.0149 | 3.60e-3 | 2.07e-5 | 0.00576 | 3.20e-3 | 2.80e-5 | 0.00878 |

Seed 59 train-loss curve (10 points) fell from `3.94e-4` to `1.12e-5`.

## Development CRCT (primary M1 `ax → Δvx`)

| Seed | C_hat | status | suff Δvx | nec Δvx | nec Δvx/Δvy | nec Δvx/Δy | CF gap | random suff |
|---:|---|---|---:|---:|---:|---:|---:|---:|
| 59 | `{act_2, act_0}` | `ARCHITECTURE_CUTSET` | 0.032 | 0.895 | 18.6 | 2.32 | 0.998 | 0 |
| 71 | `{act_2, act_1, act_4, b1_0}` | `SUFFICIENCY_FAILED` | 0.168 | 0.993 | 1.81 | 0.574 | 1.000 | 0 |
| 73 | `{act_4, act_5, b2_0}` | `SPECIFICITY_FAILED` | 0.042 | 0.339 | 3.06 | 0.257 | 0.814 | 0 |

Seed 59’s action-only coalition met the numerical sufficiency, necessity,
specificity, random-control, and counterfactual gates, but is **not a pass**
under the frozen `C ⊈ {act_*}` rule. Residual-inclusive alternates were not
sufficient (`alt` restore 0.400 / 0.883 / failed similarly). Magnitude,
gradient, and act×grad top-k coalitions were not sufficient on seed 59.
Cancellation: `NO_MEANINGFUL_CANCELLATION_DETECTED`. Gauge function MSE
`~1e-15`; gauged sufficiency on seed 59 was 0.014. Literal Jaccard 59–71 0.20,
59–73 0.00, 71–73 0.17.

Gates were not retuned after seeing the action-stem numbers.

## What this establishes

A tiny supervised PointMass MLP **can** meet the frozen full-state competence
bar at 800 Adam steps. Under the frozen CRCT gates, development did **not**
recover a passable residual-block mechanism for `ax → Δvx`. Confirmation
was not opened. This is a CRCT result on competent models, not a 001-style
competence stop, and not `MECHANISM_RECOVERY_PASSED`.

## What this does not establish

Learned-network `MECHANISM_RECOVERY_PASSED`; JEPA interpretation; Qwen;
workspace; Platonic physics; planning; MiniPush; rescue of 001 or HARD-002;
that action-embedding-only coalitions are scientifically invalid (002 forbade
them by freeze; a later ID may ask that question prospectively).
