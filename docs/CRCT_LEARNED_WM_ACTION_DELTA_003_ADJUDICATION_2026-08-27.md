# CRCT-LEARNED-WM-ACTION-DELTA-003 — adjudication (2026-08-27)

Registered outcome: **`MODEL_INCOMPETENT`**.
Evidence level: **None**.
Confirmation: **CLOSED** (not opened).
CRCT / path search: **not run**.

Parent 002 remains `INCONCLUSIVE`. Seed 59 is not a retrospective pass.
001 remains `MODEL_INCOMPETENT`. IBD-003 remains `MECHANISM_RECOVERY_PASSED`
(synthetic IBD). HARD-002 remains `NEGATIVE_RESULT`.

Freeze (before 003 training): `a23bbaa74df32c6f453e15bcc9b7a0e2bfda3a2c`.
Frozen budget: **800** Adam steps. No extra rungs.
Development seeds `79, 83, 89`. Confirmation seeds `1049, 1051, 1061`
were **not** trained.

This is a **supervised residual MLP**, not a JEPA-objective result.

## Competence (development eval, bar all four NMSE `<= 0.05`)

| Seed | Δx | Δy | Δvx | Δvy | status |
|---:|---:|---:|---:|---:|---|
| 79 | 0.0236 | **0.139** | 0.00354 | 0.00261 | `MODEL_INCOMPETENT` |
| 83 | 0.0192 | 0.0194 | 0.00545 | 0.00911 | competent, not interpreted |
| 89 | 0.0118 | 0.0477 | 0.00449 | 0.00469 | competent, not interpreted |

Seed 79 also fails Δy on the train split (NMSE 0.121). Train-loss curve is
still falling at step 800 (`1.23e-3` → `1.30e-5`). Velocity and Δx met the
bar. The frozen full-state conjunction therefore fails.

Under the freeze, one incompetent development seed closes CRCT for the
experiment. Seeds 83/89 were not given pathway tests after this outcome.

## Gateway vs pathway

Not asked. The substrate-competence conjunction did not pass on all
development seeds.

## What this establishes

The 800-step budget that was sufficient for 002’s three development seeds
is **not** sufficient for every new seed of the same architecture. 003 does
not answer whether CRCT finds a computation or only an input channel.

## What this does not establish

Pathway recovery; information-gateway-only; 002 reinterpretation; JEPA;
Qwen; workspace; Platonic physics; planning; MiniPush. Not a mechanistic
negative.
