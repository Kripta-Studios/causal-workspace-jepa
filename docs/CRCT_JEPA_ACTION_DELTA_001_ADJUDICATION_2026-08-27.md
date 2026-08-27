# CRCT-JEPA-ACTION-DELTA-001 — adjudication (2026-08-27)

Registered outcome: **`MODEL_INCOMPETENT`**.
Evidence level: **None**.
Confirmation: **CLOSED** (not opened).

Freeze commit (before outcomes): `66c5b26e4aad1c1a70fe41f1d374ebb64348c899`.
Development seeds: `43, 47, 53`. Confirmation seeds `1013, 1019, 1021` were
**not** trained or evaluated.

IBD-002 was **not** executed. IBD-003 remains `MECHANISM_RECOVERY_PASSED`
(synthetic IBD). HARD-002 remains `NEGATIVE_RESULT`.

Gates were not retuned. Architecture was not enlarged. Training budget was
not increased.

## Competence (development eval vs physics)

Frozen bar: each of `Δx, Δy, Δvx, Δvy` NMSE `<= 0.05`.

| Seed | Δx | Δy | Δvx | Δvy | status |
|---:|---:|---:|---:|---:|---|
| 43 | 0.626 | 0.424 | 0.00704 | 0.00927 | `MODEL_INCOMPETENT` |
| 47 | 0.782 | 0.782 | 0.0455 | 0.00967 | `MODEL_INCOMPETENT` |
| 53 | 0.262 | 0.448 | 0.0112 | 0.0163 | `MODEL_INCOMPETENT` |

Velocity channels met the bar. Position channels did not. The frozen
conjunction therefore fails. Circuit search was **not** run (fail-closed
before localization).

## What this establishes

Under the frozen 200-step Adam budget, this supervised residual MLP did not
become a competent one-step PointMass predictor of **full** Δ-state.
Mechanism recovery of a learned `ax → Δvx` circuit was not attempted.

## What this does not establish

- That CRCT can or cannot recover a learned neural mechanism
- Localization, necessity, sufficiency, specificity, counterfactual mediation
- Cross-seed mechanistic convergence
- Qwen, workspace, Platonic physics, planning, MiniPush
- Rescue of HARD-002
- That a longer budget would fail (not tested; not authorized here)

## Successor

A later ID may preregister a **finite training ladder** or a larger frozen
step budget **before** outcomes. Do not mutate this ID. Nonlinear/friction
and MiniPush remain unjustified until a competent learned substrate exists.
