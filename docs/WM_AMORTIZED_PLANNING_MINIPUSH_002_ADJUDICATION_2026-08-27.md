# WM-AMORTIZED-PLANNING-MINIPUSH-002 adjudication

Registered outcome: **`UNINFORMATIVE_SUBSTRATE`**.

Evidence level: **`None`**. This is not a planner competence pass, not a
`NEGATIVE_RESULT` of amortization versus CEM, and not permission to add
Rectified Flow.

The frozen T2 result is unchanged. This file does not mutate
`artifacts/metrics/wm_leflow_amortize_v1.json`.

## Protocol identity

- Protocol: `docs/WM_AMORTIZED_PLANNING_MINIPUSH_002_PROTOCOL.md`
- Config: `configs/experiments/wm_amortized_minipush_v1.json`
- Preregistration commit: `c121498`
- Implementation commit: `ccc2cde`
- Qualification commit (before confirmation): `1f71f05`
- Qualification seed 241: `QUALIFICATION_PASSED` (train + development only)
- Confirmation seeds: `251 / 257 / 263`
- Splits accessed at confirmation: `train`, `confirmation`
- Downloads: none
- Not LeFlow, not Rectified Flow, not the LeVLJEPA factorial, not stitching

## Qualification (seed 241)

WM one-step RMSE: state `0.108`, object `0.188` (both below frozen 5.0 / 4.0).
Development shooting success: `0.0` (`< 0.90`). Not all planners ≥ 0.95.
Confirmation of seed 241 was not opened.

Low WM RMSE here is largely **static-object** prediction: contact is rare in
the random-walk training distribution. It is not evidence that the linear
model learned pushing.

## Confirmation

Every seed has `search_useful: false`: CEM success equals shooting success,
so clause 1 (`s_cem >= s_shooting + 0.05`) fails. Frozen status:
`UNINFORMATIVE_SUBSTRATE`.

| seed | shooting H=5 | CEM H=5 | latent N=1 | latent N=64 | action-flow | mean d_obj (CEM) |
|---:|---:|---:|---:|---:|---:|---:|
| 251 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 8.91 |
| 257 | 0.0417 | 0.0417 | 0.0417 | 0.0417 | 0.0417 | 9.30 |
| 263 | 0.0833 | 0.0833 | 0.0833 | 0.0833 | 0.0833 | 10.09 |

The rare successes are the same 1–2 already-near-goal queries for every
planner. H=10 success matches H=5. Almost all failures are
`horizon_insufficient` (Manhattan approach bound `> H`).

WM fingerprints matched before/after planner comparison:
`009eb76c…` / `74edf826…` / `4dac6a94…`.

Δz inverse-dynamics MSE is not meaningfully better (e.g. 251: `1.40e-5` vs
`1.39e-5`). N=64 rerank does not change success. N=1 is faster than CEM
with the same (near-zero) success; that is not a Pareto **planning** win.

## Decision tree

**CASE C.** Random shooting is not a quality ceiling in the 0.90 sense, but
CEM does not outperform it. The substrate cannot support amortization-versus-
search claims. Rectified Flow is **not** justified. A generative planner is
**not** justified.

Do not retune H, N, success 1.5, or qualification gates after this outcome.

## What this does not authorize

- Relabeling as `NEGATIVE_RESULT` of the amortizer (search never separated)
- Stitching, T1/T2 reruns, Qwen 004, IBD-002, LeVLJEPA factorial
- Treating held-out MiniPush episodes as OOD
