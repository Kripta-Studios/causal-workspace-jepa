# CRCT-LEARNED-WM-ACTION-DELTA-004 — adjudication (2026-08-27)

Registered outcome: **`INCONCLUSIVE`**.
Evidence level: **None**.
Confirmation: **CLOSED** (not opened).
Selected rung: **2000** (first fully competent frozen rung).
Rung 5000: **not run** (stop rule).

Parent 003 remains `MODEL_INCOMPETENT`. Parent 002 remains `INCONCLUSIVE`.
Seed 59 is not a retrospective pass. 001 remains `MODEL_INCOMPETENT`.
IBD-003 remains `MECHANISM_RECOVERY_PASSED` (synthetic IBD). HARD-002
remains `NEGATIVE_RESULT`.

Freeze (before 004 training): `20a8c2019d696951fb05025dd1a4a8ea7e678ceb`.
Development seeds `97, 101, 107`. Confirmation seeds `1063, 1069, 1087`
were **not** trained.

This is a **supervised residual MLP**, not a JEPA-objective result.

## Competence ladder

Bar: all four Δ NMSE `<= 0.05` on development eval.

### Rung 800 — `MODEL_INCOMPETENT` (CRCT not run)

| Seed | Δx | Δy | Δvx | Δvy |
|---:|---:|---:|---:|---:|
| 97 | **0.260** | **0.431** | 0.00542 | 0.0148 |
| 101 | 0.0356 | **0.159** | 0.00500 | 0.0107 |
| 107 | **0.0572** | **0.0595** | 0.00508 | 0.00589 |

Train-split Δy also failed for 97 and 101. Loss curves were still falling.
This is undertraining, not a post-hoc extra rung.

### Rung 2000 — all seeds competent; CRCT ran

| Seed | Δx | Δy | Δvx | Δvy |
|---:|---:|---:|---:|---:|
| 97 | 0.00930 | 0.00860 | 0.00162 | 0.00388 |
| 101 | 0.0111 | 0.0305 | 0.00117 | 0.00348 |
| 107 | 0.0158 | 0.0192 | 0.00137 | 0.00103 |

## Mechanism (rung 2000 only)

| Seed | MSRS | path class | status | level |
|---:|---|---|---|---:|
| 97 | `{act_0, act_3, act_1}` | `DIRECT` | `INFORMATION_GATEWAY_ONLY` | 2 |
| 101 | `{act_3, act_1, act_5, b1_0}` | `REDUNDANT_ROUTES` | `SUFFICIENCY_FAILED` (Δvx restore 0.088) | 0 |
| 107 | `{act_5, act_0, act_1}` | `DIRECT` | `SPECIFICITY_FAILED` (Δvx/Δy 1.941) | 0 |

Seed 97 matches the 002 seed-59 *pattern* (action-stem, high `G_full` 0.992,
`G_skip` 0.998, `G_res` 0.197, sufficiency 0.0083, necessity 1.23, spec
ratios 3.30 / 3.99). Under the freeze it is **not** Level 3.

No seed is `DIRECT_PATH_MECHANISM_PASSED` or
`DISTRIBUTED_PATH_MECHANISM_PASSED`. Shared path class for a pass: none.

## What this establishes

On new seeds, 800 steps is again not seed-universal. At the first competent
rung, label-blind CRCT did not recover a residual-inclusive xor-split
pathway that passed every frozen gate on every development seed.

Seed 97 is Level-2 evidence that an action-stem MSRS can be a strong
causal mediator / skip-route carrier without counting as computational
path recovery.

## What this does not establish

Level-3 path-mechanism recovery; experiment-level
`PATH_MECHANISM_RECOVERY_PASSED`; unique residual computation; JEPA;
reinterpretation of 002 seed 59 or 003 seeds 83/89; Qwen; workspace;
planning; MiniPush. Confirmation was not opened. A JEPA-objective
successor is **not** justified by this outcome.

## Provenance

Rung 800 sidecar: freeze commit, `git_dirty` false.
Rung 2000 sidecar: same freeze commit, `git_dirty` true because the rung
800 metrics were still untracked when 2000 ran. `source_digest` matches
the freeze module. Confirmation CLI was not invoked.
