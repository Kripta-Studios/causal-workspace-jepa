# CRCT-LEARNED-WM-CONDITIONAL-MEDIATOR-006 — adjudication (2026-08-27)

Registered outcome: **`INCONCLUSIVE`**.
Evidence level: **None**.
Confirmation: **CLOSED** (not opened).
Selected rung: **800** (first fully competent frozen rung).
Rungs 2000 and 5000: **not run** (stop rule).

Parent 005 remains `INCONCLUSIVE`. 001–004, IBD-003, and HARD-002 are
unchanged. Seed 97 is not Level 3.

Freeze (before 006 training): `0d3aed7d8b6c285326d3985c2af3d16a4b5449e4`.
Development seeds `173, 179, 181`. Confirmation seeds `1171, 1181, 1187`
were **not** trained.

Supervised residual MLP, not a JEPA objective. Level 3 is not authorized.
Cached `r2_P` does not assign class. Residual-unit membership was not
required by fiat.

Independent pre-freeze reviews: protocol
[32c96334](32c96334-8148-478c-8088-3faa3772fcb2) and adversarial
[cca4ed4b](cca4ed4b-fa89-4cd5-9554-fd279ab02775), both
**FREEZE_ALLOWED** after P0 repairs.

## Competence ladder

Bar: all four Δ NMSE `<= 0.05` on development eval.

### Rung 800 — all seeds competent; CRCT ran

| Seed | Δx | Δy | Δvx | Δvy |
|---:|---:|---:|---:|---:|
| 173 | 0.0347 | 0.0105 | 0.00212 | 0.00656 |
| 179 | 0.0412 | 0.0211 | 0.00154 | 0.00298 |
| 181 | 0.00777 | 0.0112 | 0.00339 | 0.000778 |

Stop. Do not climb.

## Mechanism (rung 800)

Stage 2B is status-determining only after Stage A. No seed cleared Stage A.

| Seed | `V_up` | action-stem | Stage A stop | `G_V` (diag.) |
|---:|---|---|---|---:|
| 173 | `{act_1, act_3, act_2, b1_2}` | no | `SUFFICIENCY_FAILED` (Δvx 0.149) | 0.880 |
| 179 | `{act_4, act_0, act_5, act_2}` | yes | `SPECIFICITY_FAILED` (Δvx/Δvy 0.777) | 1.000 |
| 181 | `{act_1, b1_0, act_4, act_2}` | no | `SUFFICIENCY_FAILED` (Δvx 0.366) | 0.595 |

`downstream_class` is null for all seeds (`stage_2b_ran=false`).
Diagnostic `N_down`/`S_down` are recorded, not a pass. Seed 179’s
`G_V ≈ G_damaged`, so diagnostic `S_down` is ill-conditioned; it does
not assign a class.

Recovered coalitions were inclusion-minimal (every leave-one-out drop
exceeded 0.05) and random-control sufficient count was 0. Those facts
do not rescue Stage A.

No seed is `DIRECT_TRANSMISSION_PASSED` or `DOWNSTREAM_*_MEDIATION_PASSED`.

## What this establishes

On new seeds, 800 steps can be fully competent. At that first competent
rung, label-blind Stage A did not recover a `V_up` that passed the frozen
sufficiency/specificity conjunction, so conditional downstream mediation
was not adjudicated.

Planted instruments still distinguish early-carrier restoration from
downstream F1 computation. That is method validation, not learned-model
evidence.

## What this does not establish

Learned Level 2B `V_down`; Level 3 edges; experiment-level
`CONDITIONAL_DOWNSTREAM_MEDIATION_PASSED`; JEPA; confirmation.
A JEPA-objective successor is **not** justified.

## Provenance

Rung 800 sidecar: freeze commit `0d3aed7`, `git_dirty` false.
Confirmation CLI was not invoked. 004/005 `git_dirty` sidecars were not
rewritten.
