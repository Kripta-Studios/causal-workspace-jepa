# CRCT-LEARNED-WM-ACTION-DELTA-002 protocol

Status at freeze: `PREREGISTERED_NOT_RUN`.
Registered outcome: `INCONCLUSIVE` (confirmation closed). Thresholds unchanged.
After the freeze commit, execution of **this ID only** is authorized.

Parent: `CRCT-JEPA-ACTION-DELTA-001` (`MODEL_INCOMPETENT`, evidence `None`).
001 is not mutated, not rerun, and is **not** a mechanistic negative.
IBD-003 remains `MECHANISM_RECOVERY_PASSED` (synthetic IBD). HARD-002 remains
`NEGATIVE_RESULT`. IBD-002 remains not executed.

The draft `docs/research/CRCT_JEPA_ACTION_DELTA_002_DRAFT.md` is not this freeze.

## Nomenclature

The substrate is a **supervised residual MLP** world-model predictor.
It is **not** trained with a JEPA / joint-embedding objective.

`CRCT-JEPA-ACTION-DELTA-*` is the historical **track name** of 001 only.
This ID is `CRCT-LEARNED-WM-ACTION-DELTA-002` so a pass cannot be summarized
as “we interpreted a JEPA.”

## Question

On independently trained **competent** tiny neural world models of PointMass,
does label-blind CRCT recover a minimal causally sufficient and necessary
internal coalition for the learned map **`ax -> Delta vx`**?

## Architecture (frozen; same class as 001)

Reuse `ActionDeltaPredictor` (`H=6`). Do not enlarge width after outcomes.

## Component ontology (frozen; same as 001)

`act_0..5`, `b1_0..5`, `b2_0..5`. Encoder excluded. Residual stream not
independently searchable. Causal re-forward; coordinatewise mean-fill;
hybrid counterfactual patch; post-tanh orthogonal gauge with next-linear
compensation (`W' = W Q`).

## Finite training ladder (frozen before any 002 training)

| Rung | Adam steps | Role |
|---|---:|---|
| A | 200 | 001 budget; expected incompetent; control |
| B | 800 | 4× 001 |
| C | 2000 | 10× 001 |

Optimizer, batch, lr, loss **unchanged** from 001: Adam `3e-3`, batch 64,
pooled MSE on Δ-state. No extra rungs. No architecture/optimizer/threshold
changes after outcomes.

Climb **development models only**. Same frozen competence conjunction as 001:
each of Δx, Δy, Δvx, Δvy NMSE `<= 0.05` on that seed’s development eval.

- If **all** development seeds pass at rung R: select R, **stop climbing**,
  run development CRCT on those models, then confirmation may open.
- If a rung fails: proceed only to the next frozen rung.
- If C fails: `MODEL_INCOMPETENT`, confirmation **CLOSED**, CRCT **CLOSED**.

## Seeds (new; disjoint)

Forbidden: HARD-002; IBD-001/002/003; 001 model seeds `43,47,53,1013,1019,1021`;
Qwen `701,901`; WM `131,137,139,151,157,163,251,257,263`; factorial `21,23,29`.

- Development model seeds: `59, 71, 73`
- Confirmation model seeds: `1031, 1033, 1039`

Per seed `S` (same offsets as 001, new `S`):

```text
train:  S*1000+61   256 traj × 6 steps
dev:    S*1000+67   64 traj
confirm: S*1000+71  64 traj  (confirmation stage only, after circuit freeze)
```

## Competence records (every rung, every seed)

Persist per channel: variance, energy (NMSE denominator), raw MSE, NMSE.
Also: full Δ-state MSE/NMSE, `train_loss_final`, `dev_loss_mse`,
`train_loss_curve` (10 evenly spaced loss points; **not** a stopping rule),
`checkpoint_sha256`, `train_steps`.

## Confirmation

Authorized only if the development artifact for the **selected rung** has
all seeds competent **and** development status `MECHANISM_RECOVERY_PASSED`
(pipeline check; same conservatism as 001). Train confirmation seeds with
**exactly** that rung’s step count. Do not reselect per confirmation seed.

If a confirmation seed fails competence: do **not** run CRCT on it.
If any confirmation seed is incompetent: experiment status
`MODEL_INCOMPETENT_CONFIRMATION`. Confirmation eval is never used for
competence.

## Mechanistic gates (unchanged from 001)

Primary M1 `ax → Δvx` only. Status vocabulary includes 001 names plus
`MODEL_INCOMPETENT_CONFIRMATION` and `ARCHITECTURE_CUTSET`.

| Gate | Rule |
|---|---|
| sufficiency | restore-only `C_hat` NMSE on model `Δvx` `<= 0.05` |
| minimality | inclusion-minimal prune; drop-one NMSE `> 0.05` |
| necessity | mean-fill `C_hat` `Δvx` NMSE `>= 0.10` |
| specificity | `nec_dvx/nec_dvy >= 2` and `nec_dvx/nec_dy >= 2` |
| not action-stem only | `C ⊈ {act_*}` else `ARCHITECTURE_CUTSET`; `max_coalition=4` |
| random controls | 0 of up to 32 same-size coalitions sufficient |
| counterfactual | median gap closed `>= 0.50` |
| gauge identity | MSE `<= 1e-8` |
| gauge functional | gauged greedy+prune sufficiency `<= 0.05` |

Reported: M2–M4, cancellation (may be `NO_MEANINGFUL_CANCELLATION_DETECTED`),
magnitude/gradient/act×grad with the same necessity/sufficiency tests,
literal Jaccard across seeds (not required), RMS-matched controls.

## Provenance

Separate CLI per rung and per confirmation. Collect provenance before writing.

```text
python -m causal_workspace_jepa.experiments.cross_domain.crct_learned_wm_action_delta --stage development --rung 200 --output artifacts/metrics/crct_learned_wm_action_delta_v2.rung200.json
python -m causal_workspace_jepa.experiments.cross_domain.crct_learned_wm_action_delta --stage development --rung 800 --require-previous artifacts/metrics/crct_learned_wm_action_delta_v2.rung200.json --output artifacts/metrics/crct_learned_wm_action_delta_v2.rung800.json
python -m causal_workspace_jepa.experiments.cross_domain.crct_learned_wm_action_delta --stage confirmation --require-development <selected-rung.json> --output artifacts/metrics/crct_learned_wm_action_delta_v2.json
```

Climbing a later rung is refused unless the previous artifact is this ID,
development, competence-failed (`MODEL_INCOMPETENT`), matching digest, and
the immediately prior frozen rung. Confirmation sidecar seed is `1031`.

## Explicit non-actions

Do not mutate or rerun 001. Do not drop Δx/Δy. Do not interpret velocity-only.
Do not execute IBD-002, Qwen 004, planning, stitching, or a JEPA-objective
successor in this freeze.
