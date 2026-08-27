# 001 competence-failure audit (no new models)

Parent: `CRCT-JEPA-ACTION-DELTA-001` `MODEL_INCOMPETENT`.
This audit reconstructs **already opened** 001 development-eval splits
(`S*1000+67`, 64 traj) to interpret stored NMSE. No 001 seed was retrained.
No 002 seed was trained.

001 did **not** store training-loss curves, checkpoints, or prediction
variance. Those are reported as missing, not imputed.

## Stored NMSE (from `crct_jepa_action_delta_v1.dev.json`)

| Seed | Δx | Δy | Δvx | Δvy |
|---:|---:|---:|---:|---:|
| 43 | 0.626 | 0.424 | 0.00704 | 0.00927 |
| 47 | 0.782 | 0.782 | 0.0455 | 0.00967 |
| 53 | 0.262 | 0.448 | 0.0112 | 0.0163 |

Bar: all four `<= 0.05`. Velocity passed. Position failed. Circuit search
was not run.

## Reconstructed eval channel statistics (n=320 transitions)

NMSE denominator is `energy = mean(target²)`, matching `restoration_error`.

Seed 43:

| ch | mean | variance | energy (denom) | absmax | implied MSE | implied RMSE |
|---|---:|---:|---:|---:|---:|---:|
| dx | 3.8e-4 | 3.37e-4 | 3.37e-4 | 0.037 | 2.11e-4 | 0.0145 |
| dy | -1.9e-3 | 1.96e-4 | 2.00e-4 | 0.037 | 8.46e-5 | 0.0092 |
| dvx | 1.7e-3 | 3.18e-3 | 3.19e-3 | 0.100 | 2.24e-5 | 0.0047 |
| dvy | -2.8e-3 | 3.31e-3 | 3.32e-3 | 0.101 | 3.08e-5 | 0.0055 |

Seed 47: dx energy 1.59e-4, dvx 3.30e-3 (ratio 20.7). Implied dx MSE 1.25e-4.
Seed 53: dx energy 2.58e-4, dvx 3.49e-3 (ratio 13.5). Implied dx MSE 6.76e-5.

Train-split energies (`S*1000+61`) are the same order (dx ~2e-4, dvx ~3.4e-3).

## Diagnosis

| Hypothesis | Verdict |
|---|---|
| A. insufficient optimization budget | **Plausible, not proven.** 200 Adam steps × batch 64 ≈ 10 epochs on 1280 train transitions. Position RMSE 0.008–0.015 vs energy-implied σ(Δx)≈0.013–0.018 (NMSE 0.26–0.78 ≈ incomplete fit, not a collapsed predictor). Reaching NMSE 0.05 needs about 2–4× RMSE drop. Train losses were not stored. |
| B. optimization instability | **Unlikely.** Same v-good / x-bad pattern on all three seeds. No missing artifacts suggestive of NaN. Seed 47 Δvx is merely closer to the bar (0.046). |
| C. architecture cannot express the map | **Unlikely as primary.** The PointMass map is nearly affine. The same `H=6` residual MLP already implements Δv to NMSE 0.007–0.046. Δx is `vx_{t+1}·dt` in the frozen Euler step. |
| D. target scaling / conditioning | **Contributing.** Pooled MSE weights channels by raw energy. `energy(Δvx)/energy(Δx) ≈ 9–21`, so the loss is velocity-dominated until Δv saturates. This is a reason more steps can help, not a license to drop Δx/Δy. |
| E. ill-conditioned NMSE denominator | **Rejected.** Denominators are 1.6e-4–3.5e-4, far above the 1e-12 floor. Variance ≈ energy (means near 0). NMSE is numerically meaningful. |
| F. implementation error | **Unlikely.** One 4D linear head; velocity channels from the same tensor meet the bar. |
| G. data-generation error | **Unlikely.** Energies match `dt=0.1` scales (Δv ~ 0.1, Δx ~ 0.01). |
| H. genuine generalization failure | **Not separable.** Train losses were not stored. Eval and train energies match. Pattern is consistent with underfit of a weaker pooled-loss channel, not a shifted eval distribution. |

## Missing 001 records

- training-loss progression: **not stored**
- prediction variance: **not stored** (would require the uncheckpointed weights)
- per-channel train MSE: **not stored**

002 must persist variance, energy, raw MSE, NMSE, train/dev loss, and checkpoint hash.

## Ladder justification

A finite step ladder is a plausible repair for A+D. It is **not** justified as
architecture growth, NMSE retuning, or dropping position competence.

Candidate rungs (chosen from the 002 draft, not from new training):
`200` (001 budget, expected fail on new seeds), `800`, `2000`.
Stop at the first rung where all development seeds meet the **same**
four-channel NMSE `<= 0.05` conjunction.

## Nomenclature

001 is a supervised residual MLP, not a JEPA objective. The historical ID
`CRCT-JEPA-ACTION-DELTA-001` is a **track name**, not an objective claim
(001 protocol already says this). The successor ID must not imply JEPA
training. Use `CRCT-LEARNED-WM-ACTION-DELTA-002`.
