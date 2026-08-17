# CRCT-STAGE0-001 retained result — 2026-08-17

Status: `SYNTHETIC_POSITIVE_CONTROL_RETAINED`.

This note freezes the interpretation of the already executed full-profile CRCT Stage-0 run. It does
not modify any original threshold or artifact and it does not upgrade the evidence to a real-model
circuit, JEPA mechanism, J-space analogue, workspace, or SOTA result.

## Executed bundle

- Experiment: `CRCT-STAGE0-001`
- Base commit: `f69cc28f00faf9d5382e3a47a551410785ae9374`
- Profile: `full`
- Seeds: `7, 13, 23`
- Aggregate artifact SHA-256: `095e72aee58f4ad40fb02751edde0d13ccc5a426e8002efd5db7f3d517d2a522`
- Per-seed result SHA-256:
  - seed 7: `5016f2b97d5748296f820dd7f475d0c550587ff1fcd2f7a32f59805f3b536bdf`
  - seed 13: `84fcb4e601e5aaf2a52eede5d7a1e04abe4cdac80cc84d015107318d749efaff`
  - seed 23: `79c29ad94f862bd582067450b97a877bf5815da1d8a66d5ad3dcab776b56132f`

## Retained observations

| seed | residual power after T2 | residual AP | precision@truth-k | gauge causal rho | gauge activation rho | student replay NMSE |
|---:|---:|---:|---:|---:|---:|---:|
| 7 | 0.484252 | 1.0 | 1.0 | 1.0 | -0.210435 | 0.000702 |
| 13 | 0.715321 | 1.0 | 1.0 | 1.0 | -0.066087 | 0.001088 |
| 23 | 0.384002 | 1.0 | 1.0 | 1.0 | -0.017391 | 0.000490 |

Mean residual power is `0.527858`. The positive-control benchmark therefore demonstrates that the
implemented residual-causal score can recover a planted sparse mechanism in three independently
sampled synthetic systems while simple activation magnitude is deliberately fooled by disconnected
high-variance nuisance and a function-preserving diagonal gauge changes coordinate magnitude ranks.

## Important limitation discovered after execution

The original `matched_random_specificity` statistic is not accepted as confirmatory evidence. The
candidate top-k was selected using the same residual-causal score that was subsequently compared
against random sets, so the reported plus-one p-value `1/(256+1) = 0.003891...` is affected by
selection-on-evaluation. It remains a descriptive positive-control diagnostic only.

CRCT-STAGE0-HARD-002 replaces this control with validation-frozen circuit selection and
validation-frozen matched controls, then generates IID/OOD confirmation data only after the frozen
plan has been hashed.
