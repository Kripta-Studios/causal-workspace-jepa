# Preregistered Hypotheses

Status: `ACTIVE`. CPU smoke experiments have tested restricted forms of H-WM-02,
H-WM-03, H-WM-05, H-LLM-01, and H-LLM-02. They do not constitute broad
scientific validation.

## World Model

- H-WM-01: Physics emergence.
- H-WM-02: Action-sensitive displacement.
- H-WM-03: Causal specificity.
- H-WM-04: Planning-specific representation.
- H-WM-05: Workspace candidate.
- H-WM-06: Selective necessity.
- H-WM-07: Distributed physics.
- H-WM-08: OOD mechanism shift.
- H-WM-09: Adaptation mechanism.
- H-WM-10: Object causality.

## LLM Meta-Model

- H-LLM-01: Nonlinear advantage.
- H-LLM-02: Causal compression.
- H-LLM-03: Compositional interventions.
- H-LLM-04: Reusable mechanism features.
- H-LLM-05: Filler computation.
- H-LLM-06: Direct-verification precision.
- H-LLM-07: Cross-scale transfer.
- H-CROSS-01: Shared intervention formalism.

Metrics, splits, thresholds, layers, and magnitudes must be registered before each associated experiment.

## WM-T0-004 Preregistration

Registered before execution on 2026-07-20. The implementation and config are committed before the
scientific runner is invoked.

- Scope: test H-WM-05, H-WM-06, and a restricted H-WM-08 on a two-hidden-layer learned NumPy JEPA;
  this is not a J-space reproduction and cannot establish a full workspace.
- Data: locally generated PointMass2D only, `128` trajectories of `24` steps with seed `17`.
  Deterministic trajectory splits are approximately 70/15/15 percent. Predictor training uses only
  train trajectories; uncertainty calibration and frozen-consumer fitting use validation; all
  reported head and intervention decisions use test. OOD uses `32` new trajectories with mass
  `0.55`, drag `0.2`, and seed `10017`.
- Predictor: fixed non-collapsing 8-dimensional encoder; learned hidden widths `24` and `16`; five
  bootstrap members, `1,200` Adam steps each. Held-out one-step MSE must be at most `0.25` times the
  MSE obtained after shuffling held-out actions.
- Uncertainty: calibrate one scalar ensemble-variance scale on validation for 90 percent marginal
  coverage. It is valid only if test coverage is in `[0.80, 0.98]`, Gaussian NLL is no more than
  `0.05` above the validation-fitted homoscedastic baseline, OOD rank AUC is at least `0.65`, and
  uncertainty/error Spearman correlation across ID+OOD examples is at least `0.30`.
- Consumers: independently fit frozen quadratic heads for next-latent dynamics, value, risk,
  calibrated uncertainty, and action selection. Every test R2 must be at least `0.50`.
- Candidate: discover separately at `predictor.hidden1` and `predictor.hidden2`; at most five
  dimensions, at least `0.75` normalized Jacobian capture for every consumer, and compactness ratio
  at most `0.35`.
- In-manifold intervention: replace candidate coordinates from a validation donor selected among the
  16 nearest examples in the orthogonal complement. The test recipient complement remains fixed.
- Controls: 64 seed-fixed equal-dimensional random bases plus PCA, all using the same conditional
  donor rule. A control is matched only when its median nearest-manifold distance and perturbation
  RMS are each within a factor of two of the candidate and its density-distance ratio is at most
  `3.0`. At least 16 random controls and PCA must be matched.
- Specificity: candidate direct damage must exceed the 95th percentile of matched random controls
  and matched PCA. Its eight-step rollout damage must exceed the 95th percentile of up to 16 matched
  controls, with multistep/one-step damage ratio at least `1.25`.
- Decision: a shared causal subspace requires every predictor, uncertainty, consumer, specificity,
  and selective-necessity criterion above. `workspace_found` remains false regardless because this
  run lacks goal/instruction controllability, cross-task flexible reuse, and reportability analogue.

## WM-T0-003 Preregistration

Registered before execution on 2026-07-20.

- Scope: detector validation and a multi-consumer shared-subspace candidate in the frozen
  `WM-T0-001` representation; not a full H-WM-05 workspace test.
- Split: deterministic trajectory split with seed `0`; all heads and normalization fit on train only.
- Consumers: dynamics prediction, value, risk, OOD-uncertainty proxy, and action selection.
- Consumer validity: every held-out quadratic readout must have R2 at least `0.8`.
- Candidate: at most `4/16` dimensions, at least `0.75` normalized Jacobian sensitivity for every
  consumer, and compactness ratio at most `0.3`.
- Causal specificity: candidate projection must damage all five direct readouts more than every one
  of `32` seed-fixed equal-dimensional random subspaces.
- PCA control: privilege over high-variance PCA requires normalized damage margin greater than
  `0.05`; failure is a negative result.
- Detector controls: recover a known three-dimensional shared subspace and reject five disjoint
  private subspaces under the same discovery rule.
- Workspace decision: remains false in this study because goal/instruction controllability,
  depth/horizon evolution, and held-out task generalization are not tested.

## LLM-GPT2-002 Preregistration

Registered before execution on 2026-07-20.

- Scope: CPU GPT-2 Medium causal meta-model study; not Qwen and not a J-space/workspace test.
- Prompts: eight fixed local prompts; prompt IDs `0-5` train and `6-7` test.
- Sites: residual output of GPT-2 blocks `6`, `12`, and `18`; layers `6/12` train and layer `18`
  is held out. Only the final non-padding token is intervened on and captured.
- Interventions: two fixed residual coordinates and magnitudes
  `[-1.0, -0.5, -0.25, 0.25, 0.5, 1.0]`. Train magnitudes have absolute value at most `0.5`;
  evaluation uses `+/-1.0`.
- Context: the clean residual at the intervention site compressed by a seed-fixed 32-dimensional
  random projection. The real post-intervention target is never an input.
- Target: 32 evenly spaced final-residual coordinates and 32 logit coordinates selected using clean
  training prompts only.
- Primary split: held-out prompts and held-out `+/-1.0` magnitude at seen layers, 16 examples.
- Stress split: held-out prompts, magnitude, and layer `18`, 8 examples. It is descriptive and does
  not determine the primary hypothesis decisions.
- H-LLM-01 passes only if the better of the bilinear Intervention-JEPA and trained MLP has lower raw
  effect MSE than the prompt-local finite-difference Jacobian.
- H-LLM-02 passes only if bilinear Intervention-JEPA MSE is lower than both no-change and linear
  intervention regression.
- H-LLM-06 smoke passes only if the best primary-split predictor has effect correlation greater than
  `0.5` against direct executions.
- Baselines: no-change, train mean, linear regression, bilinear Intervention-JEPA, trained MLP,
  nearest neighbor, top-k sparse-context linear transport, prompt-local finite difference, and
  corpus-averaged Jacobian.
- Storage: 288 prompt-layer-direction-magnitude outcomes, float16 ignored activation shard, selected
  layers/positions only, and a hard 64 MB estimate budget. A checksum manifest is committed.
- Resources: local cached `gpt2-medium` only; sequence length `24`; no downloads and no GPU claim.

Measured result: H-LLM-01 failed because local-Jacobian MSE was `7.79e-7` versus `0.003499`
for bilinear Intervention-JEPA and `0.007361` for the MLP. The restricted H-LLM-02 test passed
because bilinear MSE beat no-change and linear regression on unseen prompts/magnitude, but it failed
to transfer to held-out layer 18. The restricted H-LLM-06 correlation threshold passed only because
the prompt-local Jacobian uses direct small-magnitude probes; this is not the full circuit-ranking
hypothesis from `AGENTS.md`.
