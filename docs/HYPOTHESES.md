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

## LLM-QWEN-001 Instrumentation Preregistration

Registered on 2026-07-21 before downloading weights or executing any Qwen forward pass. This is an
instrumentation acceptance test, not a test of H-LLM-01 through H-LLM-07 and not a workspace test.

- Model: public `Qwen/Qwen3-0.6B` at immutable revision
  `c1899de289a04d12100db370d81485cdf75e47ca`; estimated repository bytes `1,519,209,243`;
  Apache-2.0 model card; bfloat16 on the RTX 5070 Ti.
- Inputs: four fixed repository-authored seven-token factual/causal prompts, seed `41`, sequence cap
  `16`. This smoke has no train/test scientific split.
- Site: `blocks.14.resid_post`, final non-padding position, coordinates `0..7`. Target capture is
  `blocks.27.resid_post` plus logits.
- Operations: zero, a mean computed across the four clean prompt activations, donor resampling,
  donor patching, and magnitude-2 steering. The donor is prompt 1 and the recipient prompt 0.
- Required controls: two hooked clean forwards must be bitwise identical. Mean and resampling may
  not fall back to a batch-one identity; both require explicitly registered statistics/donors.
- Autograd: the selected maximum logit must have a finite nonzero gradient with respect to the
  captured layer-14 residual activation in a separate graph-preserving forward.
- Acceptance: the resolved model revision equals the pin; deterministic max absolute logit error is
  exactly zero; every operation has nonzero mean absolute logit delta; autograd gradient norm is
  finite and positive.
- Evidence boundary: passing establishes selected-site availability and direct causal mediation
  only. It cannot establish a Qwen circuit, feature meaning, meta-model fidelity, behavior change,
  J-space, or workspace.

Measured result from clean commit `0d6a37b`: every registered acceptance gate passed. Resolved
revision matched, replay error was exactly zero, the selected-logit autograd norm was `0.944`, and
all five operations caused nonzero downstream changes. This validates instrumentation only and does
not decide any numbered LLM hypothesis.

## LLM-INTDATA-001 Dataset Preregistration

Registered on 2026-07-21 before dataset generation. This creates causal observations for later
H-LLM-01/02/03/06 tests; dataset-generation metrics do not decide those hypotheses.

- Model/revision: the same pinned `Qwen/Qwen3-0.6B` revision `c1899de...`, bfloat16, seed `53`.
- Prompts/splits: 12 unique fixed prompts in geography and causal-physics families; prompt IDs
  `0..7` train, `8..9` validation, and `10..11` test. No prompt moves after outcomes are observed.
- Sites: residual post at layers `7`, `14`, and `21`; target is layer `27` residual plus 32 logit
  coordinates selected by mean clean training-prompt logits only.
- Grid: for every prompt/site, steering coordinates `0/64/128/256` at magnitudes `-4/+4`, plus
  zero, training-mean, donor patch, and donor resample on eight fixed coordinates: exactly 432
  direct outcomes.
- Donors: donor prompts exclude the recipient and remain within its prompt split. Mean statistics
  use training prompts only. Patch and resample use different deterministic donors where possible.
- Context/targets: seed-fixed 32-dimensional projections of clean source state and actual source
  delta; targets are 32 projected final-hidden deltas plus 32 selected logit deltas. No downstream
  post-intervention target enters the context/intervention branch.
- Strong local baseline: each registered edit gets a separate direct execution at 5 percent of its
  source-state displacement; its scaled downstream effect is stored as the prompt-local Jacobian
  approximation. Thresholds must not be tuned using these values.
- Storage: pre-capture estimate must fit 128 MB; output is resumable sharded HDF5 with 64 MB shard
  cap and SHA-256 per shard. The tensor shards remain ignored; the manifest/metrics/provenance are
  committed.
- Acceptance: exact revision, exactly 432 outcomes, all three splits present, every direct target
  effect nonzero, and stored estimate within budget. Top-token changes are descriptive, not a gate.

Measured result from clean commit `0aa80ac`: all dataset gates passed in `33.85` seconds. The single
HDF5 shard is 412,332 bytes with SHA-256 `3cf0411b...`; 17/432 edits changed the top token. Mean
prompt-local linear-approximation MSE was `139.83`. This last value only establishes that the stored
intervention mix is nonlinear/off-local for the selected targets; H-LLM-01 remains undecided until
learned models and all registered baselines are evaluated on held-out splits.

## WM-T0-005 Preregistration

Registered before execution on 2026-07-20. This is an independent multi-seed follow-up; thresholds
must not be changed after inspecting outcomes.

- Scope: test restricted H-WM-05, H-WM-06, and H-WM-08 using a goal- and dynamics-conditioned
  PointMass JEPA. This can establish at most a shared task-workspace candidate, never equivalence to
  Anthropic J-space or a full workspace.
- Data: deterministic `MultiTaskPointMass2D` with four goals and masses `0.4/1.6`, `192`
  trajectories, `20` steps, and seeds `23`, `29`, and `37`. Goal 3 plus mass mode 1 is entirely held
  out. The seven seen combinations are split 70/15/15 percent for predictor training, calibration,
  and frozen-consumer fitting. The physical OOD set uses masses `0.25/2.4` and a disjoint seed.
- Predictor: fixed non-collapsing 12-dimensional encoder, hidden widths `32/24`, and three
  bootstrap members trained for `900` Adam steps. Held-out physical-state MSE must be no more than
  `0.50` times shuffled-action MSE.
- Uncertainty: fit the ensemble interval scale on calibration only. Held-out-composition coverage
  must be in `[0.75, 0.99]`, OOD rank AUC at least `0.65`, and ID+OOD uncertainty/error Spearman at
  least `0.30`.
- Consumers: freeze the predictor, then fit separate quadratic heads for physical dynamics, value,
  risk, calibrated uncertainty, and action selection. Every held-out-composition R2 must be at
  least `0.40`.
- Candidate: discover only at `predictor.hidden2`; at most `6/24` dimensions, at least `0.70`
  normalized Jacobian capture for every consumer, and compactness at most `0.30`.
- Controls: use 32 seed-fixed random bases and 32 local tangent bases estimated from 16 nearest
  consumer-training activations. Candidate and controls use the same conditional-donor or direct
  coordinate-swap operation. At least eight controls must match within a factor of two in density
  distance and perturbation RMS, with candidate density ratio at most `3.0`.
- Counterfactual controllability: keep physical state and action fixed while swapping from the
  held-out task context to goal 0 plus mass mode 0. Candidate coordinates must recover at least
  `0.50` of mean donor-consumer change and exceed the 95th percentile of both matched random and
  local-tangent controls.
- Selective necessity: eight-step conditional-resampling damage must exceed both matched-control
  95th percentiles and have a multistep/one-step ratio of at least `1.25` for both control families.
- Replication decision: a shared task-workspace candidate requires every gate above on at least two
  of three seeds. `workspace_found` remains false because reportability analogue and published-model
  replication are absent. Partial passes and compelling plots do not rescue the joint decision.

Measured result from clean commit `7a9e510`: zero of three seeds passed. Action-conditioning ratios
were `0.712`, `1.012`, and `1.003`, all above `0.50`. OOD uncertainty passed jointly only on seed
`29`. Every seed failed the all-consumer gate; non-dynamics held-out R2 values were predominantly
negative. Minimum candidate sensitivity capture was `0.583`, `0.561`, and `0.574`, below `0.70`.
Mean task-counterfactual recovery was `-10.18`, `-17.38`, and `-3.11`, below `+0.50`. Seed `37`
exceeded one random-control p95 despite negative recovery, demonstrating why the absolute effect
gate was required; it still failed the tangent control. Restricted H-WM-05, H-WM-06, and H-WM-08
are false for this architecture and split.

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

Measured result from clean commit `6785fb1`: all registered candidate decisions failed. The action
MSE ratio was `0.684`; OOD uncertainty AUC was `0.574`; uncertainty-head R2 was `-1.327`/`0.232`;
minimum five-consumer capture was `0.635`/`0.701`; and direct plus rollout candidate damage was below
matched-control p95 at both hidden sites. Test interval coverage (`0.887`) and error correlation
(`0.698`) passed their individual gates, but partial passes do not rescue the joint uncertainty or
workspace hypotheses. Restricted H-WM-05, H-WM-06, and H-WM-08 are false for this run.

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

## LLM-GPT2-003 Preregistration

Registered before execution on 2026-07-20. No magnitude, split, threshold, or direction-construction
rule may change after direct outcomes are inspected.

- Scope: restricted H-LLM-01, H-LLM-02, and H-LLM-03 on cached GPT-2 Medium. This is not Qwen,
  J-space discovery, semantic-feature validation, or a workspace test.
- Directions: at block `12` residual output and the final non-padding token, construct one contrast
  from two positive versus two negative calibration prompts and one from two geography versus two
  biology prompts. The eight calibration prompts are disjoint from all evaluation prompts. Normalize
  the first direction and Gram-Schmidt orthogonalize the second. Labels describe construction only.
- Evaluation data: six fixed neutral prompts; IDs `0-3` train and `4-5` are held out. Per prompt,
  execute each direction singly at `+/-0.5` and `+/-6.0`, plus all four signed two-direction
  compositions at magnitude `6.0`: exactly `72` direct outcomes in 12 intervention batches.
- Split: linear, bilinear Intervention-JEPA, and MLP predictors train only on the 32 single-direction
  outcomes from prompts `0-3`. The primary split is eight composed outcomes from held-out prompts.
  The 16 composed outcomes on train prompts are descriptive. No composition target enters training.
- Context/target: seed-11 fixed random projection of the clean source residual to 24 dimensions;
  targets are 24 evenly spaced final-residual coordinates and 24 logit coordinates chosen using
  clean train prompts only. The post-intervention target never enters context.
- Baselines: no-change, train mean, linear regression, bilinear Intervention-JEPA, trained MLP,
  prompt-local additive finite difference from `+/-0.5`, corpus-averaged additive finite difference,
  and direct addition of the two same-prompt magnitude-6 single effects.
- Nonlinearity: direct composition interaction is the MSE between the observed composed effect and
  direct addition of its two large single effects, divided by observed effect power. H-LLM-01 can
  pass only if this fraction is at least `0.05` and the MLP predictor beats the
  prompt-local additive Jacobian on the primary split.
- Causal compression: restricted H-LLM-02 passes only if bilinear Intervention-JEPA beats both
  no-change and linear regression on primary raw effect MSE.
- Composition: restricted H-LLM-03 passes only if the better bilinear-or-MLP predictor beats both
  no-change and corpus-averaged Jacobian and has direct-effect correlation at least `0.50` on the
  held-out prompt compositions.
- Resources: cached `gpt2-medium`, `local_files_only=true`, sequence length `24`, 12 intervention
  batches, 16 MB activation estimate cap, 600-second runtime guard, no downloads, and no GPU claim.

Measured result from clean commit `1e57e30`: all three hypotheses failed. On held-out prompt
compositions, direct interaction was only `0.000429` of effect power, below `0.05`; MLP MSE was
`0.725` versus prompt-local Jacobian `0.000990`. Bilinear MSE was `1.346`, worse than no-change
`0.418` and linear regression `0.775`. The best learned predictor had negative effect correlation.
On seen prompts, bilinear MSE was `0.00110` with correlation `0.9985`, so the held-out split exposed
severe prompt memorization. Restricted H-LLM-01, H-LLM-02, and H-LLM-03 are false for this run.
