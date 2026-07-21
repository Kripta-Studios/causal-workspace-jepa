# World Model Track

Status: `SMOKE_VALIDATED`.

Initial implementation order:

1. deterministic Tier 0 environments: implemented;
2. tiny NumPy action-conditioned JEPA: implemented;
3. no-action and shuffled-action controls: implemented;
4. named activations: implemented for tiny JEPA;
5. interventions and matched controls: implemented;
6. planning smoke loop: implemented;
7. workspace criteria with null-result-safe controls: implemented and run with a documented null.

The former LeWorldModel placeholder is replaced by a typed, source-traceable small reproduction and
adapter. `WM-LEWM-001` ran unchanged from clean commit `4dbc388` against official revision
`8edfeb3...`. Other published adapters remain placeholders.

The reproduction retains the official end-to-end pixel encoder, action embedder, AdaLN-zero
autoregressive predictor, next-embedding MSE, and SIGReg. It deliberately scales to 20x20
PixelTinyMaze and 32 latent dimensions. The audit includes layerwise linear/nonlinear probes,
paired-action subspace replacement, norm-matched random controls, action-module suppression,
latent planning, closed-loop execution, ensemble uncertainty, five frozen consumers, and a
restricted necessity/sufficiency/faithfulness graph. It is not equivalent to a released benchmark
checkpoint.

`WM-LEWM-001` is a mixed/negative result. All seeds pass the reproduction gate, and the internal
subspace intervention selectively changes latent trajectories, planning costs, and selected actions
on two seeds. However, decoded donor recovery and the complete restricted circuit gate pass only
seed 107; the aggregate graph is rejected. Clean planning succeeds on just one of 12 cases per seed,
so this is not a successful control benchmark. All workspace candidates fail despite valid planted
controls and valid five-consumer readouts.

`WM-POPULATION-JACOBIAN-001` is implemented and prospectively registered against the three
unchanged checkpoint files. It executes every ordered categorical first-action replacement over
one and four autoregressive steps, estimates population derivatives only from 12 train-goal
families, and evaluates on five disjoint validation goals. Exact line integration checks autograd;
a train-only physical decoder and contragredient gauge transform prevent raw latent Euclidean MSE
from deciding the primary claim. The five test goals are untouched, and no result exists yet.

Validated CPU smoke:

- `WM-T0-001` ran on PointMass2D with code commit `0cab19a6c39c98b59f1a2172eb11a64ec5a566a4`.
- Conditioned latent MSE: `1.09e-09`; mean baseline: `0.249`; no-action: `0.135`; shuffled-action: `0.153`.
- `WM-T0-002` found action displacement decodability and, after correction, narrow causal
  specificity for the explicit action-input pathway; it found no workspace/J-space-like candidate.
- Adversarial review retained the displacement result and superseded the original action-patch
  metric. A clean run on commit `315d8cf` performs a replayable `predictor.input` intervention with
  L2 norm-matched controls: donor recovery `1.0`, replay error `0.0`, latent control `-12.784`, and
  random-action control `-0.316`.
- A multi-consumer audit now fits frozen readouts for dynamics prediction, value, risk, an OOD
  uncertainty proxy, and action selection. Known shared/disjoint systems validate the detector;
  random and PCA controls test whether a JEPA candidate is privileged beyond generic state variance.
- `WM-T0-003` returned a null: the detector controls pass, but uncertainty R2 is negative, PCA is
  more damaging than the candidate, and random rollout projections are off-manifold. The candidate
  is not promoted.
- `WM-T0-004` ran from clean commit `6785fb1`. Conditional donor resampling repaired the old
  off-manifold control failure, but the action gate, OOD AUC, uncertainty consumers, sensitivity
  candidates, and specificity over matched controls failed. No shared candidate was accepted.
- `WM-T0-005` ran from clean commit `7a9e510` in `57.51` seconds. Zero of three seeds passed. Weak
  action dependence, failed held-out consumer transfer, insufficient shared sensitivity, negative
  task-counterfactual recovery, and local-tangent controls reject the candidate.
- This validates deterministic execution and action-conditioning plumbing. It does not establish a
  learned workspace.
