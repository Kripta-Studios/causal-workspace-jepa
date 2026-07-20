# World Model Track

Status: `SMOKE_VALIDATED`.

Initial implementation order:

1. deterministic Tier 0 environments: implemented;
2. tiny NumPy action-conditioned JEPA: implemented;
3. no-action and shuffled-action controls: implemented;
4. named activations: implemented for tiny JEPA;
5. interventions and matched controls: implemented;
6. planning smoke loop: implemented;
7. workspace criteria with null-result-safe controls: implemented; clean `WM-T0-003` run pending.

Published world-model adapters are `BLOCKED_RESOURCE` on the CPU VPS.

Validated CPU smoke:

- `WM-T0-001` ran on PointMass2D with code commit `0cab19a6c39c98b59f1a2172eb11a64ec5a566a4`.
- Conditioned latent MSE: `1.09e-09`; mean baseline: `0.249`; no-action: `0.135`; shuffled-action: `0.153`.
- `WM-T0-002` found action displacement decodability and action-coordinate causal specificity, but no workspace/J-space-like candidate.
- Adversarial review retained the displacement result and superseded the original action-patch
  metric. A clean run on commit `315d8cf` performs a replayable `predictor.input` intervention with
  L2 norm-matched controls: donor recovery `1.0`, replay error `0.0`, latent control `-12.784`, and
  random-action control `-0.316`.
- A multi-consumer audit now fits frozen readouts for dynamics prediction, value, risk, an OOD
  uncertainty proxy, and action selection. Known shared/disjoint systems validate the detector;
  random and PCA controls test whether a JEPA candidate is privileged beyond generic state variance.
- This validates deterministic execution and action-conditioning plumbing. It does not establish a
  learned workspace.
