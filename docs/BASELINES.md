# Baselines

Status: `IMPLEMENTED_UNVALIDATED` for CPU methods; published/GPU baselines remain blocked.

## World Model

- oracle/raw-state dynamics;
- simple autoencoder dynamics;
- no-action JEPA;
- shuffled-action JEPA;
- frozen random encoder;
- linear latent dynamics;
- LeWorldModel;
- JEPA-WMs;
- Delta-JEPA objective;
- C-JEPA;
- local Jacobian;
- probe-only interpretation;
- matched random-subspace intervention.

`WM-T0-003` implements equal-dimensional random subspaces, high-variance PCA, a planted shared
subspace positive control, and disjoint-consumer negative control. Published model baselines remain
`BLOCKED_RESOURCE`.

## LLM Meta-Model

- no-change predictor;
- mean intervention effect;
- linear regression;
- bilinear predictor;
- local Jacobian;
- corpus-averaged Jacobian Lens;
- MLP predictor;
- autoregressive layer predictor;
- nearest-neighbor intervention retrieval;
- sparse-feature linear transport.

Success must not be declared from beating only a trivial baseline.
