# Workspace Criteria

Status: `SMOKE_VALIDATED_NULL`.

A workspace candidate is an experimental hypothesis, not an assumption.

Required properties:

1. decodability/reportability analogue;
2. controllability;
3. causal mediation;
4. flexible reuse by dynamics, value, risk, uncertainty, and action selection;
5. broadcast centrality controlled for norm and dimensionality;
6. selective necessity;
7. compactness;
8. depth/horizon evolution.

Controls:

- random subspaces;
- high-variance principal components;
- probe-optimal but noncausal directions;
- norm-matched directions;
- shuffled labels;
- irrelevant donor examples;
- equal-dimensional ablations outside the candidate subspace.

A null result is acceptable and must be reported.

## Current Tier 0 Result

`WM-T0-002` evaluated the tiny JEPA action-coordinate candidate.

- Candidate: `predictor.input` action coordinates.
- Positive evidence: action coordinates causally and selectively affect future latent prediction and planner cost.
- Negative evidence: the model has only dynamics and planner-cost consumers; it has no value, risk, uncertainty, or reportability consumers.
- Result: `workspace_found = false`.

Conclusion: no JEPA analogue of Anthropic J-space has been discovered.

The corrected action-input result was rerun from clean commit `315d8cf`. It establishes the explicit
action-input pathway only; exact recovery is expected for this one-layer linear predictor and does
not promote the pathway to a workspace candidate.

## Operational Detector

`WM-T0-003` first validates discovery against two known systems:

- positive: five consumers share a planted three-dimensional causal subspace;
- negative: five consumers use disjoint private subspaces.

For a JEPA candidate, every consumer Jacobian is converted to a trace-normalized sensitivity Gram
matrix. The candidate is the smallest eigensubspace that captures the preregistered fraction for
every consumer, not merely the average consumer. Direct projection-out effects are then compared
with 32 equal-dimensional random subspaces and high-variance PCA.

Passing this detector establishes only a compact shared causal-sensitivity candidate. The full
workspace decision still requires controllability, flexible reuse, selective necessity,
depth/horizon evolution, and held-out generalization.
