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
