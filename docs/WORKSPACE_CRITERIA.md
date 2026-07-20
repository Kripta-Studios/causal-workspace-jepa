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

## WM-T0-003 Result

The detector found its planted shared control and rejected the disjoint negative control. On the
tiny JEPA it proposed three of sixteen dimensions, capturing at least `0.886` of every fitted
consumer Jacobian's normalized sensitivity. The result nevertheless failed:

- uncertainty readout held-out R2 was `-3.639`, below the preregistered `0.8` validity floor;
- PCA caused more direct-readout damage (`1.527`) than the candidate (`1.306`);
- random rollout controls became severely off-manifold and produced unusably large damage.

Result: `shared_causal_subspace_candidate_found = false`, `workspace_found = false`. The most likely
explanation is an ordinary physical-state manifold reused by hand-added consumers, not a privileged
workspace.

## WM-T0-004 Registered Repair

The follow-up no longer projects arbitrary directions out of a recurrent rollout. It trains a
two-hidden-layer predictor, freezes it, calibrates a five-member ensemble on validation data, and
fits each downstream consumer independently. Candidate coordinates are interchanged from donors
that are nearby in the orthogonal complement. Every candidate and control patch reports its nearest
training-manifold distance and perturbation RMS; unmatched random or PCA controls cannot support a
specificity decision.

This repair still cannot establish J-space equivalence. Anthropic's J-space is a sparse nonnegative
token-aligned frame with report, directed modulation, reasoning, flexible reuse, and selective
necessity tests. `WM-T0-004` is only a narrower test for a shared causal hidden subspace.

## WM-T0-004 Result

The repair worked methodologically but returned a null scientific result:

- candidate patch density ratios were `1.029` and `1.018`, and `63/64` plus `64/64` random controls
  met the registered density/magnitude match;
- five-dimensional candidates captured only `0.635` and `0.701` minimum consumer sensitivity,
  below `0.75`;
- uncertainty-head R2 was `-1.327` and `0.232`, below `0.50`;
- candidate direct damage (`3.652`, `1.239`) was below random-control p95 (`9.252`, `4.369`);
- multistep/one-step ratios were high (`6.03`, `1.95`), but multistep damage still remained below
  matched-control p95, so the accumulation was not specific;
- PCA controls had much larger perturbation RMS and were correctly rejected as unmatched.

Result: no shared causal candidate and no workspace. The strongest positive finding is that
conditional donor resampling supplies usable empirical-manifold controls; it does not reveal a
privileged subspace in this model.

## WM-T0-005 Registered Follow-Up

The preregistered follow-up adds the missing pressure for flexible reuse: four goals crossed with
two dynamics modes, one held-out goal/dynamics composition, and three independent seeds. It tests a
single hidden-2 candidate against both isotropic random bases and locally estimated tangent bases.
In addition to conditional resampling and multistep necessity, it performs a task counterfactual:
state and action stay fixed while candidate coordinates are swapped toward a donor goal/dynamics
context. A candidate must recover donor consumer outputs by at least 50 percent and beat matched
controls. Passing on two seeds would establish only a shared task-workspace candidate. The full
workspace decision is fixed to false for this CPU study.

## WM-T0-005 Result

Zero of three seeds passed. The predictor did not make actions indispensable on the held-out task,
the fitted value/risk/uncertainty/action heads did not generalize compositionally, and no six-of-24
candidate captured the required 70 percent sensitivity from every consumer. Candidate task swaps
moved consumer outputs farther from the donor on average on all seeds. Seed 37 looked selective
against random controls in isolation, but recovery was still negative and local-tangent controls
were stronger. This is not controllability or selective necessity. Result: no shared task-workspace
candidate and no workspace.
