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
`8edfeb3...`. The EB-JEPA placeholder is now replaced by a typed adapter to the pinned official
source contract at `966e61e...`; published checkpoints/training outcomes remain unexecuted.

Primary-source review now prioritizes official EB-JEPA over further small-model extrapolation. Its
action-conditioned Two Rooms example is Apache-2.0, designed for a single GPU, includes recurrent
prediction plus CEM/MPPI, and reports `97 +/- 2%` planning success. The next boundary is to reproduce
that competence locally, then instrument action, recurrent gates, hidden state, latent prediction,
planning cost, action choice, and closed-loop success. Publication performance is not local evidence.

Source inspection corrects the predictor description: the official action-conditioned configuration
uses an Impala encoder with a 512-dimensional output, an identity two-dimensional action encoder,
and a **one-layer** `torch.nn.GRU`, followed by the encoder's final normalization. The adapter
decomposes the exact GRU equations into reset, update, candidate, pre-normalization hidden, and
post-normalization hidden sites. Offline contract tests pass, and the clean retained
`WM-EBJEPA-CONTRACT-001` run from `979c2d6` reconstructed native recurrence within `4.768e-7`.
A single-coordinate update-gate edit had zero same-step collateral error outside that coordinate and
changed the downstream latent by L2 `2.1505`. These are engineering gates over random weights. The
exact isolated Python 3.12/Torch 2.6+cu126 runtime detects the SM120 GPU but cannot execute matmul,
Conv2D, or GRU because the wheel contains architectures only through SM90. A matched Python
3.12/Torch 2.10+cu128 runtime includes SM120 and passes all three. Local training must therefore use
the latter as a declared compatibility deviation. The clean `WM-EBJEPA-RUNTIME-001` run from
`15d88ce` passed all eight frozen architecture/runtime/kernel gates. Remaining upstream dependencies
and planner competence are still pending.

The compatible Two Rooms dependency closure is now frozen and installed without allowing the
upstream package resolver to replace Torch. Source/import audit found undeclared runtime
requirements for scipy, pandas, and PyYAML. A manual eight-sample official training loop completed
on GPU. That exploratory run also exposed a planning-control boundary: official CEM applies
`max_norms`, official MPPI does not, and `DotWall.step` does not enforce its action space. The first
random-weight MPPI action had norm about `3.69` under a configured `2.45` maximum. A 32-seed
post-discovery confirmation and a clean integration smoke are preregistered; neither is yet retained
evidence. Planning reproduction must preserve original MPPI and add a separately labeled corrected
condition before mechanistic interpretation.

The first clean integration run from `f0e7a3e` passed its original gates but is superseded: NumPy
and Torch were seeded while Python `random`, used by the generator, was not, and no independent
replay was required. Corrected `WM-EBJEPA-INTEGRATION-002` retains the exact same model/data/planner
settings and requires exact cross-process hashes under deterministic CUDA settings.
V2 ran from clean `9a18008`; all 12 gates pass and both subprocesses match fingerprint
`16650872...234a1`. The minimal BF16 loss is `9.6593`, exact checkpoint replay error is zero, and
peak reserved memory is about 148 MiB. This makes full training executable but supplies no
competence or mechanism result.

The frozen planner confirmation ran from clean `da30443`. CEM respected the `2.45` norm in all 32
seeds, while MPPI violated it in all 32 (median maximum `6.4485`, maximum `8.3018`). The pinned MPPI
method never references `max_norms`, and `DotWall.step` does not enforce its declared action space.
This confirms a software/configuration defect. It does not establish that published planning
competence disappears; the next reproduction must report original and constraint-corrected MPPI
side by side before the planner is eligible for mechanistic claims.

A separate `ConstrainedMPPIPlanner` is implemented without modifying the official checkout. It
projects candidate actions before cost evaluation and the final returned action. Its frozen
32-seed smoke ran from clean `f58308a`: exact official equivalence holds with bounds disabled, and
bounded cost/return violations are both zero. The maximum observed cost-input and returned-action
norms are `2.45000005` and `2.44999909`. This correction is eligible for a competence comparison,
but supplies no learned-model, circuit, or workspace evidence.

The retained official-training capacity profile passes from clean `fed920e`. All eager batches
through the official 384 execute finite updates, and batch 384 peaks at 5.82 GB reserved, below the
preregistered 10-GB ceiling. The configured `torch.compile(jepa)` call returns an optimized wrapper,
but two real `unroll` updates capture zero Dynamo frames and zero graphs. Training can preserve the
flag for configuration fidelity, but cannot claim compiled execution on this path.

The official planner configuration has a second independent mismatch. Both MPPI and CEM YAMLs
declare `var_scale=1.5`; CEM consumes it, whereas MPPI expects `max_std`, absorbs the unknown key in
`**kwargs`, and runs at default `2.0`. The clean `4f0cc80` confirmation passes all six source/runtime
gates. Planning reports must separate official 2.0, bound-corrected 2.0, and any intention-corrected
1.5 arm.

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

`WM-POPULATION-JACOBIAN-001` ran against the three unchanged checkpoints from clean `89b2e14` and
is rejected on its numerical gate. The train-only physical decoder and gauge transform were stable,
but 12-node line integration badly underresolved stiff, cancelling recurrent derivatives. The
global mean reduced raw/decoded MSE largely by shrinking effects toward zero, did not improve
correlation, and failed action-column specificity. Averaging the four valid action-vertex
Jacobians within each context provisionally beat local in all seeds and both horizons, but the
numerical rejection prevents accepting that result. The five test goals remain untouched.

The naive vertex-mean continuation is stopped because validation correlation and scalar-shrinkage
controls undermine its MSE-only interpretation. `WM-ACTION-PATH-CALIBRATION-001` is instead locked
to the exposed validation goals. It profiles decoded derivative arc length versus net action effect,
local finite-chord error, 128/256-node composite-quadrature convergence, speed concentration, and
within-action-pair permutation nulls. It is calibration only; scientific test thresholds are not yet
registered and the five test goals remain protected.

Adversarial review identified that cancellation and normalized local error both divide by the
decoded net-effect norm. This can create association for small effects even within action-pair
strata. A proposed derived audit was rejected before commit because v2 lacks scalar path-length
refinement and unclamped direct norms, and only two chords are available per action pair. Separate
conditional permutations cannot supply the missing joint support. High-resolution convergence is
therefore only a numerical/vector result; the route is closed without accessing test goals.

Calibration v1 ran from clean `eb943a5` in `72.59` seconds. Horizon one converged across seeds, but
horizon four remained underresolved for seeds 101/103 even at 256 nodes. Cancellation/local-error
Spearman exceeded the within-action-pair null p95 in all three horizon-four samples only narrowly,
and median recurrence amplification occurred strongly in just seed 103. No evidence is accepted;
v2 is archived as numerical calibration and no protected-test run follows from this family.

Calibration v2 completed from clean `288f663` in `19,176.20` seconds at 512/1024 nodes. Horizon-four
maximum integration/refinement errors were `.01291/.02605`, `.03394/.03141`, and
`.001393/.000250` for seeds 101/103/107; only seed 107 is small at both resolutions. The corresponding
Spearman-minus-null-p95 margins were `.0454/.0283/.0165`, and horizon-four/horizon-one median
cancellation ratios were `1.054/2.818/.983`. The artifact records no decisions and no protected-test
access. It narrows numerical behavior but cannot identify action-path geometry under the frozen
design confound.

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
