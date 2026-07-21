# Preregistered Hypotheses

Status: `ACTIVE` research program; bounded required suite `SMOKE_VALIDATED`. Registered studies
tested restricted forms of H-WM-01/02/03/05/06/08 and H-LLM-01/02/03/06. The corrected Qwen
H-LLM-01 result is `WITHDRAWN`: an exact FP32 JVP and quadratic Taylor model decisively beat the
legacy conditional bottleneck. Its original bounded H-LLM-02/03 gates remain descriptive and do
not imply advantage over exact local transport. H-LLM-06 fails, the LeWorldModel circuit gate fails
replication, and all workspace decisions remain false. These do not constitute broad scientific
validation.

The consolidated scientific account is `papers/causal_workspace_jepa.tex`. Its present supported
boundary is: direct Qwen donor-patch behavior exists; exact/quadratic/population transports have
endpoint-dependent fidelity; the registered threshold-and-four-layer-grid onset rules fail across
the studied element/state/country relations; one target-encoder Intervention-JEPA variant is
negative; and no tested JEPA circuit/workspace proxy meets its acceptance gate. Decoded recurrent
action-path cancellation remains validation-only calibration and does not decide a numbered
hypothesis.

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

## WM-EBJEPA-INTEGRATION-001 Engineering Preregistration

Registered on 2026-07-21 before retaining an official Two Rooms integration result. Earlier manual
smokes established that the path can import and complete one tiny training loop; they are disclosed
exploration, not the retained result. This experiment tests no numbered world-model hypothesis.

- Source: clean official `facebookresearch/eb_jepa` revision
  `966e61e9285b3a876f49b9774e9720d9a99a7925`.
- Runtime: isolated Python 3.12.13, Torch 2.10.0+cu128 compatibility deviation with `sm_120`, plus
  the committed exact dependency closure
  `configs/resource/eb_jepa_two_rooms_py312_sm120.lock.txt`. The lock cannot replace Torch.
- Dataset: seed 29, eight generated training samples, four validation samples, batch size two,
  nine frames, no workers. Expected batch shapes are frozen in the config evaluator.
- Model step: official 65x65 Impala encoder, one-layer 512-dimensional GRU, inverse-dynamics and
  variance/covariance/similarity losses, four-step autoregressive BF16 unroll, backward, and AdamW.
- Gates: pinned clean source; Python/Torch/CUDA/SM120 identity; exact dataset/output shapes and
  official encoder/predictor parameter counts; finite loss/gradients; a changed parameter; exact
  checkpoint restore; finite integrated MPPI output; peak reserved memory below 12 GB.
- Packaging audit: `scipy`, `pandas`, and `PyYAML` are required by this path but absent from the
  upstream dependency declaration (`ruamel.yaml` does not provide `import yaml`). This is recorded
  independently of the integration gates.
- Boundary: one tiny batch and a random-weight planning call establish only engineering
  availability. They do not establish learned dynamics, planning competence, a circuit, or a
  workspace.

First clean outcome from `f0e7a3e`: the recorded gates passed, but the run is
`SUPERSEDED_NONDETERMINISTIC`. The probe seeded NumPy and Torch but not Python's `random`, which is
used by the official generator, and it lacked an independent-process replay gate. This was detected
by comparing the clean loss `11.5872` with the earlier same-seed exploratory loss `10.0157`. The
artifact and clean provenance are preserved, but no availability conclusion is accepted from it.

`WM-EBJEPA-INTEGRATION-002` is amended and registered before rerun. It keeps every source, dataset,
model, optimizer, planner, and memory setting above, adds `random.seed(29)`, deterministic Torch and
CUDNN settings plus `CUBLAS_WORKSPACE_CONFIG=:4096:8`, and executes the standalone probe twice in
independent Python processes. The retained gate requires an exact SHA-256 match over the generated
batch, post-update model state, loss, parameter delta, planned action, and planner losses. Its output
is `artifacts/metrics/eb_jepa_two_rooms_integration_v2.json`; v1 is never overwritten.

Final v2 result from clean `9a18008`: `SMOKE_VALIDATED`, all 12 gates pass. The two independent
subprocesses exactly match fingerprint `16650872debe76818197183888877d90e76ed27408f67c569cf3d9df478234a1`.
The official four-step BF16 loss is `9.659334`, parameter delta is `.001000002`, checkpoint restore
error is zero, and peak reserved memory is `155,189,248` bytes. No numbered hypothesis is decided.

## WM-EBJEPA-PLANNER-CONSTRAINT-001 Post-discovery Confirmation

Registered on 2026-07-21 after a manual random-weight official-model smoke returned an MPPI action
of approximate norm `3.69` despite configured `max_norms: [2.45]`, and after source inspection found
that official CEM uses `max_norms` while official MPPI does not. This is explicitly a post-discovery
engineering confirmation, not a prospective scientific hypothesis.

- Frozen comparison: official `CEMPlanner` versus `MPPIPlanner` from revision `966e61e...`, seeds
  0--31, identical deterministic cumulative-action unroll, terminal-x objective, three iterations,
  128 samples, 16 elites, plan length three, action width two, and configured maximum norm `2.45`.
- Source gates: CEM `plan` mentions/enforces `max_norms`; MPPI `plan` does not; `DotWall.step` passes
  the action to `_calculate_next_position` without checking `action_space`.
- Dynamic gates: CEM violates the norm in zero seeds and has maximum norm at most `2.450001`; MPPI
  violates it in at least 31/32 seeds, with median maximum norm at least `3.45`.
- No threshold will be changed after the retained run. Failure to reproduce freezes the observation
  as exploratory. Passing establishes an implementation/configuration defect only.
- Scientific consequence: official planning competence must be compared under original and
  constraint-corrected MPPI before any recurrent/action circuit or workspace interpretation.

Final post-discovery confirmation from clean `da30443`: `CONFIRMED_UPSTREAM_DEFECT`, all seven
gates pass. CEM violates in `0/32` seeds with maximum observed norm `2.347382`; MPPI violates in
`32/32`, with median `6.448531` and maximum `8.301845`. Source gates confirm the CEM/MPPI
enforcement asymmetry and absent `DotWall.step` action-space check. No numbered scientific
hypothesis is decided.

## WM-EBJEPA-MPPI-CORRECTION-001 Engineering Preregistration

Registered on 2026-07-21 after confirming the official constraint defect and before retaining the
correction. The official planner remains unchanged. The repository adds a separately named
`ConstrainedMPPIPlanner` solely to support a controlled reproduction arm.

- Source/control: pinned official MPPI at `966e61e...`; corrected implementation uses the same
  sampling, elite weighting, terminal selection, and return schema.
- Seeds/design: seeds 0--31, deterministic cumulative-action unroll, terminal-x objective, three
  iterations, 128 samples, 16 elites, plan length three, action width two.
- Equivalence gates: with `max_norms=None`, official and corrected returned actions and per-iteration
  minimum losses must each match within `1e-6` in every seed.
- Constraint gates: with `max_norms=[2.45]` over dimensions `[0,1]`, both every candidate action
  passed into the cost function and every returned action must have norm at most `2.450001`; zero
  of 32 seeds may violate either gate.
- The implementation projects sampled candidates before model evaluation and projects the returned
  action again after any exploration noise. It does not silently replace official MPPI.
- Passing is Availability-level engineering evidence only. Planning success under original versus
  corrected MPPI remains a separate multi-seed reproduction, and no model mechanism is tested here.

Final retained validation from clean `f58308a`: `SMOKE_VALIDATED`, all five gates pass across 32
seeds. With the constraint disabled, official and corrected actions and losses agree exactly
(maximum absolute differences `0.0`). With `max_norms=[2.45]`, there are zero cost-input and zero
returned-action violations; their observed maxima are `2.45000005` and `2.44999909`. No numbered
scientific hypothesis is decided.

## WM-EBJEPA-TRAIN-RESOURCE-001 Engineering Preregistration

Registered on 2026-07-21 before measuring official-size batches. This is a capacity/compiler
diagnostic and decides no numbered hypothesis.

- Pinned source/runtime: clean EB-JEPA `966e61e...`, Python 3.12, Torch 2.10.0+cu128, SM120.
- Exact executed training semantics: 65x65 Two Rooms trajectories of length 17, horizon eight,
  Impala encoder, one-layer 512-dimensional GRU, VC/IDM/time regularizer coefficients 8/16/12/1,
  BF16 autocast, GradScaler, clipping, AdamW, and the detached XY-probe update.
- Isolated eager processes use batch sizes 64, 128, 256, and the official default 384; each runs two
  optimizer steps and records generation time, step throughput, loss/gradient finiteness, parameter
  change, and allocated/reserved CUDA peaks. A 10,000,000,000-byte reserved-memory ceiling defines
  the locally recommended eager batch.
- One isolated batch-64 process exercises the official default `torch.compile` call and records
  Dynamo frame/graph counters around the actual custom `unroll` entrypoint. A successful wrapper
  with zero captured graphs is retained as an ineffective-compile result; success or a structured
  compiler failure is diagnostic data, not a gate that may be hidden by eager fallback.
- Acceptance requires structured results for every job, the pinned clean source/runtime, a finite
  updating eager batch 64, and valid memory/update fields for every successful job. It does not
  require batch 384 or compilation to succeed.
- Any later training configuration must disclose deviations from batch 384, 16 workers, and
  `compile=true`; this profile alone supplies no competence or mechanism evidence.

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

## LLM-IJEPA-001 Meta-Model Preregistration

Registered on 2026-07-21 after freezing `LLM-INTDATA-001` and before fitting any predictor.

- Data: the checksum-verified 432-example Qwen dataset is immutable. Training uses prompt split 0
  while excluding steering feature 256 and all resample edits. Validation uses prompt split 1 with
  the same exclusions. Primary evaluation is every edit on prompt split 2. Separate stress splits
  hold out feature 256 and the resample operation on training prompts.
- Intervention-JEPA: separate 64-unit context/intervention encoders, multiplicative interaction, a
  24-dimensional meta-state, and a downstream effect predictor. Train seeds `61/67/71` for at most
  1,000 AdamW steps with validation early stopping. Checkpoint reload must be exactly predictive.
- Required baselines: no change, training mean effect, linear and bilinear regression, trained MLP,
  nearest-neighbor retrieval, per-example direct 5-percent local Jacobian, corpus transport learned
  from training local probes, and a 16-component sparse-dictionary linear transport.
- H-LLM-01: a seed passes nonlinear advantage only if primary local-Jacobian error is at least 5
  percent of direct effect power and that seed's Intervention-JEPA primary MSE is lower than the
  per-example local Jacobian. The replicated decision requires two of three seeds.
- H-LLM-02: a seed passes causal compression only if primary effect correlation is at least `0.50`
  and its MSE is below no-change, mean-effect, linear, bilinear, and MLP. The replicated decision
  requires two of three seeds.
- H-LLM-03: a seed passes operation generalization only if resample-holdout correlation is at least
  `0.50` and MSE is below no-change, linear regression, and corpus Jacobian. Two seeds are required.
- H-LLM-06 direct verification: rank coordinates `0/64/128/256` at layer 14 from model predictions
  on four entirely new prompts. Execute every predicted edit directly on Qwen. The candidate passes
  only if precision@1 is one and its observed mean effect exceeds matched magnitude probe-ranked and
  deterministic random coordinate controls.
- Circuit boundary: the emitted JSON/GraphML graph is a directly verified ranked-coordinate
  candidate only. It is not a reconstructed circuit without necessity, sufficiency, edge
  faithfulness, minimality, and broader held-out tests.
- Acceptance of the run itself requires finite held-out metrics, exact checkpoint replay, and direct
  execution of all 16 verification predictions. Hypotheses may validly fail.

Original measured result, without threshold changes: all seeds `61/67/71` passed the registered H-LLM-01,
H-LLM-02, and H-LLM-03 gates. Primary Intervention-JEPA MSE/correlation were `3.9227`/`0.6770`,
compared with local-Jacobian MSE `116.1557` and no-change MSE `7.2429`. Feature-holdout
MSE/correlation were `8.4928`/`0.5349`. Resample-holdout MSE/correlation were `2.1405`/`0.6802`;
nearest-neighbor was slightly better on MSE (`2.0946`) even though it was not part of the registered
H-LLM-03 gate. H-LLM-06 failed: predicted top coordinate 128 did not match observed coordinate 0,
precision@1 was `0`, and the predicted candidate's observed effect did not exceed the deterministic
random control. The candidate graph is `REJECTED`; no circuit or workspace is inferred.

Scientific correction registered 2026-07-21: the stored “local Jacobian” is a one-sided 5-percent
secant executed in bfloat16. Its error can be dominated by quantization/numerical noise, so it does
not support the claimed nonlinear advantage. The model named `NeuralInterventionJEPA` in this run is
a supervised conditional bottleneck predictor; it has no target encoder, stop-gradient target, EMA,
or anti-collapse JEPA objective. Until `LLM-QWEN-JVP-AUDIT-001` resolves the baseline, H-LLM-01 is
`UNDER_REAUDIT`, not a positive result. This correction changes no observed values or old thresholds.

## LLM-QWEN-JVP-AUDIT-001 Exact-Derivative Preregistration

Registered on 2026-07-21 before any FP32 re-execution. The frozen 432-record intervention grid,
prompt/feature/operation splits, target coordinates, model revision, and predictor seeds remain
unchanged. This audit can retain or withdraw the old restricted H-LLM-01 result; it cannot discover
a circuit, behavior mechanism, or workspace.

- Execution: pinned `Qwen/Qwen3-0.6B` revision `c1899de...`, float32, eager attention, TF32 disabled,
  final-token residual edits at layers `7/14/21`, and the original layer-27 projected-hidden plus
  selected-logit targets. Every semantic intervention is directly re-executed in FP32.
- Exact local baseline: parameterize the complete source edit as `h(alpha)=h+alpha*delta` and compute
  `dF/dalpha` at zero through autograd. This is the directional JVP for the actual finite edit, not a
  one-sided secant. Symmetric central differences use the fixed epsilon sweep
  `1/64,1/32,1/16,1/8,1/4,1/2`; a directional quadratic Taylor baseline uses the second central
  difference at `1/8`.
- Numerical-validity gates: clean FP32 replay max error at most `1e-6`; semantic intervention versus
  dense `alpha=1` reconstruction max error at most `1e-5`; at epsilon `1/8`, median JVP/central row
  relative error at most `0.05` and p95 at most `0.15`; median central agreement between `1/16` and
  `1/8` at most `0.10`; and JVP/direct norm-ratio above 10 on at most 5 percent of records. Failure
  rejects the audit rather than deciding H-LLM-01.
- Precision and duplication controls: FP32 direct targets replace BF16 targets for all refits and
  scores. BF16/FP32 drift is reported. Raw primary scores are accompanied by a deterministic
  semantic-deduplicated view because the frozen grid's patch and resample donors can coincide.
- Predictor controls: refit the legacy supervised conditional bottleneck at seeds `61/67/71`
  against no-change, mean-effect, linear, bilinear, MLP, nearest-neighbor, exact-JVP, quadratic,
  and old BF16-secant baselines. Do not describe this legacy predictor as a genuine target-encoder
  JEPA.
- Finite-amplitude nonlinearity gate: exact-JVP normalized MSE must be at least `0.10` on both raw
  and deduplicated primary sets; quadratic normalized MSE must remain at least `0.05`; and at least
  three intervention operation classes must individually have JVP normalized MSE at least `0.10`.
- Corrected H-LLM-01 retention: the numerical audit and finite-amplitude nonlinearity gates must
  pass, and at least two of three predictor seeds must (i) beat exact JVP and quadratic Taylor by at
  least 10 percent raw MSE, (ii) beat exact JVP by at least 10 percent deduplicated MSE, and (iii)
  attain raw effect correlation at least `0.60`. Otherwise the original restricted H-LLM-01 result
  is `WITHDRAWN`; failure is a valid negative result and thresholds will not be retuned.

Measured v1 result from clean commit `686368e`: the audit is `REJECTED` on a numerical gate, so it does not
decide H-LLM-01. Five of six derivative gates passed. Exact JVP agreed strongly with symmetric
central differences (median relative error `0.000249`, p95 `0.00381`), but the dense `alpha=1`
downstream endpoint differed from the semantic intervention by maximum absolute `1.335e-4`, above
the preregistered `1e-5` gate. Post-result diagnosis localized this to at most `4.768e-7` float32
source-endpoint cancellation for replacement edits; that diagnosis cannot retroactively pass v1.
The preliminary scores strongly reverse the old ranking—exact-JVP MSE `0.6143`, quadratic MSE
`0.07870`, and conditional-bottleneck MSE `3.1899` raw—but remain audit evidence pending a separately
preregistered source-semantic validation. The v1 JSON's mechanically emitted `WITHDRAWN` disposition
is superseded by the preregistered rule that a numerical-gate failure rejects the audit; the runner
will be corrected prospectively. H-LLM-01 stays `UNDER_REAUDIT`.

## LLM-QWEN-JVP-AUDIT-002 Source-Semantic Confirmatory Preregistration

Registered on 2026-07-21 after fully disclosing the rejected v1 measurements and the no-write
source-cancellation diagnosis. This is explicitly post-diagnostic, not a blind replication. V1's
threshold and rejection remain immutable.

- Frozen components: model/revision, FP32 eager execution, TF32 setting, 432 intervention records,
  prompts/splits/donors, sites, target projections/logits, epsilon sweep, predictor architecture,
  predictor seeds, derivative-convergence thresholds, finite-amplitude nonlinearity thresholds,
  semantic deduplication, and corrected H-LLM-01 retention rule are identical to v1.
- Replaced invalid gate: v1 compared downstream outputs from direct feature replacement with the
  algebraic endpoint `h + (edited-h)` at an absolute `1e-5`. Float32 subtraction/addition need not
  reproduce a replacement bitwise, and downstream amplification makes that an invalid semantic
  identity gate.
- Direct semantic gate: capture the source activation after every ordinary adapter intervention and
  require maximum absolute difference from the explicitly constructed edited source to be exactly
  zero (`0.0`). This directly tests intervention implementation before downstream propagation.
- Direction endpoint gate: require every `h + (edited-h)` source discrepancy to be no larger than
  `2 * eps_float32 * max(1, |h|_max, |edited|_max)`, a scale-aware two-rounding bound computed per
  record. No observed downstream value determines this bound.
- Descriptive endpoint control: still report absolute and effect-normalized downstream endpoint
  discrepancy, but do not use it as a semantic gate once the exact source edit and roundoff-bounded
  direction have separately passed.
- Disposition: numerical validity requires all source, convergence, replay, and norm gates. If it
  fails, emit `UNRESOLVED_NUMERICAL_REJECT`. If it passes, apply v1's unchanged nonlinearity and
  two-of-three learned-predictor rules: `RETAINED` only on a full pass, otherwise `WITHDRAWN`.
- Evidence boundary: v2 can correct one restricted H-LLM-01 result on the selected projected targets.
  It cannot validate H-LLM-02/03 broadly, a genuine JEPA architecture, behavior prediction, a Qwen
  circuit, or a workspace.

Measured v2 result from clean commit `a779ff6`: all seven numerical gates passed. Direct source
semantics were exact (`0.0` max error); source direction endpoint error was at most `4.768e-7` with
zero tolerance violations; exact-JVP/central median and p95 relative errors were `0.000249` and
`0.00381`. The unchanged finite-amplitude nonlinearity gate failed: JVP normalized MSE was `0.08948`
raw and `0.09069` deduplicated, quadratic normalized MSE was `0.01146`, and zero operation classes
met the required joint rule. Zero of three learned seeds beat exact JVP or quadratic Taylor. Raw MSE
was exact JVP `0.6143`, quadratic `0.07870`, conditional bottleneck `3.1899`, and historical BF16
secant `120.8994`. Therefore the prior restricted H-LLM-01 result is `WITHDRAWN`. This is a negative
Specificity result: the tested finite edits are largely local and captured even better at second
order; it is not evidence against nonlinear meta-models on other tasks or behavior-changing edits.

## LLM-CAPITAL-PATCH-001 Behavior-Dataset Preregistration

Registered on 2026-07-21 before any forward pass on the final entity suite. Earlier engineering
calibration used only Japan, Canada, China, and Kenya to select layer 21; those four countries are
excluded from every final split and result.

- Task: 36 repository-fixed country/capital facts using the prompt `The capital of {country} is`.
  Tokenizer-only inspection, before model execution, requires each leading-space capital answer to
  be one unique Qwen token. No entity is selected or removed using model correctness or patch effect.
- Entity split: seed-211 fixed 24 train, 6 validation, and 6 test entities, recorded explicitly in
  `data/manifests/llm_prompts.yaml`. Both recipient and donor remain inside the same entity split;
  all ordered non-self pairs yield `24*23 + 6*5 + 6*5 = 612` outcomes. Thus no train entity, answer,
  recipient, or donor appears in validation/test.
- Model/intervention: pinned Qwen3-0.6B revision `c1899de...`, FP32 eager attention, source
  `blocks.21.resid_post`, final prompt position, complete 1024-dimensional donor replacement.
  Target is `blocks.27.resid_post` plus logits for all 36 preregistered answer tokens.
- Strong local controls: compute an exact autograd directional JVP for the complete donor direction,
  a symmetric central difference at epsilon `1/8`, and a directional quadratic Taylor prediction.
  Direct source semantics must be exact; `h+(donor-h)` must stay within the same two-rounding
  float32 bound as JVP audit v2; JVP/central median/p95 relative errors must be at most `0.05/0.15`;
  and all direct target effects must be nonzero. Numerical failure rejects the dataset.
- Behavior eligibility, fixed before execution: clean top-token capital accuracy must be at least
  `0.60` in every entity split; full-vocabulary donor-answer transfer must be at least `0.40` on the
  test split; and at least `0.40` of all patches must change the top token. These are eligibility
  gates for the later meta-model study, not adjustable acceptance thresholds.
- Stored arrays: clean/donor source, source delta, clean/intervened final hidden states, clean/direct
  36-answer logits, direct effect, exact JVP, central difference, quadratic Taylor, entity/split IDs,
  answer IDs, top tokens, and numerical diagnostics. Sharded HDF5 must remain below 64 MB and is
  ignored; checksum manifest, metrics, and provenance are committed.
- Evidence boundary: a passing dataset establishes behavior-changing direct causal mediation and
  valid local baselines. It does not validate a learned meta-model, target-encoder JEPA, feature
  semantics, circuit, or workspace. If behavior eligibility fails, the frozen result is retained
  and the associated learned-model hypothesis is not run on a post-selected subset.

Outcome (clean commit `95018cb`, 2026-07-21): every numerical and behavior gate passed. Clean
top-token accuracy was `0.917/0.667/1.000` on train/validation/test, donor-answer transfer was
`0.580/0.500/0.500`, and overall top-token change was `0.936`. Exact JVP/central relative error was
`0.0142` median and `0.0328` p95. Exact JVP achieved `0.5296` normalized full-target MSE and `0.351`
answer-candidate agreement; quadratic Taylor achieved `0.6391` and `0.743`, respectively. The
dataset is eligible for the prospectively registered learned-model experiment. The disagreement
between vector MSE and answer behavior is descriptive only until tested on a separate model or task.

## LLM-TARGET-IJEPA-001 Genuine Target-Encoder Preregistration

Registered on 2026-07-21 after freezing `LLM-CAPITAL-PATCH-001`, but before fitting any model to its
outcomes. This is a new restricted test `H-LLM-01B`; it does not retroactively reopen or rename the
withdrawn coordinate-grid H-LLM-01 result.

- Split: immutable entity-disjoint 552/30/30 train/validation/test outcomes from the capital dataset.
  Test entities, capital tokens, recipients, and donors are absent from training. Hyperparameters
  are fixed in `configs/experiments/qwen_target_encoder_ijepa_v1.yaml`; no test selection is allowed.
- Architecture: a shared online residual encoder processes recipient source, donor source, and clean
  final residuals; a separate intervention encoder processes the complete donor delta. Their
  embeddings, recipient/donor difference, and one multiplicative interaction enter the predictor.
  The 32-dimensional target is produced by an exponential-moving-average encoder from the directly
  executed intervened final residual and is stop-gradient. The target encoder is not a target-effect
  regression head.
- Objective: target-embedding alignment plus online/target consistency, per-stream variance floors,
  and covariance penalties. Report target and predicted mean per-dimension standard deviation and
  covariance effective rank. A run is noncollapsed only when both mean standard deviations are at
  least `0.10` and both effective ranks are at least `8/32`.
- Outcome decoder: after JEPA fitting, a fixed-ridge linear decoder is trained on train entities from
  `[intervened target embedding, clean target embedding]` to the 1060-dimensional direct effect.
  At evaluation the first term is replaced by the predicted embedding. The oracle target-embedding
  decode is reported separately. The direct effect never enters the JEPA predictor objective.
- Baselines: no change, train-mean effect, raw linear ridge, 24-dimensional PCA-bilinear ridge,
  supervised MLP, the legacy supervised conditional bottleneck, PCA nearest neighbor,
  corpus-average source-delta transport, 32-component sparse-dictionary linear transport, exact
  per-example autograd JVP, and directional quadratic Taylor.
- Metrics: full/hidden/logit normalized MSE, effect correlation, and agreement with the directly
  executed winner among the 36 preregistered answer tokens. The last metric is explicitly an
  answer-set candidate metric, not full-vocabulary top-token accuracy. All 30 test targets are
  prior direct Qwen executions.
- `H-LLM-01B` passes per seed only if answer-candidate agreement exceeds the better of exact JVP and
  quadratic Taylor by at least `0.05` and logit normalized MSE is at most `0.95` times the better
  local baseline. `H-LLM-02` passes only if the representation is noncollapsed, oracle normalized
  MSE is at most `0.75`, predicted normalized MSE beats train mean, and correlation is at least
  `0.50`. Restricted `H-LLM-04` passes only if candidate agreement is at least `0.50` and exceeds
  nearest neighbor by `0.05`.
- Replication: a claim requires at least two of seeds `311/313/317`. Ensemble scores are descriptive
  and cannot rescue a failed seed gate. A failed outcome is retained without tuning the observed
  entity roster, dimensionality, thresholds, or hyperparameters.
- Evidence boundary: even a pass establishes bounded held-out-entity causal-fidelity prediction for
  one prompt family. It is not a feature label, circuit reconstruction, cross-task result, or
  workspace. Circuit claims require a separate prospectively registered direct intervention study.

Outcome (clean commit `3086cd4`, 2026-07-21): `COMPLETED_NEGATIVE`; zero of three seeds passed any
registered hypothesis. Predicted latent effective ranks were `8.64/8.65/9.75`, but target ranks
were `7.28/6.81/6.98`, so every seed failed the rank-diversity gate despite healthy mean coordinate
standard deviations. Oracle target-embedding decoders had test normalized MSE `1.255/1.753/1.065`,
showing that the learned target geometry itself was not linearly reusable across entity splits. The
JEPA ensemble achieved normalized MSE `0.930`, correlation `0.483`, and answer-candidate agreement
`0.200`. Exact JVP was best on full-vector normalized MSE (`0.599`), raw linear ridge on logit
normalized MSE (`0.329`), and quadratic Taylor on answer-candidate agreement (`0.700`). Thus
H-LLM-01B, H-LLM-02, and the restricted H-LLM-04 are rejected for this frozen architecture/task.
The endpoint-dependent ranking is a new observation within this repository, not a literature-level
novelty claim without replication on another task/model.

## LLM-CONTEXT-GEOMETRY-001 Preregistration

Registered on 2026-07-21 before computing any final-suite full Jacobian. This is an adversarial
geometry study, not a retuning of `LLM-TARGET-IJEPA-001`.

- Definitions: for recipient context `r`, `D_r` contains the actual within-split donor residual
  differences at `blocks.21.resid_post`; these are sampled manifold chords, not an unconstrained
  intervention basis. `J_s` is the FP32 autograd Jacobian of the 36 fixed answer logits in context
  `s`, centered across answers to remove the common-logit direction. The context-paired causal
  coupling is `K_{r,s}=J_s D_r^T`.
- Gauge contract: under an invertible activation reparameterization `h'=A h`, directions transform
  as `D'=D A^T` and covectors as `J'=J A^{-1}`, hence `K` is invariant. The real-data audit applies
  a fixed diagonal transform with condition number near 100 and requires maximum relative coupling
  error at most `1e-5`. Naive Euclidean subspace overlap is reported before/after but is not assumed
  gauge invariant.
- Pooling control: separately pool the top-four Euclidean row spans of all six test `D_r` and `J_r`,
  compare that overlap with the mean of six correctly paired overlaps, and then permute the context
  labels of `J`. Separate pooling is permutation invariant by construction. An exact two-dimensional
  positive control has zero same-context coupling in both contexts but pooled overlap one.
- Context-specificity null: use 256 seed-347 fixed derangements of the six test Jacobians. For each
  real donor chord, compare the matched local logit prediction `J_r d` and mismatched `J_s d` against
  the previously executed finite patch. Also evaluate the mean Jacobian of the 24 train contexts on
  test chords. No permutation outcome changes the frozen direct targets.
- Numerical gates: clean source replacement error at most `1e-5`; stored-source replay at most
  `1e-6`; reconstructed versus stored exact-JVP median/p95 relative error at most `1e-4/1e-3`;
  diagonal-gauge coupling error at most `1e-5`; analytic pooling control exact. Numerical failure
  rejects the scientific comparisons.
- `H-GEO-01 — Real pooling illusion`: top-four separately pooled overlap exceeds mean matched
  overlap by at least `0.10` and remains unchanged to `1e-10` after context permutation.
- `H-GEO-02 — Context-specific behavior transport`: matched answer-candidate agreement exceeds the
  derangement p95 by at least `0.05`, and matched finite-effect normalized MSE is at most `0.80`
  times the derangement p05. Both gates are required.
- `H-GEO-03 — Gauge-invariant coupling`: the real diagonal-gauge contraction gate and analytic
  pooling counterexample both pass. This validates a mathematical diagnostic, not a Qwen mechanism.
- Identification assumptions: donor chords sample relevant interventions; local logit covectors may
  approximate but need not explain finite replacement effects; the 36-answer endpoint omits the
  rest of the vocabulary; context pairing is meaningful at the recipient-prompt level. Every one is
  emitted in the artifact. Passing does not establish a semantic feature, circuit, J-space, or
  workspace, and the method is not claimed novel until broader literature/model replication.

Outcome (clean commit `49d68b7`, 2026-07-21): all numerical gates passed; reconstructed-versus-
stored exact-JVP median/p95 relative error was `2.86e-7/4.58e-7`. H-GEO-01 failed because pooled
top-four overlap `0.04036` exceeded matched mean `0.03403` by only `0.00633`, not `0.10`.
H-GEO-02 failed: matched finite transport had `0.540` normalized MSE and `0.300` answer-candidate
agreement, while 256 derangements averaged `0.983` MSE but `0.396` agreement (p95 `0.467`).
H-GEO-03 passed: the analytic counterexample was exact and real `J D^T` changed only `1.68e-16`
under a diagonal transform with condition number `96.4`. The same transform changed naive pooled
overlap from `0.04036` to `0.0003345`. A non-preregistered observation is that the mean Jacobian of
24 training contexts beat the exact matched local Jacobian on all three finite endpoints: MSE
`0.358 < 0.540`, correlation `0.885 > 0.841`, candidate agreement `0.500 > 0.300`. It is labeled a
post-result hypothesis and requires a held-out confirmatory analysis.

## LLM-POPULATION-JACOBIAN-001 Confirmation Preregistration

Registered on 2026-07-21 after observing the test-split train-mean-Jacobian advantage, but before
analyzing the six validation entities. Full Jacobians have already been captured; the protected
object here is the validation analysis and its decisions, not data acquisition. Thresholds are
explicitly motivated by the test discovery and cannot be described as an independent initial
hypothesis.

- Confirmation split: exactly the 30 ordered donor patches among Hungary, Norway, Cuba, Australia,
  North Korea, and Denmark (`split_id=1`). V1 geometry decisions used only `split_id=2`. The
  population Jacobian is the unweighted mean of the 24 train-entity Jacobians; no validation/test
  Jacobian enters it.
- `H-GEO-04 — Population-Jacobian confirmation`: compared with each validation recipient's exact
  local Jacobian, population transport must have normalized logit MSE at most `0.80x`, correlation
  at least `+0.03`, answer-candidate agreement at least `+0.10`, and lower MSE in at least four of
  six recipient contexts. All four gates are required.
- `H-GEO-05 — Averaging regularization curve`: draw 128 fixed seed-353 subsets at train-context
  counts `1/2/4/8/16` and use the unique 24-context mean. Median size-16 MSE must be at most `0.80x`
  size-one MSE, correlation between log subset size and median MSE must be at most `-0.80`, and the
  24-context candidate agreement must exceed size-one median by at least `0.10`.
- `H-GEO-06 — Answer-semantic specificity`: under 256 fixed answer-row permutations of the same
  population Jacobian, the real aligned mean must have MSE at most `0.80x` the null p05 and candidate
  agreement at least `0.05` above the null p95. This norm-preserving output-label corruption tests
  whether averaging merely shrinks derivative noise without preserving answer semantics.
- Report matched local, population, and quadratic scores; predicted/direct donor rates; all
  population-size distributions; per-context MSE; and a 10,000-draw seed-359 paired bootstrap CI of
  raw per-example MSE improvement. Bootstrap significance is descriptive and cannot override gates.
- Evidence boundary: a full pass would confirm that population averaging regularizes finite
  logit-effect transport across two disjoint six-entity analyses. It would not show a novel
  algorithm, a semantic activation direction, a circuit, or a workspace. Failure is retained without
  changing subsets, thresholds, output rows, or entity rosters.

Outcome (clean commit `3725714`, 2026-07-21): all three confirmation hypotheses passed. Population
versus exact local transport scored normalized logit MSE `0.3538` versus `0.7371`, correlation
`0.8655` versus `0.8351`, and answer-candidate agreement `0.5333` versus `0.3000`; population MSE was
lower in exactly four of six contexts. The paired-bootstrap raw-MSE improvement CI was
`[2.2817, 8.2759]`. Median subset-average MSE fell from `0.5468` at one train context to `0.3538` at
24, with log-size/MSE correlation `-0.9146`; candidate agreement rose `0.4333→0.5333`. Answer-row
permutation p05 MSE was `1.4132` and p95 candidate agreement `0.0333`, so H-GEO-06 passed by a wide
margin. Quadratic Taylor retained the best discrete candidate agreement (`0.8333`) but had worse
continuous MSE (`0.7738`). This confirms the finite-effect population-regularization observation
on a second entity-disjoint split. Corpus-averaged Jacobian transport itself is prior art in the
Jacobian Lens; this is not a new algorithm or workspace claim.

## LLM-ELEMENT-LAYER-GEOMETRY-001 Preregistration

Registered on 2026-07-21 before executing any of the 36 final element prompts. Four excluded
calibration facts—Gold/Au, Silver/Ag, Tin/Sn, and Lead/Pb—were used only to choose the fixed layer
grid and hypotheses. On those excluded examples, full-residual donor patches had zero donor-symbol
control through layer 21 and complete control at layers 24/26; local versus population behavior also
reversed across that interval. These pilot outcomes set the thresholds below and are not evidence.

- Prior-art boundary: MechLens (LIT-040) already reports late factual crystallization, and AtP*
  (LIT-030) already documents nonlinear failure modes of gradient attribution. The new question is
  narrower: whether directly executed donor control and a reversal in exact-local versus population
  finite-effect predictivity coincide across an entity-disjoint factual relation. Neither late
  emergence nor population Jacobian averaging is claimed as a new algorithm.
- Data: 36 unique element/symbol facts whose leading-space symbol is exactly one unique Qwen token.
  Seed 457 fixes 24/6/6 entity-disjoint train/validation/test sets; donor and recipient always stay
  in the same split. The prompt is exactly `The chemical symbol for {element} is`. The four pilot
  elements never enter this roster. Every ordered non-self pair yields 612 patches per layer.
- Model/intervention: Qwen3-0.6B revision `c1899de...`, FP32 eager attention, final prompt token,
  full residual replacement at `blocks.18/21/24/26.resid_post`, and direct full-vocabulary behavior.
  Each clean context also receives a complete `36 x 1024` selected-answer logit Jacobian.
- Numerical gates: clean source replacement error at most `1e-5`, direct donor-source error at most
  `1e-6`, and exact-Jacobian versus symmetric central-difference relative error median/p95 at most
  `0.05/0.15` independently at every layer. Any layer failure rejects all scientific decisions.
- Endpoints/baselines: selected-logit effects are centered across the 36 answers before MSE and
  correlation, removing common-logit shifts. Direct answer candidate and full-vocabulary top token
  remain separate. Report no change, train mean effect, exact local Jacobian, quadratic Taylor, and
  train-population Jacobian at all layers/splits, plus context-count curves and 256 answer-row
  permutations. Direct execution remains the target.
- `H-LLM-08 — Late causal-control transition`: on both validation and test, maximum full-vocabulary
  donor-symbol transfer across layers 18/21 is at most `0.10`, minimum transfer across layers 24/26
  is at least `0.60`, and the late-minus-early increase is at least `0.50`.
- `H-GEO-08 — Local/population predictivity inversion`: on both validation and test, layer-21 local
  contrast MSE is at most `0.25x` population MSE; layer-24 population MSE is at most `0.60x` local;
  layer-24 local MSE is at least `3x` layer-21 local; and layer-24 population correlation exceeds
  local by at least `0.01`. All four gates on both splits are required.
- `H-GEO-09 — Late population semantic specificity`: at both layers 24 and 26 on test, the aligned
  population Jacobian has contrast MSE at most `0.80x` the p05 of 256 answer-row permutations and
  candidate agreement at least `0.05` above the null p95.
- `H-CROSS-03 — Population transport across factual relations`: supported only if H-LLM-08,
  H-GEO-08, and H-GEO-09 all pass and the already frozen capital H-GEO-04/05/06 result remains
  positive. Quadratic Taylor is always reported and may still be the stronger behavior baseline.
- Evidence boundary: a full pass would establish a replicated layerwise association among direct
  causal control, finite-effect nonlinearity, and population transport in this Qwen model. It would
  not identify the source feature, prove the transition is a circuit, validate Intervention-JEPA,
  or establish J-space/workspace structure. No layer, entity, threshold, or output row changes after
  the first registered prompt execution.

Measured result from clean commit `5d8de9a`: all numerical gates passed and the status is
`COMPLETED_MIXED`. H-LLM-08 passed: validation donor-symbol transfer was `0/0/0.60/1.00` and test
was `0/0/0.90/1.00` at layers 18/21/24/26. H-GEO-09 also passed: layer-24/26 test population MSE
was `0.1808/0.01131` against row-null p05 `2.372/1.772`, and agreement was `0.90/1.00` against
null p95 `0.167/0.108`. H-GEO-08 failed because the layer-24 population/local MSE ratios were
`0.879` validation and `0.707` test rather than at most `0.60`; test layer-21 local/population was
`0.326` rather than at most `0.25`. Therefore H-CROSS-03 also failed. The direct-control transition
and late semantic specificity survive; the preregistered strong inversion and cross-relation
conjunction do not.

## LLM-STATE-LAYER-GEOMETRY-001 Preregistration

Registered on 2026-07-21 after freezing the element result and before any forward pass on a
registered state prompt. Tokenization alone identified 49 states with unique single-token postal
abbreviations; seed 521 selected 36, and seed 523 fixed 24/6/6 entity-disjoint splits. No clean,
patched, or derivative outcome from this roster was inspected before this registration.

- Prior-art boundary: Jacobian Lens (LIT-003) already uses population-averaged Jacobians, MechLens
  (LIT-040) already establishes late factual crystallization, and Zhang/Wang (LIT-041) already
  diagnose first-order attribution-patching error and introduce HVP corrections. The directional
  quadratic Taylor result here is an HVP-style comparator, not a new method.
- Model/data: Qwen3-0.6B revision `c1899de...`, FP32 eager attention, exact prompt
  `The postal abbreviation for {state} is`, full final-token residual replacement, and layers
  18/21/24/26 inherited unchanged from the element study. Every ordered non-self pair stays within
  its split, producing 612 direct patches per layer and complete `36 x 1024` selected-logit
  Jacobians per context/layer.
- Numerical gates: independently at every layer, clean replay at most `1e-5`, donor-source error at
  most `1e-6`, and exact-Jacobian versus symmetric-central median/p95 relative error at most
  `0.05/0.15`. Any failure rejects every scientific decision.
- Behavior gate: clean full-vocabulary answer accuracy must be at least `0.75` on both validation
  and test. Failure yields `REJECTED_BEHAVIOR_GATE` and no geometry hypothesis is decided.
- Baselines/endpoints: centered 36-answer logit effects are scored with no-change, train mean,
  exact local Jacobian, directional quadratic Taylor/HVP-style correction, and 24-train-context
  population Jacobian. Full-vocabulary donor transfer, answer-candidate agreement, 1/2/4/8/16/24
  averaging curves, and 256 answer-row permutations remain separate endpoints.
- `H-LLM-10 — State late causal-control transition`: on both validation and test, maximum donor-
  abbreviation transfer at layers 18/21 is at most `0.10`, minimum transfer at layers 24/26 is at
  least `0.60`, and the late-minus-early increase is at least `0.50`.
- Define population advantage at layer `l` as
  `A_l = min(NMSE_local, NMSE_quadratic) - NMSE_population`; positive values mean population
  transport beats both exact local and the second-order comparator.
- `H-GEO-10 — Control-conditioned population advantage`: on both validation and test,
  `A_21 <= -0.05`, `A_24 >= 0`, and the Spearman correlation across all four layers between direct
  donor transfer and `A_l` is at least `0.80`. These sign/margin gates were chosen after the element
  result and are confirmatory only on this untouched relation.
- `H-GEO-11 — State late population semantic specificity`: at layers 24 and 26 on test, aligned
  population MSE is at most `0.80x` the answer-row-null p05 and candidate agreement is at least
  `0.05` above the null p95.
- `H-CROSS-04 — Control-conditioned transport across factual relations`: passes only if H-LLM-10,
  H-GEO-10, and H-GEO-11 all pass and the frozen element H-LLM-08/H-GEO-09 decisions remain true.
- Evidence boundary: a full pass would confirm a layerwise association between direct donor control
  and the relative utility of population transport across two factual relations in one small Qwen.
  It would not establish why the transition occurs, identify a feature/circuit, beat published SOTA
  across models, validate Intervention-JEPA, or establish a J-space/workspace.

Measured result from clean commit `27ebe43`: status `REJECTED_BEHAVIOR_GATE`. All numerical gates
passed, but clean validation/test accuracy was `0.667/0.667`, below the registered `0.75` floor.
Therefore H-LLM-10, H-GEO-10, H-GEO-11, and H-CROSS-04 are not scientifically decided. The artifact
stores `false` decisions because the behavior gate masks every downstream decision; these must not
be reported as hypothesis falsifications. Descriptively, donor-control/population-advantage Spearman
was `0.9487` on both splits, while advantage changed from negative through layer 24 to positive at
layer 26. That boundary-relative pattern was not the registered layer-24 sign gate and is only a
candidate for a separately preregistered, behavior-competent task.

## LLM-STATE-ONESHOT-LAYER-GEOMETRY-001 Preregistration

Registered on 2026-07-21 after freezing state v1 and before any forward pass on a target one-shot
prompt. Prompt calibration used only the 13 seed-521 states excluded from the 36-entity roster. Five
templates scored clean top-token accuracy `12/13`, `3/13`, `13/13`, `8/13`, and `3/13`; the fixed
District-of-Columbia example was selected solely for its `13/13` score. The target roster/splits are
unchanged from v1, but every exact one-shot target prompt and all of its causal outcomes are new.

- Prior-art boundary: population Jacobians (LIT-003), late crystallization (LIT-040), and HVP
  corrections (LIT-041) are established methods/phenomena. The prospective question is the equality
  of two empirical onset layers, not a new lens or second-order algorithm.
- Model/intervention: pinned Qwen3-0.6B FP32/eager; exact prompt
  `Example: The postal abbreviation for District of Columbia is DC. The postal abbreviation for
  {state} is`; final-token full-residual donor patches at layers 18/21/24/26; the same fixed 24/6/6
  entity split; 612 within-split patches per layer; complete selected-logit Jacobians.
- Numerical gates are unchanged: clean replay at most `1e-5`, donor-source error at most `1e-6`, and
  exact-Jacobian/central median/p95 relative error at most `0.05/0.15` independently at every layer.
- Behavior competence is stricter than v1: clean validation and test top-token accuracy must each be
  at least `0.90`, or every scientific decision is rejected.
- `H-LLM-12 — One-shot late causal control`: on both validation and test, maximum donor-token
  transfer at layers 18/21 is at most `0.10`, layer-26 transfer is at least `0.60`, and the increase
  is at least `0.50`. Layer 24 is reported but is not fixed as the onset.
- Population advantage remains
  `A_l = min(NMSE_local, NMSE_quadratic) - NMSE_population`. Define the control boundary as the first
  registered layer with donor-token transfer at least `0.50`, and the population boundary as the
  first registered layer with `A_l >= 0`.
- `H-GEO-12 — Control/population boundary alignment`: on both validation and test, `A_21 <= -0.05`,
  `A_26 >= 0.05`, donor-control/advantage Spearman across all four layers is at least `0.80`, and
  the two first-crossing layers are exactly equal.
- `H-GEO-13 — Boundary population semantic specificity`: on test, the aligned population transport
  at the detected control boundary and layer 26 must have MSE at most `0.80x` answer-row-null p05
  and candidate agreement at least `0.05` above null p95.
- `H-CROSS-05 — Boundary alignment across factual relations`: passes only if H-LLM-12, H-GEO-12,
  and H-GEO-13 pass and the frozen element data independently satisfy boundary equality plus the
  same `A_21 <= -0.05` and `A_26 >= 0.05` margins and their registered H-LLM-08/H-GEO-09 decisions.
  The frozen element test Spearman is `0.738`, so H-CROSS-05 does not require it to meet the new
  state-only `0.80` gate; this distinction was recorded before any target one-shot forward.
- Evidence boundary: a full pass is a cross-relation, dual-split association in one small model. It
  is not an explanation of the boundary, model-scale generalization, component localization,
  circuit reconstruction, a JEPA meta-model result, workspace/J-space evidence, or SOTA.

Measured result from clean commit `c1daa46`: status `COMPLETED_MIXED`. Clean validation/test
accuracy was `1.0/1.0` and all numerical gates passed. H-GEO-13 passed with layer-24/26 test
population NMSE `0.1193/0.02079`, agreement `1.0/1.0`, row-null p05 MSE `2.112/1.742`, and null p95
agreement `0.133/0.133`. H-GEO-12 failed because validation control onset was layer 24 but population
advantage onset was layer 26 (`A_24=-0.0266`); test aligned at layer 24. Its remaining sign margins
and Spearman gates passed. H-LLM-12 failed because test layer-21 donor transfer was `0.233`, above
`0.10`, despite layer-26 transfer `1.0`. H-CROSS-05 therefore failed. Exact boundary identity is
false for this run; semantic late population fidelity survives.

## LLM-COUNTRY-CODE-LAYER-GEOMETRY-001 Preregistration

Registered on 2026-07-21 after freezing the failed exact-equality state result and before any model
forward on the 36 target country prompts. Tokenization-only seed 601 selected a 36-country roster
from 43 candidates whose single-token ISO answers do not overlap the state-abbreviation answers;
seed 607 fixed 24/6/6 target splits. Seven excluded countries were reserved for prompt calibration.
Five templates scored `0/7`, `5/7`, `2/7`, `5/7`, and `0/7`; the first tied winner was selected by
the frozen maximum-accuracy/earliest-candidate rule. This modest `5/7` calibration makes the target
`0.90` competence gate especially important. No target-country prompt was forwarded during roster,
split, or prompt selection.

- Exact prompt: `Example: The two-letter ISO country code for Canada is CA. The two-letter ISO
  country code for {country} is`. Model, precision, intervention, derivatives, layers
  18/21/24/26, 612 within-split donor patches per layer, row nulls, and numerical thresholds are
  inherited unchanged from the one-shot state study.
- Prior-art and posthoc boundary: corpus-averaged Jacobians (LIT-003), late factual crystallization
  (LIT-040), and HVP/second-order attribution correction (LIT-041) are prior art. The zero-or-one
  grid-step rule was designed after observing element and state results. Those frozen relations are
  motivating data, not independent prospective confirmations; only country is untouched here.
- Behavior eligibility: clean full-vocabulary validation and test accuracy must each be at least
  `0.90`. Failure rejects every scientific decision without changing the roster or prompt.
- `H-LLM-14 — Country monotone causal control`: on validation and test, layer-18 donor-token
  transfer is at most `0.10`, layer-26 transfer is at least `0.60`, the increase is at least `0.50`,
  and transfer does not decrease by more than `0.05` at any adjacent registered layer. This rule
  deliberately replaces the state study's falsified layer-21 ceiling and is prospective only here.
- Define `A_l = min(NMSE_local, NMSE_quadratic) - NMSE_population`, the direct-control boundary as
  the first registered layer with donor transfer at least `0.50`, and the population boundary as
  the first with `A_l >= 0`.
- `H-GEO-14 — Bounded control/population lag`: on validation and test, both boundaries exist, the
  population boundary is no earlier than direct control and at most one registered grid step later,
  `A_21 <= -0.05`, `A_26 >= 0.05`, and four-layer Spearman between donor control and `A_l` is at
  least `0.70`. The lower correlation threshold was fixed from the already-known element test value
  `0.738`; it cannot be interpreted as a blind threshold choice for the frozen relations.
- `H-GEO-15 — Country population semantic specificity`: on test, population transport at the
  detected population boundary and layer 26 must achieve MSE at most `0.80x` answer-row-null p05
  and candidate agreement at least `0.05` above row-null p95.
- `H-CROSS-06 — Bounded lag across factual relations`: requires all three country hypotheses plus
  behavior eligibility, the same new monotone/bounded-lag rule, and the already registered semantic
  specificity pass in both frozen element and one-shot-state artifacts. Because the rule was built
  from those artifacts, a pass is a prospective country replication of a posthoc cross-relation
  pattern in one small model—not three independent confirmations.
- Evidence boundary: even a complete pass establishes only a bounded four-layer association for
  three factual relations in Qwen3-0.6B. It does not explain the lag, localize components, identify
  a circuit or feature, validate an Intervention-JEPA, transfer across models, establish SOTA, or
  support a workspace/J-space claim.

Execution audit: the first launch from clean `ab07627` failed after model computation but before
analysis/storage because the frozen element artifact did not yet contain the later-added
`behavior_eligible` field. No target metrics, manifest, shard, or outcome was emitted. The retry
adds only a compatibility fallback that recomputes frozen eligibility from the already committed
validation/test clean accuracies and the unchanged `0.90` floor; no scientific setting changed.

Measured result from clean retry commit `48226c6`: status `COMPLETED_MIXED`, runtime `283.77`
seconds. Clean validation/test accuracy was `1.0/1.0`; all numerical gates passed, including maximum
layer p95 exact-Jacobian/central error `0.0635`. H-LLM-14 passed: validation donor transfer was
`0/.367/.867/1.0`, and test was `0/.667/.967/1.0`. H-GEO-15 passed: test population NMSE/agreement
was `.2949/.5667` at layer 21 and `.01470/.9667` at layer 26, against row-null p05 MSE
`1.350/1.679` and p95 agreement `.233/.100`. H-GEO-14 failed because validation population advantage
crossed at layer 21 (`A_21=.1956`) before direct control crossed at 24; test crossed both at 21 but
also had positive `A_21=.2289`, violating the registered negative early margin. H-CROSS-06 failed.
The data reject a universal population-after-control ordering at this threshold; they do not prove
that population transport causes control or identify why relation boundaries differ.

## WM-ACTION-PATH-CALIBRATION-001 Boundary

Registered as calibration-only on 2026-07-21 before running the new dense path profiler. This ID is
hard-locked to the already exposed five validation goals and therefore makes no scientific
hypothesis decision. Across the three frozen checkpoints and horizons one/four, it samples exactly
two contexts for each of 12 ordered action pairs and evaluates exact action JVPs with 8-point
composite Gauss-Legendre rules at 16 and 32 panels. It reports decoded physical path length, net
effect, cancellation ratio, local finite-effect error, direct reconstruction error, refinement
change, speed concentration, and 256 within-action-pair permutations. It may inform a later
preregistration, but cannot validate a claim, alter v1, or access test goal IDs 12/11/10/21/2.

Local-linear world-action steering is prior art (LIT-042), and latent trajectory curvature is prior
art (LIT-012). The narrower decoded action-chord cancellation diagnostic was not found in the
bounded primary-source search; this is not proof of novelty. A future test must preserve decoded
coordinates, numerical convergence, horizon-one controls, direct effects, and stratified nulls.

Calibration result from clean `eb943a5`: `CALIBRATION_ONLY`, `72.59` seconds, no decisions, and no
test-goal access. Horizon one converged for every seed, but horizon-four maximum direct/refinement
errors were `.0680/.1369`, `.478/2.059`, and `.00197/.00093`; 256 nodes do not resolve seeds 101/103.
Horizon-four cancellation/local-error Spearman exceeded its stratified null p95 in all seeds, but
only by `.045/.027/.022`, while median recurrence amplification was `1.05/2.82/.983`. No test
hypothesis is registered from these diagnostics. A higher-resolution validation-only calibration
was required to characterize numerical stability, but could not itself authorize the protected
split.

Calibration v2 is separately registered before execution as `WM-ACTION-PATH-CALIBRATION-002`. It
uses the identical validation goals, profile seed, 24 chords per seed/horizon, decoder, checkpoints,
and stratified nulls, changing only composite panels from 16/32 to 64/128 (512/1024 nodes). It emits
no decisions and cannot access the test split. Its sole purpose is to determine whether v1's
underresolved cancellation/error association is numerically stable enough to motivate a future
prospective hypothesis.

Adversarial review before v2 completed identified a structural confound: cancellation
`L / ||delta_y||` and normalized local error both divide by `||delta_y||`. Their correlation may
therefore arise mechanically when the net decoded effect is small, and action-pair stratification
does not remove it. Even a numerically converged v2 cannot directly authorize protected-test access.

Execution audit: the first clean v2 launch from `e918d4f` failed with CUDA OOM before metrics or
provenance were written. The clean retry changes only `jacobian_chunk_size` from 16 to 2; it does
not change goals, sampled chords, quadrature nodes, outputs, or the absence of decisions.
The chunk-2 retry from clean `c72d9f5` also emitted no artifact because the implementation retained
large autograd graphs into the 1,024-node pass. The next clean retry streams 64 outer samples,
immediately projects each exact Jacobian into the registered decoded coordinates, detaches to CPU,
and frees its graph; all mathematical settings remain unchanged.

Pre-result adversarial review also rejected a proposed derived denominator audit before commit.
The parent records convergence of the integrated decoded vector, not the scalar path length `L` at
both resolutions, and its stored direct norm is floored at `1e-4`. Only two chords are available for
each ordered action pair. Thus unnormalized path excess and local residual still inherit common
scale/direction structure, separate action-pair and effect-bin permutations do not form a joint
conditional null, and leave-one-pair-out summaries cannot compensate for sparse within-pair support.
V2 therefore remains numerical/vector calibration regardless of its outcome. No protected-test
hypothesis follows, and this small-model family is closed. A materially new prospective study would
need raw norms, path lengths at both resolutions, substantially more chords per pair, row-level
split/seed guards, and a preregistered joint conditional null.

Final v2 calibration result from clean `288f663`: `CALIBRATION_ONLY`, `19,176.20` seconds,
`protected_test_goals_touched=false`, and no hypothesis decisions. Horizon-four maximum
integration/refinement errors were `.01291/.02605`, `.03394/.03141`, and `.001393/.000250` for
seeds 101/103/107. Cancellation/local-error Spearman exceeded stratified-null p95 by only
`.0454/.0283/.0165`. The refinement therefore leaves two seeds underresolved and does not alter the
pre-result design closure.

## WM-POPULATION-JACOBIAN-001 JEPA Causal-Geometry Preregistration

Registered on 2026-07-21 before loading any saved LeWorldModel checkpoint for this analysis. The
experiment ports the Qwen population-versus-local finite-transport question to autoregressive JEPA
dynamics. It uses the already frozen `WM-LEWM-001` checkpoints without retraining or checkpoint
selection; their SHA-256 values are pinned in the config. This is a new analysis of those models,
not a revision of the failed `WM-LEWM-001` circuit/workspace decisions.

- Environment-action contract: each intervention replaces the first action of a one- or four-step
  rollout between two different one-hot actions. All 12 ordered recipient/donor pairs are executed
  directly. Four-step rollouts use exactly four registered three-action suffixes. These are valid
  environment actions, not interpretability interventions or arbitrary latent directions.
- Goal split: seed 401 fixes 12 train-Jacobian goals, five validation-analysis goals, and five
  untouched test goals. V1 analyzes only validation goals; the test goals remain protected for a
  separately committed confirmation if warranted. Every goal is crossed with all 22 valid current
  positions. The checkpoint was originally trained on random trajectories spanning the maze, so
  this is a held-out Jacobian-estimation split, not a claim of model-training OOD generalization.
- Derivatives: `J(z,a,s)` is the exact FP32 autograd Jacobian of the final predicted latent with
  respect to the first action, conditional on initial latent `z` and fixed suffix `s`. The local
  predictor is `J(z,a_recipient,s)(a_donor-a_recipient)`. The population predictor averages
  Jacobians over every train goal, position, suffix, and action vertex. Within-context vertex mean,
  action-centroid, and endpoint-trapezoid transports are reported as mechanism controls.
- Direct target and path oracle: the target is the frozen model's directly executed donor-minus-
  recipient final latent. A 12-node Gauss-Legendre integral of the Jacobian along 128 fixed action
  chords per seed/horizon must reconstruct direct effects with maximum relative error at most
  `1e-4`; otherwise scientific comparisons are rejected.
- Primary gauge-safe endpoint: a frozen ridge readout is fit only on encoded next images from the
  12 train goals and maps latent states to next `x`, `y`, and Manhattan distance. Population/local
  decisions use decoded physical-effect MSE and correlation. Raw latent scores are secondary
  because an invertible latent reparameterization can change Euclidean method rankings. Under a
  condition-100 diagonal gauge change, the decoder is transformed contragrediently and must retain
  effects to `1e-10`; an analytic two-dimensional control must flip a raw-MSE method ranking.
- `H-WM-11 — Population transition smoothing`: at horizon four, the population Jacobian must have
  decoded normalized MSE at most `0.80x` local, correlation at least `+0.02` higher, and lower MSE
  in at least four of five validation goals. At least two of three checkpoint seeds must pass.
- `H-WM-12 — Averaging dose response`: using 64 fixed subsets at `1/4/16/64/256` train anchors and
  the unique full mean, size-256 median decoded MSE must be at most `0.85x` size-one and correlation
  of log subset size with median MSE at most `-0.80`. At least two seeds must pass.
- `H-WM-13 — Within-context path averaging`: the mean of the four action-vertex Jacobians in the
  same held-out context must have decoded MSE at most `0.80x` local at both horizons. This tests
  whether categorical-vertex curvature, rather than only cross-context noise, causes local failure.
- `H-WM-14 — Action-semantic specificity`: the aligned population Jacobian's horizon-four decoded
  MSE must be at most `0.80x` the p05 of all 23 nonidentity action-column permutations. At least two
  seeds must pass.
- `H-GEO-07 — Gauge-safe causal fidelity`: both path-integral and decoded-gauge numerical gates,
  plus the analytic raw-ranking counterexample, must pass in at least two seeds. This validates an
  evaluation contract, not a learned mechanism.
- `H-CROSS-02 — Finite-chord population transport`: this is supported only if the already frozen
  Qwen H-GEO-04/05/06 decisions remain positive and JEPA H-WM-11/12 pass in at least two seeds. A
  pass would be bounded cross-domain evidence that derivative averaging regularizes finite causal
  chords; corpus averaging itself is Jacobian-Lens prior art and is not claimed as a new algorithm.
- Planning boundary: first-action agreement with direct latent-cost choices is reported, but it is
  causal-behavior evidence only if the direct frozen planner selects an environment-optimal action
  on at least `60%` of contexts. The known weak `WM-LEWM-001` planner makes ineligibility plausible;
  no threshold changes after execution. No outcome can establish a workspace or circuit here.

Outcome (clean commit `89b2e14`, 2026-07-21): `REJECTED_NUMERICAL_GATE`. The decoded gauge audit
was stable to at most `2.73e-12` and its analytic ranking-flip control passed, but fixed 12-node path
integration did not reconstruct the recurrent finite effects. Horizon-four median/p95/max relative
errors were `0.116/4.23/11.18`, `0.704/8.89/49.58`, and
`1.27e-4/0.00109/0.0822` for seeds 101/103/107. Therefore no scientific hypothesis is accepted,
including the artifact's provisional 3/3 H-WM-13 flags. Descriptively, the within-context mean of
the four valid action-vertex Jacobians reduced decoded MSE versus local in every seed at both
horizons, while the global mean's low MSE had near-zero correlation and failed every action-column
specificity gate. All planners were below the 60% competence floor. A post-result diagnostic on the
worst seed-103 chord confirmed local autograd against central differences but required 192-point
quadrature to reduce relative integral error from `49.8` to `0.0058`; this indicates stiff,
cancelling recurrent derivatives rather than a simple autograd failure. A corrected confirmation
must be separately registered on the untouched test goals.

## WM-LEWM-001 Faithful-Reproduction and Circuit Preregistration

Registered on 2026-07-21 before any full scientific execution. Short reduced-data engineering
checks were used only to validate gradients, tensor shapes, and runner behavior; they are not
retained as evidence and did not use this fixed three-seed configuration.

- Published source: official LeWorldModel repository at immutable revision
  `8edfeb336732b5f3ce7b8b210d0ba370a09e2cac` (MIT license). The reproduction retains end-to-end
  pixels, action embedding, an autoregressive AdaLN-zero predictor, next-embedding MSE, and SIGReg,
  with no EMA, pretrained encoder, or auxiliary training supervision. It scales the official
  architecture to 20x20 PixelTinyMaze, 32 latent dimensions, two encoder/predictor blocks, and 64
  SIGReg projections. This is a source-informed small reproduction of selected design elements,
  not the released checkpoint result.
- Data/splits: deterministic PixelTinyMaze uses disjoint seeds `83/89/97` for 768/160/192 train,
  validation, and test trajectories. Model seeds are `101/103/107`; two of three must pass.
- H-WM-01 restricted reproduction gate: test next-embedding MSE at most `0.60`, clean/shuffled-action
  MSE ratio at most `0.85`, mean latent standard deviation at least `0.40`, and frozen nonlinear
  state-probe R2 at least `0.60`.
- H-WM-03 specificity gate: at `predictor.block1`, a four-dimensional paired-action subspace swap
  must recover at least `0.60` of the donor latent effect and `0.15` of decoded-state effect. Latent
  recovery must exceed the p95 of 16 equal-dimensional, perturbation-norm-matched random controls
  by at least `0.10`.
- Planner gate: project out that subspace during four-step latent planning across 12 fixed held-out
  start/goal pairs. It must change at least `20%` of selected first actions, latent-trajectory MSE by
  at least `0.02`, and mean candidate-cost magnitude by at least `0.02`. Either action-change rate
  or cost change must exceed the p95 of eight norm-matched random-subspace controls. Closed-loop
  success and final distance are measured but not required to improve or degrade.
- Restricted circuit gate: action-embedding suppression must increase held-out prediction error by
  at least `1.10x`; full donor action-embedding patch recovery must be at least `0.99`; last-hidden
  subspace recovery must be at least `0.50`; the specificity and planner gates must pass. Earlier
  predictor blocks are localization alternatives, not assumed serial nodes, because AdaLN action
  conditioning enters every block. Passing supports only a restricted learned action-to-planner
  circuit on this reproduction.
- Workspace boundary: five frozen physical/value/risk/uncertainty/action consumers and planted
  shared/disjoint controls are evaluated. `workspace_found` remains false unless all criteria in
  `docs/WORKSPACE_CRITERIA.md`, including reportability, voluntary control, selective multistep
  necessity, and task/OOD generalization, are met. WM-LEWM-001 cannot meet those omitted criteria.
- No threshold, seed, split, site, basis dimension, or candidate set may be changed after the full
  run begins. A failed gate is a retained negative result.

Measured result, unchanged from the preregistration: the clean run from commit `4dbc388` passed the
restricted reproduction gate on all three seeds. Test embedding MSE was `0.250/0.134/0.292`,
clean-to-shuffled ratios were `0.354/0.155/0.643`, latent standard deviations were
`0.802/0.862/0.762`, and nonlinear state-probe R2 was `0.746/0.814/0.879`. The planner gate passed
seeds 103 and 107: projection removal changed selected first actions on `0.667/0.750` of cases and
changed mean candidate costs by `0.371/0.453`, just/clearly above matched-control p95
`0.368/0.376`. It failed seed 101 because cost change `0.413` was below control p95 `0.443` and
action-change rate did not exceed its control. H-WM-03's complete donor latent-plus-decoded gate
passed only seed 107; seeds 101/103 had negative decoded recovery. The restricted circuit likewise
passed only seed 107, below the required two. Five-consumer workspace candidates failed all seeds.
The registered overall result is `FAILED_REGISTERED_GATES`; the aggregate graph is `REJECTED`.

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
