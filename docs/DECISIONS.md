# Decisions

## 2026-07-21

- Pin official EB-JEPA to `966e61e9285b3a876f49b9774e9720d9a99a7925`. Treat the actual
  action-conditioned predictor as a one-layer 512-dimensional GRU and require native-versus-
  decomposed recurrence error at most `1e-6` before gate interventions are eligible. Contract
  validation under Torch 2.10 is an engineering result; exact upstream training still requires an
  isolated Python 3.12/Torch 2.6 environment and does not inherit a reproduction claim.

- Prioritize official EB-JEPA Two Rooms over further interpretive extrapolation from the weak small
  LeWorldModel planner. First reproduce competent planning, then instrument recurrent action/gate
  routes and audit trajectory, cost, action, and closed-loop mediation across three seeds.
- Treat Qwen-Scope, Circuit Tracing, AtP*, HVP/multi-step HVP, EAP-IG/faithfulness, and direct
  activation/path patching as required comparators for new Qwen localization claims.
- Do not preregister the initial population-mediation draft. Define an upstream value-swap
  treatment, add ordered multi-site patch/restore execution, keep v1 module-only, freeze the
  smallest `k <= 4` on train, and cluster held-out inference by independent binding episode. A
  passing module set is a specific mediator set; circuit, workspace, JEPA, and SOTA remain unclaimed.

- Maintain `papers/causal_workspace_jepa.tex` as the source-of-truth scientific manuscript. Every
  result table must name its evidence level, and every empirical statement must trace to a committed
  metric/provenance artifact. Generated PDF/auxiliary products remain ignored.
- Frame the present contribution as a falsification-first audit of local, quadratic, population,
  and learned finite-intervention transports. Do not claim a new Jacobian method, SOTA result,
  Qwen circuit, learned causal compression, JEPA workspace, or cross-domain mechanism.
- Report endpoint disagreement as a result rather than selecting one favorable metric: full hidden
  vectors, selected logits, and answer behavior can rank exact JVP, quadratic Taylor, population
  transport, and learned predictors differently.
- Treat decoded JEPA action-path cancellation as a calibration-stage candidate. Cancellation and
  normalized local error share the net-effect denominator, so raw validation correlation is
  structurally confounded. High-resolution convergence is necessary but not sufficient.
- Use evidence-level columns only for the six declared hierarchy levels. Eligibility, numerical
  rejection, and run completion belong in separate status columns. Report the strongest positive
  evidence accepted, not the most ambitious held-out design attempted; the current paper has no
  accepted level-6 mechanism result.
- Reject `WM-ACTION-PATH-DENOMINATOR-AUDIT-001` before commit and before v2 output. Its derived
  unnormalized endpoints still inherit scale/direction structure, separate conditional nulls do
  not jointly condition the confounds, and two chords per action pair are too sparse for the
  proposed inference. The parent also lacks scalar path-length refinement and unclamped direct
  norms. Retain v2 as numerical/vector calibration and close the small-model path family without
  inspecting protected test goals. Reopen only with a materially new prospective acquisition.
- Supersede the positive interpretation of `LLM-IJEPA-001` H-LLM-01 while preserving its original
  run. A BF16 one-sided secant is not an exact Jacobian and may be a numerical-noise baseline.
  Resolve this through the preregistered FP32 exact-JVP audit before making any nonlinear claim.
- Refer to `NeuralInterventionJEPA` from the old run as a legacy supervised conditional bottleneck.
  The class name does not substitute for a target encoder, stop-gradient/EMA target, or anti-collapse
  JEPA loss; implement those separately.
- Force eager attention and disable TF32 for the exact derivative audit because reverse-over-reverse
  JVP through CPU/SDPA flash attention lacks the required derivative. Validate autograd JVP against
  symmetric central differences before the real run.
- When `local_files_only` is requested, resolve the pinned Hugging Face snapshot to an actual local
  directory before constructing both model and tokenizer. This prevents Transformers 5.3's Mistral-
  regex compatibility check from making an unauthorized metadata request and preserves the exact
  requested revision in adapter metadata.
- Do not claim novelty for generic controllability/observability Gramians, Hankel modes, or CoBRAS.
  Any new geometry result must be conditional on the same context, survive a cross-context pooling
  illusion/null, and predict directly executed finite-amplitude behavioral effects.
- Preserve JVP audit v1 as rejected. V2 may replace only the invalid semantic identity test with
  source-level checks: exact direct edit and a two-rounding float32 endpoint bound. Every scientific
  split, baseline, nonlinearity threshold, and learned-predictor threshold stays frozen from v1.
- Accept v2 as a valid negative Specificity result from clean commit `a779ff6`. Withdraw restricted
  H-LLM-01: exact JVP and especially quadratic Taylor explain the selected finite-edit effects much
  better than the legacy bottleneck. Do not reinterpret this as a universal linearity claim.
- Move the next Qwen dataset to full residual donor patches that transfer a known one-token answer.
  Split recipient and donor entities together, exclude all layer-calibration entities, store exact
  JVP/quadratic controls up front, and require aggregate behavior change before fitting a new model.

- Activate `gpu_12gb` after detecting an RTX 5070 Ti Laptop GPU with 12,227 MiB VRAM, CUDA-enabled
  PyTorch, approximately 370 GB free, and a passing GPU doctor check.
- Treat the old VPS resource blocks as historical. Qwen and published-JEPA work is now `ACTIVE`, but
  remains unvalidated until real direct runs produce committed metrics from clean code.
- Begin Qwen with Qwen3-0.6B to validate hooks, storage, interventions, and Windows portability before
  Qwen3-4B. Keep Qwen3-30B-A3B restricted to `gpu_cluster`.
- Normalize recorded relative metric paths in the reproducibility audit so Linux provenance can be
  checked on Windows. A fresh clone may skip ignored shard checksums, but any available shard must
  still match exactly.
- Pin the first Qwen target to public revision `c1899de289a04d12100db370d81485cdf75e47ca`
  and pass `token=False`; an invalid implicit local OAuth token returned HTTP 401 for a public model,
  and repository experiments must neither depend on nor modify user credentials.
- Require explicit training-split statistics or registered donors for Torch mean/resample operations.
  Batch-one self-means and self-resampling are invalid controls.
- Store Qwen intervention data as checksummed HDF5 shards, not monolithic NPZ. Keep prompt/donor
  split metadata in every record and choose output-logit coordinates using training prompts only.
- Attach a direct 5-percent finite-difference execution to every outcome so learned predictors must
  compete with a prompt-local baseline in the later evaluation.
- Freeze the observed Qwen dataset and test meta-models on three incompatible holdouts: unseen
  prompts, an unseen steering coordinate, and an unseen operation. Require two-of-three seed
  replication for numbered hypothesis decisions.
- Treat the direct-verification graph as a ranked-coordinate candidate, not a circuit, unless later
  necessity/sufficiency/faithfulness/minimality tests pass.
- Preserve the unchanged `LLM-IJEPA-001` recorded values and original preregistered decisions as
  history, but apply the v2 corrected disposition: H-LLM-01 `WITHDRAWN`; graph `REJECTED`. Neither
  the old nor corrective gates were revised after their scientific outcomes.
- Integrate LeWorldModel first as a source-informed small reproduction of selected design elements at official revision `8edfeb3...`
  instead of claiming benchmark equivalence without installing the larger external framework and
  datasets. Preserve its two-loss recipe and record every deliberate scaling difference.
- Do not checkpoint-select LeWM by prediction MSE alone: engineering checks showed that this rewards
  collapsed embeddings despite a high SIGReg penalty. Use the preregistered final training step.
- Reconstruct only the restricted action-embedding to final-predictor-site to planner graph. Since
  AdaLN action conditioning enters every predictor block, earlier blocks are alternatives rather
  than falsely asserted serial nodes.
- Retain `WM-LEWM-001` as a failed replicated mechanism result without threshold changes. Separate
  the 3/3 faithful-reproduction result and 2/3 selective planner-intervention result from the 1/3
  full hidden-patch/circuit result; reject the aggregate graph.
- Pin the bounded Qwen3-4B availability capture to revision `1cfa9a7...`, five residual-post sites,
  three semantic position selectors, 12 fixed prompts, and a 64 MB HDF5 budget. Commit the code and
  gates before downloading or observing the model output.
- Define repository completion through an executable audit of the explicit completion criteria.
  A passing audit must preserve the rejected Qwen/world-model circuit graphs and workspace null;
  it must not recast negative scientific outcomes as discoveries.
- Accept `LLM-QWEN-CAPTURE-001` as an Availability result after its clean run resolved the exact
  Qwen3-4B revision and passed row-count, budget, and checksum gates. Do not extrapolate it to
  autograd, causal intervention, feature, circuit, behavior, or workspace evidence at 4B scale.
- Accept the bounded repository completion claim only because the executable audit passed every
  explicit criterion from clean synchronized code. Keep `SMOKE_VALIDATED`, not
  `SCIENTIFICALLY_VALIDATED`: negative circuit/workspace outcomes and optional scale routes remain.

## 2026-07-20

- Use only the `cpu_vps` resource path for this run.
- Treat all new model-weight downloads, external datasets, CUDA, Docker, Conda, and large simulation
  suites as unavailable.
- Implement CPU smoke code with NumPy and standard-library infrastructure.
- Keep real Qwen and published JEPA checkpoints `BLOCKED_RESOURCE` until `gpu_12gb`.
- A later explicit user override permitted cached GPT-2 Medium and the minimal Transformers runtime
  under the final `AGENTS.md` CPU limits. It does not permit Qwen downloads or GPU claims.
- Use `SUMMARY.md` as the handoff log after each milestone.
- Preserve the original `WM-T0-002` artifact but withdraw its specificity interpretation: its
  “patch” result was assigned directly to the donor target.
- Validate workspace discovery against known shared and disjoint controls before applying it to a
  learned representation.
- Treat a compact five-consumer sensitivity subspace as a candidate only. It is not a workspace
  without direct ablation, PCA/random controls, controllability, selective necessity, and
  generalization.
- For `LLM-GPT2-002`, use direct held-out outcomes and compare against a prompt-local Jacobian. Do
  not infer a nonlinear advantage from beating only no-change or mean-effect.
- For `WM-T0-004`, freeze the deep predictor before fitting consumers, calibrate ensemble uncertainty
  on validation only, and replace projection-out controls with conditional donor resampling that
  explicitly measures activation-manifold distance.
- Do not call a shared Jacobian eigensubspace J-space. Anthropic defines J-space as a sparse
  nonnegative token-aligned frame and validates report, modulation, reasoning, reuse, and selectivity;
  the JEPA detector currently tests only a narrower functional analogue.
- Preserve the `WM-T0-004` null without threshold tuning. Conditional donor controls are retained as
  a useful method; the observed architecture is rejected as a workspace-discovery target because
  action dependence and uncertainty consumers fail before the workspace decision.
- For `WM-T0-005`, cross goals with dynamics modes and hold out one full composition. Require two of
  three seeds, random and local-tangent specificity, at least 50 percent task-counterfactual
  recovery, and a selective-necessity ratio of at least 1.25. Keep `workspace_found` false even if
  the narrower candidate passes.
- Preserve the `WM-T0-005` null without threshold or seed tuning. A future CPU JEPA study must change
  the architecture materially by jointly training task consumers or a planner; adding more post-hoc
  probes to the same representation is not a meaningful continuation.
- For `LLM-GPT2-003`, derive directions only from disjoint calibration prompts, orthogonalize them,
  train predictors on singles only, and reserve all composed targets for evaluation. Require direct
  large-single addition and prompt-local finite differences so a learned model cannot win against
  weak baselines alone.
- Preserve the `LLM-GPT2-003` null. Seen-prompt bilinear success is memorization evidence, not
  composition generalization. Do not add prompts or tune this observed grid; a continuation must
  change intervention class to replacement, resampling, patching, or feature clamping.
- `LLM-CAPITAL-PATCH-001` passed its frozen behavior eligibility gates. Freeze its 612-example grid
  and entity split for the learned study. Because exact-JVP full-vector MSE and candidate agreement
  rank controls differently, require both activation-space fidelity and behavior fidelity; do not
  optimize or report only the more favorable endpoint.
- The genuine target-encoder study uses the name `TargetEncoderInterventionJEPA`; the legacy class
  remains unchanged for historical checkpoint replay. Direct-effect labels are isolated to a
  train-only post-hoc linear decoder, and both oracle target-embedding decode and predicted decode
  are reported so representation failure cannot be hidden inside predictor error.
- Freeze the `LLM-TARGET-IJEPA-001` negative. Its oracle decoder shows that simply increasing
  predictor capacity cannot repair the observed held-out-entity failure. The next route must model
  context-conditioned causal geometry and keep activation, logit, and behavior endpoints separate;
  do not tune the same target encoder on the observed six test entities.
- For context geometry, treat activation differences as vectors and output gradients as covectors.
  Use their paired contraction `J D^T` for gauge-stable local effects. Separately pooled Euclidean
  reachability/observability spans are retained only as an explicitly permutation-insensitive
  anti-pattern; they cannot by themselves support a shared-subspace or workspace claim.
- V1 found no registered real pooling illusion and no context-specific finite-behavior advantage.
  Preserve both failures. The train-mean Jacobian advantage and extreme gauge sensitivity are new
  observations; only the gauge result was preregistered. Confirm the former on a separate split
  before interpretation, and never equate the invariant contraction with a unique activation basis.
- The population-Jacobian confirmation uses validation entities precisely because v1 used test
  entities for discovery. Its thresholds are openly post-discovery. Require continuous fidelity,
  discrete answer behavior, per-context replication, and answer-row specificity jointly; a favorable
  bootstrap interval or averaging curve alone cannot establish the claim.
- All population-Jacobian confirmation gates passed. Treat population averaging as the strongest
  continuous finite-logit baseline for this capital-patch regime, while retaining quadratic Taylor
  as the strongest discrete candidate baseline. The averaging algorithm is Jacobian Lens prior art;
  the repository contribution is a bounded causal-fidelity comparison and gauge audit, pending
  cross-task/model and JEPA replication.
- Port the finite-chord question to the frozen LeWorldModel reproduction before broadening the Qwen
  claim. Use only valid one-hot environment-action replacements, distinguish one-step from recurrent
  four-step endpoints, and make decoded physical effects primary because raw latent Euclidean error
  is gauge dependent. Freeze checkpoint hashes and a 12/5/5 goal partition; v1 may not inspect the
  five test goals. The inherited planner is evidentially ineligible below 60% direct competence.
- Preserve V1 as `REJECTED_NUMERICAL_GATE`; do not relax its 12-node threshold after observing
  stiffness. A separately committed v2 may use adaptive integration on the five untouched test
  goals and prospectively test the provisional within-context action-vertex mean. Do not carry the
  failed global-population hypothesis forward merely because shrinkage lowered MSE.
- Stop the vertex-mean test route before touching protected goals: its validation MSE benefit is
  compatible with scalar shrinkage and lacks replicated correlation gains. Use exposed validation
  goals only to calibrate a decoded path-cancellation profiler; require a separate clean commit for
  any future test-goal hypotheses and retain LIT-012/LIT-042 as adjacent prior art.
- Use element-symbol retrieval as the next independent Qwen relation. Calibrate layers only on
  Gold/Au, Silver/Ag, Tin/Sn, and Lead/Pb, then exclude them permanently. Freeze layers 18/21/24/26
  and 24/6/6 registered entities before execution. Do not call a late-layer transition novel:
  MechLens is direct prior art. Test only whether direct donor control coincides with a reversal in
  exact-local versus population finite-effect fidelity, with quadratic and semantic null controls.
- Freeze `LLM-ELEMENT-LAYER-GEOMETRY-001` as mixed. Direct donor control and late semantic
  population specificity pass, but the exact registered inversion and cross-relation conjunction
  fail. Do not lower the `0.25/0.60` ratios or create a posthoc coupling claim. Any continuation must
  preregister a new relation/model and distinguish known late crystallization from the narrower
  question of when context averaging helps finite causal transport.
- Use a tokenization-only state/postal-abbreviation roster for the next confirmation; inherit the
  element layer grid without another pilot. Require clean-answer competence and compare population
  transport against the better of exact local and quadratic/HVP-style correction. The new score may
  test only the prospectively frozen sign switch and control correlation; it may not reinterpret the
  failed element thresholds or claim novelty for population averaging, late crystallization, or HVP.
- Preserve state v1 as `REJECTED_BEHAVIOR_GATE`. Do not lower the clean-accuracy floor or shift its
  registered layer-24 gate to layer 26. The descriptive `0.949` control/advantage correlation can
  motivate only a new boundary-relative study whose prompt competence is calibrated on excluded
  entities before registration; never select v1 entities by their observed correctness.
- Select the one-shot prompt solely by clean accuracy on the 13 excluded states; freeze its 13/13
  winner before any target forward. Test boundary equality rather than hard-code the post-result
  layer 26, require a `0.90` clean floor, and preserve exact/quadratic/population comparators. Reusing
  v1 state identities limits independence and must remain explicit even though exact prompts are new.
- Freeze one-shot v2 as mixed: H-GEO-13 passes, but exact onset equality and the early-control gate
  fail. Do not reinterpret a one-grid-step validation lag as equality or relax layer-21 transfer.
  A weaker bounded-lag hypothesis may be tested only on an independent relation with new target
  entities/answer semantics and a preregistered competence gate.
- Use a state-answer-disjoint ISO country-code relation for that test. Choose its prompt only on
  seven excluded countries, retain the weak `5/7` calibration score, and reject below `0.90` target
  held-out accuracy. Freeze a one-grid-step maximum population lag, not equality; disclose that
  the lag and `0.70` correlation gate were selected after inspecting element/state artifacts.
- Freeze the country result as mixed: direct causal takeover and semantic population specificity
  generalize, but a population-after-control ordering does not. Validation population advantage
  precedes the 50% donor boundary by one grid step. Do not rescue H-GEO-14 with an absolute lag,
  lower control threshold, or removed `A_21` margin on these data; pivot to the independently
  protected recurrent-JEPA test goals.
