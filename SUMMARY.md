# SUMMARY

## 2026-07-21 — Corrective derivative and novelty audit

- Independent adversarial review found that the `LLM-IJEPA-001` “local Jacobian” is a one-sided
  5-percent secant executed in bfloat16. Its predicted effects are implausibly larger than the true
  effects on the frozen data, so H-LLM-01 was placed `UNDER_REAUDIT`; the old numbers are preserved but
  are not reliable nonlinear-advantage evidence.
- The class named `NeuralInterventionJEPA` is a supervised two-branch bottleneck MLP. It has no
  target encoder, stop-gradient/EMA target, or anti-collapse JEPA objective. New reporting calls it
  the legacy conditional bottleneck, and a genuine target-encoder Intervention-JEPA is a separate
  next milestone.
- Implemented and preregistered `LLM-QWEN-JVP-AUDIT-001`: full FP32 replay of the immutable 432-edit
  grid, exact autograd directional JVP, six symmetric central-difference scales, quadratic Taylor,
  BF16/FP32 drift, semantic deduplication, three refit seeds, and frozen numerical/claim gates.
- A tiny random Qwen3 end-to-end test confirms that the eager-attention exact JVP agrees with a
  symmetric central difference. The real audit must be committed and run from a clean worktree.
- The first clean real-audit attempt stopped before producing any intervention outcome: Transformers
  5.3's tokenizer regex check queried the Hub despite `local_files_only` and encountered the host's
  invalid OAuth token. The adapter now resolves the pinned local snapshot path before loading either
  model or tokenizer, avoiding the network-only compatibility branch. This is an execution fix;
  scientific thresholds and the frozen grid are unchanged.
- The clean rerun from `686368e` completed in `167.99` seconds but was correctly rejected: five of
  six numerical gates passed, while downstream semantic-endpoint max error `1.335e-4` exceeded the
  fixed `1e-5` limit. Exact JVP/central agreement itself was strong (median `2.49e-4`, p95
  `0.00381`). Preliminary raw MSE reversed the old result: quadratic `0.07870`, exact JVP `0.6143`,
  conditional bottleneck `3.1899`, and BF16 secant `120.8994`; zero learned seeds beat exact JVP.
- A no-write diagnostic localized the failed endpoint comparison to float32 cancellation of at most
  `4.768e-7` at replacement-source coordinates. This justifies a separately preregistered direct
  source-semantic validation, but cannot change v1's `REJECTED` numerical-gate result or decide H-LLM-01.
- Implemented/preregistered v2 after disclosing v1. It captures the actual edited source, requires
  exact adapter semantics, and bounds the algebraic direction endpoint by two scale-aware float32
  roundings. All JVP, nonlinearity, predictor, seed, split, and disposition thresholds remain v1's.
- V2 ran unchanged from clean commit `a779ff6` in `170.77` seconds. All numerical gates passed:
  exact direct-source error `0.0`, zero float32 endpoint-bound violations, and JVP/central relative
  error median/p95 `0.000249`/`0.00381`. The scientific result is negative: exact JVP MSE `0.6143`
  and quadratic `0.07870` beat the conditional bottleneck `3.1899`; zero of three seeds passed, and
  the registered finite-nonlinearity gate also failed. Restricted H-LLM-01 is `WITHDRAWN`.
- The corrected conclusion is that these selected Qwen3-0.6B residual edits are mostly local, with
  second-order transport explaining nearly all selected-target effect power. The old BF16 secant's
  MSE `120.8994` was a precision artifact, not evidence of a nonlinear JEPA advantage.
- Updated `AUDIT-COMPLETE-001` ran from clean synchronized commit `98a9e62`: all 14 criteria pass,
  including the corrected exact-JVP comparator/disposition, 68 tests, Ruff, diff checks, and 17
  audited metric/provenance pairs with four locally verified shard checksums.
- Literature review found that generic reachability/observability balancing is established prior
  art (control-theoretic DNN interpretation, empirical minimal realization, and CoBRAS). A future
  contribution must instead test context-conditioned finite-amplitude causal fidelity with direct
  behavior execution and cross-context pooling/permutation controls; no generic balancing novelty
  is claimed.
- Implemented/preregistered `LLM-CAPITAL-PATCH-001` to leave the mostly non-behavioral coordinate
  grid: 36 fixed single-token capitals, 24/6/6 disjoint recipient/donor entity splits, 612 full
  layer-21 residual donor patches, direct top-token behavior, exact JVP, central convergence, and
  quadratic Taylor. Japan/Canada/China/Kenya calibration examples are excluded from final data.
- The unchanged dataset ran from clean commit `95018cb` in `74.45` seconds and passed every
  numerical and behavior gate. Clean answer accuracy was `0.917/0.667/1.000` on train/validation/
  test; donor-answer transfer was `0.580/0.500/0.500`; and 93.6% of patches changed the top token.
  Exact JVP had `0.5296` normalized MSE and only 35.1% candidate agreement, while quadratic Taylor
  had `0.6391` full-vector normalized MSE but 74.3% candidate agreement. This is evidence that
  behavior fidelity and activation-space MSE can rank local approximations differently; it enables,
  but does not itself validate, the genuine target-encoder Intervention-JEPA study.
- Implemented and prospectively registered `LLM-TARGET-IJEPA-001`. A shared online residual encoder
  and intervention encoder predict a 32-dimensional EMA/stop-gradient embedding of the directly
  observed final residual. Variance/covariance losses and effective-rank gates test collapse; only
  afterward does a train-only ridge decoder map predicted plus clean target embeddings to hidden/
  logit effects. Three seeds face raw linear, PCA-bilinear, supervised MLP, legacy bottleneck,
  nearest-neighbor, corpus-average, sparse transport, exact-JVP, and quadratic controls. No real
  model fit may run until this code and its gates are committed/pushed.
- The unchanged study ran from clean `3086cd4` in `28.66` seconds. Zero of three seeds passed
  H-LLM-01B, H-LLM-02, or H-LLM-04. Predicted latent effective rank was `8.64–9.75`, but every EMA
  target latent was below the registered `8` floor (`6.81–7.28`). More decisively, oracle decoders
  given the true target embedding still had held-out normalized MSE `1.065–1.753`, localizing failure
  to entity-specific target geometry rather than only the predictor. The ensemble scored `0.930`
  normalized MSE, `0.483` correlation, and `0.20` answer-candidate agreement.
- Comparator rankings split by endpoint: exact JVP was best on full-vector fidelity (`0.599`
  normalized MSE), raw linear ridge was best on logit fidelity (`0.329`), and quadratic Taylor was
  best on direct answer-set behavior (`0.700` agreement). PCA-bilinear extrapolated catastrophically
  (`2.04e11` normalized MSE). This motivates context-conditioned, endpoint-explicit causal geometry;
  it does not support an Intervention-JEPA advantage or a circuit/workspace claim.
- Implemented/preregistered `LLM-CONTEXT-GEOMETRY-001`. For recipient context `r`, donor-direction
  matrix `D[r]`, and answer-logit Jacobian `J[s]`, it audits the paired contraction
  `K[r,s]=J[s]D[r]^T`. This quantity is invariant when activation vectors and gradient covectors are
  transformed dually, unlike naive separately pooled Euclidean subspaces. The study includes a
  two-context analytic pooling illusion, all 36 real Qwen context Jacobians, 256 fixed test-context
  derangements, a train-pooled Jacobian, direct finite-patch behavior, and numerical reconstruction
  of the already validated exact JVP. No novelty or mechanism claim is made before execution.
- The clean `49d68b7` run finished in `9.66` seconds and passed every numerical gate: full-Jacobian
  reconstruction of stored JVPs had median/p95 relative error `2.86e-7/4.58e-7`. H-GEO-01 and
  H-GEO-02 failed; H-GEO-03 passed. Real top-four pooled/matched/permuted overlap was only
  `0.04036/0.03403/0.03286`, so the preregistered real pooling gap was absent and context pairing
  barely exceeded the permutation null.
- The gauge stress exposed a concrete real-model failure of naive subspace geometry: an invertible
  diagonal coordinate change with condition number `96.4` moved pooled overlap from `0.04036` to
  `0.0003345`, while paired `J D^T` changed by only `1.68e-16` relative. This supports the diagnostic,
  not a semantic mechanism.
- Contrary to the context-specific hypothesis, the train-context mean Jacobian predicted held-out
  finite logit effects better than each recipient's exact local Jacobian: normalized MSE
  `0.358` versus `0.540`, correlation `0.885` versus `0.841`, and candidate agreement `0.500` versus
  `0.300`. A context-derangement null had worse continuous MSE (`0.983` mean) but higher average
  candidate agreement (`0.396`), further showing that discrete behavior and vector fidelity can
  disagree. This post-result pattern requires a held-out confirmation before a new claim.

## 2026-07-21 — GPU continuation begins

- Confirmed a clean, synchronized starting point at `99854eb` on `main`/`origin/main`.
- Detected an RTX 5070 Ti Laptop GPU with 12,227 MiB VRAM, CUDA-enabled PyTorch 2.10, Transformers
  5.3, 32 logical CPU cores, and roughly 370 GB free. The `gpu_12gb` doctor passes.
- The inherited full suite exposed a Windows portability failure: provenance paths recorded with
  `/` were compared to Windows `Path` strings with `\\`. The audit now compares normalized relative
  paths.
- Fresh clones intentionally lack ignored activation shards. The integration test now requires at
  least six manifest records to be either verified locally or explicitly reported missing; it does
  not mislabel missing bytes as verified.
- After the repair and GPU-doctor coverage, all 44 tests pass and the reproducibility audit reports nine metric/provenance
  pairs, zero errors, and seven skipped missing local shards.
- Regenerated the inherited 41-byte `uv.lock` and corrected the GPT-2 dependency group to include
  PyTorch and a bounded Transformers major version.
- At the continuation start, independent Qwen, world-model, and adversarial audits found that real
  Qwen and published-JEPA integrations were placeholders. The later entries below record their
  implementation, execution, and negative controls.
- Replaced the Qwen failure stub with a selected-site Torch/Hugging Face adapter. Offline tiny-Qwen3
  tests cover residual/attention/MLP/logit capture, clean replay equality, zero/mean/resample/patch/
  steer semantics, explicit donor/statistic requirements, and graph-preserving autograd.
- Pinned and preregistered `LLM-QWEN-001` at public Qwen3-0.6B revision `c1899de...` with a
  1,519,209,243-byte repository estimate. Do not run until this code milestone is committed/pushed.
- The first execution passed numerically but was rejected because the new runner collected
  provenance after writing an untracked metrics file. Fixed the ordering and pushed `0d6a37b`, then
  reran unchanged from clean code.
- Clean `LLM-QWEN-001` passes: exact deterministic replay; selected-logit autograd norm `0.944`;
  nonzero zero/mean/resample/patch/steer downstream hidden/logit effects. This meets real Qwen HF
  instrumentation only; intervention dataset, meta-model, and circuit verification remain.
- Implemented and preregistered `LLM-INTDATA-001`: 432 real direct outcomes, fixed 8/2/2 prompt
  splits, split-local donors, three layers, five operations, a direct small-step local baseline,
  and resumable SHA-256 HDF5 shards under a 128 MB cap. Commit before running.
- Ran `LLM-INTDATA-001` from clean commit `0aa80ac` in `33.85` seconds. All 432 effects were nonzero,
  17 changed the top token, and the 412,332-byte HDF5 shard checksum verifies. Prompt-local 5-percent
  linear MSE was `139.83`; this is a difficult regime, not yet a meta-model result.
- Implemented/preregistered `LLM-IJEPA-001`: three neural seeds, layer-transition and trajectory
  interfaces, exact checkpoint replay, nine baselines, prompt/feature/operation holdouts, and direct
  execution of every coordinate prediction on four new prompts.
- Ran `LLM-IJEPA-001` unchanged from clean commit `a54f2ed` in `8.23` seconds. All three seeds passed
  the registered H-LLM-01/02/03 decisions. Primary MSE/correlation were `3.923`/`0.677`; the model
  beat no-change, mean, linear, bilinear, MLP, local/corpus Jacobian, sparse-linear, and nearest-neighbor
  baselines on the primary holdout. Nearest-neighbor slightly beat it on resampling-holdout MSE
  (`2.095` versus `2.141`), an important boundary outside the registered gate.
- Directly re-executed all 16 ranked-coordinate edits on four unused prompts. Effect-size correlation
  was `0.673`, but the predicted winner (coordinate 128) was not the observed winner (coordinate 0),
  precision@1 was zero, and the effect was slightly below the random coordinate control. H-LLM-06
  failed and `qwen_meta_circuit` is recorded as `REJECTED`; no Qwen circuit or workspace is claimed.
- Verified the official LeWorldModel paper/code, MIT license, and source revision `8edfeb3...`.
  Replaced the placeholder with a faithful small reproduction retaining end-to-end pixels, action
  embedding, AdaLN-zero autoregression, next-embedding MSE, and SIGReg; added PixelTinyMaze, a typed
  adapter, layerwise probes, paired-action interventions, norm-matched planning controls, ensemble
  uncertainty, five consumers, and restricted circuit graph auditing.
- Short reduced engineering checks found and fixed a collapse-selecting validation bug: selecting by
  next-embedding error alone preferred a collapsed encoder despite SIGReg. The registered runner now
  uses the fixed final training step, as the official training recipe does. These checks are not
  scientific evidence. `WM-LEWM-001` is preregistered and must be committed before its full run.
- The first clean full attempt from `9c3239a` completed computation but failed before emitting
  metrics because `ResourceReport` was not JSON-serialized. Regenerable graphs were discarded; the
  one-line serialization repair was pushed as `4dbc388`, then the unchanged experiment was rerun.
- `WM-LEWM-001` completed from clean `4dbc388` in `54.50` seconds and failed its replicated causal
  gates. All three models passed the faithful-reproduction gates. Planner intervention gates passed
  seeds 103/107, but the full donor-decoding specificity/circuit gate passed only seed 107. The
  aggregate action-to-planner graph is `REJECTED`, not a circuit claim. All workspace candidates
  failed, while planted shared/disjoint controls behaved correctly. The intervention reduced rather
  than raised ensemble variance (`0.506` to `0.250`), a possible overconfidence failure mode.
- Replaced the obsolete Qwen capture blocker with a generic selected-site Hugging Face capture
  pipeline. The primary Qwen3-4B target is pinned to revision `1cfa9a7...` (8,060,926,626 repository
  bytes), captures only five layers and three semantic position selectors from 12 fixed prompts,
  and enforces a 64 MB checksummed HDF5 budget.
- Ran that capture from clean commit `55087ea`. The exact revision resolved, bfloat16 inference fit
  the RTX 5070 Ti, and all registered gates passed: 180 rows, 947,298 estimated uncompressed bytes,
  one 574,308-byte shard, SHA-256
  `cd1ef5e3a871740bfbd06e45c1a024257ba2ed9c1f0b1f6ac0bc2db1a11240cf`, runtime `473.14`
  seconds including the first download. This is Availability evidence only.
- The earlier `AUDIT-COMPLETE-001` run from clean synchronized commit `42492dc` passed all 14
  explicit bounded completion criteria with 63 tests, multi-seed results, direct Qwen execution,
  workspace controls, and rejected circuit graphs. It was superseded by the corrected 68-test audit
  above after the exact-JVP result. The required bounded suite is complete; this does not make either
  rejected circuit or the workspace null positive.

## 2026-07-20

### Done

- Read `AGENTS.md` and `VPS_RUNBOOK.md` fully.
- Confirmed current machine is the CPU VPS path: no GPU, 4 CPU cores, about 7 GB free at start.
- Added package metadata, resource profiles, package tree, resource guard, doctor CLI, typed interfaces, provenance helpers, and docs skeleton.
- Added standard-library unit/integration tests for config parsing, resource guards, CLI output, intervention serialization, and reproducibility scaffolding.
- Validated Milestone 0 with `doctor`, `python -m unittest discover -s tests -p 'test_*.py'`, `scripts/audit_reproducibility.py`, and `git diff --check`.
- Implemented deterministic Tier 0 generators: PointMass2D, BouncingBall2D, TwoBodyCollision, TinyMaze, and MiniPush.
- Implemented a tiny NumPy action-conditioned JEPA with fixed encoder, ridge-fitted predictor, save/load, named activation points, and no-action/shuffled-action controls.
- Implemented a random-shooting planner and PointMass closed-loop cost check.
- Added Milestone 1 tests. At that milestone the full command passed 14 tests; before the current
  `WM-T0-004` code milestone the full suite had 34 tests.
- Committed and pushed Milestone 1 code as `0cab19a6c39c98b59f1a2172eb11a64ec5a566a4`.
- Reran Tier 0 generation and `WM-T0-001` tiny JEPA smoke from clean committed code.
- Committed and pushed result summaries as `b0b7796`: `data/manifests/tier0_smoke_manifest.json`, `artifacts/metrics/tiny_jepa_smoke.json`, and `artifacts/metrics/tiny_jepa_smoke.provenance.json`.

### Discovered

- `README.md` initially only contained the project title.
- `pytest` is not installed, but NumPy is available system-wide.
- Historical decision: GPT-2 Medium was initially treated as `BLOCKED_RESOURCE`. A later explicit
  user override and the final `AGENTS.md` CPU guard permitted the cached model under strict limits.
- A first dirty-run tiny JEPA smoke pass succeeded; generated summaries were removed before the code commit so the final reported metrics can be rerun from committed code.
- Final smoke provenance reports `git_dirty: false` and code commit `0cab19a6c39c98b59f1a2172eb11a64ec5a566a4`.
- `WM-T0-001` beat the mean, no-action, and shuffled-action baselines on latent prediction, and the planner beat average random rollout cost. This is evidence level Availability only.
- Implemented intervention operators: zero, mean, resample, patch, replace_feature, steer, project_out, scale, and suppress_module.
- Implemented activation cache, stable site naming, finite-difference Jacobian, normalized patch recovery, ridge probes, random-label control, sparse dictionary, circuit graph JSON/GraphML, and mock transformer adapter.
- Implemented mock intervention-JEPA smoke runner and tests. Exploratory dirty-run metrics passed, then were removed before the code commit.
- Committed and pushed Milestone 2 code as `85c1dbfbe9c824bcca415af13f4a6f34acc95267`.
- Reran `LLM-MOCK-001` from clean committed code. Provenance reports `git_dirty: false`.
- Committed and pushed mock result summaries as `948347c`: `artifacts/metrics/mock_qwen_intervention_jepa_smoke.json` and `artifacts/metrics/mock_qwen_intervention_jepa_smoke.provenance.json`.
- Added `scripts/bootstrap_cpu.sh`, config `.gitkeep` files, and reporting stubs to complete the requested architecture.
- `uv` is not installed on the VPS. The bootstrap script checks resources and reports the exact install command instead of auto-installing tools.
- Historical audit reported `/root/.cache/pip` at about 4.4 GB and left it untouched. The later
  resource re-audit measured all of `/root/.cache` at about 233 MB, so that old size is no longer
  current.
- Historical `WM-T0-002` run reported action recovery `0.984`; adversarial review later found that
  result tautological and superseded it with the clean replay result documented below.
- Under explicit user override, created `.venv`, installed `transformers`/`safetensors`, downloaded `gpt2-medium`, and ran `LLM-GPT2-001`.
- GPT-2 Medium result: direct residual steering at `transformer.h.12.resid_post` changed logits with mean absolute delta `0.0797`; tiny intervention-JEPA MSE `0.00220` beat no-change `0.0114`.
- Historical provenance: the superseded `WM-T0-002` ran on `e5db938`; `LLM-GPT2-001` remains valid
  on `59795a4`.
- Committed and pushed GPT-2 metrics/docs as `a7ea6da`.

### Next Ideas

- On `gpu_12gb`, test GPT-2/Qwen activation replacement, resampling, patching, or sparse-feature
  clamps while retaining prompt-local Jacobians and held-out prompts as controls.

### Adversarial Milestone 3 Re-audit

- Re-read `AGENTS.md` and `VPS_RUNBOOK.md` completely and reran the baseline suite: 25 tests passed;
  doctor, reproducibility audit, and bytecode compilation passed.
- Found a material flaw in `WM-T0-002`: `action_patch` was assigned to the donor target rather than
  produced by an internal intervention. The `0.984` recovery is preserved but no longer interpreted
  as Specificity evidence.
- Implemented actual replay through `InterventionSpec` at `predictor.input`, nonzero donor sampling,
  and L2 norm-matched latent/action controls.
- Implemented a normalized multi-consumer Jacobian detector, compactness criterion, direct projection
  ablations, random and PCA controls, and known shared/disjoint detector controls.
- Added a preregistered `WM-T0-003` five-consumer study. It can identify a shared causal state
  subspace candidate, but it cannot declare a full workspace because key functional criteria remain
  untested.
- Reran corrected `WM-T0-002` from clean commit `315d8cf`. Actual `InterventionSpec` replay recovered
  the donor effect exactly (`1.0`, max error `0.0`) while L2 norm-matched controls did not. This is
  narrow Specificity evidence for the explicit action-input pathway, not a learned workspace.
- Ran `WM-T0-003` from clean commit `5223a54`. The detector passed planted shared/disjoint controls
  and proposed a `3/16` sensitivity subspace, but the preregistered candidate decision failed:
  uncertainty R2 was `-3.639`, PCA damage exceeded candidate damage, and random rollout controls were
  badly off-manifold. No workspace was found.
- This led to the separately preregistered `WM-T0-004` deeper predictor, calibrated ensemble, and
  in-manifold donor study recorded below.

### Current Handoff

- Worktree was clean at pushed commit `93431c3` before starting the next code milestone.
- After the `WM-T0-004` null, implemented and preregistered `WM-T0-005`: four goals crossed with
  two mass modes, one held-out composition, three seeds, a three-model calibrated ensemble, five
  frozen consumers, random and local-tangent controls, direct task-context counterfactual swaps,
  and multistep selective-necessity tests.
- The joint candidate gate requires valid action dependence, calibrated OOD uncertainty, all five
  held-out consumer heads, compact shared sensitivity, at least 50 percent donor recovery,
  specificity over both matched control families, and multistep selectivity on two of three seeds.
  The full workspace decision remains false by construction because reportability and published
  model replication are absent.
- The preregistered implementation was committed and pushed as `7a9e510`, then executed once from
  that clean commit. Provenance reports `git_dirty: false`; runtime was `57.51` seconds and disk
  remained near 40 GB free.
- `WM-T0-005` is a replicated null: zero of three seeds passed. Action MSE ratios were `0.712`,
  `1.012`, and `1.003` versus the registered maximum `0.50`. OOD uncertainty passed jointly only on
  seed 29. Every seed failed held-out transfer for the five-consumer set.
- Candidate dimensions saturated at `6/24`, while minimum consumer sensitivity capture was only
  `0.583`, `0.561`, and `0.574` versus `0.70`. Mean goal/dynamics counterfactual recovery was
  `-10.18`, `-17.38`, and `-3.11` versus the required `+0.50`.
- Seed 37 exceeded random-control p95 for counterfactual and rollout effects despite moving away
  from the donor. The absolute recovery gate and local-tangent controls correctly rejected this
  misleading relative win. No shared task-workspace candidate and no workspace were found.
- Practical conclusion: appending task context and fitting consumers afterward is not enough. A
  new JEPA architecture should train value/risk/action consumers or a planner jointly; do not tune
  the observed `WM-T0-005` seeds or thresholds.
- Implemented and preregistered `LLM-GPT2-003` without running it: two orthogonalized residual
  contrast directions from eight disjoint calibration prompts, six evaluation prompts, magnitude
  `0.5` finite differences, magnitude `6.0` singles/compositions, and exactly 72 direct outcomes.
- Predictors train only on 32 singles from four prompts. The primary test is eight compositions from
  two unseen prompts. Required controls are prompt-local and corpus additive Jacobians plus direct
  addition of the matching large single effects. Direction labels describe construction, not proven
  semantics. The full pre-commit suite passed 43 tests; doctor, reproducibility audit, compilation,
  and `git diff --check` passed. The implementation was committed and pushed as `1e57e30`.
- Executed `LLM-GPT2-003` once from clean commit `1e57e30`; provenance is clean, runtime was
  `392.85` seconds, estimated storage was 350,156 bytes under a 16 MB cap, and the ignored shard is
  24,933 bytes with SHA-256
  `adb4751bd3c9ca3c26139c47dac0423fd82b43515ccf9c56b5b706a78782f631`.
- Held-out compositions were almost exactly additive: direct interaction was `0.000429` of effect
  power. Prompt-local large-single addition MSE was `0.000179`; prompt-local finite-difference MSE
  was `0.000990`, correlation `0.9989`.
- Every learned held-out-prompt predictor failed: MLP MSE `0.725`, linear `0.775`, bilinear `1.346`,
  versus no-change `0.418`; learned correlations were negative. All registered H-LLM-01/02/03
  decisions are false.
- Bilinear prediction looked excellent on seen-prompt compositions (`0.00110` MSE, `0.9985`
  correlation), proving that a non-held-out evaluation would have yielded a false positive. Two of
  72 direct edits changed the top token, including one of 24 compositions, but direction semantics
  and mechanism reuse remain unvalidated.
- Practical conclusion: on this GPT-2 site, local Jacobians are useful but corpus/learned transport
  is not reusable across prompts. The next discriminating intervention should replace/resample
  activations or clamp learned sparse features, not add more steering directions to this grid.
- Post-run audit: 43 tests pass; nine metric/provenance pairs and seven local checksums validate;
  doctor reports 39.2 GB free; compilation and `git diff --check` pass. Commit and push the result
  milestone, then verify `main` is clean and synchronized with `origin/main`.
- Audited every repository Markdown file and all committed provenance logs. Corrected stale
  statements during the GPT-2 code milestone: the original action patch is superseded,
  `WM-T0-003` is no longer pending, and cached GPT-2 is allowed while Qwen remains blocked.
- Implemented `LLM-GPT2-002`: 288 batched direct interventions, local-only model loading, storage
  budget/checksum manifest, prompt/magnitude/layer holdouts, linear and bilinear predictors, trained
  MLP, nearest-neighbor and sparse-context baselines, and prompt-local/corpus Jacobians.
- Preregistered all splits and thresholds before execution. Do not change them after seeing results.
- At the preregistration checkpoint, 34 tests passed before code commit/push and clean execution.
  The measured result is recorded below.

### LLM-GPT2-002 Result

- Ran from clean commit `8fbab8c`; provenance reports `git_dirty: false` and no model download.
- Generated 288 direct residual-intervention outcomes and an ignored 87,378-byte float16 shard.
  Committed manifest checksum:
  `06a3a75c91076422d73fd62a85694c8366c4db854b2c303e203256adffe73abf`.
- Primary unseen-prompt/magnitude result: local Jacobian MSE `7.79e-7`, bilinear
  Intervention-JEPA `0.003499`, linear `0.006360`, no-change `0.006461`, MLP `0.007361`.
- H-LLM-01 nonlinear advantage failed. The narrow H-LLM-02 compression test passed because
  bilinear beat linear/no-change, but local direct linearization was about 4,490 times lower MSE.
- Held-out layer 18: bilinear MSE `0.01009` was worse than no-change `0.006228`; cross-layer transfer
  failed. Local Jacobian remained nearly exact (`3.71e-8`).
- No intervention changed the top token. This is hidden/logit causal mediation only, not behavior,
  semantic feature, J-space, workspace, Qwen, or consciousness evidence.
- Runtime was `647.85` seconds, exceeding the 10-minute CPU expectation by `47.85` seconds; record
  this as a resource-limit miss.
- Metrics, clean provenance, checksum manifest, and synchronized result docs were committed and
  pushed as `fdf6506`.
- The earlier next step was semantic/composed steering; `LLM-GPT2-003` has now tested and rejected
  that additive regime. On `gpu_12gb`, use activation replacement/resampling, sparse-feature clamps,
  and enough prompts/seeds to expose genuine nonlinearity. Keep prompt-local Jacobians as baseline.
- Strengthened `scripts/audit_reproducibility.py`: it now validates every metrics/provenance pair,
  clean commit metadata, JSON structure, and all available local dataset checksums instead of only
  checking that seven control-plane paths exist.
- The strengthened reproducibility audit and tests were committed and pushed as `3eedcb5`.
- Final handoff target: full suite, doctor, checksum/provenance audit, and `git diff --check` must
  pass; branch `main` must be clean and synchronized with `origin/main`.

### WM-T0-004 Implementation And Result

- Re-audited every Markdown handoff/registry and the committed metrics/provenance logs. Corrected the
  stale test count and the stale instruction to run the already-completed `LLM-GPT2-002`.
- Primary-source verification confirms that Anthropic's J-space is a sparse nonnegative token-aligned
  frame tested for report, directed modulation, internal reasoning, flexible reuse, and selectivity.
  A JEPA Jacobian eigensubspace is therefore only an analogue candidate, never equivalent by name.
- Added a NumPy-only two-hidden-layer action-conditioned JEPA whose hidden weights are learned and
  exposed at `predictor.hidden1` and `predictor.hidden2`; its fixed encoder prevents JEPA collapse.
- Added a five-member bootstrap ensemble, split interval calibration, held-out coverage/NLL, OOD rank
  AUC, and uncertainty/error correlation. No uncertainty claim is accepted unless all registered
  thresholds pass.
- Added conditional donor resampling: candidate coordinates come from validation activations matched
  in the orthogonal complement. Random/PCA controls must match candidate density distance and
  perturbation magnitude before they count.
- Preregistered `WM-T0-004` before execution. It fits all five consumers after freezing the predictor,
  discovers candidates separately at two depths, and requires direct plus multistep specificity.
- Resource incident: free disk fell to 3.8 GB because broken external `AppTurnos` launchers produced
  about 37 GB of PM2 restart-loop logs. Per user instruction, the systemd service is now inactive and
  masked, its PM2 entry and saved startup record are removed, port 3001 is closed, and 40+ GB is free.
  The source/database remain as a recoverable archive outside this repository.
- Committed and pushed the preregistered implementation as `6785fb1`, then executed once from that
  clean commit without changing thresholds. Runtime was `31.22` seconds and provenance is clean.
- The action-conditioning gate failed: primary MSE was `0.003416` versus shuffled-action `0.004994`,
  ratio `0.684` against the registered maximum `0.25`. Shuffling hurts, but not enough for the strong
  causal-use claim.
- In-distribution uncertainty calibration worked: test coverage was `0.887` for a 90 percent target,
  and uncertainty/error Spearman was `0.698`. The joint uncertainty claim failed because OOD AUC was
  only `0.574` versus `0.65` and hidden uncertainty-head R2 was `-1.327`/`0.232` versus `0.50`.
- Neither hidden site produced the registered compact sensitivity candidate: minimum five-consumer
  capture was `0.635` and `0.701`, below `0.75`.
- The old off-manifold confound is repaired. Candidate density ratios were `1.029`/`1.018`, and
  `63/64` plus `64/64` random controls matched density and magnitude. Candidate direct damage was
  below random-control p95 at both sites (`3.652 < 9.252`; `1.239 < 4.369`).
- Multistep amplification ratios (`6.03`, `1.95`) looked large in isolation but candidate rollout
  damage also remained below matched-control p95. This is nonspecific accumulation, not selective
  necessity. PCA controls were too large to be matched and correctly did not decide the claim.
- Scientific result: no shared causal subspace and no JEPA workspace. Restricted H-WM-05, H-WM-06,
  and H-WM-08 are false for this run. The reusable output is the conditional donor control method.
- ELI5 novelty boundary: this repository now has a fairer way to ask whether many JEPA outputs share
  a special internal whiteboard without smashing the model's activations into nonsense. The answer
  for this tiny model is no. This is methodologically useful and falsification-first, but it is not
  state-of-the-art performance, a published discovery, or evidence that larger JEPAs lack such a
  mechanism.
- Best next JEPA direction: preregister a goal-conditioned multi-task model where action and task
  variables are indispensable, use local-tangent PCA controls, and replicate across seeds. Do not
  tune `WM-T0-004` after seeing its null.
- Final pre-commit validation: 37 tests pass; seven metric/provenance pairs and six local checksums
  audit without errors; compilation and `git diff --check` pass; the CPU guard reports 40.1 GB free.
- Exact clean-code reproduction command:
  `PYTHONPATH=src ./.venv/bin/python scripts/run_experiment.py --config configs/experiments/manifold_workspace_study.yaml`.
