# TODO

## 2026-08-27 recovery

- [x] Honest reproducibility audit (move unprovenanced metric JSON; do not fabricate sidecars).
- [x] Ruff default-rule pin and tracked unused-import/variable fixes.
- [x] CPU CI for unit/scientific/integration, Ruff, audit, competence entrypoints.
- [x] Adjudicate calibration-only competence recovery without rescuing V3.
- [x] Preregister `QWEN-BINDING-COMPETENCE-CONFIRM-001` with frozen renderer and fresh tokens.
- [x] Execute confirmation on CUDA Python 3.14 after the protocol commit; do not retune 0.90.
  Result: `COMPETENCE_CONFIRMATION_PASSED` (clean 1.000, direct 0.9896).
- [x] Coalition/equivalence IBD evaluator (`CRCT-COALITION-IBD-001`); HARD-002 seeds blocked.
- [x] Do not execute `CRCT-COALITION-IBD-002` (inadequate protocol; leave `PREREGISTERED_NOT_RUN`).
- [x] Freeze then run `CRCT-COALITION-IBD-003` once; do not retune after confirmation.
  Result: `MECHANISM_RECOVERY_PASSED`.
- [ ] `CRCT-JEPA-ACTION-DELTA-001` is preregistered. Run development on 43/47/53
  only after the freeze commit. Do not open confirmation unless development
  passes. Do not retune gates.
- [x] Commit then write coalition development/confirmation metric JSON with real provenance.
- [x] Draft `LLM-QWEN-BINDING-ALGEBRA-004` / `CRCT-QWEN-BRIDGE-003` without executing them.
- [ ] Execute 004 only after a later authorization commit on origin.
- [x] Intervention-JEPA fair-baseline policy module.
- [x] Preregister secondary LeVLJEPA MiniPush factorial; do not download.
- [x] Write Platonic WM + LeFlow integration plan; do not run those papers' experiments yet.
- [x] Preregister CPU `WM-PLATONIC-MKNN-001` and `WM-LEFLOW-AMORTIZE-001`
  before seeing outcomes. Do not open stitching.
- [x] Execute MKNN-001 on frozen confirmation seeds; keep a negative if it fails.
  Result: `TRANSITION_NEIGHBORHOOD_ALIGNMENT_PASSED` with the encoder-geometry
  caveat recorded. Gates were not retuned.
- [x] Execute AMORTIZE-001 only after MKNN adjudication with no P0/P1 integrity issue.
  Result: `NEGATIVE_RESULT` (success saturated; N=64 slower than shooting).
  Gates were not retuned.
- [x] Analyze T1×T2 without retuning; untrained-predictor post-hoc is not a
  gate. Stitching remains closed. See
  `docs/WM_PLATONIC_LEFLOW_CPU_ANALYSIS_2026-08-27.md`.
- [x] Execute `WM-AMORTIZED-PLANNING-MINIPUSH-002` after its protocol
  commit. Qualification passed; confirmation is `UNINFORMATIVE_SUBSTRATE`.
  Gates were not retuned. T2 was not mutated.
- [ ] Do not open `WM-LEFLOW-TRANSFER-001`. MiniPush-002 did not make search
  useful. Rectified Flow is not justified. CRCT-on-transfer stays gated.
- [ ] `WM-AMORTIZED-PLANNING-REACHABLE-003` remains
  `DRAFT_NOT_PREREGISTERED`. Do not freeze, qualify, or confirm it until
  an explicit authorization commit.

## 2026-08-17 CRCT adjudication

- [x] Freeze `CRCT-STAGE0-001` as synthetic positive-control evidence only.
- [x] Execute `CRCT-STAGE0-HARD-002` on untouched primary seeds 1009/2027/4093 with
  validation-only discovery and controls frozen before IID/OOD generation.
- [x] Retain HARD-002 as `NEGATIVE_RESULT`; do not retune its thresholds.
- [ ] Add group/coalition-aware development metrics that distinguish full graph recall from
  epsilon-functional sufficiency, necessity, redundancy-group coverage, cancellation-group
  coverage, and equivalent circuit classes.
- [ ] Freeze that metric/selector design on non-primary development plants before generating any
  successor primary seeds.
- [ ] Integrate exact patching plus HVP reliability and QK/head routing on already-open Qwen
  capital data as development-only plumbing.
- [ ] Add Qwen-Scope SAE, raw component, transcoder/attribution-graph, and SWD substrate adapters
  with original-model intervention validation.
- [ ] Complete outcome-blind execution code for `LLM-QWEN-BINDING-ALGEBRA-002`; open learned
  residual prediction only if Phase 0 passes its independent residual-nonlinearity gate.

## July 2026 continuation portfolio

- [x] Review FlowMimic and Masked Visual Actions from their primary PDFs; register their exact
  contribution, resource boundary, limitations, and control value without treating either as
  causal-mechanism evidence.
- [x] Choose competence/geometry gates before model scaling: EB-JEPA before a larger world model,
  binding algebra and official Jacobian-Lens controls before Qwen3-1.7B, then causal 1.7B before 4B.
- [ ] Implement and unit-test sensitivity-balanced causal modes with planted recovery,
  biorthogonality, rank truncation, gauge covariance, and oblique-projector controls.
- [ ] After EB-JEPA competence, preregister finite gate/path/subspace patches against
  PCA/delta-PCA/gradient-only/random/norm/condition-matched controls.
- [ ] Build the MiniPush vector-versus-masked-visual-action and
  forward-versus-forward/inverse factorial with fixed data scales and anticausal common-warp
  controls.
- [ ] Reproduce official Jacobian-Lens convergence/rotation/row-null controls on Qwen3-0.6B, then
  preregister selected-layer Qwen3-1.7B plus Qwen-Scope sparse-feature interventions.
- [ ] Implement a differential-plus-learned-residual operator only if binding-algebra Phase 0
  leaves residual nonlinearity beyond full quadratic and relinearized-JVP baselines.

## GPU Transition (2026-07-21)

- [x] Refresh the July-2026 primary-source frontier and register EB-JEPA, Qwen-Scope, Circuit
  Tracing, NLA, causal physics steering, faithfulness/path patching, and induction-head prior art.
- [x] Adversarially reject the first Qwen population-mediation draft before preregistration.
- [x] Implement and test ordered multi-site Qwen patch/restore with exact upstream treatment and
  downstream clean-restoration replay.
- [ ] Keep the first causal localization study module-only; add pre-`o_proj` head hooks only in a
  separate milestone with exact attention reconstruction tests.
- [x] Preregister the corrected binding mediator study with independent episode units, train-only
  `k <= 4`, grouped intervals, full-vocabulary behavior, and direct necessity/sufficiency controls.
- [x] Run the tokenizer-only audit from clean `4e6624f`: all 560 episodes have exactly two changed
  positions, lengths 35/36, balanced query positions, and frozen hash `3ac7a80d...ebaf59`.
- [x] Supersede v1 before model execution after red-team found unpaired paraphrase, FP16 causal
  states, and a NaN false-pass; retain its token audit as engineering history only.
- [x] Preregister/harden v2 with exact test/paraphrase pairing, FP32 states, finite/count/group,
  runtime, content-hash, and HDF5-readback gates.
- [x] Implement and test the complete Qwen population/local/HVP/custom-AtP*/probe/magnitude
  evaluator, train-only direct-prefix selection, restoration, five direct comparators, 128 matched
  random sets, four specificity controls, self-hashed plan, and checksum-bound protected resume.
- [x] From clean `bca50e3`, run the v2 tokenizer audit: all 560 rows pass all eight gates; episode
  hash `96dc6320...f3be`; no Qwen forward executed.
- [x] Authorize capture only in a separate pre-outcome commit bound to evaluator `53cd69d` and
  token-audit result `f19d308`; downstream ranking and hypothesis decisions remain unauthorized.
- [x] Execute the 560-row FP32 capture from clean `2bf7e69`: integrity and exact replay pass, but
  task competence fails (`INELIGIBLE_TASK`); retain metrics/manifest/provenance without rescue.
- [x] Stop v2 before calibration, train rankings, or protected mediation; H-LLM-15/16 are undecided.
- [x] Implement and test a disclosed post-hoc read-only format diagnostic for token `17607`
  collapse versus the partially competent paired paraphrase. It verifies capture identity, exact
  factor/answer pairing, token mapping, finitude, and a clean committed worktree.
- [x] Execute the diagnostic from clean `31d3464` and publish metrics/provenance: the exact
  paraphrase pairing improves four-value clean/donor accuracy by `0.396/0.604` with zero paired
  losses, but cannot modify/rerun v2 or select a v3 prompt.
- [ ] If continuing binding mediation, preregister v3 with new episodes and a behavior-competent
  prompt family before any capture; preserve v2 as the negative task-eligibility result.
- [x] Preserve historical `LLM-QWEN-BINDING-ALGEBRA-001` as
  `SUPERSEDED_PRE_TOKENIZER_PRE_MODEL`; no tokenizer/model outcome was opened and every scientific
  threshold carries unchanged into v2.
- [x] Preregister/harden `LLM-QWEN-BINDING-ALGEBRA-002` with exhaustive matrix composition,
  primitive-only action input, cumulative prefix deltas from one clean origin, explicit
  identity/inverse cases, separate phase roots, affine/`S4`-equivariant controls, and a
  checksum-bound semantic roster.
- [x] Implement the pure binding-algebra protocol and deterministic tests for all 24 permutations,
  composition, inverse, conjugacy-class counts, minimal transposition rollouts, globally disjoint
  token alphabets, independent episode units, and exact test/paraphrase pairing.
- [ ] Keep `execution_authorized=false`. Implement and freeze the algebra tokenizer audit, exact
  layer-0 multi-position replay, FP32 trajectory capture, full cross-term HVP and oracle sequential
  JVP baselines, fixed delta-trajectory meta-model, predicted-state patching, episode bootstrap, and
  fail-closed aggregation before any Qwen forward.
- [ ] After a clean commit and push, open validation Phase 0 only. If its interaction/quadratic gates
  fail, record the local-differential null without training or opening protected outcomes; otherwise
  execute the already-frozen three-seed model and only then open test plus paired paraphrase.
- [x] Pin official EB-JEPA and implement/test its source-contract adapter with exact one-layer GRU
  gate decomposition and replay tolerance `1e-6`.
- [x] Retain `WM-EBJEPA-CONTRACT-001` from clean `979c2d6`; all source, shape, reconstruction, and
  targeted-intervention gates pass.
- [x] Build isolated Python 3.12/Torch 2.6+cu126 and Torch 2.10+cu128 runtimes. The exact pin fails
  all three SM120 kernels; the compatible deviation passes all three.
- [x] Retain `WM-EBJEPA-RUNTIME-001` from clean `15d88ce`; all eight frozen two-runtime gates pass.
- [x] Freeze/install the Two Rooms dependency closure without replacing compatible Torch; record
  that upstream omits scipy, pandas, and PyYAML from the required path.
- [x] Retain and supersede clean `WM-EBJEPA-INTEGRATION-001`: its original gates passed, but missing
  Python RNG seeding and an independent replay make it nondeterministic/ineligible.
- [x] Commit/push and retain deterministic `WM-EBJEPA-INTEGRATION-002` from clean `9a18008` with exact cross-process
  batch/model/loss/action fingerprint replay.
- [x] Retain the preregistered 32-seed CEM/MPPI action-constraint confirmation from clean `da30443`;
  CEM is `0/32`, MPPI `32/32`. Do not silently patch
  the upstream planner before comparing original and corrected planning.
- [x] Commit/push and retain `WM-EBJEPA-PLANNER-CONFIG-001` from clean `4f0cc80`: all six gates
  pass; MPPI silently ignores YAML `var_scale=1.5` and executes with default `max_std=2.0`.
- [x] Implement/preregister a separately named constraint-corrected MPPI adapter with exact
  no-bound official-equivalence and within-bound cost/return tests.
- [x] Commit/push and retain `WM-EBJEPA-MPPI-CORRECTION-001` from clean `f58308a`; all five gates
  pass across 32 seeds with exact unbounded equivalence and zero bounded violations;
  preserve the original official implementation as the reproduction baseline.
- [x] Commit/push and run `WM-EBJEPA-TRAIN-RESOURCE-001` unchanged from clean `fed920e`: batch 384
  peaks at 5.82 GB reserved; the default compile wrapper captures zero frames/graphs on `unroll`.
- [x] Retain the interrupted `WM-EBJEPA-TRAIN-001` launch from clean `5065108`: seed 1 completed
  epochs 0--11 and all 13 checkpoint identities revalidate; the portfolio stopped before seed 1000
  because the repository became dirty. Training loss alone decides no competence.
- [x] Harden completed-seed reuse against stale status JSON, missing/changed checkpoints, wrong
  recorded epochs, nonfinite model tensors, and a nonidentical `latest` checkpoint. Make the bounded
  MPPI arm primary for mechanistic eligibility while retaining the official arm as a paired audit.
- [x] From clean `66933fe`, validate seed 1000 as `VALIDATED_NOT_TRAINED`; repository, pinned
  upstream, source config, and compatible Torch 2.10 runtime all match.
- [ ] When `nvidia-smi` shows safe headroom beyond the measured 5.82-GB batch reservation, resume
  seeds 1000/10000 with `python scripts/run_eb_jepa_training_portfolio.py --config
  configs/experiments/eb_jepa_two_rooms_training.yaml`; retain logs/status and do not launch while
  competing user workloads occupy roughly half the 12-GB GPU.
- [ ] Retain all 12 epoch checkpoints for every training seed and evaluate frozen epochs 9/10/11
  under separately labeled planner arms.
- [ ] Reproduce the competent Two Rooms planner across three seeds before registering its recurrent
  action-circuit audit.

- [x] Adversarially audit the Qwen Jacobian baseline, model naming, behavior changes, and novelty.
- [x] Mark H-LLM-01 `UNDER_REAUDIT`; preserve the old result without continuing to claim it.
- [x] Implement, unit-test, and preregister `LLM-QWEN-JVP-AUDIT-001` with exact FP32 JVP and central
  convergence controls.
- [x] Commit/push the audit preregistration and execute v1 unchanged from clean commit `686368e`.
- [x] Retain v1 as `REJECTED` on its numerical endpoint gate; do not relax that threshold.
- [x] Preregister v2 source-semantic gates after disclosing the v1 result and cancellation diagnosis.
- [x] Commit/push v2 and execute it unchanged from clean commit `a779ff6`; all numerical gates pass.
- [x] Withdraw restricted H-LLM-01 after 0/3 seeds beat exact JVP/quadratic Taylor.
- [x] Revise and rerun the executable completion audit; latest clean run `3593475` passes all 14
  criteria, 105 tests, Ruff, and 28 metric/provenance pairs.
- [x] Implement and preregister a genuine target-encoder/stop-gradient Intervention-JEPA with
  anti-collapse tests and strong behavior/local baselines.
- [x] Commit/push and execute `LLM-TARGET-IJEPA-001` unchanged from `3086cd4`; retain the 0/3
  negative, rank-diversity failure, oracle-decoder failure, and divergent baseline rankings.
- [x] Implement/preregister a split-safe semantic donor-answer dataset with aggregate behavior gates.
- [x] Commit/push and execute `LLM-CAPITAL-PATCH-001` unchanged from `95018cb`; all gates pass.
- [x] Implement/preregister context-paired causal geometry with analytic pooling, 256 derangements,
  dual-coordinate gauge checks, manifold donor chords, and finite behavior endpoints.
- [x] Commit/push and execute `LLM-CONTEXT-GEOMETRY-001` unchanged from `49d68b7`; retain failed
  pooling/context-specificity gates and the positive gauge diagnostic.
- [x] Preregister validation-only confirmation of the post-result train-mean-Jacobian advantage with
  averaging-size, per-context, bootstrap, answer-row, and discrete/continuous endpoint controls.
- [x] Commit/push and execute `LLM-POPULATION-JACOBIAN-001` unchanged from `3725714`; all three
  validation confirmation gates pass.
- [x] Implement and preregister `WM-POPULATION-JACOBIAN-001` on three frozen LeWorldModel seeds,
  valid one-hot action swaps, held-out goals, path integration, decoded physics, and gauge controls.
- [x] Commit/push and execute `WM-POPULATION-JACOBIAN-001` unchanged from `89b2e14`; retain its
  numerical rejection, underresolved path integral, ineligible planners, and untouched test goals.
- [x] Reject the naive action-vertex-mean continuation after validation shrinkage/correlation
  controls; preserve all five test goals.
- [x] Implement a validation-only decoded action-path cancellation calibration with composite
  quadrature, refinement checks, and action-pair-stratified nulls.
- [x] Commit/push and run `WM-ACTION-PATH-CALIBRATION-001`; retain its horizon-four numerical
  underresolution and weak-but-consistent stratified association as calibration only.
- [x] Implement separately recorded `WM-ACTION-PATH-CALIBRATION-002` with the same validation chords
  and 512/1024-node refinement; retain the hard split lock and empty decisions.
- [x] Finish calibration v2 and retain it as numerical calibration only; do not touch protected
  test goals even if it converges because cancellation/local error share a denominator.
- [x] Before reading v2, adversarially reject and remove the proposed denominator audit: it lacks
  scalar path-length refinement, raw norms, dense within-pair support, and a valid joint conditional
  null. Keep protected test locked and close the current family.
- [ ] Reopen action-path inference only if prioritized as a materially new prospective acquisition
  storing path length at both resolutions, unclamped norms, many chords per pair, row-level split
  guards, and a preregistered joint conditional null; do not derive it from v2.
- [x] Draft `papers/causal_workspace_jepa.tex` with equations, evidence-level result tables,
  falsifications, novelty boundary, limitations, and exact artifact/commit references.
- [x] Compile the draft against the repository-local `papers/references.bib` and ignore generated
  PDF/auxiliary products.
- [x] Replace the manuscript's provisional calibration-v2 row with the final clean-run metrics,
  complete adversarial/doc consistency review, then commit and push the paper milestone.
- [x] Add atomic action-path checkpoints per completed seed/horizon, bound to exact config bytes and
  source commit, with fail-closed resume tests and explicit final provenance.
- [ ] Benchmark safe `jacobian_outer_batch_size`/`jacobian_chunk_size` pairs on the RTX 5070 Ti and
  preregister a bounded batching choice before another high-resolution action-path launch.
- [x] Replicate population-versus-local finite transport on the element-symbol relation; retain the
  failed strict inversion and positive late semantic-specificity decisions separately.
- [ ] Confirm the surviving causal-control/late-population association on a behavior-competent,
  newly preregistered relation or model before any general or SOTA claim.
- [x] Implement and preregister `LLM-STATE-LAYER-GEOMETRY-001` on an untouched tokenization-only
  roster with exact local, quadratic/HVP-style, population, averaging, and semantic-null controls.
- [x] Commit/push and execute state v1 unchanged from `27ebe43`; retain its behavior-gate rejection,
  numerically valid but non-evidential layer-26 pattern, and undecided hypotheses.
- [x] Calibrate a higher-competence prompt using only excluded entities, then preregister a fresh
  prompt/task before any registered forward; do not filter v1 entities by observed correctness.
- [x] Calibrate five prompts on 13 excluded states; select the 13/13 one-shot format without any
  target one-shot forward.
- [x] Implement and preregister the boundary-relative `LLM-STATE-ONESHOT-LAYER-GEOMETRY-001`.
- [x] Commit/push and execute the one-shot study unchanged from `c1daa46`; retain H-GEO-13 and the
  failed early-control, validation-boundary-equality, and cross-relation gates.
- [x] Select a state-answer-disjoint ISO-country roster and calibrate five prompts on seven excluded
  countries without forwarding any of the 36 targets.
- [x] Implement and preregister the independent zero-or-one-grid-step country-code lag study with a
  `0.90` held-out competence gate and frozen element/state audits.
- [x] Commit/push and execute `LLM-COUNTRY-CODE-LAYER-GEOMETRY-001`; retain the documented
  compatibility retry, positive H-LLM-14/H-GEO-15, and failed H-GEO-14/H-CROSS-06 without rescue.
- [x] Use four permanently excluded element-symbol facts to calibrate a fixed four-layer grid and
  audit late-crystallization/AtP prior art without touching the registered 36-entity roster.
- [x] Implement and preregister `LLM-ELEMENT-LAYER-GEOMETRY-001` with direct behavior, full selected-
  logit Jacobians, quadratic/population controls, dual entity splits, and answer-row nulls.
- [x] Commit/push and execute the element-layer study unchanged from `5d8de9a`; H-LLM-08 and
  H-GEO-09 pass, while H-GEO-08 and H-CROSS-03 fail.

- [x] Re-read `AGENTS.md`, `VPS_RUNBOOK.md`, `SUMMARY.md`, repository docs, and current Git state.
- [x] Verify RTX 5070 Ti/CUDA, RAM/CPU, disk, Python, Transformers, and the `gpu_12gb` doctor.
- [x] Run the full pre-change test and reproducibility baseline.
- [x] Repair Windows provenance-path comparison and fresh-clone checksum expectations.
- [x] Implement and offline-test the Hugging Face Qwen adapter, selected hooks, interventions, and autograd.
- [x] Implement and test the bounded resumable sharded HDF5 activation store.
- [x] Execute the preregistered Qwen3-0.6B instrumentation smoke from clean commit `0d6a37b`.
- [x] Preregister the 432-outcome Qwen dataset grid, splits, donors, local probes, and storage guard.
- [x] Commit the dataset generator, then execute it once from clean code.
- [x] Commit and execute `LLM-INTDATA-001` from clean commit `0aa80ac`; validate its shard checksum.
- [x] Train/evaluate Intervention-JEPA and all required strong baselines on held-out real-Qwen effects.
- [x] Directly execute ranked meta-model circuit candidates; reject the failed precision@1 candidate.
- [x] Implement and preregister three-seed neural Intervention-JEPA, trajectory variant, baselines,
  sparse transport, checkpoint replay, and independent direct-verification logic.
- [x] Commit `LLM-IJEPA-001`, then execute once without changing thresholds.
- [x] Integrate and execute at least one published action-conditioned JEPA or faithful reproduction.
- [x] Implement, source-pin, test, and preregister the source-informed small LeWorldModel reproduction.
- [x] Commit `WM-LEWM-001`, then execute its three seeds and restricted circuit audit unchanged;
  retain the failed replicated causal/circuit gates and rejected graph.
- [x] Run `AUDIT-COMPLETE-001` from clean synchronized commit `42492dc`; all 14 criteria pass.
- [x] Replace the Qwen activation-capture blocker with a bounded generic Hugging Face implementation.
- [x] Execute the pinned Qwen3-4B selected-site capture from clean commit `55087ea`; verify the
  resolved revision, 180 rows, 574,308-byte shard, and SHA-256 manifest.

## Milestone 0

- [x] Read local instructions.
- [x] Add resource profiles.
- [x] Add doctor CLI.
- [x] Add typed interfaces.
- [x] Run tests.
- [x] Run `git diff --check`.
- [x] Commit and push.

## Milestone 1

- [x] Implement Tier 0 generators.
- [x] Add Tier 0 smoke config and manifest update.
- [x] Implement tiny NumPy JEPA.
- [x] Add planner and smoke experiment.
- [x] Add tests.
- [x] Run tests.
- [x] Run `git diff --check`.
- [x] Commit and push code.
- [x] Rerun smoke from committed code.
- [x] Commit and push metrics/docs.

## Milestone 2

- [x] Implement intervention operators.
- [x] Implement probes and finite differences.
- [x] Implement circuit graph schema.
- [x] Implement mock Qwen adapter and Intervention-JEPA smoke.
- [x] Add tests.
- [x] Run tests.
- [x] Run `git diff --check`.
- [x] Commit and push code.
- [x] Rerun mock smoke from committed code.
- [x] Commit and push metrics/docs.

## Milestone 3

- [x] Register and run Tier 0 mechanistic study.
- [x] Test action displacement decodability.
- [x] Test action-coordinate patch specificity against controls.
- [x] Evaluate workspace criteria and record null result.
- [x] Run GPT-2 Medium hidden-state intervention smoke under user override.
- [x] Commit and push Tier 0 mechanistic metrics.
- [x] Commit and push GPT-2 metrics/docs.
- [x] Audit the original action-patch implementation and withdraw the unsupported specificity claim.
- [x] Implement a replayable action-input patch with norm-matched controls.
- [x] Implement shared-subspace discovery with positive and negative controls.
- [x] Preregister `WM-T0-003` thresholds and splits.
- [x] Commit and push repaired code before execution.
- [x] Rerun corrected `WM-T0-002` from committed code.
- [x] Run `WM-T0-003` from committed code.
- [x] Commit and push `WM-T0-003` null metrics.

## Milestone 3 Follow-up

- [x] Run `WM-T0-003` from committed code and preserve its null result.
- [x] Implement a split-calibrated deep ensemble uncertainty pipeline for `WM-T0-004`.
- [x] Implement conditional donor resampling with activation-density and perturbation matching.
- [x] Implement a deeper learned predictor with genuine internal sites and independently trained consumers.
- [x] Preregister `WM-T0-004` splits, thresholds, OOD shift, and rejection rules before execution.
- [x] Commit and push the `WM-T0-004` implementation (`6785fb1`).
- [x] Execute `WM-T0-004` from the clean committed code without changing thresholds.
- [x] Record the null result and rerun the full audit.
- [x] Commit and push the `WM-T0-004` result milestone.

## GPT-2 Medium Follow-up

- [x] Audit `LLM-GPT2-001` leakage, split, site, and baseline limitations.
- [x] Preregister `LLM-GPT2-002` before execution.
- [x] Implement batched direct intervention data and storage/checksum guard.
- [x] Implement linear, bilinear, trained MLP, nearest-neighbor, sparse-context, local Jacobian, and
  corpus-averaged Jacobian baselines.
- [x] Add offline split, predictor, and resource-limit tests.
- [x] Commit and push `LLM-GPT2-002` code before execution.
- [x] Run `LLM-GPT2-002` from clean committed code.
- [x] Commit and push its manifest, metrics, provenance, and synchronized docs (`fdf6506`).
- [x] Strengthen reproducibility audit for metric/provenance pairs and local checksums.

## Multi-Task JEPA Follow-up

- [x] Preregister `WM-T0-005` task split, seeds, controls, and decision thresholds.
- [x] Implement deterministic goal/dynamics PointMass tasks and local-tangent controls.
- [x] Implement held-out task counterfactual and three-seed joint decision logic.
- [x] Commit and push the `WM-T0-005` implementation (`7a9e510`).
- [x] Execute it once from the clean implementation commit.
- [x] Record metrics, provenance, null-safe interpretation, and synchronized docs.

## GPT-2 Semantic Composition Follow-up

- [x] Preregister `LLM-GPT2-003` prompts, directions, magnitudes, splits, and thresholds.
- [x] Implement 72 direct outcomes with singles-only predictor training.
- [x] Add prompt-local, corpus, and direct-additive composition controls.
- [x] Add offline grid, split, linearity, and interaction tests.
- [x] Commit and push the `LLM-GPT2-003` implementation (`1e57e30`).
- [x] Execute once from the clean implementation commit within 600 seconds.
- [x] Commit the checksummed manifest, metrics, provenance, and synchronized result docs.
