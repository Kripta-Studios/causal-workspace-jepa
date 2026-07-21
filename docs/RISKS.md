# Risks

Status: `ACTIVE`.

- `MITIGATED`: the bounded required suite passes `AUDIT-COMPLETE-001`, but a completion audit is not
  scientific validation. Both reconstructed circuit candidates remain rejected and the workspace
  hypothesis remains null in the tested systems.
- `RESOLVED_NEGATIVE`: the `LLM-IJEPA-001` local baseline was a BF16 one-sided secant. Corrected v2
  passed FP32 exact-JVP/central/source-semantic gates and withdrew H-LLM-01; zero of three learned
  seeds beat exact or quadratic transport.
- `RESOLVED_REPORTING_REQUIRED`: JVP audit v1 mechanically emitted `WITHDRAWN` even though its own
  preregistration says numerical failure leaves H-LLM-01 unresolved. Documentation follows the
  preregistered rule; the prospective runner must emit `UNRESOLVED_NUMERICAL_REJECT` on that path.
- `MITIGATED_POSTHOC`: JVP audit v2 was designed after v1 exposed the endpoint problem and is not a
  blind test. Its semantic repair is source-level and machine-epsilon-derived, while every observed
  scientific comparison threshold is frozen; reporting must retain this post-diagnostic boundary.
- `MITIGATED_PREREGISTRATION`: layer 21 was selected using Japan/Canada/China/Kenya engineering
  patches. Those entities are excluded from `LLM-CAPITAL-PATCH-001`; the 36 final facts and splits
  were fixed using tokenizer structure only. The frozen final run passed all gates, but shared
  prompt wording remains a deliberate task-family limitation and must not be called cross-task
  generalization.
- `ACTIVE_METRIC_CONFLICT`: on capital patches, exact JVP had better full-vector normalized MSE than
  quadratic Taylor (`0.530` versus `0.639`) while being much worse on answer-candidate agreement
  (`0.351` versus `0.743`). Learned-model reporting must retain both activation fidelity and direct
  behavior fidelity; a favorable aggregate MSE cannot substitute for the behavior endpoint.
- `MITIGATED_NAMING`: `NeuralInterventionJEPA` is a legacy supervised conditional bottleneck without
  a target encoder, stop-gradient/EMA target, or anti-collapse objective. Documentation qualifies
  it. The separately named `TargetEncoderInterventionJEPA` implements those mechanisms and completed
  a real-data three-seed comparison, but failed every registered hypothesis; neither class name is
  treated as evidence of causal compression.
- `ACTIVE_DECODER_SUPERVISION`: the genuine JEPA predicts a target embedding without direct-effect
  labels, but its post-hoc linear decoder is supervised on train-entity effects. The registered run
  failed; even its oracle target-embedding decode did not transfer, so decoder supervision did not
  create a false positive.
- `RESOLVED_NEGATIVE_GEOMETRY`: every `LLM-TARGET-IJEPA-001` seed missed the target effective-rank
  floor and its oracle decode failed on unseen entities. Do not tune this observed architecture.
  A continuation must test an explicit context-conditioned/gauge-aware geometry with null controls.
- `ACTIVE_BASELINE_INSTABILITY`: PCA-bilinear ridge extrapolated to `2.04e11` normalized MSE across
  entity splits. It is retained transparently, but cannot serve as a meaningful strong baseline;
  regularized or bounded bilinear methods require a new prospective protocol.
- `ACTIVE_NOVELTY`: controllability/observability balancing and CoBRAS are established prior art.
  Pooling reachability and observability across different contexts can create false shared modes
  even when no context has a jointly active direction. Future geometry must include within-context,
  permutation/pooling, coordinate-gauge, finite-amplitude, and direct-behavior controls.
- `ACTIVE_IDENTIFICATION`: `J D^T` identifies the local response of chosen answer logits along
  sampled donor chords, conditional on a recipient prompt. It does not identify a complete causal
  state, a unique activation subspace, or the path used by a finite patch. The experiment emits
  these assumptions explicitly and includes dormant-route/pooling warnings.
- `CONFIRMED_GAUGE_RISK`: the real diagonal reparameterization changed naive pooled subspace overlap
  from `0.04036` to `0.0003345` without changing the paired functional contraction. Euclidean
  activation-subspace centrality/overlap must not be interpreted without a coordinate-gauge contract.
- `ACTIVE_POSTHOC`: the train-mean Jacobian's finite-effect advantage was discovered on the primary
  test analysis, not preregistered. It cannot be promoted to a claim until confirmed on a separately
  frozen analysis or new task/model. Context derangements also improve discrete candidate agreement
  despite worse continuous MSE, so neither endpoint can be used alone.
- `MITIGATED_CONFIRMATION`: the population-Jacobian thresholds are intentionally chosen after the
  test-split discovery. The new runner uses only the previously unanalyzed validation split and
  records the test result as reference; even a pass remains within-family confirmation rather than
  an independent-model replication.
- `CONFIRMED_BOUNDED`: all validation confirmation gates passed, including averaging dose response
  and answer-row specificity. The effect now holds on test and validation entity groups, but both
  share one prompt template, model, layer, and intervention class. Do not call the method novel:
  corpus-averaged Jacobians are central to Anthropic's Jacobian Lens. The new evidence is limited to
  finite causal-fidelity behavior in this suite.
- `REJECTED_CONFIRMATION`: the state roster failed the clean-answer floor. Exact local and
  quadratic/HVP-style comparators were present, but no scientific score is eligible. The apparent
  layer-26 alignment was observed after rejection and cannot be promoted without a newly registered,
  behavior-competent task. One-model confirmation would still not establish SOTA.
- `COMPLETED_MIXED_V2`: one-shot competence and numerics pass, but exact boundary equality fails on
  validation and early donor control fails on test. The observed zero-or-one-step lag is posthoc;
  reporting it as confirmed would be threshold laundering. A new relation is mandatory.
- `ACTIVE_POSTHOC_LAG`: the country study is prospective, but its zero-or-one-step lag rule and
  Spearman `0.70` threshold were designed from frozen element/state results. Those two relations are
  not independent confirmations under the new rule. The country prompt also scored only `5/7` on
  excluded calibration facts; failure of the `0.90` target gate must be retained without rescue.
- `RESOLVED_NEGATIVE_LAG`: country competence passed, but H-GEO-14/H-CROSS-06 failed because the
  validation population boundary preceded the registered 50% control boundary. The surviving
  monotone-control and row-null-specificity results do not establish temporal/causal ordering,
  component localization, or a new population-Jacobian algorithm.
- `ACTIVE_PATH_CALIBRATION`: large derivative path length divided by small net decoded displacement
  can reflect real vector cancellation, decoder error, or numerical underresolution. The new
  validation calibration reports direct reconstruction, refinement change, direct-effect norm,
  speed concentration, horizon controls, and within-action-pair permutation nulls. It makes no
  claim and leaves test goals inaccessible until a separate preregistration.
- `ACTIVE_MANUSCRIPT_DRIFT`: a paper can silently overstate evolving repository evidence. The
  source-of-truth `papers/causal_workspace_jepa.tex` therefore names experiment IDs, run commits,
  evidence levels, negative dispositions, and artifact paths; it is compiled with local BibTeX and
  audited against `docs/RESULTS.md` before every paper milestone.
- `CONFIRMED_PATH_UNDERRESOLUTION`: 256-node composite integration is adequate at horizon one and
  for seed-107 horizon four, but fails refinement/direct reconstruction on seeds 101/103 horizon
  four. The small positive stratified-correlation margins may change under better integration;
  neither cancellation nor recurrent amplification is evidence yet.
- `MITIGATED_LONG_RUN_ATOMICITY`: the v2 streaming run exposed multi-hour, tens-of-thousands-kernel
  execution with no partial artifact. Future action-path runs checkpoint each completed horizon
  atomically and resume only when experiment ID, exact config bytes, and source commit match. The
  active v2 process predates this feature, and safe batch-size benchmarking remains open.

- `RESOLVED_RESOURCE`: the current host has an RTX 5070 Ti Laptop GPU with 12,227 MiB VRAM and about
  370 GB free; the historical CPU-VPS blocker no longer applies.
- `ACTIVE`: GPT-2 Medium was downloaded and run after explicit user override; weights are under `.cache/` and ignored by Git.
- `ACTIVE`: bounded Qwen3-0.6B/4B downloads are permitted by `gpu_12gb`, but storage and VRAM must be
  estimated first and all-layer/all-token capture remains prohibited.
- `RESOLVED`: pinned Qwen3-4B is 8.06 GB and 4.0B parameters. Its bounded bfloat16 selected-site
  capture fit the 12,227 MiB GPU and passed all storage/checksum gates. Broader gradients,
  interventions, and all-layer capture remain outside this result.
- `MITIGATED`: the host has an invalid implicit Hugging Face OAuth token for public API calls. Qwen
  loaders explicitly use `token=False`; no token value was printed, changed, stored, or committed.
- `RESOLVED`: `LLM-QWEN-001` validates real pinned Qwen3-0.6B hooks/interventions/autograd from clean
  code. The result remains instrumentation smoke and cannot substitute for held-out meta-model or
  circuit evidence.
- `RESOLVED`: the first Qwen run recorded `git_dirty=true` because provenance was collected after
  creating its output. Those two regenerable files were discarded; ordering was fixed in `0d6a37b`,
  and only the clean rerun is reported.
- `ACTIVE`: the Qwen dataset generator accesses adapter donor/statistic registries to compute the
  exact source edit without using downstream targets. This is tested but should become a public
  adapter method before broader reuse.
- `ACTIVE`: 432 outcomes cover two small prompt families and selected residual coordinates/sites;
  even a clean run is smoke-scale and cannot establish broad Qwen mechanism generalization.
- `ACTIVE`: local-linear MSE `139.83` is aggregated across projected hidden/logit targets and very
  different edit classes. It indicates a hard/off-local regime but is not itself a nonlinear-model win.
- `ACTIVE`: bilinear regression has more features than training examples; its implementation uses a
  dual ridge solve. Comparisons are predictive held-out scores, not parameter-identification claims.
- `ACTIVE`: direct verification ranks only four residual coordinates at one layer on four new
  prompts. It failed precision@1; the meta-ranked effect was also slightly below the random control,
  so the graph is rejected rather than promoted to a Qwen circuit.
- `ACTIVE`: the Intervention-JEPA passes its registered operation-holdout gate, but nearest-neighbor
  has slightly lower resampling MSE (`2.0946` versus `2.1405`). Claims must be limited to the fixed
  comparator set and must not imply universal nonlinear-model dominance.
- `ACTIVE`: the fitted 16-component dictionary has density `0.974`; it is a predictive baseline,
  not evidence for sparse or monosemantic Qwen features.
- `BLOCKED_EXTERNAL`: SkyJEPA remains blocked until official implementation assets are verified.
- `MITIGATED`: the faithful LeWorldModel reproduction is dimension/data scaled and is not a released
  checkpoint or benchmark replication. Metrics and docs must always retain this distinction.
- `RESOLVED_METHODOLOGY`: reduced engineering checks showed that selecting LeWM checkpoints by
  prediction MSE alone selects collapsed embeddings. The registered runner uses the fixed final
  optimization step and explicitly gates latent variance and state decodability.
- `RESOLVED_NEGATIVE`: `WM-LEWM-001` passes faithful reproduction on all seeds but only one of three
  full hidden-patch/circuit gates. The graph is rejected; do not describe the seed-107 circuit as a
  general mechanism.
- `ACTIVE`: the latent planner has only `1/12` clean closed-loop successes per seed. Its intervention
  changes are real model-mediated trajectory/cost/action effects, but not evidence of strong control.
- `ACTIVE`: intervention reduces ensemble variance from `0.506` to `0.250` despite weaker outcomes,
  exposing a possible overconfidence failure rather than a useful uncertainty response.
- `MITIGATED`: the current Windows system Python is 3.14 with CUDA PyTorch 2.10 and Transformers
  5.3, while the project supports Python 3.11+. The resolved `uv.lock` is committed; environment
  provenance still records the actual runtime for every experiment.
- `RESOLVED`: all 24 required literature entries were checked against primary paper, project,
  code, or model-card sources and now include the required registry fields.
- `RESOLVED`: `uv` is installed and the previously empty `uv.lock` has been regenerated from the
  declared dependency groups. Actual experiment provenance must still record the running versions.
- `RESOLVED`: the reproducibility audit compared POSIX provenance strings to Windows-rendered paths;
  normalized relative paths now pass on both platforms.
- `ACTIVE`: ignored GPT-2 data shards from the Linux host are absent on this machine. Their manifest
  records are audited as skipped, not verified; this is not a checksum pass for the missing bytes.
- `RESOLVED`: an earlier handoff reported `/root/.cache/pip` at about 4.4 GB. The 2026-07-20
  resource re-audit measured all of `/root/.cache` at about 233 MB; no cache deletion was needed.
- `ACTIVE`: GPT-2 Medium CPU runs are slow; the committed smoke uses 4 prompts, 2 residual coordinates, and 16 direct interventions.
- `RESOLVED`: adversarial review found that the original `WM-T0-002` action patch assigned the donor
  target directly. The corrected intervention was executed from clean commit `315d8cf`; the original
  metric is superseded.
- `ACTIVE`: a five-consumer subspace in the tiny model may simply recover its four-dimensional
  physical-state manifold. High-variance PCA is therefore a mandatory matched control.
- `ACTIVE`: `WM-T0-003` confirmed this concern: PCA ablation was more damaging than the proposed
  subspace. The current OOD-uncertainty readout also failed held-out prediction.
- `REJECTED`: arbitrary random latent projection as a multistep control on the current linear model.
  Some controls cause extreme off-manifold rollout divergence. Future controls must match activation
  density or use in-manifold donor/resampling interventions.
- `RESOLVED`: conditional donor candidate/random patches in `WM-T0-004` stayed near the empirical
  activation bank and yielded 63/64 and 64/64 matched controls. PCA patches were correctly rejected
  as unmatched because their perturbations were much larger.
- `ACTIVE`: `WM-T0-004` is one seed on simple PointMass physics. Its null blocks claims for this run;
  it does not prove that stronger goal-conditioned or multi-task JEPAs lack workspace-like structure.
- `RESOLVED_NULL`: `WM-T0-005` tested richer synthetic task context on three seeds. Its post-hoc
  consumers did not generalize to the held-out composition and task-counterfactual recovery was
  negative, so no shared state/task manifold is promoted to a workspace candidate.
- `ACTIVE`: `LLM-GPT2-002` is larger than the original smoke but still has one seed, eight local
  prompts, coordinate interventions, and selected outputs. Its local Jacobian uses extra direct
  small-magnitude executions and must be compared on fidelity, not runtime.
- `ACTIVE`: `LLM-GPT2-002` took `647.85` seconds, exceeding the CPU profile's 10-minute expectation
  by `47.85` seconds. The result is retained, but this configuration must not be described as
  within the runtime guard.
- `RESOLVED_NULL`: `LLM-GPT2-003` finished in `392.85` seconds. Lexical, syntax, and final-token
  confounds remain, so its constructed direction labels do not support semantic feature claims.
  Large compositions were additive and all learned held-out-prompt predictions failed.
