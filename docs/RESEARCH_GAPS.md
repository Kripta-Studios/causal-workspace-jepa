# Research Gaps

Status: `ACTIVE`.

- Need causal evidence beyond decodability for all world-model representations.
- Need matched generic-corruption controls for every intervention claim.
- Need train/eval split audits preventing trajectory, donor, entity, and intervention leakage.
- Bounded direct Qwen intervention verification is implemented. Broader prompt families, semantic
  features, layer/site holdouts, behavioral endpoints, and a candidate that survives direct ranking
  controls remain open; the first meta-ranked coordinate candidate was rejected.
- Conditional donor resampling repaired the random-control manifold failure in `WM-T0-004`, but the
  PCA intervention was too large to be matched. Need a preregistered local-PCA/tangent control.
- Deep-ensemble intervals calibrate in distribution, but OOD rank AUC and hidden uncertainty-head R2
  fail. Need uncertainty trained for identifiable shifts or a model that receives latent dynamics
  context; scalar post-hoc calibration alone is insufficient.
- The deep predictor's shuffled-action MSE ratio was `0.684`, much weaker than registered. A future
  workspace search needs an architecture/objective where actions are indispensable before auditing
  shared action-planning mechanisms.
- `WM-T0-005` showed that simply appending goal/mass context does not resolve those gaps: actions
  remained weak and post-hoc value/risk/uncertainty/action heads failed the held-out composition.
  The next architecture must train task-relevant consumers jointly or couple them to planning.
- `WM-LEWM-001` validates the published small-reproduction path but exposes two new gaps. Latent
  planning succeeds on only `1/12` cases per seed, and hidden donor-patch decoding/circuit evidence
  replicates on only one seed. A future route needs a stronger planner objective and a subspace whose
  decoded counterfactual is stable across initialization; the observed thresholds must not be tuned.
- Ensemble variance falls under the action-subspace intervention even as control worsens. Future
  uncertainty work needs intervention-aware calibration or epistemic/OOD objectives, not raw member
  disagreement alone.
- GPT-2 bilinear/MLP meta-models do not survive the prompt-local Jacobian. Coordinate and
  contrast-direction steering remain additive, while transport across prompts/layers fails. The
  useful next intervention classes are activation replacement/resampling, patching, and feature
  clamps; more additive steering on this grid is not discriminating.
- Qwen population-Jacobian regularization is confirmed only within one capital-prompt family.
  `WM-POPULATION-JACOBIAN-001` attempted the cross-domain port with a train-goal decoded physical
  endpoint, but its quadrature gate failed and its global population mean failed correlation and
  action-label specificity. There is therefore no positive cross-domain replication.
- V1 is numerically rejected because fixed quadrature underresolves sharply varying recurrent
  derivatives. Do not interpret its provisional vertex-mean result as evidence. A continuation must
  validate derivatives locally, adaptively integrate to convergence, retain action-label controls,
  and use only the protected test goals. The global population mean's low MSE is consistent with
  nonspecific shrinkage because correlation and semantic permutation gates failed.
- The provisional vertex-mean MSE reduction is no longer the protected-test target: validation
  correlation and a train-calibrated scalar-shrinkage diagnostic undermine its semantic reading.
  `WM-ACTION-PATH-CALIBRATION-001` instead measures decoded derivative path length, cancellation,
  local finite-effect failure, two-level quadrature convergence, and within-action-pair nulls on
  exposed validation goals. Its first resolution converged only at horizon one and seed-107
  horizon four. V2 repeats the exact chords at 512/1024 nodes with streamed Jacobians. A later
  audit must first remove the shared denominator between cancellation and normalized local error
  using unnormalized/partial endpoints and effect-size-conditioned nulls; protected test remains
  inaccessible regardless of v2 convergence until that validation audit passes.
- The element-symbol study supplies an independent relation and four-layer profile. It confirms a
  sharp late donor-control transition and late population semantic specificity, but the registered
  exact-local/population inversion ratios fail on both splits. Because late factual crystallization
  and population averaging are already published, the surviving conjunction is at most a bounded
  candidate observation. A new relation/model must prospectively confirm it; the observed element
  splits cannot be mined for revised thresholds.
- The state-abbreviation study is behavior-ineligible, so it does not provide cross-relation
  replication. A new prompt/task must establish clean competence prospectively and test a boundary-
  relative transition rather than move v1's fixed layer after the fact. Even a full future pass
  leaves cross-scale generalization, localization, circuit reconstruction, and mechanism unresolved.
- One-shot v2 repairs prompt competence prospectively but reuses the v1 state identities and split;
  its exact-boundary hypothesis is now false on validation. The remaining descriptive pattern is a
  zero-or-one-grid-step lag with high rank correlation, but that weaker statement is post-result and
  needs a new relation with new entities/answer semantics. Why population usefulness lags or tracks
  donor control remains mechanistically unexplained.
- The independent country-code relation falsifies the proposed directional lag: population
  advantage precedes the 50% donor-control boundary on validation and coincides with it on test.
  Continuous control measures, additional architectures, or component-level mediation are open,
  but the existing element/state/country thresholds cannot be weakened or re-mined.
- The working paper makes the current negative/positive boundary explicit, but publication-quality
  generalization remains open: more prompt families, larger Qwen targets, stronger world-model
  planners, independently used consumers, head/MLP localization, and direct circuit
  necessity/sufficiency/faithfulness are still required.
