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
- GPT-2 bilinear/MLP meta-models do not survive the prompt-local Jacobian. Coordinate and
  contrast-direction steering remain additive, while transport across prompts/layers fails. The
  useful next intervention classes are activation replacement/resampling, patching, and feature
  clamps; more additive steering on this grid is not discriminating.
