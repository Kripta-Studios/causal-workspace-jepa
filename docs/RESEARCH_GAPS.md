# Research Gaps

Status: `ACTIVE`.

- Need causal evidence beyond decodability for all world-model representations.
- Need matched generic-corruption controls for every intervention claim.
- Need train/eval split audits preventing trajectory, donor, entity, and intervention leakage.
- Need direct Qwen intervention verification on Hugging Face models, blocked on CPU VPS.
- Conditional donor resampling repaired the random-control manifold failure in `WM-T0-004`, but the
  PCA intervention was too large to be matched. Need a preregistered local-PCA/tangent control.
- Deep-ensemble intervals calibrate in distribution, but OOD rank AUC and hidden uncertainty-head R2
  fail. Need uncertainty trained for identifiable shifts or a model that receives latent dynamics
  context; scalar post-hoc calibration alone is insufficient.
- The deep predictor's shuffled-action MSE ratio was `0.684`, much weaker than registered. A future
  workspace search needs an architecture/objective where actions are indispensable before auditing
  shared action-planning mechanisms.
- `WM-T0-005` directly tests whether task conditioning, held-out goal/dynamics composition, and
  local-tangent controls resolve those two gaps. Until it runs, this is a method, not evidence.
- GPT-2 bilinear/MLP meta-models do not survive the prompt-local Jacobian, and bilinear transfer to an
  unseen layer fails. Need larger, semantic, combined, resampling, or feature-space interventions
  where nonlinear context dependence is plausible.
