# Research Gaps

Status: `ACTIVE`.

- Need causal evidence beyond decodability for all world-model representations.
- Need matched generic-corruption controls for every intervention claim.
- Need train/eval split audits preventing trajectory, donor, entity, and intervention leakage.
- Need direct Qwen intervention verification on Hugging Face models, blocked on CPU VPS.
- `WM-T0-004` implements conditional donor and density/magnitude matching, but execution must show
  whether enough controls remain valid and whether hybrid activations actually stay near-manifold.
- `WM-T0-004` implements calibrated ensemble uncertainty and a deeper learned predictor; held-out
  coverage, OOD ranking, and independent uncertainty-head validity remain unmeasured until execution.
- GPT-2 bilinear/MLP meta-models do not survive the prompt-local Jacobian, and bilinear transfer to an
  unseen layer fails. Need larger, semantic, combined, resampling, or feature-space interventions
  where nonlinear context dependence is plausible.
