# Research Gaps

Status: `ACTIVE`.

- Need causal evidence beyond decodability for all world-model representations.
- Need matched generic-corruption controls for every intervention claim.
- Need train/eval split audits preventing trajectory, donor, entity, and intervention leakage.
- Need direct Qwen intervention verification on Hugging Face models, blocked on CPU VPS.
- Need manifold-matched workspace controls; planted positive/disjoint negative detector controls now
  pass, but arbitrary rollout projections are invalid.
- Need calibrated JEPA uncertainty and a deeper learned predictor before rerunning workspace tests.
- GPT-2 bilinear/MLP meta-models do not survive the prompt-local Jacobian, and bilinear transfer to an
  unseen layer fails. Need larger, semantic, combined, resampling, or feature-space interventions
  where nonlinear context dependence is plausible.
