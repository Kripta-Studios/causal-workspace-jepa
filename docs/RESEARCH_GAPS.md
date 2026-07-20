# Research Gaps

Status: `ACTIVE`.

- Need causal evidence beyond decodability for all world-model representations.
- Need matched generic-corruption controls for every intervention claim.
- Need train/eval split audits preventing trajectory, donor, entity, and intervention leakage.
- Need direct Qwen intervention verification on Hugging Face models, blocked on CPU VPS.
- Need manifold-matched workspace controls; planted positive/disjoint negative detector controls now
  pass, but arbitrary rollout projections are invalid.
- Need calibrated JEPA uncertainty and a deeper learned predictor before rerunning workspace tests.
- Need GPT-2 meta-model comparisons to survive a prompt-local Jacobian, unseen magnitudes, and unseen
  layers; `LLM-GPT2-002` is implemented but not yet run.
