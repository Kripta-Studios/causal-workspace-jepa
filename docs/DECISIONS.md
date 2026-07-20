# Decisions

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
