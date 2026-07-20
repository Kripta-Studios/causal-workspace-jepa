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
- For `WM-T0-005`, cross goals with dynamics modes and hold out one full composition. Require two of
  three seeds, random and local-tangent specificity, at least 50 percent task-counterfactual
  recovery, and a selective-necessity ratio of at least 1.25. Keep `workspace_found` false even if
  the narrower candidate passes.
- Preserve the `WM-T0-005` null without threshold or seed tuning. A future CPU JEPA study must change
  the architecture materially by jointly training task consumers or a planner; adding more post-hoc
  probes to the same representation is not a meaningful continuation.
- For `LLM-GPT2-003`, derive directions only from disjoint calibration prompts, orthogonalize them,
  train predictors on singles only, and reserve all composed targets for evaluation. Require direct
  large-single addition and prompt-local finite differences so a learned model cannot win against
  weak baselines alone.
