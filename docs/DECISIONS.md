# Decisions

## 2026-07-20

- Use only the `cpu_vps` resource path for this run.
- Treat all model-weight downloads, external datasets, Transformers, CUDA, Docker, Conda, and large simulation suites as unavailable.
- Implement CPU smoke code with NumPy and standard-library infrastructure.
- Keep real Qwen, GPT-2/GPT-style real targets, and published JEPA checkpoints `BLOCKED_RESOURCE` until `gpu_12gb`.
- Use `SUMMARY.md` as the handoff log after each milestone.
- Preserve the original `WM-T0-002` artifact but withdraw its specificity interpretation: its
  “patch” result was assigned directly to the donor target.
- Validate workspace discovery against known shared and disjoint controls before applying it to a
  learned representation.
- Treat a compact five-consumer sensitivity subspace as a candidate only. It is not a workspace
  without direct ablation, PCA/random controls, controllability, selective necessity, and
  generalization.
