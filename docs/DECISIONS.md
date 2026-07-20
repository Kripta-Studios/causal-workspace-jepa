# Decisions

## 2026-07-20

- Use only the `cpu_vps` resource path for this run.
- Treat all model-weight downloads, external datasets, Transformers, CUDA, Docker, Conda, and large simulation suites as unavailable.
- Implement CPU smoke code with NumPy and standard-library infrastructure.
- Keep real Qwen, GPT-2/GPT-style real targets, and published JEPA checkpoints `BLOCKED_RESOURCE` until `gpu_12gb`.
- Use `SUMMARY.md` as the handoff log after each milestone.
