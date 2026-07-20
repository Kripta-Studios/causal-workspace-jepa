# Roadmap

## Milestone 0: Audit And Safe Bootstrap

Status: `SMOKE_VALIDATED` once tests, diff check, commit, and push complete.

- Create repository architecture.
- Add resource profiles and CPU guard.
- Add doctor command.
- Add typed interfaces.
- Add CI-safe tests.
- Synchronize Markdown.

## Milestone 1: Tier 0 Environments And Tiny JEPA

Status: `SMOKE_VALIDATED`.

- Deterministic PointMass2D, BouncingBall2D, TwoBodyCollision, TinyMaze, and MiniPush smoke generator.
- Tiny NumPy action-conditioned JEPA.
- No-action and shuffled-action controls.
- Simple planner and smoke metrics.
- Clean committed smoke metrics are recorded in `artifacts/metrics/tiny_jepa_smoke.json`.

## Milestone 2: Interpretability Foundation

Status: `NOT_STARTED`.

- Activation naming and cache.
- Intervention engine.
- Linear/nonlinear probes.
- Sparse dictionary placeholder suitable for CPU smoke.
- Jacobian finite-difference checks.
- Circuit graph schema and matched controls.

## Milestone 3+: Scientific Runs

Status: `BLOCKED_RESOURCE` for GPU-dependent work.

Use `configs/resource/gpu_12gb.yaml` for selected Qwen and published JEPA continuation. Use `gpu_cluster` for Qwen3-30B-A3B and broad Jacobian/activation studies.
