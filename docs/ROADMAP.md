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

Status: `SMOKE_VALIDATED`.

- Activation naming and cache: implemented.
- Intervention engine: implemented.
- Linear probes: implemented.
- Sparse dictionary placeholder suitable for CPU smoke: implemented.
- Jacobian finite-difference checks: implemented.
- Circuit graph schema and matched controls: implemented.
- Mock transformer and intervention-JEPA smoke runner: validated in `artifacts/metrics/mock_qwen_intervention_jepa_smoke.json`.

## Milestone 3+: Scientific Runs

Status: `BLOCKED_RESOURCE` for GPU-dependent work.

Use `configs/resource/gpu_12gb.yaml` for selected Qwen and published JEPA continuation. Use `gpu_cluster` for Qwen3-30B-A3B and broad Jacobian/activation studies.
