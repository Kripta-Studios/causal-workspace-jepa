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

Status: `ACTIVE` for CPU-safe causal audit; `BLOCKED_RESOURCE` for GPU-dependent work.

- [done] Repair the original action-patch methodology and rerun from committed code.
- [done] Validate shared-subspace discovery with known positive and negative controls.
- [done] Run `WM-T0-003` with five frozen downstream consumers and matched random/PCA controls.
- [done] Strengthen GPT-2 Medium with prompt/intervention holdouts and required baselines.

`WM-T0-003` completed with a null result. Its follow-up must use a valid uncertainty consumer and
manifold-matched rollout controls before any shared-subspace claim is reconsidered.

`WM-T0-004` completed from clean commit `6785fb1` with a null result. Conditional donor controls
stayed near-manifold and matched almost every random basis, repairing the old control failure. The
action gate, OOD AUC, uncertainty heads, shared-sensitivity thresholds, and direct/rollout
specificity all failed. Do not tune this observed configuration.

`WM-T0-005` is preregistered and implemented but not yet executed. It adds goal/dynamics task
composition, a fully held-out combination, local-tangent controls, explicit task counterfactuals,
three seeds, and minimum recovery/selectivity effect sizes.

`LLM-GPT2-002` completed with a negative nonlinear-advantage result. Continue on GPU with semantic,
combined, resampling, and larger-magnitude interventions; preserve prompt-local Jacobians as the
primary strong baseline.

Use `configs/resource/gpu_12gb.yaml` for selected Qwen and published JEPA continuation. Use `gpu_cluster` for Qwen3-30B-A3B and broad Jacobian/activation studies.
