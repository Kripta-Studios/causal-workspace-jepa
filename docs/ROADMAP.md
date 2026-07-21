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

## GPU Transition Audit (2026-07-21)

Status: `SMOKE_VALIDATED` for hardware/control-plane detection and bounded Qwen3-0.6B experiments.

- RTX 5070 Ti Laptop GPU detected with 12,227 MiB VRAM and CUDA-enabled PyTorch.
- `gpu_12gb` doctor passes with about 370 GB free.
- Windows provenance-path comparison and fresh-clone checksum audit were repaired and tested.
- Qwen3-0.6B instrumentation, a 432-effect dataset, and three-seed Intervention-JEPA evaluation are
  validated. Implement one faithful published action-conditioned JEPA reproduction next.
- Qwen3-0.6B adapter and smoke runner are `SMOKE_VALIDATED` from clean commit `0d6a37b`. Next build
  the bounded intervention dataset; do not infer a circuit from the instrumentation pass.
- The 432-outcome split-controlled HDF5 dataset is `SMOKE_VALIDATED` from clean commit `0aa80ac`.
  Next preregister and train the meta-model/baselines without changing this observed dataset.
- `LLM-IJEPA-001` is `SMOKE_VALIDATED` from clean commit `a54f2ed`. H-LLM-01/02/03 passed the fixed
  gates, while H-LLM-06 failed direct precision@1 and the candidate graph is rejected.
- Official LeWorldModel revision `8edfeb3...` is verified. A faithful small reproduction and
  intervention/planning/circuit audit are preregistered as `WM-LEWM-001`; execute only after the
  implementation milestone is committed.

## Milestone 3+: Scientific Runs

Status: `ACTIVE`; the local GPU resource blocker is removed, but the GPU-dependent code and evidence
remain incomplete.

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

`WM-T0-005` completed from clean commit `7a9e510` with a three-seed null. It added goal/dynamics
task composition, a fully held-out combination, local-tangent controls, explicit task
counterfactuals, and minimum effect sizes. No seed passed action, consumer, sensitivity, and
counterfactual gates jointly. Stop tuning this observed configuration.

`LLM-GPT2-002` completed with a negative nonlinear-advantage result. Continue on GPU with semantic,
combined, resampling, and larger-magnitude interventions; preserve prompt-local Jacobians as the
primary strong baseline.

`LLM-GPT2-003` completed from clean commit `1e57e30` inside the CPU guard. Large contrast-direction
compositions remained additive, prompt-local Jacobians were accurate, and learned transport failed
on held-out prompts despite strong seen-prompt scores. Stop tuning this grid. Future work should use
activation replacement/resampling or feature clamps that can create genuine nonlinear effects.

Use `configs/resource/gpu_12gb.yaml` for selected Qwen and published JEPA continuation. Use `gpu_cluster` for Qwen3-30B-A3B and broad Jacobian/activation studies.
