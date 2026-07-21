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

Status: `COMPLETED_NEGATIVE` for the corrected Qwen nonlinear-advantage claim; hardware,
instrumentation, storage, and direct-data gates remain validated.

- RTX 5070 Ti Laptop GPU detected with 12,227 MiB VRAM and CUDA-enabled PyTorch.
- `gpu_12gb` doctor passes with about 370 GB free.
- Windows provenance-path comparison and fresh-clone checksum audit were repaired and tested.
- Qwen3-0.6B instrumentation, a 432-effect dataset, and three-seed Intervention-JEPA evaluation are
  validated; direct ranking verification rejected the proposed coordinate circuit.
- Qwen3-0.6B adapter and smoke runner are `SMOKE_VALIDATED` from clean commit `0d6a37b`; no circuit
  is inferred from the instrumentation pass.
- The 432-outcome split-controlled HDF5 dataset is `SMOKE_VALIDATED` from clean commit `0aa80ac` and
  remains frozen for the completed meta-model comparison.
- `LLM-IJEPA-001` ran from clean commit `a54f2ed`; corrected v2 withdrew H-LLM-01 because exact and
  quadratic transport beat every learned seed. H-LLM-06 failed direct precision@1 and the candidate
  graph remains rejected.
- `LLM-QWEN-JVP-AUDIT-001` is implemented/preregistered to re-execute all targets in FP32 and compare
  exact autograd JVPs, central convergence, quadratic Taylor, and refitted predictors.
- V1 was rejected on an invalid downstream endpoint gate. Post-diagnostic v2 passed its source-level
  semantic/convergence audit and withdrew H-LLM-01: exact and quadratic transport beat all learned
  seeds on the frozen selected-target grid.
- Official LeWorldModel revision `8edfeb3...` is verified. A faithful small reproduction and
  intervention/planning/circuit audit completed as `WM-LEWM-001`. Reproduction passes 3/3 and
  planner specificity 2/3, but full hidden-patch/circuit replication fails at 1/3; graph rejected.
- The bounded Qwen3-4B selected-site capture passed from clean commit `55087ea`: exact pinned
  revision, 180 rows, and a 574,308-byte checksummed shard.
- Updated `AUDIT-COMPLETE-001` passed all 14 explicit completion criteria from clean synchronized
  commit `98a9e62`, including the corrected exact-JVP result, 68-test suite, Ruff, and
  reproducibility audit.
- `WM-POPULATION-JACOBIAN-001` ran from clean `89b2e14` and is `REJECTED_NUMERICAL_GATE`: gauge
  controls pass, but fixed 12-node integration underresolves stiff recurrent action chords. The
  global population mean fails correlation/semantic controls; a provisional within-context
  action-vertex mean improves over local 3/3 but requires test-goal confirmation with adaptive
  integration. No planner is competent enough for behavior evidence.
- `LLM-ELEMENT-LAYER-GEOMETRY-001` completed from clean `5d8de9a`. It confirms the preregistered late
  donor-control transition and late answer-row specificity, but rejects the strict local/population
  inversion on both held-out splits and therefore rejects H-CROSS-03. The surviving association
  needs a newly preregistered relation/model confirmation; the observed element thresholds are
  frozen and late crystallization/population averaging remain prior art.
- `LLM-STATE-LAYER-GEOMETRY-001` ran from clean `27ebe43` and is `REJECTED_BEHAVIOR_GATE`: clean
  held-out accuracy missed the `0.75` floor, so no hypothesis is decided. Its descriptive layer-26
  advantage sign switch and `0.949` control correlation require a new behavior-competent prompt/task
  with prospective, boundary-relative gates.
- `LLM-STATE-ONESHOT-LAYER-GEOMETRY-001` supplies that prospective test. The prompt format scored
  13/13 only on excluded calibration states, the target prompt grid is untouched, and the clean
  held-out floor is `0.90`. Control and population-advantage first-crossing layers must match on both
  splits while retaining exact-local and quadratic/HVP-style comparators.

## Milestone 3+: Scientific Runs

Status: `SMOKE_VALIDATED` for the required bounded suite; larger scaling and rejected-mechanism
follow-ups remain open research.

Immediate order: build a genuine target-encoder Intervention-JEPA and a donor-answer task with real
top-token changes; then preregister within-context causal geometry
with pooling/permutation nulls. Generic controllability/observability balancing is prior art, not the
novelty claim.

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
