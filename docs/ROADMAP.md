# Roadmap

## July 2026 research pivot

- Archive `WM-ACTION-PATH-CALIBRATION-002` as numerical/vector calibration only; the downstream
  small-model action-path route is closed by design before protected-test access.
- Integrate the official single-GPU EB-JEPA action-conditioned Two Rooms example, reproduce
  competent three-seed planning, and expose recurrent action/gate/hidden-state sites before a
  frozen necessity/sufficiency audit.
- Implement ordered multi-site Qwen interventions and treatment/restoration replay tests. Then
  preregister a module-only binding mediator benchmark comparing population/local Jacobians, HVP,
  AtP*, probes, magnitude, random controls, and direct patching. Do not call it a JEPA experiment.
- Only after a causally eligible component-outcome dataset exists, train a new trajectory
  Intervention-JEPA and evaluate composed/held-out interventions against every strong derivative
  and learned baseline.

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
- `LLM-QWEN-JVP-AUDIT-001` was implemented/preregistered to re-execute all targets in FP32 and compare
  exact autograd JVPs, central convergence, quadratic Taylor, and refitted predictors.
- V1 was rejected on an invalid downstream endpoint gate. Post-diagnostic v2 passed its source-level
  semantic/convergence audit and withdrew H-LLM-01: exact and quadratic transport beat all learned
  seeds on the frozen selected-target grid.
- Official LeWorldModel revision `8edfeb3...` is verified. A source-informed small reproduction and
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
- Do not directly confirm the vertex-MSE effect: its validation correlations and scalar-shrinkage
  control are not mechanism-specific. First run the hard-locked validation calibration of decoded
  path cancellation and local finite-effect failure. Only converged, null-separated geometry may
  motivate a new preregistered analysis of the five protected test goals.
- Calibration v1 leaves that gate closed: 256-node horizon-four reconstruction fails for seeds
  101/103, recurrence amplification is not replicated, and cancellation/error association clears
  stratified null p95 only narrowly. Refine the identical validation chords at 512/1024 nodes; stop
  this family if convergence or replicated association does not survive.
- Calibration v2 is implemented with identical chords and 512/1024 nodes. Numerical convergence
  cannot directly permit test preregistration because cancellation and normalized local error share
  a denominator. Adversarial review rejected the proposed derived audit before commit: the parent
  does not store scalar path length at both resolutions or raw direct norm, and two chords per
  action pair cannot support a joint conditional null. Keep V2 as numerical/vector calibration and
  close this family without touching test goals.
- A source-of-truth scientific manuscript now lives at `papers/causal_workspace_jepa.tex`. It must
  compile with the repository bibliography and remain synchronized with final v2 disposition,
  `docs/RESULTS.md`, `SUMMARY.md`, and the experiment/hypothesis registries.
- `LLM-ELEMENT-LAYER-GEOMETRY-001` completed from clean `5d8de9a`. It confirms the preregistered late
  donor-control transition and late answer-row specificity, but rejects the strict local/population
  inversion on both held-out splits and therefore rejects H-CROSS-03. The surviving association
  needs a newly preregistered relation/model confirmation; the observed element thresholds are
  frozen and late crystallization/population averaging remain prior art.
- `LLM-STATE-LAYER-GEOMETRY-001` ran from clean `27ebe43` and is `REJECTED_BEHAVIOR_GATE`: clean
  held-out accuracy missed the `0.75` floor, so no hypothesis is decided. Its descriptive layer-26
  advantage sign switch and `0.949` control correlation require a new behavior-competent prompt/task
  with prospective, boundary-relative gates.
- `LLM-STATE-ONESHOT-LAYER-GEOMETRY-001` ran from clean `c1daa46`. Competence/numerics and late
  semantic specificity pass, but exact boundary equality fails on validation (control 24,
  population 26), and the early-control ceiling fails on test. H-CROSS-05 is false. Any weaker lag
  hypothesis requires a new independent relation, not threshold adjustment on the state data.
- `LLM-COUNTRY-CODE-LAYER-GEOMETRY-001` is the independent prospective test. Its 36 ISO answers are
  disjoint from state answers and its 24/6/6 targets were never forwarded during selection. The
  selected prompt scored only `5/7` on excluded calibration countries, so the frozen `0.90`
  validation/test competence gate may reject it. If eligible, it tests only whether population
  advantage begins zero or one registered layer after direct control; the rule is post-state and
  must not be represented as independent discovery on all three relations.
- The clean country retry from `48226c6` is mixed. Competence, numerics, H-LLM-14, and H-GEO-15
  pass, but H-GEO-14/H-CROSS-06 fail because validation population advantage begins at layer 21
  before the 50% control boundary at 24. Freeze the directional lag null; do not replace it with an
  unsigned or partial-control threshold on these observed data. Resume the protected JEPA test-goal
  geometry audit with adaptive integration as the next independent family.

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
