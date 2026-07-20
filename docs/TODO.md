# TODO

## Milestone 0

- [x] Read local instructions.
- [x] Add resource profiles.
- [x] Add doctor CLI.
- [x] Add typed interfaces.
- [x] Run tests.
- [x] Run `git diff --check`.
- [x] Commit and push.

## Milestone 1

- [x] Implement Tier 0 generators.
- [x] Add Tier 0 smoke config and manifest update.
- [x] Implement tiny NumPy JEPA.
- [x] Add planner and smoke experiment.
- [x] Add tests.
- [x] Run tests.
- [x] Run `git diff --check`.
- [x] Commit and push code.
- [x] Rerun smoke from committed code.
- [x] Commit and push metrics/docs.

## Milestone 2

- [x] Implement intervention operators.
- [x] Implement probes and finite differences.
- [x] Implement circuit graph schema.
- [x] Implement mock Qwen adapter and Intervention-JEPA smoke.
- [x] Add tests.
- [x] Run tests.
- [x] Run `git diff --check`.
- [x] Commit and push code.
- [x] Rerun mock smoke from committed code.
- [x] Commit and push metrics/docs.

## Milestone 3

- [x] Register and run Tier 0 mechanistic study.
- [x] Test action displacement decodability.
- [x] Test action-coordinate patch specificity against controls.
- [x] Evaluate workspace criteria and record null result.
- [x] Run GPT-2 Medium hidden-state intervention smoke under user override.
- [x] Commit and push Tier 0 mechanistic metrics.
- [x] Commit and push GPT-2 metrics/docs.
- [x] Audit the original action-patch implementation and withdraw the unsupported specificity claim.
- [x] Implement a replayable action-input patch with norm-matched controls.
- [x] Implement shared-subspace discovery with positive and negative controls.
- [x] Preregister `WM-T0-003` thresholds and splits.
- [x] Commit and push repaired code before execution.
- [x] Rerun corrected `WM-T0-002` from committed code.
- [x] Run `WM-T0-003` from committed code.
- [x] Commit and push `WM-T0-003` null metrics.

## Milestone 3 Follow-up

- [x] Run `WM-T0-003` from committed code and preserve its null result.
- [x] Implement a split-calibrated deep ensemble uncertainty pipeline for `WM-T0-004`.
- [x] Implement conditional donor resampling with activation-density and perturbation matching.
- [x] Implement a deeper learned predictor with genuine internal sites and independently trained consumers.
- [x] Preregister `WM-T0-004` splits, thresholds, OOD shift, and rejection rules before execution.
- [x] Commit and push the `WM-T0-004` implementation (`6785fb1`).
- [x] Execute `WM-T0-004` from the clean committed code without changing thresholds.
- [x] Record the null result and rerun the full audit.
- [x] Commit and push the `WM-T0-004` result milestone.

## GPT-2 Medium Follow-up

- [x] Audit `LLM-GPT2-001` leakage, split, site, and baseline limitations.
- [x] Preregister `LLM-GPT2-002` before execution.
- [x] Implement batched direct intervention data and storage/checksum guard.
- [x] Implement linear, bilinear, trained MLP, nearest-neighbor, sparse-context, local Jacobian, and
  corpus-averaged Jacobian baselines.
- [x] Add offline split, predictor, and resource-limit tests.
- [x] Commit and push `LLM-GPT2-002` code before execution.
- [x] Run `LLM-GPT2-002` from clean committed code.
- [x] Commit and push its manifest, metrics, provenance, and synchronized docs (`fdf6506`).
- [x] Strengthen reproducibility audit for metric/provenance pairs and local checksums.

## Multi-Task JEPA Follow-up

- [x] Preregister `WM-T0-005` task split, seeds, controls, and decision thresholds.
- [x] Implement deterministic goal/dynamics PointMass tasks and local-tangent controls.
- [x] Implement held-out task counterfactual and three-seed joint decision logic.
- [x] Commit and push the `WM-T0-005` implementation (`7a9e510`).
- [x] Execute it once from the clean implementation commit.
- [x] Record metrics, provenance, null-safe interpretation, and synchronized docs.

## GPT-2 Semantic Composition Follow-up

- [x] Preregister `LLM-GPT2-003` prompts, directions, magnitudes, splits, and thresholds.
- [x] Implement 72 direct outcomes with singles-only predictor training.
- [x] Add prompt-local, corpus, and direct-additive composition controls.
- [x] Add offline grid, split, linearity, and interaction tests.
- [ ] Commit and push the `LLM-GPT2-003` implementation.
- [ ] Execute once from the clean implementation commit within 600 seconds.
- [ ] Commit the checksummed manifest, metrics, provenance, and synchronized result docs.
