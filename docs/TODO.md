# TODO

## GPU Transition (2026-07-21)

- [x] Adversarially audit the Qwen Jacobian baseline, model naming, behavior changes, and novelty.
- [x] Mark H-LLM-01 `UNDER_REAUDIT`; preserve the old result without continuing to claim it.
- [x] Implement, unit-test, and preregister `LLM-QWEN-JVP-AUDIT-001` with exact FP32 JVP and central
  convergence controls.
- [x] Commit/push the audit preregistration and execute v1 unchanged from clean commit `686368e`.
- [x] Retain v1 as `REJECTED` on its numerical endpoint gate; do not relax that threshold.
- [x] Preregister v2 source-semantic gates after disclosing the v1 result and cancellation diagnosis.
- [x] Commit/push v2 and execute it unchanged from clean commit `a779ff6`; all numerical gates pass.
- [x] Withdraw restricted H-LLM-01 after 0/3 seeds beat exact JVP/quadratic Taylor.
- [x] Revise and rerun the executable completion audit from clean `98a9e62`; all 14 criteria pass.
- [x] Implement and preregister a genuine target-encoder/stop-gradient Intervention-JEPA with
  anti-collapse tests and strong behavior/local baselines.
- [x] Commit/push and execute `LLM-TARGET-IJEPA-001` unchanged from `3086cd4`; retain the 0/3
  negative, rank-diversity failure, oracle-decoder failure, and divergent baseline rankings.
- [x] Implement/preregister a split-safe semantic donor-answer dataset with aggregate behavior gates.
- [x] Commit/push and execute `LLM-CAPITAL-PATCH-001` unchanged from `95018cb`; all gates pass.
- [x] Implement/preregister context-paired causal geometry with analytic pooling, 256 derangements,
  dual-coordinate gauge checks, manifold donor chords, and finite behavior endpoints.
- [x] Commit/push and execute `LLM-CONTEXT-GEOMETRY-001` unchanged from `49d68b7`; retain failed
  pooling/context-specificity gates and the positive gauge diagnostic.
- [x] Preregister validation-only confirmation of the post-result train-mean-Jacobian advantage with
  averaging-size, per-context, bootstrap, answer-row, and discrete/continuous endpoint controls.
- [x] Commit/push and execute `LLM-POPULATION-JACOBIAN-001` unchanged from `3725714`; all three
  validation confirmation gates pass.
- [x] Implement and preregister `WM-POPULATION-JACOBIAN-001` on three frozen LeWorldModel seeds,
  valid one-hot action swaps, held-out goals, path integration, decoded physics, and gauge controls.
- [x] Commit/push and execute `WM-POPULATION-JACOBIAN-001` unchanged from `89b2e14`; retain its
  numerical rejection, underresolved path integral, ineligible planners, and untouched test goals.
- [ ] Preregister a corrected adaptive path-integral audit plus action-vertex-mean confirmation on
  the five protected test goals; v1's scientific decisions must remain rejected.
- [ ] Replicate population-versus-local finite transport on another Qwen prompt family/model before
  any general or SOTA claim.

- [x] Re-read `AGENTS.md`, `VPS_RUNBOOK.md`, `SUMMARY.md`, repository docs, and current Git state.
- [x] Verify RTX 5070 Ti/CUDA, RAM/CPU, disk, Python, Transformers, and the `gpu_12gb` doctor.
- [x] Run the full pre-change test and reproducibility baseline.
- [x] Repair Windows provenance-path comparison and fresh-clone checksum expectations.
- [x] Implement and offline-test the Hugging Face Qwen adapter, selected hooks, interventions, and autograd.
- [x] Implement and test the bounded resumable sharded HDF5 activation store.
- [x] Execute the preregistered Qwen3-0.6B instrumentation smoke from clean commit `0d6a37b`.
- [x] Preregister the 432-outcome Qwen dataset grid, splits, donors, local probes, and storage guard.
- [x] Commit the dataset generator, then execute it once from clean code.
- [x] Commit and execute `LLM-INTDATA-001` from clean commit `0aa80ac`; validate its shard checksum.
- [x] Train/evaluate Intervention-JEPA and all required strong baselines on held-out real-Qwen effects.
- [x] Directly execute ranked meta-model circuit candidates; reject the failed precision@1 candidate.
- [x] Implement and preregister three-seed neural Intervention-JEPA, trajectory variant, baselines,
  sparse transport, checkpoint replay, and independent direct-verification logic.
- [x] Commit `LLM-IJEPA-001`, then execute once without changing thresholds.
- [x] Integrate and execute at least one published action-conditioned JEPA or faithful reproduction.
- [x] Implement, source-pin, test, and preregister the faithful small LeWorldModel reproduction.
- [x] Commit `WM-LEWM-001`, then execute its three seeds and restricted circuit audit unchanged;
  retain the failed replicated causal/circuit gates and rejected graph.
- [x] Run `AUDIT-COMPLETE-001` from clean synchronized commit `42492dc`; all 14 criteria pass.
- [x] Replace the Qwen activation-capture blocker with a bounded generic Hugging Face implementation.
- [x] Execute the pinned Qwen3-4B selected-site capture from clean commit `55087ea`; verify the
  resolved revision, 180 rows, 574,308-byte shard, and SHA-256 manifest.

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
- [x] Commit and push the `LLM-GPT2-003` implementation (`1e57e30`).
- [x] Execute once from the clean implementation commit within 600 seconds.
- [x] Commit the checksummed manifest, metrics, provenance, and synchronized result docs.
