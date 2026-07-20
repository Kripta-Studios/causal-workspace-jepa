# SUMMARY

## 2026-07-20

### Done

- Read `AGENTS.md` and `VPS_RUNBOOK.md` fully.
- Confirmed current machine is the CPU VPS path: no GPU, 4 CPU cores, about 7 GB free at start.
- Added package metadata, resource profiles, package tree, resource guard, doctor CLI, typed interfaces, provenance helpers, and docs skeleton.
- Added standard-library unit/integration tests for config parsing, resource guards, CLI output, intervention serialization, and reproducibility scaffolding.
- Validated Milestone 0 with `doctor`, `python -m unittest discover -s tests -p 'test_*.py'`, `scripts/audit_reproducibility.py`, and `git diff --check`.
- Implemented deterministic Tier 0 generators: PointMass2D, BouncingBall2D, TwoBodyCollision, TinyMaze, and MiniPush.
- Implemented a tiny NumPy action-conditioned JEPA with fixed encoder, ridge-fitted predictor, save/load, named activation points, and no-action/shuffled-action controls.
- Implemented a random-shooting planner and PointMass closed-loop cost check.
- Added Milestone 1 tests. Current full test command passes 14 tests.
- Committed and pushed Milestone 1 code as `0cab19a6c39c98b59f1a2172eb11a64ec5a566a4`.
- Reran Tier 0 generation and `WM-T0-001` tiny JEPA smoke from clean committed code.
- Committed and pushed result summaries as `b0b7796`: `data/manifests/tier0_smoke_manifest.json`, `artifacts/metrics/tiny_jepa_smoke.json`, and `artifacts/metrics/tiny_jepa_smoke.provenance.json`.

### Discovered

- `README.md` initially only contained the project title.
- `pytest` is not installed, but NumPy is available system-wide.
- The runbook contains a GPT-2/GPT2-medium note that conflicts with the user instruction not to download weights or install Transformers. For this run it is treated as `BLOCKED_RESOURCE`.
- A first dirty-run tiny JEPA smoke pass succeeded; generated summaries were removed before the code commit so the final reported metrics can be rerun from committed code.
- Final smoke provenance reports `git_dirty: false` and code commit `0cab19a6c39c98b59f1a2172eb11a64ec5a566a4`.
- `WM-T0-001` beat the mean, no-action, and shuffled-action baselines on latent prediction, and the planner beat average random rollout cost. This is evidence level Availability only.
- Implemented intervention operators: zero, mean, resample, patch, replace_feature, steer, project_out, scale, and suppress_module.
- Implemented activation cache, stable site naming, finite-difference Jacobian, normalized patch recovery, ridge probes, random-label control, sparse dictionary, circuit graph JSON/GraphML, and mock transformer adapter.
- Implemented mock intervention-JEPA smoke runner and tests. Exploratory dirty-run metrics passed, then were removed before the code commit.
- Committed and pushed Milestone 2 code as `85c1dbfbe9c824bcca415af13f4a6f34acc95267`.
- Reran `LLM-MOCK-001` from clean committed code. Provenance reports `git_dirty: false`.
- Committed and pushed mock result summaries as `948347c`: `artifacts/metrics/mock_qwen_intervention_jepa_smoke.json` and `artifacts/metrics/mock_qwen_intervention_jepa_smoke.provenance.json`.
- Added `scripts/bootstrap_cpu.sh`, config `.gitkeep` files, and reporting stubs to complete the requested architecture.
- `uv` is not installed on the VPS. The bootstrap script checks resources and reports the exact install command instead of auto-installing tools.
- `/root/.cache/pip` is about 4.4 GB and was not modified because it may predate this task; free disk remains above the 4 GB guard.
- Ran `WM-T0-002` Milestone 3 tiny-JEPA mechanistic study. Displacement action R2 is nearly 1.0; endpoint action R2 values are much weaker. Action-coordinate patching has recovery `0.984` versus negative matched controls. Workspace result is null: no J-space-like candidate found.
- Under explicit user override, created `.venv`, installed `transformers`/`safetensors`, downloaded `gpt2-medium`, and ran `LLM-GPT2-001`.
- GPT-2 Medium result: direct residual steering at `transformer.h.12.resid_post` changed logits with mean absolute delta `0.0797`; tiny intervention-JEPA MSE `0.00220` beat no-change `0.0114`.

### Next Ideas

- Commit and push Milestone 3 and GPT-2 Medium code, metrics, and docs.
