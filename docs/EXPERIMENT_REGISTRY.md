# Experiment Registry

No scientific experiments have run.

| ID | Hypothesis | Date | Owner | Status | Model | Dataset | Config | Command | Commit | Hardware | Seeds | Runtime | Output | Metric | Threshold | Result | Caveats | Next Action |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| CTRL-000 | none | 2026-07-20 | root agent | `SMOKE_VALIDATED` | none | none | `configs/resource/cpu_vps.yaml` | `python -m causal_workspace_jepa.cli doctor --resource-profile configs/resource/cpu_vps.yaml` | pending | CPU VPS | none | seconds | stdout | resource guard ok | free disk >= 4 GB | passed | Not a scientific result. | Proceed to Milestone 1. |
| WM-T0-001 | H-WM-02 | pending | root agent | `IMPLEMENTED_UNVALIDATED` | tiny JEPA | Tier 0 PointMass2D | `configs/experiments/tiny_jepa_smoke.yaml` | `python scripts/run_experiment.py --config configs/experiments/tiny_jepa_smoke.yaml` | pending | CPU VPS | 0 | < 10 min | `artifacts/metrics/tiny_jepa_smoke.json` | MSE vs mean baseline, no-action, shuffled-action, planner random baseline | all checks true | pending clean rerun | Smoke result only, evidence level Availability. | Commit code, rerun clean, commit metrics. |
| LLM-MOCK-001 | H-LLM-01 | pending | root agent | `NOT_STARTED` | mock transformer | synthetic prompts | `configs/experiments/mock_qwen_intervention_jepa_smoke.yaml` | `python scripts/run_experiment.py --config configs/experiments/mock_qwen_intervention_jepa_smoke.yaml` | pending | CPU VPS | 0 | < 10 min | `artifacts/metrics/mock_qwen_intervention_jepa_smoke.json` | held-out effect MSE | beats no-change | pending | Mock only, not Qwen evidence. | Implement Milestone 2. |
