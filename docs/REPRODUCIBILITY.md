# Reproducibility

Status: `SMOKE_VALIDATED` for current CPU experiments.

Every experiment must record:

- config path and serialized config;
- data manifest and checksums;
- code commit and dirty flag;
- hardware/resource profile;
- seed;
- command;
- logs;
- metrics with `evidence_level`;
- failure status when applicable.

CPU path:

```bash
export PYTHONPATH=src
python -m causal_workspace_jepa.cli doctor --resource-profile configs/resource/cpu_vps.yaml
python -m unittest discover -s tests -p 'test_*.py'
```

Validated smoke commands:

```bash
PYTHONPATH=src python scripts/generate_tier0.py --config configs/data/tier0_smoke.yaml
PYTHONPATH=src python scripts/run_experiment.py --config configs/experiments/tiny_jepa_smoke.yaml
PYTHONPATH=src python scripts/run_experiment.py --config configs/experiments/mock_qwen_intervention_jepa_smoke.yaml
PYTHONPATH=src python scripts/run_experiment.py --config configs/experiments/tier0_mechanistic_study.yaml
PYTHONPATH=src python scripts/run_experiment.py --config configs/experiments/workspace_discovery_study.yaml
PYTHONPATH=src .venv/bin/python scripts/run_experiment.py --config configs/experiments/multitask_workspace_study.yaml
PYTHONPATH=src .venv/bin/python scripts/run_experiment.py --config configs/experiments/gpt2_medium_mechanistic_study.yaml
PYTHONPATH=src .venv/bin/python scripts/run_experiment.py --config configs/experiments/gpt2_medium_semantic_composition_study.yaml
```

GPT-2 Medium uses `local_files_only: true`; the command must fail instead of downloading a missing
model. Its generated float16 shard is ignored, while the checksum manifest and summarized metrics
are committed.

No reported result may depend on uncommitted code.

`scripts/audit_reproducibility.py` checks required control-plane files, every summarized metrics and
provenance pair, `git_dirty: false`, recorded commit/path fields, JSON validity, and every available
local dataset checksum. Missing ignored datasets are reported as skipped so a fresh clone can audit
metadata before regeneration; checksum mismatches fail the audit.
