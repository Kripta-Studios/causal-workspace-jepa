# Reproducibility

Status: `SCAFFOLDED`.

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
```

No reported result may depend on uncommitted code.
