.PHONY: doctor test smoke-tier0 smoke-tiny-jepa audit

doctor:
	PYTHONPATH=src python -m causal_workspace_jepa.cli doctor --resource-profile configs/resource/cpu_vps.yaml

test:
	PYTHONPATH=src python -m unittest discover -s tests -p 'test_*.py'

smoke-tier0:
	PYTHONPATH=src python scripts/generate_tier0.py --config configs/data/tier0_smoke.yaml

smoke-tiny-jepa:
	PYTHONPATH=src python scripts/run_experiment.py --config configs/experiments/tiny_jepa_smoke.yaml

audit:
	PYTHONPATH=src python scripts/audit_reproducibility.py
