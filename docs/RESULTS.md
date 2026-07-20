# Results

Status: `SMOKE_VALIDATED`.

No causal scientific mechanism result has been produced yet. The current results are CPU smoke validation only.

| Result ID | Claim | Evidence Level | Config | Metrics | Commit | Status |
| --- | --- | --- | --- | --- | --- | --- |
| CTRL-000 | CPU resource guard can inspect the current machine without heavy downloads. | Availability | `configs/resource/cpu_vps.yaml` | stdout only | pending | `SMOKE_VALIDATED` |
| WM-T0-001 | Tiny action-conditioned JEPA predicts PointMass2D latent transitions better than mean, no-action, and shuffled-action controls in the CPU smoke setting. | Availability | `configs/experiments/tiny_jepa_smoke.yaml` | `artifacts/metrics/tiny_jepa_smoke.json` | `0cab19a6c39c98b59f1a2172eb11a64ec5a566a4` | `SMOKE_VALIDATED` |

Validation commands run before the Milestone 0 commit:

```bash
PYTHONPATH=src python -m causal_workspace_jepa.cli doctor --resource-profile configs/resource/cpu_vps.yaml
PYTHONPATH=src python -m unittest discover -s tests -p 'test_*.py'
python scripts/audit_reproducibility.py
git diff --check
```

All future result tables must include `evidence_level`.

`WM-T0-001` key metrics:

- conditioned latent MSE: `1.093934498541671e-09`;
- mean baseline latent MSE: `0.24916456639766693`;
- no-action latent MSE: `0.13531547784805298`;
- shuffled-action latent MSE: `0.15291918814182281`;
- planner true cost: `1.4791598320007324`;
- random rollout mean cost: `1.938533022068441`.
