# Results

Status: `SMOKE_VALIDATED`.

The current results are CPU-scale smoke validation, corrected tiny/deep-JEPA causal tests, and two
GPT-2 Medium direct-intervention studies. No workspace/J-space-like mechanism has been discovered.

| Result ID | Claim | Evidence Level | Config | Metrics | Commit | Status |
| --- | --- | --- | --- | --- | --- | --- |
| CTRL-000 | CPU resource guard can inspect the current machine without heavy downloads. | Availability | `configs/resource/cpu_vps.yaml` | stdout only | `12fce84` | `SMOKE_VALIDATED` |
| WM-T0-001 | Tiny action-conditioned JEPA predicts PointMass2D latent transitions better than mean, no-action, and shuffled-action controls in the CPU smoke setting. | Availability | `configs/experiments/tiny_jepa_smoke.yaml` | `artifacts/metrics/tiny_jepa_smoke.json` | `0cab19a6c39c98b59f1a2172eb11a64ec5a566a4` | `SMOKE_VALIDATED` |
| LLM-MOCK-001 | Mock intervention-JEPA predicts held-out direct intervention effects better than no-change, mean-effect, and linear-context baselines in a deterministic mock transformer. | Availability | `configs/experiments/mock_qwen_intervention_jepa_smoke.yaml` | `artifacts/metrics/mock_qwen_intervention_jepa_smoke.json` | `85c1dbfbe9c824bcca415af13f4a6f34acc95267` | `SMOKE_VALIDATED` |
| WM-T0-002 | Tiny JEPA latent displacement decodes action better than endpoints; replaying explicit action-input coordinates exactly mediates the donor action effect against norm-matched controls. | Specificity | `configs/experiments/tier0_mechanistic_study.yaml` | `artifacts/metrics/tier0_mechanistic_study.json` | `315d8cfa3e9640808e316176f45b84c31410c0f8` | `SMOKE_VALIDATED` |
| LLM-GPT2-001 | GPT-2 Medium residual-stream steering at layer 12 causes measurable downstream hidden/logit changes, and a tiny intervention-JEPA predicts held-out effects better than no-change in this smoke setting. | Causal mediation | `configs/experiments/gpt2_medium_intervention_smoke.yaml` | `artifacts/metrics/gpt2_medium_intervention_smoke.json` | `59795a4280b1c8cb372eea000f30de584476dde6` | `SMOKE_VALIDATED` |
| WM-T0-003 | The workspace detector passes planted shared/disjoint controls, but the tiny JEPA five-consumer candidate fails consumer validity, PCA specificity, and rollout-control validity; no workspace was found. | Specificity | `configs/experiments/workspace_discovery_study.yaml` | `artifacts/metrics/workspace_discovery_study.json` | `5223a54ea96fbb6b0481120301c78547e8aabff4` | `SMOKE_VALIDATED` |
| LLM-GPT2-002 | GPT-2 Medium intervention effects are almost exactly local-linear in this coordinate/magnitude regime; bilinear meta-model compression beats weak regressions on held-out prompts but not a local Jacobian and does not transfer to a held-out layer. | Causal mediation | `configs/experiments/gpt2_medium_mechanistic_study.yaml` | `artifacts/metrics/gpt2_medium_mechanistic_study.json` | `8fbab8c0a791cca8b34ba8e1e49664f16e79674d` | `SMOKE_VALIDATED` |
| WM-T0-004 | Conditional donor controls repair the prior off-manifold confound, but neither learned hidden site has a valid shared sensitivity subspace or specificity over matched controls; no workspace is found. | Specificity | `configs/experiments/manifold_workspace_study.yaml` | `artifacts/metrics/manifold_workspace_study.json` | `6785fb1684a04ff1639d6aad326ed6e11df0bf6a` | `SMOKE_VALIDATED` |

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

`LLM-MOCK-001` key metrics:

- intervention-JEPA MSE: `1.317838069780919e-07`;
- no-change MSE: `0.0021134756971150637`;
- mean-effect MSE: `0.002113459398970008`;
- linear-context MSE: `0.0021134584676474333`;
- effect correlation: `0.9999688241988822`.

This result is mock-only and must not be reported as Qwen evidence.

`WM-T0-002` key metrics:

- displacement action R2: `0.999999995077069`;
- z[t] action R2: `-0.11095967931965078`;
- z[t+1] action R2: `0.08287789864620154`;
- action-input replay recovery: `1.0`;
- replay max absolute error: `0.0`;
- L2 norm-matched latent-control recovery: `-12.783885946497321`;
- L2 norm-matched random-action-control recovery: `-0.3163457727059722`;
- workspace found: `false`.

Audit correction: the earlier `0.984` result came from setting `action_patch = target` and is
superseded. The clean rerun on commit `315d8cf` uses the intervention engine, excludes zero-effect
pairs, and L2 norm-matches controls. Exact recovery is expected because the intervention replays
explicit action input coordinates through a linear predictor. This localizes a trivial action-input
circuit; it is not a discovered learned J-space-like workspace.

`LLM-GPT2-001` key metrics:

- model: `gpt2-medium`;
- site: `transformer.h.12.resid_post`;
- prompts: `4`;
- direct intervention mean absolute logit delta: `0.07971649616956711`;
- intervention-JEPA MSE: `0.0021958956494927406`;
- no-change MSE: `0.01140519231557846`;
- mean-effect MSE: `0.011404994875192642`;
- linear-context MSE: `0.011405046097934246`;
- effect correlation: `0.9760097933956744`.

Interpretation: real GPT-2 Medium residual interventions affect downstream computation and can be meta-predicted in a tiny smoke setting. This does not establish a J-space or workspace.

`WM-T0-003` key metrics:

- planted shared control found: `true`; disjoint-consumer control found: `false`;
- proposed dimension: `3/16`; minimum normalized consumer sensitivity capture: `0.886`;
- candidate mean direct-readout damage: `1.306`; random-control maximum: `0.883`;
- PCA control damage: `1.527`, greater than the proposed candidate;
- uncertainty-head held-out R2: `-3.639`, failing the preregistered `0.8` floor;
- random rollout-control mean damage: `53050.3`, showing severe off-manifold instability;
- shared causal subspace candidate found: `false`; workspace found: `false`.

Interpretation: the detector can recognize a planted shared subspace, but this tiny JEPA result does
not survive adversarial controls. The three-dimensional direction set mostly tracks the ordinary
physical-state manifold; the uncertainty readout is invalid and arbitrary random projection is not
a valid rollout control because it drives the linear model far off its training manifold.

`LLM-GPT2-002` key metrics:

- direct outcomes: `288`; train examples: `96`; primary held-out examples: `16`;
- selected-logit mean absolute direct effect: `0.04862`; top-token change rate: `0.0`;
- primary local-Jacobian MSE/correlation: `7.79e-7` / `0.99994`;
- primary bilinear Intervention-JEPA MSE/correlation: `0.003499` / `0.6973`;
- primary linear/no-change MSE: `0.006360` / `0.006461`;
- primary trained-MLP MSE/correlation: `0.007361` / `0.0613`;
- held-out-layer bilinear MSE: `0.01009`, worse than no-change `0.006228`;
- held-out-layer local-Jacobian MSE/correlation: `3.71e-8` / `0.999998`;
- H-LLM-01 nonlinear advantage: `false`; restricted H-LLM-02 causal compression: `true`;
- runtime: `647.85` seconds, exceeding the CPU profile expectation by `47.85` seconds.

Interpretation: unit coordinate interventions up to magnitude one are effectively local-linear at
the selected GPT-2 sites and outputs. A context-conditioned bilinear predictor captures some
prompt-general effect structure, but the local direct Jacobian is approximately 4,490 times lower
MSE and cross-layer transfer fails. This is evidence against using a JEPA meta-model for this easy
linear regime, not evidence against nonlinear meta-models for larger, semantic, combined, or
off-manifold interventions. No behavior change or GPT-2 workspace was found.

`WM-T0-004` key metrics:

- runtime: `31.22` seconds from clean commit `6785fb1`;
- primary/shuffled-action MSE: `0.003416` / `0.004994`; ratio `0.684`, failing the registered `0.25`;
- validation/test 90 percent interval coverage: `0.8999` / `0.8870`;
- OOD uncertainty AUC: `0.5738`, below `0.65`; uncertainty/error Spearman: `0.6977`;
- hidden-1/hidden-2 uncertainty-head R2: `-1.327` / `0.232`, below `0.50`;
- hidden-1/hidden-2 minimum consumer sensitivity capture: `0.635` / `0.701`, below `0.75`;
- conditional-patch median density ratios: `1.029` / `1.018`; matched random controls: `63/64` and
  `64/64`, showing that the previous off-manifold random-control failure was repaired;
- candidate direct damage: `3.652` / `1.239`; matched-control p95: `9.252` / `4.369`;
- candidate multistep damage: `0.520` / `0.289`; matched-control p95: `0.644` / `0.568`;
- shared causal candidate sites: none; workspace found: `false`.

Interpretation: the ensemble produces useful in-distribution intervals and its score correlates with
error, but it does not rank the preregistered physical OOD shift well enough and the primary model's
hidden states do not make ensemble uncertainty reliably decodable. Conditional resampling keeps
candidate and random-control patches near the empirical activation bank. Under those valid controls,
the candidate directions are not privileged. Multistep amplification exists but is nonspecific.
