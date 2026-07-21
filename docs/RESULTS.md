# Results

Status: `SMOKE_VALIDATED`.

The scientific results include CPU-scale JEPA studies, three GPT-2 Medium studies, and bounded real
Qwen3-0.6B instrumentation/data/meta-model runs. No workspace/J-space-like mechanism or validated
Qwen circuit has been discovered.

| Result ID | Claim | Evidence Level | Config | Metrics | Commit | Status |
| --- | --- | --- | --- | --- | --- | --- |
| CTRL-000 | CPU resource guard can inspect the current machine without heavy downloads. | Availability | `configs/resource/cpu_vps.yaml` | stdout only | `12fce84` | `SMOKE_VALIDATED` |
| CTRL-GPU-001 | GPU resource guard and CUDA runtime detect the RTX 5070 Ti host; the cross-platform reproducibility audit accepts missing ignored shards only as skipped. | Availability | `configs/resource/gpu_12gb.yaml` | stdout only | `99854eb` plus portability fix | `SMOKE_VALIDATED` |
| WM-T0-001 | Tiny action-conditioned JEPA predicts PointMass2D latent transitions better than mean, no-action, and shuffled-action controls in the CPU smoke setting. | Availability | `configs/experiments/tiny_jepa_smoke.yaml` | `artifacts/metrics/tiny_jepa_smoke.json` | `0cab19a6c39c98b59f1a2172eb11a64ec5a566a4` | `SMOKE_VALIDATED` |
| LLM-MOCK-001 | Mock intervention-JEPA predicts held-out direct intervention effects better than no-change, mean-effect, and linear-context baselines in a deterministic mock transformer. | Availability | `configs/experiments/mock_qwen_intervention_jepa_smoke.yaml` | `artifacts/metrics/mock_qwen_intervention_jepa_smoke.json` | `85c1dbfbe9c824bcca415af13f4a6f34acc95267` | `SMOKE_VALIDATED` |
| WM-T0-002 | Tiny JEPA latent displacement decodes action better than endpoints; replaying explicit action-input coordinates exactly mediates the donor action effect against norm-matched controls. | Causal mediation | `configs/experiments/tier0_mechanistic_study.yaml` | `artifacts/metrics/tier0_mechanistic_study.json` | `315d8cfa3e9640808e316176f45b84c31410c0f8` | `SMOKE_VALIDATED` |
| LLM-GPT2-001 | GPT-2 Medium residual-stream steering at layer 12 causes measurable downstream hidden/logit changes, and a tiny intervention-JEPA predicts held-out effects better than no-change in this smoke setting. | Causal mediation | `configs/experiments/gpt2_medium_intervention_smoke.yaml` | `artifacts/metrics/gpt2_medium_intervention_smoke.json` | `59795a4280b1c8cb372eea000f30de584476dde6` | `SMOKE_VALIDATED` |
| WM-T0-003 | The workspace detector passes planted shared/disjoint controls, but the tiny JEPA five-consumer candidate fails consumer validity, PCA specificity, and rollout-control validity; no workspace was found. | Specificity | `configs/experiments/workspace_discovery_study.yaml` | `artifacts/metrics/workspace_discovery_study.json` | `5223a54ea96fbb6b0481120301c78547e8aabff4` | `SMOKE_VALIDATED` |
| LLM-GPT2-002 | GPT-2 Medium intervention effects are almost exactly local-linear in this coordinate/magnitude regime; bilinear meta-model compression beats weak regressions on held-out prompts but not a local Jacobian and does not transfer to a held-out layer. | Causal mediation | `configs/experiments/gpt2_medium_mechanistic_study.yaml` | `artifacts/metrics/gpt2_medium_mechanistic_study.json` | `8fbab8c0a791cca8b34ba8e1e49664f16e79674d` | `SMOKE_VALIDATED` |
| WM-T0-004 | Conditional donor controls repair the prior off-manifold confound, but neither learned hidden site has a valid shared sensitivity subspace or specificity over matched controls; no workspace is found. | Specificity | `configs/experiments/manifold_workspace_study.yaml` | `artifacts/metrics/manifold_workspace_study.json` | `6785fb1684a04ff1639d6aad326ed6e11df0bf6a` | `SMOKE_VALIDATED` |
| WM-T0-005 | A multi-task three-seed JEPA fails held-out action, consumer-transfer, shared-sensitivity, and task-counterfactual gates; no shared task-workspace candidate or workspace is found. | Generalization | `configs/experiments/multitask_workspace_study.yaml` | `artifacts/metrics/multitask_workspace_study.json` | `7a9e510e84e7166ce862ca7f52fd598630e8f06a` | `SMOKE_VALIDATED` |
| LLM-GPT2-003 | Magnitude-6 contrast-direction compositions are almost additive; prompt-local Jacobians predict them, but learned/corpus transports fail on unseen prompts despite excellent seen-prompt scores. | Generalization | `configs/experiments/gpt2_medium_semantic_composition_study.yaml` | `artifacts/metrics/gpt2_medium_semantic_composition_study.json` | `1e57e30218295a6e6802c2db16cf81c353d8d77d` | `SMOKE_VALIDATED` |
| LLM-QWEN-001 | Pinned Qwen3-0.6B selected-site hooks are deterministic and preserve autograd; five direct residual interventions change downstream hidden states and logits. | Causal mediation | `configs/experiments/qwen3_0_6b_instrumentation_smoke.yaml` | `artifacts/metrics/qwen3_0_6b_instrumentation_smoke.json` | `0d6a37b2f27a862b5d272f254e451fb41b7837e4` | `SMOKE_VALIDATED` |
| LLM-INTDATA-001 | A split-controlled real Qwen3-0.6B intervention dataset contains 432 nonzero directly executed effects with a checksum-verified HDF5 shard. | Causal mediation | `configs/experiments/qwen_intervention_dataset_v1.yaml` | `artifacts/metrics/qwen3_0_6b_intervention_dataset.json` | `0aa80acc9a6fb17d3fc90dba5b2a5bc358326fb2` | `SMOKE_VALIDATED` |
| LLM-IJEPA-001 | A nonlinear Intervention-JEPA generalizes across fixed prompt, coordinate, and operation holdouts better than the registered parametric/Jacobian baselines across three seeds, but its top-ranked coordinate fails direct causal ranking verification. | Generalization | `configs/experiments/intervention_jepa_v1.yaml` | `artifacts/metrics/qwen_intervention_jepa_v1.json` | `a54f2ed6a2491fb905978cb3c10af655a36c7b42` | `SMOKE_VALIDATED`; candidate `REJECTED` |
| WM-LEWM-001A | A source-traceable faithful small LeWorldModel reproduction learns noncollapsed action-conditioned pixel dynamics across all three registered seeds. | Generalization | `configs/experiments/lewm_small_reproduction_v1.yaml` | `artifacts/metrics/lewm_small_reproduction_v1.json` | `4dbc38856b2f1aa6e42754ade72941f0399d3b93` | `SMOKE_VALIDATED` sub-result |
| WM-LEWM-001B | A four-dimensional hidden action-subspace projection changes future latent/decoded trajectories, planning costs, and selected actions beyond a matched-control cost gate on two of three seeds. | Specificity | `configs/experiments/lewm_small_reproduction_v1.yaml` | `artifacts/metrics/lewm_small_reproduction_v1.json` | `4dbc38856b2f1aa6e42754ade72941f0399d3b93` | `SMOKE_VALIDATED` sub-result |
| WM-LEWM-001C | Donor decoded recovery and the full restricted action-to-planner circuit pass only one seed; the replicated gate and every workspace candidate fail. | Circuit reconstruction | `configs/experiments/lewm_small_reproduction_v1.yaml` | `artifacts/metrics/lewm_small_reproduction_v1.json`; rejected graph | `4dbc38856b2f1aa6e42754ade72941f0399d3b93` | `COMPLETED_NEGATIVE`; graph `REJECTED` |

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
circuit; it is not a discovered learned J-space-like workspace. The later 2026-07-21 audit
downgrades the top-level evidence from Specificity to Causal mediation: the controls are input-L2
matched but not matched on activation distribution or downstream effect, and the planner comparison
edits environment actions rather than an internal learned representation. The original artifact's
`evidence_level` field is retained as historical run output and is superseded by this audit.

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

`WM-T0-005` key metrics:

- runtime: `57.51` seconds from clean commit `7a9e510`; passing seeds: `0/3`;
- action-conditioning MSE ratios: `0.712`, `1.012`, `1.003`, all above the `0.50` maximum;
- held-out coverage: `0.946`, `0.986`, `0.855`; OOD AUC: `0.604`, `0.652`, `0.560`; only seed
  `29` passed the joint uncertainty gate;
- all-consumer held-out validity: false on every seed; dynamics R2 was `-2.872`, `0.398`, `0.726`,
  while value, risk, uncertainty, and action-selection transfer was predominantly negative;
- six-of-24 candidate minimum sensitivity capture: `0.583`, `0.561`, `0.574`, below `0.70`;
- task-counterfactual mean donor recovery: `-10.18`, `-17.38`, `-3.11`, below `+0.50`;
- seed `37` random-control rollout damage appeared selective, but only four local-tangent controls
  matched and their p95 exceeded candidate damage; its counterfactual recovery was still negative;
- shared task-workspace candidate: `false`; workspace found: `false`.

Interpretation: adding task context did not make the learned latent computation compositionally
reusable. The post-hoc consumers mostly failed on the unseen goal/dynamics pairing, and candidate
coordinate swaps did not move their outputs toward the donor task. Relative wins against weak
random controls can occur even with the wrong absolute effect; the recovery floor and tangent
controls prevented that false positive. This is evidence against this architecture, not against
workspace-like mechanisms in larger or jointly trained JEPAs.

`LLM-GPT2-003` key metrics:

- runtime: `392.85` seconds from clean commit `1e57e30`; outcomes: `72`; shard: 24,933 bytes;
- held-out composition effect power: `0.4177`; direct interaction MSE: `0.000179`; interaction
  fraction: `0.000429`, below the preregistered `0.05` nonlinear-regime floor;
- held-out prompt-local large-additive MSE/correlation: `0.000179` / `0.99979`;
- held-out prompt-local Jacobian MSE/correlation: `0.000990` / `0.99892`;
- held-out no-change/MLP/linear/bilinear MSE: `0.4177` / `0.7252` / `0.7746` / `1.3460`;
- held-out learned-model correlations were negative; H-LLM-01/02/03 all failed;
- seen-prompt bilinear MSE/correlation: `0.001096` / `0.99851`, demonstrating prompt memorization;
- selected-logit mean absolute delta: `0.3992`; top-token changes: `2/72` overall and `1/24`
  composed outcomes;
- no Qwen, semantic feature, J-space, workspace, or reusable circuit claim is supported.

Interpretation: these constructed contrast edits causally affect GPT-2 output, but their downstream
effects compose almost linearly even at magnitude six. A few same-prompt finite-difference probes
provide an accurate local transport; averaging across prompts or fitting four prompt contexts does
not. The sharp seen/held-out gap is the most useful result: a meta-model can appear to learn
composition while only memorizing prompt-specific Jacobians.

`LLM-QWEN-001` key metrics:

- pinned/resolved revision: `c1899de289a04d12100db370d81485cdf75e47ca`;
- clean-code commit: `0d6a37b2f27a862b5d272f254e451fb41b7837e4`;
- deterministic repeat maximum absolute logit error: `0.0`;
- selected-logit gradient norm with respect to `blocks.14.resid_post`: `0.9443`;
- mean absolute logit delta: zero `0.05647`, mean `0.01115`, donor patch/resample `0.008046`,
  magnitude-2 steer `0.03208`;
- target hidden L2 change: `13.42` to `76.51` across operations.

Interpretation: Hugging Face Qwen hidden-state instrumentation, direct interventions, replay, and
autograd work on this host. Resample and patch intentionally share one donor in this acceptance
test and therefore have identical effects. This result has no held-out effect-prediction split and
does not support a feature, circuit, meta-model, behavior, J-space, or workspace claim.

`LLM-INTDATA-001` key metrics:

- outcomes/prompts/sites: `432` / `12` / layers `7,14,21`;
- runtime: `33.85` seconds on RTX 5070 Ti;
- shard: 412,332 bytes, SHA-256
  `3cf0411b321c87e07465eeab1fdd53a00d366d257cf8b651d90be18279655e8f`;
- top-token changes: `17/432` (`3.94%`);
- mean target-effect L2: `19.65`;
- mean prompt-local 5-percent linear-approximation MSE: `139.83`.

Interpretation: the real intervention data is generated and leakage/storage gates pass. The large
local-linear error makes the mixed edit regime discriminating, but no learned predictor has yet
been evaluated at this dataset-only stage. Dataset existence and nonlinearity alone do not establish
H-LLM-01 or causal compression.

`LLM-IJEPA-001` key metrics:

- primary MSE/correlation: Intervention-JEPA `3.9227`/`0.6770`, no-change `7.2429`, mean `7.1189`,
  linear `27.1688`, bilinear `357.3145`, MLP `9.6423`, nearest-neighbor `5.7200`, local Jacobian
  `116.1557`, corpus Jacobian `26.5092`, sparse-linear `12.0765`;
- unseen coordinate MSE/correlation: `8.4928`/`0.5349`; unseen resampling operation:
  `2.1405`/`0.6802`;
- nearest-neighbor narrowly won unseen-resampling MSE at `2.0946`, limiting the nonlinear advantage;
- the 16 direct verification edits had predicted/observed effect correlation `0.6725`, but
  precision@1 was `0`, meta/random effect ratio `0.9975`, and no top token changed;
- the learned dictionary density was `0.974`, so it is not evidence for sparse or monosemantic
  features.

Interpretation: the fixed H-LLM-01/02/03 gates passed on all three seeds, supporting bounded causal
compression/generalization for this selected target. H-LLM-06 failed, the coordinate graph is
`REJECTED`, and neither a circuit, semantic feature, behavior mechanism, J-space, nor workspace is
supported.

`WM-LEWM-001` key metrics:

- official source revision `8edfeb3...`; clean repository commit `4dbc388`; runtime `54.50` seconds;
- three/three faithful-reproduction seeds passed, including noncollapsed latents and nonlinear
  physical-state probe R2 `0.746`-`0.879`;
- action-embedding suppression raised prediction error `1.59x`-`6.66x`; full action-embedding donor
  patches recovered `1.0` of the donor latent effect on every seed;
- planner subspace removal changed `0.667`-`0.833` of first actions and latent trajectories by
  `0.372`-`0.598` MSE; registered matched-control planning specificity passed two seeds;
- complete hidden donor-patch specificity and restricted circuit gates passed only seed 107, so the
  two-of-three replicated claims failed and the aggregate graph is `REJECTED`;
- clean/intervened closed-loop success averaged `0.083/0.0`; the planner is weak and no successful
  control result is claimed;
- five-consumer shared-sensitivity candidates failed all seeds; ensemble variance decreased from
  `0.506` to `0.250` under intervention, suggesting overconfidence rather than useful uncertainty.

Interpretation: the published recipe is reproduced at small scale, and a targeted internal
intervention has selective trajectory/cost/action effects on two seeds. The proposed hidden circuit
does not generalize across seeds, decoded donor recovery is unstable, and there is no workspace.
