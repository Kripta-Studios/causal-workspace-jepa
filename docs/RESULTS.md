# Results

Status: `SMOKE_VALIDATED`; corrected Qwen H-LLM-01 result is negative.

The scientific results include CPU-scale JEPA studies, three GPT-2 Medium studies, and bounded real
Qwen3-0.6B instrumentation/data/meta-model runs. No workspace/J-space-like mechanism or validated
Qwen circuit has been discovered.

| Result ID | Claim | Evidence Level | Config | Metrics | Commit | Status |
| --- | --- | --- | --- | --- | --- | --- |
| WM-ACTION-PATH-CALIBRATION-001 | Validation-only action-path profiling exposes unresolved horizon-four derivatives and weak candidate cancellation/error association; it makes no scientific claim. | Availability/calibration | `configs/experiments/lewm_action_path_calibration_v1.yaml` | `artifacts/metrics/lewm_action_path_calibration_v1.json` | `eb943a5` | `CALIBRATION_ONLY` |
| LLM-COUNTRY-CODE-LAYER-GEOMETRY-001 | Country donor control becomes monotone and late population transport is semantically specific, but population advantage can precede the 50% direct-control boundary; directional bounded lag is rejected. | Generalization | `configs/experiments/qwen_country_code_layer_geometry_v1.yaml` | `artifacts/metrics/qwen_country_code_layer_geometry_v1.json` | clean retry `48226c6` | `COMPLETED_MIXED` |
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
| LLM-IJEPA-001 | The legacy conditional bottleneck beat the originally registered comparators, but its BF16 secant was invalidated by exact-JVP v2; the nonlinear claim is withdrawn and its ranked coordinate failed direct verification. | Generalization | `configs/experiments/intervention_jepa_v1.yaml` | `artifacts/metrics/qwen_intervention_jepa_v1.json` | `a54f2ed6a2491fb905978cb3c10af655a36c7b42` | `SUPERSEDED_NEGATIVE`; candidate `REJECTED` |
| LLM-QWEN-JVP-AUDIT-001 | FP32 exact JVP agrees with central differences and preliminarily dominates the learned bottleneck, but the audit failed its absolute semantic-endpoint gate and therefore cannot decide H-LLM-01. | Specificity | `configs/experiments/qwen_jvp_audit_v1.yaml` | `artifacts/metrics/qwen_jvp_audit_v1.json` | `686368e792598aaeb3d0aff7349d34f8f70a3c36` | `REJECTED` numerical gate |
| LLM-QWEN-JVP-AUDIT-002 | Exact FP32 JVP and quadratic Taylor beat all three legacy conditional-bottleneck seeds after source semantics and derivative convergence pass; restricted H-LLM-01 is withdrawn. | Specificity | `configs/experiments/qwen_jvp_audit_v2.yaml` | `artifacts/metrics/qwen_jvp_audit_v2.json` | `a779ff6ea617f77e2b0c252c79b5a1a1fa66cfdc` | `COMPLETED_NEGATIVE` |
| LLM-CAPITAL-PATCH-001 | Entity-disjoint layer-21 donor patches change Qwen's top-token capital answer in 93.6% of cases and transfer the donor answer in 50% of held-out test pairs; exact and quadratic controls disagree between vector and behavior metrics. | Causal mediation | `configs/experiments/qwen_capital_patch_dataset_v1.yaml` | `artifacts/metrics/qwen_capital_patch_dataset_v1.json` | `95018cb326d5604ed45f128338f66f51b13d04ae` | `SMOKE_VALIDATED`; behavior eligible |
| LLM-TARGET-IJEPA-001 | A genuine target-encoder Intervention-JEPA fails all three replicated held-out-entity hypotheses; even oracle target-embedding decode fails, while exact JVP, linear ridge, and quadratic Taylor win different fidelity endpoints. | Generalization | `configs/experiments/qwen_target_encoder_ijepa_v1.yaml` | `artifacts/metrics/qwen_target_ijepa_v1.json` | `3086cd484fb819c3a11525ee9886542049780955` | `COMPLETED_NEGATIVE` |
| LLM-CONTEXT-GEOMETRY-001 | Real Qwen rejects fixed pooling/context-specificity gaps but confirms that naive Euclidean overlap is gauge-sensitive while paired `J D^T` is invariant; a train-mean Jacobian unexpectedly beats matched local finite transport. | Specificity | `configs/experiments/qwen_context_geometry_v1.yaml` | `artifacts/metrics/qwen_context_geometry_v1.json` | `49d68b72200328657683b9760a084e0d952948b1` | `COMPLETED_MIXED` |
| LLM-POPULATION-JACOBIAN-001 | A preregistered validation-only analysis confirms that the 24-train-context mean Jacobian predicts finite held-out logit effects better than exact local Jacobians, with a context-count dose response and answer-row specificity. | Generalization | `configs/experiments/qwen_population_jacobian_v1.yaml` | `artifacts/metrics/qwen_population_jacobian_v1.json` | `3725714` | `COMPLETED_POSITIVE` |
| LLM-ELEMENT-LAYER-GEOMETRY-001 | A second factual relation confirms a sharp late donor-control transition and semantically specific late population transport, but rejects the preregistered strong local/population inversion and cross-relation conjunction. | Generalization | `configs/experiments/qwen_element_layer_geometry_v1.yaml` | `artifacts/metrics/qwen_element_layer_geometry_v1.json` | `5d8de9a` | `COMPLETED_MIXED` |
| LLM-STATE-LAYER-GEOMETRY-001 | The state-abbreviation confirmation is behavior-ineligible despite valid derivatives; all hypotheses remain undecided. | Availability/eligibility rejection | `configs/experiments/qwen_state_layer_geometry_v1.yaml` | `artifacts/metrics/qwen_state_layer_geometry_v1.json` | `27ebe43` | `REJECTED_BEHAVIOR_GATE` |
| LLM-STATE-ONESHOT-LAYER-GEOMETRY-001 | A behavior-competent one-shot state study confirms late population semantic specificity but rejects exact control/population onset equality and the strict early-control gate. | Generalization | `configs/experiments/qwen_state_oneshot_layer_geometry_v1.yaml` | `artifacts/metrics/qwen_state_oneshot_layer_geometry_v1.json` | `c1daa46` | `COMPLETED_MIXED` |
| WM-POPULATION-JACOBIAN-001 | The recurrent-JEPA port is rejected because fixed quadrature fails; its provisional action-vertex averaging signal is not accepted, while the global mean fails correlation and semantic specificity. | Numerical rejection | `configs/experiments/lewm_population_geometry_v1.yaml` | `artifacts/metrics/lewm_population_geometry_v1.json` | `89b2e14` | `REJECTED_NUMERICAL_GATE` |
| WM-LEWM-001A | A source-traceable faithful small LeWorldModel reproduction learns noncollapsed action-conditioned pixel dynamics across all three registered seeds. | Generalization | `configs/experiments/lewm_small_reproduction_v1.yaml` | `artifacts/metrics/lewm_small_reproduction_v1.json` | `4dbc38856b2f1aa6e42754ade72941f0399d3b93` | `SMOKE_VALIDATED` sub-result |
| WM-LEWM-001B | A four-dimensional hidden action-subspace projection changes future latent/decoded trajectories, planning costs, and selected actions beyond a matched-control cost gate on two of three seeds. | Specificity | `configs/experiments/lewm_small_reproduction_v1.yaml` | `artifacts/metrics/lewm_small_reproduction_v1.json` | `4dbc38856b2f1aa6e42754ade72941f0399d3b93` | `SMOKE_VALIDATED` sub-result |
| WM-LEWM-001C | Donor decoded recovery and the full restricted action-to-planner circuit pass only one seed; the replicated gate and every workspace candidate fail. | Circuit reconstruction | `configs/experiments/lewm_small_reproduction_v1.yaml` | `artifacts/metrics/lewm_small_reproduction_v1.json`; rejected graph | `4dbc38856b2f1aa6e42754ade72941f0399d3b93` | `COMPLETED_NEGATIVE`; graph `REJECTED` |
| LLM-QWEN-CAPTURE-001 | The pinned Qwen3-4B revision fits bounded bfloat16 inference and yields a budgeted, checksummed five-layer/three-position activation capture. | Availability | `configs/llm/qwen3_4b_selected_layers.yaml` | `artifacts/metrics/qwen3_4b_selected_activation_capture.json` | `55087ea1fe12bc361830eb501aed86aaf850e50e` | `SMOKE_VALIDATED` |
| AUDIT-COMPLETE-001 | All 14 explicit bounded repository criteria and the corrected 68-test/Ruff/provenance/JVP checks pass while retaining the circuit rejections and workspace null. | Availability | none | `artifacts/metrics/completion_audit.json` | clean execution `98a9e62` | `SMOKE_VALIDATED` |

Validation commands run before the Milestone 0 commit:

```bash
PYTHONPATH=src python -m causal_workspace_jepa.cli doctor --resource-profile configs/resource/cpu_vps.yaml
PYTHONPATH=src python -m unittest discover -s tests -p 'test_*.py'
python scripts/audit_reproducibility.py
git diff --check
```

All future result tables must include `evidence_level`.

`AUDIT-COMPLETE-001` is a software/evidence-coverage result. It does not elevate any scientific
claim: the Qwen coordinate graph and aggregate LeWorldModel graph remain `REJECTED`, and no
workspace candidate passed the registered tests.

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

Historical interpretation: the fixed H-LLM-01/02/03 gates passed on all three seeds against the
then-registered comparators. H-LLM-01 is now `WITHDRAWN` because the BF16 secant was not an
exact Jacobian, and the model is a supervised conditional bottleneck rather than a genuine JEPA.
H-LLM-06 failed, the coordinate graph is
`REJECTED`, and neither a circuit, semantic feature, behavior mechanism, J-space, nor workspace is
supported.

`LLM-QWEN-JVP-AUDIT-001` key metrics:

- numerical gates: five of six pass; JVP/central median relative error `0.000249`, p95 `0.00381`;
- failed gate: dense-versus-semantic downstream endpoint maximum absolute error `1.335e-4` versus
  preregistered `1e-5`;
- preliminary raw MSE: exact JVP `0.6143`, quadratic Taylor `0.07870`, conditional bottleneck
  `3.1899`, no-change `6.8654`, and historical BF16 secant `120.8994`;
- exact-JVP normalized MSE was `0.08948` raw and `0.09069` after semantic deduplication, below the
  registered `0.10` nonlinearity floor; zero of three learned seeds beat exact JVP;
- FP32 direct execution produced 18/432 top-token changes; BF16/FP32 direct effects had correlation
  `0.9746` but mean row-relative error `0.5146`.

Interpretation: v1 is rejected and cannot withdraw or retain H-LLM-01. A post-result diagnostic found
that replacement paths incur up to `4.768e-7` source rounding when reconstructing `h+(donor-h)`,
which can propagate above an absolute downstream tolerance. A new audit may validate source
semantics directly, but v1's gate and rejected status are immutable.

`LLM-QWEN-JVP-AUDIT-002` key metrics:

- all seven numerical gates passed; direct-source semantic max error `0.0`, no direction-endpoint
  tolerance violations, and JVP/central median/p95 relative errors `0.000249`/`0.00381`;
- raw MSE/correlation: exact JVP `0.6143`/`0.9548`, quadratic Taylor `0.07870`/`0.9947`, conditional
  bottleneck `3.1899`/`0.7347`, nearest neighbor `4.9026`/`0.5862`;
- JVP normalized MSE `0.08948` raw and `0.09069` deduplicated; quadratic normalized MSE `0.01146`;
- finite-amplitude nonlinearity gate failed and learned predictor passing seeds were `0/3`;
- status `SMOKE_VALIDATED` as an audit, scientific disposition `WITHDRAWN` for restricted H-LLM-01.

Interpretation: within this fixed selected-target grid, the apparent nonlinear advantage was a BF16
finite-difference artifact. First- and especially second-order local transport are strong. This does
not generalize to semantic behavior-changing interventions that were not tested.

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

`WM-POPULATION-JACOBIAN-001` key metrics:

- clean execution commit/runtime: `89b2e14`, `55.20` seconds; validation goals only;
- horizon-four decoded local/population/within-context-vertex MSE by seed:
  `19.18/0.997/4.48`, `38.29/1.415/13.16`, and `2.401/0.987/0.885`;
- population correlations were only `0.073/0.013/0.116`, no action-column specificity gate passed,
  and only one of three averaging-dose gates passed, so low population MSE is not semantic fidelity;
- the vertex mean provisionally beat local at both horizons in all seeds, but cannot be accepted
  because the registered numerical gate rejected the run;
- horizon-four 12-node path median/p95/max relative errors were
  `0.116/4.23/11.18`, `0.704/8.89/49.58`, and `1.27e-4/0.00109/0.0822`;
- decoded gauge invariance held to at most `2.73e-12`; direct planner competence was
  `0.364/0.514/0.295`, below the fixed `0.60` floor for every seed.

Interpretation: exact local derivatives can become very large and cancel along recurrent finite
action chords. The fixed quadrature was inadequate, so the experiment decides no scientific
hypothesis. A post-result diagnostic verified local derivatives by central differences and found
that the worst chord needed 192 quadrature nodes to reduce error from `49.8` to `0.0058`. The
untouched test goals remain protected; validation calibration must converge first.

`WM-ACTION-PATH-CALIBRATION-001` key metrics:

- clean commit/runtime: `eb943a5`, `72.59` seconds; validation goals only; 24 sampled chords per
  seed/horizon; `protected_test_goals_touched=false`; no hypothesis decisions;
- horizon-one maximum direct/refinement error by seed was `.000645/.00140`,
  `.000767/.00168`, and `.000170/.000162`;
- horizon-four maximum direct/refinement error was `.0680/.1369`, `.478/2.059`, and
  `.00197/.000933`; seeds 101/103 remain numerically underresolved at 256 nodes;
- horizon-four cancellation/local-error Spearman was `.415/.483/.863`, exceeding stratified-null
  p95 by `.045/.027/.022`; these small margins are provisional until convergence;
- median horizon-four/horizon-one cancellation amplification was `1.05/2.82/.983`, so recurrence
  amplification does not replicate at the calibration sample size.

Interpretation: exact path profiling is feasible and suggests chord-specific cancellation may track
local linearization failure, but the stiffest seeds are not numerically stable and the effect is
weak relative to a strong within-action-pair null. This is calibration, not evidence or novelty.

`LLM-ELEMENT-LAYER-GEOMETRY-001` key metrics:

- clean execution commit/runtime: `5d8de9a`, `190.28` seconds; 2,448 direct patches and 144 complete
  `36 x 1024` selected-answer Jacobians;
- all four numerical gates passed; exact-Jacobian/central p95 relative error was at most `0.0402`,
  with exact clean replay and donor-source semantics;
- validation full-vocabulary donor-symbol transfer by layer 18/21/24/26: `0/0/0.60/1.00`; test:
  `0/0/0.90/1.00`; H-LLM-08 passed;
- test local/population normalized MSE by layer: `0.1138/0.3672`, `0.0809/0.2483`,
  `0.2558/0.1808`, and `0.0916/0.01131`;
- layer-24/26 test population correlation was `0.9767/0.9951` and candidate agreement `0.90/1.00`;
  corresponding answer-row-null p05 MSE was `2.372/1.772` and p95 agreement `0.167/0.108`, so
  H-GEO-09 passed;
- H-GEO-08 failed: layer-24 population/local MSE ratios were `0.879` validation and `0.707` test,
  above the registered `0.60`, while test layer-21 local/population was `0.326` above `0.25`;
- H-CROSS-03 failed because it required H-GEO-08; quadratic Taylor remains reported and was best at
  early layers but unstable at layer 24.

Interpretation: donor patches become directly behavior-controlling only in late layers, and late
population Jacobians preserve the correct answer-row semantics with an averaging-size dose response.
The preregistered strong predictivity inversion is nevertheless false. MechLens already covers late
factual crystallization and Jacobian Lens already covers corpus averaging, so this is a bounded
causal-geometry observation that needs a new relation/model confirmation, not a SOTA, circuit, or
workspace claim.

`LLM-STATE-LAYER-GEOMETRY-001` key metrics:

- clean execution commit/runtime: `27ebe43`, `183.92` seconds; all four numerical gates passed and
  exact-Jacobian/central p95 relative error was at most `0.0354`;
- clean train/validation/test full-vocabulary accuracy: `0.625/0.667/0.667`; validation/test failed
  the fixed `0.75` competence floor, so status is `REJECTED_BEHAVIOR_GATE`;
- validation/test donor transfer by layer 18/21/24/26: `0/0/0.067/0.667` and
  `0/0/0.400/0.667`;
- validation population advantage over the better local/quadratic comparator:
  `-0.567/-0.314/-0.073/+0.074`; test: `-0.510/-0.310/-0.010/+0.078`;
- descriptive donor-control/advantage Spearman: `0.9487` on both splits;
- ignored shard: 30,844,253 bytes, SHA-256
  `dabdfcf5e2ef751fe9379183e397517f6e36ca92b7c1b0398aa657b23857d8c0`.

Interpretation: no scientific hypothesis is decided. The clean-answer gate did its intended job:
it prevents model formatting/factual failures from being mistaken for causal geometry. The apparent
boundary-relative sign change at layer 26 was observed after rejection and can only motivate a new
behavior-competent preregistration; it cannot rescue H-LLM-10, H-GEO-10, H-GEO-11, or H-CROSS-04.

`LLM-STATE-ONESHOT-LAYER-GEOMETRY-001` key metrics:

- clean execution commit/runtime: `c1daa46`, `235.12` seconds; clean validation/test accuracy
  `1.0/1.0`; all numerical gates passed with maximum p95 derivative error `0.0492`;
- validation donor transfer by layer 18/21/24/26: `0/0.033/0.800/1.0`; test:
  `0/0.233/1.0/1.0`;
- validation population advantage: `-0.575/-0.220/-0.0266/+0.112`, Spearman `1.0`; test:
  `-0.600/-0.203/+0.114/+0.127`, Spearman `0.9487`;
- test control/population boundaries: `24/24`; validation: `24/26`, rejecting H-GEO-12;
- layer-24/26 test population NMSE/agreement: `0.1193/1.0` and `0.02079/1.0`; row-null p05 MSE:
  `2.112/1.742`; row-null p95 agreement: `0.133/0.133`; H-GEO-13 passed;
- H-LLM-12 failed its test early-control ceiling; H-CROSS-05 failed; ignored shard 30,889,987 bytes,
  SHA-256 `d0c83b0fff029990cb4ca7bd7151bd84e62fef0a43caa63608708e914b9974fa`.

Interpretation: population transport is semantically faithful late in the network, but its exact
onset need not coincide with the onset of majority donor control on each entity split. The observed
one-grid-step validation lag is descriptive and cannot replace the registered equality claim. This
is a discriminating negative for exact coupling, not a circuit, workspace, JEPA-meta-model, or SOTA
result.

`LLM-COUNTRY-CODE-LAYER-GEOMETRY-001` key metrics:

- clean retry commit/runtime: `48226c6`, `283.77` seconds; clean validation/test accuracy `1.0/1.0`;
  all numerical gates passed with maximum layer p95 derivative error `0.0635`;
- validation donor transfer at layers 18/21/24/26: `0/.367/.867/1.0`; test:
  `0/.667/.967/1.0`; H-LLM-14 passed;
- validation population advantage: `-.574/+.196/+.392/+.301`, Spearman `.800`; test:
  `-.529/+.229/+.268/+.284`, Spearman `1.0`;
- validation control/population boundaries were `24/21` (lag `-1`); test boundaries were `21/21`;
  the registered direction and negative `A_21` margin failed, so H-GEO-14/H-CROSS-06 failed;
- layer-21/26 test population NMSE/agreement was `.2949/.5667` and `.01470/.9667`, versus row-null
  p05 MSE `1.350/1.679` and p95 agreement `.233/.100`; H-GEO-15 passed;
- ignored shard: 30,887,576 bytes, SHA-256
  `13f3792ab221f2a795d9529010cfef4da6e622cf21ecc70e08e46e0570224235`.

Interpretation: direct donor-answer control and semantically aligned population transport generalize
to a third factual relation, but population advantage is not universally ordered after majority
donor control. This rejects the preregistered bounded-direction account. It does not establish what
causes either transition, a component circuit, a JEPA meta-model, workspace geometry, or SOTA.
