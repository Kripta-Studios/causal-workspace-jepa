# SUMMARY

## 2026-07-21 — EB-JEPA Two Rooms import closure and planner constraint correction

- Installed a separate Python 3.12 Two Rooms closure while preserving Torch 2.10.0+cu128. The
  exact resolved non-Torch environment is committed as a lock, and the installer fails if Torch or
  the pinned clean official revision changes.
- Found an upstream packaging gap: the executed Two Rooms path imports scipy, pandas, and PyYAML,
  but the official project dependencies declare none of them; `ruamel.yaml` does not provide
  `import yaml`.
- An exploratory eight-sample official loop completed dataset generation, Impala/GRU BF16
  forward/backward, AdamW, and checkpoint save on GPU.
- The same exploratory integration returned an MPPI action of norm about `3.69` despite configured
  `max_norms: [2.45]`. Source inspection shows CEM enforces the norm while MPPI does not, and the
  environment passes actions into dynamics without checking its action space.
- A matched deterministic control found exploratory MPPI violations in `32/32` seeds (median
  maximum norm `6.45`, maximum `8.30`) versus CEM `0/32` (maximum `2.35`). These numbers are not yet
  retained evidence: exact gates and the post-discovery status are preregistered for clean runs.
- The first clean integration artifact from `f0e7a3e` is superseded despite passing its original
  gates: it omitted `random.seed`, and the official generator uses Python randomness. The same-seed
  clean/exploratory losses differed (`11.5872` versus `10.0157`). Corrected v2 preserves all model
  settings and adds exact independent-process hashes over data, updated weights, loss, and planner.
- Corrected `WM-EBJEPA-INTEGRATION-002` then ran from clean `9a18008`; all 12 gates passed. Both
  subprocesses produced fingerprint `16650872...234a1`, the BF16 loss was `9.6593`, checkpoint
  restore error was zero, and peak reserved GPU memory was `155,189,248` bytes. This validates only
  the minimal official execution path, not learned prediction or planning competence.
- `WM-EBJEPA-PLANNER-CONSTRAINT-001` then ran from clean `da30443`; all seven frozen gates passed.
  CEM violated `max_norms=2.45` in `0/32` seeds (maximum `2.3474`), while MPPI violated it in
  `32/32` (median `6.4485`, maximum `8.3018`). Official MPPI does not use `max_norms`, and
  `DotWall.step` does not check `action_space` before applying the transition. This confirms an
  upstream planner defect, not failure of the reported trained model; original and corrected MPPI
  must now be compared during competence reproduction.
- Added a separately named constraint-corrected MPPI without touching upstream. It projects every
  sampled candidate before model-cost evaluation and the final returned action.
  `WM-EBJEPA-MPPI-CORRECTION-001` ran from clean `f58308a`; all five gates passed across 32 seeds.
  With bounds disabled, maximum official/corrected action and loss differences were exactly `0.0`.
  With the `2.45` bound enabled, both cost-input and returned-action violations were `0/32`; maxima
  were `2.45000005` and `2.44999909`. This validates the controlled planner arm only, not competence.

## 2026-07-21 — High-resolution action-path calibration completed and closed

- `WM-ACTION-PATH-CALIBRATION-002` completed from clean `288f663` in `19,176.20` seconds. It used
  only exposed validation goals, records `protected_test_goals_touched=false`, and has empty
  hypothesis decisions.
- At horizon four, seeds 101/103/107 had maximum integration errors
  `.01291/.03394/.001393` and maximum 512-to-1024-node changes `.02605/.03141/.000250`.
  Thus the refinement materially improves v1 but remains underresolved for seeds 101 and 103.
- Horizon-four cancellation/local-error Spearman values `.4148/.4783/.8574` exceed their
  within-action-pair null p95 values `.3693/.4500/.8409` by only `.0454/.0283/.0165`.
  Median horizon-four/horizon-one cancellation ratios remain `1.054/2.818/.983`, so recurrent
  amplification is not replicated across seeds.
- These values do not repair the preregistered design boundary: both normalized endpoints share a
  direct-effect denominator, and the artifact lacks scalar path length at both resolutions,
  unclamped norms, dense within-pair support, and a joint conditional null. The route remains
  `CLOSED_DESIGN`; no protected-test run or mechanism claim is authorized.

## 2026-07-21 — EB-JEPA exact-pin GPU incompatibility isolated

- Built two ignored Python 3.12.13 environments that differ at the Torch/CUDA boundary. The exact
  upstream pin uses Torch 2.6.0+cu126; the local compatibility runtime uses Torch 2.10.0+cu128.
- On the same RTX 5070 Ti (compute capability 12.0), Torch 2.6 reports compiled architectures only
  through `sm_90`. Matmul, Conv2D, and GRU all fail with `no kernel image is available`.
- Torch 2.10 includes `sm_120`; the same three kernels execute with finite outputs and no operation
  warning. Thus exact upstream dependency reproduction is CPU-only on this host, while GPU training
  requires a disclosed Torch/CUDA deviation. This is runtime Availability evidence, not a JEPA
  result.
- Primary PyTorch sources agree with the local diagnosis: the 2.6 release shipped at most CUDA 12.6,
  and stable SM120 support begins with 2.7 builds using CUDA 12.8.
- `WM-EBJEPA-RUNTIME-001` ran from clean `15d88ce`; all eight frozen gates passed. The exact runtime
  advertised `sm_50` through `sm_90` and every matched kernel failed with the missing-image error.
  The compatible runtime advertised `sm_70` through `sm_120`; matmul, Conv2D, and GRU all returned
  finite outputs. The committed artifact is runtime Availability evidence only.

## 2026-07-21 — Official EB-JEPA contract and exact recurrent decomposition

- Pinned the official `facebookresearch/eb_jepa` source at immutable commit
  `966e61e9285b3a876f49b9774e9720d9a99a7925`; the local ignored checkout is clean and can be
  recreated with `scripts/prepare_eb_jepa.py`.
- Replaced the published-adapter placeholder boundary with a typed EB-JEPA adapter for the actual
  official object contract: Impala encoder, identity two-dimensional action encoder, and a
  one-layer 512-dimensional `torch.nn.GRU`. This corrects an earlier informal description of the
  predictor as two-layer.
- Implemented an explicit PyTorch-GRU decomposition into reset, update, candidate, pre-normalized
  hidden, and post-normalized hidden sites. Unit tests compare it to native `torch.nn.GRU` and test
  position/feature-specific gate intervention plus downstream recurrence.
- The retained clean `WM-EBJEPA-CONTRACT-001` run from `979c2d6` used the pinned official classes
  and produced latent shape `[1,1,512]`, three-step prediction/gate shapes `[1,3,512]`, maximum
  native-versus-decomposed recurrence error `4.768e-7`, zero same-step collateral error outside the
  edited coordinate, and downstream latent L2 effect `2.1505`.
- Added and retained a configured source-contract smoke and an EB-JEPA-specific doctor. Hardware passes, but the
  current Python 3.14.2/PyTorch 2.10.0 runtime differs from upstream's exact Python 3.12/PyTorch
  2.6 pin and lacks nine declared packages. This blocks an exact upstream training-reproduction
  claim, not the separately tested source contract.
- No trained checkpoint, planning result, causal mechanism, circuit, or workspace result follows
  from this milestone. The next scientific gate remains competent Two Rooms planning across three
  seeds, followed by a separately preregistered necessity/sufficiency audit.

## 2026-07-21 — Primary-source SOTA refresh and next experimental boundary

- Verified the July-2026 causal-interpretability frontier against primary papers and official
  repositories: Jacobian Lens/J-space, AtP*, HVP correction, Circuit Tracing, Qwen-Scope, Natural
  Language Autoencoders, MIB, path patching, faithfulness, Physics Emergence/steering, EB-JEPA,
  Delta-JEPA, C-JEPA, LeWorldModel, JEPA-WMs, Temporal Straightening, AdaJEPA, and WAM steering.
- The bounded search found no published system that trains an intervention-conditioned JEPA to
  predict hidden/logit/behavior changes in Qwen and then validates a compact mediator with direct
  necessity, sufficiency, specificity, and held-out generalization. This is a research gap, not a
  novelty or SOTA claim; the repository's current target-encoder Intervention-JEPA failed 0/3.
- Added EB-JEPA as the highest-priority published world-model target: its official single-GPU
  action-conditioned Two Rooms example reports `97 +/- 2%` planning success and exposes recurrent
  action dynamics suitable for gate/hidden-state mediation. Corrected official-code status for
  LeJEPA, V-JEPA 2.1, Temporal Straightening, and AdaJEPA.
- Added Qwen-Scope as the official sparse-feature comparator and Qwen3-1.7B as a feasible
  intermediate target. Circuit Tracing/NLA are explicit comparators, not evidence that sparse or
  verbalizable features are native causal circuits.
- Adversarial review rejected the first Qwen population-mediation draft. It lacked a defined
  upstream treatment, multi-site intervention support, clustered units, and separation of binding
  from answer copying. The corrected design is module-only, defines sufficiency from clean plus
  treated mediator states and necessity by restoring clean mediators in the treated run, freezes
  `k <= 4` on train, clusters uncertainty by episode/key/value, and calls a positive result a
  mediator set rather than a circuit or JEPA result.
- Implemented ordered multi-site Qwen intervention programs. Tiny-Qwen tests establish exact
  upstream treatment replay by patching the changed layer-0 input position, exact clean recovery by
  restoring a downstream residual state, and explicit order sensitivity for repeated same-site
  operations. Head-level intervention remains deferred until a pre-`o_proj` hook has an exact
  attention reconstruction test.

## 2026-07-21 — Resumable long-run action-path execution

- Added atomic per-horizon progress checkpoints to the recurrent JEPA action-path runner. Progress
  is bound to the exact YAML bytes, experiment ID, and Git commit; stale, duplicate-seed, or
  cross-experiment progress fails closed instead of silently mixing scientific runs.
- A resumed final artifact records whether progress was used and how many seed/horizon blocks were
  loaded. Temporary progress JSON is ignored and removed only after final metrics and provenance
  are written.
- A seed is resumably complete only with an explicit `seed_complete` marker, both horizon blocks,
  and the horizon-4/horizon-1 amplification summary. This closes the crash window between the last
  expensive block and the derived seed summary.
- This hardening applies to future launches. The then-running clean v2 process had imported commit
  `288f663` before the change and therefore wrote only at completion.
- Before the v2 metrics existed, adversarial review rejected a proposed derived denominator audit
  before commit. V2 checks convergence of an integrated vector, not convergence of the scalar path
  length used by cancellation; it stores no 512-node path length and clamps the recorded direct
  norm. With only two chords per action pair, neither separate action-pair/effect-bin permutations
  nor leave-one-pair-out summaries identify denominator-independent geometry. V2 is therefore
  numerical calibration only, protected test remains locked, and this small-model route is closed
  unless a materially new prospective design records both path lengths, raw norms, substantially
  more chords, row-level split guards, and a valid joint conditional null.

## 2026-07-21 — Scientific manuscript and consolidated discovery boundary

- Added `papers/causal_workspace_jepa.tex`, a source-of-truth working paper covering the
  shared finite-intervention formalism, exact/quadratic/population/learned transports, direct Qwen
  behavior patches, the target-encoder Intervention-JEPA null, the small LeWorldModel circuit and
  workspace nulls, decoded recurrent action-path equations, evidence governance, related work,
  limitations, and exact metric/commit provenance.
- The manuscript compiles with MiKTeX/`latexmk` against the repository-local
  `papers/references.bib`. Generated PDFs and auxiliary files are ignored. `papers/README.md` and
  `VPS_RUNBOOK.md` contain the exact build command and warn that this host has an unrelated global
  `references.bib`; compilation must run inside `papers/`.
- Consolidated the strongest defensible findings in `docs/RESULTS.md`: direct capital donor patches
  cause behavior changes; the old nonlinear learned advantage was a BF16 precision artifact;
  vector/logit/behavior endpoints rank transports differently; population averaging helps in one
  bounded relation; registered threshold/grid onset rules fail; one target-encoder JEPA variant
  fails 0/3; gauge-safe dual coupling is a valid control; and no tested JEPA circuit/workspace proxy
  meets its acceptance gate.
- Adversarial manuscript review forbids novelty/SOTA, circuit, workspace, causal-compression, or
  cross-domain-equivalence claims. No positive result reaches circuit-reconstruction level 5 or a
  broad level-6 mechanism. It also identified that cancellation and normalized local error share
  the net-effect denominator; quadrature convergence alone cannot authorize protected testing.

## 2026-07-21 — JEPA action-path calibration implementation

- Stopped the naive protected-test vertex-mean confirmation: validation correlation and the already
  recorded scalar-shrinkage control show that its MSE gain is not sufficiently semantic.
- Literature audit added LIT-042. Local-linear world-action steering and temporal straightening are
  prior art; no source found directly tests decoded path-length cancellation along finite JEPA
  action chords. Absence from this bounded search is not proof of novelty.
- Implemented validation-only `WM-ACTION-PATH-CALIBRATION-001`. It profiles exact action JVPs on
  two contexts per ordered action pair, using 8-point composite Gauss-Legendre at 16/32 panels.
  Metrics include decoded path length/net displacement, cancellation ratio, local finite-effect
  error, direct reconstruction, refinement change, speed concentration, and action-pair-stratified
  permutation nulls across horizons one/four and seeds 101/103/107.
- The calibration ID is hard-locked to the five already exposed validation goals. It emits no
  scientific decision and cannot touch protected test goals. Test hypotheses/thresholds will be
  registered only after calibration, in a separate clean pushed commit.
- The clean run from `eb943a5` completed in `72.59` seconds with `protected_test_goals_touched=false`
  and no hypothesis decisions. Horizon-one maximum direct/refinement errors were below
  `.00077/.00169` across seeds. Horizon four converged for seed 107 (`.00197/.00093`) but not seeds
  101 (`.0680/.1369`) or 103 (`.478/2.059`), so 256 nodes are still insufficient for stiff chords.
- Horizon-four cancellation/local-error Spearman was `.415/.483/.863` for seeds 101/103/107,
  exceeding action-pair-stratified null p95 by only `.045/.027/.022`. Median horizon-four versus
  horizon-one cancellation ratios were `1.05/2.82/.983`; recurrence amplification is not replicated.
  These are calibration diagnostics, not evidence. A 512/1024-node validation-only refinement on
  the same chords was warranted solely to resolve numerical behavior, not to authorize test access.
- `WM-ACTION-PATH-CALIBRATION-002` is implemented as a separate validation-only artifact using the
  identical profile seed/chords and 8-point rules over 64/128 panels (512/1024 nodes). The runner's
  split lock now names both calibration IDs. V2 still emits empty decisions and cannot authorize a
  claim by itself; it tests whether v1's association survives numerical refinement.
- The first clean v2 launch from `e918d4f` wrote no artifact: 1,024-node outer `vmap(jacrev)` with
  chunk size 16 exhausted 12 GB VRAM. The retry changes only resource batching to chunk size 2;
  goals, chords, nodes, metrics, and the decision-free calibration contract are unchanged.
- The chunk-2 retry from `c72d9f5` also wrote no artifact: it reached the 1,024-node stage but
  retained the complete 512-node autograd graph, eventually exhausting VRAM after 788 seconds. The
  next retry streams 64 outer samples at a time, projects exact Jacobians into the registered
  decoded coordinates, detaches them to CPU, and releases each graph immediately. This is a
  tensor-lifetime repair, not a mathematical or scientific change.
- The streamed run from clean `288f663` subsequently completed in `19,176.20` seconds. Its final
  numerical values and the closed scientific boundary are recorded in the newest summary entry.

## 2026-07-21 — Independent country-code bounded-lag preregistration

- The first clean execution from `ab07627` completed target model computation but failed before
  analysis or storage because the older element artifact lacks the later `behavior_eligible` key.
  It wrote no metrics, manifest, shard, or displayed country outcome. A mechanical fallback now
  derives frozen-artifact eligibility from its committed validation/test accuracy; all country
  prompts, splits, thresholds, computations, and decisions remain unchanged for the clean retry.
- Tokenization-only seed 601 selected 36 unique single-token ISO codes whose answers are disjoint
  from all state-abbreviation answers; seed 607 froze 24/6/6 target splits. No target prompt has
  been forwarded.
- Five prompt templates were evaluated only on seven excluded calibration countries and scored
  `0/7`, `5/7`, `2/7`, `5/7`, and `0/7`. The deterministic earliest tied winner is the Canada
  one-shot template. Its weak calibration performance is disclosed; target validation/test clean
  accuracy must each reach `0.90`, with no rescue or entity filtering.
- `LLM-COUNTRY-CODE-LAYER-GEOMETRY-001` implements the post-state bounded-lag question while keeping
  the same Qwen revision, FP32/eager path, layers 18/21/24/26, direct patches, exact Jacobians,
  quadratic/HVP-style corrections, population transport, averaging curves, and answer-row nulls.
- H-GEO-14 requires the population-advantage first crossing to occur at or one registered layer
  after the direct-control first crossing on both held-out splits, with `A_21 <= -0.05`,
  `A_26 >= 0.05`, and Spearman at least `0.70`. H-LLM-14 requires monotone early-to-terminal donor
  control; H-GEO-15 requires semantic row-null specificity. H-CROSS-06 also audits frozen element
  and one-shot-state artifacts under the new rule.
- The zero-or-one-step rule was designed after those two frozen relations. Therefore a country pass
  would be one prospective replication of a posthoc pattern in one small model—not a new method,
  circuit, JEPA result, workspace, cross-model result, or SOTA claim.
- The clean retry from `48226c6` completed in `283.77` seconds. Validation/test accuracy was
  `1.0/1.0`; every numerical gate passed (maximum layer p95 derivative error `0.0635`). H-LLM-14
  passed with donor transfer `0/.367/.867/1.0` on validation and `0/.667/.967/1.0` on test.
- H-GEO-15 passed: test population NMSE/agreement at layers 21/26 was `.2949/.5667` and
  `.01470/.9667`, versus row-null p05 MSE `1.350/1.679` and p95 agreement `.233/.100`.
- H-GEO-14 failed rather than merely missing a margin. Validation population advantage became
  positive at layer 21 (`A_21=.1956`) before the direct-control boundary at 24; test crossed both at
  21 with `A_21=.2289`. The preregistered direction required population to be no earlier and
  `A_21<=-.05`. H-CROSS-06 therefore failed. Across these relations, population usefulness is not
  ordered after a 50% donor-takeover threshold; partial causal control can already suffice.
- The ignored 30,887,576-byte shard has SHA-256
  `13f3792ab221f2a795d9529010cfef4da6e622cf21ecc70e08e46e0570224235`.

## 2026-07-21 — Behavior-competent boundary-relative confirmation result

- Calibrated five prompt formats on only the 13 states excluded by seed 521. Accuracy was
  `12/13`, `3/13`, `13/13`, `8/13`, and `3/13`; the fixed District-of-Columbia example won. No
  one-shot target prompt from the 36-entity roster was forwarded before registration.
- Implemented/preregistered `LLM-STATE-ONESHOT-LAYER-GEOMETRY-001` with a stricter `0.90` clean
  held-out floor. It retains FP32 exact Jacobians, directional quadratic/HVP-style correction,
  24-context population transport, averaging curves, row nulls, and the fixed four-layer grid.
- The new H-GEO-12 does not move a fixed layer: it defines control onset as the first layer with
  donor transfer at least `0.50` and population onset as the first layer where population NMSE beats
  both exact-local and quadratic NMSE. These first-crossing layers must be equal on validation and
  test, with early/terminal margins and Spearman at least `0.80`. H-CROSS-05 requires the frozen
  element boundary equality and sign margins, but not the new Spearman gate: its already-known test
  Spearman is `0.738`. This pre-execution distinction prevents an impossible cross hypothesis.
- V2 reuses state identities/splits but every exact prompt and causal outcome is new. A full pass is
  still one-model cross-relation evidence, not an explained circuit, JEPA meta-model, workspace, or
  SOTA result.
- The unchanged run from clean `c1daa46` completed in `235.12` seconds. Clean validation/test
  accuracy was `1.0/1.0`; all numerical gates passed, with maximum p95 derivative error `0.0492`.
- H-GEO-13 passed: test population NMSE/agreement was `0.1193/1.0` at layer 24 and `0.02079/1.0`
  at layer 26, versus row-null p05 MSE `2.112/1.742` and p95 agreement `0.133/0.133`.
- H-GEO-12 failed one decisive dual-split subgate. Test control/population boundaries both occurred
  at layer 24, but validation control crossed at 24 while population advantage stayed negative
  (`-0.0266`) until layer 26. All early/terminal advantage margins and Spearman (`1.0` validation,
  `0.949` test) passed. Exact onset equality is therefore false for this run.
- H-LLM-12 failed because test layer-21 donor transfer was `0.233`, above the registered `0.10`
  early maximum; terminal transfer was `1.0`. H-CROSS-05 consequently failed. The robust surviving
  observation is late semantic population fidelity, not exact boundary identity.

## 2026-07-21 — State-abbreviation causal-geometry confirmation result

- Primary-source audit added LIT-041, *When Attribution Patching Lies*. Second-order/HVP correction
  is prior art, so the existing directional quadratic Taylor remains a mandatory comparator and no
  novelty is assigned to curvature correction itself.
- Without running the model, tokenization identified 49 U.S. postal abbreviations represented by
  unique single Qwen tokens. Seed 521 selected 36; seed 523 froze a 24/6/6 entity split. The exact
  prompt, four layers inherited from the element study, 2,448 direct patches, 144 full selected-logit
  Jacobians, storage budget, numerical gates, and clean-accuracy floor are preregistered.
- H-GEO-10 uses a post-element but prospective endpoint:
  `A_l = min(local NMSE, quadratic NMSE) - population NMSE`. Both validation and test must show
  `A_21 <= -0.05`, `A_24 >= 0`, and Spearman correlation at least `0.80` between four-layer donor
  control and `A_l`. H-LLM-10 repeats the causal-control transition; H-GEO-11 repeats late answer-row
  specificity; H-CROSS-04 requires all three plus the frozen element positives.
- This study can confirm only a cross-relation association in Qwen3-0.6B. It cannot by itself explain
  the mechanism, identify a circuit/workspace, validate a JEPA meta-model, or establish SOTA.
- The unchanged run from clean `27ebe43` completed in `183.92` seconds. All four numerical gates
  passed, with exact-Jacobian/central p95 relative error at most `0.0354` and exact source replay.
- The preregistered clean-behavior gate rejected the study: train/validation/test full-vocabulary
  accuracy was `0.625/0.667/0.667`, below the `0.75` validation/test floor. Consequently H-LLM-10,
  H-GEO-10, H-GEO-11, and H-CROSS-04 are not scientifically decided; false artifact flags denote
  gate rejection rather than negative hypothesis evidence.
- Descriptive diagnostics are retained without promotion: validation/test donor control by layer was
  `0/0/0.067/0.667` and `0/0/0.400/0.667`; population advantage was
  `-0.567/-0.314/-0.073/+0.074` and `-0.510/-0.310/-0.010/+0.078`. Their Spearman correlation was
  `0.949` on both splits. The sign change follows layer 26 here rather than the fixed layer 24. A
  new prompt/task with prospective competence must confirm that boundary-relative hypothesis.

## 2026-07-21 — Qwen element-layer causal-geometry result

- Rejected the tempting JEPA vertex-dispersion follow-up after post-result controls showed that
  train-calibrated scalar shrinkage beat the vertex mean on seeds 101/103 and simple derivative norm
  explained most of the apparent failure prediction. No protected test goal was touched.
- On four permanently excluded element-symbol calibration facts, donor control was 0% through Qwen
  layer 21 and 100% at layers 24/26. Over the same transition, exact local selected-logit MSE changed
  from about `0.021` to `0.361`, while the four-context mean changed from `0.327` to `0.053`.
  These pilot values are hypothesis-design inputs, not evidence.
- Literature audit found MechLens already establishes late factual crystallization and AtP* already
  establishes nonlinear gradient-attribution failures. The prospective novelty is restricted to a
  direct causal-control/local-predictivity inversion, if it generalizes across disjoint entities.
- Implemented/preregistered `LLM-ELEMENT-LAYER-GEOMETRY-001`: 36 unique single-token symbols,
  fixed 24/6/6 entities, 612 within-split patches at each of layers 18/21/24/26, full `36x1024`
  Jacobians per context/layer, exact local/quadratic/population baselines, central numerical checks,
  context-count curves, 256 row nulls, and full-vocabulary behavior. The preregistration was pushed
  as `5d8de9a` before any registered prompt was executed.
- The unchanged clean run completed in `190.28` seconds and passed every numerical gate. Exact
  Jacobian versus symmetric-central median/p95 relative error ranged from `0.00094/0.00300` at layer
  18 to `0.01093/0.04018` at layer 24; clean replay and donor-source errors were exactly zero.
- H-LLM-08 passed on both entity-disjoint splits: full-vocabulary donor-symbol transfer was `0/0%`
  at layers 18/21 and `60/100%` at 24/26 on validation, versus `0/0/90/100%` on test. This is direct
  causal mediation/generalization for the donor answer, not localization of a feature or circuit.
- H-GEO-09 passed: on test, aligned population normalized MSE was `0.1808/0.01131` at layers 24/26
  versus row-permutation p05 `2.372/1.772`, while candidate agreement was `0.90/1.00` versus null
  p95 `0.167/0.108`. Increasing train contexts reduced layer-24 median MSE from `0.2617` at one
  context to `0.1808` at 24, and layer-26 from `0.0237` to `0.01131`.
- H-GEO-08 failed exactly as registered. At layer 24, population/local MSE ratios were `0.879` on
  validation and `0.707` on test, both above the required `0.60`; on test, layer-21 local/population
  was `0.326`, above the required `0.25`. The local-error jump and correlation-margin subgates did
  pass, but partial conjunctions do not count. H-CROSS-03 is therefore false.
- Bounded conclusion: causal donor control crystallizes sharply in late Qwen3-0.6B layers and the
  late population transport is answer-row aligned, while the preregistered claim that control
  coincides with a strong local-to-population reversal is rejected. MechLens covers late factual
  crystallization and Jacobian Lens covers population averaging; the remaining layer-conditioned
  conjunction needs a new relation/model confirmation before any novelty or SOTA claim.

## 2026-07-21 — JEPA population-geometry preregistration

- Implemented `WM-POPULATION-JACOBIAN-001` as a prospective cross-domain test of the confirmed Qwen
  population-Jacobian observation. It freezes all three `WM-LEWM-001` checkpoint hashes, a seed-401
  12/5/5 goal partition, every valid one-hot first-action swap, horizons one/four, and four fixed
  suffixes. V1 may inspect validation goals only; test goals remain protected.
- The primary comparison uses decoded `x/y/distance` effects from a train-goal-only readout because
  raw latent Euclidean MSE is coordinate dependent. A condition-100 reparameterization must preserve
  decoded effects to `1e-10`, while an analytic control demonstrates raw method-ranking reversal.
  Twelve-node path integration, within-context vertex/centroid controls, an averaging-size curve,
  and all 23 nonidentity action-column permutations distinguish curvature smoothing from label loss.
- The weak inherited planner cannot support behavior evidence unless its direct action choice is
  environment-optimal on at least 60% of contexts. No checkpoint has yet been evaluated under this
  analysis. Commit and push the preregistration before running it unchanged.
- The unchanged run from clean `89b2e14` completed in `55.20` seconds and is
  `REJECTED_NUMERICAL_GATE`. Gauge-safe decoded effects were invariant to at most `2.73e-12`, but
  12-node path-integral errors reached `11.18/49.58/0.0822` at horizon four, so no scientific
  hypothesis is accepted. Every direct planner was below the 60% competence floor.
- Descriptively, global population transport drove MSE toward one but had near-zero correlation and
  failed all action-column semantic nulls. The within-context mean of the four valid action-vertex
  Jacobians beat the local derivative at both horizons in all three seeds; because v1's numerical
  gate failed, this is a post-result candidate for protected-test confirmation, not evidence yet.
- A no-write diagnostic confirmed local autograd against central differences and localized the
  failure to sharp recurrent cancellation: the worst seed-103 chord improved from `49.8` relative
  integration error with 12 nodes to `0.0058` with 192. V2 must use preregistered adaptive
  integration and cannot change v1's rejected status.

## 2026-07-21 — Corrective derivative and novelty audit

- Independent adversarial review found that the `LLM-IJEPA-001` “local Jacobian” is a one-sided
  5-percent secant executed in bfloat16. Its predicted effects are implausibly larger than the true
  effects on the frozen data, so H-LLM-01 was placed `UNDER_REAUDIT`; the old numbers are preserved but
  are not reliable nonlinear-advantage evidence.
- The class named `NeuralInterventionJEPA` is a supervised two-branch bottleneck MLP. It has no
  target encoder, stop-gradient/EMA target, or anti-collapse JEPA objective. New reporting calls it
  the legacy conditional bottleneck, and a genuine target-encoder Intervention-JEPA is a separate
  next milestone.
- Implemented and preregistered `LLM-QWEN-JVP-AUDIT-001`: full FP32 replay of the immutable 432-edit
  grid, exact autograd directional JVP, six symmetric central-difference scales, quadratic Taylor,
  BF16/FP32 drift, semantic deduplication, three refit seeds, and frozen numerical/claim gates.
- A tiny random Qwen3 end-to-end test confirms that the eager-attention exact JVP agrees with a
  symmetric central difference. The real audit must be committed and run from a clean worktree.
- The first clean real-audit attempt stopped before producing any intervention outcome: Transformers
  5.3's tokenizer regex check queried the Hub despite `local_files_only` and encountered the host's
  invalid OAuth token. The adapter now resolves the pinned local snapshot path before loading either
  model or tokenizer, avoiding the network-only compatibility branch. This is an execution fix;
  scientific thresholds and the frozen grid are unchanged.
- The clean rerun from `686368e` completed in `167.99` seconds but was correctly rejected: five of
  six numerical gates passed, while downstream semantic-endpoint max error `1.335e-4` exceeded the
  fixed `1e-5` limit. Exact JVP/central agreement itself was strong (median `2.49e-4`, p95
  `0.00381`). Preliminary raw MSE reversed the old result: quadratic `0.07870`, exact JVP `0.6143`,
  conditional bottleneck `3.1899`, and BF16 secant `120.8994`; zero learned seeds beat exact JVP.
- A no-write diagnostic localized the failed endpoint comparison to float32 cancellation of at most
  `4.768e-7` at replacement-source coordinates. This justifies a separately preregistered direct
  source-semantic validation, but cannot change v1's `REJECTED` numerical-gate result or decide H-LLM-01.
- Implemented/preregistered v2 after disclosing v1. It captures the actual edited source, requires
  exact adapter semantics, and bounds the algebraic direction endpoint by two scale-aware float32
  roundings. All JVP, nonlinearity, predictor, seed, split, and disposition thresholds remain v1's.
- V2 ran unchanged from clean commit `a779ff6` in `170.77` seconds. All numerical gates passed:
  exact direct-source error `0.0`, zero float32 endpoint-bound violations, and JVP/central relative
  error median/p95 `0.000249`/`0.00381`. The scientific result is negative: exact JVP MSE `0.6143`
  and quadratic `0.07870` beat the conditional bottleneck `3.1899`; zero of three seeds passed, and
  the registered finite-nonlinearity gate also failed. Restricted H-LLM-01 is `WITHDRAWN`.
- The corrected conclusion is that these selected Qwen3-0.6B residual edits are mostly local, with
  second-order transport explaining nearly all selected-target effect power. The old BF16 secant's
  MSE `120.8994` was a precision artifact, not evidence of a nonlinear JEPA advantage.
- Updated `AUDIT-COMPLETE-001` ran from clean synchronized commit `3593475`: all 14 criteria pass,
  including the corrected exact-JVP comparator/disposition, 105 tests, Ruff, diff checks, and 28
  audited metric/provenance pairs with 10 locally verified shard checksums.
- Literature review found that generic reachability/observability balancing is established prior
  art (control-theoretic DNN interpretation, empirical minimal realization, and CoBRAS). A future
  contribution must instead test context-conditioned finite-amplitude causal fidelity with direct
  behavior execution and cross-context pooling/permutation controls; no generic balancing novelty
  is claimed.
- Implemented/preregistered `LLM-CAPITAL-PATCH-001` to leave the mostly non-behavioral coordinate
  grid: 36 fixed single-token capitals, 24/6/6 disjoint recipient/donor entity splits, 612 full
  layer-21 residual donor patches, direct top-token behavior, exact JVP, central convergence, and
  quadratic Taylor. Japan/Canada/China/Kenya calibration examples are excluded from final data.
- The unchanged dataset ran from clean commit `95018cb` in `74.45` seconds and passed every
  numerical and behavior gate. Clean answer accuracy was `0.917/0.667/1.000` on train/validation/
  test; donor-answer transfer was `0.580/0.500/0.500`; and 93.6% of patches changed the top token.
  Exact JVP had `0.5296` normalized MSE and only 35.1% candidate agreement, while quadratic Taylor
  had `0.6391` full-vector normalized MSE but 74.3% candidate agreement. This is evidence that
  behavior fidelity and activation-space MSE can rank local approximations differently; it enables,
  but does not itself validate, the genuine target-encoder Intervention-JEPA study.
- Implemented and prospectively registered `LLM-TARGET-IJEPA-001`. A shared online residual encoder
  and intervention encoder predict a 32-dimensional EMA/stop-gradient embedding of the directly
  observed final residual. Variance/covariance losses and effective-rank gates test collapse; only
  afterward does a train-only ridge decoder map predicted plus clean target embeddings to hidden/
  logit effects. Three seeds face raw linear, PCA-bilinear, supervised MLP, legacy bottleneck,
  nearest-neighbor, corpus-average, sparse transport, exact-JVP, and quadratic controls. No real
  model fit may run until this code and its gates are committed/pushed.
- The unchanged study ran from clean `3086cd4` in `28.66` seconds. Zero of three seeds passed
  H-LLM-01B, H-LLM-02, or H-LLM-04. Predicted latent effective rank was `8.64–9.75`, but every EMA
  target latent was below the registered `8` floor (`6.81–7.28`). More decisively, oracle decoders
  given the true target embedding still had held-out normalized MSE `1.065–1.753`, localizing failure
  to entity-specific target geometry rather than only the predictor. The ensemble scored `0.930`
  normalized MSE, `0.483` correlation, and `0.20` answer-candidate agreement.
- Comparator rankings split by endpoint: exact JVP was best on full-vector fidelity (`0.599`
  normalized MSE), raw linear ridge was best on logit fidelity (`0.329`), and quadratic Taylor was
  best on direct answer-set behavior (`0.700` agreement). PCA-bilinear extrapolated catastrophically
  (`2.04e11` normalized MSE). This motivates context-conditioned, endpoint-explicit causal geometry;
  it does not support an Intervention-JEPA advantage or a circuit/workspace claim.
- Implemented/preregistered `LLM-CONTEXT-GEOMETRY-001`. For recipient context `r`, donor-direction
  matrix `D[r]`, and answer-logit Jacobian `J[s]`, it audits the paired contraction
  `K[r,s]=J[s]D[r]^T`. This quantity is invariant when activation vectors and gradient covectors are
  transformed dually, unlike naive separately pooled Euclidean subspaces. The study includes a
  two-context analytic pooling illusion, all 36 real Qwen context Jacobians, 256 fixed test-context
  derangements, a train-pooled Jacobian, direct finite-patch behavior, and numerical reconstruction
  of the already validated exact JVP. No novelty or mechanism claim is made before execution.
- The clean `49d68b7` run finished in `9.66` seconds and passed every numerical gate: full-Jacobian
  reconstruction of stored JVPs had median/p95 relative error `2.86e-7/4.58e-7`. H-GEO-01 and
  H-GEO-02 failed; H-GEO-03 passed. Real top-four pooled/matched/permuted overlap was only
  `0.04036/0.03403/0.03286`, so the preregistered real pooling gap was absent and context pairing
  barely exceeded the permutation null.
- The gauge stress exposed a concrete real-model failure of naive subspace geometry: an invertible
  diagonal coordinate change with condition number `96.4` moved pooled overlap from `0.04036` to
  `0.0003345`, while paired `J D^T` changed by only `1.68e-16` relative. This supports the diagnostic,
  not a semantic mechanism.
- Contrary to the context-specific hypothesis, the train-context mean Jacobian predicted held-out
  finite logit effects better than each recipient's exact local Jacobian: normalized MSE
  `0.358` versus `0.540`, correlation `0.885` versus `0.841`, and candidate agreement `0.500` versus
  `0.300`. A context-derangement null had worse continuous MSE (`0.983` mean) but higher average
  candidate agreement (`0.396`), further showing that discrete behavior and vector fidelity can
  disagree. This post-result pattern requires a held-out confirmation before a new claim.
- Implemented/preregistered `LLM-POPULATION-JACOBIAN-001` as that confirmation. V1's geometry
  decisions used only test entities; v2 is fixed to the six previously unanalyzed validation
  entities. It requires the 24-context mean to improve local NMSE by 20%, correlation by `0.03`,
  candidate agreement by `0.10`, and at least four of six contexts. Separate gates test a monotonic
  1/2/4/8/16/24-context averaging curve and specificity over 256 answer-row permutations. These
  thresholds were chosen after the test discovery and are valid only as validation-split confirmation.
- The clean `3725714` confirmation passed H-GEO-04/05/06. On 30 validation outcomes, the 24-context
  mean versus exact local Jacobian achieved normalized logit MSE `0.354` versus `0.737`, correlation
  `0.866` versus `0.835`, and answer-candidate agreement `0.533` versus `0.300`, with lower MSE in
  exactly 4/6 recipient contexts. Paired-bootstrap raw-MSE improvement CI was `[2.28, 8.28]` with
  `0.9997` positive probability.
- The averaging curve improved median MSE from `0.547` at one context to `0.354` at 24, with fixed
  log-size/MSE correlation `-0.915`; candidate agreement rose `0.433→0.533`. Answer-row permutations
  had p05 MSE `1.413` and p95 agreement `0.033`, so aligned averaging is not merely norm shrinkage.
  Quadratic Taylor still won candidate behavior (`0.833`) while scoring worse continuous MSE
  (`0.774`), preserving the endpoint distinction.
- This confirms population-Jacobian regularization of finite Qwen donor-patch logit effects across
  two disjoint six-entity analyses. Anthropic's Jacobian Lens already establishes corpus averaging,
  so the algorithm is not claimed novel. The bounded new empirical finding is that averaging can
  outperform the exact local derivative for finite behavior-changing replacements; model/task
  replication remains required before calling it a general mechanism or SOTA.

## 2026-07-21 — GPU continuation begins

- Confirmed a clean, synchronized starting point at `99854eb` on `main`/`origin/main`.
- Detected an RTX 5070 Ti Laptop GPU with 12,227 MiB VRAM, CUDA-enabled PyTorch 2.10, Transformers
  5.3, 32 logical CPU cores, and roughly 370 GB free. The `gpu_12gb` doctor passes.
- The inherited full suite exposed a Windows portability failure: provenance paths recorded with
  `/` were compared to Windows `Path` strings with `\\`. The audit now compares normalized relative
  paths.
- Fresh clones intentionally lack ignored activation shards. The integration test now requires at
  least six manifest records to be either verified locally or explicitly reported missing; it does
  not mislabel missing bytes as verified.
- After the repair and GPU-doctor coverage, all 44 tests pass and the reproducibility audit reports nine metric/provenance
  pairs, zero errors, and seven skipped missing local shards.
- Regenerated the inherited 41-byte `uv.lock` and corrected the GPT-2 dependency group to include
  PyTorch and a bounded Transformers major version.
- At the continuation start, independent Qwen, world-model, and adversarial audits found that real
  Qwen and published-JEPA integrations were placeholders. The later entries below record their
  implementation, execution, and negative controls.
- Replaced the Qwen failure stub with a selected-site Torch/Hugging Face adapter. Offline tiny-Qwen3
  tests cover residual/attention/MLP/logit capture, clean replay equality, zero/mean/resample/patch/
  steer semantics, explicit donor/statistic requirements, and graph-preserving autograd.
- Pinned and preregistered `LLM-QWEN-001` at public Qwen3-0.6B revision `c1899de...` with a
  1,519,209,243-byte repository estimate. Do not run until this code milestone is committed/pushed.
- The first execution passed numerically but was rejected because the new runner collected
  provenance after writing an untracked metrics file. Fixed the ordering and pushed `0d6a37b`, then
  reran unchanged from clean code.
- Clean `LLM-QWEN-001` passes: exact deterministic replay; selected-logit autograd norm `0.944`;
  nonzero zero/mean/resample/patch/steer downstream hidden/logit effects. This meets real Qwen HF
  instrumentation only; intervention dataset, meta-model, and circuit verification remain.
- Implemented and preregistered `LLM-INTDATA-001`: 432 real direct outcomes, fixed 8/2/2 prompt
  splits, split-local donors, three layers, five operations, a direct small-step local baseline,
  and resumable SHA-256 HDF5 shards under a 128 MB cap. Commit before running.
- Ran `LLM-INTDATA-001` from clean commit `0aa80ac` in `33.85` seconds. All 432 effects were nonzero,
  17 changed the top token, and the 412,332-byte HDF5 shard checksum verifies. Prompt-local 5-percent
  linear MSE was `139.83`; this is a difficult regime, not yet a meta-model result.
- Implemented/preregistered `LLM-IJEPA-001`: three neural seeds, layer-transition and trajectory
  interfaces, exact checkpoint replay, nine baselines, prompt/feature/operation holdouts, and direct
  execution of every coordinate prediction on four new prompts.
- Ran `LLM-IJEPA-001` unchanged from clean commit `a54f2ed` in `8.23` seconds. All three seeds passed
  the registered H-LLM-01/02/03 decisions. Primary MSE/correlation were `3.923`/`0.677`; the model
  beat no-change, mean, linear, bilinear, MLP, local/corpus Jacobian, sparse-linear, and nearest-neighbor
  baselines on the primary holdout. Nearest-neighbor slightly beat it on resampling-holdout MSE
  (`2.095` versus `2.141`), an important boundary outside the registered gate.
- Directly re-executed all 16 ranked-coordinate edits on four unused prompts. Effect-size correlation
  was `0.673`, but the predicted winner (coordinate 128) was not the observed winner (coordinate 0),
  precision@1 was zero, and the effect was slightly below the random coordinate control. H-LLM-06
  failed and `qwen_meta_circuit` is recorded as `REJECTED`; no Qwen circuit or workspace is claimed.
- Verified the official LeWorldModel paper/code, MIT license, and source revision `8edfeb3...`.
  Replaced the placeholder with a source-informed small reproduction of selected design elements,
  retaining end-to-end pixels, action
  embedding, AdaLN-zero autoregression, next-embedding MSE, and SIGReg; added PixelTinyMaze, a typed
  adapter, layerwise probes, paired-action interventions, norm-matched planning controls, ensemble
  uncertainty, five consumers, and restricted circuit graph auditing.
- Short reduced engineering checks found and fixed a collapse-selecting validation bug: selecting by
  next-embedding error alone preferred a collapsed encoder despite SIGReg. The registered runner now
  uses the fixed final training step, as the official training recipe does. These checks are not
  scientific evidence. `WM-LEWM-001` is preregistered and must be committed before its full run.
- The first clean full attempt from `9c3239a` completed computation but failed before emitting
  metrics because `ResourceReport` was not JSON-serialized. Regenerable graphs were discarded; the
  one-line serialization repair was pushed as `4dbc388`, then the unchanged experiment was rerun.
- `WM-LEWM-001` completed from clean `4dbc388` in `54.50` seconds and failed its replicated causal
  gates. All three models passed the faithful-reproduction gates. Planner intervention gates passed
  seeds 103/107, but the full donor-decoding specificity/circuit gate passed only seed 107. The
  aggregate action-to-planner graph is `REJECTED`, not a circuit claim. All workspace candidates
  failed, while planted shared/disjoint controls behaved correctly. The intervention reduced rather
  than raised ensemble variance (`0.506` to `0.250`), a possible overconfidence failure mode.
- Replaced the obsolete Qwen capture blocker with a generic selected-site Hugging Face capture
  pipeline. The primary Qwen3-4B target is pinned to revision `1cfa9a7...` (8,060,926,626 repository
  bytes), captures only five layers and three semantic position selectors from 12 fixed prompts,
  and enforces a 64 MB checksummed HDF5 budget.
- Ran that capture from clean commit `55087ea`. The exact revision resolved, bfloat16 inference fit
  the RTX 5070 Ti, and all registered gates passed: 180 rows, 947,298 estimated uncompressed bytes,
  one 574,308-byte shard, SHA-256
  `cd1ef5e3a871740bfbd06e45c1a024257ba2ed9c1f0b1f6ac0bc2db1a11240cf`, runtime `473.14`
  seconds including the first download. This is Availability evidence only.
- The earlier `AUDIT-COMPLETE-001` run from clean synchronized commit `42492dc` passed all 14
  explicit bounded completion criteria with 63 tests, multi-seed results, direct Qwen execution,
  workspace controls, and rejected circuit graphs. It was superseded by the corrected 68-test audit
  above after the exact-JVP result. The required bounded suite is complete; this does not make either
  rejected circuit or the workspace null positive.

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
- Added Milestone 1 tests. At that milestone the full command passed 14 tests; before the current
  `WM-T0-004` code milestone the full suite had 34 tests.
- Committed and pushed Milestone 1 code as `0cab19a6c39c98b59f1a2172eb11a64ec5a566a4`.
- Reran Tier 0 generation and `WM-T0-001` tiny JEPA smoke from clean committed code.
- Committed and pushed result summaries as `b0b7796`: `data/manifests/tier0_smoke_manifest.json`, `artifacts/metrics/tiny_jepa_smoke.json`, and `artifacts/metrics/tiny_jepa_smoke.provenance.json`.

### Discovered

- `README.md` initially only contained the project title.
- `pytest` is not installed, but NumPy is available system-wide.
- Historical decision: GPT-2 Medium was initially treated as `BLOCKED_RESOURCE`. A later explicit
  user override and the final `AGENTS.md` CPU guard permitted the cached model under strict limits.
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
- Historical audit reported `/root/.cache/pip` at about 4.4 GB and left it untouched. The later
  resource re-audit measured all of `/root/.cache` at about 233 MB, so that old size is no longer
  current.
- Historical `WM-T0-002` run reported action recovery `0.984`; adversarial review later found that
  result tautological and superseded it with the clean replay result documented below.
- Under explicit user override, created `.venv`, installed `transformers`/`safetensors`, downloaded `gpt2-medium`, and ran `LLM-GPT2-001`.
- GPT-2 Medium result: direct residual steering at `transformer.h.12.resid_post` changed logits with mean absolute delta `0.0797`; tiny intervention-JEPA MSE `0.00220` beat no-change `0.0114`.
- Historical provenance: the superseded `WM-T0-002` ran on `e5db938`; `LLM-GPT2-001` remains valid
  on `59795a4`.
- Committed and pushed GPT-2 metrics/docs as `a7ea6da`.

### Next Ideas

- On `gpu_12gb`, test GPT-2/Qwen activation replacement, resampling, patching, or sparse-feature
  clamps while retaining prompt-local Jacobians and held-out prompts as controls.

### Adversarial Milestone 3 Re-audit

- Re-read `AGENTS.md` and `VPS_RUNBOOK.md` completely and reran the baseline suite: 25 tests passed;
  doctor, reproducibility audit, and bytecode compilation passed.
- Found a material flaw in `WM-T0-002`: `action_patch` was assigned to the donor target rather than
  produced by an internal intervention. The `0.984` recovery is preserved but no longer interpreted
  as Specificity evidence.
- Implemented actual replay through `InterventionSpec` at `predictor.input`, nonzero donor sampling,
  and L2 norm-matched latent/action controls.
- Implemented a normalized multi-consumer Jacobian detector, compactness criterion, direct projection
  ablations, random and PCA controls, and known shared/disjoint detector controls.
- Added a preregistered `WM-T0-003` five-consumer study. It can identify a shared causal state
  subspace candidate, but it cannot declare a full workspace because key functional criteria remain
  untested.
- Reran corrected `WM-T0-002` from clean commit `315d8cf`. Actual `InterventionSpec` replay recovered
  the donor effect exactly (`1.0`, max error `0.0`) while L2 norm-matched controls did not. This is
  narrow Specificity evidence for the explicit action-input pathway, not a learned workspace.
- Ran `WM-T0-003` from clean commit `5223a54`. The detector passed planted shared/disjoint controls
  and proposed a `3/16` sensitivity subspace, but the preregistered candidate decision failed:
  uncertainty R2 was `-3.639`, PCA damage exceeded candidate damage, and random rollout controls were
  badly off-manifold. No workspace was found.
- This led to the separately preregistered `WM-T0-004` deeper predictor, calibrated ensemble, and
  in-manifold donor study recorded below.

### Current Handoff

- Worktree was clean at pushed commit `93431c3` before starting the next code milestone.
- After the `WM-T0-004` null, implemented and preregistered `WM-T0-005`: four goals crossed with
  two mass modes, one held-out composition, three seeds, a three-model calibrated ensemble, five
  frozen consumers, random and local-tangent controls, direct task-context counterfactual swaps,
  and multistep selective-necessity tests.
- The joint candidate gate requires valid action dependence, calibrated OOD uncertainty, all five
  held-out consumer heads, compact shared sensitivity, at least 50 percent donor recovery,
  specificity over both matched control families, and multistep selectivity on two of three seeds.
  The full workspace decision remains false by construction because reportability and published
  model replication are absent.
- The preregistered implementation was committed and pushed as `7a9e510`, then executed once from
  that clean commit. Provenance reports `git_dirty: false`; runtime was `57.51` seconds and disk
  remained near 40 GB free.
- `WM-T0-005` is a replicated null: zero of three seeds passed. Action MSE ratios were `0.712`,
  `1.012`, and `1.003` versus the registered maximum `0.50`. OOD uncertainty passed jointly only on
  seed 29. Every seed failed held-out transfer for the five-consumer set.
- Candidate dimensions saturated at `6/24`, while minimum consumer sensitivity capture was only
  `0.583`, `0.561`, and `0.574` versus `0.70`. Mean goal/dynamics counterfactual recovery was
  `-10.18`, `-17.38`, and `-3.11` versus the required `+0.50`.
- Seed 37 exceeded random-control p95 for counterfactual and rollout effects despite moving away
  from the donor. The absolute recovery gate and local-tangent controls correctly rejected this
  misleading relative win. No shared task-workspace candidate and no workspace were found.
- Practical conclusion: appending task context and fitting consumers afterward is not enough. A
  new JEPA architecture should train value/risk/action consumers or a planner jointly; do not tune
  the observed `WM-T0-005` seeds or thresholds.
- Implemented and preregistered `LLM-GPT2-003` without running it: two orthogonalized residual
  contrast directions from eight disjoint calibration prompts, six evaluation prompts, magnitude
  `0.5` finite differences, magnitude `6.0` singles/compositions, and exactly 72 direct outcomes.
- Predictors train only on 32 singles from four prompts. The primary test is eight compositions from
  two unseen prompts. Required controls are prompt-local and corpus additive Jacobians plus direct
  addition of the matching large single effects. Direction labels describe construction, not proven
  semantics. The full pre-commit suite passed 43 tests; doctor, reproducibility audit, compilation,
  and `git diff --check` passed. The implementation was committed and pushed as `1e57e30`.
- Executed `LLM-GPT2-003` once from clean commit `1e57e30`; provenance is clean, runtime was
  `392.85` seconds, estimated storage was 350,156 bytes under a 16 MB cap, and the ignored shard is
  24,933 bytes with SHA-256
  `adb4751bd3c9ca3c26139c47dac0423fd82b43515ccf9c56b5b706a78782f631`.
- Held-out compositions were almost exactly additive: direct interaction was `0.000429` of effect
  power. Prompt-local large-single addition MSE was `0.000179`; prompt-local finite-difference MSE
  was `0.000990`, correlation `0.9989`.
- Every learned held-out-prompt predictor failed: MLP MSE `0.725`, linear `0.775`, bilinear `1.346`,
  versus no-change `0.418`; learned correlations were negative. All registered H-LLM-01/02/03
  decisions are false.
- Bilinear prediction looked excellent on seen-prompt compositions (`0.00110` MSE, `0.9985`
  correlation), proving that a non-held-out evaluation would have yielded a false positive. Two of
  72 direct edits changed the top token, including one of 24 compositions, but direction semantics
  and mechanism reuse remain unvalidated.
- Practical conclusion: on this GPT-2 site, local Jacobians are useful but corpus/learned transport
  is not reusable across prompts. The next discriminating intervention should replace/resample
  activations or clamp learned sparse features, not add more steering directions to this grid.
- Post-run audit: 43 tests pass; nine metric/provenance pairs and seven local checksums validate;
  doctor reports 39.2 GB free; compilation and `git diff --check` pass. Commit and push the result
  milestone, then verify `main` is clean and synchronized with `origin/main`.
- Audited every repository Markdown file and all committed provenance logs. Corrected stale
  statements during the GPT-2 code milestone: the original action patch is superseded,
  `WM-T0-003` is no longer pending, and cached GPT-2 is allowed while Qwen remains blocked.
- Implemented `LLM-GPT2-002`: 288 batched direct interventions, local-only model loading, storage
  budget/checksum manifest, prompt/magnitude/layer holdouts, linear and bilinear predictors, trained
  MLP, nearest-neighbor and sparse-context baselines, and prompt-local/corpus Jacobians.
- Preregistered all splits and thresholds before execution. Do not change them after seeing results.
- At the preregistration checkpoint, 34 tests passed before code commit/push and clean execution.
  The measured result is recorded below.

### LLM-GPT2-002 Result

- Ran from clean commit `8fbab8c`; provenance reports `git_dirty: false` and no model download.
- Generated 288 direct residual-intervention outcomes and an ignored 87,378-byte float16 shard.
  Committed manifest checksum:
  `06a3a75c91076422d73fd62a85694c8366c4db854b2c303e203256adffe73abf`.
- Primary unseen-prompt/magnitude result: local Jacobian MSE `7.79e-7`, bilinear
  Intervention-JEPA `0.003499`, linear `0.006360`, no-change `0.006461`, MLP `0.007361`.
- H-LLM-01 nonlinear advantage failed. The narrow H-LLM-02 compression test passed because
  bilinear beat linear/no-change, but local direct linearization was about 4,490 times lower MSE.
- Held-out layer 18: bilinear MSE `0.01009` was worse than no-change `0.006228`; cross-layer transfer
  failed. Local Jacobian remained nearly exact (`3.71e-8`).
- No intervention changed the top token. This is hidden/logit causal mediation only, not behavior,
  semantic feature, J-space, workspace, Qwen, or consciousness evidence.
- Runtime was `647.85` seconds, exceeding the 10-minute CPU expectation by `47.85` seconds; record
  this as a resource-limit miss.
- Metrics, clean provenance, checksum manifest, and synchronized result docs were committed and
  pushed as `fdf6506`.
- The earlier next step was semantic/composed steering; `LLM-GPT2-003` has now tested and rejected
  that additive regime. On `gpu_12gb`, use activation replacement/resampling, sparse-feature clamps,
  and enough prompts/seeds to expose genuine nonlinearity. Keep prompt-local Jacobians as baseline.
- Strengthened `scripts/audit_reproducibility.py`: it now validates every metrics/provenance pair,
  clean commit metadata, JSON structure, and all available local dataset checksums instead of only
  checking that seven control-plane paths exist.
- The strengthened reproducibility audit and tests were committed and pushed as `3eedcb5`.
- Final handoff target: full suite, doctor, checksum/provenance audit, and `git diff --check` must
  pass; branch `main` must be clean and synchronized with `origin/main`.

### WM-T0-004 Implementation And Result

- Re-audited every Markdown handoff/registry and the committed metrics/provenance logs. Corrected the
  stale test count and the stale instruction to run the already-completed `LLM-GPT2-002`.
- Primary-source verification confirms that Anthropic's J-space is a sparse nonnegative token-aligned
  frame tested for report, directed modulation, internal reasoning, flexible reuse, and selectivity.
  A JEPA Jacobian eigensubspace is therefore only an analogue candidate, never equivalent by name.
- Added a NumPy-only two-hidden-layer action-conditioned JEPA whose hidden weights are learned and
  exposed at `predictor.hidden1` and `predictor.hidden2`; its fixed encoder prevents JEPA collapse.
- Added a five-member bootstrap ensemble, split interval calibration, held-out coverage/NLL, OOD rank
  AUC, and uncertainty/error correlation. No uncertainty claim is accepted unless all registered
  thresholds pass.
- Added conditional donor resampling: candidate coordinates come from validation activations matched
  in the orthogonal complement. Random/PCA controls must match candidate density distance and
  perturbation magnitude before they count.
- Preregistered `WM-T0-004` before execution. It fits all five consumers after freezing the predictor,
  discovers candidates separately at two depths, and requires direct plus multistep specificity.
- Resource incident: free disk fell to 3.8 GB because broken external `AppTurnos` launchers produced
  about 37 GB of PM2 restart-loop logs. Per user instruction, the systemd service is now inactive and
  masked, its PM2 entry and saved startup record are removed, port 3001 is closed, and 40+ GB is free.
  The source/database remain as a recoverable archive outside this repository.
- Committed and pushed the preregistered implementation as `6785fb1`, then executed once from that
  clean commit without changing thresholds. Runtime was `31.22` seconds and provenance is clean.
- The action-conditioning gate failed: primary MSE was `0.003416` versus shuffled-action `0.004994`,
  ratio `0.684` against the registered maximum `0.25`. Shuffling hurts, but not enough for the strong
  causal-use claim.
- In-distribution uncertainty calibration worked: test coverage was `0.887` for a 90 percent target,
  and uncertainty/error Spearman was `0.698`. The joint uncertainty claim failed because OOD AUC was
  only `0.574` versus `0.65` and hidden uncertainty-head R2 was `-1.327`/`0.232` versus `0.50`.
- Neither hidden site produced the registered compact sensitivity candidate: minimum five-consumer
  capture was `0.635` and `0.701`, below `0.75`.
- The old off-manifold confound is repaired. Candidate density ratios were `1.029`/`1.018`, and
  `63/64` plus `64/64` random controls matched density and magnitude. Candidate direct damage was
  below random-control p95 at both sites (`3.652 < 9.252`; `1.239 < 4.369`).
- Multistep amplification ratios (`6.03`, `1.95`) looked large in isolation but candidate rollout
  damage also remained below matched-control p95. This is nonspecific accumulation, not selective
  necessity. PCA controls were too large to be matched and correctly did not decide the claim.
- Scientific result: no shared causal subspace and no JEPA workspace. Restricted H-WM-05, H-WM-06,
  and H-WM-08 are false for this run. The reusable output is the conditional donor control method.
- ELI5 novelty boundary: this repository now has a fairer way to ask whether many JEPA outputs share
  a special internal whiteboard without smashing the model's activations into nonsense. The answer
  for this tiny model is no. This is methodologically useful and falsification-first, but it is not
  state-of-the-art performance, a published discovery, or evidence that larger JEPAs lack such a
  mechanism.
- Best next JEPA direction: preregister a goal-conditioned multi-task model where action and task
  variables are indispensable, use local-tangent PCA controls, and replicate across seeds. Do not
  tune `WM-T0-004` after seeing its null.
- Final pre-commit validation: 37 tests pass; seven metric/provenance pairs and six local checksums
  audit without errors; compilation and `git diff --check` pass; the CPU guard reports 40.1 GB free.
- Exact clean-code reproduction command:
  `PYTHONPATH=src ./.venv/bin/python scripts/run_experiment.py --config configs/experiments/manifold_workspace_study.yaml`.
