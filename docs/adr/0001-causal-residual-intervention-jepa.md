# ADR 0001: Causal-Residual Intervention JEPA

- Status: `ACCEPTED_AFTER_SOL_REVIEW`
- Date: 2026-08-02
- Scope: CR-LLM-IJEPA and CR-AC-JEPA
- Decision owner: primary Sol architect
- Architecture review: fresh Sol review on 2026-08-02; first verdict `change`, revised verdict
  `ship` for the architecture boundary. Protected execution remains unauthorized.

## Context

The repository has already falsified a favorable complete-effect learned comparison: on the fixed
Qwen grid, exact FP32 JVP and quadratic transport beat the legacy learned bottleneck. A later
EMA/stop-gradient target encoder over complete treated-state geometry also failed all registered
held-out gates. In Track A, the source-informed small LeWorldModel passes prediction gates but its
planner, circuit, and workspace candidates fail; official EB-JEPA mechanism work remains ineligible
until the frozen three-seed competence portfolio passes.

The new framework therefore must not make complete treated-state prediction its primary learned
hypothesis, must preserve earlier negative results, and must stop rather than train when an eligible
analytical baseline already explains the finite effect.

## Decision

For a clean source state `x`, intervention or environment action `u`, and directly executed treated
state `x_u`, define

```text
delta_u = x_u - x
baseline_star = select_train_validation_only(eligible_baselines)
residual_u = delta_u - baseline_star(x, u)
predicted_delta_u = baseline_star(x, u) + predicted_residual_u
x_hat_u = x + predicted_delta_u
```

The learned hypothesis predicts `residual_u`. Complete-state prediction is retained only as a
parameter-matched registered control. `baseline_star` is selected only from **deployable
residualizers** whose inputs, source-model executions, and inference cost are available to the
learned model under its declared deployment contract. Fair but non-residualizing comparators remain
in the leaderboard, and methods that consume exact treated prefixes or other unavailable executions
are labelled **oracle ceilings** and cannot define the target. Before residual-model fitting, one
machine-readable selection record freezes and hashes the eligible set, deployment inputs, endpoint
metric, aggregation rule, tie-breaking rule, fitted baseline artifacts, normalization,
dimensionality reduction, split identities, and selected `baseline_star`.

Target encoding, nuisance removal, routing configuration, and stopping are also fit on
train/validation groups only. Protected rows are inaccessible until a self-hashed plan is committed
and the relevant authorization flag is true.

### Shared core

Both tracks use one typed causal-effect schema, split policy, provenance contract, and metric
vocabulary. They do not share source semantics or scientific decisions.

- Deployable residualizers: no change, mean effect, exact local JVP, directional HVP/quadratic
  Taylor, population differential transport, PCA/ridge causal-delta transport, train-fit affine or
  equivariant transports, and domain-specific transports only when their complete declared inputs
  are available at inference.
- Fair comparators: every eligible residualizer plus parameter/compute/information-matched
  bilinear, MLP, sparse-student, complete-delta, and complete-state controls. Predictor tuning and
  data budgets are matched and fixed without protected outcomes.
- Oracle ceilings: observed-prefix relinearization and any method that re-executes an exact treated
  prefix unavailable to the deployed learned predictor. Oracle results quantify headroom but never
  select `baseline_star`, define a residual target, or count as an inherited model input.
- Target: residual endpoints or residual trajectories only. The target encoder cannot receive IDs,
  answer labels, protected metadata, treated states as online input, or features unavailable at
  inference.
- Predictor candidates: no-routing local model, parameter-matched MLP, standard cross-attention,
  and conditional low-rank routing. Low-rank keys route across layer/site/object/time coordinates;
  values retain source dimensionality. Routing keys and transported values have independent
  ablations.
- Heads: residual state, endpoint/logit or task effect, replay behavior, inverse/composition, and
  uncertainty.
- Losses: stop-gradient/EMA JEPA alignment, normalized residual reconstruction, direct replay
  endpoint/KL, identity, inverse restoration, sequential composition, commutator interaction,
  matched-control specificity, uncertainty calibration, nuisance/anti-shortcut, optional
  within-context or within-trajectory NCE, and optional residual VISReg.
- Decision: contrastive, VISReg, low-rank routing, and adaptive sampling are candidates, not default
  winners. They are retained only if validation selects them and protected direct replay/control
  beats the strongest eligible analytical **and learned** comparator, including capacity-matched
  complete-delta/state, affine/equivariant, MLP/bilinear, and sparse-student controls.

### CR-LLM-IJEPA

The primary prospective source is the already preregistered Qwen3-0.6B binding-algebra family. Its
existing competence, `0.10` residual/nonlinearity gate, three seeds, `0.80` error ratio, `0.10`
behavior margin, `0.50` direct-patch recovery, `0.20` control margin, and `0.75` inverse-restoration
thresholds remain frozen. The CR residual architecture is a new preregistration amendment,
`QWEN-BINDING-ALGEBRA-CR-V1`, with its own canonical digest and `execution_authorized=false`; it does
not edit, relabel, or silently inherit authorization from binding-algebra v2.

1. Gate B0 reproduces clean/direct-treatment competence and exact layer-0 replay by prompt/entity
   group. Failure is `INELIGIBLE_TASK`.
2. Gate B1 compares exact JVP, full quadratic/HVP, population, registered recurrent-path transports,
   learned controls, and the separately labelled observed-prefix oracle. If residual/interaction
   power after the strongest deployable residualizer is below `0.10`, stop
   `COMPLETED_NEGATIVE_LOCALLY_DIFFERENTIAL` without training. The oracle is reported but cannot
   define this gate's residual target.
3. Gate B2 fits residual target encoders on training groups and requires the exact target embedding
   oracle to preserve direct replay and candidate behavior beyond train-only causal-delta
   PCA/ridge. Donor, prompt, position, and token-ID nuisance dominance rejects the target.
4. Gate B3 trains three frozen seeds only after B0--B2. It holds out donor entities, recipient
   entities, prompt forms, compositions, and required intervention families.
5. Direct replay first establishes observed-target replay, then constructs `x_hat_u`, performs a
   complete replacement at every registered site, and executes the suffix under the same pinned
   model revision, runtime, precision, tokenizer, RNG state, and decoding contract as the exact
   intervention. Activation, role-logit, candidate behavior, inverse/composition, calibration, and
   matched-control metrics are separate endpoints. Recovery ratios use preregistered positive
   ratio-of-sums denominators. Missing sites, partial-state replacement, non-finite denominators, or
   failed observed-target replay fail closed; they cannot be reported as zero-effect success.
6. SAE, Qwen-Scope, transcoder, AtP*, and Jacobian rankings remain interpretability comparators.
   Names or router weights are hypotheses until direct ablation and matched controls pass.

The current `execution_authorized=false` remains authoritative. This milestone may implement and
dry-run the complete outcome-blind stack, but it may not tokenize or execute the protected Qwen
study until a later clean, pushed authorization milestone.

### CR-AC-JEPA

Track A separates source/planner competence from representation experiments.

1. Gate A0 completes the frozen official EB-JEPA seeds and competence matrix. Mechanistic claims
   remain locked unless the bound-corrected primary arm reaches `0.80` overall and `0.70` per seed.
2. Deterministic Stage 0 defines residuals in observable ground-truth physical state or under a
   frozen encoder. EMA targets remain an ablation, not a guarantee against co-adaptation; they are
   anchored by frozen replay/decoding, collapse, drift, residual-power, and repeatability gates. A
   frozen baseline defines every protected residual target, target drift is reported, and a
   preregistered finite-residual/repeatability floor must pass before residual learning begins.
3. The small architecture factorial compares pooled versus dense patch/object tokens, vector versus
   supported masked-visual actions, forward-only versus inverse/action decoding, next-latent versus
   delta/residual targets, counterfactual masking, and planted nuisances. Dense tokens use
   block-causal attention.
4. Same-trajectory NCE uses eligible transitions from the same episode; false-negative handling
   excludes near-time states and/or admits multiple dynamically equivalent positives. True actions
   are compared with shuffled, sign-flipped, norm-matched, and counterfactual actions.
5. The INTACT candidate uses one shared conditional action-law operator with the four-slot grammar
   `[z_t, m_t, z_t*m_t, action_embedding(a_{t-1})]`. Local intent keeps the real successor attached;
   deployable goal intent uses `stop_gradient(z_goal)-z_t`. It optimizes separate proper action
   likelihoods and never imposes pointwise equality between local and goal intents.
6. Direct control recurrently predicts action chunks without candidate sequences or terminal
   latent-cost calls. Its deployment mask exposes only the current observation/latent, frozen goal
   anchor, previous executed-action context, and registered model state; future successors,
   trajectories, rewards, success flags, absolute/relative time, episode length, and policy identity
   are unavailable. Broad CEM/MPPI stays actor-disabled; an optional verifier may search only
   locally around the Direct proposal. Gaussian and mixture action heads are compared where
   multimodality exists.
7. Evaluation separates action-family kNN/CKA/R2/rank/NLL from actor-disabled planning, Direct
   closed-loop success, optional verified success, latent drift, and paired actor/coordinate
   compatibility interventions. Required controls include actor-only and representation-shuffle
   baselines, future/time/trajectory/policy nuisance probes, residual-path ablation and patching, and
   action-shuffled/counterfactual executions. Family metrics and actor imitation cannot substitute
   for closed-loop success or evidence that the CR-JEPA path mediates it.

The current GPU has about 6.6 GB free while the frozen EB-JEPA batch reserves 5.82 GB before safety
headroom. Full EB-JEPA continuation is therefore presently `EXECUTION_BLOCKED_RESOURCE`; Stage 0,
implementation, dry runs, and resource-independent tests remain eligible.

## Alternatives and literature disposition

| Source or alternative | Disposition |
| --- | --- |
| Exact local JVP | Mandatory baseline; never called a learned causal model. |
| Directional HVP / second-order Taylor | Mandatory baseline and B1 gate. |
| Prefix-relinearized transport | Mandatory oracle ceiling when it consumes directly executed treated prefixes; a deployable variant is eligible only if it uses exactly the same declared source executions, information, and cost available to the learned predictor. |
| Population Jacobian transport | Mandatory train-only baseline; existing bounded Qwen result is preserved. |
| PCA/ridge causal-delta transport | Mandatory target-sufficiency and prediction baseline. |
| Existing target-encoder Intervention-JEPA | Retained frozen negative and parameter-matched complete-state control; not modified or rescued. |
| EB-JEPA and LeWorldModel | Source/planner competence baselines before Track-A interpretation. |
| AtP*, attribution patching, HVP rankings | Localization comparators only; direct patching decides mediation. |
| SAE, Qwen-Scope, transcoders | Sparse representation comparators only; feature labels are hypotheses. |
| arXiv:2607.09694, Low-Rank Attention Residuals | Adopt decoupled low-dimensional routing keys and full-dimensional values as an ablated candidate, not mechanism evidence. |
| arXiv:2211.10831, JEPA slow features | Adopt persistent/slow nuisance planting and same-episode controls; variance regularization alone is insufficient. |
| arXiv:2607.18236, Patch Policy | Adopt dense patch tokens with block-causal attention as a Track-A candidate. |
| arXiv:2605.02735, Silenced Visual Latents | Adopt actor-disabled and path-ablation controls for bypass/silencing; reject its inference-time reward optimization as outside the reward-free primary design. |
| arXiv:2602.14338, AERO | Retain adaptive allocation as a train-only sampler ablation; reject importing GRPO/RL into this program. |
| arXiv:2607.24653 | Resolves to Kimi K3, not INTACT; no feature is adopted from this identifier. |
| arXiv:2607.26056, INTACT | Adopt the shared local/goal four-slot grammar and Direct candidate as a controlled Track-A ablation; do not assume its reported success transfers. |
| “Obsessed Encoder” article | Treat the supplied nuisance-dominance idea as motivation only; the source was not uniquely identifiable in the initial search and must be recorded `UNVERIFIED` until a canonical URL is supplied or found. |

## Experimental funnel and stopping

1. Stage 0: deterministic linear, quadratic, nonlinear-compositional, and predictable-nuisance
   systems. Linear/quadratic cases must leave zero learned discovery; nonlinear replay must improve;
   nuisance guards must detect a nominally high-rank but causally weak representation.
2. Stage 1: one development seed, train/validation only.
3. Stage 2: at most two retained configurations per track, selected by frozen validation criteria.
   Before Stage 3, validation selects exactly one protected primary per track. A second retained
   configuration is an explicitly labelled secondary ablation and cannot replace the primary after
   any protected result. If a conjunctive claim is ever desired instead, its multiplicity-corrected
   rule must be preregistered here before execution.
4. Stage 3: three registered seeds and mandatory controls for the frozen primary and required
   baselines.
5. Stage 4: one protected execution after configs, thresholds, seeds, manifests, stopping rules,
   code, and authorization are committed and pushed. Protected results never select methods.

Failed gates stop their branch without threshold changes. Status is one of `IMPLEMENTED`,
`SMOKE_VALIDATED`, `EXECUTION_BLOCKED`, `INELIGIBLE_TASK`, `NEGATIVE_RESULT`,
`POSITIVE_EXPLORATORY`, or `POSITIVE_PROTECTED`.

## Work packages

All implementation packages use non-overlapping ownership; shared registries, package initializers,
runner dispatch, paper, and summary are updated serially after code interfaces stabilize.

1. **Terra CR core and Stage 0**: causal-effect schemas; train/validation-only baseline selection;
   exact residual targets; JVP/HVP/relinearized/population contracts; router/MLP/cross-attention/no-
   route candidates; composable objectives; nuisance suite; deterministic four-case synthetic
   benchmark; leakage, checkpoint, and provenance tests.
2. **Terra Track B stack**: new outcome-blind `QWEN-BINDING-ALGEBRA-CR-V1` amendment and digest;
   algebra tokenizer/capture contracts; FP32 causal residual trajectory;
   analytical baselines; target sufficiency; CR-LLM-IJEPA; within-context hard negatives; direct
   Qwen predicted-state replay; phase/authorization fail-closed runner; synthetic tiny-Qwen tests.
3. **Terra Track A stack**: stable physical residual targets; dense/pooled model; same-trajectory
   residual contrast; nuisance/action controls; shared and independent action-law actors; Gaussian
   and mixture heads; Direct controller; centered verifier; action-quotient and paired-compatibility
   metrics; deterministic closed-loop tests.
4. **Luna plumbing and fixtures**: frozen configs, schemas, serialization, CLI/dry-run paths,
   deterministic fixtures, protected-study authorization tests, and focused test registration.
5. **Luna scientific records**: preregistration, baseline/literature/approach/experiment registries,
   README/SUMMARY/track/risks/results synchronization, manuscript update, manifests, result cards,
   and exact continuation commands. Existing negative artifacts remain immutable.
6. **Primary Sol verification**: inspect each diff, rerun focused/full tests and audits, execute only
   eligible Stage-0/smoke studies, commit/push coherent milestones, and obtain a fresh context-clean
   Sol `ship` verdict after exact before/after state checks.

## Consequences

This design can ship a rigorous no-go outcome. A positive label is unavailable unless finite
residual signal exists, learned residual prediction improves direct replay or closed-loop control,
all registered matched controls pass, protected held-outs remain untouched until authorization,
and the decision replicates across the required seeds. Rank, variance, retrieval, probe score,
router weight, or imitation accuracy alone cannot support a causal, circuit, or workspace claim.
