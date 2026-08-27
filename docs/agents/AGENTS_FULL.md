# AGENTS.md — Causal Workspace JEPA

## Current task statement

Build a complete, reproducible research codebase for two connected research tracks:

1. **Mechanistic interpretability of action-conditioned JEPA world models.** Discover, localize, causally test, and manipulate internal circuits and latent subspaces that encode dynamics, object interactions, action effects, controllability, uncertainty, goals, value, risk, and planning.

2. **JEPA as a causal meta-model of Qwen.** Treat the internal computation of an open-weight Qwen model as a dynamical system. Train an intervention-conditioned JEPA that predicts how downstream activations, sparse features, logits, and behavior change after controlled internal interventions.

The central research thesis is:

> An action-conditioned JEPA may develop a compact, causally privileged and flexibly reused latent workspace that broadcasts task-relevant state to prediction, uncertainty, value, risk, planning, and action selection. The same predictive-interventional architecture may serve as a nonlinear causal meta-model of an LLM's internal computation.

This is a research and engineering task, not a narrative exercise. Implement working code, tests, manifests, reproducible experiments, causal controls, reports, commits, and pushes. Negative results are valid. Unsupported interpretability claims are not.

Distinguish these meanings throughout the repository:

- **Environment action:** a control input applied to a simulator or robot.
- **Interpretability intervention:** an operation applied to a model activation, feature, head, MLP, token, object slot, or subspace.
- **Agent action:** a software operation performed by an implementation agent.
- **J-space:** the Jacobian-derived, token-verbalizable subspace studied in Anthropic's 2026 global-workspace work.
- **JEPA:** Joint-Embedding Predictive Architecture.
- **Workspace candidate:** an experimentally proposed JEPA subspace; its existence must not be assumed.
- **Decodability:** information can be extracted by a probe.
- **Causal use:** intervening on a representation selectively changes downstream computation or behavior.

Do not claim consciousness, sentience, subjective experience, or literal equivalence between Anthropic's J-space and any JEPA representation.

---

## Operating style inherited from the CDC research prompt

Use the following principles, adapted from OpenAI's Cycle Double Cover research prompt:

- Begin with a genuinely diverse portfolio of technical approaches.
- Do not allocate a fixed number of agents to each method.
- Preserve independence during early exploration so agents do not all converge on one attractive explanation.
- Maintain an explicit registry of approach families.
- Redirect effort when too many agents converge on one family.
- Do not let an elegant reduction, visualization, high probe score, or compelling feature label dominate without causal evidence.
- Mark a route `BLOCKED` when it depends on missing code, unavailable hardware, an untestable assumption, or a theorem-strength missing mechanism.
- Reopen blocked routes only when a materially new method, resource, dataset, or discriminating experiment becomes available.
- Keep incompatible hypotheses alive long enough to test them.
- Use adversarial agents throughout.
- Require concrete outputs: code diffs, tests, configs, manifests, causal diagrams, equations, counterexamples, metric tables, or documented negative results.
- Reject vague progress reports and unsupported optimism.
- The root agent must repeatedly synthesize, challenge, redirect, and launch new rounds.
- Do not stop after the first failed approach.
- Do not report completion until all completion criteria survive scientific and software audit.
- When resources prevent execution, finish every resource-independent component, mark the exact blocked boundary, and leave tested continuation commands.

Unlike the mathematical prompt, do not impose an artificial wall-clock requirement. Optimize for verified progress, reproducibility, and discriminating experiments.

---

## Exact required outcome

Produce a repository that can:

1. Generate deterministic action-conditioned physical datasets with known ground-truth variables.
2. Train, save, load, and evaluate a small JEPA world model.
3. Integrate or adapt at least one published action-conditioned JEPA world model.
4. Capture stable, named intermediate activations.
5. Fit and evaluate:
   - linear and nonlinear probes;
   - sparse autoencoders or sparse dictionaries;
   - Jacobian-based lenses;
   - action-effect readouts;
   - uncertainty heads;
   - intervention-conditioned JEPA meta-models.
6. Apply reproducible interventions:
   - zero, mean, and resampling ablation;
   - activation patching;
   - feature replacement;
   - feature steering;
   - subspace projection/removal;
   - module suppression;
   - head or MLP suppression when applicable.
7. Measure intervention effects on:
   - future latent predictions;
   - decoded physical variables;
   - uncertainty;
   - value, risk, and cost;
   - selected actions;
   - closed-loop task success.
8. Reconstruct and audit candidate circuit graphs.
9. Operationalize and test a JEPA workspace hypothesis using positive and negative controls.
10. Instrument a small open-weight Qwen model using Hugging Face Transformers, hooks, and autograd.
11. Generate a real dataset of Qwen intervention outcomes.
12. Train an intervention-conditioned JEPA that predicts downstream Qwen computation.
13. Compare that meta-model against:
    - no-change and mean-effect baselines;
    - linear and bilinear regression;
    - MLP predictors;
    - local Jacobian approximations;
    - corpus-averaged Jacobian Lens transports;
    - sparse-feature linear transports.
14. Validate every meta-model explanation by executing the predicted intervention directly on Qwen.
15. Preserve experiment provenance: config, data manifest, code commit, hardware, seed, logs, metrics, and failure status.
16. Keep `README.md` and all project Markdown documents synchronized with the actual repository state.
17. Run `git add`, `git commit`, and `git push` after every coherent milestone.

A notebook collection, probe-only study, visualization-only study, partial scaffold, or qualitative story does not complete the project.

---

## Evidence hierarchy

Assign every scientific claim an evidence level:

1. **Availability** — a probe decodes information.
2. **Localization** — particular layers, tokens, objects, modules, or learned features carry more information.
3. **Causal mediation** — intervention changes downstream computation or behavior.
4. **Specificity** — the targeted effect exceeds matched generic-corruption controls.
5. **Circuit reconstruction** — a compact set of nodes and edges accounts for a substantial fraction of the effect.
6. **Generalization** — the mechanism survives held-out states, prompts, tasks, magnitudes, seeds, and relevant distribution shifts.

Rules:

- Never describe level 1 evidence as causal.
- Never call a direction a “feature for X” solely because examples correlate with X.
- Never call a set of components a “circuit” without necessity, sufficiency, and faithfulness tests.
- Never claim a workspace from decodability or centrality alone.
- Every result table must include an `evidence_level` column.

---

## Primary research questions

### Track A — Interpret the JEPA world model

1. At what depth do physical variables become decodable?
2. Are physical variables localized, factorized, manifold-valued, or distributed?
3. Where and how does action information enter latent dynamics?
4. Is action identity more recoverable from `z[t+1] - z[t]` than from either endpoint?
5. Which modules causally mediate an action's effect on a future prediction?
6. Which variables influence planning but are not the strongest physical probe directions?
7. Are object interactions represented by object-local, relational, or distributed circuits?
8. Is uncertainty explicit, geometric, distributional, ensemble-based, or not stably represented?
9. Which representations are reused by dynamics, value, risk, uncertainty, and action selection?
10. Does ablating a candidate shared subspace selectively damage flexible multistep planning while sparing simple prediction?
11. How do mechanisms change under visual, physical, object, policy, task, and action-support shift?
12. Does test-time adaptation preserve, replace, or obscure the mechanism?
13. Can feature swaps induce a predicted counterfactual future and planner action?
14. Does a compact workspace-like subspace exist at all?

### Track B — Use a JEPA to interpret Qwen

1. Can an intervention-conditioned JEPA predict downstream effects better than local linear approximations?
2. Can it model nonlinear, context-dependent, multi-layer and multi-token effects?
3. Does it generalize to unseen prompts, entities, positions, layers, magnitudes, features, and intervention combinations?
4. Can it predict hidden-state and output/logit changes simultaneously?
5. Does it identify Qwen circuit candidates that survive direct causal verification?
6. Does its own meta-latent become interpretable, or does it create a second opaque model?
7. Can it model hidden computation across filler tokens?
8. Can it separate semantic content, intervention identity, and downstream effect?
9. Can a meta-model trained on a smaller dense Qwen transfer partially to Qwen3-30B-A3B?
10. How much causal fidelity is lost by compressing Qwen's computation?

---

## Mandatory preregistered hypotheses

Record these in `docs/HYPOTHESES.md` before running the associated experiment.

### World-model hypotheses

- **H-WM-01 — Physics emergence:** physical variables become sharply more decodable within a bounded intermediate-depth region.
- **H-WM-02 — Action-sensitive displacement:** action identity is more recoverable from latent displacement than from either endpoint.
- **H-WM-03 — Causal specificity:** patching an action-sensitive subspace changes the predicted trajectory toward the donor transition more than norm-matched controls.
- **H-WM-04 — Planning-specific representation:** some features have a larger causal effect on planner choice than their physical-state probe score predicts.
- **H-WM-05 — Workspace candidate:** a compact subspace is jointly read by prediction, value, risk, uncertainty, and action-selection modules.
- **H-WM-06 — Selective necessity:** workspace-candidate ablation harms flexible multistep planning more than one-step prediction and matched-dimensional controls.
- **H-WM-07 — Distributed physics:** at least one physical quantity requires coordinated multidimensional intervention rather than a single feature.
- **H-WM-08 — OOD mechanism shift:** impending OOD failures are preceded by measurable changes in circuit usage, latent geometry, or uncertainty.
- **H-WM-09 — Adaptation mechanism:** AdaJEPA-style test-time updates alter specific causal pathways rather than merely lowering mean prediction error.
- **H-WM-10 — Object causality:** object-level interventions identify interaction circuits that generalize across identities and layouts.

### LLM meta-model hypotheses

- **H-LLM-01 — Nonlinear advantage:** Intervention-JEPA predicts large or context-dependent effects better than a first-order Jacobian.
- **H-LLM-02 — Causal compression:** a compact meta-state predicts a meaningful fraction of downstream Qwen activation and logit deltas.
- **H-LLM-03 — Compositional interventions:** a model trained mostly on single interventions generalizes to some held-out pairs or sequences.
- **H-LLM-04 — Reusable mechanism features:** some meta-features predict effects across prompt families and behavioral tasks.
- **H-LLM-05 — Filler computation:** intermediate computation at content-light filler positions is decodable and causally relevant.
- **H-LLM-06 — Direct-verification precision:** circuit candidates ranked by the meta-model outperform random and probe-ranked candidates under direct Qwen intervention.
- **H-LLM-07 — Cross-scale transfer:** a representation learned on Qwen3-4B transfers partially to aligned sites in Qwen3-30B-A3B.
- **H-CROSS-01 — Shared intervention formalism:** one intervention schema and causal-fidelity metric can describe physical and LLM internal dynamics without asserting semantic equivalence.

Failed hypotheses are valid results. Do not silently change metrics, splits, thresholds, layer sets, or intervention magnitudes after observing results.

---

## Completion criteria

The project is complete only when all conditions hold:

1. Fast unit and integration tests pass.
2. A deterministic CPU-scale JEPA experiment runs end to end.
3. At least one published action-conditioned JEPA checkpoint or faithful small reproduction is integrated.
4. At least one world-model intervention causes a selective predicted change in:
   - future trajectory;
   - planning objective;
   - selected action or closed-loop behavior.
5. At least one candidate circuit is reconstructed and audited.
6. Workspace criteria are evaluated with positive and negative controls; a null result is acceptable.
7. A small open-weight Qwen model is instrumented through Hugging Face, not only Ollama.
8. A real Qwen intervention-outcome dataset is generated.
9. Intervention-JEPA is evaluated against meaningful baselines on held-out interventions.
10. Meta-model circuit predictions are verified by direct execution on Qwen.
11. Principal results use multiple seeds or justified deterministic replication.
12. Every result has config, logs, metrics, environment metadata, and commit hash.
13. Documentation reflects the actual state.
14. No reported result depends on uncommitted code.
15. The final branch is pushed to the configured remote.

---

## Insufficient outcomes

The following are explicitly insufficient:

- reproducing a paper without a mechanistic audit;
- only fitting probes or an SAE;
- only plotting PCA, t-SNE, UMAP, attention maps, feature examples, or latent trajectories;
- calling correlation causal;
- calling decodability understanding;
- using a frozen physical-state decoder as proof the planner uses the decoded variable;
- using an LLM-generated feature label as evidence;
- using Ollama as though it exposed hidden activations;
- fabricating intervention outcomes;
- overlapping training and evaluation prompts, entities, trajectories, or intervention donors;
- cherry-picked examples without aggregate metrics;
- one-seed mechanism claims;
- claiming a workspace from a small bottleneck alone;
- treating SkyJEPA as reproducible before its official code, data, checkpoints, and scripts are released;
- downloading large datasets or weights in constrained mode without resource approval;
- reporting skipped GPU tests as passed;
- returning a “best effort” summary while core interfaces or tests remain missing.

---

## Required literature and source registry

Create `docs/LITERATURE.md`, `papers/README.md`, and `papers/references.bib`. For each source record:

- title;
- authors;
- year/version;
- paper URL;
- official code URL;
- dataset/checkpoint URL;
- claimed contribution;
- relevance;
- assumptions;
- limitations;
- reproduction status;
- last verification date.

### Global workspace and Jacobian Lens

1. Anthropic, **A global workspace in language models**  
   https://www.anthropic.com/research/global-workspace

2. **Verbalizable Representations Form a Global Workspace in Language Models**  
   https://transformer-circuits.pub/2026/workspace/index.html

3. Anthropic Jacobian Lens implementation  
   https://github.com/anthropics/jacobian-lens

Extract and convert into falsifiable tests:

- Jacobian transport;
- reportability;
- voluntary control;
- causal mediation;
- flexible reuse;
- broadcast connectivity;
- selective necessity;
- limitations of token-aligned verbalizable subspaces.

Do not assume these properties transfer to a JEPA.

### JEPA and world-model foundations

4. **LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics**  
   https://arxiv.org/abs/2511.08544

5. **LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels**  
   https://arxiv.org/abs/2603.19312  
   Official code: https://github.com/lucas-maes/le-wm

6. **What Drives Success in Physical Planning with Joint-Embedding Predictive World Models?**  
   https://arxiv.org/abs/2512.24497  
   Official code/checkpoints/data: https://github.com/facebookresearch/jepa-wms

7. **Delta-JEPA: Learning Action-Sensitive World Models via Latent Difference Decoding**  
   https://arxiv.org/abs/2606.31232

8. **Causal-JEPA: Learning World Models through Object-Level Latent Interventions**  
   https://arxiv.org/abs/2602.11389  
   Official code: https://github.com/galilai-group/cjepa

9. **V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning**  
   https://arxiv.org/abs/2506.09985  
   Official code: https://github.com/facebookresearch/vjepa2

10. **V-JEPA 2.1**  
    https://arxiv.org/abs/2603.14482

11. **SkyJEPA: Learning Long-Horizon World Models for Zero-Shot Sim-to-Real Control of Quadrotors**  
    https://arxiv.org/abs/2606.23444  
    Status repository: https://github.com/arplaboratory/SkyJEPA

12. **Temporal Straightening for Latent Planning**  
    https://arxiv.org/abs/2603.12231

13. **AdaJEPA: An Adaptive Latent World Model**  
    https://arxiv.org/abs/2606.32026

14. **A Generalization Theory for JEPA-Based World Models**  
    https://arxiv.org/abs/2606.27014

### Interpretability of learned physics

15. **Interpreting Physics in Video World Models**  
    https://arxiv.org/abs/2602.07050

16. **Do Video Foundation Models Understand Intuitive Physics? A Layerwise Probing Analysis**  
    https://arxiv.org/abs/2606.09646

17. **Probing the Latent World: Emergent Discrete Symbols and Physical Structure in Latent Representations**  
    https://arxiv.org/abs/2603.20327

Use these works to motivate:

- layerwise probing;
- a possible Physics Emergence Zone;
- subspace geometry;
- high-dimensional population coding;
- targeted ablations;
- passive discretization;
- the distinction between information being present and being causally used.

### Hidden LLM computation

18. **Reading Between the Dots: Decoding Hidden Computation across Filler Tokens**  
    https://arxiv.org/abs/2607.03502  
    Code: https://github.com/kaleybrauer/filler-token-reasoning

19. **Let's Think Dot by Dot: Hidden Computation in Transformer Language Models**  
    https://arxiv.org/abs/2404.15758

20. **When Chain-of-Thought Fails, the Solution Hides in the Hidden States**  
    https://arxiv.org/abs/2604.23351

21. **Demystifying Hidden-State Recurrence: Switchable Latent Reasoning with On-Policy Reinforcement Learning**  
    https://arxiv.org/abs/2606.13106

### Qwen targets

22. **Qwen3-0.6B** — smoke and resource-limited target  
    https://huggingface.co/Qwen/Qwen3-0.6B

23. **Qwen3-4B** — primary mechanistic target  
    https://huggingface.co/Qwen/Qwen3-4B

24. **Qwen3-30B-A3B** — later-scale target and user's Ollama family  
    https://huggingface.co/Qwen/Qwen3-30B-A3B

Qwen3-30B-A3B has approximately 30.5B total parameters and 3.3B activated per token. Full instrumentation still requires storing the complete model and substantial intermediate state. Do not treat “3.3B activated” as a 3.3B-memory experiment.

---

## Model hierarchy and implementation priority

### World-model track

Implement in this order:

1. **Tiny synthetic action-conditioned JEPA**
   - exact state and causal ground truth;
   - CPU tests;
   - rapid intervention loops.

2. **LeWorldModel-compatible model or adapter**
   - small end-to-end pixel JEPA;
   - stable training;
   - physical probes;
   - suitable for a 12 GB GPU.

3. **Delta-JEPA-compatible objective**
   - action-sensitive latent displacement;
   - direct test of LDAD.

4. **JEPA-WMs pretrained adapter**
   - stronger published planning baselines;
   - prefer released checkpoints before retraining.

5. **C-JEPA adapter**
   - object-centric interactions and counterfactual tests;
   - prefer released pre-extracted slots.

6. **AdaJEPA and temporal-straightening variants**
   - OOD mechanism analysis and planning geometry.

7. **SkyJEPA adapter**
   - final long-horizon sim-to-real validation;
   - mark `BLOCKED_EXTERNAL` until the required official implementation assets exist.

### LLM track

Implement in this order:

1. Mock transformer with known activation dynamics.
2. Qwen3-0.6B when a resource-approved machine can store it.
3. Qwen3-4B as the primary mechanistic target.
4. Qwen3-30B-A3B only on suitable high-memory GPU hardware.
5. Ollama Qwen3-30B-A3B as a research assistant and tentative feature labeler, never as the hidden-activation source.

---

## Dataset plan

Create `docs/DATASETS.md` and `data/manifests/datasets.yaml`.

Each dataset record must include:

- canonical name;
- source and version;
- license;
- checksum when available;
- compressed/extracted size;
- modalities;
- observation/action/state shapes;
- train/validation/test split;
- OOD split definitions;
- download command;
- preprocessing command;
- resource modes allowed;
- purpose;
- leakage risks.

### Tier 0 — generated locally and mandatory

Keep total Tier 0 data below 512 MB by default.

1. **PointMass2D**
   - state: position and velocity;
   - action: bounded acceleration;
   - interventions: mass, drag, force scale, observation noise.

2. **BouncingBall2D**
   - state: position and velocity;
   - factors: gravity and restitution;
   - interventions: gravity, restitution, obstacle layout.

3. **TwoBodyCollision**
   - variables: mass, velocity, contact, restitution;
   - interventions: mass swap, object removal, restitution change.

4. **MiniPush**
   - small pixel-based manipulation environment;
   - ground-truth object masks and state;
   - default resolution 32×32 or 64×64.

5. **TinyMaze / TwoRooms**
   - action-conditioned navigation;
   - supports value, planning, and workspace tests.

All generation must be deterministic from explicit seeds.

### Tier 1 — primary published datasets

Use only after resource checks:

1. LeWorldModel datasets/checkpoints:
   - TwoRooms;
   - Push-T;
   - Cube;
   - Reacher.

2. JEPA-WMs:
   - Push-T;
   - PointMaze;
   - Wall;
   - MetaWorld.

3. C-JEPA:
   - CLEVRER;
   - Push-T object-slot data;
   - prefer pre-extracted slot representations before retraining a slot encoder.

4. CausalWorld:
   - generate only task families required by registered hypotheses;
   - use controlled physical and causal OOD shifts.

### Tier 2 — scaling and realism

Use only after Tier 1 succeeds:

- DROID subset used by JEPA-WMs;
- RoboCasa subset used by JEPA-WMs;
- selected ManiSkill 3 tasks;
- V-JEPA 2-AC compatible trajectories;
- SkyJEPA simulation and flight data when released.

Do not begin with Tier 2.

### LLM text and intervention datasets

Create `data/manifests/llm_prompts.yaml`.

Use:

1. Anthropic Jacobian Lens synthetic/evaluation prompts from the official repository.
2. Tiny CPU smoke corpus:
   - generated strings;
   - a tiny checked-in factual corpus;
   - optionally WikiText-2 on a suitable machine.
3. GPU research corpus:
   - WikiText-103 or a streamed C4 subset;
   - never materialize full C4 locally.
4. Synthetic mechanistic suites:
   - indirect object identification;
   - entity–attribute binding;
   - country–capital/currency/language composition;
   - multi-hop retrieval;
   - arithmetic composition;
   - string transformation;
   - in-context function execution;
   - code-bug detection;
   - goal/plan maintenance;
   - filler-token computation.
5. CounterFact-style factual intervention data with train/test entities separated.
6. Reading Between the Dots task families where model and license are compatible.

Do not use private user conversations, credentials, or secrets as research data.

### Activation storage constraints

Before capture, estimate storage:

`examples × selected layers × selected positions × hidden size × bytes per value`

Then add:
- logits/features;
- intervention metadata;
- indexes;
- checkpoint overhead.

Rules:

- never save all layers and all tokens by default;
- use selected layers and positions;
- use float16/bfloat16 where valid;
- support random projections;
- store sharded Zarr or HDF5;
- keep shards around 256–512 MB;
- add checksums and resumability;
- abort when the configured budget would be exceeded.

---

## Repository structure

Create or converge toward:

```text
.
├── AGENTS.md
├── README.md
├── LICENSE
├── CITATION.cff
├── pyproject.toml
├── uv.lock
├── Makefile
├── .gitignore
├── .env.example
├── configs/
│   ├── resource/
│   │   ├── cpu_vps.yaml
│   │   ├── gpu_12gb.yaml
│   │   └── gpu_cluster.yaml
│   ├── data/
│   ├── world_model/
│   ├── llm/
│   ├── interpretability/
│   ├── planning/
│   └── experiments/
├── src/
│   └── causal_workspace_jepa/
│       ├── __init__.py
│       ├── cli.py
│       ├── common/
│       │   ├── config.py
│       │   ├── types.py
│       │   ├── seeds.py
│       │   ├── resources.py
│       │   ├── provenance.py
│       │   ├── registry.py
│       │   └── logging.py
│       ├── data/
│       │   ├── synthetic/
│       │   ├── world_model/
│       │   ├── llm_prompts/
│       │   ├── activation_store.py
│       │   ├── manifests.py
│       │   └── splits.py
│       ├── models/
│       │   ├── components/
│       │   ├── tiny_jepa.py
│       │   ├── lewm.py
│       │   ├── delta_jepa.py
│       │   ├── intervention_jepa.py
│       │   ├── uncertainty.py
│       │   ├── probes.py
│       │   └── sparse_dictionary.py
│       ├── adapters/
│       │   ├── lewm_adapter.py
│       │   ├── jepa_wms_adapter.py
│       │   ├── cjepa_adapter.py
│       │   ├── skyjepa_adapter.py
│       │   ├── qwen_hf_adapter.py
│       │   ├── ollama_assistant.py
│       │   └── mock_transformer.py
│       ├── hooks/
│       │   ├── names.py
│       │   ├── capture.py
│       │   ├── interventions.py
│       │   ├── patching.py
│       │   └── gradients.py
│       ├── interpretability/
│       │   ├── probing.py
│       │   ├── selectivity.py
│       │   ├── sae.py
│       │   ├── jacobian_lens.py
│       │   ├── world_model_lens.py
│       │   ├── activation_patching.py
│       │   ├── causal_scrubbing.py
│       │   ├── circuit_graph.py
│       │   ├── workspace_tests.py
│       │   └── feature_labeling.py
│       ├── planning/
│       │   ├── cem.py
│       │   ├── mppi.py
│       │   ├── mpc.py
│       │   ├── costs.py
│       │   └── closed_loop.py
│       ├── experiments/
│       │   ├── world_model/
│       │   ├── llm/
│       │   └── cross_domain/
│       └── reporting/
│           ├── tables.py
│           ├── plots.py
│           ├── reports.py
│           └── cards.py
├── scripts/
│   ├── bootstrap_cpu.sh
│   ├── doctor.py
│   ├── generate_tier0.py
│   ├── download_dataset.py
│   ├── capture_qwen_activations.py
│   ├── generate_qwen_interventions.py
│   ├── run_experiment.py
│   ├── aggregate_results.py
│   └── audit_reproducibility.py
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── scientific/
│   └── fixtures/
├── data/
│   ├── README.md
│   ├── manifests/
│   ├── raw/
│   ├── processed/
│   └── activations/
├── artifacts/
│   ├── README.md
│   ├── checkpoints/
│   ├── metrics/
│   ├── figures/
│   ├── tables/
│   └── reports/
├── docs/
│   ├── ROADMAP.md
│   ├── LITERATURE.md
│   ├── RESEARCH_GAPS.md
│   ├── HYPOTHESES.md
│   ├── APPROACH_REGISTRY.md
│   ├── EXPERIMENT_REGISTRY.md
│   ├── DATASETS.md
│   ├── BASELINES.md
│   ├── REPRODUCIBILITY.md
│   ├── WORLD_MODEL_TRACK.md
│   ├── LLM_TRACK.md
│   ├── WORKSPACE_CRITERIA.md
│   ├── RESULTS.md
│   ├── RISKS.md
│   ├── DECISIONS.md
│   └── TODO.md
└── papers/
    ├── README.md
    └── references.bib
```

Preserve useful existing files. Migrate carefully. Do not delete working code merely to match the tree.

Do not commit:
- raw datasets;
- model weights;
- activation shards;
- caches;
- WandB directories;
- secrets;
- large generated artifacts.

Commit:
- manifests;
- checksums;
- small fixtures;
- configs;
- summarized metrics;
- source code;
- tests;
- Markdown reports.

---

## Core typed interfaces

Define and test interfaces before integrating large external repositories.

### World model

```python
class ActionConditionedWorldModel(Protocol):
    def encode(self, observation: Tensor) -> "LatentState": ...
    def predict(
        self,
        latent: "LatentState",
        actions: Tensor,
        *,
        return_intermediates: bool = False,
    ) -> "WorldModelOutput": ...
    def named_activation_points(self) -> Sequence[str]: ...
    def decode_state(self, latent: "LatentState") -> Mapping[str, Tensor]: ...
```

`WorldModelOutput` must include:

- predicted latent trajectory;
- optional uncertainty;
- optional decoded state;
- named intermediate activations;
- action embeddings;
- planner-facing cost features.

### Intervention specification

```python
@dataclass(frozen=True)
class InterventionSpec:
    site: str
    operation: Literal[
        "zero",
        "mean",
        "resample",
        "patch",
        "replace_feature",
        "steer",
        "project_out",
        "scale",
        "suppress_module",
    ]
    positions: tuple[int, ...] | None
    feature_ids: tuple[int, ...] | None
    magnitude: float
    donor_example_id: str | None
    seed: int
```

Every intervention must be serializable, hashable, replayable, and logged.

### Instrumented Qwen

```python
class InstrumentedCausalLM(Protocol):
    def tokenize(self, prompts: Sequence[str]) -> "TokenBatch": ...
    def forward_with_cache(
        self,
        batch: "TokenBatch",
        sites: Sequence[str],
    ) -> "LLMRun": ...
    def forward_with_intervention(
        self,
        batch: "TokenBatch",
        intervention: InterventionSpec,
        sites: Sequence[str],
    ) -> "LLMRun": ...
    def named_activation_points(self) -> Sequence[str]: ...
```

The Hugging Face adapter must expose:

- residual stream;
- attention output;
- MLP output;
- selected head output where practical;
- logits;
- token metadata;
- autograd-compatible tensors when requested.

The Ollama adapter must state that hidden activations and autograd are unavailable.

### Intervention-JEPA

Implement at least:

1. **Layer-transition JEPA**
   - context: activation at layer `l`;
   - intervention: operation at or before `l`;
   - target: activation at `l+k`, features, and logit delta.

2. **Trajectory JEPA**
   - context: selected layer/token state sequence;
   - intervention: one or more edits;
   - target: downstream activation trajectory and output effect.

Do not leak the real post-intervention target into the context branch.

---

## World-model analogues of a Jacobian Lens

Implement:

1. **Dynamics lens** — influence on future latent states.
2. **Physical-state lens** — influence on decoded physical variables.
3. **Action lens** — influence on predicted action effects or selected action.
4. **Value/cost lens** — influence on goal-conditioned cost.
5. **Risk lens** — influence on failure/constraint prediction.
6. **Uncertainty lens** — influence on predictive variance, disagreement, or OOD score.

Each lens must support:

- local Jacobian mode;
- corpus-averaged transport where appropriate;
- finite-difference verification;
- comparison with direct interventions;
- error as intervention magnitude grows;
- layerwise and horizon-wise evaluation.

A lens is an approximation, not causal proof.

---

## Workspace tests

Create `docs/WORKSPACE_CRITERIA.md`. Test these functional properties:

1. **Decodability/reportability analogue**
   - a compact subspace predicts multiple explicit state variables or symbolic descriptions.

2. **Controllability**
   - goals, instructions, or environment conditions selectively modulate the candidate subspace.

3. **Causal mediation**
   - swap or ablation changes prediction and planning in the predicted direction.

4. **Flexible reuse**
   - the same representation is read by dynamics, value, risk, uncertainty, action selection, and optional symbolic explanation.

5. **Broadcast centrality**
   - unusually broad causal influence after controlling for activation norm and dimensionality.

6. **Selective necessity**
   - ablation harms flexible multistep planning more than matched controls and simple tasks.

7. **Compactness**
   - dimension required to recover a fixed fraction of the causal effect.

8. **Depth/horizon evolution**
   - how workspace content changes across predictor depth and rollout horizon.

Controls:

- random subspaces;
- high-variance principal components;
- probe-optimal but noncausal directions;
- norm-matched directions;
- shuffled labels;
- irrelevant donor examples;
- equal-dimensional ablations outside the candidate subspace.

A null result is acceptable.

---

## Interpretability methods and controls

### Probes

Implement:

- ridge/logistic linear probes;
- small MLP probes;
- layerwise and horizon-wise probes;
- trajectory-, seed-, layout-, object-, and policy-separated splits;
- random-label controls;
- selectivity and complexity reporting;
- confidence intervals.

Probes must never update the base model.

### Sparse representation learning

Implement an SAE or nonnegative sparse dictionary with:

- configurable expansion factor;
- sparsity control;
- reconstruction score;
- dead-feature fraction;
- feature density;
- seed stability;
- feature matching across runs;
- feature-space interventions;
- donor-example retrieval.

Do not equate sparsity with monosemanticity.

### Activation patching

Support:

- clean/corrupted/donor runs;
- patching by layer, token, object, time, feature, or module;
- normalized recovery;
- behavioral and planning recovery;
- irrelevant-donor controls.

### Circuit graph

Evaluate circuit candidates by:

- node ablation;
- edge ablation;
- necessity;
- sufficiency;
- faithfulness;
- minimality;
- held-out generalization.

Store graphs as JSON and GraphML.

### Uncertainty

Start with:

1. ensemble/bootstrap disagreement;
2. heteroscedastic residual or probabilistic latent head.

Evaluate:

- NLL where valid;
- calibration;
- ECE;
- Brier score;
- selective risk;
- OOD AUROC/AUPRC;
- relation to planning failure.

Do not assume uncertainty has a dedicated feature.

---

## Baselines

Record in `docs/BASELINES.md`.

### World-model baselines

- oracle/raw-state dynamics;
- simple autoencoder dynamics;
- no-action JEPA;
- shuffled-action JEPA;
- frozen random encoder;
- linear latent dynamics;
- LeWorldModel;
- JEPA-WMs;
- Delta-JEPA objective;
- C-JEPA where applicable;
- DINO-WM where released;
- V-JEPA-2-AC where released;
- local Jacobian;
- probe-only interpretation;
- matched random-subspace intervention.

### LLM meta-model baselines

- no-change predictor;
- mean intervention effect;
- linear regression;
- bilinear predictor;
- local Jacobian;
- corpus-averaged Jacobian Lens;
- MLP predictor;
- autoregressive layer predictor;
- nearest-neighbor intervention retrieval;
- SAE-feature linear transport.

Do not declare success from beating only a trivial baseline.

---

## Splits and leakage prevention

### World-model splits

Separate by:

- trajectory;
- simulator seed;
- initial state;
- goal;
- layout;
- object identity/configuration;
- behavior policy;
- action distribution;
- environment parameter.

OOD categories:

- visual OOD;
- physical OOD;
- object OOD;
- policy OOD;
- action-support OOD;
- task OOD.

### LLM splits

Separate by:

- prompt template;
- entity;
- answer;
- reasoning depth;
- token length;
- intervention site;
- intervention type;
- magnitude;
- feature identity;
- prompt family.

Prevent donor/recipient leakage in causal patching.

Fit normalization, PCA, SAE, probes, labelers, and averaged Jacobians only on training data unless an experiment is explicitly registered as transductive.

---

## Metrics

### World-model prediction and planning

- latent MSE/cosine error;
- multistep rollout error;
- decoded state RMSE;
- event detection;
- action reconstruction;
- trajectory curvature;
- planner success;
- planning time;
- closed-loop return.

### Causal fidelity

- direct-effect magnitude;
- predicted-versus-observed effect correlation;
- sign accuracy;
- normalized recovery;
- KL divergence or logit-delta error;
- trajectory endpoint error;
- action agreement;
- specificity ratio;
- circuit faithfulness.

### Workspace

- causal effect captured versus dimension;
- number/diversity of consumers;
- broadcast centrality;
- selective-necessity ratio;
- cross-task reuse;
- stability across seeds.

### LLM meta-model

- downstream activation cosine/MSE;
- sparse-feature delta correlation;
- logit-delta correlation;
- KL divergence;
- top-k changed-token overlap;
- answer-change classification;
- calibration;
- held-out layer/site score;
- combined-intervention score;
- runtime versus direct execution, reported separately from fidelity.

Always report sample count, seed count, dispersion, and confidence intervals where appropriate.

---

## Milestones

### Milestone 0 — Audit and safe bootstrap

1. Read `AGENTS.md`, `README.md`, every `docs/*.md`, Git status, and recent history.
2. Preserve existing scientific results.
3. Create missing structure.
4. Implement resource detection and profiles.
5. Implement a `doctor` command.
6. Add CI-safe tests.
7. Update Markdown.
8. Commit and push.

Acceptance:

- CPU-mode install succeeds;
- doctor runs;
- fast tests pass;
- no heavy download occurs.

### Milestone 1 — Tier 0 environments and tiny JEPA

1. Implement deterministic environments.
2. Generate tiny datasets.
3. Implement encoder, target encoder, action-conditioned predictor, and regularization.
4. Add no-action and shuffled-action controls.
5. Add one-step/multistep training.
6. Add checkpointing and manifests.
7. Add a simple planner.
8. Commit and push.

Acceptance:

- CPU smoke training finishes;
- prediction beats a mean baseline;
- planner beats random in one toy task;
- tests confirm action conditioning.

### Milestone 2 — Interpretability foundation

1. Named activation points.
2. Activation cache/store.
3. Probes with controls.
4. SAE/dictionary.
5. Intervention engine.
6. Activation patching.
7. Jacobian and finite-difference utilities.
8. Circuit graph schema.
9. Commit and push.

Acceptance:

- a synthetic variable is decoded;
- a targeted intervention changes the expected downstream variable;
- matched controls are weaker.

### Milestone 3 — Tier 0 mechanistic study

1. Register hypotheses.
2. Run layerwise physical probing.
3. Study latent displacement and actions.
4. Localize dynamics circuits.
5. Compare single-direction with multidimensional control.
6. Compare physical decodability with planner influence.
7. Evaluate workspace criteria.
8. Commit results and negative results; push.

### Milestone 4 — LeWorldModel integration

1. Pin official repository version.
2. Implement an adapter.
3. Prefer checkpoint plus small dataset before retraining.
4. Verify prediction/planning.
5. Run the same mechanistic protocol.
6. Commit and push.

### Milestone 5 — Action and object causality

1. Implement LDAD/Delta-JEPA objective.
2. Compare endpoint and displacement action decoding.
3. Integrate C-JEPA slots.
4. Run object swap/removal/contact interventions.
5. Audit causal use.
6. Commit and push.

### Milestone 6 — Uncertainty, OOD, and planning

1. Add uncertainty methods.
2. Add all OOD split families.
3. Detect mechanism shifts before failures.
4. Add selective planning/abstention.
5. Optionally add AdaJEPA-style test-time adaptation.
6. Audit mechanism changes caused by adaptation.
7. Commit and push.

### Milestone 7 — Qwen instrumentation

1. Implement Hugging Face Qwen adapter.
2. Implement selected-layer capture.
3. Add mock model for CPU tests.
4. Reproduce a small Jacobian Lens example on suitable hardware.
5. Implement residual/MLP/attention interventions.
6. Add filler-token and compositional prompt generators.
7. Commit and push.

Acceptance:

- direct intervention changes activations and logits as expected;
- replay is deterministic;
- Ollama is not used as a hidden-state source.

### Milestone 8 — Qwen intervention dataset

1. Preregister prompts, sites, operations, and splits.
2. Estimate storage.
3. Generate normal and intervened runs.
4. Store pre-state, intervention, post-state, logits, behavior, and metadata.
5. Validate shards/checksums.
6. Commit manifests/scripts, not tensors.
7. Push.

### Milestone 9 — Intervention-JEPA

1. Implement layer-transition and trajectory variants.
2. Train on Qwen intervention data.
3. Compare against baselines.
4. Test held-out prompts, layers, magnitudes, and combinations.
5. Calibrate uncertainty.
6. Commit and push.

### Milestone 10 — Interpret the interpreting JEPA

1. Probe meta-model features.
2. Fit sparse dictionaries.
3. Patch and steer meta-model activations.
4. Test whether explanations predict real Qwen effects.
5. Compare meta-model circuits with direct Qwen circuits.
6. Quantify compression loss.
7. Commit and push.

### Milestone 11 — Cross-domain synthesis

1. Unify intervention schema and metrics.
2. Compare physical and LLM dynamics without asserting semantic equivalence.
3. Identify which methods transfer and which fail.
4. Produce final report and reproducibility bundle.
5. Commit and push.

### Milestone 12 — SkyJEPA validation

Run only when official resources are available.

1. Integrate official implementation.
2. Reproduce simulator results.
3. Apply validated interpretability methods.
4. Test payload, thrust, inertia, drag, and delay changes.
5. Attempt sim-to-real only with safe procedures.
6. Never run unvalidated feature steering on real flight hardware.
7. Commit and push.

---

## Dynamic multi-agent strategy

Launch independent approach families such as:

- architecture/interfaces;
- world-model reproduction;
- action sensitivity;
- object causality;
- uncertainty;
- planning;
- probes and geometry;
- sparse features;
- causal intervention;
- workspace criteria;
- Qwen hooks;
- Jacobian methods;
- Intervention-JEPA;
- provenance and data;
- resource/storage audit;
- adversarial scientific audit.

Maintain `docs/APPROACH_REGISTRY.md` with:

- approach family;
- owner/agent;
- exact question;
- current evidence;
- blockers;
- next discriminating experiment;
- status: `ACTIVE`, `BLOCKED`, `REJECTED`, `MERGED`, `COMPLETE`.

Do not tell most early subagents the favored explanation. Require concrete artifacts. Redirect duplicated effort. Reopen blocked approaches only with a genuinely new mechanism.

Adversarial audits must check:

- leakage;
- probe overinterpretation;
- control mismatch;
- generic corruption;
- cherry-picking;
- seed sensitivity;
- storage/compute assumptions;
- train/eval mode;
- donor leakage;
- causal language;
- nondeterminism;
- stale documentation;
- broken external dependencies.

---

## Resource modes

### `cpu_vps`

Target:

- 4 CPU cores around 2 GHz;
- 8 GB RAM;
- no GPU;
- approximately 14 GB storage.

Hard rules:

- do not download Qwen weights;
- do not download V-JEPA, JEPA-WMs, LeWM, C-JEPA, or SkyJEPA weights;
- do not download Tier 1/Tier 2 datasets;
- do not install CUDA packages;
- do not use Docker or a large Conda environment;
- use `uv`;
- preserve at least 4 GB free;
- cap project caches at 2 GB;
- cap generated Tier 0 data at 512 MB;
- do not run large SAEs, Jacobian fitting, or activation captures;
- do not launch hours-long jobs without a tiny smoke config.

Allowed:

- audit and documentation;
- package scaffolding;
- typed interfaces;
- resource guards;
- toy simulators;
- tiny CPU JEPA;
- mock Qwen adapter;
- intervention schemas;
- activation-store code;
- unit/integration tests;
- CLI and CI;
- tiny end-to-end experiments;
- Git commits and pushes.

### `gpu_12gb`

Target: RTX 5070 Ti Laptop with about 12 GB VRAM.

Allowed:

- LeWorldModel small experiments;
- Tier 0 and selected Tier 1 data;
- Qwen3-0.6B and Qwen3-4B with short sequences and selected layers;
- sharded activation capture;
- small SAEs/probes;
- direct interventions;
- reduced Jacobian experiments;
- mixed precision and gradient accumulation.

Rules:

- estimate VRAM first;
- default to Qwen3-4B, not Qwen3-30B-A3B;
- avoid all-layer/all-token Jacobians;
- stop and record OOM;
- do not silently change a preregistered experiment.

### `gpu_cluster`

Required for:

- Qwen3-30B-A3B full hidden-state/autograd studies;
- large intervention datasets;
- all-layer Jacobian fitting;
- large V-JEPA 2.1;
- DROID/RoboCasa training;
- broad multi-seed scaling;
- SkyJEPA-scale reproduction.

Record GPU model/count/memory, wall time, estimate, and actual cost.

---

## CPU VPS bootstrap commands

The agent must implement and keep these commands valid:

```bash
git status --short
git branch --show-current
git remote -v
git log -5 --oneline
df -h
free -h
nproc

curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"

uv venv --python 3.11
source .venv/bin/activate

uv pip install -e ".[dev,cpu]"

python -m causal_workspace_jepa.cli doctor \
  --resource-profile configs/resource/cpu_vps.yaml

pytest -q tests/unit tests/integration -m "not gpu and not slow"

python scripts/generate_tier0.py \
  --config configs/data/tier0_smoke.yaml

python scripts/run_experiment.py \
  --config configs/experiments/tiny_jepa_smoke.yaml

python scripts/audit_reproducibility.py
```

If CPU PyTorch would violate storage, implement initial interfaces/tests with NumPy and report the estimate before installing PyTorch.

A skipped GPU test must be recorded as `SKIPPED_RESOURCE`, never `PASS`.

---

## GPU continuation commands

Keep exact working variants in `README.md`:

```bash
source .venv/bin/activate

python -m causal_workspace_jepa.cli doctor \
  --resource-profile configs/resource/gpu_12gb.yaml

python scripts/download_dataset.py \
  --name lewm_pusht \
  --manifest data/manifests/datasets.yaml \
  --confirm-size

python scripts/run_experiment.py \
  --config configs/experiments/lewm_pusht_probe.yaml

python scripts/run_experiment.py \
  --config configs/experiments/lewm_pusht_causal_patch.yaml

python scripts/capture_qwen_activations.py \
  --config configs/llm/qwen3_4b_selected_layers.yaml

python scripts/generate_qwen_interventions.py \
  --config configs/experiments/qwen_intervention_dataset_v1.yaml

python scripts/run_experiment.py \
  --config configs/experiments/intervention_jepa_v1.yaml
```

---

## Git discipline

At the start of every milestone:

```bash
git status --short
git branch --show-current
git remote -v
git log -5 --oneline
```

Before editing:

- read relevant files fully;
- preserve user changes;
- do not reset, clean, force-checkout, or rewrite history;
- do not delete untracked files without understanding them;
- never commit secrets.

After every coherent milestone:

```bash
git status --short
git diff --check
pytest -q <relevant-tests>
git add -A
git commit -m "<type>: <concise milestone description>"
git push
```

Preferred commit prefixes:

- `chore:`
- `docs:`
- `feat:`
- `test:`
- `fix:`
- `refactor:`
- `exp:`
- `audit:`

Rules:

- no giant final commit;
- no `git push --force`;
- if no remote exists, record `PUSH_BLOCKED_NO_REMOTE`;
- if auth fails, preserve local commits and record the exact error;
- every experiment records its Git commit;
- every Markdown result links to its config and metrics artifact.

---

## Markdown discipline

`README.md` is the public source of truth. It must always contain:

- objective;
- current status;
- implemented features;
- blocked work;
- quick start;
- resource modes;
- dataset policy;
- exact experiment commands;
- latest validated results;
- limitations;
- repository map.

Update after every milestone:

- `README.md`;
- `docs/ROADMAP.md`;
- `docs/EXPERIMENT_REGISTRY.md`;
- `docs/RESULTS.md`;
- `docs/TODO.md`;
- `docs/DECISIONS.md`;
- `docs/RISKS.md`.

Use status labels:

- `NOT_STARTED`;
- `SCAFFOLDED`;
- `IMPLEMENTED_UNVALIDATED`;
- `SMOKE_VALIDATED`;
- `SCIENTIFICALLY_VALIDATED`;
- `BLOCKED_RESOURCE`;
- `BLOCKED_EXTERNAL`;
- `REJECTED`.

Never write “complete” when only scaffolding exists.

---

## Testing requirements

### Unit tests

Cover:

- tensor shapes/dtypes;
- deterministic seeds;
- dataset splits;
- intervention serialization;
- hook names;
- patch semantics;
- resource guards;
- manifest validation;
- metrics;
- graph I/O.

### Integration tests

Cover:

- tiny JEPA train/save/load;
- activation capture;
- intervention replay;
- probe training;
- tiny SAE training;
- planner integration;
- mock-Qwen intervention data;
- Intervention-JEPA train/evaluate.

### Scientific tests

At minimum:

- shuffled actions reduce action reconstruction;
- zero intervention has near-zero effect;
- donor patch recovers a known synthetic variable;
- matched random direction has lower targeted effect;
- train/eval trajectories do not overlap;
- meta-model does not receive the post-intervention target;
- intervention prediction is evaluated on held-out prompts/interventions;
- a high probe direction that fails causal tests is not promoted.

Scientific tests may be `slow` or `gpu`, but must exist.

---

## Experiment registry

Every `docs/EXPERIMENT_REGISTRY.md` entry must include:

- experiment ID;
- hypothesis ID;
- date;
- owner/agent;
- status;
- model/checkpoint;
- dataset/split;
- config;
- command;
- commit;
- hardware;
- seeds;
- expected runtime;
- output path;
- primary metric;
- decision threshold;
- result;
- interpretation;
- caveats;
- next action.

Use unique IDs such as:

- `WM-T0-001`;
- `WM-LEWM-001`;
- `WM-DELTA-001`;
- `WM-CJ-001`;
- `WM-OOD-001`;
- `LLM-JLENS-001`;
- `LLM-INTDATA-001`;
- `LLM-IJEPA-001`;
- `CROSS-001`.

---

## Qwen through Ollama

The user's Ollama Qwen3-30B may assist with:

- literature synthesis;
- source-grounded paper cards;
- code review;
- candidate feature descriptions;
- clustering examples;
- proposing falsifying tests;
- drafting reports.

It must not be used as evidence. It must not be described as exposing hidden states.

For feature labeling:

1. Provide only redacted, non-secret examples.
2. Ask for several competing labels.
3. Ask for confounds and falsifying tests.
4. Store the prompt, model tag, parameters, and response.
5. Validate labels using held-out examples and direct interventions.
6. Mark labels `TENTATIVE` until causal evidence exists.

Recommended labeler prompt:

```text
You are labeling a learned feature for a mechanistic-interpretability study.

Evidence:
- top activating examples;
- negatively activating examples;
- donor/recipient patch outcomes;
- downstream effect metrics;
- counterexamples.

Return:
1. three competing interpretations;
2. evidence for each;
3. likely confounds;
4. a minimal falsifying experiment;
5. a calibrated confidence score.

Do not treat correlation as causation.
Do not infer a human-like thought unless the evidence directly supports that wording.
```

---

## Safety and integrity

- Never exfiltrate prompts, credentials, API tokens, private data, or proprietary model activations.
- Never commit `.env`, tokens, private SSH material, or user conversations.
- Respect dataset and model licenses.
- Do not bypass gated-model terms.
- Do not perform unsafe real-robot interventions.
- Do not hide failed runs, NaNs, OOMs, or negative results.
- Do not fabricate citations, metrics, logs, or completed experiments.
- If a source cannot be verified, mark it `UNVERIFIED`.
- If a paper lacks released code, distinguish conceptual reimplementation from reproduction.
- Store checksums and environment metadata.
- Keep model-generated scientific text separate from measured evidence.

---

## First execution order on the current CPU VPS

Do this now:

1. Audit Git and disk/RAM.
2. Install only `uv` and a minimal CPU development environment.
3. Create the repository tree and Markdown registries.
4. Implement resource profiles and `doctor`.
5. Implement typed interfaces and mock adapters.
6. Implement deterministic Tier 0 generators.
7. Implement a very small CPU JEPA and one smoke experiment.
8. Implement intervention serialization and unit tests.
9. Implement mock Qwen activation dynamics and a tiny Intervention-JEPA test.
10. Update `README.md`.
11. Commit and push each coherent milestone.

Do not do this now:

- download Qwen weights;
- download published JEPA checkpoints;
- download Tier 1/Tier 2 datasets;
- run real Qwen hooks;
- fit Jacobian Lens;
- train large SAEs;
- run V-JEPA/SkyJEPA;
- claim scientific validation.

The root agent must leave the repository in a state that can be pulled on the GPU machine and continued without redesign.

---

## Final report requirements

The final report must include:

1. exact tested claims;
2. evidence level;
3. positive and negative results;
4. causal controls;
5. generalization boundaries;
6. resource use;
7. reproducibility instructions;
8. known limitations;
9. failed approaches and why;
10. whether a workspace candidate was found;
11. whether Intervention-JEPA beat Jacobian and regression baselines;
12. whether predictions survived direct Qwen verification;
13. which experiments remain blocked by hardware or external releases.

Return success only after the repository, tests, experiments, documentation, commits, and pushes survive adversarial audit.

LLM implementation order:

1. Mock transformer with known circuits.
2. GPT-2 Medium  as the first real mechanistic target.
3. Qwen3-0.6B.
4. Qwen3-4B.
5. Qwen3-30B-A3B.

In cpu_vps mode, GPT-2 Medium is permitted only when:
- at least 6 GB of disk is free before installation;
- sequence length is at most 64;
- activation storage is estimated before capture;
- no more than six layers and selected token positions are stored initially;
- the smoke dataset is at most 200 prompts;
- the first research dataset is at most 3,000 prompt-intervention pairs;
- hidden states are stored in float16 shards;
- Jacobians are computed only for selected layers, positions, and outputs.
