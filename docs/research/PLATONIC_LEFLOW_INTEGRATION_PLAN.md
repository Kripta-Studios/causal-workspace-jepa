# Platonic world-model geometry and LeFlow — integration plan

Status: paper-scale work remains `PREREGISTERED_PLAN_ONLY`. CPU Track A IDs
`WM-PLATONIC-MKNN-001` and `WM-LEFLOW-AMORTIZE-001` have dedicated protocols
and are **not run** until those protocol commits exist. No DINO-WM/LeWM
download. `WM-PLATONIC-STITCH-001` is not opened.
Papers are arXiv preprints (not peer-reviewed):

- [2608.23720](https://arxiv.org/abs/2608.23720) Platonic Representation Hypothesis on World Models (v2, 2026-08-26)
- [2608.24855](https://arxiv.org/abs/2608.24855) LeFlow: Generative Latent Flow Planning for World Models (v1, 2026-08-25)

This plan does **not** claim a workspace, platonic physics, or SOTA. It maps
what those papers actually measured onto existing causal-workspace-jepa
tracks, with HARD-002 and Qwen competence constraints intact.

The joint question below is **stronger than either paper** and is not
supported by them. It is a prospective research program, not a result.

## What the papers actually support

Platonic WM: frozen heterogeneous **visual encoders** (DINOv2/SigLIP/MAE/ResNet)
feeding **the same DINO-WM-style Transformer predictor** on five DINO-WM
simulators. Evidence is m-kNN geometric alignment plus **adapter-mediated**
layer stitching used for CEM planning. ResNet stays geometrically apart and
was excluded from stitching. Alignment is not identity of coordinates, not
recovery of F=ma, and not architecture-universal.

LeFlow: frozen LeWM (and a compressed DINO-WM adapter) plus a rectified-flow
latent trajectory prior, inverse dynamics, and world-model rollout reranking.
It replaces **iterative CEM inside each replan**, not replanning itself.
Short horizon (H=5) is the supported regime; H=10/20 collapse. Holdout is
in-distribution episodes, not new worlds.

## Joint question: transferable planning geometry, then platonic causal computations

The two preprints stack conceptually as:

```text
same world dynamics
        │
        ▼
partially aligned latent geometries     (Platonic WM; m-kNN, adapter stitch)
        │
        ▼
those latents can carry short-horizon plans   (LeFlow; flow + ID + WM verify)
```

The scientifically interesting move for **this** repository is not to repeat
either paper. It is the transfer test they did not run:

> If `WM_A` and `WM_B` learn the same dynamics, can a LeFlow-style planner
> trained on `A` control `B` after a simple coordinate map, and if so, is
> that because both implement equivalent **causal computations**?

```text
REAL WORLD DYNAMICS
        │
┌───────┴───────┐
▼               ▼
JEPA A         JEPA B
▼               ▼
Z_A  ~structure~  Z_B     geometry (m-kNN) is not enough
        │
        ▼
shared planning geometry?   LeFlow transfer A → B
        │
        ▼
equivalent causal computations?   CRCT on why transfer holds
```

That last step is the CRCT contribution. HARD-002 already forbids collapsing
“true circuit” into one node set. The platonic claim worth testing is
**equivalence of computations**, not equality of embeddings:

```text
not:  h_A = h_B
yes:  does both models implement interchangeable
      goal-displacement / reachability / contact / action-consequence
      computations, up to a simple gauge?
```

Proposed named computation classes are **hypotheses to plant and test**, not
discovered Qwen/JEPA mechanisms:

| class | operational meaning | matched decoy |
|---|---|---|
| goal displacement | `z_goal - z_t` predicts remaining change better than either endpoint | appearance/color features with high energy, zero effect on `z_{t+1}` |
| reachability | which `z` paths are realizable by legal actions | Euclidean nearness across a wall |
| contact dynamics | collisions/constraints change the next-state residual | high-activation unused channels |
| action consequence | `(z_t, a_t) → Δz` / inverse `(z_t, z_{t+1}) → a_t` | shuffled-action copy of the same activations |

Do **not** call a positive m-kNN “platonic causal computations.” Do not call
successful planner transfer a workspace. Workspace still requires prospective
necessity, sufficiency, specificity, faithfulness, and held-out generalization
(`docs/WORKSPACE_CRITERIA.md`, H-WM-05/06). This program at most licenses a
later H-WM-04-style claim: some features affect planner choice more than a
physical probe predicts, **and** those features transfer.

### Evidence ladder (fail-closed)

Report four columns. Never promote a lower column into a higher claim.

1. **Geometry.** m-kNN(`Z_A`,`Z_B`) vs shuffled-action and random-observation-map
   controls. Evidence level: Availability / localization.
2. **Interface.** Adapter-mediated stitch of predictors (Platonic WM). Report
   zero-adapter vs trained MLP separately. Not a circuit.
3. **Planner transfer.** Train LeFlow on frozen `A`; evaluate on frozen `B`.
   Stronger than stitching if the **flow prior** stays frozen.
4. **Causal equivalence.** CRCT: which computation classes are
   epsilon-sufficient / necessary / substitutable for the transferred
   planner. Uses coalition ontology (`CRCT-COALITION-IBD-002` prospective),
   not HARD-002 seeds.

### Transfer factorial (must be frozen before any run)

Train amortized planner on `A` only. Then on `B` evaluate:

| arm | flow prior | inverse dynamics | `g: Z_A → Z_B` | what a pass would mean |
|---|---|---|---|---|
| T0 | retrain on B | retrain on B | n/a | B is independently plannable (competence, not transfer) |
| T1 | freeze from A | retrain on B | none / tiny linear | trajectory geometry transfers; action interface does not |
| T2 | retrain on B | freeze from A | none / tiny linear | local `Δz → a` transfers; global path prior does not |
| T3 | freeze from A | freeze from A | none | strongest: planning geometry is already in the same coordinates |
| T4 | freeze from A | freeze from A | 2-layer MLP | same as stitching caveat: `g` may be doing the work |
| T5 | freeze from A | freeze from A | identity after Procrustes | coordinate gauge only |

Primary transfer metric: planning success of T1/T3/T4 versus T0 and versus a
**wrong-world** control (same architecture, different dynamics). Wall-clock
is secondary. Horizon `H` frozen a priori (`{5,10}`), with the LeFlow collapse
at long `H` treated as an expected failure mode, not a surprise.

Fail closed:

- T4 pass + T3 fail ⇒ **not** platonic coordinates; only an adapter exists
  (same limitation as the stitching paper).
- Geometry high + all transfer arms fail ⇒ neighborhoods are not a planning
  geometry (Euclidean ≠ reachability).
- Transfer to a model trained on different dynamics ⇒ leak / trivial map;
  do not interpret as shared physics.
- CRCT without a transferred planner that beats matched controls ⇒ no
  “why it transfers” question exists yet.

Verifier stays the **target** world model. Transferring a planner that only
looks good under `A`'s rollouts is not control of `B`.

### CRCT object (only after transfer competence)

Ask whether both models contain interchangeable circuits for the four
computation classes, not whether named neurons match.

Minimum CRCT report, copied from the coalition successor rather than HARD-002:

- literal graph recall of planted computation nodes (diagnostic);
- epsilon-sufficient sets for **planner success / next-state residual**,
  not activation reconstruction;
- at least two equivalent minimal circuits (e.g. two reachability routes);
- signed cancellation (equal-and-opposite contact);
- necessity of reachability vs appearance decoys;
- in-support vs out-of-support interventions;
- gauge: invertible reparameterization of `Z` that preserves planner transfer.

Positive control first: a tiny interpretable-by-design plant where those four
classes are explicit, two equivalent reachability circuits exist, and an
appearance decoy is loud. Development seeds must not be HARD-002
`1009/2027/4093` and must not reuse IBD-001 confirmation seeds. Freeze the
selector, then new confirmation seeds.

This is a **new** CRCT application. Stage-0 did not demonstrate it.
IBD-001 is only an ontology smoke, with a recorded tautological-gauge caveat.

### Proposed later IDs (not opened, not run)

- `WM-PLATONIC-MKNN-001` — geometry column. **Opened and run** as a CPU
  control; see `docs/WM_PLATONIC_MKNN_001_ADJUDICATION_2026-08-27.md`.
- `WM-PLATONIC-STITCH-001` — interface column. **Not opened.**
- `WM-LEFLOW-AMORTIZE-001` — planner competence on one frozen tiny JEPA.
  **Opened and run** as a CPU control; `NEGATIVE_RESULT`.
- `WM-LEFLOW-TRANSFER-001` — T0–T5 on two tiny JEPAs, same dynamics,
  different frozen observation maps. CPU first. **Not opened.**
- `WM-CRCT-PLATONIC-COMPUTE-001` — causal-equivalence column. Authorized only
  if `WM-LEFLOW-TRANSFER-001` beats wrong-world and adapter-only controls
  under a protocol committed before those outcomes. **Not opened.**

Do not skip to `WM-CRCT-PLATONIC-COMPUTE-001` because the diagram is attractive.

## Mapping onto this repository

### 1. Geometry ≠ causal use (Platonic × CRCT)

The platonic result is closest to evidence levels 1–2 (availability /
localization of similar neighborhoods) plus a functional stitching test that
is still not a circuit. CRCT in this repo already refuses to treat
reconstructability as unique-graph recovery (`CRCT-STAGE0-HARD-002`,
`NEGATIVE_RESULT`).

Successor questions that **are** in-scope after coalition IBD:

- Do two predictors with high m-kNN share **epsilon-sufficient** action-effect
  circuits, or only similar neighborhoods?
- Are stitched layers **necessary** for planning, or is the trained MLP adapter
  doing the causal work?
- Do equivalent minimal circuits exist across encoder families (functional
  substitutability), matching HARD-002's ontology rather than node equality?

Do **not** treat m-kNN rise as a workspace. Do not rerun HARD-002 seeds.

Proposed later IDs (not opened here):

- `WM-PLATONIC-MKNN-001` — CPU/GPU-12GB, tiny JEPA or frozen LeWM/DINO-WM
  adapters already in-repo; same trajectories; m-kNN vs shuffled-action and
  frozen-encoder-only controls.
- `WM-PLATONIC-STITCH-001` — only after planning competence exists; report
  adapter-trained vs zero-adapter stitching separately.

### 2. Reachability geometry vs semantic geometry (LeFlow × workspace tests)

LeFlow's controllable manifold point is already implicit in this repo's
planner/cost stack: Euclidean latent distance can be small while actions
cannot realize the path. `docs/WORKSPACE_CRITERIA.md` selective necessity
and H-WM-04 (planner-specific features) are the right hypotheses, not
“latent interpolation is planning.”

Proposed later IDs:

- `WM-LEFLOW-AMORTIZE-001` — tiny JEPA / MiniPush / PixelTinyMaze; compare
  existing CEM/random-shooting vs a **CPU-scale** flow or even a
  non-generative amortized latent interpolator; frozen world model;
  inverse-dynamics `(z_t, z_{t+1}, z_{t+1}-z_t)`; WM rollout rerank.
- Ablations frozen in advance: action-flow vs latent-flow; generative vs
  deterministic; N=1 vs N=64 rerank; H in {5,10} with a preregistered
  collapse warning, not a surprise.

Do not claim elimination of search. Report wall-clock separately from
success. Do not treat 80/20 episode holdout as OOD.

### 3. Inverse dynamics and H-WM-02

This repo already tests action identity from latent displacement. LeFlow's
explicit Δz input is a cheap architectural prior for that hypothesis. A
fair test is: same inverse-dynamics MLP with and without concatenated
`z_{t+1}-z_t`, capacity-matched, on Tier 0 PointMass2D / MiniPush, before
any DINO-WM download.

### 4. Qwen track: do not analogize blindly

Platonic convergence under a **shared predictor family** is not evidence that
Qwen residual streams across prompts share a platonic binding subspace.
Qwen work stays gated on competence. `QWEN-BINDING-COMPETENCE-CONFIRM-001`
passed. That still does **not** open a Qwen stitching or flow-planner
experiment. A later mechanistic question, only after `LLM-QWEN-BINDING-ALGEBRA-004`
B0, is whether intervention-JEPA meta-latents for the same intervention family
align across seeds/sites (m-kNN on meta-states) and whether those neighborhoods
predict direct Qwen replay. That is H-LLM-02/04 territory, not a workspace claim.

### 5. What not to do

- Do not download DINO-WM / LeWM weights in `cpu_vps` mode.
- Do not treat encoder swapping as comparing Dreamer vs JEPA vs diffusion.
- Do not stitch ResNet and ViT without an explicit interface protocol.
- Do not lower HARD-002 gates because stitching “worked” in a paper.
- Do not run Steerling-8B to “explain” platonic features.
- Do not execute LeFlow-scale CEM (N=50, 30 iters) as ordinary CI.
- Do not treat planner transfer as proven because m-kNN rose or because
  stitching with a trained adapter planned.
- Do not start CRCT-on-transfer until `WM-LEFLOW-TRANSFER-001` is
  preregistered, committed, and competent against wrong-world controls.

## Recommended order

1. Close Qwen confirmation and keep V3 ineligible. Done: confirmation passed;
   004 drafted not run. Qwen is **not** the first substrate for planner
   transfer.
2. Keep coalition IBD as ontology control; use IBD-002 (real gauge) before
   any “platonic computation” plant.
3. `WM-PLATONIC-MKNN-001`: two tiny JEPAs, different frozen observation maps,
   same dynamics, m-kNN vs shuffled-action and random maps.
4. `WM-LEFLOW-AMORTIZE-001`: CPU amortized planner vs existing
   `planning/cem.py` on **one** frozen tiny JEPA.
5. `WM-LEFLOW-TRANSFER-001`: T0–T5. This is the distinctive experiment.
6. `WM-CRCT-PLATONIC-COMPUTE-001` only if transfer survives the fail-closed
   gates.
7. Only then, GPU-12GB LeWM/DINO-WM adapters if checkpoints are already local
   and a new protocol is committed.

## Immediate non-claim

Nothing in this plan is evidence that JEPA discovered platonic physics,
that CEM is obsolete, that LeFlow transfers across models, or that CRCT
found shared causal computations. The useful transfer is **measurement
design**: geometry vs stitch-function vs planner portability vs equivalent
causal computations, reported as separate columns with `evidence_level`.
A transferred planner is still not a workspace.
