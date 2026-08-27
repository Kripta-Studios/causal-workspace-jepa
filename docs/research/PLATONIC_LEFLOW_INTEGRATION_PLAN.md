# Platonic world-model geometry and LeFlow — integration plan

Status: `PREREGISTERED_PLAN_ONLY`. No experiment in this document has been run.
Papers are arXiv preprints (not peer-reviewed):

- [2608.23720](https://arxiv.org/abs/2608.23720) Platonic Representation Hypothesis on World Models (v2, 2026-08-26)
- [2608.24855](https://arxiv.org/abs/2608.24855) LeFlow: Generative Latent Flow Planning for World Models (v1, 2026-08-25)

This plan does **not** claim a workspace, platonic physics, or SOTA. It maps
what those papers actually measured onto existing causal-workspace-jepa
tracks, with HARD-002 and Qwen competence constraints intact.

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
Qwen work stays gated on competence confirmation
(`QWEN-BINDING-COMPETENCE-CONFIRM-001`). If confirmation fails, this plan
does not open a Qwen stitching or flow-planner experiment.

If confirmation passes, a **later** mechanistic question is whether
intervention-JEPA meta-latents for the same intervention family align across
seeds/sites (m-kNN on meta-states) and whether those neighborhoods predict
direct Qwen replay. That is H-LLM-02/04 territory, not a workspace claim.

### 5. What not to do

- Do not download DINO-WM / LeWM weights in `cpu_vps` mode.
- Do not treat encoder swapping as comparing Dreamer vs JEPA vs diffusion.
- Do not stitch ResNet and ViT without an explicit interface protocol.
- Do not lower HARD-002 gates because stitching “worked” in a paper.
- Do not run Steerling-8B to “explain” platonic features.
- Do not execute LeFlow-scale CEM (N=50, 30 iters) as ordinary CI.

## Recommended order

1. Close Qwen confirmation and keep V3 ineligible.
2. Keep `CRCT-COALITION-IBD-001` as the ontology positive control.
3. CPU-scale platonic control: two tiny JEPAs, different frozen linear
   observation maps, same dynamics, m-kNN + matched random maps.
4. CPU-scale amortized planning on tiny JEPA vs existing `planning/cem.py`.
5. Only then, GPU-12GB LeWM/DINO-WM adapters if checkpoints are already local
   and a new protocol is committed.

## Immediate non-claim

Nothing in this plan is evidence that JEPA discovered platonic physics or
that CEM is obsolete in this repository. The useful transfer is
**measurement design**: geometry vs stitch-function vs reachability vs
causal circuits, reported as separate columns with `evidence_level`.
