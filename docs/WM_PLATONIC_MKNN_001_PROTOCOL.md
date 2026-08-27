# WM-PLATONIC-MKNN-001 protocol

Status: adjudicated `TRANSITION_NEIGHBORHOOD_ALIGNMENT_PASSED` (Availability)
on 2026-08-27. Frozen gates, seeds, maps, and k in this file were not
changed after outcomes. See
`docs/WM_PLATONIC_MKNN_001_ADJUDICATION_2026-08-27.md`.

This is a CPU-scale **geometry control**, not a Platonic WM paper
reproduction, not stitching, not CRCT, not a workspace, and not permission
to relabel `CRCT-STAGE0-HARD-002`.

Parent plan: `docs/research/PLATONIC_LEFLOW_INTEGRATION_PLAN.md`.
Config: `configs/experiments/wm_platonic_mknn_v1.json`.

## Scientific question

If two tiny action-conditioned JEPAs see the **same PointMass2D dynamics**
through **different frozen linear observation maps**, and are trained on the
**same trajectories** with the **same predictor family and budget**, how much
**transition-neighborhood** geometry aligns under mutual k-NN?

This is **not** a test that latent coordinates become identical.

## Evidence level if the primary gate passes

`Availability` — localization of similar transition neighborhoods.

A pass does **not** establish: shared coordinates, shared causal circuits,
functional necessity, workspace, platonic physics, recovery of `F=ma`, or
architecture-universal convergence.

A fail is recorded as `NEGATIVE_RESULT` and is not retuned.

## Frozen substrate

| field | frozen value |
|---|---|
| environment | PointMass2D (`generate_pointmass2d`) |
| state dim | 4 (`x,y,vx,vy`) |
| action dim | 2 |
| observation maps | frozen Gaussian linear `R^4 → R^16`, not trained |
| encoder | frozen identity `I_16`; never fitted |
| predictor | ridge linear action-conditioned TinyJEPA |
| latent dim | 16 |
| ridge | `1e-4` |
| trajectories | 80 |
| steps | 16 |
| split | train 48 / development 16 / confirmation 16 |
| split seed | 227 |
| environment base seed | 211; dataset seed = `211 + confirmation_seed` |
| map A seed | 801 |
| map B seed | 809 |
| random-map seed | 817 |
| noise seed | 223 |
| k | 5 |
| n_eval | 128 confirmation states, trajectory-major prefix |
| resource | `cpu_vps`; no downloads |

Development trajectories exist so the split is fully specified. They are
**not** an adjudicating split and are **not** a threshold-calibration set.
Thresholds below are frozen from chance arithmetic, not from data.

There is no `test` or `paraphrase` split. Those names must not be accessed.

## Frozen observation-map construction

```text
W ~ Normal(0, 1/sqrt(4)) in R^{4×16}
obs = state @ W
```

Maps are generated once from the frozen map seeds and reused for train and
confirmation. They are never updated. No learned adapter is allowed.

Random-map control: a third frozen map applied to **isotropic Gaussian noise**
of the same shape as states (not a map of the true state). Train and evaluate
that arm on that noise, paired by index only.

## Frozen seeds

Confirmation seeds, chosen before any target metric is computed: `131, 137, 139`.

Forbidden (must raise): HARD-002 `1009/2027/4093`; IBD-001 `11/13/17/811/823/829`;
IBD-002 `21/23/29/941/947/953`; Qwen confirmation `701`; Qwen 004 `901`.

Do not add seeds after seeing outcomes.

## Frozen metrics (report separately; do not collapse)

1. `encoder_mknn` — m-kNN of encoded confirmation states `z_t`.
2. `predictor_mknn` — m-kNN of one-step predicted `z_{t+1}` under **matched
   true confirmation actions**.
3. `action_conditioned_mknn` — m-kNN of one-step predicted `z_{t+1}` under a
   **frozen probe action** `[0.5, 0.0]` on the same states.

Primary adjudicating metric: `predictor_mknn` of pair `(A, B)`.

Controls on the same primary metric:

- shuffled-action: TinyJEPA `action_mode="shuffled_action"` on B observations;
  evaluate with true actions.
- random-map: noise-map model vs A.

Chance reference: `k / (n_eval - 1) = 5/127`.

## Frozen gates (pre-outcome)

Pass (`TRANSITION_NEIGHBORHOOD_ALIGNMENT_PASSED`) iff **every** confirmation
seed satisfies all three:

1. `predictor_mknn(A,B) > predictor_mknn(A, shuffled-action)`
2. `predictor_mknn(A,B) > predictor_mknn(A, random-map)`
3. `predictor_mknn(A,B) > 2 * chance`

Rationale, frozen before outcomes: chance is the index-overlap rate of a
uniform k-subset of `n_eval-1` others. The `2×` floor is a pre-outcome
convention that the pair must beat that null, **and** both mechanistic
controls. It is not calibrated on this task. Encoder geometry is reported,
not gated, because observation maps are allowed to scramble coordinates.

Fail-closed: `NEGATIVE_RESULT`. Do not lower k, change maps, change seeds, or
train an adapter to obtain alignment.

## Interpretation constraints

m-kNN increase may support availability/localization of similar transition
neighborhoods. If `encoder_mknn` is already above the predictor gate, say so:
predictor alignment may be inherited from observation geometry rather than
learned transitions. That caveat does not change the frozen gate.

Do not implement stitching. `WM-PLATONIC-STITCH-001` remains unopened.

## Artifacts

- `artifacts/metrics/wm_platonic_mknn_v1.json`
- `artifacts/metrics/wm_platonic_mknn_v1.provenance.json`

Provenance `git_dirty` must be false. Protocol and implementation commits
must predate the result commit.

## Stopping rule

One confirmation pass over the three frozen seeds. No extra seeds, no k
search, no map search, no threshold edit after metrics exist.
