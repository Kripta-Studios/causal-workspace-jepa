# CRCT-JEPA-ACTION-DELTA-001 protocol

Status at freeze: `PREREGISTERED_NOT_RUN`.
After the freeze commit, execution of **this ID only** is authorized.

Parent: `CRCT-COALITION-IBD-003` (`MECHANISM_RECOVERY_PASSED`, synthetic IBD).
This experiment asks whether CRCT recovers a mechanism a network **learned**,
not one planted by design.

HARD-002 remains `NEGATIVE_RESULT`. IBD-002 remains not executed.
Qwen 004, stitching, Reachable-003, Rectified Flow, DINO-WM, and LeWM remain
unauthorized. TinyJEPA ridge/identity is **not** the substrate.

The draft `docs/research/CRCT_JEPA_ACTION_DELTA_001_DRAFT.md` is not this freeze.

## Question

Does label-blind CRCT recover a minimal causally sufficient and necessary
internal coalition for the learned map **`ax -> Delta vx`**, with target
specificity, matched-control rejection, and a counterfactual activation patch,
on independently trained confirmation seeds?

Decoding `ax` or high activation magnitude is not the claim.

## Architecture (frozen)

CPU PyTorch **supervised residual MLP** (not a JEPA/joint-embedding objective),
residual stream width `H=6`:

```text
state[4] --Linear-Tanh-Linear-Tanh--> z[6]
action[2] --Linear-Tanh--> e[6]
cat(z,e) --Linear-Tanh--> h0[6]
h1 = h0 + Linear(Tanh(Linear(h0)))     # block 1
h2 = h1 + Linear(Tanh(Linear(h1)))     # block 2
Delta_hat[4] = Linear(h2)
```

No attention. Output is **Delta state**, not next-state.

## Component ontology (searchable sites)

Eighteen scalar units, unnamed to the selector:

```text
act_0..act_5   action-embedding coordinates of e
b1_0..b1_5     block-1 MLP hidden (post-tanh, pre-second linear)
b2_0..b2_5     block-2 MLP hidden
```

Encoder units are trained but **not** in the search set (frozen omission).
Interventions are mean-ablate / restore / patch of these coordinates using
**training-split mean activations** (in-support). Random Gaussian fills are
out-of-support and are labeled, not used as pass tokens.

## Physics (known) vs mechanism (unknown)

PointMass2D via `generate_pointmass2d` (`dt=0.1`, `drag=0.05`, `mass=1`).
External maps (define the *function*, not the circuit):

```text
M1 primary: ax -> Delta vx
M2: ay -> Delta vy
M3: vx -> Delta x
M4: vy -> Delta y
```

Cross-axis `ax -> Delta vy` and `ay -> Delta vx` are negative-control
readouts, not extra circuits to pass.

## Seeds and splits (chosen before generation)

Forbidden: HARD-002 `1009,2027,4093`; IBD-001/002/003 seeds.

- Development **model** seeds: `43, 47, 53`
- Confirmation **model** seeds: `1013, 1019, 1021`

Per model seed `S`:

```text
train trajectories:  generate_pointmass2d(256 traj, 6 steps, seed=S*1000+61)
development eval:    generate_pointmass2d(64 traj, 6 steps, seed=S*1000+67)
confirmation eval:   generate_pointmass2d(64 traj, 6 steps, seed=S*1000+71)
```

Confirmation eval (`S*1000+71`) is generated **only in the confirmation
stage**, and only **after** that seed’s circuit is frozen on development
eval. Development invocations must not load or generate confirmation
trajectories. Train only on train transitions. No confirmation in
training, selection, or competence.

## Training (frozen)

Adam, `lr=3e-3`, `batch=64`, **200** steps, seed = model seed.
Loss: MSE of predicted vs true Delta state.
Checkpoint = the frozen-budget weights (no confirmation-based early stop).

## Competence (development only, fail-closed)

On that seed’s development eval (`S*1000+67`), NMSE vs **physics** for each
of `Delta x, Delta y, Delta vx, Delta vy` must be `<= 0.05`. Confirmation
eval is never used for competence, selection, or stopping.

If any seed fails: that seed is `MODEL_INCOMPETENT`. If any development seed
is incompetent: confirmation stays **CLOSED** and the aggregate is
`MODEL_INCOMPETENT`. No architecture enlargement. Confirmation seeds apply
the same competence rule on *their* development eval before interpretation.

## Selector (label-blind, development eval)

Target for discovery: the **network’s** original `Delta vx` channel
(explain the model, not a physics oracle).

Greedy restore-only, max size **4** (cannot select the entire 6-unit action
embedding), min step **0.02 NMSE** (absolute energy-normalized improvement,
i.e. 0.02 of original target energy because NMSE already divides by that
energy). Start from all-sites mean-filled. Add the site that most reduces
NMSE(restored `Delta vx`, original `Delta vx`). Stop when NMSE `<= 0.05` or
no step clears `min_step_nmse`. Then **inclusion-minimal prune**: drop any
member whose removal keeps sufficiency `<= 0.05`. The pruned set is `C_hat`.

Selecting all six `act_*` units is `INCONCLUSIVE` (architecture cut-set, not
a discovered sparse circuit). The full action-embedding cut-set is still
**reported** as `architecture_action_cutset`.

Physics equations are not used to name or include sites. The Euler PointMass
update writes velocity before position, so `Δx` depends on `ax` in the same
step; `Δx` is **not** a negative control. Negative controls are `Δvy` and `Δy`.

## Interventions (in-support, causal re-forward)

Sites are a DAG. Overriding a site **recomputes all later sites** unless
those later sites are also overridden. Frozen-downstream overwrite (keeping
cached `b1`/`b2` while changing `act`) is **not** the necessity test; that
would leave action information in cached MLP units and systematically
understate necessity of upstream sites.

- **mean-fill C**: override members of C to the **coordinatewise** training-split
  mean and recompute non-overridden sites. This is a baseline intervention,
  **not** claimed to lie on the joint activation manifold.
- **restore-only C**: override *non-members* of C to the coordinatewise
  train mean; members of C are computed live. If C is downstream, those
  units recompute under any still-silenced upstream sites (not cached
  clean activations).
- **counterfactual patch**: labeled `hybrid_activation_patch`. Pairs differ
  **only** in `ax` (or `ay` for M2), sampled from the PointMass ranges
  (`x,y ∈ [-1,1]`, `vx,vy ∈ [-0.2,0.2]`, actions `∈ [-1,1]`); override C
  to the values C took on `a'` while running `a`; recompute non-C.
  Metric: median over 64 pairs of
  `1 - ||Δvx_patch - Δvx(a')||² / ||Δvx(a) - Δvx(a')||²`.
  `a' = clip(a+0.7,-1,1)` (if that equals `a`, use `a-0.7`).
- **gauge**: orthogonal `Q` on post-nonlinearity action-embedding, block-1
  hidden, and block-2 hidden coordinates (`h' = h Q` for row-batch
  activations). Compensate the **next** linear: `W' = W Q` for
  `nn.Linear` weight `(out, in)`. Do **not** push `Q` through `tanh`
  (that would not preserve the map). Full-map MSE vs the ungauged network
  must be `<= 1e-8`. Re-run greedy on the gauged model. Literal Jaccard is
  diagnostic; functional sufficiency of some coalition is the gauge claim.

Residual-stream coordinates (`h0`, `h1`, `h2`) are **not** independently
searchable (frozen omission). They are still affected because later
computations re-forward after an override. Encoder units remain excluded.

## Frozen primary gates (M1 only, all confirmation seeds)

| Gate | Rule |
|---|---|
| competence | already passed on development for that training recipe; confirmation models trained identically |
| sufficiency | restore-only `C_hat` NMSE on original `Delta vx` `<= 0.05` |
| minimality | inclusion-minimal prune; every leftover drop-one NMSE `> 0.05` |
| necessity | mean-fill `C_hat`: `Delta vx` NMSE vs original `>= 0.10` |
| specificity | `nec_dvx / max(nec_dvy, 1e-6) >= 2.0` **and** `nec_dvx / max(nec_dy, 1e-6) >= 2.0` |
| not whole action layer | `C_hat` is not exactly `{act_0..act_5}` |
| random controls | up to 32 distinct same-size coalitions; **zero** of them sufficient |
| counterfactual | median gap closed `>= 0.50` on 64 pairs |
| gauge identity | gauged vs original output MSE `<= 1e-8` |
| gauge functional | gauged greedy+prune coalition meets sufficiency `<= 0.05` |

Not primary (reported): M2–M4, minimality/drop-one, alternate greedy 2nd-best,
cancellation pairs, magnitude / gradient top-k vs CRCT, literal Jaccard
across seeds, RMS-matched controls.

Cancellation: if no opposing pair with member ablation NMSE `>= 0.02` and
joint NMSE `<= 0.5 * min(members)`, record
`NO_MEANINGFUL_CANCELLATION_DETECTED` (not a fail).

## Status vocabulary

`MODEL_INCOMPETENT` | `LOCALIZATION_FAILED` | `MINIMALITY_FAILED` |
`NECESSITY_FAILED` | `SUFFICIENCY_FAILED` | `SPECIFICITY_FAILED` |
`MECHANISM_RECOVERY_PASSED` | `INCONCLUSIVE`

The substrate is a **supervised residual MLP** trained to predict Δ-state.
The experiment ID keeps the JEPA-program name; a pass is not a JEPA-objective
result.

Primary pass: every confirmation seed is `MECHANISM_RECOVERY_PASSED`.
Evidence level on pass: `Causal effect` (learned tiny WM, not Qwen/workspace).

## Provenance

Separate CLI invocations. Do not fuse stages.

```powershell
$env:PYTHONPATH = "src"
python -m causal_workspace_jepa.experiments.cross_domain.crct_jepa_action_delta --stage development --output artifacts/metrics/crct_jepa_action_delta_v1.dev.json
python -m causal_workspace_jepa.experiments.cross_domain.crct_jepa_action_delta --stage confirmation --output artifacts/metrics/crct_jepa_action_delta_v1.json --require-development artifacts/metrics/crct_jepa_action_delta_v1.dev.json
```

Confirmation CLI must refuse unless the development artifact has
`status=MECHANISM_RECOVERY_PASSED`, `all_seeds_passed=true`, matching
`threshold_digest`, matching `source_digest` of this module, and development
seeds `{43,47,53}`. If a development provenance sidecar exists, it must be
`--stage development` only (no fused confirmation). Seed field in a
confirmation sidecar is `1013`. Confirmation `command` includes
`--stage confirmation` and `--require-development` and must not include
`--stage development` or `&&`.

## Explicit non-actions

Do not retune epsilon after confirmation. Do not pick the best seed. Do not
rerun IBD-003/HARD-002. Do not execute Qwen 004 or planning tracks.
