# CRCT-LEARNED-WM-ACTION-DELTA-003 protocol

Status at freeze: `PREREGISTERED_NOT_RUN`.
After the freeze commit, execution of **this ID only** is authorized.

Parent: `CRCT-LEARNED-WM-ACTION-DELTA-002` (`INCONCLUSIVE`, evidence `None`).
002 is not mutated, not rerun, and seed 59 is **not** a retrospective pass.
001 remains `MODEL_INCOMPETENT`. IBD-003 remains `MECHANISM_RECOVERY_PASSED`
(synthetic IBD). HARD-002 remains `NEGATIVE_RESULT`. IBD-002 was not executed.

The draft `docs/research/CRCT_LEARNED_WM_ACTION_DELTA_003_DRAFT.md` is not
this freeze. That draft asked whether action-only coalitions “should be
allowed.” This protocol does **not** ask that question.

## Nomenclature

Supervised residual-MLP PointMass predictor. **Not** a JEPA objective.
A pass is not “we interpreted a JEPA.”

## Question

When label-blind CRCT finds a sparse action-embedding coalition for
learned `ax → Δvx`, is that coalition:

1. an **information carrier / gateway** (Level 1),
2. a **causal mediator** (Level 2), or
3. part of a resolved **computational pathway** including skip vs residual
   routes (Level 3)?

A mechanism-recovery pass cannot be based solely on Level 1.

## What this is not

Not an automatic deletion of 002’s `ARCHITECTURE_CUTSET` rule.
Action-stem coalitions are **eligible to be tested**, not declared mechanisms.

## Competing hypotheses (not privileged)

| ID | Claim |
|---|---|
| `H_GATEWAY` | Action-stem units carry `ax` but do not characterize target-specific computation. |
| `H_DIRECT` | Sparse action-stem + skip path implements most of `ax → Δvx`; residual branches are not required for frozen counterfactual mediation. |
| `H_DISTRIBUTED` | Action-stem initiates; residual-branch computation is additionally required for frozen mediation. |
| `H_EQUIVALENT` | Cross-seed: different literal sites, same frozen pathway class. Assigned only after ≥2 seeds independently pass pathway gates. |

## Substrate

Same `ActionDeltaPredictor` family as 001/002: `H=6`, Adam `lr=3e-3`,
batch 64, pooled MSE, **800** Adam steps (002’s selected competence rung;
not re-climbed). New seeds. Competence conjunction unchanged: all four
Δ-channel NMSE `<= 0.05` vs physics on that seed’s development eval.

If any development seed is incompetent: experiment `MODEL_INCOMPETENT`,
CRCT skipped on that seed, confirmation **CLOSED**. No extra rungs.

## Seeds (new; disjoint)

Forbidden: all 002 forbidden seeds plus 002 model seeds
`59,71,73,1031,1033,1039`.

- Development: `79, 83, 89`
- Confirmation: `1049, 1051, 1061`

Per seed `S` (same split offsets as 001/002):

```text
train:    S*1000+61   256 traj × 6 steps
dev:      S*1000+67   64 traj
confirm:  S*1000+71   64 traj  (confirmation stage only)
```

## Physics / specificity ontology

See `docs/CRCT_LEARNED_WM_ACTION_DELTA_003_PHYSICS_GRAPH_2026-08-27.md`.

For `ax → Δvx`:

- direct target: `Δvx`
- downstream of ax (diagnostic, **not** a negative control): `Δx`
- independent negative controls: `Δvy`, `Δy`

Specificity gate uses **only** independent controls:
`nec_Δvx / nec_Δvy >= 2` and `nec_Δvx / nec_Δy >= 2`.
Do **not** require a ratio vs `Δx`. Persist the full 4×4 necessity matrix.

## Component ontology

Searchable CRCT sites (unchanged): `act_0..5`, `b1_0..5`, `b2_0..5`.
Encoder excluded. Residual stream `h0`/`h1` are **path-intervention**
sites, not localization sites.

Selector: 001/002 label-blind greedy restore + inclusion-minimal prune on
the model’s own `Δvx`. Mean-fill = coordinatewise baseline.

## MSRS vs MCP

| Term | Definition |
|---|---|
| **MSRS** | Minimal sufficient restoration set: pruned greedy coalition restoring model `Δvx` to NMSE `<= 0.05` under mean-fill of all other searchable sites. |
| **MCP** | Mechanistic causal pathway: MSRS plus the frozen path class (`DIRECT` / `DISTRIBUTED` / `INTERACTING` / `REDUNDANT_ROUTES`) that transmits counterfactual `ax` effects. |

An MSRS is not automatically the complete circuit. Persist both.

Action-carrier set: searchable `act_*` members of the MSRS (may be empty).
Downstream set: `b1_*`/`b2_*` members of the MSRS. If the MSRS is
action-only, a **diagnostic** downstream-only greedy restore (act sites
forbidden) is recorded; it is not a second primary claim.

## Path interventions (paired A/B, state and `ay` fixed, `ax` differs)

1. `G_full`: patch MSRS from B into A; re-forward.
2. `G_skip`: patch MSRS; hold residual-branch activations `hid1`,`hid2` at A.
3. `G_res`: patch MSRS; hold only `skip1=h0_A` (direct mix skip into
   block 1). Residual branches recompute; block 2 sees residual-updated `h1`.

Same pair generator and median gap-closed statistic as 002
(`counterfactual_gap_min = 0.50`).

Path class (only if `G_full >= 0.50`):

| class | rule |
|---|---|
| `DIRECT` | `G_skip >= 0.50` and `G_res < 0.50` |
| `DISTRIBUTED` | `G_skip < 0.50` and `G_res >= 0.50` |
| `REDUNDANT_ROUTES` | `G_skip >= 0.50` and `G_res >= 0.50` |
| `INTERACTING` | `G_skip < 0.50` and `G_res < 0.50` |

`DIRECT` is a valid mechanism. Do not call it a failure because it is simple.

## Information vs causal controls

Fit a **train-split** linear probe `ax ~ site` per searchable site.
Report `R^2`. Diagnostic only; not a pass gate.

Information localization baseline: top-`k` sites by probe `R^2`, `k=|MSRS|`.
Apply the **same** necessity, sufficiency, and `G_full` tests.

Also evaluate the full action stem `{act_*}` as a generic cut-set
(necessity/specificity). A generic gateway is expected to damage `Δvx`
and `Δvy` similarly.

`INFORMATION_GATEWAY_ONLY` if the MSRS (or probe top-k) decodes `ax` but
fails the frozen causal conjunction (sufficiency or necessity or
independent-control specificity or `G_full`).

## Other retained tests

Magnitude / gradient / act×grad top-k; random same-size coalitions (0
sufficient allowed); cancellation (may be `NO_MEANINGFUL_CANCELLATION_DETECTED`);
post-tanh gauge `h'=hQ`, `W'=WQ` with function MSE `<= 1e-8`. Gauge
records literal Jaccard **and** whether sufficiency, necessity, `G_full`,
and path class are preserved. Literal-unit invariance is **not** required.

## Per-seed status (first matching rule)

1. competence fail → `MODEL_INCOMPETENT`
2. empty MSRS → `LOCALIZATION_FAILED`
3. restore NMSE `Δvx` `> 0.05` → `SUFFICIENCY_FAILED`
4. not inclusion-minimal → `MINIMALITY_FAILED`
5. necessity `Δvx` `< 0.10` → `NECESSITY_FAILED`
6. independent specificity fail → `SPECIFICITY_FAILED`
7. random-control sufficient count `> 0` → `INCONCLUSIVE`
8. `G_full < 0.50` and probe top-k `R^2` mean `>=` MSRS mean `R^2` →
   `INFORMATION_GATEWAY_ONLY`
9. `G_full < 0.50` otherwise → `INCONCLUSIVE`
10. gauge function MSE `> 1e-8` → `INCONCLUSIVE`
11. gauged MSRS restore NMSE `> 0.05` or gauged `G_full < 0.50` →
    `INCONCLUSIVE` (literal identity may change; functional mediation must
    survive)
12. otherwise → `PATH_MECHANISM_RECOVERY_PASSED` with a path class
    (`DIRECT` / `DISTRIBUTED` / `REDUNDANT_ROUTES` / `INTERACTING`)

`MEDIATOR_FOUND_PATH_UNRESOLVED` is reserved. It is **not** assigned: when
`G_full >= 0.50` the path classifier always returns one of the four classes
above. `INTERACTING` means neither isolated route meets the CF bar; the
joint skip+residual computation is the MCP. That is still a pass.

There is **no** `ARCHITECTURE_CUTSET` automatic fail.

Per-seed hypothesis label (descriptive, after status):

- `H_GATEWAY` if status is `INFORMATION_GATEWAY_ONLY`
- `H_DIRECT` if passed and path class `DIRECT`
- `H_DISTRIBUTED` if passed and path class in `{DISTRIBUTED, INTERACTING}`
- `H_DIRECT` if passed and `REDUNDANT_ROUTES` (skip sufficient; residual
  also sufficient) — residual redundancy is recorded, not a fail
- else `H_UNASSIGNED`

## Experiment-level status

- any development seed incompetent → `MODEL_INCOMPETENT`
- all development seeds `PATH_MECHANISM_RECOVERY_PASSED` → that status
- else if all remaining interpreted seeds share one failure status → that
  status
- else `INCONCLUSIVE`

Confirmation opens **only** if development status is
`PATH_MECHANISM_RECOVERY_PASSED` and `all_seeds_passed` is true.
Confirmation trains the frozen 800 steps. Incompetent confirmation seed:
no CRCT; experiment `MODEL_INCOMPETENT_CONFIRMATION`.

`H_EQUIVALENT` is recorded only if ≥2 passing seeds have **different**
literal MSRS (Jaccard `< 1`) and the **same** path class.

## Evidence

`PATH_MECHANISM_RECOVERY_PASSED` → evidence `Causal effect`.
Otherwise `None`.

## Provenance

Separate development and confirmation CLIs. Sidecar seed is the first
development seed (`79`) or first confirmation seed (`1049`).
`source_digest` hashes 001 + 002 + 003 modules.

```text
python -m causal_workspace_jepa.experiments.cross_domain.crct_learned_wm_action_delta_003 --stage development --output artifacts/metrics/crct_learned_wm_action_delta_v3.dev.json
python -m causal_workspace_jepa.experiments.cross_domain.crct_learned_wm_action_delta_003 --stage confirmation --require-development artifacts/metrics/crct_learned_wm_action_delta_v3.dev.json --output artifacts/metrics/crct_learned_wm_action_delta_v3.json
```

## Explicit non-actions

Do not mutate 001/002. Do not reuse 002 seeds. Do not execute JEPA
objective, friction, MiniPush, planning, stitching, Qwen 004, or IBD-002.
Do not select only residual-inclusive seeds after outcomes.
