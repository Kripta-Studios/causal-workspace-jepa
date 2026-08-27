# Adversarial review — CRCT-JEPA-ACTION-DELTA-001 (pre-freeze)

Reviewer role: try to break the claim *before* outcomes exist.
This is not permission to execute confirmation or retune gates.

## Verdict

**No remaining P0.** Freeze is allowed. Residual risks are frozen P1s, not
license to mutate the protocol after inspection of trained seeds.

## Attack: circular mechanism

Explaining a network’s own `Δvx` by turning on units that compute `Δvx` is
vacuous if any large-enough set passes. The protocol blocks that with
max size 6, greedy min-step, random same-size plus-one p, RMS-matched
controls (reported), and a counterfactual patch that must move `Δvx`
toward `ax=a'` while holding other inputs fixed. Competence vs physics is
a separate fail-closed gate, so a mechanism pass cannot be claimed for an
incompetent interpolator.

## Attack: physics leakage into the circuit

The selector never receives planted node names or the PointMass equations.
Site names (`act_*`, `b1_*`, `b2_*`) are not features of the search; every
remaining site is scored. External physics defines the *function*
`ax → Δvx`, not membership of C.

## Attack: confirmation leakage / best-seed selection

Development seeds `{43,47,53}` are disjoint from confirmation
`{1013,1019,1021}`. Circuit freeze uses `S*1000+67`. Confirmation
trajectories `S*1000+71` are generated only in the confirmation stage.
CLI requires a development artifact with `MECHANISM_RECOVERY_PASSED`.
Primary pass requires **all** confirmation seeds; no post-hoc best-seed
rule.

## Attack: invalid patching

Mean-ablate and restore-only use train-split means (in-support).
Counterfactual overrides copy activations from a real `a'` forward.
Gaussian fills are not a pass token. Re-forward is required so upstream
ablation cannot hide in cached downstream activations.

## Attack: gauge tautology

The allowed transform is an invertible mix of hidden coordinates with a
compensating next linear. It is not an output-energy rescaling. A
function-preservation MSE gate is explicit. Literal unit Jaccard may
drop; that is diagnostic.

## Attack: attribution baselines already solve it

If magnitude or gradient top-k also satisfy necessity/sufficiency, the
result must report that. Baselines are not decoyed. That would weaken a
“CRCT-specific discovery” story without invalidating a causal coalition
claim.

## Attack: skip connections make necessity impossible

Possible. Residual skips can carry `mix(z,e)` around silenced MLP units.
If greedy returns a coalition that is sufficient but not necessary, the
frozen status is `NECESSITY_FAILED`, not a silent pass. That is an
acceptable negative.

## Attack: specificity is impossible because ax/ay share the embedding

Possible. Then `SPECIFICITY_FAILED`. Do not lower the ratio after seeing
confirmation.

## P1 accepted

- Encoder and residual-stream coordinates are not in the search set.
- Cancellation may be `NO_MEANINGFUL_CANCELLATION_DETECTED`.
- Cross-seed literal overlap is reported, not required.
- Model is tiny CPU MLP, not a JEPA trained at paper scale.

## Second pass

Independent adversarial review required NO-FREEZE until minimality, action-layer
cut-set, min-step formula, and confirmation binding were repaired. Those repairs
are now in the protocol/code. Remaining attacks (skip connections, shared
ax/ay embedding, omitted encoder) can still produce a frozen negative
(`NECESSITY_FAILED` / `SPECIFICITY_FAILED` / `LOCALIZATION_FAILED`). That is
acceptable.

Revised freeze verdict: **freeze allowed**. Do not mutate gates after trained
seeds are opened.

## Must not claim even on pass

Learned Qwen circuits; workspace; Platonic physics; planning; MiniPush
contact; rescue of HARD-002; that IBD-002 ran; that IBD-003 was a learned
network.

