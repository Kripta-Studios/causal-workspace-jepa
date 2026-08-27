# Adversarial review — CRCT-LEARNED-WM-ACTION-DELTA-003 (pre-freeze)

## Verdict

**Freeze allowed.** Attacks that survive are fail-closed outcomes, not silent
passes. Seed 59 is not converted into a 003 pass.

## Attacks

| # | Attack | Result |
|---|---|---|
| 1 | Relax 002 `ARCHITECTURE_CUTSET` after seed 59 | Blocked. New ID, new seeds, new question (gateway vs pathway). 002 unchanged. |
| 2 | `Δx` as negative control | Blocked. Downstream diagnostic only. |
| 3 | Misuse `Δy` | Blocked. `Δy` is an independent control; 002 seed 73 remains a valid spec fail. |
| 4 | Level 1 as pass | Blocked. Probe `R^2` is diagnostic; pass requires CF + spec + path class. |
| 5 | MSRS = complete circuit | Blocked. MSRS and MCP persisted separately. |
| 6 | Path holds not isolating | P0 repaired: `G_res` holds only `skip1`. Sequential caveat documented. |
| 7 | Invalid gauge | Same post-tanh `W'=WQ` as 001/002; function MSE gated; functional CF gated. |
| 8 | Privilege `H_DIRECT` | Path class uses the same 0.50 bar on skip vs residual. `REDUNDANT_ROUTES` recorded. |
| 9 | Reuse 002 seeds | Forbidden set includes 002 seeds. |
| 10 | JEPA naming | Claim boundary forbids it. |
| 11 | Confirmation on mixed development | Requires `PATH_MECHANISM_RECOVERY_PASSED` and `all_seeds_passed`. |
| 12 | Post-hoc path thresholds | Reuses frozen CF bar 0.50. |
| 13 | Residual-stream search | `h0` is path-only, not in greedy search. |
| 14 | Full `{act_*}` as pass | max_coalition 4; full stem is a diagnostic cut-set, not the MSRS selector. |
| 15 | Select residual-inclusive seeds | All three development seeds are run; no post-hoc subsetting. |

## Strongest honest pass claim

In a competent learned supervised PointMass world model, label-blind CRCT
identified a target-specific causal pathway mediating `ax → Δvx`,
distinguished from mere ax information availability by conditional
interventions and matched controls. If that pathway is skip-dominated, say
so; do not downgrade it for simplicity.

## Later independent verdict (after freeze)

Independent reviewers returned **NO-FREEZE** after `a23bbaa` /
`5219a9f`. See
`docs/CRCT_LEARNED_WM_ACTION_DELTA_003_INDEPENDENT_REVIEW_POST_FREEZE_2026-08-27.md`.
This file is not rewritten to look pre-outcome.
