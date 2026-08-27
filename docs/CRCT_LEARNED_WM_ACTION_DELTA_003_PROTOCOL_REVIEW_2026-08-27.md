# Independent protocol review — CRCT-LEARNED-WM-ACTION-DELTA-003

Pre-freeze. No 003 training outcomes. 002 is not mutated. Seed 59 is not a
pass.

## Verdict

**Ready to freeze** after P0 repairs below. No remaining P0.

## P0 repairs made before freeze

1. Residual-only hold no longer freezes `skip2=h1_A` while `hid2` still
   depends on a different `h1`. `G_res` holds only `skip1=h0_A`.
2. Gauge records are pass-relevant: gauged MSRS sufficiency and gauged
   `G_full` must meet the same frozen bars. Literal Jaccard is not a gate.
3. The superseded draft’s “allow action-only” question is not the freeze
   question. Action-stem coalitions are eligible for pathway tests, not
   declared mechanisms.

## Explicit checks

| Question | Finding |
|---|---|
| PointMass dependency graph? | Correct for implemented Euler (`Δx` downstream of `ax`; `Δy` independent). |
| `Δy` valid negative control? | Yes, for `ax → Δvx`. |
| 800 steps outcome-independent for 003? | Yes. Parent 002 already selected 800 before 003 existed. New seeds still face the competence gate. No extra rungs. |
| Removing `ARCHITECTURE_CUTSET` converts seed 59? | No. New seeds `79/83/89`. 002 artifact remains `INCONCLUSIVE`. |
| Path holds meaningful? | Skip-only freezes residual branches; residual-only freezes the mix skip into block 1. Unit-tested. |
| MSRS vs MCP operational? | Persisted separately; path class is the MCP annotation. |
| Label leakage? | Selector is 001/002 greedy on the model’s `Δvx`, not unit names. |
| Confirmation conservative? | Requires development `PATH_MECHANISM_RECOVERY_PASSED` and `all_seeds_passed`. |
| Post-hoc numeric gates? | Path classes reuse `counterfactual_gap_min=0.50`. No 0.70 ratio. |

## What a pass may claim

On new competent supervised PointMass residual MLPs, label-blind CRCT
identified a target-specific causal pathway for `ax → Δvx`, distinguished
from ax-decodability by conditional skip/residual holds and independent
controls.

## What a pass may not claim

JEPA; Qwen; workspace; Platonic physics; planning; MiniPush; 002 seed 59
pass; generic mechanistic interpretability.
