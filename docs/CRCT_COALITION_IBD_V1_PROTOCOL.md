# CRCT coalition IBD V1 protocol

Experiment: `CRCT-COALITION-IBD-001`

HARD-002 remains `NEGATIVE_RESULT`. Its primary seeds `1009/2027/4093` are
constructor-blocked. This successor does not retune HARD-002 gates.

## Question

Can a prospective evaluator distinguish:

1. literal planted-graph recall,
2. epsilon-functional sufficiency,
3. multiple equivalent minimal circuits,
4. signed cancellation,
5. high-activation noncausal decoys,
6. matched-control insufficiency,
7. in-support vs out-of-support interventions,

on an interpretable-by-design concept-bottleneck plant?

## Frozen thresholds (a priori)

- epsilon = 0.02
- redundancy correlation min = 0.99
- cancellation sum-energy ratio max = 0.02
- cancellation min member energy = 1e-4
- gauge Spearman min = 0.99
- decoy causal energy max = 1e-8

Development seeds: `11, 13, 17`.
Confirmation seeds: `811, 823, 829`.

A positive evaluator result is synthetic method evidence only. It is not a
Qwen, JEPA, or workspace claim.

## Commands

```powershell
$env:PYTHONPATH = "src"
python -m causal_workspace_jepa.experiments.cross_domain.crct_coalition_ibd --stage development --output artifacts/metrics/crct_coalition_ibd_dev.json
python -m causal_workspace_jepa.experiments.cross_domain.crct_coalition_ibd --stage confirmation --output artifacts/metrics/crct_coalition_ibd_confirm.json
```

Run confirmation only after the development code/thresholds are committed.
