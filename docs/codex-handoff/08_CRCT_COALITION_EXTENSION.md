# CRCT Coalition / Equivalence Extension

HARD-002 exposed an ontology mismatch: a small selected circuit can reconstruct >99% of effect while
omitting planted nodes that are redundant, cancelling, or functionally substitutable.

A successor must not collapse “true circuit” into one node set.

## Proposed objects

Let a candidate circuit `C` act on the frozen system and let `E(C)` be intervention-effect error
relative to the full target mechanism.

Measure:

### Literal recall
Recovery of the full planted generative graph.

### Epsilon-functional sufficiency
`C` is sufficient if `E(C) <= epsilon`.

### Minimality
`C` is epsilon-minimal if it is sufficient and removing any selected atomic/group element violates
the sufficiency threshold.

### Necessity
For component/group `g`, measure degradation after ablating `g` from an otherwise sufficient
circuit. Report effect size and uncertainty, not only a boolean.

### Redundancy groups
A group contains mechanisms where one of several members can substitute for another within epsilon.

### Cancellation groups
Members have material signed effects that cancel in aggregate. Absolute attribution must not hide
this.

### Equivalence classes
Two circuits are equivalent under the frozen metric when their finite intervention behavior is
indistinguishable within epsilon across the required IID/OOD evaluation distribution.

## Positive control

Build a tiny transformer/concept-bottleneck plant with:
- explicit known, unknown, residual factors;
- redundant routes;
- cancellation;
- at least two equivalent minimal circuits;
- decoys with high activation magnitude;
- gauge/basis transformations that preserve function.

The test should prove that:
- exhaustive graph recall is distinguishable from functional sufficiency;
- the evaluator recognizes multiple equivalent minimal circuits;
- activation magnitude can be fooled without fooling causal criteria;
- signed cancellation is visible.

## Temporal separation

1. develop metrics/selectors on development plants;
2. freeze code/config/thresholds;
3. choose fresh primary seeds before generation;
4. commit freeze;
5. generate confirmation plants;
6. run once and adjudicate.

Do not reuse HARD-002 primary seeds for confirmation of the redesigned metric.
