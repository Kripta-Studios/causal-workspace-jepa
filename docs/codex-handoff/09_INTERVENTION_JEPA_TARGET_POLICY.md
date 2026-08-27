# Intervention-JEPA Target Policy

## Current evidence constraint

HARD-002 did not justify privileging a learned residual target. Its direct-delta MLP was better than
the residual MLP across all primary seeds IID and OOD.

Therefore architecture preference must not be mistaken for evidence.

## Required baseline family

Every relevant experiment should compare, under matched data/capacity/evaluation:

- zero/simple baseline where meaningful;
- JVP/first-order;
- T2/quadratic;
- relinearized JVP;
- capacity-matched direct-delta predictor;
- differential + learned residual;
- any proposed known/unknown/residual decomposition.

## Residual eligibility

Before training a residual learner for a confirmatory claim:

1. measure finite-effect power unexplained by the strongest frozen differential baseline;
2. require a preregistered minimum residual-power floor;
3. show that the residual is stable and predictable;
4. show differential+residual beats direct-delta on frozen held-out evaluation.

## Decomposed targets

A `known / unknown / residual` decomposition may be implemented as a development track if it adds:

- signed reconstruction;
- explicit residual budget;
- component/group support labels;
- no target leakage from protected outcomes;
- direct original-model intervention replay;
- fair direct-delta control;
- out-of-support intervention diagnostics.

Exact algebraic reconstruction by definition is not causal evidence by itself.
