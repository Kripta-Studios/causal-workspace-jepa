# Acceptance Gates

The branch is not “done” unless every applicable gate is satisfied.

## Engineering gate

- full non-protected test suite: pass;
- focused Qwen suite: pass;
- focused CRCT scientific suite: pass;
- Ruff: zero errors;
- reproducibility audit: pass without fabricated provenance;
- no unexplained untracked scientific artifacts;
- CI configured and green for CPU/non-protected checks;
- no accidental protected data in Git history/diff.

## Qwen competence-development gate

Only adjudicate the supplied development result if:
- exact report exists;
- source snapshot exists;
- manifest/hashes verify;
- access ledger shows calibration-only model forwards;
- selected renderer and metrics match the artifact;
- protected split list is empty.

## Fresh Qwen confirmation gate

- prospective protocol committed before execution;
- fresh examples generated only after freeze;
- clean full-vocabulary accuracy >= 0.90;
- direct-permuted full-vocabulary accuracy >= 0.90;
- candidate-only metric not used for eligibility;
- no test/paraphrase access;
- outcome adjudicated without threshold changes.

Failure means the path closes negative; it does not trigger a new prompt search on confirmation data.

## Coalition CRCT gate

A successor positive claim must report all of:
- literal graph recovery;
- epsilon-functional sufficiency;
- minimality;
- necessity;
- redundancy coverage;
- cancellation coverage;
- equivalence-class correctness;
- matched-control specificity;
- gauge stability;
- IID/OOD confirmation;
- intervention-support validity.

Development and confirmation seeds must be disjoint and temporally separated by the freeze commit.

## Learned residual gate

Do not promote learned residual modeling unless:
- residual power beyond strong differential baselines exceeds the frozen eligibility floor;
- the signal is stable across independent examples/seeds;
- differential+residual beats direct-delta and other fair baselines on frozen held-out metrics;
- result survives original-model intervention replay.
