# CI, Reproducibility, and Repository Hygiene

## Immediate failures to resolve

Re-verify the handoff report:
- one reproducibility-related test failure;
- five Ruff issues;
- 12 untracked entries.

## Untracked artifacts

For every untracked file record:
- path;
- bytes;
- SHA-256;
- creation/source context if recoverable;
- whether it is input, source, result, scratch, or generated cache;
- whether the repo's reproducibility scanner should include it.

Never add fake sidecars.

Possible legitimate resolutions:
- commit the artifact and real provenance;
- move a scratch result outside audited paths;
- add a narrow ignore rule for known transient outputs;
- delete only if it is reproducibly generated scratch and not evidence needed for adjudication.

## CI design

CPU/non-protected GitHub Actions should include:
- install base + dev dependencies;
- `python -m pytest` on unit/integration/scientific tests that do not require GPU/protected data;
- `python -m ruff check .`;
- reproducibility/static artifact audit;
- competence entrypoint syntax regression test.

Do not:
- download Qwen weights in ordinary CI;
- open protected splits;
- run confirmation experiments automatically;
- mutate artifacts from CI.

## Entrypoint regression

The competence-recovery suite should itself execute the entrypoint syntax regression test if it is
part of its critical preflight.

The same test/source file should be included in the suite's source snapshot if the snapshot is
intended to represent all code enforcing the run contract.

## Warnings

Do not blindly suppress the reported warnings. Classify:
- benign third-party deprecation;
- repo deprecation needing repair;
- scientific numerical warning;
- resource/runtime warning.

Document any accepted warning class.
