# AGENTS.md — candidate compact map

> Do not replace the current root `AGENTS.md` until every binding rule in it has been inventoried and
> preserved. This file demonstrates the desired compact shape.

## Mission

Maintain a reproducible research codebase for causal/mechanistic JEPA and LLM intervention studies.
Scientific integrity outranks getting a positive result.

## Start here

Read:
- `docs/codex-handoff/00_INDEX.md` for the current recovery task;
- `docs/REPRODUCIBILITY.md`;
- `docs/EXPERIMENT_REGISTRY.md`;
- `docs/RESULTS.md`;
- the protocol/result document for the experiment you are touching.

## Non-negotiable rules

- Never change frozen thresholds after outcome access.
- Never reuse development outcomes as confirmatory evidence.
- Never open protected splits unless a prospective protocol explicitly authorizes them.
- Never fabricate provenance.
- Preserve historical negative/ineligible results.
- Commit preregistrations before governed outcome access.
- Keep direct-delta and strong differential baselines in residual-learning claims.
- Separate literal graph recovery from functional sufficiency/equivalence.
- No concurrent write agents in the same checkout.

## Engineering

Before finalizing any change:
- run the relevant tests;
- run Ruff;
- run reproducibility checks;
- inspect Git diff/status/untracked files;
- synchronize affected docs;
- request an independent review for scientific changes.

## Plans

Complex work must use a versioned execution plan with progress, decision log, commands, tests, and
artifact hashes. Do not rely on chat history as the source of truth.
