# Documentation Synchronization

The repository's documentation must describe the actual latest scientific state.

Synchronize at least:
- `README.md`;
- `SUMMARY.md`;
- `docs/RESULTS.md`;
- `docs/ROADMAP.md`;
- `docs/TODO.md`;
- `docs/EXPERIMENT_REGISTRY.md`;
- `docs/DECISIONS.md`;
- `docs/REPRODUCIBILITY.md`;
- `docs/LLM_TRACK.md`;
- CRCT/Qwen protocol/result docs affected by new milestones.

## Required statements

Global docs must clearly preserve:
- Stage0 basic = synthetic positive control only;
- HARD-002 = frozen negative scientific result;
- V3 B0 = `INELIGIBLE_TASK_PHASE0`;
- competence recovery = calibration-only development selection if verified;
- fresh confirmation = separate new milestone;
- direct-delta currently beats residual learner on HARD-002;
- coalition/equivalence redesign is prospective;
- workspace claim remains closed;
- LeVLJEPA is secondary;
- Steerling-8B execution is resource-blocked on the current local profile.

## AGENTS.md migration

The current repository has a very large root `AGENTS.md`. Large always-injected instruction files
waste context and become stale.

Do not overwrite it mechanically.

Instead:
1. inventory every binding rule;
2. move stable detail into indexed docs;
3. keep root `AGENTS.md` as a short navigation map plus non-negotiable commands;
4. add links to deeper instructions;
5. verify no rule was lost;
6. add a structural/doc test if practical.

Use `AGENTS_CANDIDATE.md` only as a target shape.
