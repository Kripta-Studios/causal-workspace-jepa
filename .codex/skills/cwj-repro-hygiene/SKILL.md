---
name: cwj-repro-hygiene
description: Repair tests, Ruff, CI, untracked artifacts, provenance, source snapshots, and reproducibility audits in causal-workspace-jepa without falsifying scientific metadata. Use for repository hygiene, CI failures, artifact audits, and pre-experiment cleanup.
---
Read `../../../docs/codex-handoff/10_CI_REPRO_HYGIENE.md`.

Inventory before deleting or ignoring anything. For each suspicious artifact, prefer evidence-backed
classification over making the audit green.

Run:
- targeted failing test first;
- focused subsystem suite;
- full non-protected suite;
- Ruff;
- reproducibility audit.

When a regression previously broke an experiment entrypoint, ensure the regression test is part of
the experiment runner's own preflight and source snapshot where appropriate.
