---
name: cwj-orchestrate
description: Orchestrate complex work in causal-workspace-jepa with bounded subagents, model routing, scientific gates, versioned execution plans, and independent review. Use for multi-step repo recovery, experiment implementation, CI/provenance repair, Qwen/CRCT milestones, or any task spanning multiple subsystems.
---
Read `../../../docs/codex-handoff/00_INDEX.md`, then maintain one authoritative execution plan.

Use read-only subagents in parallel for discovery. Assign write work narrowly and sequentially unless
isolated Codex worktrees are used.

For every task:
1. establish current Git/test/lint/provenance state;
2. identify frozen scientific boundaries;
3. delegate bounded analyses;
4. integrate evidence before editing;
5. implement test-first where practical;
6. run the acceptance gate;
7. request independent Sol High review.

Never let a worker change preregistered thresholds, protected-split policy, or historical result
status. Use `$cwj-scientific-integrity` whenever experiment semantics are involved.
