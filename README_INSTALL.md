# Causal Workspace JEPA — Codex Orchestration Pack
Version: 2026-08-20

This pack is designed for the repository:

- `Kripta-Studios/causal-workspace-jepa`
- target branch: `crct-stage0-001`
- expected remote HEAD at handoff: `3f58ec5a25ad8374c4678f94171a18892fdc0ead`

It does **not** replace scientific results or silently modify historical protocol files.
It adds an agent harness and a task-local source of truth so Codex can repair the repository,
close the Qwen competence milestone correctly, and implement the next CRCT research stage without
contaminating protected evidence.

## How to install

1. Make sure the repository is checked out on `crct-stage0-001`.
2. Extract the contents of this pack into the **repository root**.
3. Do not overwrite the existing `AGENTS.md` with `AGENTS_CANDIDATE.md` manually.
   Codex must migrate it only after checking that no binding instruction is lost.
4. Restart Codex so project-local `.codex/skills` and agent configuration are reloaded.
5. Open a **new Codex session** in the repository root.
6. Start with **GPT-5.6 Sol / High** for the first planning and integrity pass.
7. Paste the contents of `CODEX_MASTER_PROMPT.md`.

## Why a new session

The branch has moved through several scientific adjudications. A fresh session avoids stale
assumptions such as “V3 is still pending” or “competence recovery has not run”. The repository and
the files in `docs/codex-handoff/` must be the system of record.

## What this pack intentionally does not do

- It does not open protected Qwen test/paraphrase splits.
- It does not retune HARD-002 thresholds.
- It does not reinterpret `INELIGIBLE_TASK_PHASE0` as a positive V3 result.
- It does not fabricate provenance for untracked JSON files.
- It does not authorize a Steerling-8B run on insufficient hardware.
- It does not authorize parallel write agents in one checkout.

See `docs/codex-handoff/00_INDEX.md`.
