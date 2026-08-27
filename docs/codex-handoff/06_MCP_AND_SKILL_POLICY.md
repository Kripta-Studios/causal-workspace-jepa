# MCP and Skill Policy

## MCPs

### Tier 1 — use by default when relevant

**GitHub**
- commits, compare, branches, PRs, Actions, review comments, upstream code;
- use it to verify remote provenance rather than relying on copied text.

**Context7**
- current library APIs and migration details;
- especially PyTorch, Transformers, pytest, Ruff, h5py, safetensors.

**Hugging Face**
- exact Qwen model/tokenizer revision, config, tokenizer behavior, model cards;
- pin revisions in scientific protocols.

**DeepWiki**
- inspect upstream repository architecture and implementation details;
- useful for Qwen/Transformers, LeVLJEPA, Steerling, and interpretability code.

**Exa**
- current papers, primary technical sources, related work;
- research only; do not let literature search change frozen gates after outcome access.

### Tier 2 — conditional

**W&B**
- development experiment tracking only if deliberately added to a new prospective protocol;
- do not retrofit into an already-frozen confirmation run;
- local artifacts remain canonical unless the protocol explicitly says otherwise.

**OpenAI Developer Docs**
- Codex, model routing, MCP, skills, agent configuration;
- not needed for scientific claims about Qwen/CRCT.

### Usually skip

**Sentry**
- useful for deployed service/runtime debugging, not a priority for this research-repo recovery.

## Existing skills

Use if installed:
- `superpowers` — decomposition, test-first work, debugging.
- `sol-advisor` — model routing advisory.
- `gh-fix-ci` — after GitHub Actions exists/fails.
- `gh-address-comments` — PR review feedback.
- `stop-slop` — final cleanup after correctness.
- `frontend-design` — skip for this task.

## Project-local skills supplied by this pack

- `$cwj-orchestrate`
- `$cwj-scientific-integrity`
- `$cwj-repro-hygiene`
- `$cwj-qwen-confirmation`
- `$cwj-crct-coalitions`
- `$cwj-final-review`

Use the smallest set needed for each subtask.
