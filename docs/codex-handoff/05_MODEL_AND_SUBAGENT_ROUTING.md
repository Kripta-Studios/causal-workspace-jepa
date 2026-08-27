# Model and Subagent Routing

## Recommended mode for this recovery: safety-first

Because this branch contains protected evaluation boundaries and frozen negative results, start the
root session on **GPT-5.6 Sol / High**.

Preferred subagents:

| role | model | effort | permissions | purpose |
|---|---|---|---|---|
| `sol_architect` | GPT-5.6 Sol | High | read-only | protocol/architecture/leakage analysis |
| `sol_verifier` | GPT-5.6 Sol | Medium | read-only | repo survey, tests, consistency |
| `luna_worker` | GPT-5.6 Luna | Max | workspace-write | bounded implementation |
| `luna_test_worker` | GPT-5.6 Luna | Max | workspace-write | tests, CI, hygiene |
| `sol_reviewer` | GPT-5.6 Sol | High | read-only | adversarial final review |

## Why Sol High is the parent initially

If a current Codex build ignores a child model/effort override, the child may inherit the parent.
For this scientific recovery, inheriting Sol High is safer than unexpectedly downgrading a critical
review to a cheaper model.

After the repository is clean and the protocol is frozen, routine coordination can move to
Sol Medium to reduce cost.

## Routing smoke

Before relying on role routing:

1. Spawn `sol_verifier` on a trivial read-only task.
2. Spawn `luna_worker` on a no-write diagnostic task.
3. Inspect whatever model/effort metadata the current Codex UI/tool exposes.
4. If model/effort cannot be observed, mark routing `UNVERIFIED`.
5. If overrides are ignored, do one of:
   - keep the root on Sol High and accept inheritance;
   - use separate Codex tasks/worktrees with the desired model selected manually.

Never claim that a Luna Max worker ran if the runtime does not expose evidence.

## Parallelism rules

Parallelize:
- repository exploration;
- independent protocol review;
- documentation inconsistency scans;
- static test/lint triage.

Do **not** parallelize writes into the same checkout.

For parallel implementation, use isolated Codex app worktrees/tasks. A subagent prompt telling a
writer to “stay in directory X” is not a security/integrity boundary.

Recommended concurrency: at most 4 active threads for this repo. More agents increase merge and
context risk without clear benefit.

## Task sizing

A Luna worker gets:
- one bounded goal;
- explicit files/directories;
- acceptance tests;
- prohibited files/splits;
- no authority to change preregistered thresholds.

A Sol reviewer gets:
- the diff;
- frozen protocol/result docs;
- raw test output;
- no implementer rationale unless required.
