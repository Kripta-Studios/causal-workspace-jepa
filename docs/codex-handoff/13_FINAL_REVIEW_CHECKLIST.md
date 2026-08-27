# Final Adversarial Review Checklist

A Sol High reviewer should answer each item with PASS / FAIL / NOT APPLICABLE and evidence.

## Git / provenance
- Is the final branch/HEAD unambiguous?
- Are preregistration commits temporally before outcomes?
- Are remote provenance claims actually pushed when required?
- Are there unexplained untracked scientific artifacts?
- Is every committed result traceable to config/source/run evidence?

## Data leakage
- Any protected test/paraphrase access?
- Any prompt selected using confirmation outcomes?
- Any threshold/layer/seed changed after seeing the governed result?
- Any training examples derived from validation/test outputs?
- Any accidental model-forward outside allowed split ledger?

## Scientific claims
- Does HARD-002 remain negative?
- Does V3 remain ineligible?
- Is competence recovery labeled development-only?
- Is confirmation genuinely fresh?
- Are candidate-only metrics kept diagnostic?
- Are direct-delta baselines retained?
- Are graph recall and functional sufficiency separated?
- Are cancellation and redundancy treated explicitly?
- Are workspace/SOTA claims still appropriately closed?

## Engineering
- full non-protected tests green?
- Ruff green?
- reproducibility audit green?
- CI green?
- failure-path tests present?
- entrypoint regression included?
- docs synchronized?

## Review independence
The final reviewer should inspect raw artifacts and diff before reading the implementer's narrative
where possible. This reduces confirmation bias.
