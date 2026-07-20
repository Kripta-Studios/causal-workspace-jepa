# Approach Registry

| Family | Owner | Question | Evidence | Blockers | Next Experiment | Status |
| --- | --- | --- | --- | --- | --- | --- |
| Architecture/interfaces | root agent | Can the project run reproducibly in CPU mode? | Resource guard, typed interfaces, tests, and audits pass. | None for CPU smoke. | Keep audit green. | `SMOKE_VALIDATED` |
| Tier 0 dynamics | root agent | Can deterministic simulators expose ground truth? | Five deterministic generators and manifests. | Scientific scale. | Add OOD parameter splits. | `SMOKE_VALIDATED` |
| Tiny JEPA | root agent | Does action conditioning improve toy prediction? | Conditioned model beats no-action and shuffled controls. | Linear/random encoder limits mechanism claims. | Multi-consumer audit. | `SMOKE_VALIDATED` |
| Intervention primitives | root agent | Are interventions serializable and replayable? | Operators and replay tests pass. | Module-specific adapters remain. | Use direct replay in every causal experiment. | `SMOKE_VALIDATED` |
| Workspace discovery | root agent | Can a compact shared causal subspace be distinguished from private channels and PCA? | Detector controls pass; tiny JEPA candidate fails uncertainty validity and PCA specificity. | Random rollout controls are off-manifold; model is too shallow. | Add calibrated uncertainty and manifold-matched controls. | `ACTIVE` |
| Mock Qwen | root agent | Can known activation dynamics validate LLM pipeline? | Mock intervention-JEPA smoke passes. | Not evidence about a real LLM. | Add stronger baselines. | `SMOKE_VALIDATED` |
| GPT-2 Medium | root agent | Can intervention effects be predicted on a real transformer? | Four-prompt causal smoke exists; 288-outcome follow-up is preregistered and implemented. | CPU runtime; one seed and selected outputs. | Commit, then run `LLM-GPT2-002`. | `IMPLEMENTED_UNVALIDATED` |
| Published JEPA adapters | future GPU agent | Can published world models support the same audit? | None on VPS. | Hardware/data/weights. | `gpu_12gb` continuation. | `BLOCKED_RESOURCE` |
| Real Qwen hooks | future GPU agent | Can HF Qwen interventions be predicted and verified? | None on VPS. | Hardware/weights/Transformers. | `gpu_12gb` continuation. | `BLOCKED_RESOURCE` |
| SkyJEPA | future external agent | Can long-horizon flight model be audited? | None. | Official assets unavailable/unverified. | Recheck official release. | `BLOCKED_EXTERNAL` |
