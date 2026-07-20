# Approach Registry

| Family | Owner | Question | Evidence | Blockers | Next Experiment | Status |
| --- | --- | --- | --- | --- | --- | --- |
| Architecture/interfaces | root agent | Can the project run reproducibly in CPU mode? | Resource guard and typed interfaces. | None for Milestone 0. | Doctor and tests. | `ACTIVE` |
| Tier 0 dynamics | root agent | Can deterministic simulators expose ground truth? | None yet. | Not implemented. | Generate smoke datasets. | `NOT_STARTED` |
| Tiny JEPA | root agent | Does action conditioning improve toy prediction? | None yet. | Not implemented. | Tiny smoke training. | `NOT_STARTED` |
| Intervention primitives | root agent | Are interventions serializable and replayable? | Dataclass interface only. | Operators not implemented. | Unit tests for zero/patch/project-out. | `SCAFFOLDED` |
| Mock Qwen | root agent | Can known activation dynamics validate LLM pipeline? | Protocol only. | Adapter not implemented. | Mock intervention dataset. | `NOT_STARTED` |
| Published JEPA adapters | future GPU agent | Can published world models support the same audit? | None on VPS. | Hardware/data/weights. | `gpu_12gb` continuation. | `BLOCKED_RESOURCE` |
| Real Qwen hooks | future GPU agent | Can HF Qwen interventions be predicted and verified? | None on VPS. | Hardware/weights/Transformers. | `gpu_12gb` continuation. | `BLOCKED_RESOURCE` |
| SkyJEPA | future external agent | Can long-horizon flight model be audited? | None. | Official assets unavailable/unverified. | Recheck official release. | `BLOCKED_EXTERNAL` |
