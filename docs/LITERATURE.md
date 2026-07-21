# Literature Registry

Last primary-source check: 2026-07-20. Unchecked entries remain `UNVERIFIED_FROM_PROMPT`.

| ID | Source | URL | Code/Data | Relevance | Status |
| --- | --- | --- | --- | --- | --- |
| LIT-001 | Anthropic, A global workspace in language models | https://www.anthropic.com/research/global-workspace | Anthropic article | Converts J-space claims into falsifiable workspace tests without assuming transfer to JEPA. | `UNVERIFIED_FROM_PROMPT` |
| LIT-002 | Verbalizable Representations Form a Global Workspace in Language Models | https://transformer-circuits.pub/2026/workspace/index.html | Web publication, published 2026-07-06 | Defines a sparse nonnegative token-aligned J-space and tests report, modulation, reasoning, reuse, selectivity, depth, capacity, and broadcast. | `PRIMARY_VERIFIED_2026-07-20` |
| LIT-003 | Anthropic Jacobian Lens implementation | https://github.com/anthropics/jacobian-lens | Official companion code; not cloned | Baseline corpus-averaged Jacobian transport and sparse J-space decomposition. | `PRIMARY_VERIFIED_2026-07-20` |
| LIT-004 | LeJEPA | https://arxiv.org/abs/2511.08544 | Not verified | JEPA theory and scalable objective assumptions. | `UNVERIFIED_FROM_PROMPT` |
| LIT-005 | LeWorldModel | https://arxiv.org/abs/2603.19312 | https://github.com/lucas-maes/le-wm | Published action-conditioned pixel JEPA target. | `BLOCKED_RESOURCE` |
| LIT-006 | What Drives Success in Physical Planning with JEPA WMs? | https://arxiv.org/abs/2512.24497 | https://github.com/facebookresearch/jepa-wms | Planning baselines and checkpoints for GPU continuation. | `BLOCKED_RESOURCE` |
| LIT-007 | Delta-JEPA | https://arxiv.org/abs/2606.31232 | arXiv primary record | Introduces latent-difference action decoding; directly motivates H-WM-02 but not a workspace claim. | `PRIMARY_VERIFIED_2026-07-20` |
| LIT-008 | Causal-JEPA | https://arxiv.org/abs/2602.11389 | https://github.com/galilai-group/cjepa | Object-level latent intervention comparison. | `BLOCKED_RESOURCE` |
| LIT-009 | V-JEPA 2 | https://arxiv.org/abs/2506.09985 | https://github.com/facebookresearch/vjepa2 | Video foundation world-model comparison. | `BLOCKED_RESOURCE` |
| LIT-010 | V-JEPA 2.1 | https://arxiv.org/abs/2603.14482 | Not verified | Scaling continuation. | `BLOCKED_RESOURCE` |
| LIT-011 | SkyJEPA | https://arxiv.org/abs/2606.23444 | https://github.com/arplaboratory/SkyJEPA | Long-horizon sim-to-real validation. | `BLOCKED_EXTERNAL` |
| LIT-012 | Temporal Straightening for Latent Planning | https://arxiv.org/abs/2603.12231 | Not verified | Planning geometry baseline. | `UNVERIFIED_FROM_PROMPT` |
| LIT-013 | AdaJEPA | https://arxiv.org/abs/2606.32026 | Not verified | Test-time adaptation mechanism tests. | `UNVERIFIED_FROM_PROMPT` |
| LIT-014 | A Generalization Theory for JEPA-Based World Models | https://arxiv.org/abs/2606.27014 | Not verified | OOD/generalization framing. | `UNVERIFIED_FROM_PROMPT` |
| LIT-015 | Interpreting Physics in Video World Models | https://arxiv.org/abs/2602.07050 | arXiv primary record | Reports an intermediate Physics Emergence Zone and distributed circular direction geometry; motivates depth and coordinated-intervention tests. | `PRIMARY_VERIFIED_2026-07-20` |
| LIT-016 | Do Video Foundation Models Understand Intuitive Physics? | https://arxiv.org/abs/2606.09646 | Not verified | Layerwise probing controls. | `UNVERIFIED_FROM_PROMPT` |
| LIT-017 | Probing the Latent World | https://arxiv.org/abs/2603.20327 | Not verified | Discrete symbols and physical structure. | `UNVERIFIED_FROM_PROMPT` |
| LIT-018 | Reading Between the Dots | https://arxiv.org/abs/2607.03502 | https://github.com/kaleybrauer/filler-token-reasoning | Filler-token hidden computation prompt families. | `BLOCKED_RESOURCE` |
| LIT-019 | Let's Think Dot by Dot | https://arxiv.org/abs/2404.15758 | Not verified | Hidden computation across filler tokens. | `UNVERIFIED_FROM_PROMPT` |
| LIT-020 | When Chain-of-Thought Fails | https://arxiv.org/abs/2604.23351 | Not verified | Hidden-state answer evidence. | `UNVERIFIED_FROM_PROMPT` |
| LIT-021 | Switchable Latent Reasoning | https://arxiv.org/abs/2606.13106 | Not verified | Hidden-state recurrence. | `UNVERIFIED_FROM_PROMPT` |
| LIT-022 | Qwen3-0.6B | https://huggingface.co/Qwen/Qwen3-0.6B | Hugging Face; Apache-2.0; pinned revision `c1899de...` | Official 28-layer, 0.6B-parameter smoke target; selected-site instrumentation only. | `PRIMARY_VERIFIED_2026-07-21` |
| LIT-023 | Qwen3-4B | https://huggingface.co/Qwen/Qwen3-4B | Hugging Face | Primary mechanistic target on GPU profile. | `BLOCKED_RESOURCE` |
| LIT-024 | Qwen3-30B-A3B | https://huggingface.co/Qwen/Qwen3-30B-A3B | Hugging Face | Cluster-scale target. Full weights required despite sparse activation. | `BLOCKED_RESOURCE` |
