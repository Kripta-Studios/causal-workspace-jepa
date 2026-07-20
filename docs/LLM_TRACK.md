# LLM Track

Status: `SMOKE_VALIDATED`.

CPU VPS implementation uses a mock transformer with known activation dependencies. It is valid for interface, leakage, and intervention-pipeline tests only.

Real Hugging Face Qwen instrumentation is `BLOCKED_RESOURCE` on this VPS because it requires model weights, Transformers, selected-layer activation storage, and suitable GPU memory. Ollama is not a hidden-state or autograd source.

Current mock implementation:

- exposes `resid_pre`, `attn_out`, `mlp_out`, `resid_post`, and `logits`;
- supports direct replayable interventions through the shared `InterventionSpec`;
- trains a small intervention-conditioned JEPA-style ridge predictor against no-change, mean-effect, and linear-context baselines.

Validated CPU mock smoke:

- `LLM-MOCK-001` ran with code commit `85c1dbfbe9c824bcca415af13f4a6f34acc95267`.
- Intervention-JEPA MSE: `1.32e-07`; no-change: `0.002113`; mean-effect: `0.002113`; linear-context: `0.002113`.
- Effect correlation: `0.99997`.
- This is not Qwen evidence.

Validated GPT-2 Medium smoke:

- `LLM-GPT2-001` loaded `gpt2-medium` through Hugging Face Transformers in `.venv`.
- Direct residual steering at `transformer.h.12.resid_post` changed downstream logits with mean absolute logit delta `0.0797`.
- Tiny intervention-JEPA MSE was `0.00220` vs no-change `0.0114` on a 4-example held-out smoke split.
- This is causal-mediation smoke evidence for a directly edited residual coordinate, not a J-space or workspace discovery.

Strengthened GPT-2 Medium study:

- `LLM-GPT2-002` ran from clean commit `8fbab8c` and generated a checksummed 288-outcome dataset.
- It batches 288 direct residual interventions across eight prompts, layers `6/12/18`, two
  coordinates, and six magnitudes while storing only selected positions and outputs.
- Prompt, magnitude, and layer holdouts are explicit; training uses six prompts, layers `6/12`, and
  magnitudes through `0.5`, while evaluation uses two unseen prompts, magnitude `1.0`, and a separate
  layer-18 stress split.
- Required baselines include a prompt-local finite-difference Jacobian, corpus-averaged Jacobian,
  linear and bilinear regressions, trained MLP, nearest neighbor, and sparse-context transport.
- On held-out prompts/magnitude, local-Jacobian MSE was `7.79e-7`, bilinear Intervention-JEPA
  `0.003499`, and no-change `0.006461`. The bilinear model captured some context-dependent effect but
  did not beat the strong local baseline.
- On held-out layer 18, bilinear MSE `0.01009` was worse than no-change `0.006228`; cross-layer
  transfer failed. The MLP also failed and no intervention changed the top token.
- The model loaded from the existing local cache only. This remains selected-output GPT-2 causal
  smoke evidence, not Qwen, behavior-level, feature-semantic, or workspace evidence.
