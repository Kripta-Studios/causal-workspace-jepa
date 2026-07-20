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
