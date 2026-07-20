# LLM Track

Status: `SCAFFOLDED`.

CPU VPS implementation uses a mock transformer with known activation dependencies. It is valid for interface, leakage, and intervention-pipeline tests only.

Real Hugging Face Qwen instrumentation is `BLOCKED_RESOURCE` on this VPS because it requires model weights, Transformers, selected-layer activation storage, and suitable GPU memory. Ollama is not a hidden-state or autograd source.
