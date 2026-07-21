#!/usr/bin/env python
"""Capture bounded selected-site Hugging Face Qwen activations."""

from __future__ import annotations

import argparse
import json

from causal_workspace_jepa.data.llm_prompts.capture_qwen import run_qwen_activation_capture


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    result = run_qwen_activation_capture(args.config)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
