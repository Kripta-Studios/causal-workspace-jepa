#!/usr/bin/env bash
set -euo pipefail

PROFILE="${1:-configs/resource/cpu_vps.yaml}"

export PATH="$HOME/.local/bin:$PATH"
export PYTHONPATH="${PYTHONPATH:-src}"

python -m causal_workspace_jepa.cli doctor --resource-profile "$PROFILE"

if ! command -v uv >/dev/null 2>&1; then
  echo "uv is not installed. Install with: curl -LsSf https://astral.sh/uv/install.sh | sh" >&2
  echo "SKIPPED_RESOURCE: bootstrap did not install tools automatically on this constrained VPS" >&2
  exit 2
fi

PYTHON_BIN="${PYTHON_BIN:-python3}"
uv venv --python "$PYTHON_BIN"
source .venv/bin/activate
uv pip install -e ".[dev,cpu]"
python -m causal_workspace_jepa.cli doctor --resource-profile "$PROFILE"
pytest -q tests/unit tests/integration -m "not gpu and not slow"
