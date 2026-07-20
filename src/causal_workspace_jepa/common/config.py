"""Small configuration loader for CPU-safe YAML subsets.

The CPU VPS path intentionally avoids a hard PyYAML dependency. The parser
supports the simple mappings and scalar values used by repository configs.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if value == "":
        return {}
    if value in {"true", "True"}:
        return True
    if value in {"false", "False"}:
        return False
    if value in {"null", "None", "~"}:
        return None
    if value.startswith('"') and value.endswith('"'):
        return value[1:-1]
    if value.startswith("'") and value.endswith("'"):
        return value[1:-1]
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(part.strip()) for part in inner.split(",")]
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a small YAML mapping from ``path``.

    This is deliberately conservative. It rejects top-level lists and complex
    YAML constructs so config parsing remains deterministic without optional
    dependencies.
    """

    path = Path(path)
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for lineno, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.split("#", 1)[0].rstrip()
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip(" "))
        if indent % 2:
            raise ValueError(f"{path}:{lineno}: indentation must use multiples of two spaces")
        stripped = line.strip()
        if stripped.startswith("- "):
            raise ValueError(f"{path}:{lineno}: list blocks are not supported in CPU config parser")
        if ":" not in stripped:
            raise ValueError(f"{path}:{lineno}: expected key: value")
        key, value = stripped.split(":", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"{path}:{lineno}: empty key")
        while stack and indent <= stack[-1][0]:
            stack.pop()
        if not stack:
            raise ValueError(f"{path}:{lineno}: invalid indentation")
        parsed = _parse_scalar(value)
        stack[-1][1][key] = parsed
        if isinstance(parsed, dict):
            stack.append((indent, parsed))
    return root


def get_nested(config: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    """Return a dotted config value, or ``default`` when absent."""

    value: Any = config
    for part in dotted_key.split("."):
        if not isinstance(value, dict) or part not in value:
            return default
        value = value[part]
    return value
