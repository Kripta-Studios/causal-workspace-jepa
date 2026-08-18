from __future__ import annotations

import ast
from pathlib import Path
import shutil
import subprocess

import pytest


ROOT = Path(__file__).resolve().parents[2]
PS_WRAPPER = ROOT / "scripts/run_qwen_competence_recovery.ps1"
PYTHON_ENTRYPOINTS = (
    ROOT / "scripts/run_qwen_competence_recovery_suite.py",
    ROOT
    / "src/causal_workspace_jepa/experiments/llm/qwen_binding_competence_recovery.py",
)


def test_competence_recovery_entrypoints_have_no_leading_junk() -> None:
    ps_text = PS_WRAPPER.read_text(encoding="utf-8-sig")
    assert ps_text.startswith("param(")

    for path in PYTHON_ENTRYPOINTS:
        text = path.read_text(encoding="utf-8-sig")
        assert text.startswith("from __future__ import annotations")


def test_competence_recovery_python_entrypoints_parse() -> None:
    for path in PYTHON_ENTRYPOINTS:
        ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))


def test_competence_recovery_powershell_wrapper_parses_when_available() -> None:
    shell = shutil.which("pwsh") or shutil.which("powershell")
    if shell is None:
        pytest.skip("PowerShell is unavailable on this test host")

    escaped_path = str(PS_WRAPPER).replace("'", "''")
    parser_command = (
        "$tokens = $null; "
        "$errors = $null; "
        "$null = [System.Management.Automation.Language.Parser]::ParseFile("
        f"'{escaped_path}', [ref]$tokens, [ref]$errors); "
        "if ($errors.Count -gt 0) { "
        "$errors | ForEach-Object { Write-Error $_.Message }; "
        "exit 1 "
        "}"
    )
    result = subprocess.run(
        [shell, "-NoProfile", "-NonInteractive", "-Command", parser_command],
        cwd=ROOT,
        text=True,
        capture_output=True,
        errors="replace",
        timeout=30,
        check=False,
    )
    assert result.returncode == 0, (
        "PowerShell parser rejected competence recovery wrapper:\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )
