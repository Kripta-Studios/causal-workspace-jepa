$ErrorActionPreference = "Stop"

Write-Host "=== Git identity ==="
git branch --show-current
git rev-parse HEAD
git remote -v
git status --short

Write-Host "`n=== Recent commits ==="
git log -10 --oneline --decorate

Write-Host "`n=== Python ==="
python --version

Write-Host "`n=== Targeted entrypoint regression ==="
python -m pytest -q tests/unit/test_qwen_competence_recovery_entrypoints.py

Write-Host "`n=== Focused Qwen ==="
python -m pytest -q `
  tests/unit/test_qwen_binding_competence_recovery.py `
  tests/unit/test_qwen_bridge_phase0_guard.py `
  tests/unit/test_qwen_bridge_phase0_v2_guard.py

Write-Host "`n=== Focused CRCT ==="
python -m pytest -q tests/scientific/test_crct_stage0.py tests/scientific/test_crct_stage0_hard.py

Write-Host "`n=== Ruff ==="
python -m ruff check .

Write-Host "`n=== Full non-protected tests ==="
python -m pytest -q

Write-Host "`n=== Reproducibility audit ==="
python scripts/audit_reproducibility.py

Write-Host "`n=== Final status ==="
git status --short
