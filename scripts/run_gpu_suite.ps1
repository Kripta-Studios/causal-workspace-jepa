param(
    [string]$Python = ""
)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
if (-not $Python) {
    $candidate = "C:\Program Files\Python314\python.exe"
    if (Test-Path $candidate) {
        $Python = $candidate
    } else {
        $Python = "python"
    }
}

$env:PYTHONPATH = Join-Path $root "src"
$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"

& $Python -c "import torch; assert torch.cuda.is_available(), 'SKIPPED_RESOURCE: GPU suite requires CUDA'; print(torch.cuda.get_device_name(0))"
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

& $Python -m pytest -q tests/unit tests/integration tests/scientific -m "not slow"
exit $LASTEXITCODE
