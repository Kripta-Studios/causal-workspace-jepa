\
param(
    [string]$Device = "cuda",
    [int]$BatchSize = 16,
    [int]$TimeoutSeconds = 3600
)

$ErrorActionPreference = "Stop"
$Repo = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Repo

Write-Host "Qwen Binding Competence Recovery V1"
Write-Host "Repo:      $Repo"
Write-Host "Device:    $Device"
Write-Host "HEAD:      $(git rev-parse HEAD)"
Write-Host "Boundary:  CALIBRATION MODEL FORWARDS ONLY"
Write-Host "Forbidden: train / validation / test / paraphrase"

$env:PYTHONPATH = "$Repo\src" + [IO.Path]::PathSeparator + $env:PYTHONPATH
$env:PYTHONFAULTHANDLER = "1"
$env:PYTHONUNBUFFERED = "1"

python -X faulthandler .\scripts\run_qwen_competence_recovery_suite.py `
    --device $Device `
    --batch-size $BatchSize `
    --timeout-seconds $TimeoutSeconds

if ($LASTEXITCODE -ne 0) {
    Write-Warning "Competence recovery suite exited with code $LASTEXITCODE. Upload newest ZIP."
}
