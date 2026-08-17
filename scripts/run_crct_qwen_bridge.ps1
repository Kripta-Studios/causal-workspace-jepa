[CmdletBinding()]
param(
    [string]$Device = 'cuda',
    [int]$ForwardBatch = 16,
    [int]$ReplayBatch = 8,
    [int]$DerivativeBatch = 1,
    [int]$Phase0TimeoutSeconds = 21600,
    [switch]$SkipCapitalDev,
    [switch]$SkipOntologyAudit
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location $RepoRoot
$env:PYTHONPATH = if ($env:PYTHONPATH) { "$RepoRoot\src;$env:PYTHONPATH" } else { "$RepoRoot\src" }
$env:PYTHONFAULTHANDLER = '1'
$env:PYTHONUNBUFFERED = '1'

Write-Host "CRCT -> Qwen Bridge V1" -ForegroundColor Cyan
Write-Host "Repo:       $RepoRoot"
Write-Host "Device:     $Device"
Write-Host "HEAD:       $(git rev-parse HEAD)"
Write-Host "Boundary:   calibration/train/validation ONLY; no test/paraphrase CLI exists"

$pythonArgs = @(
    'scripts/run_crct_qwen_bridge_suite.py',
    '--device', $Device,
    '--forward-batch', $ForwardBatch.ToString(),
    '--replay-batch', $ReplayBatch.ToString(),
    '--derivative-batch', $DerivativeBatch.ToString(),
    '--phase0-timeout-seconds', $Phase0TimeoutSeconds.ToString()
)
if ($SkipCapitalDev) { $pythonArgs += '--skip-capital-dev' }
if ($SkipOntologyAudit) { $pythonArgs += '--skip-ontology-audit' }

& python @pythonArgs
$code = $LASTEXITCODE
if ($code -ne 0) {
    Write-Warning "Bridge suite exited with code $code. Inspect/upload the newest ZIP; fail-safe diagnostics are retained."
}
exit $code
