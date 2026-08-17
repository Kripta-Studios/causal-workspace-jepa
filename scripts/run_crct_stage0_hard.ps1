[CmdletBinding()]
param(
    [ValidateSet('smoke', 'full')]
    [string]$Profile = 'full',
    [string]$Device = 'auto',
    [int[]]$Seeds = @(1009, 2027, 4093),
    [string]$PreflightDevice = 'auto',
    [int]$PreflightTimeoutSeconds = 300,
    [int]$SeedTimeoutSeconds = 1800,
    [switch]$FullTestSuite,
    [switch]$SkipTests,
    [switch]$AllowDifferentBase,
    [switch]$AllowCpuFull
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location $RepoRoot
$env:PYTHONPATH = if ($env:PYTHONPATH) { "$RepoRoot\src;$env:PYTHONPATH" } else { "$RepoRoot\src" }
$env:PYTHONFAULTHANDLER = '1'
$env:PYTHONUNBUFFERED = '1'

Write-Host "CRCT Stage-0 HARD-002 suite" -ForegroundColor Cyan
Write-Host "Repo:      $RepoRoot"
Write-Host "Profile:   $Profile"
Write-Host "Device:    $Device"
Write-Host "Seeds:     $($Seeds -join ', ')"
Write-Host "HEAD:      $(git rev-parse HEAD)"

$pythonArgs = @(
    '-X', 'faulthandler',
    'scripts/run_crct_stage0_hard_suite.py',
    '--profile', $Profile,
    '--device', $Device,
    '--preflight-device', $PreflightDevice,
    '--preflight-timeout-seconds', $PreflightTimeoutSeconds.ToString(),
    '--seed-timeout-seconds', $SeedTimeoutSeconds.ToString(),
    '--seeds'
) + ($Seeds | ForEach-Object { $_.ToString() })

if ($FullTestSuite) { $pythonArgs += '--full-test-suite' }
if ($SkipTests) { $pythonArgs += '--skip-tests' }
if ($AllowDifferentBase) { $pythonArgs += '--allow-different-base' }
if ($AllowCpuFull) { $pythonArgs += '--allow-cpu-full' }

python @pythonArgs
$exitCode = $LASTEXITCODE
if ($exitCode -ne 0) {
    Write-Warning "CRCT HARD-002 suite exited with code $exitCode. Inspect the retained diagnostics/ZIP."
}
exit $exitCode
