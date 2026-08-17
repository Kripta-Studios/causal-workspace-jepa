[CmdletBinding()]
param(
    [ValidateSet('smoke', 'full', 'max')]
    [string]$Profile = 'full',
    [string]$Device = 'auto',
    [int[]]$Seeds = @(7, 13, 23),
    [switch]$FullTestSuite,
    [switch]$SkipTests,
    [switch]$AllowDifferentBase,
    [switch]$AllowCpuFull,
    [string]$PreflightDevice = 'auto',
    [int]$PreflightTimeoutSeconds = 300,
    [switch]$SkipPreflight
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location $RepoRoot
$env:PYTHONPATH = if ($env:PYTHONPATH) { "$RepoRoot\src;$env:PYTHONPATH" } else { "$RepoRoot\src" }

Write-Host "CRCT Stage-0 suite" -ForegroundColor Cyan
Write-Host "Repo:    $RepoRoot"
Write-Host "Profile: $Profile"
Write-Host "Device:  $Device"
Write-Host "Seeds:   $($Seeds -join ', ')"
Write-Host "HEAD:    $(git rev-parse HEAD)"

$pythonArgs = @(
    'scripts/run_crct_stage0_suite.py',
    '--profile', $Profile,
    '--device', $Device,
    '--preflight-device', $PreflightDevice,
    '--preflight-timeout-seconds', $PreflightTimeoutSeconds.ToString(),
    '--seeds'
) + ($Seeds | ForEach-Object { $_.ToString() })

if ($FullTestSuite) { $pythonArgs += '--full-test-suite' }
if ($SkipTests) { $pythonArgs += '--skip-tests' }
if ($AllowDifferentBase) { $pythonArgs += '--allow-different-base' }
if ($AllowCpuFull) { $pythonArgs += '--allow-cpu-full' }
if ($SkipPreflight) { $pythonArgs += '--skip-preflight' }

python @pythonArgs
$exitCode = $LASTEXITCODE
if ($exitCode -ne 0) {
    Write-Warning "CRCT suite exited with code $exitCode. The diagnostic directory/ZIP is still retained when creation reached that stage."
}
exit $exitCode
