$ErrorActionPreference = "Stop"

$files = git ls-files --others --exclude-standard
if (-not $files) {
    Write-Host "No untracked files."
    exit 0
}

$rows = foreach ($path in $files) {
    if (Test-Path -LiteralPath $path -PathType Leaf) {
        $item = Get-Item -LiteralPath $path
        $hash = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLower()
        [pscustomobject]@{
            Path = $path
            Bytes = $item.Length
            SHA256 = $hash
        }
    } else {
        [pscustomobject]@{
            Path = $path
            Bytes = $null
            SHA256 = "<directory>"
        }
    }
}

$rows | Sort-Object Path | Format-Table -AutoSize
