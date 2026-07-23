# Pull results/ and logs/ back down from the EC2 instance (PowerShell version).
# Safe to run at any time, including mid-run -- results/leaderboard.md reflects
# whatever the last periodic rebuild on the instance produced.
#
# Usage:
#   .\deploy\fetch_results.ps1 -EC2Host ec2-user@1.2.3.4 -EC2Key ~\.ssh\my-key.pem
param(
    [Parameter(Mandatory=$true)][string]$EC2Host,
    [string]$EC2Key,
    [string]$RemoteDir = "~/trialbench-classification-benchmark"
)
$ErrorActionPreference = "Stop"

$Here = Split-Path -Parent $PSScriptRoot
Set-Location $Here
New-Item -ItemType Directory -Force -Path results, logs | Out-Null

$scpArgs = @("-o", "StrictHostKeyChecking=accept-new", "-r")
if ($EC2Key) { $scpArgs += @("-i", $EC2Key) }

Write-Host "==> fetching results/ and logs/ from ${EC2Host}:${RemoteDir}"
& scp @scpArgs "${EC2Host}:${RemoteDir}/results" .
if ($LASTEXITCODE -ne 0) { throw "scp results failed (exit $LASTEXITCODE)" }

& scp @scpArgs "${EC2Host}:${RemoteDir}/logs" .
# logs/ may not exist yet on a fresh instance -- don't fail the whole fetch over it.
if ($LASTEXITCODE -ne 0) { Write-Warning "scp logs failed (exit $LASTEXITCODE) -- continuing" }

Write-Host ""
Write-Host "==> done. See results\leaderboard.md"
