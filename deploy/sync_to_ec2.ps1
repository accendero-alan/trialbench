# Push this repo up to an EC2 instance from Windows PowerShell.
# Uses the tar.exe + ssh.exe that ship with Windows 10/11 (OpenSSH client) --
# no rsync/WSL required. Excludes data/, results/, logs/, .venv/, etc.,
# which are produced/managed on the instance itself (see bootstrap_ec2.sh).
#
# Usage:
#   .\deploy\sync_to_ec2.ps1 -EC2Host ec2-user@1.2.3.4 -EC2Key ~\.ssh\my-key.pem
#   .\deploy\sync_to_ec2.ps1 -EC2Host my-instance-alias                 # ~\.ssh\config
#   .\deploy\sync_to_ec2.ps1 -EC2Host ec2-user@1.2.3.4 -RemoteDir /home/ec2-user/trialbench
param(
    [Parameter(Mandatory=$true)][string]$EC2Host,
    [string]$EC2Key,
    [string]$RemoteDir = "~/trialbench-classification-benchmark"
)
$ErrorActionPreference = "Stop"

$Here = Split-Path -Parent $PSScriptRoot
Set-Location $Here

$sshArgs = @("-o", "StrictHostKeyChecking=accept-new")
if ($EC2Key) { $sshArgs += @("-i", $EC2Key) }

Write-Host "==> syncing $Here -> ${EC2Host}:${RemoteDir}"

& ssh @sshArgs $EC2Host "mkdir -p '$RemoteDir'"
if ($LASTEXITCODE -ne 0) { throw "ssh mkdir failed (exit $LASTEXITCODE)" }

# tar.exe (bsdtar, built into Windows) streamed over ssh -- mirrors the bash
# fallback path so both entry points behave the same way.
$excludes = @(".git", ".venv", "__pycache__", ".pytest_cache", "data", "results", "logs", "catboost_info")
$tarArgs = @("czf", "-")
foreach ($e in $excludes) { $tarArgs += @("--exclude=./$e") }
$tarArgs += "."

$tarCmd = "tar $($tarArgs -join ' ')"
$sshCmd = "ssh $($sshArgs -join ' ') $EC2Host `"tar xzf - -C '$RemoteDir'`""

cmd /c "$tarCmd | $sshCmd"
if ($LASTEXITCODE -ne 0) { throw "tar | ssh pipeline failed (exit $LASTEXITCODE)" }

Write-Host ""
Write-Host "==> done. Next, on the instance:"
Write-Host "    ssh $(if ($EC2Key) {"-i $EC2Key "})$EC2Host"
Write-Host "    cd $RemoteDir && ./deploy/bootstrap_ec2.sh"
