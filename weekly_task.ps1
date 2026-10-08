# Weekly Tenrai retrain task helper (Windows Task Scheduler).
#
#   powershell -ExecutionPolicy Bypass -File weekly_task.ps1 -Action run
#   powershell -ExecutionPolicy Bypass -File weekly_task.ps1 -Action install
#   powershell -ExecutionPolicy Bypass -File weekly_task.ps1 -Action uninstall
#   powershell -ExecutionPolicy Bypass -File weekly_task.ps1 -Action install -Top 200
#
# - `run`     : same as the Linux cron / GitHub Action default (stale 7d, stages 2-6)
# - `install` : registers "AnimeWeeklyRetrain" -> every Monday 03:00, runs `run`
# - `uninstall`: removes the scheduled task again
param(
  [ValidateSet("run", "install", "uninstall")]
  [string]$Action = "run",
  [int]$StaleDays = 7,
  [int]$Top = 0,
  [string]$Stages = "2 3 4 5 6",
  [switch]$Force
)

$ErrorActionPreference = "Stop"
$Repo = Split-Path -Parent $MyInvocation.MyCommand.Path
$TaskName = "AnimeWeeklyRetrain"

if ($Action -eq "run") {
  Set-Location -LiteralPath $Repo
  $args = @()
  if ($Top -gt 0) { $args += @("--top", "$Top") }
  else { $args += @("--stale", "$StaleDays") }
  $args += @("--stages") + ($Stages -split '\s+')
  if ($Force) { $args += "--force" }
  Write-Host "running: python retrain_weekly.py $($args -join ' ')"
  & python retrain_weekly.py @args
  exit $LASTEXITCODE
}

if ($Action -eq "install") {
  $extra = ""
  if ($Top -gt 0) { $extra += " -Top $Top" }
  else { $extra += " -StaleDays $StaleDays" }
  if ($Stages -ne "2 3 4 5 6") { $extra += " -Stages '$Stages'" }
  if ($Force) { $extra += " -Force" }
  $tr = "powershell.exe -NoProfile -ExecutionPolicy Bypass -File `"$Repo\weekly_task.ps1`" -Action run$extra"
  # Weekly, Monday 03:00. Requires an elevated prompt only if /RU SYSTEM is used;
  # default runs as the current user.
  schtasks /Create /TN $TaskName /TR $tr /SC WEEKLY /D MON /ST 03:00 /F
  Write-Host "installed '$TaskName' -> every Monday 03:00"
  schtasks /Query /TN $TaskName
}

if ($Action -eq "uninstall") {
  schtasks /Delete /TN $TaskName /F
  Write-Host "removed '$TaskName'"
}
