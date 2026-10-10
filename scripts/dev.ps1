# Runs the API and the dashboard locally, for development and debugging:
#
#   .\scripts\dev.ps1
#
# One uvicorn process on http://localhost:8000 serves both the dashboard (at
# /, straight from frontend/index.html - just refresh the browser after
# editing it) and the API (docs at /docs), reloading on every change under
# app/ and config/. Being one process, the dashboard can't be up without the
# API. No login locally - the dashboard skips Cognito on localhost, and
# uvicorn has no authorizer in front of it.
#
# It uses .env: the same MongoDB and S3 bucket (FILES_BUCKET, reached with
# AWS_PROFILE) as production, so what you do locally - generating a bid
# pack, marking a tender applied, uploading documents - changes real data.
# Ctrl+C stops it.
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

$port = 8000
$busy = Get-NetTCPConnection -State Listen -LocalPort $port -ErrorAction SilentlyContinue | Select-Object -First 1
if ($busy) {
  $proc = Get-CimInstance Win32_Process -Filter "ProcessId=$($busy.OwningProcess)"
  Write-Host "Port $port is already in use by process $($busy.OwningProcess):" -ForegroundColor Red
  Write-Host "  $($proc.CommandLine)"
  Write-Host "Stop it (close its terminal, or: Stop-Process -Id $($busy.OwningProcess)) and run this again."
  exit 1
}

$python = if (Test-Path "venv\Scripts\python.exe") { "venv\Scripts\python.exe" } else { "python" }

$env:SERVE_DASHBOARD = "1"
# --reload misses saves in a OneDrive folder (file events don't arrive), so
# the API kept serving old code - poll for changes instead.
$env:WATCHFILES_FORCE_POLLING = "true"
Write-Host "Dashboard + API: http://localhost:$port (API docs at /docs)"
& $python -m uvicorn app.api.main:app --host 127.0.0.1 --port $port --reload --reload-dir app --reload-dir config
