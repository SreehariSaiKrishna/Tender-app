# Runs the API and the dashboard locally, for development and debugging:
#
#   .\scripts\dev.ps1
#
# - API: uvicorn on http://127.0.0.1:8001, reloading on every change under
#   app/ and config/. No login locally - the dashboard skips Cognito on
#   localhost, and uvicorn has no authorizer in front of it.
# - Dashboard: http://localhost:8000, served straight from frontend/ - just
#   refresh the browser after editing index.html.
#
# Both use .env: the same MongoDB and S3 bucket (FILES_BUCKET, reached with
# AWS_PROFILE) as production, so what you do locally - generating a bid
# pack, marking a tender applied, uploading documents - changes real data.
# Ctrl+C stops both.
$ErrorActionPreference = "Stop"
Set-Location (Join-Path $PSScriptRoot "..")

$python = if (Test-Path "venv\Scripts\python.exe") { "venv\Scripts\python.exe" } else { "python" }

$frontend = Start-Process -FilePath $python -PassThru -NoNewWindow `
  -ArgumentList "-m", "http.server", "8000", "--bind", "127.0.0.1", "--directory", "frontend"
Write-Host "Dashboard: http://localhost:8000"
Write-Host "API:       http://127.0.0.1:8001 (docs at /docs)"
try {
  & $python -m uvicorn app.api.main:app --host 127.0.0.1 --port 8001 --reload --reload-dir app --reload-dir config
} finally {
  Stop-Process -Id $frontend.Id -ErrorAction SilentlyContinue
}
