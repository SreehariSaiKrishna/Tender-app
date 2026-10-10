#!/usr/bin/env bash
# WSL / Git Bash entry point for scripts/dev.ps1 - the venv is a Windows one
# (venv\Scripts\python.exe), so this hands off to Windows PowerShell:
#
#   ./scripts/dev.sh
#
# Dashboard + API on http://localhost:8000. Ctrl+C stops it.
set -euo pipefail
cd "$(dirname "$0")/.."
ps=$(command -v powershell.exe || true)
for candidate in /mnt/c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe \
                 /c/Windows/System32/WindowsPowerShell/v1.0/powershell.exe; do
  [ -n "$ps" ] && break
  [ -x "$candidate" ] && ps=$candidate
done
[ -n "$ps" ] || { echo "Can't find powershell.exe - run .\\scripts\\dev.ps1 from PowerShell instead." >&2; exit 1; }
exec "$ps" -NoProfile -ExecutionPolicy Bypass -File "scripts\\dev.ps1"
