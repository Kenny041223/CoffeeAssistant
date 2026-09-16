# Local webhook server; run run_whatsapp_worker.ps1 in a second terminal.
[CmdletBinding()]
param(
    [string]$BindHost = '127.0.0.1',
    [int]$Port = 8000,
    [string]$PythonPath = '.venv/Scripts/python.exe'
)

$ErrorActionPreference = 'Stop'
Push-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
try {
    . (Join-Path $PSScriptRoot 'load_env.ps1')
    & $PythonPath -m connect_whatsapp.whatsapp_server --host $BindHost --port $Port
    if ($LASTEXITCODE -ne 0) { throw "WhatsApp webhook exited with code $LASTEXITCODE" }
} finally {
    Pop-Location
}
