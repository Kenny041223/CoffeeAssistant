[CmdletBinding()]
param(
    [string]$PythonPath = '.venv/Scripts/python.exe',
    [switch]$Status,
    [string]$RetryFailed
)

$ErrorActionPreference = 'Stop'
Push-Location -LiteralPath (Split-Path -Parent $PSScriptRoot)
try {
    . (Join-Path $PSScriptRoot 'load_env.ps1')
    $workerArguments = @('-m', 'connect_whatsapp.message_worker')
    if ($Status) { $workerArguments += '--status' }
    if ($RetryFailed) { $workerArguments += @('--retry-failed', $RetryFailed) }
    & $PythonPath @workerArguments
    if ($LASTEXITCODE -ne 0) { throw "WhatsApp worker exited with code $LASTEXITCODE" }
} finally {
    Pop-Location
}
