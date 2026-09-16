# Runs the WhatsApp webhook server. Loads PINECONE_API_KEY, GEMINI_API_KEY,
# WHATSAPP_ACCESS_TOKEN, WHATSAPP_PHONE_NUMBER_ID, WHATSAPP_VERIFY_TOKEN, and
# WHATSAPP_APP_SECRET from the project's .env automatically (never printed);
# an already-set environment variable still wins over the file.
#
# This starts a server that stays running and listens for Meta's webhook
# calls -- it does not exit on its own. Meta also needs a public HTTPS URL
# pointing at it; see connect_whatsapp/whatsapp-integration.md for the full
# setup (Meta app config, getting each credential, and exposing this server
# to the internet during local testing vs. a real deployment).
[CmdletBinding()]
param(
    [string]$MenuFile = 'structure.json',
    [string]$IndexName,
    [string]$Namespace,
    [ValidateRange(1, 50)][int]$TopK = 15,
    [string]$BindHost = '0.0.0.0',
    [int]$Port = 8000,
    [string]$PythonPath = '.venv/Scripts/python.exe'
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $projectRoot
try {
    if (-not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) {
        throw "Python executable not found at '$PythonPath'. Create .venv and install requirements.txt, generate_embedding/requirements-gemini.txt, generate_embedding/requirements-pinecone.txt, and connect_whatsapp/requirements-whatsapp.txt first."
    }

    $envFile = Join-Path $projectRoot '.env'
    if (Test-Path -LiteralPath $envFile -PathType Leaf) {
        foreach ($line in Get-Content -LiteralPath $envFile) {
            if ($line -match '^\s*#' -or $line -notmatch '=') { continue }
            $key, $value = $line -split '=', 2
            $key = $key.Trim()
            $value = $value.Trim()
            if ($key -and $value -and -not (Test-Path "env:$key")) {
                Set-Item -Path "env:$key" -Value $value
            }
        }
    }

    foreach ($name in @('PINECONE_API_KEY', 'GEMINI_API_KEY', 'WHATSAPP_ACCESS_TOKEN',
                         'WHATSAPP_PHONE_NUMBER_ID', 'WHATSAPP_VERIFY_TOKEN', 'WHATSAPP_APP_SECRET')) {
        if (-not (Get-Item "env:$name" -ErrorAction SilentlyContinue)) {
            throw "$name is not set. Fill it in in .env, or set it directly: `$env:$name = `"...`""
        }
    }

    $serverArguments = @(
        '-m', 'connect_whatsapp.whatsapp_server',
        '--menu', $MenuFile,
        '--top-k', $TopK,
        '--host', $BindHost,
        '--port', $Port
    )
    if ($IndexName) { $serverArguments += @('--index', $IndexName) }
    if ($Namespace) { $serverArguments += @('--namespace', $Namespace) }
    & $PythonPath @serverArguments
} finally {
    Pop-Location
}
