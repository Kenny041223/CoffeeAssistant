# Runs the customer chat assistant. Loads PINECONE_API_KEY and
# GEMINI_API_KEY from the project's .env automatically (never printed);
# an already-set environment variable still wins over the file.
[CmdletBinding()]
param(
    [string]$MenuFile = 'structure.json',
    [string]$IndexName,
    [string]$Namespace,
    [ValidateRange(1, 50)][int]$TopK = 15,
    [string]$PythonPath = '.venv/Scripts/python.exe'
)

$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $projectRoot
try {
    if (-not (Test-Path -LiteralPath $PythonPath -PathType Leaf)) {
        throw "Python executable not found at '$PythonPath'. Create .venv and install requirements.txt, generate_embedding/requirements-gemini.txt, and generate_embedding/requirements-pinecone.txt first."
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

    if (-not $env:PINECONE_API_KEY) {
        throw 'PINECONE_API_KEY is not set. Fill it in in .env, or set it directly: $env:PINECONE_API_KEY = "..."'
    }
    if (-not $env:GEMINI_API_KEY) {
        throw 'GEMINI_API_KEY is not set. Fill it in in .env, or set it directly: $env:GEMINI_API_KEY = "..."'
    }

    $chatArguments = @(
        '-m', 'generate_prompt.chatbot',
        '--menu', $MenuFile,
        '--top-k', $TopK
    )
    if ($IndexName) { $chatArguments += @('--index', $IndexName) }
    if ($Namespace) { $chatArguments += @('--namespace', $Namespace) }
    & $PythonPath @chatArguments
} finally {
    Pop-Location
}
