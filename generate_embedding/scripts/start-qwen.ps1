param([switch]$PullModel, [switch]$PullTextModel)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$ollamaExe = Join-Path $projectRoot '.tools/ollama/ollama.exe'
if (-not (Test-Path -LiteralPath $ollamaExe)) {
    throw 'Portable Ollama is missing. See docs/ocr.md for installation instructions.'
}
$env:OLLAMA_HOST = '127.0.0.1:11435'
$env:OLLAMA_MODELS = Join-Path $projectRoot '.tools/ollama-models'
$env:OLLAMA_NO_CLOUD = '1'
$env:OLLAMA_NUM_PARALLEL = '1'
$env:OLLAMA_MAX_LOADED_MODELS = '1'
$running = $false
try {
    $null = Invoke-RestMethod 'http://127.0.0.1:11435/api/version' -TimeoutSec 2
    $running = $true
} catch {}
if (-not $running) {
    Start-Process -FilePath $ollamaExe -ArgumentList 'serve' -WindowStyle Hidden `
        -RedirectStandardOutput (Join-Path $projectRoot '.tools/ollama-server.out.log') `
        -RedirectStandardError (Join-Path $projectRoot '.tools/ollama-server.err.log') | Out-Null
    for ($attempt = 0; $attempt -lt 30; $attempt++) {
        Start-Sleep -Seconds 1
        try {
            $null = Invoke-RestMethod 'http://127.0.0.1:11435/api/version' -TimeoutSec 2
            $running = $true
            break
        } catch {}
    }
    if (-not $running) { throw 'Ollama did not start. Inspect .tools/ollama-server.err.log.' }
}
if ($PullModel) {
    & $ollamaExe pull 'qwen3-vl:4b-instruct'
    if ($LASTEXITCODE -ne 0) { throw 'Qwen model download failed.' }
}
if ($PullTextModel) {
    & $ollamaExe pull 'qwen3:4b-instruct-2507-q4_K_M'
    if ($LASTEXITCODE -ne 0) { throw 'Qwen text model download failed.' }
}
Write-Output 'Local Qwen server ready at http://127.0.0.1:11435'
