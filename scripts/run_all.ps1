param(
    [switch]$ForceNpmInstall,
    [string]$DbHost = "",
    [string]$DbPort = "",
    [string]$DbName = "",
    [string]$DbUser = "",
    [string]$DbPassword = "",
    [string]$MailHost = "",
    [string]$MailPort = "",
    [string]$MailUsername = "",
    [string]$MailPassword = "",
    [string]$MailFrom = "",
    [string]$PythonBaseUrl = "",
    [string]$SttModel = "",
    [string]$SttLanguage = "",
    [string]$SttDevice = "",
    [string]$SttComputeType = ""
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$backendDir = Join-Path $root "backend"
$frontendDir = Join-Path $root "frontend"
$venvPython = Join-Path $root ".venv\Scripts\python.exe"
$venvDlPython = Join-Path $root ".venv_dl\Scripts\python.exe"
$envFilePath = Join-Path $root ".env"

function Read-DotEnvFile {
    param([string]$Path)
    $values = @{}
    if (-not (Test-Path $Path)) {
        return $values
    }
    foreach ($line in Get-Content $Path) {
        $trimmed = $line.Trim()
        if (-not $trimmed -or $trimmed.StartsWith("#")) {
            continue
        }
        $separatorIndex = $trimmed.IndexOf("=")
        if ($separatorIndex -lt 1) {
            continue
        }
        $key = $trimmed.Substring(0, $separatorIndex).Trim()
        $value = $trimmed.Substring($separatorIndex + 1).Trim()
        $values[$key] = $value
    }
    return $values
}

function Resolve-Setting {
    param(
        [string]$CurrentValue,
        [string[]]$FallbackKeys,
        [string]$DefaultValue,
        [hashtable]$DotEnv
    )
    $trimmedCurrent = if ($null -eq $CurrentValue) { "" } else { $CurrentValue.Trim() }
    if ($trimmedCurrent) {
        return $trimmedCurrent
    }
    foreach ($key in $FallbackKeys) {
        if ($DotEnv.ContainsKey($key) -and $DotEnv[$key].Trim()) {
            return $DotEnv[$key].Trim()
        }
    }
    return $DefaultValue
}

$dotEnv = Read-DotEnvFile -Path $envFilePath

$DbHost = Resolve-Setting -CurrentValue $DbHost -FallbackKeys @("DB_HOST") -DefaultValue "127.0.0.1" -DotEnv $dotEnv
$DbPort = Resolve-Setting -CurrentValue $DbPort -FallbackKeys @("DB_PORT") -DefaultValue "5432" -DotEnv $dotEnv
$DbName = Resolve-Setting -CurrentValue $DbName -FallbackKeys @("DB_NAME", "POSTGRES_DB") -DefaultValue "therapist_ai" -DotEnv $dotEnv
$DbUser = Resolve-Setting -CurrentValue $DbUser -FallbackKeys @("DB_USER", "POSTGRES_USER") -DefaultValue "postgres" -DotEnv $dotEnv
$DbPassword = Resolve-Setting -CurrentValue $DbPassword -FallbackKeys @("DB_PASSWORD", "POSTGRES_PASSWORD") -DefaultValue "admin" -DotEnv $dotEnv
$MailHost = Resolve-Setting -CurrentValue $MailHost -FallbackKeys @("MAIL_HOST") -DefaultValue "" -DotEnv $dotEnv
$MailPort = Resolve-Setting -CurrentValue $MailPort -FallbackKeys @("MAIL_PORT") -DefaultValue "" -DotEnv $dotEnv
$MailUsername = Resolve-Setting -CurrentValue $MailUsername -FallbackKeys @("MAIL_USERNAME") -DefaultValue "" -DotEnv $dotEnv
$MailPassword = Resolve-Setting -CurrentValue $MailPassword -FallbackKeys @("MAIL_PASSWORD") -DefaultValue "" -DotEnv $dotEnv
$MailFrom = Resolve-Setting -CurrentValue $MailFrom -FallbackKeys @("MAIL_FROM") -DefaultValue "" -DotEnv $dotEnv
$PythonBaseUrl = Resolve-Setting -CurrentValue $PythonBaseUrl -FallbackKeys @("PYTHON_BASE_URL") -DefaultValue "http://127.0.0.1:8000" -DotEnv $dotEnv
$SttModel = Resolve-Setting -CurrentValue $SttModel -FallbackKeys @("STT_MODEL") -DefaultValue "small" -DotEnv $dotEnv
$SttLanguage = Resolve-Setting -CurrentValue $SttLanguage -FallbackKeys @("STT_LANGUAGE") -DefaultValue "en" -DotEnv $dotEnv
$SttDevice = Resolve-Setting -CurrentValue $SttDevice -FallbackKeys @("STT_DEVICE") -DefaultValue "cpu" -DotEnv $dotEnv
$SttComputeType = Resolve-Setting -CurrentValue $SttComputeType -FallbackKeys @("STT_COMPUTE_TYPE") -DefaultValue "int8" -DotEnv $dotEnv

function Test-UvicornInstalled {
    param([string]$PythonExe)
    if (-not (Test-Path $PythonExe)) {
        return $false
    }
    try {
        & $PythonExe -c "import importlib.util,sys; sys.exit(0 if importlib.util.find_spec('uvicorn') else 1)" 2>$null
        return ($LASTEXITCODE -eq 0)
    } catch {
        return $false
    }
}

if (Test-UvicornInstalled -PythonExe $venvPython) {
    $pythonExe = $venvPython
} elseif (Test-UvicornInstalled -PythonExe $venvDlPython) {
    $pythonExe = $venvDlPython
} elseif (Test-Path $venvPython) {
    $pythonExe = $venvPython
} elseif (Test-Path $venvDlPython) {
    $pythonExe = $venvDlPython
} else {
    $pythonExe = "python"
}

$forceInstallLiteral = if ($ForceNpmInstall) { '$true' } else { '$false' }

$pythonCmd = @"
`$env:STT_MODEL='$SttModel'
`$env:STT_LANGUAGE='$SttLanguage'
`$env:STT_DEVICE='$SttDevice'
`$env:STT_COMPUTE_TYPE='$SttComputeType'
`$env:THERAPIST_LLM_ENABLED='true'
`$env:THERAPIST_LLM_PROVIDER='llama_cpp'
`$env:THERAPIST_LLAMA_MODEL_PATH='LLMA/tinyllama-1.1b-chat-v1.0/tinyllama-1.1b-chat-v1.0.Q2_K.gguf'
`$env:THERAPIST_LLAMA_RUNTIME_PATH='vendor/llama_cpp_runtime'
`$env:THERAPIST_LLAMA_HELPER_PYTHON='.tmp/python312_runtime/python.exe'
`$env:THERAPIST_LLAMA_HELPER_RUNTIME_PATH='vendor/llama_cpp_runtime312'
`$env:THERAPIST_LLAMA_PREFER_HELPER='true'
`$env:THERAPIST_LLAMA_N_CTX='1024'
`$env:THERAPIST_LLAMA_N_THREADS='1'
`$env:THERAPIST_LLM_MAX_TOKENS='16'
`$env:THERAPIST_LLM_HISTORY_MAX_TURNS='1'
`$env:THERAPIST_LLM_TEMPERATURE='0.3'
`$env:THERAPIST_LLM_TOP_P='0.8'
`$env:THERAPIST_LLM_NON_BLOCKING_STARTUP='true'
`$env:THERAPIST_WARMUP_ON_STARTUP='true'
& '$pythonExe' -m uvicorn src.api.server:app --reload --port 8000
"@
$backendCmd = "Set-Location '$root'; .\scripts\run_backend.ps1 -DbHost '$DbHost' -DbPort '$DbPort' -DbName '$DbName' -DbUser '$DbUser' -DbPassword '$DbPassword' -MailHost '$MailHost' -MailPort '$MailPort' -MailUsername '$MailUsername' -MailPassword '$MailPassword' -MailFrom '$MailFrom' -PythonBaseUrl '$PythonBaseUrl'"
$frontendCmd = "Set-Location '$frontendDir'; if ($forceInstallLiteral -or -not (Test-Path 'node_modules')) { npm install }; npm run dev"

Start-Process powershell -ArgumentList "-NoExit", "-Command", $pythonCmd -WorkingDirectory $root
Start-Process powershell -ArgumentList "-NoExit", "-Command", $backendCmd -WorkingDirectory $backendDir
Start-Process powershell -ArgumentList "-NoExit", "-Command", $frontendCmd -WorkingDirectory $frontendDir

Write-Host "Services demarres dans 3 fenetres PowerShell:"
Write-Host "- Python IA: http://127.0.0.1:8000"
Write-Host "- Backend Spring: http://127.0.0.1:8080"
Write-Host "- Frontend Vite: http://localhost:5173"
