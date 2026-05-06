param(
    [string]$DbHost = "127.0.0.1",
    [string]$DbPort = "5432",
    [string]$DbName = "therapist_ai",
    [string]$DbUser = "postgres",
    [string]$DbPassword = "admin",
    [string]$MailHost = "",
    [string]$MailPort = "",
    [string]$MailUsername = "",
    [string]$MailPassword = "",
    [string]$MailFrom = "",
    [string]$PythonBaseUrl = "http://127.0.0.1:8000",
    [int]$ServerPort = 8080
)

$ErrorActionPreference = "Stop"

function Get-ListeningPids([int]$Port) {
    $lines = netstat -ano | Select-String ":$Port" | Select-String "LISTENING"
    if (-not $lines) { return @() }

    $pids = @()
    foreach ($line in $lines) {
        $parts = ($line.ToString().Trim() -split "\s+")
        if ($parts.Length -ge 5) {
            $listenerPidText = $parts[$parts.Length - 1]
            if ($listenerPidText -match "^\d+$") {
                $pids += [int]$listenerPidText
            }
        }
    }
    return $pids | Sort-Object -Unique
}

function Stop-ExistingTherapistBackend([int]$Port) {
    $pids = Get-ListeningPids -Port $Port
    if (-not $pids -or $pids.Count -eq 0) { return }

    foreach ($listenerPid in $pids) {
        try {
            $proc = Get-CimInstance Win32_Process -Filter "ProcessId = $listenerPid"
            if (-not $proc) { continue }

            $cmd = [string]$proc.CommandLine
            $isTherapistBackend = $cmd -match "TherapistBackendApplication|spring-boot:run|com\.therapist\.backend"

            if ($isTherapistBackend) {
                Write-Host "Arret ancien backend PID=$listenerPid sur le port $Port..."
                Stop-Process -Id $listenerPid -Force -ErrorAction SilentlyContinue
                Start-Sleep -Seconds 1
            } else {
                throw "Le port $Port est utilise par un autre processus (PID=$listenerPid, Name=$($proc.Name)). Ferme-le puis relance le backend."
            }
        } catch {
            throw $_
        }
    }
}

function Set-OptionalEnvVar([string]$Name, [string]$Value) {
    $envPath = "Env:$Name"
    if ($null -eq $Value -or $Value.Trim().Length -eq 0) {
        Remove-Item -Path $envPath -ErrorAction SilentlyContinue
        return
    }
    Set-Item -Path $envPath -Value $Value.Trim()
}

function Normalize-MailPassword([string]$SmtpHost, [string]$Password) {
    if ($null -eq $Password) {
        return ""
    }
    $trimmed = $Password.Trim()
    if (
        $SmtpHost -match "gmail\.com" -and
        $trimmed -match "^[A-Za-z0-9]{4}( [A-Za-z0-9]{4}){3}$"
    ) {
        return ($trimmed -replace " ", "")
    }
    return $trimmed
}

$env:DB_HOST = $DbHost
$env:DB_PORT = $DbPort
$env:DB_NAME = $DbName
$env:DB_USER = $DbUser
$env:DB_PASSWORD = $DbPassword
$env:PYTHON_BASE_URL = $PythonBaseUrl
$env:AUTH_EMAIL_CONSOLE_FALLBACK = "false"
Set-OptionalEnvVar -Name "MAIL_HOST" -Value $MailHost
Set-OptionalEnvVar -Name "MAIL_PORT" -Value $MailPort
Set-OptionalEnvVar -Name "MAIL_USERNAME" -Value $MailUsername
Set-OptionalEnvVar -Name "MAIL_PASSWORD" -Value (Normalize-MailPassword -SmtpHost $MailHost -Password $MailPassword)
Set-OptionalEnvVar -Name "MAIL_FROM" -Value $MailFrom

Stop-ExistingTherapistBackend -Port $ServerPort

Set-Location backend
mvn spring-boot:run
