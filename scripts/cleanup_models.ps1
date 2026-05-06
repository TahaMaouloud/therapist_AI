param(
    [switch]$WhatIfOnly
)

$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$modelsRoot = Join-Path $projectRoot "models"

if (-not (Test-Path -LiteralPath $modelsRoot)) {
    throw "Dossier models introuvable: $modelsRoot"
}

$keepRelative = @(
    ".gitkeep",
    "emotion_bert",
    "emotion_metrics.json",
    "emotion_audio_model.joblib",
    "emotion_audio_metrics.json",
    "emotion_audio_model_tuned.joblib",
    "emotion_audio_metrics_tuned.json",
    "_tmp_w2v_tuned_model.joblib",
    "_tmp_w2v_tuned_metrics.json",
    "wav2vec2_embedding_cache"
)

$keepAbsolute = [System.Collections.Generic.HashSet[string]]::new([System.StringComparer]::OrdinalIgnoreCase)
foreach ($item in $keepRelative) {
    $full = [System.IO.Path]::GetFullPath((Join-Path $modelsRoot $item))
    [void]$keepAbsolute.Add($full)
}

$allChildren = Get-ChildItem -LiteralPath $modelsRoot -Force
$toDelete = @()
foreach ($child in $allChildren) {
    $full = [System.IO.Path]::GetFullPath($child.FullName)
    if ($keepAbsolute.Contains($full)) {
        continue
    }
    $toDelete += $child
}

Write-Host ""
Write-Host "Elements gardes dans models/:"
$keepRelative | ForEach-Object { Write-Host "  - $_" }

Write-Host ""
Write-Host "Elements a supprimer:"
if (-not $toDelete) {
    Write-Host "  - Rien a supprimer."
    exit 0
}
$toDelete | ForEach-Object { Write-Host "  - $($_.Name)" }

if ($WhatIfOnly) {
    Write-Host ""
    Write-Host "Mode simulation termine. Aucun fichier supprime."
    exit 0
}

$acl = Get-Acl -LiteralPath $modelsRoot
$denyRules = @()
foreach ($rule in @($acl.Access)) {
    try {
        $sid = $rule.IdentityReference.Translate([System.Security.Principal.SecurityIdentifier]).Value
    } catch {
        $sid = ""
    }
    if (
        $rule.AccessControlType -eq [System.Security.AccessControl.AccessControlType]::Deny -and
        $sid -eq "S-1-1-0" -and
        ($rule.FileSystemRights -band [System.Security.AccessControl.FileSystemRights]::DeleteSubdirectoriesAndFiles)
    ) {
        $denyRules += $rule
    }
}

if ($denyRules.Count -gt 0) {
    Write-Host ""
    Write-Host "Suppression temporaire de la regle ACL qui bloque les suppressions dans models/ ..."
    foreach ($rule in $denyRules) {
        [void]$acl.RemoveAccessRuleSpecific($rule)
    }
    Set-Acl -LiteralPath $modelsRoot -AclObject $acl
}

try {
    foreach ($item in $toDelete) {
        Write-Host "Suppression: $($item.FullName)"
        Remove-Item -LiteralPath $item.FullName -Recurse -Force
    }
}
finally {
    if ($denyRules.Count -gt 0) {
        Write-Host ""
        Write-Host "Restauration de la regle ACL sur models/ ..."
        $restoreAcl = Get-Acl -LiteralPath $modelsRoot
        foreach ($rule in $denyRules) {
            [void]$restoreAcl.AddAccessRule($rule)
        }
        Set-Acl -LiteralPath $modelsRoot -AclObject $restoreAcl
    }
}

Write-Host ""
Write-Host "Nettoyage termine."
