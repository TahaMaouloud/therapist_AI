param(
    [string]$DataPath = "data/goemotions_3.csv",
    [string]$TextCol = "text",
    [string]$LabelCol = "emotion",
    [string]$ModelName = "models/emotion_bert",
    [float]$TestSize = 0.2,
    [float]$ValidationSize = 0.1,
    [float]$NumTrainEpochs = 4.0,
    [int]$TrainBatchSize = 16,
    [int]$EvalBatchSize = 32,
    [int]$EarlyStoppingPatience = 2,
    [string]$ModelOutDir = "models/emotion_bert",
    [string]$MetricsOutPath = "models/emotion_metrics.json",
    [bool]$AutoPrepareGoEmotions = $true,
    [string]$PreparedDataPath = "data/processed/goemotions_5class.csv",
    [string]$PreparedMetricsPath = "data/processed/goemotions_5class_metrics.json",
    [bool]$DedupeByText = $true,
    [int]$MaxPerClass = 0,
    [string]$ResumeFromCheckpoint = "",
    [switch]$SkipTrain,
    [switch]$LocalFilesOnly
)

$PythonExe = if (Test-Path ".\\.venv_dl\\Scripts\\python.exe") { ".\\.venv_dl\\Scripts\\python.exe" } else { "python" }
$TrainingDataPath = $DataPath

function Test-GoEmotionsRawCsv {
    param([string]$Path)
    if (-not (Test-Path $Path)) {
        return $false
    }
    try {
        $header = Get-Content -Path $Path -TotalCount 1
    } catch {
        return $false
    }
    if (-not $header) {
        return $false
    }

    return (
        $header.Contains("text,id,author,subreddit") -and
        $header.Contains("example_very_unclear") -and
        $header.Contains("admiration") -and
        $header.Contains("anger") -and
        $header.Contains("neutral")
    )
}

if ($AutoPrepareGoEmotions -and (Test-GoEmotionsRawCsv -Path $DataPath)) {
    Write-Host "GoEmotions brut detecte, conversion vers 5 emotions en cours..."
    $PrepareArgs = @(
        "-m", "src.nlp.prepare_goemotions_5class",
        "--input", $DataPath,
        "--output", $PreparedDataPath,
        "--metrics-out", $PreparedMetricsPath,
        "--min-top-votes", "1",
        "--min-margin", "1",
        "--max-unclear-ratio", "0.5"
    )

    if ($DedupeByText) {
        $PrepareArgs += "--dedupe-by-text"
    }

    if ($MaxPerClass -gt 0) {
        $PrepareArgs += @("--max-per-class", "$MaxPerClass")
    }

    & $PythonExe @PrepareArgs
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }

    $TrainingDataPath = $PreparedDataPath
    $TextCol = "text"
    $LabelCol = "emotion"
    Write-Host "Dataset converti: $TrainingDataPath"
}

$Args = @(
    "-m", "src.nlp.train_emotion_model",
    "--data", $TrainingDataPath,
    "--text-col", $TextCol,
    "--label-col", $LabelCol,
    "--model-name", $ModelName,
    "--test-size", "$TestSize",
    "--validation-size", "$ValidationSize",
    "--num-train-epochs", "$NumTrainEpochs",
    "--train-batch-size", "$TrainBatchSize",
    "--eval-batch-size", "$EvalBatchSize",
    "--early-stopping-patience", "$EarlyStoppingPatience",
    "--model-out-dir", $ModelOutDir,
    "--metrics-out", $MetricsOutPath
)

if ($LocalFilesOnly) {
    $Args += "--local-files-only"
}

if ($ResumeFromCheckpoint -ne "") {
    $Args += @("--resume-from-checkpoint", $ResumeFromCheckpoint)
}

if ($SkipTrain) {
    $Args += "--skip-train"
}

& $PythonExe @Args
exit $LASTEXITCODE
