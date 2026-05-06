param(
    [string[]]$DatasetRoots = @("data/raw_audio/archive", "data/AudioWAV"),
    [string[]]$Models = @("wav2vec2")
)

$PythonExe = if (Test-Path ".\\.venv_dl\\Scripts\\python.exe") { ".\\.venv_dl\\Scripts\\python.exe" } else { "python" }

Write-Host "Python utilise: $PythonExe"
Write-Host "Modeles demandes: $($Models -join ', ')"

& $PythonExe -m src.nlp.train_emotion_audio_model `
  --dataset-roots $DatasetRoots `
  --models $Models `
  --model-out "models/emotion_audio_model.joblib" `
  --metrics-out "models/emotion_audio_metrics.json"

exit $LASTEXITCODE
