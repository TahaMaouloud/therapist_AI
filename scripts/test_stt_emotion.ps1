param(
    [string]$AudioPath = "",
    [string]$SttModel = "small",
    [string]$SttLanguage = "en"
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

if ([string]::IsNullOrWhiteSpace($AudioPath)) {
    $latest = Get-ChildItem -File "data/raw_audio/uploads" -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1

    if (-not $latest) {
        throw "Aucun fichier audio trouve dans data/raw_audio/uploads. Passe -AudioPath."
    }
    $AudioPath = $latest.FullName
}

if (-not (Test-Path $AudioPath)) {
    throw "Audio introuvable: $AudioPath"
}

$pythonCandidates = @(
    (Join-Path $root ".venv_dl\Scripts\python.exe"),
    (Join-Path $root ".venv\Scripts\python.exe"),
    "python"
)

$pythonExe = $null
foreach ($candidate in $pythonCandidates) {
    if ($candidate -eq "python") {
        $pythonExe = $candidate
        break
    }
    if (Test-Path $candidate) {
        $pythonExe = $candidate
        break
    }
}

if (-not $pythonExe) {
    throw "Python introuvable."
}

$env:STT_MODEL = $SttModel
$env:STT_LANGUAGE = $SttLanguage

$pyCode = @'
from pathlib import Path
import os
import sys

from src.nlp.emotion_audio import predict_emotion_from_audio_with_confidence
from src.nlp.emotion_text import predict_emotion_from_text
from src.stt.transcriber import transcribe


def fuse_emotion(audio_emotion: str, audio_conf: float, text_emotion: str) -> str:
    if text_emotion != "neutral":
        if audio_conf < 0.75:
            return text_emotion
        if audio_emotion == "neutral":
            return text_emotion
        if text_emotion in {"fearful", "sad", "angry"} and audio_emotion in {
            "neutral",
            "happy",
            "surprised",
        }:
            return text_emotion
    return audio_emotion


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("Usage: provide audio_path argument")

    audio_path = str(Path(sys.argv[1]).resolve())
    if not Path(audio_path).exists():
        raise SystemExit(f"Audio introuvable: {audio_path}")

    stt_model = os.getenv("STT_MODEL", "small")
    stt_lang = os.getenv("STT_LANGUAGE", "en")

    transcript = transcribe(
        audio_path,
        model=stt_model,
        language=stt_lang,
    )
    audio_emotion, confidence = predict_emotion_from_audio_with_confidence(audio_path)
    text_emotion = predict_emotion_from_text(transcript)
    fused = fuse_emotion(audio_emotion, float(confidence), text_emotion)

    print(f"FILE={audio_path}")
    print("STT_ENGINE=faster-whisper")
    print(f"STT_MODEL={stt_model}")
    print(f"STT_LANGUAGE={stt_lang}")
    print(f"TRANSCRIPT={transcript}")

    if not transcript.strip():
        print("EMOTION=unavailable")
        print("EMOTION_CONFIDENCE=0.0000")
        print("NOTE=No speech detected in this file.")
        return

    print(f"AUDIO_EMOTION={audio_emotion}")
    print(f"TEXT_EMOTION={text_emotion}")
    print(f"EMOTION={fused}")
    print(f"EMOTION_CONFIDENCE={float(confidence):.4f}")


if __name__ == "__main__":
    main()
'@

$pyCode | & $pythonExe - $AudioPath
