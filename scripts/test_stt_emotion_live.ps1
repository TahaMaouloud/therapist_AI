param(
    [string]$SttModel = "small",
    [string]$SttLanguage = "en",
    [int]$ChunkSec = 2,
    [string]$InputDevice = "auto",
    [double]$MinRms = 0.01,
    [switch]$ListDevices
)

$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

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

$oldPythonPath = $env:PYTHONPATH
$oldPythonIoEncoding = $env:PYTHONIOENCODING
$env:PYTHONPATH = $root
$env:PYTHONIOENCODING = "utf-8"

if ($ListDevices) {
    $listCode = @'
import sounddevice as sd

devices = sd.query_devices()
for idx, dev in enumerate(devices):
    max_in = int(dev.get("max_input_channels", 0))
    max_out = int(dev.get("max_output_channels", 0))
    if max_in <= 0:
        continue
    name = str(dev.get("name", "unknown"))
    hostapi = dev.get("hostapi", "")
    print(f"{idx}: in={max_in} out={max_out} hostapi={hostapi} name={name}")
'@
    try {
        $listCode | & $pythonExe -
    } finally {
        $env:PYTHONPATH = $oldPythonPath
        $env:PYTHONIOENCODING = $oldPythonIoEncoding
    }
    return
}

$env:STT_MODEL = $SttModel
$env:STT_LANGUAGE = $SttLanguage
$env:STT_INPUT_DEVICE = $InputDevice
$env:STT_MIN_RMS = [string]$MinRms

$pyCode = @'
import os
import sys

from src.nlp.emotion_audio import predict_emotion_from_audio_with_confidence
from src.nlp.emotion_text import predict_emotion_from_text
from src.stt.transcriber import transcribe_live_until_enter


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
    chunk_sec = int(sys.argv[1]) if len(sys.argv) > 1 else 2

    stt_model = os.getenv("STT_MODEL", "small")
    stt_lang = os.getenv("STT_LANGUAGE", "en")
    stt_input_device = os.getenv("STT_INPUT_DEVICE", "auto")
    stt_min_rms = os.getenv("STT_MIN_RMS", "0.01")

    print("=== LIVE TEST: STT + EMOTION ===")
    print("STT_ENGINE=faster-whisper")
    print(f"STT_MODEL={stt_model}")
    print(f"STT_LANGUAGE={stt_lang}")
    print(f"STT_INPUT_DEVICE={stt_input_device}")
    print(f"STT_MIN_RMS={stt_min_rms}")
    print(f"CHUNK_SEC={chunk_sec}")
    print("Parle au micro, puis appuie sur Entree pour arreter.")

    transcript, audio_path = transcribe_live_until_enter(
        chunk_sec=chunk_sec,
        model=stt_model,
        language=stt_lang,
        input_device=stt_input_device,
    )

    print(f"FILE={audio_path}")
    print(f"TRANSCRIPT={(transcript if transcript else '<VIDE>')}")

    if not audio_path:
        print("EMOTION=unavailable")
        print("EMOTION_CONFIDENCE=0.0000")
        return

    if not transcript.strip():
        print("EMOTION=unavailable")
        print("EMOTION_CONFIDENCE=0.0000")
        print("NOTE=No speech detected. Check microphone input device or lower STT_MIN_RMS.")
        return

    audio_emotion, confidence = predict_emotion_from_audio_with_confidence(audio_path)
    text_emotion = predict_emotion_from_text(transcript)
    fused = fuse_emotion(audio_emotion, float(confidence), text_emotion)

    print(f"AUDIO_EMOTION={audio_emotion}")
    print(f"TEXT_EMOTION={text_emotion}")
    print(f"EMOTION={fused}")
    print(f"EMOTION_CONFIDENCE={float(confidence):.4f}")


if __name__ == "__main__":
    main()
'@

$tmpPy = Join-Path $env:TEMP ("therapist_live_test_{0}.py" -f $PID)
try {
    Set-Content -Path $tmpPy -Value $pyCode -Encoding UTF8
    & $pythonExe $tmpPy $ChunkSec
} finally {
    $env:PYTHONPATH = $oldPythonPath
    $env:PYTHONIOENCODING = $oldPythonIoEncoding
    if (Test-Path $tmpPy) {
        Remove-Item -Path $tmpPy -Force -ErrorAction SilentlyContinue
    }
}
