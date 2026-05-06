from __future__ import annotations

from datetime import datetime
from pathlib import Path


def record_audio(
    duration_sec: int = 8, sample_rate: int = 16000, channels: int = 1
) -> str:
    import sounddevice as sd
    import soundfile as sf

    output_dir = Path("data/raw_audio")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"

    print(f"Recording {duration_sec}s... parle maintenant.")
    audio = sd.rec(
        int(duration_sec * sample_rate),
        samplerate=sample_rate,
        channels=channels,
        dtype="float32",
    )
    sd.wait()
    sf.write(output_path, audio, sample_rate)
    print(f"Audio saved: {output_path}")
    return str(output_path)
