from __future__ import annotations

# Backward-compatible STT entrypoints.
# STT is now local-only through faster_whisper.
from src.stt.transcriber import transcribe, transcribe_live, transcribe_live_until_enter

__all__ = ["transcribe", "transcribe_live", "transcribe_live_until_enter"]
