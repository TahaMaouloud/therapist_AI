from __future__ import annotations

NEGATIVE_EMOTIONS = {"angry", "fearful", "sad", "disgust"}

AUDIO_CONFIDENCE_THRESHOLD = 0.45
AUDIO_PRIMARY_CONFIDENCE = 0.72
TEXT_CONFIDENCE_THRESHOLD = 0.55
TEXT_PRIMARY_CONFIDENCE = 0.72
POSITIVE_TEXT_OVERRIDE_CONFIDENCE = 0.85


def normalize_emotion(label: str | None) -> str:
    value = str(label or "").strip().lower()
    return value or "neutral"


def voice_source_from_confidence(audio_confidence: float) -> str:
    return "voice" if float(audio_confidence) >= AUDIO_CONFIDENCE_THRESHOLD else "voice-low-confidence"


def fuse_text_and_voice_emotion(
    text_emotion: str,
    text_confidence: float,
    audio_emotion: str,
    audio_confidence: float,
    audio_source: str,
) -> tuple[str, str]:
    text = normalize_emotion(text_emotion)
    audio = normalize_emotion(audio_emotion)
    text_conf = float(text_confidence)
    audio_conf = float(audio_confidence)

    if text == audio:
        return audio, "agreement"

    if audio_source != "voice":
        if text != "neutral" and text_conf >= TEXT_CONFIDENCE_THRESHOLD:
            return text, "text-priority-low-audio"
        return (text if text != "neutral" else audio), "fallback-low-audio"

    if text == "happy" and audio in NEGATIVE_EMOTIONS:
        if text_conf >= TEXT_CONFIDENCE_THRESHOLD and audio_conf < POSITIVE_TEXT_OVERRIDE_CONFIDENCE:
            return text, "text-positive-override"

    if text != "neutral" and text_conf < TEXT_CONFIDENCE_THRESHOLD:
        return audio, "voice-priority-low-text"

    if text != "neutral" and text_conf >= TEXT_PRIMARY_CONFIDENCE and audio_conf < AUDIO_PRIMARY_CONFIDENCE:
        return text, "text-priority-high-confidence"

    if text != "neutral" and audio_conf < AUDIO_PRIMARY_CONFIDENCE:
        return text, "text-priority"

    return audio, "voice-priority"
