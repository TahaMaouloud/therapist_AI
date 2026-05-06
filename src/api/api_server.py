from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import threading
from typing import Any

from src.env_loader import load_local_env_file

load_local_env_file()

from fastapi import FastAPI
from pydantic import BaseModel

from src.nlp.emotion_audio import predict_emotion_top_k_from_audio
from src.nlp.emotion_fusion import (
    fuse_text_and_voice_emotion,
    normalize_emotion,
    voice_source_from_confidence,
)
from src.nlp.emotion_text import predict_emotion_with_confidence_from_text
from src.nlp.therapist_agent import (
    generate_reply,
    therapist_backend_status,
    therapist_start_warmup,
)
from src.stt.transcriber import transcribe_with_language_detection

app = FastAPI(title="Therapist API")
_WARMUP_LOCK = threading.Lock()
_WARMUP_STARTED = False


def _env_bool(name: str, default: bool) -> bool:
    raw = str(__import__("os").getenv(name, str(default))).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _warmup_therapist_model() -> None:
    try:
        therapist_start_warmup(wait=True)
    except Exception:
        return


@app.on_event("startup")
def startup_warmup() -> None:
    global _WARMUP_STARTED
    if not _env_bool("THERAPIST_WARMUP_ON_STARTUP", True):
        return
    with _WARMUP_LOCK:
        if _WARMUP_STARTED:
            return
        _WARMUP_STARTED = True
    thread = threading.Thread(target=_warmup_therapist_model, daemon=True)
    thread.start()


class TextRequest(BaseModel):
    text: str
    session_id: str | None = None
    history: list[dict[str, str]] | None = None


class AudioRequest(BaseModel):
    audio_path: str
    session_id: str | None = None
    history: list[dict[str, str]] | None = None


def _normalize_history(history: list[dict[str, str]] | None) -> list[dict[str, str]] | None:
    if not history:
        return None
    normalized: list[dict[str, str]] = []
    for item in history:
        role = str(item.get("role", "")).strip()
        content = str(item.get("content", "")).strip()
        if not role or not content:
            continue
        normalized.append({"role": role, "content": content})
    return normalized or None

def _build_non_english_reply(language_name: str) -> str:
    safe_language = str(language_name or "").strip() or "unknown language"
    return (
        f"Thank you for your message. I detected {safe_language}. "
        "At the moment, I can respond only in English. "
        "Please continue in English so I can support you better. "
        "If you prefer, you can also speak with a licensed therapist in your language."
    )


def _detect_voice_emotion(
    audio_path: str,
) -> tuple[str, float, str, list[dict[str, Any]]]:
    try:
        top2 = predict_emotion_top_k_from_audio(
            audio_path=audio_path,
            k=2,
        )
        normalized_top2 = [
            {"emotion": normalize_emotion(label), "confidence": float(confidence)}
            for label, confidence in top2
        ]
        if not normalized_top2:
            return "neutral", 0.0, "voice-unavailable", [{"emotion": "neutral", "confidence": 0.0}]

        primary = normalized_top2[0]
        source = voice_source_from_confidence(float(primary["confidence"]))
        return (
            str(primary["emotion"]),
            float(primary["confidence"]),
            source,
            normalized_top2,
        )
    except Exception:
        return "neutral", 0.0, "voice-unavailable", [{"emotion": "neutral", "confidence": 0.0}]


def _analyze_audio_parallel(
    audio_path: str,
) -> tuple[str, dict[str, Any], str, float, str, list[dict[str, Any]]]:
    with ThreadPoolExecutor(max_workers=2) as executor:
        transcript_future = executor.submit(transcribe_with_language_detection, audio_path)
        emotion_future = executor.submit(_detect_voice_emotion, audio_path)

        stt_meta = dict(transcript_future.result())
        transcript = str(stt_meta.get("text", ""))
        audio_emotion, audio_confidence, emotion_source, audio_top2 = emotion_future.result()

    return (
        transcript,
        stt_meta,
        audio_emotion,
        float(audio_confidence),
        emotion_source,
        audio_top2,
    )


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/diagnostics/therapist-backend")
def diagnostics_therapist_backend() -> dict:
    return therapist_backend_status()


@app.post("/session/text")
def session_text(req: TextRequest) -> dict:
    emotion, text_confidence, text_source = predict_emotion_with_confidence_from_text(req.text)
    reply = generate_reply(
        req.text,
        emotion=emotion,
        session_id=req.session_id,
        conversation_history=_normalize_history(req.history),
    )
    backend_info = therapist_backend_status()
    return {
        "input_text": req.text,
        "emotion": emotion,
        "text_emotion": emotion,
        "text_emotion_confidence": text_confidence,
        "text_emotion_source": text_source,
        "emotion_source": text_source,
        "reply_backend": backend_info.get("backend", "unknown"),
        "reply_backend_reason": backend_info.get("reason", ""),
        "reply": reply,
    }


@app.post("/session/audio")
def session_audio(req: AudioRequest) -> dict:
    transcript, stt_meta, audio_emotion, audio_confidence, audio_source, audio_top2 = (
        _analyze_audio_parallel(req.audio_path)
    )
    detected_language = stt_meta.get("detected_language")
    detected_language_name = str(stt_meta.get("detected_language_name", "Unknown language"))
    detected_language_probability = float(stt_meta.get("detected_language_probability", 0.0))
    language_supported = bool(stt_meta.get("language_supported", True))

    if not language_supported:
        reply = _build_non_english_reply(detected_language_name)
        return {
            "audio_path": req.audio_path,
            "transcript": transcript,
            "detected_language": detected_language,
            "detected_language_name": detected_language_name,
            "detected_language_probability": detected_language_probability,
            "language_supported": False,
            "emotion": "neutral",
            "audio_emotion": audio_emotion,
            "audio_confidence": audio_confidence,
            "audio_emotion_top2": audio_top2,
            "audio_emotion_primary": audio_top2[0] if audio_top2 else {"emotion": audio_emotion, "confidence": audio_confidence},
            "audio_emotion_secondary": audio_top2[1] if len(audio_top2) > 1 else None,
            "audio_source": audio_source,
            "emotion_source": "language-gate",
            "text_emotion": "neutral",
            "text_emotion_confidence": 0.0,
            "text_emotion_source": "language-gate",
            "reply": reply,
        }

    text_emotion, text_confidence, text_source = predict_emotion_with_confidence_from_text(transcript)
    emotion, emotion_source = fuse_text_and_voice_emotion(
        text_emotion=text_emotion,
        text_confidence=text_confidence,
        audio_emotion=audio_emotion,
        audio_confidence=audio_confidence,
        audio_source=audio_source,
    )
    reply = generate_reply(
        transcript,
        emotion=emotion,
        session_id=req.session_id,
        conversation_history=_normalize_history(req.history),
    )
    backend_info = therapist_backend_status()
    return {
        "audio_path": req.audio_path,
        "transcript": transcript,
        "detected_language": detected_language,
        "detected_language_name": detected_language_name,
        "detected_language_probability": detected_language_probability,
        "language_supported": True,
        "emotion": emotion,
        "audio_emotion": audio_emotion,
        "audio_confidence": audio_confidence,
        "audio_emotion_top2": audio_top2,
        "audio_emotion_primary": audio_top2[0] if audio_top2 else {"emotion": audio_emotion, "confidence": audio_confidence},
        "audio_emotion_secondary": audio_top2[1] if len(audio_top2) > 1 else None,
        "audio_source": audio_source,
        "emotion_source": emotion_source,
        "text_emotion": text_emotion,
        "text_emotion_confidence": text_confidence,
        "text_emotion_source": text_source,
        "reply_backend": backend_info.get("backend", "unknown"),
        "reply_backend_reason": backend_info.get("reason", ""),
        "reply": reply,
    }
