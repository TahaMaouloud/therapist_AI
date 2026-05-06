from __future__ import annotations

import math
import os
import threading
from datetime import datetime
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

_WHISPER_MODELS: dict[tuple[str, str, str], object] = {}
_DEFAULT_MODEL_SIZE = (os.getenv("STT_MODEL", "small") or "small").strip() or "small"
_DEFAULT_LANGUAGE = (os.getenv("STT_LANGUAGE", "en") or "en").strip() or "en"
_DEFAULT_DEVICE = (os.getenv("STT_DEVICE", "cpu") or "cpu").strip() or "cpu"
_DEFAULT_COMPUTE_TYPE = (os.getenv("STT_COMPUTE_TYPE", "int8") or "int8").strip() or "int8"
_DEFAULT_INPUT_DEVICE = (os.getenv("STT_INPUT_DEVICE", "") or "").strip() or None
_MIN_RMS = float((os.getenv("STT_MIN_RMS", "0.01") or "0.01").strip() or "0.01")

_DECISIVE_TRANSCRIBE_OPTIONS = {
    "vad_filter": True,
    "vad_parameters": {"min_silence_duration_ms": 450, "speech_pad_ms": 250},
    "beam_size": 3,
    "best_of": 3,
    "temperature": 0.0,
    "condition_on_previous_text": True,
    "repetition_penalty": 1.05,
    "no_repeat_ngram_size": 2,
    "compression_ratio_threshold": 2.2,
    "log_prob_threshold": -1.1,
    "no_speech_threshold": 0.55,
}

_RELAXED_TRANSCRIBE_OPTIONS = {
    "vad_filter": False,
    "beam_size": 5,
    "best_of": 5,
    "temperature": 0.0,
    "condition_on_previous_text": False,
    "repetition_penalty": 1.02,
    "compression_ratio_threshold": 2.6,
    "log_prob_threshold": -1.6,
    "no_speech_threshold": 0.35,
}

_SUPPORTED_STT_LANGUAGES = {"en"}
_LANGUAGE_NAME_MAP = {
    "af": "Afrikaans",
    "ar": "Arabic",
    "bn": "Bengali",
    "bg": "Bulgarian",
    "ca": "Catalan",
    "cs": "Czech",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "et": "Estonian",
    "fa": "Persian",
    "fi": "Finnish",
    "fr": "French",
    "gu": "Gujarati",
    "he": "Hebrew",
    "hi": "Hindi",
    "hr": "Croatian",
    "hu": "Hungarian",
    "id": "Indonesian",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "lt": "Lithuanian",
    "lv": "Latvian",
    "ml": "Malayalam",
    "mr": "Marathi",
    "nl": "Dutch",
    "no": "Norwegian",
    "pa": "Punjabi",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "sk": "Slovak",
    "sl": "Slovenian",
    "sv": "Swedish",
    "sw": "Swahili",
    "ta": "Tamil",
    "te": "Telugu",
    "th": "Thai",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "ur": "Urdu",
    "vi": "Vietnamese",
    "zh": "Chinese",
}


def _normalize_language_tag(language: str | None) -> str | None:
    if language is None:
        return None
    cleaned = language.strip().lower()
    if not cleaned:
        return None
    if cleaned in {"auto", "none", "null"}:
        return None
    cleaned = cleaned.split(".", 1)[0].replace("-", "_")
    if cleaned in {"c", "posix"}:
        return None
    if "_" in cleaned:
        cleaned = cleaned.split("_", 1)[0]
    return cleaned or None


def _resolve_language(language: str | None) -> str | None:
    if language is None:
        return _normalize_language_tag(_DEFAULT_LANGUAGE)
    return _normalize_language_tag(language)


def _resolve_input_device(input_device: str | int | None) -> str | int | None:
    raw = _DEFAULT_INPUT_DEVICE if input_device is None else input_device
    if raw is None:
        return None
    cleaned = str(raw).strip()
    if not cleaned or cleaned.lower() == "auto":
        return None
    if cleaned.isdigit():
        return int(cleaned)
    return cleaned


def _language_name_from_tag(language: str | None) -> str:
    normalized = _normalize_language_tag(language)
    if normalized is None:
        return "Unknown language"
    return _LANGUAGE_NAME_MAP.get(normalized, normalized.upper())


def _is_supported_language(language: str | None) -> bool:
    normalized = _normalize_language_tag(language)
    if normalized is None:
        # Unknown language should not hard-block the user.
        return True
    return normalized in _SUPPORTED_STT_LANGUAGES


def _transcribe_auto_with_info(
    audio_path: str, model_size: str | None = None
) -> tuple[str, str | None, float]:
    model = _get_local_model(model_size=model_size)
    detected_language: str | None = None
    detected_language_probability = 0.0

    first_pass = dict(_DECISIVE_TRANSCRIBE_OPTIONS)
    segments, info = model.transcribe(audio_path, **first_pass)
    detected_language = _normalize_language_tag(getattr(info, "language", None))
    try:
        detected_language_probability = float(
            getattr(info, "language_probability", 0.0) or 0.0
        )
    except Exception:
        detected_language_probability = 0.0
    primary_text = _join_decisive_segments(list(segments))
    if primary_text:
        return primary_text, detected_language, detected_language_probability

    second_pass = dict(_RELAXED_TRANSCRIBE_OPTIONS)
    segments, info = model.transcribe(audio_path, **second_pass)
    second_language = _normalize_language_tag(getattr(info, "language", None))
    if second_language is not None:
        detected_language = second_language
    try:
        second_probability = float(getattr(info, "language_probability", 0.0) or 0.0)
        if second_probability > detected_language_probability:
            detected_language_probability = second_probability
    except Exception:
        pass
    relaxed_text = _join_decisive_segments(list(segments))
    return relaxed_text, detected_language, detected_language_probability

def _get_local_model(model_size: str | None = None):
    resolved_model_size = (model_size or _DEFAULT_MODEL_SIZE).strip() or _DEFAULT_MODEL_SIZE
    cache_key = (resolved_model_size, _DEFAULT_DEVICE, _DEFAULT_COMPUTE_TYPE)
    model = _WHISPER_MODELS.get(cache_key)
    if model is None:
        from faster_whisper import WhisperModel

        model = WhisperModel(
            resolved_model_size,
            device=_DEFAULT_DEVICE,
            compute_type=_DEFAULT_COMPUTE_TYPE,
        )
        _WHISPER_MODELS[cache_key] = model
    return model


def _clean_text(text: str) -> str:
    return " ".join(text.split()).strip()


def _chunk_rms(audio_chunk: np.ndarray) -> float:
    try:
        arr = np.asarray(audio_chunk, dtype=np.float32)
        if arr.size == 0:
            return 0.0
        return float(np.sqrt(np.mean(np.square(arr))))
    except Exception:
        return 0.0


def _join_decisive_segments(segments: list[object]) -> str:
    raw_parts: list[str] = []
    decisive_parts: list[str] = []
    last_text = ""

    for seg in segments:
        text = _clean_text(str(getattr(seg, "text", "")))
        if not text:
            continue

        raw_parts.append(text)

        avg_logprob = float(getattr(seg, "avg_logprob", -1.0))
        no_speech_prob = float(getattr(seg, "no_speech_prob", 0.0))
        if avg_logprob < -1.3 and no_speech_prob > 0.75:
            continue

        normalized = text.casefold()
        if normalized == last_text:
            continue

        decisive_parts.append(text)
        last_text = normalized

    if decisive_parts:
        return " ".join(decisive_parts).strip()
    return " ".join(raw_parts).strip()

def _transcribe_with_local_whisper(
    audio_path: str, language: str | None = None, model_size: str | None = None
) -> str:
    model = _get_local_model(model_size=model_size)
    resolved_language = _resolve_language(language)

    first_pass = dict(_DECISIVE_TRANSCRIBE_OPTIONS)
    if resolved_language:
        first_pass["language"] = resolved_language
    segments, _info = model.transcribe(audio_path, **first_pass)
    primary_text = _join_decisive_segments(list(segments))
    if primary_text:
        return primary_text

    # Retry with less aggressive decoding when primary pass returns empty text.
    second_pass = dict(_RELAXED_TRANSCRIBE_OPTIONS)
    if resolved_language:
        second_pass["language"] = resolved_language
    segments, _info = model.transcribe(audio_path, **second_pass)
    relaxed_text = _join_decisive_segments(list(segments))
    if relaxed_text:
        return relaxed_text

    # Last chance: auto-detect language if forced language was too restrictive.
    if resolved_language:
        auto_lang_pass = dict(_RELAXED_TRANSCRIBE_OPTIONS)
        segments, _info = model.transcribe(audio_path, **auto_lang_pass)
        auto_text = _join_decisive_segments(list(segments))
        if auto_text:
            return auto_text

    return ""


def transcribe(
    audio_path: str,
    model: str | None = None,
    language: str | None = None,
) -> str:
    try:
        return _transcribe_with_local_whisper(
            audio_path=audio_path, language=language, model_size=model
        )
    except Exception as exc:
        return f"[Erreur STT local] {exc}"


def transcribe_with_language_detection(
    audio_path: str,
    model: str | None = None,
    language: str | None = None,
) -> dict[str, object]:
    try:
        transcript, detected_language, detected_probability = _transcribe_auto_with_info(
            audio_path=audio_path, model_size=model
        )

        # Preserve the client language transcript by decoding again in detected language.
        if detected_language:
            in_lang_text = _transcribe_with_local_whisper(
                audio_path=audio_path,
                language=detected_language,
                model_size=model,
            )
            if in_lang_text:
                transcript = in_lang_text

        # If still empty, fallback to auto-language decoding (not forced English).
        if not transcript:
            transcript = _transcribe_with_local_whisper(
                audio_path=audio_path, language="auto", model_size=model
            )

        language_name = _language_name_from_tag(detected_language)
        language_supported = _is_supported_language(detected_language)

        return {
            "text": str(transcript),
            "detected_language": detected_language,
            "detected_language_name": language_name,
            "detected_language_probability": float(detected_probability),
            "language_supported": bool(language_supported),
            "error": "",
        }
    except Exception as exc:
        return {
            "text": f"[Erreur STT local] {exc}",
            "detected_language": None,
            "detected_language_name": "Unknown language",
            "detected_language_probability": 0.0,
            "language_supported": True,
            "error": str(exc),
        }


def transcribe_live(
    duration_sec: int = 12,
    chunk_sec: int = 2,
    sample_rate: int = 16000,
    channels: int = 1,
    model: str | None = None,
    language: str | None = None,
    input_device: str | int | None = None,
) -> tuple[str, str]:
    """
    Live STT by short chunks:
    - records audio chunk by chunk
    - transcribes each chunk immediately
    - prints incremental transcript
    Returns (full_transcript, saved_audio_path).
    """
    try:
        import sounddevice as sd
        import soundfile as sf
    except Exception as exc:
        return (f"[Erreur dependances STT live] {exc}", "")

    chunk_sec = max(1, int(chunk_sec))
    total_chunks = max(1, math.ceil(max(1, int(duration_sec)) / chunk_sec))
    resolved_input_device = _resolve_input_device(input_device)
    all_chunks: list[np.ndarray] = []
    transcript_parts: list[str] = []
    voiced_chunks = 0

    print(
        f"STT live demarre ({duration_sec}s, chunk={chunk_sec}s, input={resolved_input_device if resolved_input_device is not None else 'default'}). Parle maintenant..."
    )
    for idx in range(total_chunks):
        rec_kwargs: dict[str, object] = {
            "frames": int(chunk_sec * sample_rate),
            "samplerate": sample_rate,
            "channels": channels,
            "dtype": "float32",
        }
        if resolved_input_device is not None:
            rec_kwargs["device"] = resolved_input_device
        audio_chunk = sd.rec(**rec_kwargs)
        sd.wait()
        all_chunks.append(audio_chunk.copy())
        rms = _chunk_rms(audio_chunk)
        if rms < _MIN_RMS:
            print(f"[Live {idx + 1}/{total_chunks}] silence (rms={rms:.4f})")
            continue
        voiced_chunks += 1

        with NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            sf.write(tmp_path, audio_chunk, sample_rate)
            part = _transcribe_with_local_whisper(
                audio_path=str(tmp_path), language=language, model_size=model
            )
            if part:
                transcript_parts.append(part)
            current = " ".join(transcript_parts).strip()
            print(f"[Live {idx + 1}/{total_chunks}] {current}")
        except Exception as exc:
            print(f"[Live {idx + 1}/{total_chunks}] Erreur STT: {exc}")
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass

    output_dir = Path("data/raw_audio")
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_path = output_dir / f"session_live_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
    full_audio = np.concatenate(all_chunks, axis=0)
    sf.write(audio_path, full_audio, sample_rate)

    full_transcript = " ".join(transcript_parts).strip()
    if not full_transcript and voiced_chunks > 0:
        try:
            full_transcript = _transcribe_with_local_whisper(
                audio_path=str(audio_path), language=language, model_size=model
            )
        except Exception as exc:
            print(f"[Live final] Erreur STT: {exc}")

    return (full_transcript, str(audio_path))


def transcribe_live_until_enter(
    chunk_sec: int = 2,
    sample_rate: int = 16000,
    channels: int = 1,
    model: str | None = None,
    language: str | None = None,
    input_device: str | int | None = None,
) -> tuple[str, str]:
    """
    Live STT with keyboard stop:
    - records chunk by chunk
    - transcribes each chunk immediately
    - stops when user presses Enter
    Returns (full_transcript, saved_audio_path).
    """
    try:
        import sounddevice as sd
        import soundfile as sf
    except Exception as exc:
        return (f"[Erreur dependances STT live] {exc}", "")

    all_chunks: list[np.ndarray] = []
    transcript_parts: list[str] = []
    stop_event = threading.Event()
    resolved_input_device = _resolve_input_device(input_device)
    voiced_chunks = 0

    def wait_for_enter() -> None:
        try:
            input("Appuie sur Entree pour arreter l'enregistrement...\n")
        except EOFError:
            print("Entree indisponible sur ce terminal. Arret auto en cours...")
        finally:
            stop_event.set()

    threading.Thread(target=wait_for_enter, daemon=True).start()
    print(
        f"STT live demarre (chunk={chunk_sec}s, input={resolved_input_device if resolved_input_device is not None else 'default'}). Parle maintenant..."
    )

    idx = 0
    chunk_sec = max(1, int(chunk_sec))

    while not stop_event.is_set():
        idx += 1
        rec_kwargs: dict[str, object] = {
            "frames": int(chunk_sec * sample_rate),
            "samplerate": sample_rate,
            "channels": channels,
            "dtype": "float32",
        }
        if resolved_input_device is not None:
            rec_kwargs["device"] = resolved_input_device
        audio_chunk = sd.rec(**rec_kwargs)
        sd.wait()
        all_chunks.append(audio_chunk.copy())
        rms = _chunk_rms(audio_chunk)
        if rms < _MIN_RMS:
            print(f"[Live {idx}] silence (rms={rms:.4f})")
            continue
        voiced_chunks += 1

        with NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        try:
            sf.write(tmp_path, audio_chunk, sample_rate)
            part = _transcribe_with_local_whisper(
                audio_path=str(tmp_path), language=language, model_size=model
            )
            if part:
                transcript_parts.append(part)
            current = " ".join(transcript_parts).strip()
            print(f"[Live {idx}] {current}")
        except Exception as exc:
            print(f"[Live {idx}] Erreur STT: {exc}")
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass

    if not all_chunks:
        return ("", "")

    output_dir = Path("data/raw_audio")
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_path = output_dir / f"session_live_{datetime.now().strftime('%Y%m%d_%H%M%S')}.wav"
    full_audio = np.concatenate(all_chunks, axis=0)
    sf.write(audio_path, full_audio, sample_rate)

    full_transcript = " ".join(transcript_parts).strip()
    if not full_transcript and voiced_chunks > 0:
        try:
            full_transcript = _transcribe_with_local_whisper(
                audio_path=str(audio_path), language=language, model_size=model
            )
        except Exception as exc:
            print(f"[Live final] Erreur STT: {exc}")

    return (full_transcript, str(audio_path))
