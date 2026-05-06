from __future__ import annotations

from src.nlp.emotion_fusion import (
    fuse_text_and_voice_emotion,
    voice_source_from_confidence,
)
from src.nlp.emotion_text import predict_emotion_with_confidence_from_text
from src.nlp.therapist_agent import generate_reply
from src.tts.synthesizer import synthesize


def _build_non_english_reply(language_name: str) -> str:
    safe_language = str(language_name or "").strip() or "unknown language"
    return (
        f"I understand that you are speaking {safe_language}. "
        "I do not understand your language. "
        "Please speak in English. "
        "Otherwise, please consult a real therapist."
    )


def run_session() -> None:
    print("Therapist IA")
    print("Choisis un mode: 1) texte  2) audio")
    mode = input("Mode (1/2): ").strip()

    if mode == "1":
        user_text = input("Ecris ton message: ").strip()
        emotion, text_confidence, text_source = predict_emotion_with_confidence_from_text(user_text)
        print(
            f"Emotion detectee (texte): {emotion} "
            f"[confidence={text_confidence:.2f}, source={text_source}]"
        )
        reply = generate_reply(user_text, emotion=emotion)
        print(f"Therapist: {reply}")
        tts_path = synthesize(reply)
        print(f"Sortie reponse enregistree: {tts_path}")
        return

    if mode == "2":
        from src.nlp.emotion_audio import predict_emotion_from_audio_with_confidence
        from src.stt.transcriber import (
            transcribe_live_until_enter,
            transcribe_with_language_detection,
        )

        print("Mode audio: 1) Parler au micro (STT live)  2) Televerser un fichier audio")
        audio_mode = input("Choix audio (1/2): ").strip()

        stt_meta: dict[str, object] = {}
        if audio_mode == "1":
            _live_transcript, audio_path = transcribe_live_until_enter(chunk_sec=2)
            stt_meta = transcribe_with_language_detection(audio_path)
            transcript = str(stt_meta.get("text", _live_transcript))
            print(f"Transcription STT finale: {transcript}")
        elif audio_mode == "2":
            audio_path = input("Chemin du fichier audio (.wav/.mp3/...): ").strip().strip('"')
            stt_meta = transcribe_with_language_detection(audio_path)
            transcript = str(stt_meta.get("text", ""))
            print(f"Transcription STT: {transcript}")
        else:
            print("Choix audio invalide.")
            return

        detected_language_name = str(
            stt_meta.get("detected_language_name", "Unknown language")
        )
        language_supported = bool(stt_meta.get("language_supported", True))
        if not language_supported:
            reply = _build_non_english_reply(detected_language_name)
            print(f"Therapist: {reply}")
            tts_path = synthesize(reply)
            print(f"Sortie reponse enregistree: {tts_path}")
            return

        try:
            if not audio_path:
                raise RuntimeError("Audio live non disponible.")
            audio_emotion, audio_confidence = predict_emotion_from_audio_with_confidence(audio_path)
            audio_source = voice_source_from_confidence(audio_confidence)
        except Exception as exc:
            audio_emotion = "neutral"
            audio_confidence = 0.0
            audio_source = "voice-unavailable"
            print(f"Emotion model indisponible ({exc}), fallback sur 'neutral'.")

        text_emotion, text_confidence, text_source = predict_emotion_with_confidence_from_text(transcript)
        emotion, emotion_source = fuse_text_and_voice_emotion(
            text_emotion=text_emotion,
            text_confidence=text_confidence,
            audio_emotion=audio_emotion,
            audio_confidence=audio_confidence,
            audio_source=audio_source,
        )

        print(f"Emotion detectee (texte): {text_emotion} [confidence={text_confidence:.2f}, source={text_source}]")
        print(f"Emotion detectee (voix): {audio_emotion} [confidence={audio_confidence:.2f}, source={audio_source}]")
        print(f"Emotion finale: {emotion} [fusion={emotion_source}]")
        reply = generate_reply(transcript, emotion=emotion)
        print(f"Therapist: {reply}")
        tts_path = synthesize(reply)
        print(f"Sortie reponse enregistree: {tts_path}")
        return

    print("Mode invalide. Relance et choisis 1 ou 2.")
