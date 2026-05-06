from src.nlp.emotion_text import predict_emotion_with_confidence_from_text


def test_violent_intent_overrides_happy_like_text() -> None:
    emotion, confidence, source = predict_emotion_with_confidence_from_text(
        "hello guys how are you felling today i wanna tell you today i ganna kill my friend"
    )
    assert emotion == "angry"
    assert confidence >= 0.97
    assert source == "safety-violence-override"


def test_self_harm_text_overrides_to_fearful() -> None:
    emotion, confidence, source = predict_emotion_with_confidence_from_text(
        "I want to die and I might hurt myself tonight."
    )
    assert emotion == "fearful"
    assert confidence >= 0.97
    assert source == "safety-self-harm-override"
