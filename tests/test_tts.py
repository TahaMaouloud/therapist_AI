from src.tts.synthesizer import _get_default_tts_lang, _normalize_lang_tag


def test_default_tts_lang_is_en_by_default(monkeypatch):
    monkeypatch.delenv("TTS_LANG", raising=False)
    monkeypatch.delenv("LANG", raising=False)
    assert _get_default_tts_lang() == "en-us"


def test_default_tts_lang_uses_tts_lang(monkeypatch):
    monkeypatch.setenv("TTS_LANG", "en-gb")
    assert _get_default_tts_lang() == "en-gb"


def test_normalize_lang_tag_from_locale():
    assert _normalize_lang_tag("en_US.UTF-8") == "en-us"
