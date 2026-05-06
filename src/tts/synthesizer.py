from __future__ import annotations

import base64
import os
import shutil
import subprocess
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

_ENGINE_LOCK = threading.Lock()
_TtsEngineFn = Callable[[str, Path, str, int], str | None]


def _env_bool(name: str, default: bool) -> bool:
    raw = str(os.getenv(name, str(default))).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _normalize_lang_tag(raw_lang: str | None) -> str:
    value = str(raw_lang or "").strip().lower().replace("_", "-")
    if not value:
        return "en-us"

    value = value.split(".", 1)[0]
    if value in {"c", "c-utf-8", "posix"}:
        return "en-us"
    if value == "fr":
        return "fr-fr"
    if value == "en":
        return "en-us"
    return value


def _get_default_tts_lang() -> str:
    lang = os.getenv("TTS_LANG") or os.getenv("LANG") or "en-us"
    return _normalize_lang_tag(lang)


_DEFAULT_LANG = _get_default_tts_lang()


def _parse_timeout_seconds(name: str, default: float, minimum: float, maximum: float) -> float:
    raw = str(os.getenv(name, str(default))).strip()
    try:
        value = float(raw)
    except Exception:
        value = default
    return max(minimum, min(maximum, value))


def _parse_tts_rate() -> int:
    rate_env = os.getenv("TTS_RATE", "165").strip()
    try:
        return int(rate_env)
    except Exception:
        return 165


def _output_dirs() -> tuple[Path, Path]:
    base_dir = Path("data/outputs")
    tts_dir = base_dir / "tts"
    base_dir.mkdir(parents=True, exist_ok=True)
    tts_dir.mkdir(parents=True, exist_ok=True)
    return base_dir, tts_dir


def _pick_voice_id(engine: object, lang: str) -> str | None:
    lang = _normalize_lang_tag(lang)
    voices = engine.getProperty("voices") or []
    preferred = [lang]
    if lang.startswith("fr"):
        preferred.extend(["french", "fr-fr"])
    if lang.startswith("en"):
        preferred.extend(["english", "en-us", "en_us", "en-gb", "en_gb"])

    for voice in voices:
        voice_id = str(getattr(voice, "id", ""))
        voice_name = str(getattr(voice, "name", ""))
        voice_langs = str(getattr(voice, "languages", ""))
        blob = f"{voice_id} {voice_name} {voice_langs}".lower()
        if any(key in blob for key in preferred):
            return voice_id
    return str(getattr(voices[0], "id", "")) if voices else None


def _synthesize_with_pyttsx3(
    text: str,
    output_path: Path,
    lang: str = _DEFAULT_LANG,
    tts_rate: int = 165,
) -> str | None:
    try:
        import pyttsx3
    except Exception as exc:
        return f"pyttsx3 indisponible: {exc}"

    timeout_seconds = _parse_timeout_seconds(
        "TTS_PYTTSX3_TIMEOUT_SECONDS",
        default=8.0,
        minimum=2.0,
        maximum=30.0,
    )
    state: dict[str, str | None] = {"error": None}

    def _worker() -> None:
        engine = None
        try:
            engine = pyttsx3.init()
            engine.setProperty("rate", tts_rate)
            voice_id = _pick_voice_id(engine, lang=lang)
            if voice_id:
                engine.setProperty("voice", voice_id)
            engine.save_to_file(text, str(output_path))
            engine.runAndWait()
        except Exception as exc:
            state["error"] = f"echec synthese pyttsx3: {exc}"
        finally:
            try:
                if engine is not None:
                    engine.stop()
            except Exception:
                pass

    with _ENGINE_LOCK:
        worker = threading.Thread(target=_worker, daemon=True, name="pyttsx3-tts")
        worker.start()
        worker.join(timeout_seconds)
        if worker.is_alive():
            return f"pyttsx3 timeout apres {timeout_seconds:.1f}s"

    if state["error"] is not None:
        return str(state["error"])

    for _ in range(40):
        if output_path.exists() and output_path.stat().st_size > 128:
            return None
        time.sleep(0.05)
    return "audio WAV non cree par pyttsx3"


def _espeak_voice_for_lang(lang: str) -> str:
    normalized = _normalize_lang_tag(lang)
    if normalized.startswith("fr"):
        return "fr"
    if normalized.startswith("en-gb"):
        return "en-gb"
    if normalized.startswith("en"):
        return "en-us"
    return normalized


def _synthesize_with_espeak_ng(
    text: str,
    output_path: Path,
    lang: str = _DEFAULT_LANG,
    tts_rate: int = 165,
) -> str | None:
    executable = shutil.which("espeak-ng") or shutil.which("espeak")
    if not executable:
        return "espeak-ng indisponible"

    voice = _espeak_voice_for_lang(lang)
    safe_rate = max(80, min(320, int(tts_rate)))
    cmd = [executable, "-v", voice, "-s", str(safe_rate), "-w", str(output_path), text]
    try:
        completed = subprocess.run(
            cmd,
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except Exception as exc:
        return f"echec synthese espeak-ng: {exc}"

    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip()
        if stderr:
            return f"espeak-ng exit {completed.returncode}: {stderr}"
        return f"espeak-ng exit {completed.returncode}"

    if output_path.exists() and output_path.stat().st_size > 128:
        return None
    return "audio WAV non cree par espeak-ng"


def _synthesize_with_windows_sapi(
    text: str,
    output_path: Path,
    lang: str = _DEFAULT_LANG,
    tts_rate: int = 165,
) -> str | None:
    if os.name != "nt":
        return "windows-sapi non applicable"

    powershell = shutil.which("powershell") or shutil.which("pwsh")
    if not powershell:
        return "powershell indisponible"

    normalized = _normalize_lang_tag(lang)
    sapi_lang = ""
    if normalized.startswith("fr"):
        sapi_lang = "fr-FR"
    elif normalized.startswith("en"):
        sapi_lang = "en-US"

    sapi_rate = max(-10, min(10, int(round((int(tts_rate) - 165) / 15.0))))
    encoded_text = base64.b64encode(text.encode("utf-8")).decode("ascii")
    script = """param(
    [Parameter(Mandatory=$true)][string]$EncodedText,
    [Parameter(Mandatory=$true)][string]$OutputPath,
    [Parameter(Mandatory=$false)][string]$Culture = "",
    [Parameter(Mandatory=$false)][int]$Rate = 0
)
$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Speech
$text = [System.Text.Encoding]::UTF8.GetString([System.Convert]::FromBase64String($EncodedText))
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
try {
    if ($Culture) {
        try {
            $ci = New-Object System.Globalization.CultureInfo($Culture)
            $synth.SelectVoiceByHints(
                [System.Speech.Synthesis.VoiceGender]::NotSet,
                [System.Speech.Synthesis.VoiceAge]::NotSet,
                0,
                $ci
            )
        } catch {}
    }
    $synth.Rate = $Rate
    $synth.SetOutputToWaveFile($OutputPath)
    $synth.Speak($text)
} finally {
    $synth.Dispose()
}
"""
    with tempfile.NamedTemporaryFile(
        mode="w",
        suffix=".ps1",
        encoding="utf-8",
        delete=False,
    ) as handle:
        script_path = Path(handle.name)
        handle.write(script)
    try:
        completed = subprocess.run(
            [
                powershell,
                "-NoProfile",
                "-NonInteractive",
                "-ExecutionPolicy",
                "Bypass",
                "-File",
                str(script_path),
                "-EncodedText",
                encoded_text,
                "-OutputPath",
                str(output_path),
                "-Culture",
                sapi_lang,
                "-Rate",
                str(sapi_rate),
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=30,
        )
    except Exception as exc:
        return f"echec synthese windows-sapi: {exc}"
    finally:
        try:
            script_path.unlink(missing_ok=True)
        except Exception:
            pass

    if completed.returncode != 0:
        stderr = (completed.stderr or "").strip()
        stdout = (completed.stdout or "").strip()
        detail = stderr or stdout
        if detail:
            return f"windows-sapi exit {completed.returncode}: {detail}"
        return f"windows-sapi exit {completed.returncode}"

    if output_path.exists() and output_path.stat().st_size > 128:
        return None
    return "audio WAV non cree par windows-sapi"


def synthesize_payload(text: str, lang: str | None = None) -> dict:
    safe_text = (text or "").strip()
    if not safe_text:
        safe_text = "I am here to help you."

    requested_lang = _normalize_lang_tag(
        lang or os.getenv("TTS_LANG") or os.getenv("LANG") or _DEFAULT_LANG
    )
    tts_rate = _parse_tts_rate()

    base_dir, tts_dir = _output_dirs()
    wav_path = tts_dir / f"reply_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.wav"
    errors: list[str] = []
    selected_engine = ""
    pyttsx3_enabled = _env_bool("TTS_ENABLE_PYTTSX3", os.name != "nt")
    attempts: list[tuple[str, _TtsEngineFn]] = []
    if os.name == "nt":
        attempts.append(("windows-sapi", _synthesize_with_windows_sapi))
        attempts.append(("espeak-ng", _synthesize_with_espeak_ng))
        if pyttsx3_enabled:
            attempts.append(("pyttsx3", _synthesize_with_pyttsx3))
        else:
            errors.append("pyttsx3: disabled on Windows (set TTS_ENABLE_PYTTSX3=true to enable)")
    else:
        if pyttsx3_enabled:
            attempts.append(("pyttsx3", _synthesize_with_pyttsx3))
        attempts.append(("espeak-ng", _synthesize_with_espeak_ng))
    for engine_name, engine_fn in attempts:
        error = engine_fn(safe_text, wav_path, lang=requested_lang, tts_rate=tts_rate)
        if error is None:
            selected_engine = engine_name
            break
        errors.append(f"{engine_name}: {error}")

    if selected_engine and wav_path.exists():
        encoded = base64.b64encode(wav_path.read_bytes()).decode("ascii")
        return {
            "tts_path": str(wav_path),
            "tts_audio_base64": encoded,
            "tts_audio_mime": "audio/wav",
            "tts_engine": selected_engine,
            "tts_error": "",
        }

    fallback_txt = base_dir / "reply.txt"
    fallback_txt.write_text(safe_text, encoding="utf-8")
    return {
        "tts_path": str(fallback_txt),
        "tts_audio_base64": "",
        "tts_audio_mime": "",
        "tts_engine": "fallback-text",
        "tts_error": " | ".join(errors) if errors else "tts indisponible",
    }


def synthesize(text: str) -> str:
    payload = synthesize_payload(text)
    return str(payload["tts_path"])
