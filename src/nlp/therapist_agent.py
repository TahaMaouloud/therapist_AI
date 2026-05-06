from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import threading
from collections import deque
from pathlib import Path
from typing import Iterable, Mapping

from src.env_loader import load_local_env_file

load_local_env_file()

# Therapist response engine:
# - prefers local LLaMA when available
# - keeps per-session history in memory
# - enforces safety guardrails
# - falls back to deterministic rule-based replies

THERAPIST_MAX_CHARS = 520
_DEFAULT_SESSION_ID = "default"


def _env_bool(name: str, default: bool) -> bool:
    raw = str(os.getenv(name, str(default))).strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = str(os.getenv(name, str(default))).strip()
    try:
        return int(raw)
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    raw = str(os.getenv(name, str(default))).strip()
    try:
        return float(raw)
    except Exception:
        return default


def _generation_max_tokens() -> int:
    # Keep local CPU latency predictable for real-time chat UX.
    requested = max(8, _env_int("THERAPIST_LLM_MAX_TOKENS", 16))
    return min(24, requested)


def _llm_history_max_turns() -> int:
    requested = _env_int("THERAPIST_LLM_HISTORY_MAX_TURNS", 1)
    return min(4, max(0, requested))


def _llama_runtime_candidates() -> list[Path]:
    explicit_runtime = str(os.getenv("THERAPIST_LLAMA_RUNTIME_PATH", "")).strip()
    candidates: list[Path] = []
    if explicit_runtime:
        candidates.append(Path(explicit_runtime).expanduser())
    candidates.extend(
        [
            Path("vendor/llama_cpp_runtime"),
            Path(".vendor/llama_cpp_runtime"),
        ]
    )
    unique_candidates: list[Path] = []
    for candidate in candidates:
        if candidate not in unique_candidates:
            unique_candidates.append(candidate)
    return unique_candidates


def _llama_helper_runtime_candidates() -> list[Path]:
    explicit_runtime = str(os.getenv("THERAPIST_LLAMA_HELPER_RUNTIME_PATH", "")).strip()
    candidates: list[Path] = []
    if explicit_runtime:
        candidates.append(Path(explicit_runtime).expanduser())
    candidates.extend(
        [
            Path("vendor/llama_cpp_runtime312"),
            Path("vendor/llama_cpp_runtime"),
            Path(".vendor/llama_cpp_runtime312"),
            Path(".vendor/llama_cpp_runtime"),
        ]
    )
    unique_candidates: list[Path] = []
    for candidate in candidates:
        if candidate not in unique_candidates:
            unique_candidates.append(candidate)
    return unique_candidates


def _llama_helper_python_candidates() -> list[Path]:
    explicit_python = str(os.getenv("THERAPIST_LLAMA_HELPER_PYTHON", "")).strip()
    candidates: list[Path] = []
    if explicit_python:
        candidates.append(Path(explicit_python).expanduser())
    candidates.extend(
        [
            Path(".tmp/python312_runtime/python.exe"),
            Path("vendor/python312_embed/python.exe"),
            Path("vendor/python312_runtime/python.exe"),
        ]
    )
    unique_candidates: list[Path] = []
    for candidate in candidates:
        if candidate not in unique_candidates:
            unique_candidates.append(candidate)
    return unique_candidates


# --- 1) Rule-based fallback library ---
EMOTION_VALIDATIONS = {
    "sad": "That sounds heavy to carry, and what you feel makes sense.",
    "sadness": "That sounds heavy to carry, and what you feel makes sense.",
    "angry": "I understand how this can create a lot of tension for you.",
    "fear": "I can see anxiety is taking up a lot of space right now.",
    "fearful": "I can see anxiety is taking up a lot of space right now.",
    "happy": "There is a positive momentum in you, and that is valuable.",
    "surprised": "When something is unexpected, it is normal to feel unsettled.",
    "disgust": "That feeling of rejection matters and deserves to be heard.",
    "calm": "You seem more grounded right now, and that is a strength.",
    "neutral": "Thank you for sharing this here; we can move step by step.",
}

EMOTION_GUIDANCE = {
    "sad": "We can start with one concrete point: what hurts the most today.",
    "sadness": "We can start with one concrete point: what hurts the most today.",
    "angry": "We can first slow down and name what crossed your limit.",
    "fear": "We can come back to something concrete here and now to help you stabilize.",
    "fearful": "We can come back to something concrete here and now to help you stabilize.",
    "happy": "We can identify what helps you keep this balance more often.",
    "surprised": "We can take a step back and organize what happened.",
    "disgust": "We can explore what this reaction is trying to protect in you.",
    "calm": "We can build on what helps you feel good to strengthen this stability.",
    "neutral": "We can clarify one priority so it does not feel overwhelming.",
}

FOLLOW_UP_QUESTIONS = {
    "sad": "What hurts the most right now?",
    "sadness": "What hurts the most right now?",
    "angry": "At what exact moment did you feel it overflow?",
    "fear": "What thought comes back the most when anxiety rises?",
    "fearful": "What thought comes back the most when anxiety rises?",
    "happy": "What contributed most to this state in recent days?",
    "surprised": "What unsettled you the most in this situation?",
    "disgust": "What made you say no inside?",
    "calm": "What concretely helps you keep this calm?",
    "neutral": "What would be the first small helpful step today?",
}

EMOTION_TONE = {
    "sad": "Use a warm, validating, and supportive tone.",
    "sadness": "Use a warm, validating, and supportive tone.",
    "angry": "Use a calm, de-escalating, and non-confrontational tone.",
    "fear": "Use a grounding and reassuring tone.",
    "fearful": "Use a grounding and reassuring tone.",
    "happy": "Use a positive and reinforcing tone while staying thoughtful.",
    "surprised": "Use a steady tone that helps organize the situation.",
    "disgust": "Use a respectful tone that validates boundaries.",
    "calm": "Use a constructive tone that helps preserve balance.",
    "neutral": "Use a gentle and exploratory tone.",
}

OPENERS = (
    "Thank you for sharing this.",
    "You did the right thing by putting this into words.",
    "I am with you in what you are going through.",
)

_SYSTEM_PROMPT_TEMPLATE = """You are a calm, supportive therapist-style assistant.
Respond in 3 to 5 short sentences.
Always validate feelings first, then give one practical coping step, then end with one gentle question.
Never provide harmful, illegal, violent, or self-harm instructions.
If danger or self-harm appears, prioritize immediate local emergency help.
Emotion: {emotion}
Tone: {tone_guidance}
Coping focus: {emotion_guidance}
Question focus: {emotion_question}
"""

_CRISIS_PATTERN = re.compile(
    r"\b("
    r"suicide|kill myself|end my life|self harm|self-harm|hurt myself|want to die|"
    r"overdose|cut myself|se suicider|me tuer|en finir|mourir|je veux mourir"
    r")\b",
    re.IGNORECASE,
)

_UNSAFE_REPLY_PATTERN = re.compile(
    r"\b(step\s*\d+|first[,:\s]|second[,:\s]|instructions?:|here(?:'s| is) how|guide to)\b"
    r".{0,160}\b(kill|self[- ]?harm|suicide|bomb|poison|attack|hack|tuer|bombe|empoisonner|pirater)\b",
    re.IGNORECASE | re.DOTALL,
)

_DANGEROUS_REQUEST_PATTERN = re.compile(
    r"\b(how\s+to|ways?\s+to|instructions?\s+(?:to|for)|steps?\s+(?:to|for)|"
    r"guide\s+(?:to|for)|teach me(?:\s+how)?\s+to|comment\s+(?:faire|fabriquer|tuer|agresser|pirater)|"
    r"methode\s+pour|moyen\s+de|etapes?\s+pour)\b"
    r".{0,80}\b(kill|self[- ]?harm|suicide|bomb|poison|attack|hack|"
    r"tuer|se suicider|bombe|empoisonner|agresser|pirater)\b",
    re.IGNORECASE,
)

_VIOLENT_INTENT_PATTERN = re.compile(
    r"\b("
    r"kill\s+(?:my\s+friend|him|her|them|someone|somebody|a\s+person|person)|"
    r"hurt\s+(?:my\s+friend|him|her|them|someone|somebody|a\s+person|person)|"
    r"attack\s+(?:him|her|them|someone|somebody|a\s+person|person)|"
    r"stab\s+(?:him|her|them|someone|somebody)|"
    r"shoot\s+(?:him|her|them|someone|somebody)|"
    r"murder\s+(?:my\s+friend|him|her|them|someone|somebody)|"
    r"tuer\s+(?:mon\s+ami|mon\s+amie|quelqu(?:'| )un)|"
    r"blesser\s+(?:quelqu(?:'| )un|mon\s+ami|mon\s+amie)|"
    r"agresser\s+(?:quelqu(?:'| )un|mon\s+ami|mon\s+amie)"
    r")\b",
    re.IGNORECASE,
)

_HARM_TOPIC_PATTERN = re.compile(
    r"\b(kill|self[- ]?harm|suicide|bomb|poison|attack|hack|tuer|se suicider|bombe|empoisonner|pirater)\b",
    re.IGNORECASE,
)

_REFUSAL_PATTERN = re.compile(
    r"\b(i cannot|i can't|i cant|i will not|i won't|i do not|i don't|i dont)\b",
    re.IGNORECASE,
)

_EMPATHY_HINTS = (
    "i hear",
    "i understand",
    "that sounds",
    "thank you",
    "what you feel",
    "you are not alone",
    "it makes sense",
)

_PROMPT_ECHO_PATTERN = re.compile(
    r"\b(core role|safety and boundaries|response structure|detected emotion|"
    r"tone guidance|emotion-oriented|here(?:'s| is) how you would respond|"
    r"here(?:'s| is) how i can support you|revised version of the prompt|"
    r"therapist-style assistant|validate your feelings first|"
    r"practical coping step|reply naturally|as a calm,\s*supportive)\b",
    re.IGNORECASE,
)

_ROLE_ARTIFACT_PATTERN = re.compile(r"\b(?:user|assistant|therapist)\s*:\s*", re.IGNORECASE)
_TRAILING_ROLE_ARTIFACT_PATTERN = re.compile(
    r"(?:\s+\b(?:user|assistant|therapist)\.?\s*)+$",
    re.IGNORECASE,
)
_STAGE_DIRECTION_PATTERN = re.compile(
    r"\((?:smil(?:e|ing)|laugh(?:s|ing)?|sigh(?:s|ing)?|pause(?:s|d)?|"
    r"breath(?:e|es|ing)|nod(?:s|ding))\)",
    re.IGNORECASE,
)
_TRAILING_GARBAGE_PATTERN = re.compile(
    r"(?:\s+(?:rep|reply|resp|response)\??)\s*$",
    re.IGNORECASE,
)
_QUESTIONISH_SENTENCE_PATTERN = re.compile(
    r"\b(what|how|why|when|where|who|which|can|could|would|do|does|did|is|are|should|will)\b"
    r"[^!?]{0,180}\.$",
    re.IGNORECASE,
)


# --- 2) In-memory history store ---
_HISTORY_LOCK = threading.Lock()
_SESSION_HISTORY: dict[str, deque[dict[str, str]]] = {}
_LAST_REPLY_SOURCE = "rule_based"


def _history_max_turns() -> int:
    return max(1, _env_int("THERAPIST_HISTORY_MAX_TURNS", 8))


def _session_key(session_id: str | None) -> str:
    key = _clean_text(session_id or "") or _DEFAULT_SESSION_ID
    return key[:128]


def _history_snapshot(session_id: str | None) -> list[dict[str, str]]:
    key = _session_key(session_id)
    with _HISTORY_LOCK:
        if key not in _SESSION_HISTORY:
            return []
        return list(_SESSION_HISTORY[key])


def _remember_turn(session_id: str | None, user_text: str, assistant_text: str) -> None:
    key = _session_key(session_id)
    with _HISTORY_LOCK:
        max_items = _history_max_turns() * 2
        history = _SESSION_HISTORY.get(key)
        if history is None:
            history = deque(maxlen=max_items)
            _SESSION_HISTORY[key] = history
        elif history.maxlen != max_items:
            history = deque(history, maxlen=max_items)
            _SESSION_HISTORY[key] = history

        history.append({"role": "user", "content": user_text})
        history.append({"role": "assistant", "content": assistant_text})


def clear_conversation_history(session_id: str | None = None) -> None:
    """Clear all cached turns or one session cache when session_id is provided."""
    with _HISTORY_LOCK:
        if session_id is None:
            _SESSION_HISTORY.clear()
            return
        _SESSION_HISTORY.pop(_session_key(session_id), None)


def _set_last_reply_source(source: str) -> None:
    global _LAST_REPLY_SOURCE
    _LAST_REPLY_SOURCE = str(source or "rule_based")


# --- 3) LLM backend loader ---
class _LocalLlamaEngine:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._loaded = False
        self._loading_started = False
        self._backend = ""
        self._failure_reason = ""
        self._resolved_model_path = ""
        self._resolved_runtime_path = ""
        self._resolved_helper_python_path = ""

        self._llama_cpp_model = None
        self._llama_helper_process: subprocess.Popen[str] | None = None

    def _start_background_load_if_needed(self) -> None:
        if self._loaded or self._loading_started:
            return

        with self._lock:
            if self._loaded or self._loading_started:
                return
            self._loading_started = True

        thread = threading.Thread(target=self._background_load_worker, daemon=True)
        thread.start()

    def _background_load_worker(self) -> None:
        try:
            self._ensure_loaded()
        except Exception:
            # Keep worker failures silent: generate_reply handles fallback paths.
            return

    def _model_path(self) -> Path:
        explicit_path = (
            os.getenv("THERAPIST_LLAMA_MODEL_PATH")
            or os.getenv("THERAPIST_LLM_MODEL_PATH")
            or ""
        )
        if explicit_path:
            return Path(str(explicit_path)).expanduser()

        # Auto-detect a common local model location before falling back.
        candidates = (
            Path("models/llama"),
            Path("LLMA/tinyllama-1.1b-chat-v1.0"),
            Path("LLMA"),
        )
        for candidate in candidates:
            if candidate.exists():
                return candidate

        return Path("models/llama")

    def _provider(self) -> str:
        raw = str(os.getenv("THERAPIST_LLM_PROVIDER", "llama_cpp")).strip().lower()
        if raw == "rule_based":
            return "rule_based"
        return "llama_cpp"

    def configured_provider(self) -> str:
        return self._provider()

    def configured_model_path(self) -> str:
        return str(self._model_path())

    def configured_runtime_path(self) -> str:
        return self._resolved_runtime_path

    def configured_helper_python_path(self) -> str:
        return self._resolved_helper_python_path

    def enabled(self) -> bool:
        if not _env_bool("THERAPIST_LLM_ENABLED", True):
            return False
        return self._provider() != "rule_based"

    def failure_reason(self) -> str:
        return self._failure_reason

    def backend_name(self) -> str:
        return self._backend

    def is_loading(self) -> bool:
        return bool(self._loading_started and not self._loaded)

    def _find_gguf_file(self, path: Path) -> Path:
        if path.is_file() and path.suffix.lower() == ".gguf":
            return path
        if path.is_dir():
            candidates = sorted(path.glob("*.gguf"))
            if candidates:
                return candidates[0]
        raise FileNotFoundError(f"No .gguf model file found under: {path}")

    def _helper_script_path(self) -> Path:
        return Path(__file__).with_name("llama_cpp_worker.py")

    def _helper_python_path(self) -> Path:
        for candidate in _llama_helper_python_candidates():
            if candidate.exists() and candidate.is_file():
                return candidate
        raise FileNotFoundError("No Python helper runtime found for llama_cpp")

    def _helper_runtime_path(self) -> Path:
        for candidate in _llama_helper_runtime_candidates():
            if candidate.exists() and candidate.is_dir():
                return candidate
        raise FileNotFoundError("No helper llama_cpp runtime package directory found")

    def _prefer_llama_helper(self) -> bool:
        if _env_bool("THERAPIST_LLAMA_PREFER_HELPER", False):
            return True
        return sys.version_info >= (3, 14)

    def _prepare_llama_runtime_import(self) -> None:
        self._resolved_runtime_path = ""
        for candidate in _llama_runtime_candidates():
            if not candidate.exists():
                continue
            resolved = str(candidate.resolve())
            if resolved not in sys.path:
                sys.path.insert(0, resolved)
            self._resolved_runtime_path = resolved
            return

    def _close_llama_helper_process(self) -> None:
        process = self._llama_helper_process
        self._llama_helper_process = None
        if process is None:
            return
        try:
            if process.poll() is None and process.stdin is not None:
                process.stdin.write(json.dumps({"command": "shutdown"}) + "\n")
                process.stdin.flush()
        except Exception:
            pass
        try:
            if process.poll() is None:
                process.terminate()
        except Exception:
            pass
        try:
            if process.poll() is None:
                process.kill()
        except Exception:
            pass

    def _read_helper_message(self, process: subprocess.Popen[str]) -> dict[str, object]:
        stdout = process.stdout
        if stdout is None:
            raise RuntimeError("llama_cpp helper stdout is not available")

        skipped_lines: list[str] = []
        while True:
            line = stdout.readline()
            if not line:
                tail = " | ".join(skipped_lines[-3:])
                if tail:
                    raise RuntimeError(f"llama_cpp helper exited before sending JSON: {tail}")
                raise RuntimeError("llama_cpp helper exited before sending JSON")
            clean = line.strip()
            if not clean:
                continue
            try:
                return json.loads(clean)
            except json.JSONDecodeError:
                skipped_lines.append(clean[:240])
                continue

    def _load_llama_cpp(self, path: Path) -> None:
        gguf = self._find_gguf_file(path)
        self._prepare_llama_runtime_import()
        try:
            from llama_cpp import Llama
        except Exception as exc:
            raise RuntimeError(f"llama_cpp not available: {exc}") from exc

        n_ctx = max(512, _env_int("THERAPIST_LLAMA_N_CTX", 1024))
        n_threads = max(
            1,
            _env_int("THERAPIST_LLAMA_N_THREADS", 1),
        )
        n_gpu_layers = _env_int("THERAPIST_LLAMA_N_GPU_LAYERS", 0)

        try:
            self._llama_cpp_model = Llama(
                model_path=str(gguf),
                n_ctx=n_ctx,
                n_threads=n_threads,
                n_gpu_layers=n_gpu_layers,
                verbose=False,
            )
        except TypeError:
            self._llama_cpp_model = Llama(
                model_path=str(gguf),
                n_ctx=n_ctx,
                n_threads=n_threads,
            )

    def _load_llama_cpp_helper(self, path: Path) -> None:
        gguf = self._find_gguf_file(path)
        helper_python = self._helper_python_path()
        helper_runtime = self._helper_runtime_path()
        helper_script = self._helper_script_path()
        if not helper_script.exists():
            raise FileNotFoundError(f"llama_cpp helper script not found: {helper_script}")

        self._close_llama_helper_process()

        n_ctx = max(512, _env_int("THERAPIST_LLAMA_N_CTX", 1024))
        n_threads = max(
            1,
            _env_int("THERAPIST_LLAMA_N_THREADS", 1),
        )
        n_gpu_layers = _env_int("THERAPIST_LLAMA_N_GPU_LAYERS", 0)

        process = subprocess.Popen(
            [
                str(helper_python.resolve()),
                str(helper_script.resolve()),
                "--runtime-path",
                str(helper_runtime.resolve()),
                "--model",
                str(gguf.resolve()),
                "--n-ctx",
                str(n_ctx),
                "--n-threads",
                str(n_threads),
                "--n-gpu-layers",
                str(n_gpu_layers),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        try:
            ready = self._read_helper_message(process)
        except Exception:
            try:
                process.kill()
            except Exception:
                pass
            raise

        if not bool(ready.get("ok")):
            error = str(ready.get("error") or "helper failed to start")
            try:
                process.kill()
            except Exception:
                pass
            raise RuntimeError(error)

        self._llama_helper_process = process
        self._resolved_runtime_path = str(helper_runtime.resolve())
        self._resolved_helper_python_path = str(helper_python.resolve())

    def _helper_generate(self, messages: list[dict[str, str]]) -> str:
        process = self._llama_helper_process
        if process is None or process.stdin is None:
            raise RuntimeError("llama_cpp helper is not running")
        if process.poll() is not None:
            raise RuntimeError("llama_cpp helper process already exited")

        payload = {
            "command": "generate",
            "messages": messages,
            "max_tokens": _generation_max_tokens(),
            "temperature": max(0.0, _env_float("THERAPIST_LLM_TEMPERATURE", 0.3)),
            "top_p": min(1.0, max(0.1, _env_float("THERAPIST_LLM_TOP_P", 0.8))),
        }
        try:
            process.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n")
            process.stdin.flush()
            response = self._read_helper_message(process)
        except Exception:
            self._close_llama_helper_process()
            raise

        if not bool(response.get("ok")):
            raise RuntimeError(str(response.get("error") or "helper generation failed"))
        return str(response.get("text") or "").strip()

    def _ensure_loaded(self) -> bool:
        if self._loaded:
            return bool(self._backend)

        with self._lock:
            if self._loaded:
                return bool(self._backend)

            provider = self._provider()
            model_path = self._model_path()
            self._resolved_model_path = str(model_path)
            self._resolved_runtime_path = ""
            self._resolved_helper_python_path = ""
            last_error = ""
            try:
                if provider != "llama_cpp":
                    raise RuntimeError(f"Unsupported provider: {provider}")
                attempts: list[tuple[str, object, str]]
                if self._prefer_llama_helper():
                    attempts = [
                        ("helper", self._load_llama_cpp_helper, "llama_cpp_helper"),
                        ("inline", self._load_llama_cpp, "llama_cpp"),
                    ]
                else:
                    attempts = [
                        ("inline", self._load_llama_cpp, "llama_cpp"),
                        ("helper", self._load_llama_cpp_helper, "llama_cpp_helper"),
                    ]

                errors: list[str] = []
                for label, loader, backend_name in attempts:
                    try:
                        loader(model_path)
                        self._backend = backend_name
                        last_error = ""
                        break
                    except Exception as exc:
                        errors.append(f"{label}: {exc}")

                if not self._backend:
                    raise RuntimeError(" | ".join(errors))
            except Exception as exc:
                last_error = f"llama_cpp: {exc}"

            self._failure_reason = last_error
            self._loaded = True
            return bool(self._backend)

    def _messages_to_text_prompt(self, messages: list[dict[str, str]]) -> str:
        parts: list[str] = []
        for msg in messages:
            role = str(msg.get("role", "user")).strip().lower()
            content = _clean_text(msg.get("content", ""))
            if not content:
                continue
            if role == "system":
                parts.append(f"System: {content}")
            elif role == "assistant":
                parts.append(f"Assistant: {content}")
            else:
                parts.append(f"User: {content}")
        parts.append("Assistant:")
        return "\n".join(parts)

    def _generate_llama_cpp(self, messages: list[dict[str, str]]) -> str:
        if self._llama_cpp_model is None:
            raise RuntimeError("llama_cpp model is not loaded")

        max_tokens = _generation_max_tokens()
        temperature = max(0.0, _env_float("THERAPIST_LLM_TEMPERATURE", 0.3))
        top_p = min(1.0, max(0.1, _env_float("THERAPIST_LLM_TOP_P", 0.8)))

        try:
            response = self._llama_cpp_model.create_chat_completion(
                messages=messages,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
            )
            choices = response.get("choices") or []
            if choices:
                message = choices[0].get("message") or {}
                content = str(message.get("content") or "").strip()
                if content:
                    return content
        except Exception:
            # Fallback to plain completion prompt for older/incompatible chat formats.
            pass

        prompt = self._messages_to_text_prompt(messages)
        completion = self._llama_cpp_model.create_completion(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            stop=["User:", "System:"],
        )
        choices = completion.get("choices") or []
        text = str((choices[0] if choices else {}).get("text") or "")
        return text.strip()

    def generate(self, messages: list[dict[str, str]]) -> str:
        if not self.enabled():
            raise RuntimeError("LLM backend disabled by configuration")

        # Background warmup can stay non-blocking for diagnostics/startup, but once a
        # real user message arrives we prefer waiting for TinyLlama to load instead of
        # falling back immediately to the deterministic rule-based reply.
        if self.is_loading():
            if not self._ensure_loaded():
                reason = self._failure_reason or "model backend not available"
                raise RuntimeError(reason)

        # On the first real generation request, force a synchronous load so TinyLlama
        # gets the first chance to answer whenever the model/runtime is available.
        if not self._loaded:
            if not self._ensure_loaded():
                reason = self._failure_reason or "model backend not available"
                raise RuntimeError(reason)

        if not self._ensure_loaded():
            reason = self._failure_reason or "model backend not available"
            raise RuntimeError(reason)

        with self._lock:
            if self._backend == "llama_cpp":
                return self._generate_llama_cpp(messages)
            if self._backend == "llama_cpp_helper":
                return self._helper_generate(messages)
            raise RuntimeError("No active LLM backend")


_LLM_ENGINE = _LocalLlamaEngine()


# --- 4) Text helpers ---
def _clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _pick(options: Iterable[str], seed: str) -> str:
    # Deterministic choice to avoid random shifts between runs.
    items = tuple(options)
    if not items:
        return ""
    digest = hashlib.sha256(seed.encode("utf-8")).digest()
    idx = digest[0] % len(items)
    return items[idx]


def _trim_reply(text: str) -> str:
    clean = _clean_text(text)
    clean = _ROLE_ARTIFACT_PATTERN.sub("", clean)
    clean = _TRAILING_ROLE_ARTIFACT_PATTERN.sub("", clean).strip()
    clean = _STAGE_DIRECTION_PATTERN.sub("", clean)
    clean = _clean_text(clean)
    clean = _TRAILING_GARBAGE_PATTERN.sub("", clean).strip()
    if not clean:
        return ""
    if "?" not in clean and _QUESTIONISH_SENTENCE_PATTERN.search(clean):
        clean = clean[:-1].rstrip() + "?"
    if len(clean) > THERAPIST_MAX_CHARS:
        clean = clean[: THERAPIST_MAX_CHARS - 3].rstrip() + "..."
    if clean[-1] not in ".!?":
        clean += "."
    return clean


def _normalize_emotion(emotion: str) -> str:
    key = _clean_text(emotion).lower()
    if not key:
        return "neutral"
    if key == "sadness":
        return "sad"
    if key == "fear":
        return "fearful"
    return key


def _merge_histories(
    session_history: list[dict[str, str]],
    external_history: list[dict[str, str]],
) -> list[dict[str, str]]:
    max_items = _history_max_turns() * 2
    merged: list[dict[str, str]] = []

    for item in [*session_history, *external_history]:
        role = str(item.get("role", "")).strip().lower()
        content = _clean_text(str(item.get("content", "")))
        if role not in {"user", "assistant"} or not content:
            continue
        candidate = {"role": role, "content": content[:1000]}
        if merged and merged[-1] == candidate:
            continue
        merged.append(candidate)

    return merged[-max_items:]


def _sanitize_history(history: list[Mapping[str, str]] | None) -> list[dict[str, str]]:
    if not history:
        return []

    clean_items: list[dict[str, str]] = []
    max_items = _history_max_turns() * 2
    for item in history[-max_items:]:
        role = str(item.get("role", "")).strip().lower()
        if role not in {"user", "assistant"}:
            continue
        content = _clean_text(str(item.get("content", "")))
        if not content:
            continue
        clean_items.append({"role": role, "content": content[:1000]})
    return clean_items


def _is_crisis_text(text: str) -> bool:
    return bool(_CRISIS_PATTERN.search(text))


def _is_dangerous_request(text: str) -> bool:
    if _is_crisis_text(text):
        return False
    return bool(_DANGEROUS_REQUEST_PATTERN.search(text))


def _is_violent_intent_text(text: str) -> bool:
    if _is_crisis_text(text):
        return False
    return bool(_VIOLENT_INTENT_PATTERN.search(text))


def _safety_reply_emotion(emotion: str) -> str:
    emotion_key = _normalize_emotion(emotion)
    if emotion_key in {"happy", "calm", "surprised"}:
        return "neutral"
    return emotion_key


def _crisis_safe_reply(emotion: str) -> str:
    emotion = _safety_reply_emotion(emotion)
    tone = {
        "sad": "I am really glad you shared this.",
        "fearful": "I hear how intense this feels right now.",
        "angry": "I can hear how overwhelmed things feel.",
    }.get(emotion, "Thank you for telling me this.")

    return _trim_reply(
        f"{tone} Your safety matters most. "
        "If you might hurt yourself or someone else, please contact local emergency services now. "
        "If possible, reach out to a trusted person who can stay with you right now. "
        "Would you like us to focus on one immediate grounding step together?"
    )


def _violent_intent_safe_reply(emotion: str) -> str:
    emotion = _safety_reply_emotion(emotion)
    tone = {
        "angry": "I hear how intense and unsafe this feels right now.",
        "fearful": "I hear how serious this feels right now.",
        "sad": "Thank you for saying this out loud.",
    }.get(emotion, "Thank you for telling me this.")
    return _trim_reply(
        f"{tone} I cannot help with hurting someone. "
        "If you think you may act on this, move away from the person and from any weapon right now, "
        "and contact local emergency services or a trusted adult immediately. "
        "What is one immediate step you can take to create distance and stay safe now?"
    )


def _dangerous_request_safe_reply(emotion: str) -> str:
    emotion_key = _safety_reply_emotion(emotion)
    validation = EMOTION_VALIDATIONS.get(
        emotion_key,
        "I hear that things feel intense right now.",
    )
    question = FOLLOW_UP_QUESTIONS.get(
        emotion_key,
        "What would help you feel safer right now?",
    )
    return _trim_reply(
        f"{validation} I cannot help with dangerous or illegal instructions. "
        "I can help you with a safer plan to calm your mind and protect yourself right now. "
        f"{question}"
    )


def _contains_unsafe_reply(text: str) -> bool:
    lowered = str(text or "")
    if _REFUSAL_PATTERN.search(lowered) and _HARM_TOPIC_PATTERN.search(lowered):
        return False
    return bool(_UNSAFE_REPLY_PATTERN.search(text))


def _enforce_therapeutic_shape(reply: str, emotion: str) -> str:
    emotion_key = _normalize_emotion(emotion)
    cleaned = _trim_reply(reply)
    if not cleaned:
        return ""

    lowered = cleaned.lower()
    if not any(marker in lowered for marker in _EMPATHY_HINTS):
        validation = EMOTION_VALIDATIONS.get(
            emotion_key,
            "What you feel matters, and we can take it seriously.",
        )
        cleaned = _trim_reply(f"{validation} {cleaned}")

    if "?" not in cleaned:
        follow_up = FOLLOW_UP_QUESTIONS.get(
            emotion_key,
            "What feels most important to explore now?",
        )
        cleaned = _trim_reply(f"{cleaned} {follow_up}")

    return cleaned


def _looks_like_prompt_echo(text: str) -> bool:
    return bool(_PROMPT_ECHO_PATTERN.search(str(text or "")))


# --- 5) Fallback generation ---
def _rule_based_reply(user_text: str, emotion: str = "neutral") -> str:
    emotion_key = _normalize_emotion(emotion)
    opener = _pick(OPENERS, seed=f"{emotion_key}:{user_text}")
    validation = EMOTION_VALIDATIONS.get(
        emotion_key, "What you feel matters, and we can take it seriously."
    )
    guidance = EMOTION_GUIDANCE.get(
        emotion_key, "We can move in small steps so this stays manageable."
    )
    question = FOLLOW_UP_QUESTIONS.get(
        emotion_key, "What seems most helpful to you right now?"
    )
    return _trim_reply(f"{opener} {validation} {guidance} {question}")


# --- 6) LLaMA prompt + generation ---
def _build_llm_messages(
    user_text: str,
    emotion: str,
    history: list[dict[str, str]],
) -> list[dict[str, str]]:
    emotion_key = _normalize_emotion(emotion)
    tone_guidance = EMOTION_TONE.get(emotion_key, EMOTION_TONE["neutral"])
    emotion_guidance = EMOTION_GUIDANCE.get(
        emotion_key,
        "Offer one simple grounding or coping step the user can try now.",
    )
    emotion_question = FOLLOW_UP_QUESTIONS.get(
        emotion_key,
        "What feels most helpful to focus on right now?",
    )

    system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(
        emotion=emotion_key,
        tone_guidance=tone_guidance,
        emotion_guidance=emotion_guidance,
        emotion_question=emotion_question,
    )

    messages: list[dict[str, str]] = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_text})
    return messages


def _llm_reply(
    user_text: str,
    emotion: str,
    history: list[dict[str, str]],
) -> str:
    messages = _build_llm_messages(
        user_text=user_text,
        emotion=emotion,
        history=history,
    )
    raw_reply = _LLM_ENGINE.generate(messages)
    clean_reply = _enforce_therapeutic_shape(raw_reply, emotion=emotion)
    if (
        not _clean_text(raw_reply)
        or not clean_reply
        or _looks_like_prompt_echo(clean_reply)
        or _contains_unsafe_reply(clean_reply)
    ):
        # Retry with a compact prompt when the small local model echoes instructions
        # or struggles with the full therapist system prompt.
        compact_messages = [
            {
                "role": "system",
                "content": (
                    "You are a caring therapist. "
                    "Reply naturally in 1 or 2 short sentences, offer one simple coping step, "
                    "and end with one gentle question. "
                    "Do not explain your instructions."
                ),
            },
            {"role": "user", "content": user_text},
        ]
        raw_reply = _LLM_ENGINE.generate(compact_messages)
        clean_reply = _enforce_therapeutic_shape(raw_reply, emotion=emotion)
    if not clean_reply:
        raise RuntimeError("Empty LLaMA reply")
    if _looks_like_prompt_echo(clean_reply):
        raise RuntimeError("Prompt echo detected")
    if _contains_unsafe_reply(clean_reply):
        raise RuntimeError("Unsafe LLaMA reply detected")
    return clean_reply


# --- 7) Public API ---
def generate_reply(
    user_text: str,
    emotion: str = "neutral",
    session_id: str | None = None,
    conversation_history: list[Mapping[str, str]] | None = None,
) -> str:
    # Main entry point used by API and CLI pipeline.
    clean_user_text = _clean_text(user_text)
    emotion_key = _normalize_emotion(emotion)

    if not clean_user_text:
        _set_last_reply_source("rule_based")
        return "I am here with you. Tell me what you are going through right now."

    if _is_crisis_text(clean_user_text):
        reply = _crisis_safe_reply(emotion_key)
        _set_last_reply_source("rule_based")
        _remember_turn(session_id, clean_user_text, reply)
        return reply

    if _is_violent_intent_text(clean_user_text):
        reply = _violent_intent_safe_reply(emotion_key)
        _set_last_reply_source("rule_based")
        _remember_turn(session_id, clean_user_text, reply)
        return reply

    if _is_dangerous_request(clean_user_text):
        reply = _dangerous_request_safe_reply(emotion_key)
        _set_last_reply_source("rule_based")
        _remember_turn(session_id, clean_user_text, reply)
        return reply

    external_history = _sanitize_history(conversation_history)
    session_history = _sanitize_history(_history_snapshot(session_id))
    history = _merge_histories(
        session_history=session_history,
        external_history=external_history,
    )

    llm_history = history
    max_llm_items = _llm_history_max_turns() * 2
    if max_llm_items <= 0:
        llm_history = []
    elif len(llm_history) > max_llm_items:
        llm_history = llm_history[-max_llm_items:]

    try:
        if _LLM_ENGINE.enabled():
            reply = _llm_reply(
                user_text=clean_user_text,
                emotion=emotion_key,
                history=llm_history,
            )
            _set_last_reply_source(_LLM_ENGINE.backend_name() or "llm")
            _remember_turn(session_id, clean_user_text, reply)
            return reply
    except Exception:
        # Keep silent and continue with deterministic fallback.
        pass

    fallback_reply = _rule_based_reply(clean_user_text, emotion=emotion_key)
    _set_last_reply_source("rule_based")
    _remember_turn(session_id, clean_user_text, fallback_reply)
    return fallback_reply


def therapist_backend_status() -> dict[str, str]:
    """Simple status helper for diagnostics/logs."""
    status = {
        "provider": _LLM_ENGINE.configured_provider(),
        "model_path": _LLM_ENGINE.configured_model_path(),
    }
    runtime_path = _LLM_ENGINE.configured_runtime_path()
    if runtime_path:
        status["runtime_path"] = runtime_path
    helper_python_path = _LLM_ENGINE.configured_helper_python_path()
    if helper_python_path:
        status["helper_python_path"] = helper_python_path

    if not _LLM_ENGINE.enabled():
        return {
            **status,
            "backend": "rule_based",
            "reason": "THERAPIST_LLM_ENABLED=false or provider=rule_based",
        }

    if _LLM_ENGINE.is_loading():
        return {**status, "backend": "rule_based", "reason": "warmup in progress"}

    if _LLM_ENGINE.backend_name():
        return {**status, "backend": _LLM_ENGINE.backend_name(), "reason": "loaded"}

    reason = _LLM_ENGINE.failure_reason() or "not loaded yet"
    return {**status, "backend": "rule_based", "reason": reason}


def therapist_start_warmup(wait: bool = False) -> None:
    if not _LLM_ENGINE.enabled():
        return
    if wait:
        try:
            _LLM_ENGINE._ensure_loaded()
            return
        except Exception:
            return
    _LLM_ENGINE._start_background_load_if_needed()


def therapist_last_reply_source() -> str:
    return _LAST_REPLY_SOURCE
