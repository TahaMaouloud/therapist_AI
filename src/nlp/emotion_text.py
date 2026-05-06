from __future__ import annotations

import os
import re
import unicodedata
from pathlib import Path
from typing import Any

from src.env_loader import load_local_env_file

load_local_env_file()

try:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
except Exception:  # pragma: no cover - optional dependency fallback
    torch = None
    AutoModelForSequenceClassification = None
    AutoTokenizer = None

# Text emotion detection:
# 1) prefer trained model if available
# 2) fallback to keyword heuristic

EMOTION_KEYWORDS: dict[str, tuple[str, ...]] = {
    "fearful": (
        "stress",
        "anxiety",
        "anxious",
        "afraid",
        "scared",
        "fear",
        "worry",
        "worried",
        "nervous",
        "panic",
        "overwhelmed",
        "cannot",
        "cant",
        "can t",
        "cannot do it",
        "cant do it",
        "can t do it",
        "i cannot do it",
        "i cant do it",
        "i can t do it",
    ),
    "sad": (
        "sad",
        "depressed",
        "down",
        "tired",
        "exhausted",
        "empty",
        "lonely",
        "alone",
        "hopeless",
        "cry",
        "crying",
    ),
    "angry": (
        "angry",
        "mad",
        "furious",
        "frustrated",
        "annoyed",
        "rage",
    ),
    "happy": (
        "happy",
        "better",
        "good",
        "great",
        "relieved",
        "joyful",
        "content",
    ),
    "surprised": (
        "surprised",
        "shocked",
        "unexpected",
        "suddenly",
    ),
    "disgust": (
        "disgusted",
        "disappointed",
        "rejected",
        "nauseous",
    ),
}

_DEFAULT_MODEL_CANDIDATES = (
    Path(os.getenv("TEXT_EMOTION_MODEL_PATH", "models/emotion_bert")),
    Path("models/_tmp_emotion_bert"),
)

_TEXT_MODEL: tuple[Any, Any, Any] | None = None
_MODEL_LOOKUP_DONE = False
_NEGATED_HAPPY_PATTERNS = (
    re.compile(r"\bnot happy\b"),
    re.compile(r"\bunhappy\b"),
    re.compile(r"\bnot feeling happy\b"),
    re.compile(r"\bdo not feel happy\b"),
    re.compile(r"\bdont feel happy\b"),
    re.compile(r"\bdon t feel happy\b"),
    re.compile(r"\bno longer happy\b"),
    re.compile(r"\bnever happy\b"),
)
_SELF_HARM_TEXT_PATTERN = re.compile(
    r"\b("
    r"kill myself|hurt myself|want to die|end my life|self harm|selfharm|"
    r"suicide|overdose|cut myself|me tuer|je veux mourir|en finir"
    r")\b"
)
_VIOLENT_INTENT_PATTERN = re.compile(
    r"\b("
    r"kill\s+(?:my\s+friend|him|her|them|someone|somebody|a\s+person|person)|"
    r"hurt\s+(?:my\s+friend|him|her|them|someone|somebody|a\s+person|person)|"
    r"attack\s+(?:him|her|them|someone|somebody|a\s+person|person)|"
    r"stab\s+(?:him|her|them|someone|somebody)|"
    r"shoot\s+(?:him|her|them|someone|somebody)|"
    r"murder\s+(?:my\s+friend|him|her|them|someone|somebody)|"
    r"tuer\s+(?:mon\s+ami|mon\s+amie|quelqu\s+un)|"
    r"blesser\s+(?:quelqu\s+un|mon\s+ami|mon\s+amie)|"
    r"agresser\s+(?:quelqu\s+un|mon\s+ami|mon\s+amie)"
    r")\b"
)
_DANGEROUS_INSTRUCTION_PATTERN = re.compile(
    r"\b("
    r"how\s+to|ways?\s+to|instructions?\s+(?:to|for)|steps?\s+(?:to|for)|"
    r"guide\s+(?:to|for)|teach\s+me(?:\s+how)?\s+to|"
    r"comment\s+(?:faire|fabriquer|tuer|agresser|pirater)|"
    r"methode\s+pour|moyen\s+de|etapes?\s+pour"
    r")\b.{0,80}\b("
    r"kill|self\s*harm|suicide|bomb|poison|attack|hack|"
    r"tuer|bombe|empoisonner|agresser|pirater"
    r")\b"
)


def _normalize_text(text: str) -> str:
    lowered = text.lower()
    ascii_like = unicodedata.normalize("NFKD", lowered)
    ascii_like = "".join(ch for ch in ascii_like if not unicodedata.combining(ch))
    cleaned = re.sub(r"[^a-z0-9\s]", " ", ascii_like)
    return re.sub(r"\s+", " ", cleaned).strip()


def _safety_override_emotion(normalized: str) -> tuple[str, float, str] | None:
    if _SELF_HARM_TEXT_PATTERN.search(normalized):
        return "fearful", 0.99, "safety-self-harm-override"
    if _VIOLENT_INTENT_PATTERN.search(normalized):
        return "angry", 0.99, "safety-violence-override"
    if _DANGEROUS_INSTRUCTION_PATTERN.search(normalized):
        return "angry", 0.97, "safety-danger-override"
    return None


def _predict_emotion_heuristic(normalized: str) -> str:
    emotion, _confidence = _predict_emotion_heuristic_with_confidence(normalized)
    return emotion


def _predict_emotion_heuristic_with_confidence(normalized: str) -> tuple[str, float]:
    scores: dict[str, int] = {emotion: 0 for emotion in EMOTION_KEYWORDS}
    for emotion, keywords in EMOTION_KEYWORDS.items():
        for kw in keywords:
            kw_norm = _normalize_text(kw)
            if kw_norm and kw_norm in normalized:
                scores[emotion] += 1

    best_emotion = "neutral"
    best_score = 0
    for emotion, score in scores.items():
        if score > best_score:
            best_emotion = emotion
            best_score = score

    if best_score <= 0:
        return "neutral", 0.0

    total_score = sum(scores.values())
    confidence = min(0.49, float(best_score) / max(total_score, 1))
    return best_emotion, confidence


def _checkpoint_step(path: Path) -> int:
    prefix = "checkpoint-"
    if not path.is_dir() or not path.name.startswith(prefix):
        return -1
    try:
        return int(path.name[len(prefix) :])
    except ValueError:
        return -1


def _model_weight_mtime(path: Path) -> float:
    mtimes: list[float] = []
    for filename in ("model.safetensors", "pytorch_model.bin"):
        weight_path = path / filename
        if not weight_path.exists():
            continue
        try:
            mtimes.append(float(weight_path.stat().st_mtime))
        except OSError:
            continue
    return max(mtimes) if mtimes else 0.0


def _ordered_model_candidates(base_path: Path) -> list[Path]:
    if not base_path.exists() or not base_path.is_dir():
        return [base_path]

    checkpoints = [path for path in base_path.glob("checkpoint-*") if _checkpoint_step(path) >= 0]
    if not checkpoints:
        return [base_path]

    checkpoints.sort(key=_checkpoint_step, reverse=True)
    latest_checkpoint = checkpoints[0]
    if _model_weight_mtime(latest_checkpoint) > _model_weight_mtime(base_path):
        return [latest_checkpoint, base_path, *checkpoints[1:]]
    return [base_path, *checkpoints]


def _candidate_model_paths() -> list[Path]:
    unique_paths: list[Path] = []
    for path in _DEFAULT_MODEL_CANDIDATES:
        for candidate in _ordered_model_candidates(path):
            if candidate not in unique_paths:
                unique_paths.append(candidate)
    return unique_paths


def _get_text_model() -> Any | None:
    global _TEXT_MODEL, _MODEL_LOOKUP_DONE
    if _MODEL_LOOKUP_DONE:
        return _TEXT_MODEL

    _MODEL_LOOKUP_DONE = True
    if torch is None or AutoTokenizer is None or AutoModelForSequenceClassification is None:
        return None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for path in _candidate_model_paths():
        if not path.exists():
            continue
        try:
            tokenizer = AutoTokenizer.from_pretrained(path)
            model = AutoModelForSequenceClassification.from_pretrained(path)
            model.to(device)
            model.eval()
            _TEXT_MODEL = (tokenizer, model, device)
            return _TEXT_MODEL
        except Exception:
            continue
    return None


def _predict_emotion_model(normalized: str) -> str | None:
    label, _confidence = _predict_emotion_model_with_confidence(normalized)
    return label


def _predict_emotion_model_with_confidence(normalized: str) -> tuple[str | None, float]:
    bundle = _get_text_model()
    if bundle is None:
        return None, 0.0

    try:
        tokenizer, model, device = bundle
        max_length = int(os.getenv("TEXT_EMOTION_MAX_LENGTH", "256"))
        inputs = tokenizer(normalized, return_tensors="pt", truncation=True, max_length=max_length)
        inputs = {key: value.to(device) for key, value in inputs.items()}

        with torch.inference_mode():
            logits = model(**inputs).logits

        probabilities = torch.softmax(logits, dim=-1)
        pred_id = int(torch.argmax(logits, dim=-1).item())
        confidence = float(probabilities[0, pred_id].item())
        id2label = model.config.id2label or {}
        label = id2label.get(pred_id) or id2label.get(str(pred_id))
        if label is None:
            return None, 0.0
        label = str(label).strip().lower()
        return (label or "neutral"), confidence
    except Exception:
        return None, 0.0


def _apply_post_rules(normalized: str, label: str, confidence: float, source: str) -> tuple[str, float, str]:
    lowered_label = str(label or "neutral").strip().lower() or "neutral"

    # Small guardrail for explicit negation that the tiny dataset does not cover well.
    if any(pattern.search(normalized) for pattern in _NEGATED_HAPPY_PATTERNS):
        if lowered_label == "happy" or confidence < 0.55:
            return "sad", max(confidence, 0.56), f"{source}-negation-override"

    return lowered_label, confidence, source


def predict_emotion_with_confidence_from_text(text: str) -> tuple[str, float, str]:
    normalized = _normalize_text(text)
    if not normalized:
        return "neutral", 0.0, "empty"

    safety_override = _safety_override_emotion(normalized)
    if safety_override is not None:
        return safety_override

    model_label, model_confidence = _predict_emotion_model_with_confidence(normalized)
    if model_label is not None:
        return _apply_post_rules(normalized, model_label, model_confidence, "model")

    heuristic_label, heuristic_confidence = _predict_emotion_heuristic_with_confidence(normalized)
    return _apply_post_rules(normalized, heuristic_label, heuristic_confidence, "heuristic")


def predict_emotion_from_text(text: str) -> str:
    label, _confidence, _source = predict_emotion_with_confidence_from_text(text)
    return label
