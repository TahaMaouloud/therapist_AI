from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

import joblib
import librosa
import numpy as np

# Module d'inference emotion audio:
# fichier audio -> extraction features/embeddings -> prediction emotion (+ confiance).

# --- 1) Mapping des emotions selon les conventions de noms de fichiers ---
RAVDESS_EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}

# Example filename: 1001_DFA_ANG_XX.wav
UNDERSCORE_EMOTION_MAP = {
    "ANG": "angry",
    "DIS": "disgust",
    "FEA": "fearful",
    "HAP": "happy",
    "NEU": "neutral",
    "SAD": "sad",
}

# Language-agnostic emotion grouping that generalizes better on real-world audio.
CORE5_EMOTION_MAP = {
    "neutral": "neutral",
    "calm": "neutral",
    "happy": "happy",
    "surprised": "happy",
    "sad": "sad",
    "fearful": "fearful",
    "angry": "angry",
    "disgust": "angry",
}


# --- 2) Parsing et harmonisation des labels ---
def parse_ravdess_emotion(audio_path: Path) -> Optional[str]:
    # Example filename: 03-01-05-01-01-01-01.wav
    parts = audio_path.stem.split("-")
    if len(parts) < 3:
        return None
    emotion_code = parts[2]
    return RAVDESS_EMOTION_MAP.get(emotion_code)


def parse_underscore_emotion(audio_path: Path) -> Optional[str]:
    # Example filename: 1001_DFA_ANG_XX.wav
    parts = audio_path.stem.split("_")
    if len(parts) < 3:
        return None
    emotion_code = parts[2].upper()
    return UNDERSCORE_EMOTION_MAP.get(emotion_code)


def parse_emotion_label(audio_path: Path) -> Optional[str]:
    """
    Parse emotion label from supported dataset filename formats.
    """
    return parse_ravdess_emotion(audio_path) or parse_underscore_emotion(audio_path)


def map_emotion_label(emotion: str | None, scheme: str = "core5") -> Optional[str]:
    # Harmonise les labels source vers le schema choisi (full ou core5).
    if emotion is None:
        return None
    key = emotion.lower()
    if scheme == "full":
        return key
    if scheme == "core5":
        return CORE5_EMOTION_MAP.get(key)
    raise ValueError(f"Unsupported label scheme: {scheme}")


# --- 3) Outils de resume statistique pour les features ---
def _stats_2d(feat: np.ndarray) -> np.ndarray:
    return np.hstack([np.mean(feat, axis=1), np.std(feat, axis=1)])


def _stats_1d(feat: np.ndarray) -> np.ndarray:
    return np.array([float(np.mean(feat)), float(np.std(feat))], dtype=np.float32)


# --- 4) Extraction principale de features audio (utilisee a l'entrainement) ---
def extract_audio_features(audio_path: Path, sr: int = 22050) -> np.ndarray:
    y, sample_rate = _load_audio(audio_path, sr=sr)
    if len(y) == 0:
        raise ValueError(f"Empty audio: {audio_path}")

    y = librosa.util.normalize(y)

    # Features temps/frequence + descripteurs spectraux.
    mfcc = librosa.feature.mfcc(y=y, sr=sample_rate, n_mfcc=40)
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    chroma = librosa.feature.chroma_stft(y=y, sr=sample_rate)
    mel = librosa.power_to_db(
        librosa.feature.melspectrogram(y=y, sr=sample_rate), ref=np.max
    )
    zcr = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)
    centroid = librosa.feature.spectral_centroid(y=y, sr=sample_rate)
    bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sample_rate)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sample_rate)
    flatness = librosa.feature.spectral_flatness(y=y)

    try:
        # Peut echouer sur des signaux tres courts.
        contrast = librosa.feature.spectral_contrast(y=y, sr=sample_rate)
    except Exception:
        contrast = np.zeros((7, 1), dtype=np.float32)

    try:
        # Tonnetz repose sur une decomposition harmonique.
        harmonic = librosa.effects.harmonic(y)
        tonnetz = librosa.feature.tonnetz(y=harmonic, sr=sample_rate)
    except Exception:
        tonnetz = np.zeros((6, 1), dtype=np.float32)

    try:
        # Tempo moyen comme feature de prosodie.
        tempo = librosa.feature.tempo(y=y, sr=sample_rate)
        tempo_stats = np.array([float(np.mean(tempo))], dtype=np.float32)
    except Exception:
        tempo_stats = np.array([0.0], dtype=np.float32)

    features = np.hstack(
        [
            _stats_2d(mfcc),
            _stats_2d(mfcc_delta),
            _stats_2d(mfcc_delta2),
            _stats_2d(chroma),
            _stats_2d(mel),
            _stats_2d(contrast),
            _stats_2d(tonnetz),
            _stats_1d(zcr),
            _stats_1d(rms),
            _stats_1d(centroid),
            _stats_1d(bandwidth),
            _stats_1d(rolloff),
            _stats_1d(flatness),
            tempo_stats,
        ]
    )
    return features.astype(np.float32)


# --- 5) Extraction legacy pour compatibilite anciens modeles ---
def extract_audio_features_legacy(audio_path: Path, sr: int = 22050) -> np.ndarray:
    """
    Backward-compatible feature extractor (364 features) used by old saved models.
    """
    y, sample_rate = _load_audio(audio_path, sr=sr)
    if len(y) == 0:
        raise ValueError(f"Empty audio: {audio_path}")

    mfcc = librosa.feature.mfcc(y=y, sr=sample_rate, n_mfcc=40)
    chroma = librosa.feature.chroma_stft(y=y, sr=sample_rate)
    mel = librosa.feature.melspectrogram(y=y, sr=sample_rate)
    zcr = librosa.feature.zero_crossing_rate(y)
    rms = librosa.feature.rms(y=y)

    features = np.hstack(
        [
            np.mean(mfcc, axis=1),
            np.std(mfcc, axis=1),
            np.mean(chroma, axis=1),
            np.std(chroma, axis=1),
            np.mean(mel, axis=1),
            np.std(mel, axis=1),
            np.mean(zcr, axis=1),
            np.std(zcr, axis=1),
            np.mean(rms, axis=1),
            np.std(rms, axis=1),
        ]
    )
    return features.astype(np.float32)


# --- 6) Chargement audio robuste: librosa puis fallback PyAV ---
def _load_audio(audio_path: Path, sr: int = 22050) -> tuple[np.ndarray, int]:
    ext = audio_path.suffix.lower()
    prefer_av_ext = {".webm", ".m4a", ".aac", ".ogg", ".opus", ".flac", ".mp3"}
    if ext in prefer_av_ext:
        return _load_audio_with_av(audio_path, sr=sr)
    try:
        y, sample_rate = librosa.load(str(audio_path), sr=sr)
        return y.astype(np.float32), int(sample_rate)
    except Exception:
        return _load_audio_with_av(audio_path, sr=sr)


def _load_audio_with_av(audio_path: Path, sr: int = 22050) -> tuple[np.ndarray, int]:
    import av

    chunks: list[np.ndarray] = []
    native_sr = None

    with av.open(str(audio_path)) as container:
        audio_stream = next((s for s in container.streams if s.type == "audio"), None)
        if audio_stream is None:
            raise ValueError(f"No audio stream found in {audio_path}")

        for frame in container.decode(audio_stream):
            arr = frame.to_ndarray()
            if arr.ndim == 2:
                if arr.shape[0] <= 8:
                    arr = arr.mean(axis=0)
                else:
                    arr = arr.mean(axis=1)

            if np.issubdtype(arr.dtype, np.integer):
                max_abs = max(abs(np.iinfo(arr.dtype).min), np.iinfo(arr.dtype).max)
                arr = arr.astype(np.float32) / float(max_abs if max_abs else 1)
            else:
                arr = arr.astype(np.float32)

            chunks.append(arr)
            if native_sr is None:
                native_sr = frame.sample_rate

    if not chunks:
        raise ValueError(f"Could not decode audio frames from {audio_path}")

    y = np.concatenate(chunks, axis=0)
    native_sr = int(native_sr or sr)
    if native_sr != sr:
        y = librosa.resample(y, orig_sr=native_sr, target_sr=sr)
        native_sr = sr
    return y.astype(np.float32), native_sr


# --- 7) Adaptation features <-> modele charge (nouveau vs legacy) ---
def _resolve_feature_input(model: Any, audio_path: Path) -> np.ndarray:
    # Etape 1: tenter les features actuelles.
    expected_features = getattr(model, "n_features_in_", None)
    x = extract_audio_features(audio_path).reshape(1, -1)
    if expected_features is None or x.shape[1] == expected_features:
        return x

    # Etape 2: fallback legacy si le modele attend l'ancien format.
    x_legacy = extract_audio_features_legacy(audio_path).reshape(1, -1)
    if x_legacy.shape[1] == expected_features:
        return x_legacy

    raise ValueError(
        f"Feature mismatch: model expects {expected_features}, "
        f"new={x.shape[1]}, legacy={x_legacy.shape[1]}"
    )


# --- 8) Utilitaires probabilites / top-k ---
def _build_one_hot(pred: int, num_classes: int) -> np.ndarray:
    probs = np.zeros(num_classes, dtype=np.float32)
    probs[int(pred)] = 1.0
    return probs


def _top_k_from_probs(
    probs: np.ndarray,
    labels: list[str],
    k: int,
) -> list[tuple[str, float]]:
    flat = np.asarray(probs, dtype=np.float32).reshape(-1)
    if flat.size != len(labels):
        raise ValueError(
            f"Probability size mismatch: probs={flat.size}, labels={len(labels)}"
        )

    top_k = max(1, min(int(k), flat.size))
    idx = np.argsort(flat)[::-1][:top_k]
    return [(str(labels[int(i)]), float(flat[int(i)])) for i in idx]


def _align_wav2vec2_runtime_config(model_bundle: dict[str, Any], model: Any) -> None:
    bundle_pooling = str(model_bundle.get("wav2vec2_pooling", "")).strip().lower()
    feature_version = str(model_bundle.get("feature_version", "")).strip().lower()
    resolved_pooling = bundle_pooling

    if not resolved_pooling:
        if "meanstd" in feature_version:
            resolved_pooling = "meanstd"
        elif "mean_pool" in feature_version or feature_version.endswith("_mean"):
            resolved_pooling = "mean"

    if not resolved_pooling:
        expected_features = getattr(model, "n_features_in_", None)
        if expected_features == 768:
            resolved_pooling = "mean"
        elif expected_features == 1536:
            resolved_pooling = "meanstd"

    if resolved_pooling and getattr(model, "pooling", "") != resolved_pooling:
        setattr(model, "pooling", resolved_pooling)
        if hasattr(model, "_build_cache_signature"):
            model._cache_signature = model._build_cache_signature()
        if hasattr(model, "_resolve_cache_dir_path"):
            model._cache_dir_path = model._resolve_cache_dir_path()


# --- 9) API de prediction emotion audio ---
def predict_emotion_top_k_from_audio(
    audio_path: str,
    model_path: str = "models/emotion_audio_model.joblib",
    k: int = 2,
) -> list[tuple[str, float]]:
    # Etape 1: charger le bundle (modele + encodeur labels + metadata).
    model_bundle = joblib.load(model_path)
    model = model_bundle["model"]
    label_encoder = model_bundle["label_encoder"]
    labels = [str(x) for x in label_encoder.classes_]
    model_name = str(model_bundle.get("model_name", "")).lower()
    sample_path = Path(audio_path)

    if model_name == "wav2vec2":
        _align_wav2vec2_runtime_config(model_bundle, model)

    probs: np.ndarray | None = None

    # Etape 2: calculer les probabilites selon le type de modele.
    # Branche Wav2Vec2: inference basee sur chemins audio.
    if model_name == "wav2vec2" and hasattr(model, "predict_proba_paths"):
        probs = np.asarray(model.predict_proba_paths([sample_path])[0], dtype=np.float32)
    elif model_name == "wav2vec2" and hasattr(model, "predict_paths"):
        pred = int(model.predict_paths([sample_path])[0])
        probs = _build_one_hot(pred=pred, num_classes=len(labels))
    else:
        # Branche tabulaire: extraction features puis prediction classique.
        x = _resolve_feature_input(model=model, audio_path=sample_path)
        if hasattr(model, "predict_proba"):
            try:
                probs = np.asarray(model.predict_proba(x)[0], dtype=np.float32)
            except Exception:
                probs = None
        if probs is None:
            pred = int(model.predict(x)[0])
            probs = _build_one_hot(pred=pred, num_classes=len(labels))

    # Etape 3: retourner le top-k trie par probabilite.
    return _top_k_from_probs(probs=probs, labels=labels, k=k)


def predict_emotion_from_audio_with_confidence(
    audio_path: str, model_path: str = "models/emotion_audio_model.joblib"
) -> tuple[str, float]:
    # Wrapper top-1: renvoie (emotion, confiance).
    top = predict_emotion_top_k_from_audio(
        audio_path=audio_path,
        model_path=model_path,
        k=1,
    )
    if not top:
        return "neutral", 0.0
    return str(top[0][0]), float(top[0][1])


def predict_emotion_from_audio(
    audio_path: str, model_path: str = "models/emotion_audio_model.joblib"
) -> str:
    # Wrapper simple: renvoie seulement l'emotion dominante.
    emotion, _confidence = predict_emotion_from_audio_with_confidence(
        audio_path=audio_path, model_path=model_path
    )
    return emotion
