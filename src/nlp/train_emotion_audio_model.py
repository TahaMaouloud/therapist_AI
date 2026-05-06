from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.nlp.emotion_audio import (
    _load_audio,
    extract_audio_features,
    map_emotion_label,
    parse_emotion_label,
)

# Entrainement emotion audio multi-modeles:
# dataset audio -> benchmark (RF/CNN/LSTM/Wav2Vec2) -> selection best model -> export.

os.environ.setdefault("DISABLE_SAFETENSORS_CONVERSION", "1")

# Dependances deep learning optionnelles:
# - si absentes, les modeles sklearn continuent de fonctionner
# - les modeles deep (CNN/LSTM/Wav2Vec2) sont ignores proprement
try:
    import torch
    from torch import nn
except Exception:
    torch = None
    nn = None

try:
    from transformers import AutoFeatureExtractor, AutoModel
except Exception:
    AutoFeatureExtractor = None
    AutoModel = None


def _log(message: str) -> None:
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


if nn is not None:
    # CNN sur des features audio tabulaires (features hand-crafted).

    class TabularCNNNet(nn.Module):
        def __init__(self, num_classes: int) -> None:
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv1d(1, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool1d(2),
                nn.Conv1d(32, 64, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(p=0.2),
                nn.Linear(64, num_classes),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = x.unsqueeze(1)
            x = self.features(x)
            return self.classifier(x)


    # LSTM sur les memes features audio tabulaires.
    class TabularLSTMNet(nn.Module):
        def __init__(self, num_classes: int, hidden_size: int = 64) -> None:
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=1,
                hidden_size=hidden_size,
                num_layers=1,
                batch_first=True,
                bidirectional=True,
            )
            self.classifier = nn.Sequential(
                nn.Dropout(p=0.2),
                nn.Linear(hidden_size * 2, num_classes),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = x.unsqueeze(-1)
            output, _ = self.lstm(x)
            last_step = output[:, -1, :]
            return self.classifier(last_step)


class TorchTabularClassifier:
    # Wrapper style sklearn pour comparer CNN/LSTM avec la meme interface.
    def __init__(
        self,
        kind: str,
        num_classes: int,
        random_state: int = 42,
        epochs: int = 18,
        batch_size: int = 64,
        learning_rate: float = 1e-3,
    ) -> None:
        self.kind = kind
        self.num_classes = num_classes
        self.random_state = random_state
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.n_features_in_: int | None = None
        self._model: Any = None
        self._device: Any = None

    def _build_model(self) -> Any:
        if nn is None:
            raise RuntimeError("PyTorch is not available.")
        if self.kind == "cnn":
            return TabularCNNNet(num_classes=self.num_classes)
        if self.kind == "lstm":
            return TabularLSTMNet(num_classes=self.num_classes)
        raise ValueError(f"Unsupported torch model kind: {self.kind}")

    def _set_seed(self) -> None:
        random.seed(self.random_state)
        np.random.seed(self.random_state)
        if torch is not None:
            torch.manual_seed(self.random_state)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(self.random_state)

    def fit(self, x: np.ndarray, y: np.ndarray) -> "TorchTabularClassifier":
        if torch is None or nn is None:
            raise RuntimeError("PyTorch is required for CNN/LSTM training.")

        # Etape 1: initialiser seed et device.
        self._set_seed()
        self.n_features_in_ = int(x.shape[1])
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Etape 2: creer modele + optimiseur + loss.
        model = self._build_model().to(self._device)
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)
        loss_fn = nn.CrossEntropyLoss()

        # Etape 3: convertir numpy -> tensors et construire DataLoader.
        x_tensor = torch.from_numpy(x.astype(np.float32))
        y_tensor = torch.from_numpy(y.astype(np.int64))
        dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=False,
        )

        # Etape 4: boucle d'entrainement sur plusieurs epochs.
        model.train()
        for _ in range(self.epochs):
            for batch_x, batch_y in loader:
                batch_x = batch_x.to(self._device)
                batch_y = batch_y.to(self._device)

                optimizer.zero_grad()
                logits = model(batch_x)
                loss = loss_fn(logits, batch_y)
                loss.backward()
                optimizer.step()

        # Etape 5: passer en mode eval et stocker le modele sur CPU.
        model.eval()
        self._model = model.to("cpu")
        self._device = torch.device("cpu")
        return self

    def _predict_logits(self, x: np.ndarray) -> np.ndarray:
        if torch is None or self._model is None:
            raise RuntimeError("Model is not trained or PyTorch is unavailable.")

        x_tensor = torch.from_numpy(np.asarray(x, dtype=np.float32))
        if x_tensor.ndim == 1:
            x_tensor = x_tensor.reshape(1, -1)

        with torch.no_grad():
            logits = self._model(x_tensor)
        return logits.numpy()

    def predict(self, x: np.ndarray) -> np.ndarray:
        logits = self._predict_logits(x)
        return np.argmax(logits, axis=1)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if torch is None:
            raise RuntimeError("PyTorch is unavailable.")
        logits = self._predict_logits(x)
        probs = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
        return probs


class Wav2Vec2EmbeddingClassifier:
    # Signal audio brut -> embeddings Wav2Vec2 -> classifieur LogisticRegression.
    def __init__(
        self,
        backbone: str = "facebook/wav2vec2-base",
        random_state: int = 42,
        max_seconds: float = 6.0,
        sample_rate: int = 16000,
        pooling: str = "meanstd",
        classifier_c_grid: tuple[float, ...] = (0.5, 1.0, 2.0, 4.0, 8.0),
        cv_folds: int = 3,
        embedding_cache_dir: str | None = None,
    ) -> None:
        self.backbone = backbone
        self.random_state = random_state
        self.max_seconds = max_seconds
        self.sample_rate = sample_rate
        self.pooling = pooling
        self.classifier_c_grid = tuple(float(c) for c in classifier_c_grid)
        self.cv_folds = max(2, int(cv_folds))
        self.embedding_cache_dir = str(embedding_cache_dir or "").strip()
        self.n_features_in_: int | None = None
        self._extractor: Any = None
        self._encoder: Any = None
        self._classifier: Any = None
        self._embedding_cache: dict[str, np.ndarray] = {}
        self._cache_signature = self._build_cache_signature()
        self._cache_dir_path = self._resolve_cache_dir_path()
        self._disk_cache_reads = 0
        self._disk_cache_writes = 0
        self.classifier_best_params_: dict[str, Any] | None = None
        self.classifier_cv_best_score_: float | None = None
        self.classifier_cv_results_: list[dict[str, float]] = []

    def _build_cache_signature(self) -> str:
        raw = "|".join(
            [
                str(self.backbone),
                str(self.sample_rate),
                str(self.max_seconds),
                str(self.pooling),
            ]
        )
        return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]

    def _resolve_cache_dir_path(self) -> Path | None:
        if not self.embedding_cache_dir:
            return None

        cache_dir = Path(self.embedding_cache_dir) / self._cache_signature
        try:
            cache_dir.mkdir(parents=True, exist_ok=True)
            meta_path = cache_dir / "meta.json"
            if not meta_path.exists():
                metadata = {
                    "backbone": self.backbone,
                    "sample_rate": self.sample_rate,
                    "max_seconds": self.max_seconds,
                    "pooling": self.pooling,
                    "signature": self._cache_signature,
                }
                meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        except Exception:
            return None

        return cache_dir

    def _cache_file_path(self, cache_key: str) -> Path | None:
        if self._cache_dir_path is None:
            return None
        digest = hashlib.sha1(cache_key.encode("utf-8")).hexdigest()
        return self._cache_dir_path / f"{digest}.npy"

    def _load_cached_embedding_from_disk(self, cache_key: str) -> np.ndarray | None:
        file_path = self._cache_file_path(cache_key)
        if file_path is None or not file_path.exists():
            return None
        try:
            loaded = np.load(file_path, allow_pickle=False)
            embedding = np.asarray(loaded, dtype=np.float32).reshape(-1)
            self._disk_cache_reads += 1
            return embedding
        except Exception:
            return None

    def _save_cached_embedding_to_disk(self, cache_key: str, embedding: np.ndarray) -> None:
        file_path = self._cache_file_path(cache_key)
        if file_path is None:
            return

        temp_path = file_path.with_suffix(".tmp.npy")
        try:
            np.save(temp_path, np.asarray(embedding, dtype=np.float32), allow_pickle=False)
            temp_path.replace(file_path)
            self._disk_cache_writes += 1
        except Exception:
            try:
                temp_path.unlink(missing_ok=True)
            except Exception:
                pass

    def _ensure_backbone(self) -> None:
        if torch is None or AutoFeatureExtractor is None or AutoModel is None:
            raise RuntimeError(
                "Wav2Vec2 requires torch + transformers (non installs dans cet environnement)."
            )
        if self._extractor is None or self._encoder is None:
            try:
                self._extractor = AutoFeatureExtractor.from_pretrained(
                    self.backbone, local_files_only=True
                )
                self._encoder = AutoModel.from_pretrained(
                    self.backbone, local_files_only=True, use_safetensors=False
                )
            except Exception:
                self._extractor = AutoFeatureExtractor.from_pretrained(self.backbone)
                self._encoder = AutoModel.from_pretrained(
                    self.backbone, use_safetensors=False
                )
            self._encoder.eval()

    def __getstate__(self) -> dict[str, Any]:
        # Keep serialized model lightweight and avoid pickling PyTorch modules directly.
        state = self.__dict__.copy()
        state["_extractor"] = None
        state["_encoder"] = None
        state["_embedding_cache"] = {}
        state["_cache_dir_path"] = None
        state["_disk_cache_reads"] = 0
        state["_disk_cache_writes"] = 0
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        legacy_defaults = {
            "backbone": "facebook/wav2vec2-base",
            "random_state": 42,
            "max_seconds": 6.0,
            "sample_rate": 16000,
            "pooling": "meanstd",
            "classifier_c_grid": (0.5, 1.0, 2.0, 4.0, 8.0),
            "cv_folds": 3,
            "embedding_cache_dir": "",
            "n_features_in_": None,
            "classifier_best_params_": None,
            "classifier_cv_best_score_": None,
            "classifier_cv_results_": [],
        }
        for attr_name, default_value in legacy_defaults.items():
            if not hasattr(self, attr_name):
                setattr(self, attr_name, default_value)
        self._extractor = None
        self._encoder = None
        self._embedding_cache = {}
        self._cache_signature = self._build_cache_signature()
        self._cache_dir_path = self._resolve_cache_dir_path()
        self._disk_cache_reads = 0
        self._disk_cache_writes = 0

    def _build_classifier(self, c_value: float) -> Pipeline:
        return Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=1200,
                        class_weight="balanced",
                        random_state=self.random_state,
                        C=float(c_value),
                        solver="lbfgs",
                    ),
                ),
            ]
        )

    def _cache_key(self, audio_path: Path) -> str:
        stat = audio_path.stat()
        resolved = str(audio_path.resolve()).lower()
        return f"{resolved}|{stat.st_size}|{int(stat.st_mtime_ns)}"

    def _pool_hidden_states(
        self,
        hidden_states: Any,
        attention_mask: Any | None,
    ) -> np.ndarray:
        hidden = hidden_states.squeeze(0)  # (seq_len, hidden_dim)
        mask = None
        if attention_mask is not None:
            # Assure que le masque est 1D et a la bonne forme
            mask = attention_mask.squeeze(0).squeeze(0)  # Enlève les dimensions batch
            if mask.dim() == 1:
                mask = mask.unsqueeze(-1)  # (seq_len, 1)
            mask = mask.to(hidden.dtype)

        if mask is None:
            pooled_mean = hidden.mean(dim=0)
            if self.pooling == "mean":
                return pooled_mean.cpu().numpy().astype(np.float32)
            pooled_std = hidden.std(dim=0, unbiased=False)
        else:
            # Vérifie que seq_len correspond
            if mask.shape[0] != hidden.shape[0]:
                # Si les dimensions ne correspondent pas, force le masque à être tous les 1
                mask = torch.ones((hidden.shape[0], 1), dtype=hidden.dtype, device=hidden.device)
            
            denom = mask.sum(dim=0).clamp(min=1.0)
            pooled_mean = (hidden * mask).sum(dim=0) / denom
            if self.pooling == "mean":
                return pooled_mean.cpu().numpy().astype(np.float32)
            centered = (hidden - pooled_mean.unsqueeze(0)) * mask
            pooled_std = torch.sqrt((centered.square().sum(dim=0) / denom).clamp(min=1e-6))

        pooled = torch.cat([pooled_mean, pooled_std], dim=0)
        return pooled.cpu().numpy().astype(np.float32)

    def _fit_classifier(self, x: np.ndarray, y: np.ndarray) -> None:
        unique_counts = np.unique(y, return_counts=True)[1]
        max_cv_folds = int(unique_counts.min()) if len(unique_counts) else 0
        use_search = len(self.classifier_c_grid) > 1 and max_cv_folds >= 2

        if not use_search:
            default_c = self.classifier_c_grid[0] if self.classifier_c_grid else 1.0
            _log(f"[wav2vec2] classifier: fitting logistic head with C={default_c}")
            self._classifier = self._build_classifier(default_c)
            self._classifier.fit(x, y)
            self.classifier_best_params_ = {"clf__C": float(default_c)}
            self.classifier_cv_best_score_ = None
            self.classifier_cv_results_ = []
            return

        cv_folds = min(self.cv_folds, max_cv_folds)
        _log(
            "[wav2vec2] classifier: grid search "
            f"over {len(self.classifier_c_grid)} values x {cv_folds} folds"
        )
        grid = GridSearchCV(
            estimator=self._build_classifier(self.classifier_c_grid[0]),
            param_grid={"clf__C": list(self.classifier_c_grid)},
            scoring="f1_macro",
            cv=StratifiedKFold(
                n_splits=cv_folds,
                shuffle=True,
                random_state=self.random_state,
            ),
            n_jobs=None,
            refit=True,
        )
        grid.fit(x, y)
        _log(
            "[wav2vec2] classifier: best "
            f"C={grid.best_params_.get('clf__C')} | macro_f1={float(grid.best_score_):.4f}"
        )
        self._classifier = grid.best_estimator_
        self.classifier_best_params_ = {
            key: float(value) if isinstance(value, (float, int, np.floating, np.integer)) else value
            for key, value in grid.best_params_.items()
        }
        self.classifier_cv_best_score_ = float(grid.best_score_)
        self.classifier_cv_results_ = [
            {
                "clf__C": float(params["clf__C"]),
                "mean_test_score": float(score),
            }
            for params, score in zip(
                grid.cv_results_["params"],
                grid.cv_results_["mean_test_score"],
                strict=False,
            )
        ]

    def _embed_one(self, audio_path: Path) -> np.ndarray:
        if torch is None:
            raise RuntimeError("PyTorch is required for Wav2Vec2.")

        cache_key = self._cache_key(audio_path)
        cached = self._embedding_cache.get(cache_key)
        if cached is not None:
            return cached
        disk_cached = self._load_cached_embedding_from_disk(cache_key)
        if disk_cached is not None:
            self._embedding_cache[cache_key] = disk_cached
            return disk_cached

        # Etape 1: charger backbone + audio (resample en 16kHz).
        self._ensure_backbone()
        y, sr = _load_audio(audio_path, sr=self.sample_rate)
        if len(y) == 0:
            raise ValueError(f"Empty audio: {audio_path}")

        # Etape 2: tronquer pour limiter cout memoire/temps.
        max_len = int(self.max_seconds * sr)
        if max_len > 0 and len(y) > max_len:
            y = y[:max_len]

        # Etape 3: encoder audio et faire un mean-pooling temporel.
        inputs = self._extractor(
            y,
            sampling_rate=sr,
            return_tensors="pt",
            padding=True,
        )

        with torch.no_grad():
            outputs = self._encoder(**inputs)
        pooled = self._pool_hidden_states(
            hidden_states=outputs.last_hidden_state,
            attention_mask=inputs.get("attention_mask"),
        )
        pooled = np.asarray(pooled, dtype=np.float32).reshape(-1)
        self._embedding_cache[cache_key] = pooled
        self._save_cached_embedding_to_disk(cache_key, pooled)
        return pooled

    def embed_paths(self, audio_paths: list[Path], stage: str = "embeddings") -> np.ndarray:
        embeddings: list[np.ndarray] = []
        total = len(audio_paths)
        reads_before = self._disk_cache_reads
        writes_before = self._disk_cache_writes
        _log(f"[wav2vec2] {stage}: starting {total} files")
        if self._cache_dir_path is not None:
            _log(f"[wav2vec2] {stage}: disk cache enabled -> {self._cache_dir_path}")
        for idx, path in enumerate(audio_paths, start=1):
            if idx == 1 or idx % 100 == 0 or idx == total:
                _log(f"[wav2vec2] {stage}: {idx}/{total} | {path.name}")
            embeddings.append(self._embed_one(path))
        if self._cache_dir_path is not None:
            disk_hits = self._disk_cache_reads - reads_before
            disk_new = self._disk_cache_writes - writes_before
            _log(f"[wav2vec2] {stage}: disk cache hits={disk_hits} | newly saved={disk_new}")
        return np.vstack(embeddings)

    def fit_paths(self, audio_paths: list[Path], y: np.ndarray) -> "Wav2Vec2EmbeddingClassifier":
        x = self.embed_paths(audio_paths, stage="fit embeddings")
        self.n_features_in_ = int(x.shape[1])
        self._fit_classifier(x, y)
        return self

    def predict_paths(self, audio_paths: list[Path]) -> np.ndarray:
        if self._classifier is None:
            raise RuntimeError("Wav2Vec2 classifier is not fitted.")
        x = self.embed_paths(audio_paths, stage="predict embeddings")
        return self._classifier.predict(x)

    def predict_proba_paths(self, audio_paths: list[Path]) -> np.ndarray:
        if self._classifier is None:
            raise RuntimeError("Wav2Vec2 classifier is not fitted.")
        x = self.embed_paths(audio_paths, stage="predict_proba embeddings")
        return self._classifier.predict_proba(x)


# Ensure custom estimators are pickled with a stable import path.
# This avoids "__main__" references when script is launched with "-m".
TorchTabularClassifier.__module__ = "src.nlp.train_emotion_audio_model"
Wav2Vec2EmbeddingClassifier.__module__ = "src.nlp.train_emotion_audio_model"

if __name__ == "__main__":
    sys.modules.setdefault("src.nlp.train_emotion_audio_model", sys.modules[__name__])


def resolve_dataset_roots(args: argparse.Namespace) -> list[Path]:
    # Resolution des dossiers dataset (compatible ancien usage).
    if args.dataset_roots:
        raw_roots = args.dataset_roots
    elif args.dataset_root:
        raw_roots = [args.dataset_root]
    else:
        raw_roots = ["data/raw_audio/archive", "data/AudioWAV"]

    roots: list[Path] = []
    for root in raw_roots:
        path = Path(root)
        if path.exists():
            roots.append(path)
        else:
            print(f"[warning] Dataset root not found, skipped: {path}")

    if not roots:
        raise RuntimeError("No valid dataset root found.")

    return roots


def find_audio_files(dataset_roots: list[Path]) -> list[Path]:
    # Collecte et deduplication des .wav sur plusieurs dossiers source.
    wav_files: list[Path] = []
    seen_keys: set[str] = set()
#Il scanne tous les .wav des 2 dossiers
    for root in dataset_roots:
        for path in root.rglob("*.wav"):
            try:
                file_size = path.stat().st_size
            except OSError:
                continue

            # Some archives duplicate directory trees; keep one copy per filename+size.
            dedupe_key = f"{path.name.lower()}::{file_size}"
            if dedupe_key in seen_keys:
                continue
            seen_keys.add(dedupe_key)
            wav_files.append(path)

    return wav_files


def build_dataset(
    dataset_roots: list[Path],
    label_scheme: str,
    max_samples: int | None,
    random_state: int,
    compute_features: bool = True,
) -> tuple[np.ndarray | None, np.ndarray, list[Path]]:
    # Construction des features tabulaires + labels a partir des audios bruts.
    audio_paths = find_audio_files(dataset_roots)
    if max_samples is not None and max_samples > 0 and len(audio_paths) > max_samples:
        rng = np.random.default_rng(random_state)
        selected_idx = rng.choice(len(audio_paths), size=max_samples, replace=False)
        audio_paths = [audio_paths[int(i)] for i in selected_idx]

    features: list[np.ndarray] = []
    labels: list[str] = []
    kept_paths: list[Path] = []
#Pour chaque audio, il construit le dataset d’entraînement
    for idx, path in enumerate(audio_paths, start=1):
        if idx % 200 == 0:
            _log(f"[dataset] scanned {idx}/{len(audio_paths)} files")

        # Etape A: label depuis le nom de fichier + mapping vers schema cible.
        raw_emotion = parse_emotion_label(path)
        emotion = map_emotion_label(raw_emotion, scheme=label_scheme)
        if emotion is None:
            continue

        if compute_features:
            try:
                # Etape B: extraction du vecteur de features.
                feats = extract_audio_features(path)
            except Exception:
                continue
            features.append(feats)
        labels.append(emotion)
        kept_paths.append(path)

    if not labels:
        roots_msg = ", ".join(str(root) for root in dataset_roots)
        raise RuntimeError(
            f"No usable audio found under: {roots_msg}. "
            "Check filename conventions and audio readability."
        )

    x = np.array(features) if compute_features else None
    return x, np.array(labels), kept_paths


def train_and_score_tabular(
    estimator: Any,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
) -> tuple[Any, float, np.ndarray]:
    # Evaluation commune RF/CNN/LSTM sur le split de validation.
    estimator.fit(x_train, y_train)
    pred = estimator.predict(x_eval)
    score = float(f1_score(y_eval, pred, average="macro"))
    return estimator, score, pred


def train_and_score_wav2vec2(
    estimator: Wav2Vec2EmbeddingClassifier,
    train_paths: list[Path],
    y_train: np.ndarray,
    eval_paths: list[Path],
    y_eval: np.ndarray,
) -> tuple[Any, float, np.ndarray]:
    # Evaluation dediee aux modeles qui prennent des chemins de fichiers.
    estimator.fit_paths(train_paths, y_train)
    pred = estimator.predict_paths(eval_paths)
    score = float(f1_score(y_eval, pred, average="macro"))
    return estimator, score, pred


def main() -> None:
    # Pipeline CLI:
    # 1) lire args, 2) construire dataset, 3) split train/val/test
    # 4) comparer les modeles via macro-F1 sur val
    # 5) re-entrainer le meilleur sur train, evaluer sur test
    # 6) sauvegarder le bundle modele + metriques
    parser = argparse.ArgumentParser(
        description="Train and benchmark audio emotion models over multiple datasets" 
    )
    parser.add_argument( 
        "--dataset-root", 
        type=str,   
        default=None,  
        help="Single dataset root (backward-compatible option)",  
    )
    parser.add_argument(
        "--dataset-roots",
        nargs="+",
        default=None,
        help="One or more dataset roots (ex: data/raw_audio/archive data/AudioWAV)",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["wav2vec2"],
        help="Models to benchmark among: wav2vec2 cnn lstm random_forest",
    )
    parser.add_argument(
        "--label-scheme",
        type=str,
        default="core5",
        choices=["full", "core5"],
        help="Emotion granularity for training",
    )
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--val-size", type=float, default=0.15)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--deep-epochs", type=int, default=18)
    parser.add_argument(
        "--rf-n-jobs",
        type=int,
        default=1,
        help="Parallel workers for RandomForest (-1 for all cores, 1 for safer Windows runs)",
    )
    parser.add_argument(
        "--wav2vec2-backbone",
        type=str,
        default="facebook/wav2vec2-base",
    )
    parser.add_argument("--wav2vec2-max-seconds", type=float, default=6.0)
    parser.add_argument(
        "--wav2vec2-pooling",
        type=str,
        default="meanstd",
        choices=["mean", "meanstd"],
        help="Temporal pooling strategy for Wav2Vec2 embeddings",
    )
    parser.add_argument(
        "--wav2vec2-c-grid",
        nargs="+",
        type=float,
        default=[0.5, 1.0, 2.0, 4.0, 8.0],
        help="Candidate C values for the Wav2Vec2 logistic classifier",
    )
    parser.add_argument(
        "--wav2vec2-cv-folds",
        type=int,
        default=3,
        help="Inner CV folds to tune the Wav2Vec2 classifier head",
    )
    parser.add_argument(
        "--wav2vec2-cache-dir",
        type=str,
        default="models/wav2vec2_embedding_cache",
        help=(
            "Persistent cache directory for Wav2Vec2 embeddings. "
            "Reuse computed embeddings across reruns and resume after interruption."
        ),
    )
    parser.add_argument(
        "--model-out",
        type=str,
        default="models/emotion_audio_model.joblib",
    )
    parser.add_argument(
        "--metrics-out",
        type=str,
        default="models/emotion_audio_metrics.json",
    )
    args = parser.parse_args()

    requested_models = [m.lower() for m in args.models]
    supported_models = {"wav2vec2", "cnn", "lstm", "random_forest"}
    unknown_models = [m for m in requested_models if m not in supported_models]
    if unknown_models:
        raise ValueError(f"Unsupported model names: {unknown_models}")
    needs_tabular_features = any(
        model_name in {"random_forest", "cnn", "lstm"} for model_name in requested_models
    )

    dependency_status = {
        "torch": bool(torch is not None),
        "transformers": bool(
            AutoFeatureExtractor is not None and AutoModel is not None
        ),
    }
    _log(f"[env] python={sys.executable}")
    _log(
        "[env] dependencies: "
        f"torch={dependency_status['torch']} | "
        f"transformers={dependency_status['transformers']}"
    )
    _log(f"[env] requested models: {', '.join(requested_models)}")

    # Etape 1: preparation chemins sortie + construction dataset.
    dataset_roots = resolve_dataset_roots(args)
    model_out = Path(args.model_out)
    metrics_out = Path(args.metrics_out)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    metrics_out.parent.mkdir(parents=True, exist_ok=True)

    x, y_raw, paths = build_dataset(
        dataset_roots=dataset_roots,
        label_scheme=args.label_scheme,
        max_samples=args.max_samples,
        random_state=args.random_state,
        compute_features=needs_tabular_features,
    )
    _log(
        f"[dataset] usable samples={len(y_raw)} | tabular_features={'yes' if x is not None else 'no'}"
    )

    # Etape 2: encodage labels texte -> classes numeriques.
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw)

    # Etape 3: split train/test puis split interne fit/val pour benchmark.
    indices = np.arange(len(y))
    idx_train, idx_test = train_test_split(
        indices,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )
    idx_fit, idx_val = train_test_split(
        idx_train,
        test_size=args.val_size,
        random_state=args.random_state,
        stratify=y[idx_train],
    )

    candidate_factories: dict[str, Any] = {}
    skipped_models: dict[str, str] = {}
    trained_models: dict[str, Any] = {}

    # Baseline ML classique sur features hand-crafted.
    if "random_forest" in requested_models:
        candidate_factories["random_forest"] = lambda: RandomForestClassifier(
            n_estimators=700,
            random_state=args.random_state,
            class_weight="balanced",
            n_jobs=args.rf_n_jobs,
        )

    # Modeles deep sur features tabulaires.
    if "cnn" in requested_models:
        if torch is None or nn is None:
            skipped_models["cnn"] = "PyTorch not available in this Python environment."
        else:
            candidate_factories["cnn"] = lambda: TorchTabularClassifier(
                kind="cnn",
                num_classes=len(label_encoder.classes_),
                random_state=args.random_state,
                epochs=args.deep_epochs,
            )

    if "lstm" in requested_models:
        if torch is None or nn is None:
            skipped_models["lstm"] = "PyTorch not available in this Python environment."
        else:
            candidate_factories["lstm"] = lambda: TorchTabularClassifier(
                kind="lstm",
                num_classes=len(label_encoder.classes_),
                random_state=args.random_state,
                epochs=args.deep_epochs,
            )

    # Representation speech pre-entrainee + classifieur leger.
    if "wav2vec2" in requested_models:
        if torch is None or AutoFeatureExtractor is None or AutoModel is None:
            skipped_models["wav2vec2"] = (
                "torch + transformers not available in this Python environment."
            )
        else:
            candidate_factories["wav2vec2"] = lambda: Wav2Vec2EmbeddingClassifier(
                backbone=args.wav2vec2_backbone,
                random_state=args.random_state,
                max_seconds=args.wav2vec2_max_seconds,
                pooling=args.wav2vec2_pooling,
                classifier_c_grid=tuple(args.wav2vec2_c_grid),
                cv_folds=args.wav2vec2_cv_folds,
                embedding_cache_dir=args.wav2vec2_cache_dir,
            )

    if not candidate_factories:
        raise RuntimeError(
            "No candidate model available. Install missing dependencies or change --models."
        )

    # Etape 4: benchmark de chaque modele sur le meme split validation.
    val_scores: dict[str, float] = {}
    validation_details: dict[str, dict[str, Any]] = {}

    # Benchmark validation: chaque modele utilise le meme split fit/val.
    for name, factory in candidate_factories.items():
        try:
            model = factory()
            _log(f"[validation] training {name} ...")

            if name == "wav2vec2":
                _, score, val_pred = train_and_score_wav2vec2(
                    estimator=model,
                    train_paths=[paths[i] for i in idx_fit],
                    y_train=y[idx_fit],
                    eval_paths=[paths[i] for i in idx_val],
                    y_eval=y[idx_val],
                )
            else:
                _, score, val_pred = train_and_score_tabular(
                    estimator=model,
                    x_train=x[idx_fit],
                    y_train=y[idx_fit],
                    x_eval=x[idx_val],
                    y_eval=y[idx_val],
                )

            trained_models[name] = model
            val_acc = float(accuracy_score(y[idx_val], val_pred))
            val_report = classification_report(
                label_encoder.inverse_transform(y[idx_val]),
                label_encoder.inverse_transform(val_pred),
                output_dict=True,
                zero_division=0,
            )

            val_scores[name] = score
            validation_details[name] = {
                "macro_f1": score,
                "accuracy": val_acc,
                "classification_report": val_report,
            }
            _log(f"[validation] {name} macro_f1={score:.4f} | accuracy={val_acc:.4f}")
        except Exception as exc:
            skipped_models[name] = str(exc)
            safe_exc = str(exc).encode("ascii", "ignore").decode("ascii")
            _log(f"[validation] skipped {name}: {safe_exc}")

    if not val_scores:
        raise RuntimeError(
            "All candidate models failed during validation. "
            "Check metrics file for failure reasons."
        )

    best_name = max(val_scores, key=val_scores.get)
    best_val_score = val_scores[best_name]

    validation_ranking = sorted(
        (
            {
                "model": model_name,
                "macro_f1": float(model_metrics.get("macro_f1", 0.0)),
                "accuracy": float(model_metrics.get("accuracy", 0.0)),
            }
            for model_name, model_metrics in validation_details.items()
        ),
        key=lambda item: item["macro_f1"],
        reverse=True,
    )
    inferior_models = [
        {
            "model": row["model"],
            "macro_f1": row["macro_f1"],
            "accuracy": row["accuracy"],
            "delta_macro_f1_vs_best": float(best_val_score - row["macro_f1"]),
        }
        for row in validation_ranking
        if row["model"] != best_name
    ]

    # Etape 5: re-entrainement du meilleur modele sur train complet.
    # Refit best model on train split (fit+val), evaluate on holdout test split.
    best_model = trained_models.get(best_name) or candidate_factories[best_name]()
    _log(f"[final] best model={best_name} | validation_macro_f1={best_val_score:.4f}")
    if best_name == "wav2vec2":
        _log("[final] reusing cached wav2vec2 embeddings for the refit stage")
        best_model.fit_paths([paths[i] for i in idx_train], y[idx_train])
        y_pred = best_model.predict_paths([paths[i] for i in idx_test])
        feature_version = f"wav2vec2_{args.wav2vec2_pooling}_pool"
    else:
        _log(f"[final] refitting {best_name} on full train split")
        best_model.fit(x[idx_train], y[idx_train])
        y_pred = best_model.predict(x[idx_test])
        feature_version = "v2_prosody_spectral"

    y_test = y[idx_test]
    acc = float(accuracy_score(y_test, y_pred))
    report = classification_report(
        label_encoder.inverse_transform(y_test),
        label_encoder.inverse_transform(y_pred),
        output_dict=True,
    )

    # Etape 6: sauvegarder le bundle d'inference.
    bundle = {
        "model": best_model,
        "model_name": best_name,
        "label_encoder": label_encoder,
        "label_scheme": args.label_scheme,
        "feature_version": feature_version,
    }
    if best_name == "wav2vec2":
        bundle["wav2vec2_backbone"] = args.wav2vec2_backbone
        bundle["wav2vec2_pooling"] = args.wav2vec2_pooling
        bundle["wav2vec2_cache_dir"] = args.wav2vec2_cache_dir
        bundle["wav2vec2_classifier_best_params"] = best_model.classifier_best_params_
        bundle["wav2vec2_classifier_cv_best_score"] = best_model.classifier_cv_best_score_

    _log(f"[save] writing model bundle to {model_out}")
    joblib.dump(bundle, model_out)

    label_counts = {
        label: int(count)
        for label, count in zip(*np.unique(y_raw, return_counts=True), strict=False)
    }

    metrics = {
        "accuracy": acc,
        "selected_model": best_name,
        "validation_macro_f1": best_val_score,
        "validation_scores": val_scores,
        "validation_details": validation_details,
        "validation_ranking": validation_ranking,
        "inferior_models": inferior_models,
        "skipped_models": skipped_models,
        "models_requested": requested_models,
        "dataset_roots": [str(root) for root in dataset_roots],
        "samples_total": int(len(y)),
        "samples_train": int(len(idx_train)),
        "samples_test": int(len(idx_test)),
        "label_scheme": args.label_scheme,
        "python_executable": sys.executable,
        "dependency_status": dependency_status,
        "labels": list(label_encoder.classes_),
        "label_counts": label_counts,
        "classification_report": report,
    }
    if best_name == "wav2vec2":
        metrics["wav2vec2_backbone"] = args.wav2vec2_backbone
        metrics["wav2vec2_pooling"] = args.wav2vec2_pooling
        metrics["wav2vec2_cache_dir"] = args.wav2vec2_cache_dir
        metrics["wav2vec2_classifier_best_params"] = best_model.classifier_best_params_
        metrics["wav2vec2_classifier_cv_best_score"] = best_model.classifier_cv_best_score_
        metrics["wav2vec2_classifier_cv_results"] = best_model.classifier_cv_results_
    # Etape 7: exporter les metriques (json) pour analyse/comparaison.
    _log(f"[save] writing metrics to {metrics_out}")
    metrics_out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("\n=== Validation ranking (macro_f1 desc) ===", flush=True)
    for row in validation_ranking:
        print(
            f"- {row['model']}: macro_f1={row['macro_f1']:.4f} "
            f"| accuracy={row['accuracy']:.4f}"
        , flush=True)
    if inferior_models:
        print("\n=== Inferior models vs best ===", flush=True)
        for row in inferior_models:
            print(
                f"- {row['model']}: delta_macro_f1_vs_best={row['delta_macro_f1_vs_best']:.4f}"
            , flush=True)
    if skipped_models:
        print("\n=== Skipped models ===", flush=True)
        for name, reason in skipped_models.items():
            safe_reason = str(reason).encode("ascii", "ignore").decode("ascii")
            print(f"- {name}: {safe_reason}", flush=True)

    _log(f"Training completed. Best={best_name} | Accuracy={acc:.4f}")
    _log(f"Model saved to: {model_out}")
    _log(f"Metrics saved to: {metrics_out}")


if __name__ == "__main__":
    main()
