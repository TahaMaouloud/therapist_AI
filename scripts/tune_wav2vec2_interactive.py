#!/usr/bin/env python3
"""
Script interactif pour fine-tuner Wav2Vec2 avec paramètres personnalisés.
Demande à l'utilisateur d'entrer les paramètres et lance l'entraînement.
"""
from __future__ import annotations

import sys
import hashlib
from datetime import datetime
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import argparse
import json
import numpy as np


def configure_console_output() -> None:
    """
    Avoid Windows console crashes when printing unicode symbols.
    """
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None:
            continue
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(errors="replace")
            except Exception:
                pass


def ensure_wav2vec2_dependencies() -> None:
    """
    Fail fast with a clear action plan when required deps are missing.
    """
    errors: list[str] = []

    try:
        import torch  # noqa: F401
    except Exception as exc:
        errors.append(f"torch: {exc}")

    try:
        import transformers  # noqa: F401
    except Exception as exc:
        errors.append(f"transformers: {exc}")

    if not errors:
        return

    venv_python = project_root / ".venv_dl" / "Scripts" / "python.exe"
    if venv_python.exists():
        rerun_hint = f"\"{venv_python}\" scripts/tune_wav2vec2_interactive.py"
    else:
        rerun_hint = "python -m pip install torch transformers"

    details = "; ".join(errors)
    raise RuntimeError(
        "Wav2Vec2 requires torch + transformers, but they are unavailable in the current interpreter.\n"
        f"Current python: {sys.executable}\n"
        f"Details: {details}\n"
        f"Try: {rerun_hint}"
    )


def load_training_components() -> tuple[object, ...]:
    """
    Import heavy training dependencies only when training is about to start.
    This keeps the interactive prompts responsive instead of waiting silently.
    """
    print(
        "\nChargement des modules d'entraînement "
        "(scikit-learn / scipy / wav2vec2). Patiente 30 a 60 secondes au premier lancement...\n",
        flush=True,
    )
    from sklearn.metrics import accuracy_score, classification_report, f1_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import LabelEncoder

    from src.nlp.train_emotion_audio_model import (
        Wav2Vec2EmbeddingClassifier,
        build_dataset,
        resolve_dataset_roots,
        train_and_score_wav2vec2,
    )

    return (
        Wav2Vec2EmbeddingClassifier,
        build_dataset,
        resolve_dataset_roots,
        train_and_score_wav2vec2,
        train_test_split,
        LabelEncoder,
        accuracy_score,
        classification_report,
        f1_score,
    )


def build_embedding_cache_signature(
    backbone: str,
    sample_rate: int,
    max_seconds: float,
    pooling: str,
) -> str:
    raw = "|".join(
        [
            str(backbone),
            str(sample_rate),
            str(max_seconds),
            str(pooling),
        ]
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def inspect_embedding_cache(
    embedding_cache_dir: str,
    backbone: str,
    max_seconds: float,
    pooling: str,
    sample_rate: int = 16000,
) -> tuple[Path, int]:
    signature = build_embedding_cache_signature(
        backbone=backbone,
        sample_rate=sample_rate,
        max_seconds=max_seconds,
        pooling=pooling,
    )
    cache_path = Path(embedding_cache_dir) / signature
    cache_count = len(list(cache_path.glob("*.npy"))) if cache_path.exists() else 0
    return cache_path, cache_count


def get_float_input(prompt: str, default: float) -> float:
    """Demande un nombre float à l'utilisateur avec valeur par défaut."""
    while True:
        try:
            user_input = input(f"{prompt} [défaut: {default}]: ").strip()
            if not user_input:
                return default
            return float(user_input)
        except ValueError:
            print("❌ Veuillez entrer un nombre valide.")


def get_int_input(prompt: str, default: int) -> int:
    """Demande un nombre entier à l'utilisateur avec valeur par défaut."""
    while True:
        try:
            user_input = input(f"{prompt} [défaut: {default}]: ").strip()
            if not user_input:
                return default
            return int(user_input)
        except ValueError:
            print("❌ Veuillez entrer un nombre entier valide.")


def get_choice_input(prompt: str, choices: list[str], default: str) -> str:
    """Demande un choix parmi une liste à l'utilisateur."""
    print(f"\n{prompt}")
    for i, choice in enumerate(choices, 1):
        marker = "✓" if choice == default else " "
        print(f"  {i}. [{marker}] {choice}")

    while True:
        user_input = input("Numéro du choix [défaut: 1]: ").strip()
        if not user_input:
            return default
        try:
            idx = int(user_input) - 1
            if 0 <= idx < len(choices):
                return choices[idx]
            else:
                print(f"❌ Veuillez entrer un numéro entre 1 et {len(choices)}")
        except ValueError:
            print("❌ Veuillez entrer un nombre valide.")


def get_grid_input(prompt: str, default_str: str) -> list[float]:
    """Demande une grille de nombres à l'utilisateur."""
    print(f"\n{prompt}")
    print(f"Format: entrez les nombres séparés par des espaces")
    print(f"Défaut: {default_str}")

    while True:
        user_input = input("Grille C: ").strip()
        if not user_input:
            try:
                return [float(x) for x in default_str.split()]
            except ValueError:
                print("❌ Format invalide pour la grille par défaut.")
                continue
        try:
            values = [float(x) for x in user_input.split()]
            if len(values) < 2:
                print("❌ Veuillez entrer au moins 2 valeurs.")
                continue
            if any(v <= 0 for v in values):
                print("❌ Toutes les valeurs doivent être positives.")
                continue
            return sorted(values)
        except ValueError:
            print("❌ Format invalide. Exemple: 0.1 0.5 1.0 2.0 4.0")


def print_header():
    """Affiche le titre du script."""
    print("\n" + "="*70)
    print(" 🎵 FINE-TUNING WAV2VEC2 POUR DÉTECTION D'ÉMOTIONS AUDIO 🎵")
    print("="*70 + "\n")


def print_section(title: str):
    """Affiche un titre de section."""
    print(f"\n{'─'*70}")
    print(f"  {title}")
    print(f"{'─'*70}\n")


def ensure_unique_output_path(path: str) -> Path:
    """Renvoie un chemin unique en ajoutant un suffixe si le fichier existe déjà."""
    output_path = Path(path)
    if not output_path.exists():
        return output_path

    parent = output_path.parent
    stem = output_path.stem
    suffix = output_path.suffix
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    unique_path = parent / f"{stem}_{timestamp}{suffix}"
    counter = 1
    while unique_path.exists():
        unique_path = parent / f"{stem}_{timestamp}_{counter}{suffix}"
        counter += 1
    return unique_path


def main():
    configure_console_output()
    ensure_wav2vec2_dependencies()
    print_header()

    # ==================== Configuration des Backbones ====================
    print_section("1️⃣  CHOIX DU MODÈLE BACKBONE")
    print("Les modèles pré-entraînés disponibles :\n")
    backbones = [
        ("facebook/wav2vec2-base", "Petit, rapide (12M params)"),
        ("facebook/wav2vec2-large", "Moyen (300M params)"),
        ("facebook/wav2vec2-large-xlsr-53", "Multilingue (300M params)"),
        ("microsoft/wavlm-base-plus", "Meilleur compromis (94M params) ⭐"),
        ("microsoft/wavlm-large", "Le plus puissant = meilleur accuracy (316M params) 🏆"),
    ]

    print("Options entre les backbones:")
    for i, (name, desc) in enumerate(backbones, 1):
        print(f"  {i}. {name}")
        print(f"     └─ {desc}\n")

    backbone = input("Entrez le backbone (1-5) ou le chemin exact [défaut: 4]: ").strip()
    if backbone == "":
        backbone = backbones[3][0]  # microsoft/wavlm-base-plus
    elif backbone.isdigit() and 1 <= int(backbone) <= len(backbones):
        backbone = backbones[int(backbone) - 1][0]

    print(f"✓ Backbone sélectionné: {backbone}\n")

    # ==================== Paramètres Wav2Vec2 ====================
    print_section("2️⃣  PARAMÈTRES WAV2VEC2")

    max_seconds = get_float_input(
        "Durée maximale d'audio en secondes (plus = mieux mais lent)",
        10.0,
    )

    pooling = get_choice_input(
        "Stratégie de pooling temporel:",
        ["mean", "meanstd"],
        "meanstd",
    )

    c_grid_default = "0.001 0.01 0.1 0.5 1.0 2.0 4.0 8.0 16.0"
    c_grid = get_grid_input(
        "Grille de régularisation C pour le classifeur LogisticRegression",
        c_grid_default,
    )
    print(f"✓ Grille C: {c_grid}\n")

    cv_folds = get_int_input(
        "Nombre de folds pour la validation croisée (plus = meilleur mais lent)",
        10,
    )
    cache_dir_input = input(
        "Dossier de cache embeddings [defaut: models/wav2vec2_embedding_cache, 'none' pour desactiver]: "
    ).strip()
    if cache_dir_input.lower() in {"none", "off", "disable", "no", "0"}:
        embedding_cache_dir = ""
    elif cache_dir_input:
        embedding_cache_dir = cache_dir_input
    else:
        embedding_cache_dir = "models/wav2vec2_embedding_cache"

    # ==================== Paramètres Dataset ====================
    print_section("3️⃣  PARAMÈTRES DATASET")

    label_scheme = get_choice_input(
        "Schéma d'étiquettes:",
        ["core5", "full"],
        "core5",
    )

    test_size = get_float_input(
        "Proportion de test (0.1 = 10%)",
        0.2,
    )

    val_size = get_float_input(
        "Proportion de validation (0.1 = 10%)",
        0.15,
    )

    max_samples = input(
        "Nombre maximum d'exemples (laisser vide = tous): "
    ).strip()
    if max_samples:
        try:
            max_samples = int(max_samples)
        except ValueError:
            max_samples = None
            print("⚠️  Utilisation de tous les exemples")
    else:
        max_samples = None

    # ==================== Paramètres Sortie ====================
    print_section("4️⃣  CHEMINS DE SORTIE")

    model_out = input(
        "Chemin du modèle de sortie [défaut: models/emotion_audio_model_tuned.joblib]: "
    ).strip() or "models/emotion_audio_model_tuned.joblib"
    model_out = str(ensure_unique_output_path(model_out))

    metrics_out = input(
        "Chemin des métriques de sortie [défaut: models/emotion_audio_metrics_tuned.json]: "
    ).strip() or "models/emotion_audio_metrics_tuned.json"
    metrics_out = str(ensure_unique_output_path(metrics_out))

    # ==================== Résumé ====================
    print_section("📋 RÉSUMÉ DE LA CONFIGURATION")

    config_str = f"""
    🔧 Modèle:
       • Backbone: {backbone}
       • Max seconds: {max_seconds}s
       • Pooling: {pooling}
       • Grille C: {c_grid}
       • CV Folds: {cv_folds}

    📊 Dataset:
       • Schéma: {label_scheme}
       • Test size: {test_size}
       • Validation size: {val_size}
       • Max samples: {max_samples or 'Tous'}

    💾 Sortie:
       • Modèle: {model_out}
       • Métriques: {metrics_out}
    """
    print(config_str)
    if embedding_cache_dir:
        print(f"    Embedding cache: {embedding_cache_dir}")
        cache_path, cache_count = inspect_embedding_cache(
            embedding_cache_dir=embedding_cache_dir,
            backbone=backbone,
            max_seconds=max_seconds,
            pooling=pooling,
        )
        print(f"    Cache signature path: {cache_path}")
        if cache_count > 0:
            print(f"    Embeddings deja sauvegardes: {cache_count}")
            print("    Reprise possible: ce run reutilisera ces embeddings.")
        else:
            print("    Embeddings deja sauvegardes: 0")
            print("    Attention: aucun cache exploitable trouve pour cette configuration.")
    else:
        print("    Embedding cache: disabled")

    confirm = input("\nLancer l'entraînement? (o/n) [défaut: o]: ").strip().lower()
    if confirm in ["n", "non"]:
        print("❌ Entraînement annulé.")
        return

    # ==================== Entraînement ====================
    print_section("🚀 DÉBUT DE L'ENTRAÎNEMENT")

    try:
        (
            Wav2Vec2EmbeddingClassifier,
            build_dataset,
            resolve_dataset_roots,
            train_and_score_wav2vec2,
            train_test_split,
            LabelEncoder,
            accuracy_score,
            classification_report,
            f1_score,
        ) = load_training_components()

        # 1. Préparation dataset
        print("[1/5] Chargement du dataset...")
        dataset_roots = resolve_dataset_roots(
            argparse.Namespace(
                dataset_roots=None,
                dataset_root=None,
            )
        )

        x, y_raw, paths = build_dataset(
            dataset_roots=dataset_roots,
            label_scheme=label_scheme,
            max_samples=max_samples,
            random_state=42,
        )

        print(f"     ✓ {len(x)} exemples chargés")
        print(f"     ✓ Classes: {set(y_raw)}\n")

        # 2. Encodage labels
        print("[2/5] Encodage des labels...")
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y_raw)
        print(f"     ✓ {len(label_encoder.classes_)} classes\n")

        # 3. Split train/test/val
        print("[3/5] Création des splits train/test/validation...")
        indices = np.arange(len(y))
        idx_train, idx_test = train_test_split(
            indices,
            test_size=test_size,
            random_state=42,
            stratify=y,
        )
        idx_fit, idx_val = train_test_split(
            idx_train,
            test_size=val_size,
            random_state=42,
            stratify=y[idx_train],
        )

        print(f"     ✓ Train: {len(idx_fit)} | Val: {len(idx_val)} | Test: {len(idx_test)}\n")

        # 4. Création et entraînement du modèle Wav2Vec2
        print("[4/5] Entraînement du modèle Wav2Vec2...")
        print(f"     (Cela peut prendre plusieurs minutes...)\n")

        model = Wav2Vec2EmbeddingClassifier(
            backbone=backbone,
            random_state=42,
            max_seconds=max_seconds,
            sample_rate=16000,
            pooling=pooling,
            classifier_c_grid=tuple(c_grid),
            cv_folds=cv_folds,
            embedding_cache_dir=embedding_cache_dir or None,
        )

        train_paths = [paths[i] for i in idx_fit]
        val_paths = [paths[i] for i in idx_val]
        test_paths = [paths[i] for i in idx_test]

        # Entraînement
        model, val_score, val_pred = train_and_score_wav2vec2(
            estimator=model,
            train_paths=train_paths,
            y_train=y[idx_fit],
            eval_paths=val_paths,
            y_eval=y[idx_val],
        )

        print(f"     ✓ Entraînement terminé!\n")

        # 5. Évaluation sur le test set
        print("[5/5] Évaluation sur le test set...")

        test_pred = model.predict_paths(test_paths)
        test_acc = accuracy_score(y[idx_test], test_pred)
        test_f1 = f1_score(y[idx_test], test_pred, average="macro")
        test_report = classification_report(
            label_encoder.inverse_transform(y[idx_test]),
            label_encoder.inverse_transform(test_pred),
            output_dict=True,
            zero_division=0,
        )

        print(f"     ✓ Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
        print(f"     ✓ Macro F1: {test_f1:.4f}\n")

        # ==================== Sauvegarde ====================
        print_section("💾 SAUVEGARDE DES RÉSULTATS")

        # Sauvegarder le modèle
        Path(model_out).parent.mkdir(parents=True, exist_ok=True)
        import joblib

        joblib.dump(model, model_out)
        print(f"     ✓ Modèle sauvegardé: {model_out}")

        # Sauvegarder les métriques
        metrics = {
            "backbone": backbone,
            "max_seconds": max_seconds,
            "pooling": pooling,
            "classifier_c_grid": c_grid,
            "cv_folds": cv_folds,
            "embedding_cache_dir": embedding_cache_dir,
            "label_scheme": label_scheme,
            "validation_macro_f1": float(val_score),
            "test_accuracy": float(test_acc),
            "test_macro_f1": float(test_f1),
            "test_classification_report": test_report,
            "classifier_best_params": model.classifier_best_params_,
            "classifier_cv_best_score": model.classifier_cv_best_score_,
        }

        Path(metrics_out).parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_out, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"     ✓ Métriques sauvegardées: {metrics_out}\n")

        # ==================== Résultats Finaux ====================
        print_section("✅ RÉSULTATS FINAUX")

        results = f"""
╔═══════════════════════════════════════════════════════════╗
║         RÉSULTATS DU FINE-TUNING WAV2VEC2              ║
╠═══════════════════════════════════════════════════════════╣
║                                                           ║
║  🎯 Métriques de Test:                                   ║
║     • Accuracy:  {test_acc:.4f} ({test_acc*100:6.2f}%)    ║
║     • Macro F1:  {test_f1:.4f}                           ║
║                                                           ║
║  ⚙️  Configuration:                                       ║
║     • Backbone:  {backbone:<35} ║
║     • Max Sec:   {max_seconds}                              ║
║     • Pooling:   {pooling}                               ║
║     • CV Folds:  {cv_folds}                               ║
║                                                           ║
║  💾 Fichiers Sauvegardés:                                ║
║     • Modèle:    {model_out:<35} ║
║     • Métriques: {metrics_out:<35} ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
        """
        print(results)

    except Exception as e:
        print(f"\n❌ ERREUR LORS DE L'ENTRAÎNEMENT:")
        print(f"   {type(e).__name__}: {e}\n")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        print(f"\nERROR: {exc}\n")
        sys.exit(1)
