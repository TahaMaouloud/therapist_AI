from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset as TorchDataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

# Fine-tuning pipeline:
# CSV -> preprocessing -> train/validation/test split -> tokenization -> Trainer -> save best model + metrics.

_PREDICTOR_CACHE: dict[str, tuple[Any, AutoModelForSequenceClassification, torch.device]] = {}


def load_dataset(path: Path, text_col: str, label_col: str) -> pd.DataFrame:
    if path.suffix.lower() != ".csv":
        raise ValueError("Unsupported file format. Expected a .csv file.")

    df = pd.read_csv(path)
    if text_col not in df.columns or label_col not in df.columns:
        raise ValueError(
            f"Columns not found. Expected text='{text_col}', label='{label_col}'. "
            f"Available: {list(df.columns)}"
        )

    df = df[[text_col, label_col]].dropna().copy()
    df[text_col] = df[text_col].astype(str).str.strip()
    df[label_col] = df[label_col].astype(str).str.strip()
    df = df[(df[text_col] != "") & (df[label_col] != "")]
    if df.empty:
        raise ValueError("Dataset is empty after preprocessing.")

    return df.rename(columns={text_col: "text", label_col: "label"})


def build_label_mappings(df: pd.DataFrame) -> tuple[list[str], dict[str, int], dict[int, str]]:
    labels = sorted(df["label"].unique().tolist())
    if len(labels) < 2:
        raise ValueError("Need at least 2 unique labels for classification.")

    label2id = {label: idx for idx, label in enumerate(labels)}
    id2label = {idx: label for label, idx in label2id.items()}
    return labels, label2id, id2label


def _validate_split_size(df: pd.DataFrame, split_size: float, split_name: str) -> None:
    class_count = int(df["label"].nunique())
    split_count = int(round(len(df) * split_size))
    if split_count < class_count:
        min_split_size = class_count / max(len(df), 1)
        raise ValueError(
            f"{split_name} is too small for stratified split. "
            f"Need at least {class_count} samples for {class_count} classes. "
            f"Use {split_name} >= {min_split_size:.3f} or add more rows."
        )


def split_dataset(
    df: pd.DataFrame,
    test_size: float,
    validation_size: float,
    random_state: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be between 0 and 1.")
    if not 0.0 < validation_size < 1.0:
        raise ValueError("validation_size must be between 0 and 1.")
    if test_size + validation_size >= 1.0:
        raise ValueError("test_size + validation_size must be < 1.")

    _validate_split_size(df, test_size, "test_size")
    _validate_split_size(df, validation_size, "validation_size")

    train_val_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["label"],
    )
    validation_share = validation_size / (1.0 - test_size)
    _validate_split_size(train_val_df, validation_share, "validation_size")

    train_df, val_df = train_test_split(
        train_val_df,
        test_size=validation_share,
        random_state=random_state,
        stratify=train_val_df["label"],
    )
    return (
        train_df.reset_index(drop=True),
        val_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


class EmotionDataset(TorchDataset):
    def __init__(self, texts: list[str], labels: list[int], tokenizer: Any, max_length: int) -> None:
        self.encodings = tokenizer(
            texts,
            truncation=True,
            padding=False,
            max_length=max_length,
        )
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = {key: torch.tensor(value[idx], dtype=torch.long) for key, value in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


def to_torch_dataset(df: pd.DataFrame, label2id: dict[str, int], tokenizer: Any, max_length: int) -> EmotionDataset:
    mapped = df["label"].map(label2id)
    if mapped.isna().any():
        raise ValueError("Some labels are missing from label2id mapping.")
    labels = mapped.astype(int).tolist()
    texts = df["text"].tolist()
    return EmotionDataset(texts=texts, labels=labels, tokenizer=tokenizer, max_length=max_length)


def _checkpoint_step(path: Path) -> int:
    prefix = "checkpoint-"
    if not path.is_dir() or not path.name.startswith(prefix):
        return -1
    try:
        return int(path.name[len(prefix) :])
    except ValueError:
        return -1


def find_latest_checkpoint(model_out_dir: Path) -> Path | None:
    checkpoints = [path for path in model_out_dir.glob("checkpoint-*") if path.is_dir()]
    if not checkpoints:
        return None
    return max(checkpoints, key=_checkpoint_step)


def _is_windows_mapped_file_error(exc: Exception) -> bool:
    error_text = f"{type(exc).__name__}: {exc}".lower()
    return (
        "os error 1224" in error_text
        or "section mapp" in error_text
        or "mapped section" in error_text
        or ("safetensor" in error_text and "i/o error" in error_text)
    )


def resolve_model_sources(
    model_name: str,
    model_out_dir: Path,
    resume_from_checkpoint: str | None,
) -> tuple[str, str | None, str]:
    if not resume_from_checkpoint:
        return model_name, None, "fresh"

    checkpoint_path = (
        find_latest_checkpoint(model_out_dir)
        if resume_from_checkpoint.strip().lower() == "auto"
        else Path(resume_from_checkpoint)
    )
    if checkpoint_path is None or not checkpoint_path.exists():
        raise ValueError("Checkpoint not found. Provide a valid path or use --resume-from-checkpoint auto.")

    checkpoint_path = checkpoint_path.resolve()
    trainer_state_path = checkpoint_path / "trainer_state.json"
    if trainer_state_path.exists():
        return str(checkpoint_path), str(checkpoint_path), "resume"
    return str(checkpoint_path), None, "warm_start"


def resolve_tokenizer_source(model_source: str, fallback_model_name: str) -> str:
    source_path = Path(model_source)
    if source_path.exists():
        has_tokenizer_files = any(
            (source_path / name).exists()
            for name in ("tokenizer.json", "tokenizer_config.json", "vocab.txt", "special_tokens_map.json")
        )
        if has_tokenizer_files:
            return str(source_path)
    return fallback_model_name


def build_compute_metrics() -> Any:
    def _compute_metrics(eval_pred: Any) -> dict[str, float]:
        logits = eval_pred.predictions if hasattr(eval_pred, "predictions") else eval_pred[0]
        labels = eval_pred.label_ids if hasattr(eval_pred, "label_ids") else eval_pred[1]
        pred_ids = np.argmax(logits, axis=-1)
        return {
            "accuracy": float(accuracy_score(labels, pred_ids)),
            "macro_f1": float(f1_score(labels, pred_ids, average="macro")),
            "weighted_f1": float(f1_score(labels, pred_ids, average="weighted")),
        }

    return _compute_metrics


def _estimate_warmup_steps(train_size: int, args: argparse.Namespace) -> int:
    steps_per_epoch = max(1, math.ceil(train_size / max(args.train_batch_size, 1)))
    total_steps = max(1, math.ceil(steps_per_epoch * args.num_train_epochs))
    return max(0, int(total_steps * args.warmup_ratio))


def _extract_eval_metrics(raw_metrics: dict[str, Any], prefix: str) -> dict[str, float]:
    return {
        "accuracy": float(raw_metrics.get(f"{prefix}_accuracy", 0.0)),
        "macro_f1": float(raw_metrics.get(f"{prefix}_macro_f1", 0.0)),
        "weighted_f1": float(raw_metrics.get(f"{prefix}_weighted_f1", 0.0)),
        "eval_loss": float(raw_metrics.get(f"{prefix}_loss", 0.0)),
    }


def train_and_evaluate(
    train_ds: EmotionDataset,
    val_ds: EmotionDataset,
    test_ds: EmotionDataset,
    tokenizer: Any,
    model_source: str,
    model_out_dir: Path,
    label2id: dict[str, int],
    id2label: dict[int, str],
    resume_from_checkpoint: str | None,
    args: argparse.Namespace,
) -> dict[str, Any]:
    try:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_source,
            num_labels=len(label2id),
            label2id=label2id,
            id2label=id2label,
            local_files_only=bool(args.local_files_only),
        )
    except OSError as exc:
        raise RuntimeError(
            "Unable to load base model weights. "
            "If you are offline, use a local model directory or run once with internet access."
        ) from exc

    training_args = TrainingArguments(
        output_dir=str(model_out_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        save_only_model=True,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.train_batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        num_train_epochs=args.num_train_epochs,
        weight_decay=args.weight_decay,
        warmup_steps=_estimate_warmup_steps(len(train_ds), args),
        load_best_model_at_end=True,
        metric_for_best_model="eval_macro_f1",
        greater_is_better=True,
        save_total_limit=2,
        report_to="none",
        seed=args.random_state,
        fp16=bool(args.fp16 and torch.cuda.is_available()),
    )

    callbacks: list[Any] = []
    if args.early_stopping_patience > 0:
        callbacks.append(EarlyStoppingCallback(early_stopping_patience=args.early_stopping_patience))

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=build_compute_metrics(),
        callbacks=callbacks,
    )

    if not args.skip_train:
        if resume_from_checkpoint:
            trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        else:
            trainer.train()
    saved_model_dir = model_out_dir
    try:
        trainer.save_model(str(model_out_dir))
        tokenizer.save_pretrained(str(model_out_dir))
    except Exception as exc:
        if not _is_windows_mapped_file_error(exc):
            raise

        latest_checkpoint = find_latest_checkpoint(model_out_dir)
        if latest_checkpoint is None:
            raise RuntimeError(
                "Model save failed due to a Windows file lock and no checkpoint was found. "
                "Close any running app that uses this model, then retry with a new --model-out-dir."
            ) from exc

        saved_model_dir = latest_checkpoint
        print(
            "[warning] Final model save failed because the target file is locked. "
            f"Using latest checkpoint as exported model: {saved_model_dir}"
        )

    validation_metrics_raw = trainer.evaluate(eval_dataset=val_ds, metric_key_prefix="validation")
    test_metrics_raw = trainer.evaluate(eval_dataset=test_ds, metric_key_prefix="test")
    predictions = trainer.predict(test_ds)
    pred_ids = np.argmax(predictions.predictions, axis=-1)
    true_ids = predictions.label_ids

    ordered_labels = [id2label[idx] for idx in range(len(id2label))]
    report = classification_report(
        true_ids,
        pred_ids,
        labels=list(range(len(ordered_labels))),
        target_names=ordered_labels,
        output_dict=True,
        zero_division=0,
    )

    validation_metrics = _extract_eval_metrics(validation_metrics_raw, prefix="validation")
    test_metrics = _extract_eval_metrics(test_metrics_raw, prefix="test")

    return {
        **test_metrics,
        "labels": ordered_labels,
        "saved_model_dir": str(saved_model_dir),
        "validation": validation_metrics,
        "best_checkpoint": str(trainer.state.best_model_checkpoint) if trainer.state.best_model_checkpoint else None,
        "best_validation_macro_f1": (
            float(trainer.state.best_metric) if trainer.state.best_metric is not None else validation_metrics["macro_f1"]
        ),
        "classification_report": report,
    }


def _load_predictor_bundle(
    model_dir: str | Path,
    device: str | None = None,
) -> tuple[Any, AutoModelForSequenceClassification, torch.device]:
    key = str(model_dir)
    if key in _PREDICTOR_CACHE:
        return _PREDICTOR_CACHE[key]

    resolved_device = torch.device(device) if device else torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(resolved_device)
    model.eval()

    bundle = (tokenizer, model, resolved_device)
    _PREDICTOR_CACHE[key] = bundle
    return bundle


@torch.inference_mode()
def predict_emotion(
    text: str,
    model_dir: str | Path = "models/emotion_bert",
    max_length: int = 256,
    device: str | None = None,
) -> str:
    tokenizer, model, model_device = _load_predictor_bundle(model_dir=model_dir, device=device)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    inputs = {key: value.to(model_device) for key, value in inputs.items()}
    logits = model(**inputs).logits
    pred_id = int(torch.argmax(logits, dim=-1).item())

    label = model.config.id2label.get(pred_id) if model.config.id2label else None
    if label is None and model.config.id2label:
        label = model.config.id2label.get(str(pred_id))
    return str(label) if label is not None else str(pred_id)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune bert-base-uncased for emotion classification")
    parser.add_argument("--data", type=str, required=True, help="Path to CSV dataset")
    parser.add_argument("--text-col", type=str, default="text")
    parser.add_argument("--label-col", type=str, default="label")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--validation-size", type=float, default=0.1)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--model-name", type=str, default="bert-base-uncased")
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--num-train-epochs", type=float, default=4.0)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument("--logging-steps", type=int, default=50)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--resume-from-checkpoint",
        type=str,
        default=None,
        help="Checkpoint directory to resume/warm-start from, or 'auto' for the latest checkpoint in model-out-dir.",
    )
    parser.add_argument("--skip-train", action="store_true", help="Only export/evaluate the loaded model.")
    parser.add_argument("--model-out-dir", type=str, default="models/emotion_bert")
    parser.add_argument("--metrics-out", type=str, default="models/emotion_metrics.json")
    args = parser.parse_args()

    data_path = Path(args.data)
    model_out_dir = Path(args.model_out_dir)
    metrics_out = Path(args.metrics_out)

    model_out_dir.mkdir(parents=True, exist_ok=True)
    metrics_out.parent.mkdir(parents=True, exist_ok=True)

    set_seed(args.random_state)
    df = load_dataset(data_path, args.text_col, args.label_col)
    labels, label2id, id2label = build_label_mappings(df)
    train_df, val_df, test_df = split_dataset(
        df,
        test_size=args.test_size,
        validation_size=args.validation_size,
        random_state=args.random_state,
    )
    model_source, resume_checkpoint, training_mode = resolve_model_sources(
        model_name=args.model_name,
        model_out_dir=model_out_dir,
        resume_from_checkpoint=args.resume_from_checkpoint,
    )
    tokenizer_source = resolve_tokenizer_source(model_source=model_source, fallback_model_name=args.model_name)

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_source,
            local_files_only=bool(args.local_files_only),
        )
    except OSError as exc:
        raise RuntimeError(
            "Unable to load tokenizer. "
            "If you are offline, use a local model directory or run once with internet access."
        ) from exc
    train_ds = to_torch_dataset(train_df, label2id, tokenizer=tokenizer, max_length=args.max_length)
    val_ds = to_torch_dataset(val_df, label2id, tokenizer=tokenizer, max_length=args.max_length)
    test_ds = to_torch_dataset(test_df, label2id, tokenizer=tokenizer, max_length=args.max_length)

    eval_metrics = train_and_evaluate(
        train_ds=train_ds,
        val_ds=val_ds,
        test_ds=test_ds,
        tokenizer=tokenizer,
        model_source=model_source,
        model_out_dir=model_out_dir,
        label2id=label2id,
        id2label=id2label,
        resume_from_checkpoint=resume_checkpoint,
        args=args,
    )

    metrics = {
        "data_path": str(data_path),
        "text_col": args.text_col,
        "label_col": args.label_col,
        "samples_total": int(len(df)),
        "samples_train": int(len(train_df)),
        "samples_validation": int(len(val_df)),
        "samples_test": int(len(test_df)),
        "model_name": args.model_name,
        "model_source": model_source,
        "model_out_dir": str(model_out_dir),
        "label2id": label2id,
        "id2label": {str(key): value for key, value in id2label.items()},
        "labels": labels,
        "training": {
            "num_train_epochs": args.num_train_epochs,
            "train_batch_size": args.train_batch_size,
            "eval_batch_size": args.eval_batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "warmup_ratio": args.warmup_ratio,
            "max_length": args.max_length,
            "test_size": args.test_size,
            "validation_size": args.validation_size,
            "random_state": args.random_state,
            "local_files_only": bool(args.local_files_only),
            "early_stopping_patience": args.early_stopping_patience,
            "resume_from_checkpoint": args.resume_from_checkpoint,
            "resume_mode": training_mode,
            "skip_train": bool(args.skip_train),
        },
        "eval": eval_metrics,
    }
    metrics_out.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(
        "Training done. "
        f"Accuracy={eval_metrics['accuracy']:.4f}, "
        f"Macro-F1={eval_metrics['macro_f1']:.4f}, "
        f"Val-Macro-F1={eval_metrics['validation']['macro_f1']:.4f}"
    )
    effective_model_dir = eval_metrics.get("saved_model_dir", str(model_out_dir))
    print(f"Model saved to: {effective_model_dir}")
    print(f"Metrics saved to: {metrics_out}")


if __name__ == "__main__":
    main()
