from __future__ import annotations

import argparse
import csv
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


GOEMOTIONS_LABELS = (
    "admiration",
    "amusement",
    "anger",
    "annoyance",
    "approval",
    "caring",
    "confusion",
    "curiosity",
    "desire",
    "disappointment",
    "disapproval",
    "disgust",
    "embarrassment",
    "excitement",
    "fear",
    "gratitude",
    "grief",
    "joy",
    "love",
    "nervousness",
    "optimism",
    "pride",
    "realization",
    "relief",
    "remorse",
    "sadness",
    "surprise",
    "neutral",
)


TARGET_MAPPING = {
    "angry": ("anger", "annoyance", "disapproval", "disgust"),
    "fearful": ("fear", "nervousness"),
    "happy": (
        "admiration",
        "amusement",
        "approval",
        "caring",
        "excitement",
        "gratitude",
        "joy",
        "love",
        "optimism",
        "pride",
        "relief",
        "surprise",
    ),
    "sad": ("disappointment", "embarrassment", "grief", "remorse", "sadness"),
    "neutral": ("neutral", "realization"),
}


_WHITESPACE_RE = re.compile(r"\s+")


@dataclass
class AggregatedExample:
    example_id: str
    text: str
    source_counts: Counter[str]
    total_raters: int
    unclear_votes: int


def _normalize_text(text: str) -> str:
    return _WHITESPACE_RE.sub(" ", str(text or "").strip()).lower()


def _iter_goemotions_rows(path: Path) -> Iterable[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            yield row


def aggregate_goemotions(path: Path) -> list[AggregatedExample]:
    grouped: dict[str, AggregatedExample] = {}
    for row in _iter_goemotions_rows(path):
        example_id = str(row.get("id", "")).strip()
        if not example_id:
            continue

        text = str(row.get("text", "")).strip()
        if not text:
            continue

        item = grouped.get(example_id)
        if item is None:
            item = AggregatedExample(
                example_id=example_id,
                text=text,
                source_counts=Counter(),
                total_raters=0,
                unclear_votes=0,
            )
            grouped[example_id] = item

        item.total_raters += 1
        if str(row.get("example_very_unclear", "")).strip().lower() == "true":
            item.unclear_votes += 1

        for label in GOEMOTIONS_LABELS:
            try:
                if int(str(row.get(label, "0")).strip() or "0") == 1:
                    item.source_counts[label] += 1
            except ValueError:
                continue

    return list(grouped.values())


def target_scores_from_source_counts(source_counts: Counter[str]) -> Counter[str]:
    scores = Counter()
    for target, source_labels in TARGET_MAPPING.items():
        scores[target] = sum(int(source_counts.get(label, 0)) for label in source_labels)
    return scores


def choose_target_label(
    scores: Counter[str],
    min_top_votes: int,
    min_margin: int,
) -> tuple[str | None, int, int]:
    ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    if not ranked or ranked[0][1] <= 0:
        return None, 0, 0

    top_label, top_votes = ranked[0]
    second_votes = ranked[1][1] if len(ranked) > 1 else 0
    if top_votes < min_top_votes:
        return None, top_votes, top_votes - second_votes
    if top_votes - second_votes < min_margin:
        return None, top_votes, top_votes - second_votes
    return top_label, top_votes, top_votes - second_votes


def convert_examples(
    examples: list[AggregatedExample],
    min_top_votes: int,
    min_margin: int,
    max_unclear_ratio: float,
    dedupe_by_text: bool,
) -> list[dict[str, object]]:
    converted: list[dict[str, object]] = []
    for item in examples:
        unclear_ratio = item.unclear_votes / max(item.total_raters, 1)
        if unclear_ratio > max_unclear_ratio:
            continue

        target_scores = target_scores_from_source_counts(item.source_counts)
        label, top_votes, margin = choose_target_label(
            scores=target_scores,
            min_top_votes=min_top_votes,
            min_margin=min_margin,
        )
        if label is None:
            continue

        converted.append(
            {
                "id": item.example_id,
                "text": item.text,
                "emotion": label,
                "target_votes": int(top_votes),
                "target_margin": int(margin),
                "total_raters": int(item.total_raters),
                "unclear_ratio": round(float(unclear_ratio), 4),
            }
        )

    if not dedupe_by_text:
        return converted

    best_by_text: dict[str, dict[str, object]] = {}
    for row in converted:
        key = _normalize_text(str(row["text"]))
        previous = best_by_text.get(key)
        if previous is None:
            best_by_text[key] = row
            continue

        candidate_rank = (
            int(row["target_votes"]),
            int(row["target_margin"]),
            int(row["total_raters"]),
            len(str(row["text"])),
        )
        previous_rank = (
            int(previous["target_votes"]),
            int(previous["target_margin"]),
            int(previous["total_raters"]),
            len(str(previous["text"])),
        )
        if candidate_rank > previous_rank:
            best_by_text[key] = row

    return list(best_by_text.values())


def maybe_balance_examples(
    rows: list[dict[str, object]],
    max_per_class: int | None,
    random_state: int,
) -> list[dict[str, object]]:
    if not max_per_class or max_per_class <= 0:
        return rows

    rng = random.Random(random_state)
    grouped: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["emotion"])].append(row)

    balanced: list[dict[str, object]] = []
    for emotion, items in grouped.items():
        items = list(items)
        rng.shuffle(items)
        balanced.extend(items[:max_per_class])
        print(f"[balance] {emotion}: kept {min(len(items), max_per_class)}/{len(items)}")

    rng.shuffle(balanced)
    return balanced


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "text",
        "emotion",
        "id",
        "target_votes",
        "target_margin",
        "total_raters",
        "unclear_ratio",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def build_metrics(rows: list[dict[str, object]], args: argparse.Namespace, source_path: Path) -> dict[str, object]:
    label_counts = Counter(str(row["emotion"]) for row in rows)
    return {
        "source_path": str(source_path),
        "output_path": str(args.output),
        "samples_total": int(len(rows)),
        "label_counts": dict(sorted(label_counts.items())),
        "min_top_votes": int(args.min_top_votes),
        "min_margin": int(args.min_margin),
        "max_unclear_ratio": float(args.max_unclear_ratio),
        "dedupe_by_text": bool(args.dedupe_by_text),
        "max_per_class": int(args.max_per_class) if args.max_per_class else None,
        "target_mapping": {key: list(values) for key, values in TARGET_MAPPING.items()},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert GoEmotions CSV to 5 emotion classes.")
    parser.add_argument("--input", type=str, default="data/goemotions_3.csv")
    parser.add_argument("--output", type=str, default="data/processed/goemotions_5class.csv")
    parser.add_argument("--metrics-out", type=str, default="data/processed/goemotions_5class_metrics.json")
    parser.add_argument("--min-top-votes", type=int, default=1)
    parser.add_argument("--min-margin", type=int, default=1)
    parser.add_argument("--max-unclear-ratio", type=float, default=0.5)
    parser.add_argument("--max-per-class", type=int, default=None)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--dedupe-by-text", action="store_true")
    args = parser.parse_args()

    source_path = Path(args.input)
    output_path = Path(args.output)
    metrics_path = Path(args.metrics_out)

    aggregated = aggregate_goemotions(source_path)
    print(f"[aggregate] unique examples: {len(aggregated)}")

    converted = convert_examples(
        examples=aggregated,
        min_top_votes=args.min_top_votes,
        min_margin=args.min_margin,
        max_unclear_ratio=args.max_unclear_ratio,
        dedupe_by_text=bool(args.dedupe_by_text),
    )
    print(f"[convert] kept after mapping/filtering: {len(converted)}")

    balanced = maybe_balance_examples(
        rows=converted,
        max_per_class=args.max_per_class,
        random_state=args.random_state,
    )

    label_counts = Counter(str(row["emotion"]) for row in balanced)
    print(f"[convert] final label counts: {dict(sorted(label_counts.items()))}")

    write_csv(output_path, balanced)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(
        json.dumps(build_metrics(balanced, args=args, source_path=source_path), indent=2),
        encoding="utf-8",
    )

    print(f"[done] dataset saved to: {output_path}")
    print(f"[done] metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
