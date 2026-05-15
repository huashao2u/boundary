from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


def _field(item: Any, name: str, default: Any = None) -> Any:
    if isinstance(item, dict):
        return item.get(name, default)
    return getattr(item, name, default)


def _metadata(item: Any) -> dict[str, Any]:
    value = _field(item, "metadata", {})
    return value if isinstance(value, dict) else {}


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    text = str(value)
    match = re.search(r"\d+", text)
    if match is None:
        return None
    return int(match.group(0))


def _take_first(examples: list[Any], limit: int | None) -> list[Any]:
    if limit is None:
        return list(examples)
    return list(examples[:limit])


def _math_level(item: Any) -> int | None:
    return _as_int(_metadata(item).get("math_level"))


def _take_math_v026(items: list[Any], total: int = 3000, easy_ratio: float = 0.8) -> list[Any]:
    easy_limit = int(total * easy_ratio)
    hard_limit = total - easy_limit
    easy_items = [item for item in items if (_math_level(item) or 99) <= 3]
    hard_items = [item for item in items if (_math_level(item) or 0) >= 4]
    selected = _take_first(easy_items, easy_limit) + _take_first(hard_items, hard_limit)
    if len(selected) < total:
        selected_ids = {id(item) for item in selected}
        remainder = [item for item in items if id(item) not in selected_ids]
        selected.extend(_take_first(remainder, total - len(selected)))
    return selected


def load_fixed_example_ids(path: str | Path) -> set[str]:
    """Load fixed example ids from txt, JSON list, or JSONL records."""
    input_path = Path(path)
    text = input_path.read_text(encoding="utf-8").strip()
    if not text:
        return set()
    if text.startswith("["):
        payload = json.loads(text)
        return {str(item) for item in payload}
    ids: set[str] = set()
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("{"):
            payload = json.loads(stripped)
            value = payload.get("example_id") or payload.get("id")
        else:
            value = stripped.split()[0]
        if value:
            ids.add(str(value))
    return ids


def filter_by_example_ids(examples: list[Any], example_ids: set[str]) -> list[Any]:
    if not example_ids:
        return list(examples)
    return [example for example in examples if str(_field(example, "example_id", "")) in example_ids]


def apply_selection_preset(examples: list[Any], preset: str | None) -> list[Any]:
    if not preset or preset == "none":
        return list(examples)
    if preset not in {"v023_full_rollout", "v026_full_rollout"}:
        raise ValueError(f"Unknown dataset selection preset: {preset}")

    by_dataset: dict[str, list[Any]] = {}
    dataset_order: list[str] = []
    for example in examples:
        dataset = str(getattr(example, "dataset", ""))
        if dataset not in by_dataset:
            by_dataset[dataset] = []
            dataset_order.append(dataset)
        by_dataset[dataset].append(example)

    selected_by_dataset: dict[str, list[Any]] = {}
    for dataset, items in by_dataset.items():
        if dataset == "gsm8k":
            selected_by_dataset[dataset] = _take_first(items, 3000)
        elif dataset == "math":
            if preset == "v026_full_rollout":
                selected_by_dataset[dataset] = _take_math_v026(items, total=3000, easy_ratio=0.8)
            else:
                easy_items = [
                    item
                    for item in items
                    if (_math_level(item) or 99) <= 3
                ]
                selected_by_dataset[dataset] = _take_first(easy_items, 3000)
        elif dataset == "or_bench":
            benign: list[Any] = []
            hard: list[Any] = []
            toxic: list[Any] = []
            other: list[Any] = []
            for item in items:
                label = str(_metadata(item).get("or_bench_label", "")).lower()
                if label == "benign":
                    benign.append(item)
                elif label == "hard":
                    hard.append(item)
                elif label == "toxic":
                    toxic.append(item)
                else:
                    other.append(item)
            selected_by_dataset[dataset] = _take_first(benign, 4000) + hard + toxic + other
        elif dataset in {"mintqa", "in3"}:
            selected_by_dataset[dataset] = list(items)
        else:
            selected_by_dataset[dataset] = list(items)

    selected: list[Any] = []
    for dataset in dataset_order:
        selected.extend(selected_by_dataset.get(dataset, []))
    return selected


def selection_summary(examples: list[Any]) -> dict[str, Any]:
    by_dataset = Counter(str(_field(example, "dataset", "")) for example in examples)
    by_or_bench_label = Counter(
        str(_metadata(example).get("or_bench_label", ""))
        for example in examples
        if str(_field(example, "dataset", "")) == "or_bench"
    )
    math_levels = Counter(
        str(_metadata(example).get("math_level", "unknown"))
        for example in examples
        if str(_field(example, "dataset", "")) == "math"
    )
    return {
        "total": len(examples),
        "by_dataset": dict(by_dataset),
        "or_bench_by_label": dict(by_or_bench_label),
        "math_levels": dict(math_levels),
    }
