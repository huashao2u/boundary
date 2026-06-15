from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mcagent_boundary.config import load_boundary_config, resolve_repo_path
from mcagent_boundary.adapters import build_adapter_registry
from mcagent_boundary.adapters.base import StandardizedExample
from mcagent_boundary.io import write_json, write_jsonl


def _iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _example_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    ids: set[str] = set()
    for record in _iter_jsonl(path):
        if record.get("example_id") is not None:
            ids.add(str(record.get("example_id")))
    return ids


def _scan_pair_files(paths: list[Path]) -> dict[str, Any]:
    pair_ids: set[str] = set()
    rows_by_dataset: Counter[str] = Counter()
    unique_by_dataset: dict[str, set[str]] = defaultdict(set)
    or_rows_by_label: Counter[str] = Counter()
    or_unique_by_label: dict[str, set[str]] = defaultdict(set)
    for path in paths:
        if not path.exists():
            continue
        for record in _iter_jsonl(path):
            example_id = record.get("example_id")
            if example_id is None:
                continue
            example_id = str(example_id)
            pair_ids.add(example_id)
            dataset = str(record.get("dataset"))
            rows_by_dataset[dataset] += 1
            unique_by_dataset[dataset].add(example_id)
            if dataset == "or_bench":
                label = _or_label(example_id, record.get("metadata") or {})
                or_rows_by_label[label] += 1
                or_unique_by_label[label].add(example_id)
    return {
        "pair_ids": pair_ids,
        "rows_by_dataset": rows_by_dataset,
        "unique_by_dataset": unique_by_dataset,
        "or_rows_by_label": or_rows_by_label,
        "or_unique_by_label": or_unique_by_label,
    }


def _or_label(example_id: str, metadata: dict[str, Any] | None = None) -> str:
    metadata = metadata or {}
    label = metadata.get("or_bench_label")
    if label:
        return str(label)
    prefix = "or_bench-"
    if example_id.startswith(prefix):
        rest = example_id[len(prefix) :]
        return rest.split("-", 1)[0]
    return "unknown"


def _as_record(example: Any, *, test_source: str) -> dict[str, Any]:
    record = example.to_record()
    metadata = dict(record.get("metadata") or {})
    metadata["test_source"] = test_source
    record["metadata"] = metadata
    return record


def _sample(items: list[Any], n: int, rng: random.Random) -> list[Any]:
    if len(items) <= n:
        return list(items)
    return rng.sample(items, n)


def _math_level_label(item: Any) -> str:
    metadata = item.metadata if hasattr(item, "metadata") else {}
    text = str(metadata.get("math_level", "unknown"))
    for value in range(1, 6):
        if str(value) in text:
            return f"Level {value}"
    return "unknown"


def _sample_math_by_level(items: list[Any], n: int, rng: random.Random) -> tuple[list[Any], dict[str, Any]]:
    by_level: dict[str, list[Any]] = defaultdict(list)
    for item in items:
        by_level[_math_level_label(item)].append(item)
    target_levels = [f"Level {value}" for value in range(1, 6)]
    base = n // len(target_levels)
    remainder = n - base * len(target_levels)
    selected: list[Any] = []
    per_level: dict[str, int] = {}
    shortfall = 0
    for index, level in enumerate(target_levels):
        quota = base + (1 if index < remainder else 0)
        taken = _sample(by_level[level], quota, rng)
        selected.extend(taken)
        per_level[level] = len(taken)
        shortfall += max(0, quota - len(taken))
    if shortfall:
        selected_ids = {item.example_id for item in selected}
        fallback_pool = [
            item
            for level in target_levels
            for item in by_level[level]
            if item.example_id not in selected_ids
        ]
        fill = _sample(fallback_pool, shortfall, rng)
        selected.extend(fill)
        for item in fill:
            level = _math_level_label(item)
            per_level[level] = per_level.get(level, 0) + 1
    return selected[:n], {
        "levels_available": {level: len(values) for level, values in sorted(by_level.items())},
        "levels_selected": per_level,
    }


def _load_examples(config: dict[str, Any], dataset_name: str, *, split: str) -> list[Any]:
    registry = build_adapter_registry()
    dataset_root = resolve_repo_path(config["paths"]["dataset_root"], config)
    return registry[dataset_name].load(dataset_root=dataset_root, limit=None, split=split)


def _load_mintqa_sample(config: dict[str, Any], *, split: str, n: int, rng: random.Random) -> list[StandardizedExample]:
    import pyarrow.parquet as pq

    dataset_root = resolve_repo_path(config["paths"]["dataset_root"], config)
    shards = sorted((dataset_root / "MintQA-Ti-v0.1" / "data").glob(f"{split}-*.parquet"))
    offsets: list[tuple[Path, int, int]] = []
    cursor = 0
    for shard in shards:
        num_rows = pq.ParquetFile(shard).metadata.num_rows
        offsets.append((shard, cursor, num_rows))
        cursor += num_rows
    selected_global = sorted(rng.sample(range(cursor), min(n, cursor))) if cursor > n else list(range(cursor))
    by_shard: dict[Path, list[tuple[int, int]]] = defaultdict(list)
    shard_index = 0
    for global_index in selected_global:
        while shard_index + 1 < len(offsets) and global_index >= offsets[shard_index][1] + offsets[shard_index][2]:
            shard_index += 1
        shard, start, _num_rows = offsets[shard_index]
        by_shard[shard].append((global_index, global_index - start))

    examples: list[StandardizedExample] = []
    columns = ["id", "question", "answer", "q_entity", "a_entity"]
    for shard, wanted in by_shard.items():
        rows = pq.read_table(shard, columns=columns).to_pylist()
        for global_index, local_index in wanted:
            row = rows[local_index]
            metadata = {key: value for key, value in row.items() if key not in {"question", "answer"}}
            examples.append(
                StandardizedExample(
                    example_id=f"mintqa-{split}-{global_index}",
                    dataset="mintqa",
                    split=split,
                    question=str(row.get("question")),
                    gold_answer=row.get("answer"),
                    metadata={
                        "split": split,
                        **metadata,
                        "boundary_type": "factual",
                        "task_type": "factual_boundary",
                        "can_search": True,
                        "can_calculate": False,
                        "can_clarify": False,
                        "allow_refuse": True,
                        "legacy_dataset": "mintqa",
                        "legacy_id": f"mintqa-{split}-{global_index}",
                    },
                )
            )
    return examples


def main() -> None:
    parser = argparse.ArgumentParser(description="Build held-out test examples for the latest DPO pair set.")
    parser.add_argument("--artifact-dir", required=True, help="Directory containing all_rollouts and final DPO pairs.")
    parser.add_argument("--output-dir", default=None, help="Output directory. Defaults to ARTIFACT/test_set.")
    parser.add_argument("--seed", type=int, default=20260517)
    parser.add_argument("--gsm8k-n", type=int, default=500)
    parser.add_argument("--mintqa-n", type=int, default=500)
    parser.add_argument("--commonsenseqa-n", type=int, default=500)
    parser.add_argument("--math-n", type=int, default=500)
    parser.add_argument("--or-bench-n", type=int, default=800)
    parser.add_argument("--or-benign-n", type=int, default=500)
    parser.add_argument("--or-hard-n", type=int, default=250)
    parser.add_argument("--or-toxic-n", type=int, default=50)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    config = load_boundary_config()
    artifact_dir = resolve_repo_path(args.artifact_dir, config)
    output_dir = resolve_repo_path(args.output_dir, config) if args.output_dir else artifact_dir / "test_set"
    output_dir.mkdir(parents=True, exist_ok=True)

    pair_paths = [artifact_dir / "train_step_dpo_pairs.jsonl", artifact_dir / "eval_step_dpo_pairs.jsonl"]
    pair_scan = _scan_pair_files(pair_paths)
    pair_ids = pair_scan["pair_ids"]
    rollout_ids = _example_ids(artifact_dir / "all_rollouts.jsonl")

    gsm8k_test = _load_examples(config, "gsm8k", split="test")
    mintqa_test = _load_mintqa_sample(config, split="test", n=args.mintqa_n, rng=rng)
    commonsenseqa_test = _load_examples(config, "commonsenseqa", split="validation")
    in3_test = _load_examples(config, "in3", split="test")
    math_all = _load_examples(config, "math", split="train")
    or_all = _load_examples(config, "or_bench", split="train")

    gsm8k_selected = _sample(gsm8k_test, args.gsm8k_n, rng)
    mintqa_selected = list(mintqa_test)
    commonsenseqa_pool = [
        item for item in commonsenseqa_test if item.example_id not in rollout_ids and item.example_id not in pair_ids
    ]
    commonsenseqa_selected = _sample(commonsenseqa_pool, args.commonsenseqa_n, rng)
    in3_selected = list(in3_test)
    math_pool = [item for item in math_all if item.example_id not in rollout_ids and item.example_id not in pair_ids]
    math_selected, math_summary = _sample_math_by_level(math_pool, args.math_n, rng)

    or_by_label: dict[str, list[Any]] = defaultdict(list)
    for item in or_all:
        or_by_label[_or_label(item.example_id, item.metadata)].append(item)
    or_selected: list[Any] = []
    or_summary: dict[str, Any] = {"quota": {"benign": args.or_benign_n, "hard": args.or_hard_n, "toxic": args.or_toxic_n}}
    for label, quota in [("hard", args.or_hard_n), ("toxic", args.or_toxic_n)]:
        pool = [item for item in or_by_label[label] if item.example_id not in pair_ids]
        taken = _sample(pool, quota, rng)
        or_selected.extend(taken)
        or_summary[f"{label}_available_not_in_pairs"] = len(pool)
        or_summary[f"{label}_selected"] = len(taken)
    benign_pool = [item for item in or_by_label["benign"] if item.example_id not in rollout_ids and item.example_id not in pair_ids]
    benign_quota = max(0, args.or_bench_n - len(or_selected))
    benign_quota = min(args.or_benign_n, benign_quota)
    benign_selected = _sample(benign_pool, benign_quota, rng)
    or_selected.extend(benign_selected)
    or_summary["benign_available_not_in_rollout"] = len(benign_pool)
    or_summary["benign_selected"] = len(benign_selected)
    if len(or_selected) < args.or_bench_n:
        selected_ids = {item.example_id for item in or_selected}
        fallback_pool = [
            item
            for item in or_all
            if item.example_id not in selected_ids and item.example_id not in pair_ids and item.example_id not in rollout_ids
        ]
        fallback = _sample(fallback_pool, args.or_bench_n - len(or_selected), rng)
        or_selected.extend(fallback)
        or_summary["fallback_selected"] = len(fallback)

    records_by_dataset = {
        "gsm8k": [_as_record(item, test_source="official_test") for item in gsm8k_selected],
        "mintqa": [_as_record(item, test_source="official_test") for item in mintqa_selected],
        "commonsenseqa": [
            _as_record(item, test_source="validation_not_in_rollout_or_pairs") for item in commonsenseqa_selected
        ],
        "in3": [_as_record(item, test_source="official_test_all") for item in in3_selected],
        "math": [_as_record(item, test_source="train_not_in_rollout_level_balanced") for item in math_selected],
        "or_bench": [
            _as_record(
                item,
                test_source=(
                    "not_in_final_pairs_hard_toxic"
                    if _or_label(item.example_id, item.metadata) in {"hard", "toxic"}
                    else "benign_not_in_rollout"
                ),
            )
            for item in or_selected[: args.or_bench_n]
        ],
    }
    all_records = [
        record
        for dataset in ("gsm8k", "mintqa", "commonsenseqa", "in3", "math", "or_bench")
        for record in records_by_dataset[dataset]
    ]

    write_jsonl(output_dir / "pair_test_examples.jsonl", all_records)
    for dataset, records in records_by_dataset.items():
        write_jsonl(output_dir / f"{dataset}_test_examples.jsonl", records)

    summary = {
        "artifact_dir": str(artifact_dir),
        "output_dir": str(output_dir),
        "seed": args.seed,
        "total": len(all_records),
        "by_dataset": {dataset: len(records) for dataset, records in records_by_dataset.items()},
        "final_pair_rows_by_dataset": dict(pair_scan["rows_by_dataset"]),
        "final_pair_unique_examples_by_dataset": {
            dataset: len(ids) for dataset, ids in pair_scan["unique_by_dataset"].items()
        },
        "or_bench_final_pair_rows_by_label": dict(pair_scan["or_rows_by_label"]),
        "or_bench_final_pair_unique_examples_by_label": {
            label: len(ids) for label, ids in pair_scan["or_unique_by_label"].items()
        },
        "or_bench_selected_by_label": dict(
            Counter(_or_label(record["example_id"], record.get("metadata") or {}) for record in records_by_dataset["or_bench"])
        ),
        "math": math_summary,
        "or_bench": or_summary,
        "overlap_with_final_pairs": len({record["example_id"] for record in all_records} & pair_ids),
        "overlap_with_rollout": len({record["example_id"] for record in all_records} & rollout_ids),
    }
    write_json(output_dir / "pair_test_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
