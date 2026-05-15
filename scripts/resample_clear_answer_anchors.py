#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mcagent_boundary.io import read_json, read_jsonl, write_json, write_jsonl, write_jsonl_by_dataset


def _ranked(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        records,
        key=lambda item: (
            float(item.get("utility_gap", item.get("boundary_score", 0.0)) or 0.0),
            str(item.get("state_id", "")),
        ),
        reverse=True,
    )


def _sample(
    records: list[dict[str, Any]],
    *,
    total_limit: int,
    quota_by_dataset: dict[str, int],
    redistribute_unused: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        buckets.setdefault(str(record.get("dataset", "unknown")), []).append(record)
    for dataset, bucket in list(buckets.items()):
        buckets[dataset] = _ranked(bucket)

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    selected_by_dataset: Counter[str] = Counter()
    for dataset, limit in quota_by_dataset.items():
        for record in buckets.get(dataset, [])[: max(0, limit)]:
            selected.append(record)
            selected_ids.add(str(record.get("state_id")))
            selected_by_dataset[dataset] += 1

    if redistribute_unused and len(selected) < total_limit:
        for record in _ranked(records):
            if len(selected) >= total_limit:
                break
            state_id = str(record.get("state_id"))
            if state_id in selected_ids:
                continue
            selected.append(record)
            selected_ids.add(state_id)
            selected_by_dataset[str(record.get("dataset", "unknown"))] += 1

    if len(selected) > total_limit:
        selected = _ranked(selected)[:total_limit]
        selected_by_dataset = Counter(str(item.get("dataset", "unknown")) for item in selected)

    available_by_dataset = Counter(str(item.get("dataset", "unknown")) for item in records)
    shortages = {
        dataset: {
            "quota": target,
            "available": available_by_dataset.get(dataset, 0),
            "selected": selected_by_dataset.get(dataset, 0),
            "short_by": max(0, target - selected_by_dataset.get(dataset, 0)),
        }
        for dataset, target in quota_by_dataset.items()
        if selected_by_dataset.get(dataset, 0) < target
    }
    return selected, {
        "input": len(records),
        "output": len(selected),
        "quota_by_dataset": quota_by_dataset,
        "redistribute_unused": redistribute_unused,
        "available_by_dataset": dict(available_by_dataset),
        "selected_by_dataset": dict(selected_by_dataset),
        "shortages": shortages,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Downsample existing clear_answer_anchors.jsonl.")
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--total", type=int, default=740)
    parser.add_argument("--quota-json", default='{"gsm8k":20,"math":40,"in3":40,"mintqa":0,"or_bench":640}')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir
    quota = {str(key): int(value) for key, value in json.loads(args.quota_json).items()}
    clear_answer_path = run_dir / "clear_answer_anchors.jsonl"
    mined_path = run_dir / "mined_boundary.json"
    records = read_jsonl(clear_answer_path)
    sampled, report = _sample(records, total_limit=args.total, quota_by_dataset=quota)
    write_jsonl(clear_answer_path, sampled)
    write_jsonl_by_dataset(clear_answer_path, sampled)

    summary = read_json(mined_path)
    summary.setdefault("sampling_report", {})["clear_answer_anchors"] = report
    write_json(mined_path, summary)
    print(
        json.dumps(
            {
                "clear_answer_output": str(clear_answer_path),
                "selected": report,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
