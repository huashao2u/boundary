from __future__ import annotations

from typing import Any


def _safe_rate(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def evaluate_task_metrics(rollouts: list[dict[str, Any]]) -> dict[str, Any]:
    per_dataset: dict[str, dict[str, Any]] = {}
    total_utility = 0.0
    total_records = len(rollouts)
    all_correct = 0
    all_correct_denom = 0
    clarify_resolved = 0
    clarify_total = 0
    for record in rollouts:
        dataset = record["dataset"]
        natural_branch = record.get("natural_branch") or {}
        per_dataset.setdefault(dataset, {"correct": 0, "correct_denom": 0, "utility_sum": 0.0, "count": 0})
        correctness = natural_branch.get("correctness")
        if correctness is not None:
            per_dataset[dataset]["correct_denom"] += 1
            all_correct_denom += 1
            if correctness:
                per_dataset[dataset]["correct"] += 1
                all_correct += 1
        if record["boundary_type"] == "intention" and record["natural_action"] == "CLARIFY":
            clarify_total += 1
            clarify_resolved += int(natural_branch.get("final_status") == "answered_after_clarify")
        utility = float(natural_branch.get("utility", 0.0))
        per_dataset[dataset]["utility_sum"] += utility
        per_dataset[dataset]["count"] += 1
        total_utility += utility

    dataset_metrics = {
        dataset: {
            "exact_match": _safe_rate(block["correct"], block["correct_denom"]),
            "expected_utility": _safe_rate(block["utility_sum"], block["count"]),
            "num_samples": block["count"],
        }
        for dataset, block in per_dataset.items()
    }
    return {
        "overall_exact_match": _safe_rate(all_correct, all_correct_denom),
        "expected_utility": _safe_rate(total_utility, total_records),
        "clarify_resolution_rate": _safe_rate(clarify_resolved, clarify_total),
        "num_samples": total_records,
        "per_dataset": dataset_metrics,
    }

