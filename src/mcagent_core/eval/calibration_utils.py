from __future__ import annotations

from collections import defaultdict
from typing import Any


def _safe_div(numerator: float, denominator: float) -> float | None:
    return None if denominator == 0 else numerator / denominator


def binary_auroc(labels: list[int], scores: list[float]) -> float | None:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None
    ranked = sorted(zip(scores, labels), key=lambda item: item[0])
    rank_sum = 0.0
    for rank, (_, label) in enumerate(ranked, start=1):
        if label:
            rank_sum += rank
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def expected_calibration_error(labels: list[int], scores: list[float], num_bins: int = 10) -> float | None:
    if not labels:
        return None
    bins = defaultdict(list)
    for label, score in zip(labels, scores):
        clipped = min(max(float(score), 0.0), 1.0)
        index = min(num_bins - 1, int(clipped * num_bins))
        bins[index].append((label, clipped))
    total = len(labels)
    ece = 0.0
    for bucket in bins.values():
        avg_label = sum(label for label, _ in bucket) / len(bucket)
        avg_score = sum(score for _, score in bucket) / len(bucket)
        ece += abs(avg_label - avg_score) * (len(bucket) / total)
    return ece


def brier_score(labels: list[int], scores: list[float]) -> float | None:
    if not labels:
        return None
    return sum((float(score) - float(label)) ** 2 for label, score in zip(labels, scores)) / len(labels)


def summarize_risk_coverage(labels: list[int], scores: list[float]) -> dict[str, float | None]:
    if not labels:
        return {"best_coverage_at_90_precision": None, "utility_coverage": None}
    ranked = sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)
    covered = 0
    correct = 0
    coverage_at_precision = None
    for index, (_, label) in enumerate(ranked, start=1):
        covered += 1
        correct += int(label)
        precision = correct / covered
        if coverage_at_precision is None and precision >= 0.9:
            coverage_at_precision = covered / len(ranked)
    utility_coverage = sum(labels) / len(labels)
    return {
        "best_coverage_at_90_precision": coverage_at_precision,
        "utility_coverage": utility_coverage,
    }


def extract_binary_action_targets(records: list[dict[str, Any]]) -> tuple[list[int], list[float]]:
    labels: list[int] = []
    scores: list[float] = []
    for record in records:
        score = record.get("action_confidence")
        oracle = record.get("oracle_action")
        action = (record.get("decision") or {}).get("action")
        if score is None or action is None or oracle is None:
            continue
        labels.append(int(action == oracle))
        scores.append(float(score))
    return labels, scores


def extract_binary_answer_targets(records: list[dict[str, Any]]) -> tuple[list[int], list[float]]:
    labels: list[int] = []
    scores: list[float] = []
    for record in records:
        score = record.get("answer_confidence")
        correctness = record.get("correctness")
        if score is None or correctness is None:
            continue
        labels.append(int(bool(correctness)))
        scores.append(float(score))
    return labels, scores
