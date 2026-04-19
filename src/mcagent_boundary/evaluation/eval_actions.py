from __future__ import annotations

from typing import Any


def _safe_rate(numerator: int, denominator: int) -> float | None:
    return None if denominator == 0 else numerator / denominator


def evaluate_actions(rollouts: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(rollouts)
    action_counts: dict[str, int] = {}
    action_correct = 0
    search_total = 0
    unnecessary_searches = 0
    calculate_total = 0
    helpful_calculates = 0
    clarify_total = 0
    helpful_clarifies = 0
    refuse_total = 0
    justified_refuses = 0
    answer_total = 0
    over_answers = 0
    total_utility = 0.0

    for record in rollouts:
        natural_action = record["natural_action"]
        best_action = record.get("best_action")
        natural_branch = record.get("natural_branch") or {}
        action_counts[natural_action] = action_counts.get(natural_action, 0) + 1
        action_correct += int(natural_action == best_action)
        total_utility += float(natural_branch.get("utility", 0.0))
        if natural_action == "SEARCH":
            search_total += 1
            unnecessary_searches += int(natural_branch.get("outcome_label") != "SEARCH_helpful")
        if natural_action == "CALCULATE":
            calculate_total += 1
            helpful_calculates += int(natural_branch.get("outcome_label") == "CALCULATE_helpful")
        if natural_action == "CLARIFY":
            clarify_total += 1
            helpful_clarifies += int(natural_branch.get("outcome_label") == "CLARIFY_helpful")
        if natural_action == "REFUSE":
            refuse_total += 1
            justified_refuses += int(natural_branch.get("outcome_label") == "REFUSE_justified")
        if natural_action == "ANSWER":
            answer_total += 1
            over_answers += int(best_action != "ANSWER")

    non_answer_rate = None if total == 0 else 1 - (action_counts.get("ANSWER", 0) / total)
    return {
        "action_accuracy": _safe_rate(action_correct, total),
        "unnecessary_search_rate": _safe_rate(unnecessary_searches, search_total),
        "calculate_helpfulness": _safe_rate(helpful_calculates, calculate_total),
        "clarify_helpfulness": _safe_rate(helpful_clarifies, clarify_total),
        "justified_refusal_rate": _safe_rate(justified_refuses, refuse_total),
        "over_refusal_rate": _safe_rate(refuse_total - justified_refuses, refuse_total),
        "over_answer_rate": _safe_rate(over_answers, answer_total),
        "non_answer_rate": non_answer_rate,
        "expected_utility": None if total == 0 else total_utility / total,
        "action_counts": action_counts,
    }

