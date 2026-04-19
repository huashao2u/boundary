from __future__ import annotations

from typing import Any

from mcagent_core.eval.calibration_utils import binary_auroc, brier_score, expected_calibration_error, summarize_risk_coverage


def evaluate_calibration(rollouts: list[dict[str, Any]], default_confidence: float = 0.5) -> dict[str, Any]:
    labels: list[int] = []
    scores: list[float] = []
    for record in rollouts:
        natural_action = record.get("natural_action")
        best_action = record.get("best_action")
        if natural_action is None or best_action is None:
            continue
        confidence = record.get("natural_action_confidence")
        try:
            score = float(default_confidence if confidence is None else confidence)
        except (TypeError, ValueError):
            score = float(default_confidence)
        labels.append(int(natural_action == best_action))
        scores.append(max(0.0, min(1.0, score)))
    summary = summarize_risk_coverage(labels, scores) if labels else {"best_coverage_at_90_precision": None, "utility_coverage": None}
    return {
        "decision_auroc": binary_auroc(labels, scores),
        "decision_ece": expected_calibration_error(labels, scores),
        "decision_brier": brier_score(labels, scores),
        "num_samples": len(labels),
        **summary,
    }

