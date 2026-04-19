from __future__ import annotations

from typing import Any


def _stable_process(process_features: dict[str, bool]) -> bool:
    return not any(bool(value) for value in process_features.values())


def mine_boundary_states(rollouts: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    mining_cfg = config["mining"]
    min_delta = float(mining_cfg["min_delta_u"])
    competition_window = float(mining_cfg["competition_window"])
    anchor_margin = float(mining_cfg["anchor_margin"])

    boundary_candidates: list[dict[str, Any]] = []
    clear_answer: list[dict[str, Any]] = []
    clear_external: list[dict[str, Any]] = []
    other: list[dict[str, Any]] = []

    for rollout in rollouts:
        ranked = rollout.get("ranked_actions") or []
        utilities = rollout.get("candidate_utilities") or {}
        if len(ranked) < 2 or not utilities:
            other.append({**rollout, "pool": "other", "pool_reason": "insufficient_actions"})
            continue
        best = ranked[0]
        second = ranked[1]
        answer_u = float(utilities.get("ANSWER", float("-inf")))
        best_u = float(best["utility"])
        gap = best_u - float(ranked[-1]["utility"])
        rollout["utility_gap"] = gap
        stable = _stable_process(rollout.get("process_features") or {})

        if best["action"] == "ANSWER" and stable:
            external_best = max(
                [float(value) for action, value in utilities.items() if action != "ANSWER"],
                default=float("-inf"),
            )
            if best_u - external_best >= anchor_margin:
                clear_answer.append({**rollout, "pool": "clear_answer_anchor"})
                continue

        if best["action"] != "ANSWER" and best_u - answer_u >= anchor_margin and best_u - float(second["utility"]) >= competition_window:
            clear_external.append({**rollout, "pool": "clear_external_anchor"})
            continue

        answer_competes = abs(best_u - answer_u) >= competition_window if "ANSWER" in utilities else False
        action_competes = abs(best_u - float(second["utility"])) <= competition_window
        natural_differs = rollout.get("natural_action") != best["action"]
        if gap >= min_delta and (answer_competes or action_competes or natural_differs):
            boundary_candidates.append({**rollout, "pool": "boundary_critical"})
        else:
            other.append({**rollout, "pool": "other", "pool_reason": "not_boundary_critical"})

    return {
        "summary": {
            "num_rollouts": len(rollouts),
            "num_boundary_candidates": len(boundary_candidates),
            "num_clear_answer_anchors": len(clear_answer),
            "num_clear_external_anchors": len(clear_external),
            "num_other": len(other),
        },
        "boundary_candidates": boundary_candidates,
        "clear_answer_anchors": clear_answer,
        "clear_external_anchors": clear_external,
        "other": other,
    }
