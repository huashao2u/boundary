from __future__ import annotations

from typing import Any


def _stable_process(process_features: dict[str, bool]) -> bool:
    return not any(bool(value) for value in process_features.values())


_EXCLUDED_ROLLOUT_ISSUES = {
    "invalid_candidate_output",
    "legacy_single_decision_fallback",
    "invalid_schema",
    "invalid_missing_answer",
    "invalid_single_action",
}

_ACTION_TAGS = {
    "SEARCH": ("TIME_SENSITIVE", "NEW_OR_TAIL_KNOWLEDGE", "TOOL_REQUIRED"),
    "CALCULATE": ("CALCULATION_REQUIRED",),
    "CLARIFY": ("MISSING_INFO",),
    "REFUSE": ("FALSE_PREMISE", "JUSTIFIED_REFUSE"),
}


def _excluded_rollout_issues(rollout: dict[str, Any]) -> list[str]:
    issues = []
    for diagnostic in rollout.get("diagnostics") or []:
        issue = str(diagnostic.get("issue", ""))
        if issue in _EXCLUDED_ROLLOUT_ISSUES:
            issues.append(issue)
    return sorted(set(issues))


def _as_float(value: Any) -> float | None:
    try:
        return None if value is None else float(value)
    except (TypeError, ValueError):
        return None


def _candidate_actions(candidates: list[dict[str, Any]]) -> list[str]:
    actions = []
    for candidate in candidates:
        action = str(candidate.get("action", "")).upper()
        if action and action not in actions:
            actions.append(action)
    return actions


def _confidence_margin(candidates: list[dict[str, Any]]) -> float | None:
    if len(candidates) < 2:
        return None
    first = _as_float(candidates[0].get("confidence"))
    second = _as_float(candidates[1].get("confidence"))
    if first is None or second is None:
        return None
    return abs(first - second)


def _semantic_supports(action: str, semantic_tags: dict[str, bool]) -> bool:
    return any(bool(semantic_tags.get(tag)) for tag in _ACTION_TAGS.get(action, ()))


def _has_non_answer_pressure(natural_action: str, actions: list[str], semantic_tags: dict[str, bool]) -> bool:
    if natural_action == "ANSWER":
        return any(
            action != "ANSWER" and action in actions and _semantic_supports(action, semantic_tags)
            for action in _ACTION_TAGS
        )
    return "ANSWER" in actions


def _boundary_score(rollout: dict[str, Any], candidates: list[dict[str, Any]]) -> tuple[float, list[str]]:
    process_features = rollout.get("process_features") or {}
    semantic_tags = rollout.get("semantic_tags") or {}
    actions = _candidate_actions(candidates)
    natural_action = str(rollout.get("natural_action") or (actions[0] if actions else "")).upper()
    margin = _confidence_margin(candidates)
    reasons: list[str] = []
    score = 0.0

    process_weights = {
        "LOW_LOGIT_MARGIN": 0.25,
        "HIGH_BRANCHING": 0.20,
        "STRUGGLE_LONG": 0.15,
        "HAS_SELF_REPAIR": 0.15,
    }
    for feature, weight in process_weights.items():
        if process_features.get(feature):
            score += weight
            reasons.append(feature)

    if len(actions) >= 3:
        score += 0.15
        reasons.append("three_way_action_competition")
    elif len(actions) == 2:
        score += 0.08
        reasons.append("two_way_action_competition")

    if margin is None:
        score += 0.05
        reasons.append("missing_confidence_margin")
    elif margin <= 0.20:
        score += 0.20
        reasons.append("low_confidence_margin")
    elif margin <= 0.40:
        score += 0.10
        reasons.append("moderate_confidence_margin")

    if _has_non_answer_pressure(natural_action, actions, semantic_tags):
        score += 0.20
        reasons.append("answer_external_boundary")

    for action in actions:
        if action != "ANSWER" and _semantic_supports(action, semantic_tags):
            score += 0.05
            reasons.append(f"semantic_supports_{action.lower()}")

    return round(score, 6), reasons


def _legacy_utility_pool(
    rollout: dict[str, Any],
    min_delta: float,
    competition_window: float,
    anchor_margin: float,
) -> dict[str, Any] | None:
    ranked = rollout.get("ranked_actions") or []
    utilities = rollout.get("candidate_utilities") or {}
    if len(ranked) < 2 or not utilities or "utility" not in ranked[0]:
        return None
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
            return {"pool": "clear_answer_anchor", "record": {**rollout, "pool": "clear_answer_anchor"}}

    if best["action"] != "ANSWER" and best_u - answer_u >= anchor_margin and best_u - float(second["utility"]) >= competition_window:
        return {"pool": "clear_external_anchor", "record": {**rollout, "pool": "clear_external_anchor"}}

    answer_competes = abs(best_u - answer_u) >= competition_window if "ANSWER" in utilities else False
    action_competes = abs(best_u - float(second["utility"])) <= competition_window
    natural_differs = rollout.get("natural_action") != best["action"]
    if gap >= min_delta and (answer_competes or action_competes or natural_differs):
        return {"pool": "boundary_critical", "record": {**rollout, "pool": "boundary_critical"}}
    return {"pool": "other", "record": {**rollout, "pool": "other", "pool_reason": "not_boundary_critical"}}


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
        excluded_issues = _excluded_rollout_issues(rollout)
        if excluded_issues:
            other.append({**rollout, "pool": "other", "pool_reason": "excluded_invalid_rollout", "excluded_issues": excluded_issues})
            continue

        legacy_pool = _legacy_utility_pool(rollout, min_delta, competition_window, anchor_margin)
        if legacy_pool is not None:
            pool = legacy_pool["pool"]
            record = legacy_pool["record"]
            if pool == "clear_answer_anchor":
                clear_answer.append(record)
            elif pool == "clear_external_anchor":
                clear_external.append(record)
            elif pool == "boundary_critical":
                boundary_candidates.append(record)
            else:
                other.append(record)
            continue

        candidates = list(rollout.get("candidates") or [])
        actions = _candidate_actions(candidates)
        if len(actions) < 2 or "ANSWER" not in actions:
            other.append({**rollout, "pool": "other", "pool_reason": "insufficient_valid_candidates"})
            continue

        score, reasons = _boundary_score(rollout, candidates)
        natural_action = str(rollout.get("natural_action") or actions[0]).upper()
        stable = _stable_process(rollout.get("process_features") or {})
        high_confidence = (_as_float(rollout.get("natural_action_confidence")) or 0.0) >= 0.75
        enriched = {
            **rollout,
            "boundary_score": score,
            "utility_gap": score,
            "candidate_actions": actions,
            "boundary_reasons": reasons,
        }

        if stable and high_confidence and natural_action == "ANSWER" and not _has_non_answer_pressure(natural_action, actions, rollout.get("semantic_tags") or {}):
            clear_answer.append({**enriched, "pool": "clear_answer_anchor"})
        elif stable and high_confidence and natural_action != "ANSWER" and _semantic_supports(natural_action, rollout.get("semantic_tags") or {}):
            clear_external.append({**enriched, "pool": "clear_external_anchor"})
        elif score >= min_delta:
            boundary_candidates.append({**enriched, "pool": "boundary_critical"})
        else:
            other.append({**enriched, "pool": "other", "pool_reason": "low_boundary_score"})

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
