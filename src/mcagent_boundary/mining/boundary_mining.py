from __future__ import annotations

from collections import Counter
from typing import Any

from mcagent_boundary.rollout.candidate_schema import is_valid_candidate, required_input_value, valid_as_rejected


def _stable_process(process_features: dict[str, bool]) -> bool:
    deprecated = {"LOW_LOGIT_MARGIN"}
    return not any(bool(value) for key, value in process_features.items() if key not in deprecated)


_EXCLUDED_ROLLOUT_ISSUES = {
    "invalid_candidate_output",
    "legacy_single_decision_fallback",
    "invalid_schema",
}

_ACTION_TAGS = {
    "SEARCH": ("SEARCH_REQUIRED", "TIME_SENSITIVE", "NEW_OR_TAIL_KNOWLEDGE"),
    "CALCULATE": ("CALCULATION_REQUIRED",),
    "CLARIFY": ("CLARIFY_REQUIRED", "MISSING_INFO"),
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


def _valid_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [candidate for candidate in candidates if is_valid_candidate(candidate)]


def _trainable_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [candidate for candidate in candidates if valid_as_rejected(candidate)]


def _confidence_margin(candidates: list[dict[str, Any]]) -> float | None:
    if len(candidates) < 2:
        return None
    first = _as_float(candidates[0].get("confidence"))
    second = _as_float(candidates[1].get("confidence"))
    if first is None or second is None:
        return None
    return abs(first - second)


def _u_rel_margin(candidates: list[dict[str, Any]]) -> float | None:
    values = sorted(
        [value for value in (_as_float(candidate.get("u_rel")) for candidate in candidates) if value is not None],
        reverse=True,
    )
    if len(values) < 2:
        return None
    return abs(values[0] - values[1])


def _semantic_supports(action: str, semantic_tags: dict[str, bool]) -> bool:
    return any(bool(semantic_tags.get(tag)) for tag in _ACTION_TAGS.get(action, ()))


def _has_non_answer_pressure(natural_action: str, actions: list[str], semantic_tags: dict[str, bool]) -> bool:
    if natural_action == "ANSWER":
        return any(
            action != "ANSWER" and action in actions and _semantic_supports(action, semantic_tags)
            for action in _ACTION_TAGS
        )
    return "ANSWER" in actions


def _has_real_action_competition(
    natural_action: str,
    actions: list[str],
    candidates: list[dict[str, Any]],
    semantic_tags: dict[str, bool],
    competition_window: float,
) -> bool:
    confidence_margin = _confidence_margin(candidates)
    if confidence_margin is not None and confidence_margin <= competition_window:
        return True
    u_rel_margin = _u_rel_margin(candidates)
    if u_rel_margin is not None and u_rel_margin <= competition_window:
        return True
    if _has_non_answer_pressure(natural_action, actions, semantic_tags):
        return True
    return False


def _dataset_threshold(dataset: str, mining_cfg: dict[str, Any], *, key: str, default_key: str) -> float:
    by_dataset = mining_cfg.get(key) or {}
    if dataset in by_dataset:
        return float(by_dataset[dataset])
    return float(mining_cfg.get(default_key, mining_cfg.get("min_delta_u", 0.5)))


def _boundary_score(rollout: dict[str, Any], candidates: list[dict[str, Any]], config: dict[str, Any]) -> tuple[float, list[str]]:
    process_features = rollout.get("process_features") or {}
    semantic_tags = rollout.get("semantic_tags") or {}
    actions = _candidate_actions(candidates)
    natural_action = str(rollout.get("natural_action") or (actions[0] if actions else "")).upper()
    margin = _confidence_margin(candidates)
    reasons: list[str] = []
    score = 0.0

    configured_weights = config.get("mining", {}).get("boundary_score_weights", {})
    process_weights = {
        "LOW_CANDIDATE_LOGPROB_MARGIN": float(configured_weights.get("low_candidate_logprob_margin", 0.15)),
        "RANK_DISAGREEMENT": float(configured_weights.get("rank_disagreement", 0.10)),
        "HIGH_SCORE_ENTROPY": float(configured_weights.get("high_score_entropy", 0.10)),
        "CONFIDENCE_LOGPROB_MISMATCH": float(configured_weights.get("confidence_logprob_mismatch", 0.10)),
        "HIGH_BRANCHING": 0.04,
        "STRUGGLE_LONG": 0.03,
        "HAS_SELF_REPAIR": 0.03,
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
            score += 0.04
            reasons.append(f"semantic_supports_{action.lower()}")

    return round(score, 6), reasons


def _legacy_utility_pool(
    rollout: dict[str, Any],
    min_delta: float,
    competition_window: float,
    anchor_margin: float,
) -> dict[str, Any] | None:
    """Classify pre-v0.2 rollout records that already contain utility fields.

    DEPRECATED(mainline): current rollout records do not populate
    candidate_utilities / ranked_actions[*].utility before teacher labeling.
    The active path below uses student candidate competition, process features,
    and semantic tags instead.
    """
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
    pool_distribution: Counter[str] = Counter()
    tag_distribution: Counter[str] = Counter()
    reason_distribution: Counter[str] = Counter()
    total_candidates = 0
    invalid_candidates = 0
    empty_action_input_count = 0

    for rollout in rollouts:
        semantic_tags = rollout.get("semantic_tags") or {}
        for tag, enabled in semantic_tags.items():
            if enabled:
                tag_distribution[tag] += 1

        excluded_issues = _excluded_rollout_issues(rollout)
        if excluded_issues:
            record = {**rollout, "pool": "other", "pool_reason": "excluded_invalid_rollout", "excluded_issues": excluded_issues}
            other.append(record)
            pool_distribution["other"] += 1
            reason_distribution["excluded_invalid_rollout"] += 1
            continue

        raw_candidates = list(rollout.get("candidates") or [])
        valid_candidate_list = _trainable_candidates(raw_candidates)
        total_candidates += len(raw_candidates)
        invalid_candidates += len(raw_candidates) - len(valid_candidate_list)
        empty_action_input_count += int(rollout.get("empty_action_input_count") or 0)

        rollout_for_pool = {
            **rollout,
            "candidates": valid_candidate_list,
            "num_valid_candidates": len(valid_candidate_list),
            "invalid_candidate_count": len(raw_candidates) - len(valid_candidate_list),
        }

        legacy_pool = _legacy_utility_pool(rollout_for_pool, min_delta, competition_window, anchor_margin)
        if legacy_pool is not None:
            pool = legacy_pool["pool"]
            record = legacy_pool["record"]
            pool_distribution[pool] += 1
            if pool == "clear_answer_anchor":
                clear_answer.append(record)
            elif pool == "clear_external_anchor":
                clear_external.append(record)
            elif pool == "boundary_critical":
                boundary_candidates.append(record)
            else:
                other.append(record)
            continue

        candidates = valid_candidate_list
        actions = _candidate_actions(candidates)
        if not candidates:
            reason = "not_enough_valid_candidates"
            other.append({**rollout_for_pool, "pool": "other", "pool_reason": reason})
            pool_distribution["other"] += 1
            reason_distribution[reason] += 1
            continue

        score, reasons = _boundary_score(rollout_for_pool, candidates, config)
        boundary_threshold = _dataset_threshold(
            str(rollout.get("dataset", "")),
            mining_cfg,
            key="boundary_threshold_by_dataset",
            default_key="boundary_threshold_default",
        )
        natural_action = str(rollout.get("natural_action") or candidates[0].get("action") or actions[0]).upper()
        stable = _stable_process(rollout.get("process_features") or {})
        high_confidence = (_as_float(candidates[0].get("confidence")) or 0.0) >= 0.75
        real_competition = _has_real_action_competition(
            natural_action,
            actions,
            candidates,
            semantic_tags,
            competition_window,
        )
        enriched = {
            **rollout_for_pool,
            "boundary_score": score,
            "boundary_threshold": boundary_threshold,
            "utility_gap": score,
            "candidate_actions": actions,
            "boundary_reasons": reasons,
            "has_real_action_competition": real_competition,
        }
        boundary_candidate_ready = (
            len(candidates) >= 2
            and len(actions) >= 2
            and "ANSWER" in actions
            and any(action != "ANSWER" for action in actions)
        )

        if (
            stable
            and high_confidence
            and natural_action == "ANSWER"
            and required_input_value(candidates[0])
            and not _has_non_answer_pressure(natural_action, actions, semantic_tags)
        ):
            clear_answer.append({**enriched, "pool": "clear_answer_anchor"})
            pool_distribution["clear_answer_anchor"] += 1
        elif (
            stable
            and high_confidence
            and natural_action != "ANSWER"
            and required_input_value(candidates[0])
            and _semantic_supports(natural_action, semantic_tags)
        ):
            clear_external.append({**enriched, "pool": "clear_external_anchor"})
            pool_distribution["clear_external_anchor"] += 1
        elif not boundary_candidate_ready:
            reason = "not_enough_valid_candidates" if len(candidates) < 2 else "missing_answer_or_external_competitor"
            other.append({**enriched, "pool": "other", "pool_reason": reason})
            pool_distribution["other"] += 1
            reason_distribution[reason] += 1
        elif score >= boundary_threshold and real_competition:
            boundary_candidates.append({**enriched, "pool": "boundary_critical"})
            pool_distribution["boundary_critical"] += 1
            for reason in reasons:
                reason_distribution[reason] += 1
        else:
            reason = "no_real_action_competition" if score >= boundary_threshold else "low_boundary_score"
            other.append({**enriched, "pool": "other", "pool_reason": reason})
            pool_distribution["other"] += 1
            reason_distribution[reason] += 1

    return {
        "summary": {
            "num_rollouts": len(rollouts),
            "num_valid_candidates": total_candidates - invalid_candidates,
            "invalid_candidate_rate": (invalid_candidates / total_candidates) if total_candidates else 0.0,
            "empty_action_input_count": empty_action_input_count,
            "num_boundary_candidates": len(boundary_candidates),
            "num_clear_answer_anchors": len(clear_answer),
            "num_clear_external_anchors": len(clear_external),
            "num_other": len(other),
            "pool_distribution": dict(pool_distribution),
            "tag_distribution": dict(tag_distribution),
            "boundary_reason_distribution": dict(reason_distribution),
            "clear_answer_anchor_count": len(clear_answer),
            "clear_external_anchor_count": len(clear_external),
        },
        "boundary_candidates": boundary_candidates,
        "clear_answer_anchors": clear_answer,
        "clear_external_anchors": clear_external,
        "other": other,
    }
