from __future__ import annotations

import math
import re
from typing import Any


SELF_REPAIR_PATTERN = re.compile(
    r"\b(wait|actually|let me revise|on second thought|correction|recalculate|re-compute|"
    r"let me redo|i should revise|i misread)\b",
    re.IGNORECASE,
)
BRANCHING_PATTERN = re.compile(
    r"\b(alternatively|either way|option\s+[abc]|another path|two possible actions|"
    r"could either|one option is)\b",
    re.IGNORECASE,
)


def _sorted_numeric(values: list[Any]) -> list[float]:
    numeric: list[float] = []
    for value in values:
        try:
            numeric.append(float(value))
        except (TypeError, ValueError):
            continue
    return sorted(numeric, reverse=True)


def _top2_margin(values: list[Any]) -> float | None:
    numeric = _sorted_numeric(values)
    if len(numeric) < 2:
        return None
    return abs(numeric[0] - numeric[1])


def _entropy(probabilities: dict[str, float] | None) -> float | None:
    if not probabilities:
        return None
    vals = [max(0.0, float(value)) for value in probabilities.values()]
    total = sum(vals)
    if total <= 0:
        return None
    normalized = [value / total for value in vals if value > 0]
    return -sum(value * math.log(value) for value in normalized)


def _softmax(values: list[float], temperature: float) -> list[float]:
    if not values:
        return []
    temp = max(float(temperature), 1e-6)
    scaled = [value / temp for value in values]
    max_value = max(scaled)
    exp_values = [math.exp(value - max_value) for value in scaled]
    normalizer = sum(exp_values)
    if normalizer <= 0:
        return []
    return [value / normalizer for value in exp_values]


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    q = max(0.0, min(1.0, float(q)))
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(math.floor(q * (len(ordered) - 1)))))
    return ordered[index]


def _candidate_logprob_diagnostics(
    candidate_action_logprobs: list[dict[str, Any]] | None,
    candidates: list[dict[str, Any]] | None,
    *,
    low_margin_delta: float,
    high_score_entropy_threshold: float,
    confidence_high_threshold: float,
    logprob_low_percentile: float,
    entropy_temperature: float,
) -> tuple[dict[str, bool], dict[str, Any]]:
    scores: list[dict[str, Any]] = []
    for entry in candidate_action_logprobs or []:
        if not isinstance(entry, dict):
            continue
        try:
            mean = float(entry["action_json_logprob_mean"])
        except (KeyError, TypeError, ValueError):
            continue
        scores.append({
            "rank": entry.get("rank"),
            "action": str(entry.get("action", "")).upper(),
            "mean": mean,
        })

    features = {
        "LOW_CANDIDATE_LOGPROB_MARGIN": False,
        "RANK_DISAGREEMENT": False,
        "HIGH_SCORE_ENTROPY": False,
        "CONFIDENCE_LOGPROB_MISMATCH": False,
    }
    diagnostics: dict[str, Any] = {"candidate_logprob_scoring_available": bool(scores)}
    if len(scores) < 2:
        return features, diagnostics

    ranked = sorted(scores, key=lambda item: item["mean"], reverse=True)
    margin = ranked[0]["mean"] - ranked[1]["mean"]
    features["LOW_CANDIDATE_LOGPROB_MARGIN"] = margin < low_margin_delta
    diagnostics["candidate_logprob_margin"] = round(margin, 6)
    diagnostics["candidate_logprob_rank1_action"] = ranked[0]["action"]

    candidate_list = [candidate for candidate in (candidates or []) if isinstance(candidate, dict)]
    if candidate_list:
        student_rank1 = str(candidate_list[0].get("action", "")).upper()
        features["RANK_DISAGREEMENT"] = bool(student_rank1 and student_rank1 != ranked[0]["action"])
        diagnostics["student_rank1_action"] = student_rank1

    probs = _softmax([entry["mean"] for entry in scores], entropy_temperature)
    if probs:
        entropy = -sum(prob * math.log(prob) for prob in probs if prob > 0)
        normalized_entropy = entropy / math.log(len(probs)) if len(probs) > 1 else 0.0
        features["HIGH_SCORE_ENTROPY"] = normalized_entropy >= high_score_entropy_threshold
        diagnostics["candidate_score_entropy"] = round(normalized_entropy, 6)

    low_threshold = _percentile([entry["mean"] for entry in scores], logprob_low_percentile)
    if low_threshold is not None and candidate_list:
        mean_by_key = {(entry["rank"], entry["action"]): entry["mean"] for entry in scores}
        for candidate in candidate_list:
            try:
                confidence = float(candidate.get("confidence"))
            except (TypeError, ValueError):
                continue
            key = (candidate.get("rank"), str(candidate.get("action", "")).upper())
            mean = mean_by_key.get(key)
            if mean is None:
                continue
            if confidence >= confidence_high_threshold and mean <= low_threshold:
                features["CONFIDENCE_LOGPROB_MISMATCH"] = True
                break
        diagnostics["candidate_logprob_low_threshold"] = round(low_threshold, 6)

    return features, diagnostics


def extract_process_feature_details(
    *,
    reason: str = "",
    raw_text: str = "",
    reasoning_attempt: str | None = None,
    candidate_confidences: list[Any] | None = None,
    action_probabilities: dict[str, float] | None = None,
    action_scores: dict[str, float] | None = None,
    candidate_action_logprobs: list[dict[str, Any]] | None = None,
    candidates: list[dict[str, Any]] | None = None,
    low_margin_delta: float = 0.10,
    high_score_entropy_threshold: float = 0.85,
    confidence_high_threshold: float = 0.75,
    logprob_low_percentile: float = 0.30,
    entropy_temperature: float = 1.0,
    confidence_source: str | None = None,
    token_threshold: int = 80,
) -> dict[str, Any]:
    """Return process uncertainty features and diagnostics.

    These features describe the student's generation process, not the task
    semantics. Raw JSON/tool observations are intentionally excluded from
    STRUGGLE_LONG and self-report phrases such as "uncertain" do not create a
    LOW_LOGIT_MARGIN signal.
    """
    reasoning_text = (reasoning_attempt if reasoning_attempt is not None else reason) or ""
    approx_tokens = len(reasoning_text.split())

    diagnostics: dict[str, Any] = {
        "reason_token_count": approx_tokens,
        "confidence_source": confidence_source,
    }

    probability_margin = _top2_margin(list((action_probabilities or {}).values()))
    score_margin = _top2_margin(list((action_scores or {}).values()))
    candidate_margin = _top2_margin(candidate_confidences or [])

    margin_source = None
    margin = None
    if probability_margin is not None:
        margin = probability_margin
        margin_source = "action_probabilities"
    elif score_margin is not None:
        margin = score_margin
        margin_source = "action_scores"
    elif candidate_margin is not None:
        margin = candidate_margin
        margin_source = "candidate_confidences"
    else:
        diagnostics["missing_action_margin_signal"] = True

    if margin is not None:
        diagnostics["action_margin"] = round(margin, 6)
        diagnostics["action_margin_source"] = margin_source

    entropy = _entropy(action_probabilities)
    if entropy is not None:
        diagnostics["action_entropy"] = round(entropy, 6)

    low_margin = bool(margin is not None and margin <= 0.15)
    high_entropy = bool(entropy is not None and entropy >= 1.25)
    close_candidates = bool(candidate_margin is not None and candidate_margin <= 0.10)
    explicit_branching = len(BRANCHING_PATTERN.findall(reasoning_text)) >= 1
    logprob_features, logprob_diagnostics = _candidate_logprob_diagnostics(
        candidate_action_logprobs,
        candidates,
        low_margin_delta=low_margin_delta,
        high_score_entropy_threshold=high_score_entropy_threshold,
        confidence_high_threshold=confidence_high_threshold,
        logprob_low_percentile=logprob_low_percentile,
        entropy_temperature=entropy_temperature,
    )
    diagnostics.update(logprob_diagnostics)

    return {
        "features": {
            "STRUGGLE_LONG": approx_tokens >= token_threshold,
            "HAS_SELF_REPAIR": bool(SELF_REPAIR_PATTERN.search(reasoning_text)),
            # Deprecated compatibility alias: v0.2.3 mining uses the
            # candidate action JSON logprob tags below instead.
            "LOW_LOGIT_MARGIN": low_margin,
            "HIGH_BRANCHING": bool(high_entropy or close_candidates or explicit_branching or logprob_features["HIGH_SCORE_ENTROPY"]),
            **logprob_features,
        },
        "diagnostics": diagnostics,
    }


def extract_process_features(
    reason: str,
    raw_text: str,
    tool_observation: dict[str, Any] | None = None,
    token_threshold: int = 80,
    reasoning_attempt: str | None = None,
    candidate_confidences: list[Any] | None = None,
    action_probabilities: dict[str, float] | None = None,
    action_scores: dict[str, float] | None = None,
    candidate_action_logprobs: list[dict[str, Any]] | None = None,
    candidates: list[dict[str, Any]] | None = None,
    confidence_source: str | None = None,
) -> dict[str, bool]:
    del tool_observation  # v0.2 process features must not include tool observations.
    details = extract_process_feature_details(
        reason=reason,
        raw_text=raw_text,
        reasoning_attempt=reasoning_attempt,
        candidate_confidences=candidate_confidences,
        action_probabilities=action_probabilities,
        action_scores=action_scores,
        candidate_action_logprobs=candidate_action_logprobs,
        candidates=candidates,
        confidence_source=confidence_source,
        token_threshold=token_threshold,
    )
    return details["features"]
