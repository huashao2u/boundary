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


def extract_process_feature_details(
    *,
    reason: str = "",
    raw_text: str = "",
    reasoning_attempt: str | None = None,
    candidate_confidences: list[Any] | None = None,
    action_probabilities: dict[str, float] | None = None,
    action_scores: dict[str, float] | None = None,
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

    return {
        "features": {
            "STRUGGLE_LONG": approx_tokens >= token_threshold,
            "HAS_SELF_REPAIR": bool(SELF_REPAIR_PATTERN.search(reasoning_text)),
            "LOW_LOGIT_MARGIN": low_margin,
            "HIGH_BRANCHING": bool(high_entropy or close_candidates or explicit_branching),
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
        confidence_source=confidence_source,
        token_threshold=token_threshold,
    )
    return details["features"]
