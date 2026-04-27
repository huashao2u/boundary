from __future__ import annotations

from typing import Any

from mcagent_core.features.extract_process_features import (
    extract_process_feature_details,
    extract_process_features,
)


def compute_process_feature_details(
    reason_prefix: str,
    raw_text: str,
    token_threshold: int,
    reasoning_attempt: str | None = None,
    candidate_confidences: list[Any] | None = None,
    action_probabilities: dict[str, float] | None = None,
    action_scores: dict[str, float] | None = None,
    confidence_source: str | None = None,
) -> dict[str, Any]:
    return extract_process_feature_details(
        reason=reason_prefix,
        raw_text=raw_text,
        reasoning_attempt=reasoning_attempt,
        candidate_confidences=candidate_confidences,
        action_probabilities=action_probabilities,
        action_scores=action_scores,
        confidence_source=confidence_source,
        token_threshold=token_threshold,
    )


def compute_process_features(
    reason_prefix: str,
    raw_text: str,
    token_threshold: int,
    reasoning_attempt: str | None = None,
    candidate_confidences: list[Any] | None = None,
    action_probabilities: dict[str, float] | None = None,
    action_scores: dict[str, float] | None = None,
    confidence_source: str | None = None,
) -> dict[str, bool]:
    return extract_process_features(
        reason=reason_prefix,
        raw_text=raw_text,
        tool_observation=None,
        token_threshold=token_threshold,
        reasoning_attempt=reasoning_attempt,
        candidate_confidences=candidate_confidences,
        action_probabilities=action_probabilities,
        action_scores=action_scores,
        confidence_source=confidence_source,
    )
