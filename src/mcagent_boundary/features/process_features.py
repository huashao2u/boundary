from __future__ import annotations

from mcagent_core.features.extract_process_features import extract_process_features


def compute_process_features(reason_prefix: str, raw_text: str, token_threshold: int) -> dict[str, bool]:
    return extract_process_features(
        reason=reason_prefix,
        raw_text=raw_text,
        tool_observation=None,
        token_threshold=token_threshold,
    )

