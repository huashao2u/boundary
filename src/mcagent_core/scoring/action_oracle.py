from __future__ import annotations

from mcagent_core.features.semantic_tags import build_semantic_tags


def choose_oracle_action(
    sample,
    semantic_tags: dict[str, bool] | None = None,
    clarify_allowed: bool = True,
    retrieval_allowed: bool = True,
    calculation_allowed: bool = True,
) -> str:
    tags = build_semantic_tags(sample) if semantic_tags is None else semantic_tags
    if tags["MISSING_INFO"] and clarify_allowed:
        return "CLARIFY"
    if tags["FALSE_PREMISE"] or tags["JUSTIFIED_REFUSE"]:
        return "REFUSE"
    if tags["CALCULATION_REQUIRED"] and calculation_allowed:
        return "CALCULATE"
    if tags["TOOL_REQUIRED"] and retrieval_allowed:
        return "SEARCH"
    return "ANSWER"
