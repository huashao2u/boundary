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
    if tags.get("CLARIFY_REQUIRED") and clarify_allowed:
        return "CLARIFY"
    if tags.get("FALSE_PREMISE") or tags.get("JUSTIFIED_REFUSE"):
        return "REFUSE"
    if tags.get("CALCULATION_REQUIRED") and calculation_allowed:
        return "CALCULATE"
    if (
        tags.get("SEARCH_REQUIRED")
        or tags.get("TIME_SENSITIVE")
        or tags.get("NEW_OR_TAIL_KNOWLEDGE")
    ) and retrieval_allowed:
        return "SEARCH"
    return "ANSWER"
