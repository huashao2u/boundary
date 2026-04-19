from __future__ import annotations

from mcagent_boundary.scoring.correctness import needs_calculation


def is_search_helpful(example, branch: dict, branch_map: dict[str, dict], semantic_tags: dict[str, bool]) -> bool:
    results = ((branch.get("observation") or {}).get("results")) or []
    answer_branch = branch_map.get("ANSWER", {})
    return bool(results) and (
        branch.get("correctness") is True
        or semantic_tags.get("TOOL_REQUIRED")
        or semantic_tags.get("TIME_SENSITIVE")
        or answer_branch.get("correctness") is not True
    )


def is_calculate_helpful(example, branch: dict, branch_map: dict[str, dict], semantic_tags: dict[str, bool]) -> bool:
    observation = branch.get("observation") or {}
    answer_branch = branch_map.get("ANSWER", {})
    return bool(observation.get("result")) and (
        branch.get("correctness") is True
        or needs_calculation(example)
        or answer_branch.get("correctness") is not True
    )


def is_clarify_helpful(example, branch: dict, branch_map: dict[str, dict], semantic_tags: dict[str, bool]) -> bool:
    observation = branch.get("observation") or {}
    return semantic_tags.get("MISSING_INFO", False) and bool(observation.get("user_reply"))


def is_refuse_justified(example, branch: dict, branch_map: dict[str, dict], semantic_tags: dict[str, bool]) -> bool:
    answer_branch = branch_map.get("ANSWER", {})
    return bool(
        semantic_tags.get("FALSE_PREMISE")
        or semantic_tags.get("JUSTIFIED_REFUSE")
        or (semantic_tags.get("MISSING_INFO") and not example.can_clarify)
        or (semantic_tags.get("TOOL_REQUIRED") and not example.can_search and answer_branch.get("correctness") is not True)
    )

