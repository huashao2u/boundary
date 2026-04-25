from __future__ import annotations

"""helpfulness.py — v0.2 two-entry-point design.

Training annotation path:  teacher_helpfulness(branch, teacher_label) -> float
Eval path:                 outcome_helpfulness(example, branch, branch_map, semantic_tags) -> bool

Gold is only used in outcome_helpfulness for offline correctness scoring (eval only).
"""

from typing import Any

from mcagent_boundary.scoring.correctness import needs_calculation


# ---------------------------------------------------------------------------
# Training annotation path — reads teacher_label.candidate_helpfulness
# ---------------------------------------------------------------------------

def teacher_helpfulness(branch: dict[str, Any], teacher_label: dict[str, Any] | None) -> float:
    """Return teacher-judged helpfulness score in [0,1] for branch.action.

    Reads candidate_helpfulness list from teacher_label.
    Returns 0.5 (neutral) if teacher_label is unavailable or action not found.
    """
    if teacher_label is None:
        return 0.5
    helpfulness_list = teacher_label.get("candidate_helpfulness") or []
    action = str(branch.get("action", "")).upper()
    for entry in helpfulness_list:
        if isinstance(entry, dict) and str(entry.get("action", "")).upper() == action:
            try:
                return max(0.0, min(1.0, float(entry["score"])))
            except (KeyError, TypeError, ValueError):
                return 0.5
    return 0.5


# ---------------------------------------------------------------------------
# Eval-only path — based on real tool execution outcomes
# ---------------------------------------------------------------------------

def outcome_helpfulness(
    example,
    branch: dict[str, Any],
    branch_map: dict[str, dict[str, Any]],
    semantic_tags: dict[str, bool],
) -> bool:
    """Determine if a branch is helpful based on real execution outcome (eval only).

    Gold labels are used only for offline correctness scoring here;
    they are NOT injected into student action_input.
    """
    action = str(branch.get("action", "")).upper()
    if action == "ANSWER":
        return branch.get("correctness") is True
    if action == "SEARCH":
        return _search_helpful(branch, branch_map, semantic_tags)
    if action == "CALCULATE":
        return _calculate_helpful(example, branch, branch_map, semantic_tags)
    if action == "CLARIFY":
        return _clarify_helpful(branch, semantic_tags)
    if action == "REFUSE":
        return _refuse_justified(example, branch, branch_map, semantic_tags)
    return False


def _search_helpful(
    branch: dict[str, Any],
    branch_map: dict[str, dict[str, Any]],
    semantic_tags: dict[str, bool],
) -> bool:
    """SEARCH is helpful only when:
    - observation is non-empty,
    - finalize after search is correct or semantic tags indicate external evidence needed,
    - AND student's own ANSWER candidate was wrong OR tags require external evidence.
    """
    results = ((branch.get("observation") or {}).get("results")) or []
    if not results:
        return False
    answer_branch = branch_map.get("ANSWER", {})
    needs_external = (
        semantic_tags.get("TIME_SENSITIVE")
        or semantic_tags.get("NEW_OR_TAIL_KNOWLEDGE")
        or semantic_tags.get("TOOL_REQUIRED")
    )
    answer_wrong = answer_branch.get("correctness") is not True
    return branch.get("correctness") is True and (answer_wrong or needs_external)


def _calculate_helpful(
    example,
    branch: dict[str, Any],
    branch_map: dict[str, dict[str, Any]],
    semantic_tags: dict[str, bool],
) -> bool:
    """CALCULATE is helpful only when:
    - calculator returned a result,
    - finalize after calculate is correct,
    - AND student's ANSWER was wrong OR calculation is genuinely needed.
    """
    observation = branch.get("observation") or {}
    if not observation.get("result"):
        return False
    answer_branch = branch_map.get("ANSWER", {})
    return (
        branch.get("correctness") is True
        and (answer_branch.get("correctness") is not True or needs_calculation(example, semantic_tags))
    )


def _clarify_helpful(branch: dict[str, Any], semantic_tags: dict[str, bool]) -> bool:
    """CLARIFY is helpful only when MISSING_INFO is present and a meaningful reply was received."""
    observation = branch.get("observation") or {}
    return bool(semantic_tags.get("MISSING_INFO") and observation.get("user_reply"))


def _refuse_justified(
    example,
    branch: dict[str, Any],
    branch_map: dict[str, dict[str, Any]],
    semantic_tags: dict[str, bool],
) -> bool:
    """REFUSE is justified based on semantic tags and structural properties."""
    answer_branch = branch_map.get("ANSWER", {})
    return bool(
        semantic_tags.get("FALSE_PREMISE")
        or semantic_tags.get("JUSTIFIED_REFUSE")
        or (semantic_tags.get("MISSING_INFO") and not example.can_clarify)
        or (
            semantic_tags.get("TOOL_REQUIRED")
            and not example.can_search
            and answer_branch.get("correctness") is not True
        )
    )


# ---------------------------------------------------------------------------
# Backward-compat wrappers (v0.1 callers)
# ---------------------------------------------------------------------------

def is_search_helpful(example, branch, branch_map, semantic_tags):
    return _search_helpful(branch, branch_map, semantic_tags)


def is_calculate_helpful(example, branch, branch_map, semantic_tags):
    return _calculate_helpful(example, branch, branch_map, semantic_tags)


def is_clarify_helpful(example, branch, branch_map, semantic_tags):
    return _clarify_helpful(branch, semantic_tags)


def is_refuse_justified(example, branch, branch_map, semantic_tags):
    return _refuse_justified(example, branch, branch_map, semantic_tags)
