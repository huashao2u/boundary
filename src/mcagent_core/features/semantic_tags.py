from __future__ import annotations

import re
from typing import Any


FRESH_FACT_WORDS = ("latest", "current", "today", "recent", "now", "this year", "updated", "as of")
MISCONCEPTION_WORDS = ("always", "never", "prove that", "is it true", "does it exist")
MATH_PATTERN = re.compile(r"[\d\)\(]+\s*[\+\-\*\/=]|\bsolve\b|\bcompute\b|\bcalculate\b|\bfind the value\b", re.IGNORECASE)
SEARCH_CUE_WORDS = ("search", "look up", "retrieve", "verify", "check current", "up-to-date", "external evidence")
CLARIFY_CUE_WORDS = ("clarify", "unspecified", "underspecified", "ambiguous", "need more information", "missing information")
REFUSE_CUE_WORDS = ("false premise", "premise is false", "cannot verify", "unjustified", "does not exist", "not possible")
CALC_CUE_WORDS = ("calculate", "compute", "arithmetic", "equation", "formula", "solve")


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.lower()
    if isinstance(value, dict):
        return " ".join(f"{key} {_normalize_text(item)}" for key, item in value.items())
    if isinstance(value, (list, tuple, set)):
        return " ".join(_normalize_text(item) for item in value)
    return str(value).lower()


def build_semantic_tags_from_state(
    *,
    question: str,
    metadata: dict[str, Any] | None = None,
    task_type: str | None = None,
    dataset: str | None = None,
    reason_prefix: str | None = None,
    history_prefix: list[dict[str, Any]] | None = None,
) -> dict[str, bool]:
    metadata = metadata or {}
    question_text = question.lower()
    reason_text = _normalize_text(reason_prefix)
    history_text = _normalize_text(history_prefix)
    combined_state_text = " ".join(part for part in (question_text, reason_text, history_text) if part)

    false_premise = bool(metadata.get("false_premise")) or "false premise" in combined_state_text
    missing_info = (
        bool(metadata.get("vague"))
        or bool(metadata.get("missing_details"))
        or any(word in combined_state_text for word in CLARIFY_CUE_WORDS)
    )
    fresh_fact = any(word in combined_state_text for word in FRESH_FACT_WORDS) or bool(metadata.get("effective_year"))
    new_or_tail_knowledge = fresh_fact or bool(metadata.get("source")) or bool(metadata.get("graph_preview"))
    misconception_risk = any(word in combined_state_text for word in MISCONCEPTION_WORDS) or false_premise
    calculation_required = (
        (task_type == "math" if task_type is not None else False)
        and (bool(MATH_PATTERN.search(question)) or (dataset or "") in {"gsm8k", "competition_math"})
    ) or any(word in combined_state_text for word in CALC_CUE_WORDS)
    tool_required = (
        (task_type == "factual_boundary" if task_type is not None else False) and (fresh_fact or new_or_tail_knowledge)
    ) or any(word in combined_state_text for word in SEARCH_CUE_WORDS)
    justified_refuse = false_premise or any(word in combined_state_text for word in REFUSE_CUE_WORDS)
    return {
        "FRESH_FACT": fresh_fact,
        "FALSE_PREMISE": false_premise,
        "MISCONCEPTION_RISK": misconception_risk,
        "NEW_OR_TAIL_KNOWLEDGE": new_or_tail_knowledge,
        "MISSING_INFO": missing_info,
        "TOOL_REQUIRED": tool_required,
        "JUSTIFIED_REFUSE": justified_refuse,
        "CALCULATION_REQUIRED": calculation_required,
    }


def build_semantic_tags(sample) -> dict[str, bool]:
    return build_semantic_tags_from_state(
        question=sample.question,
        metadata=sample.metadata or {},
        task_type=sample.task_type,
        dataset=sample.dataset,
    )


def active_semantic_tags(sample) -> list[str]:
    tags = build_semantic_tags(sample)
    return [name for name, enabled in tags.items() if enabled]


def active_semantic_tags_from_state(
    *,
    question: str,
    metadata: dict[str, Any] | None = None,
    task_type: str | None = None,
    dataset: str | None = None,
    reason_prefix: str | None = None,
    history_prefix: list[dict[str, Any]] | None = None,
) -> list[str]:
    tags = build_semantic_tags_from_state(
        question=question,
        metadata=metadata,
        task_type=task_type,
        dataset=dataset,
        reason_prefix=reason_prefix,
        history_prefix=history_prefix,
    )
    return [name for name, enabled in tags.items() if enabled]
