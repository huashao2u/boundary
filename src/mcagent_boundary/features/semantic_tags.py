from __future__ import annotations

import re
from typing import Any


TIME_WORDS = ("latest", "current", "today", "recent", "now", "this year", "updated", "as of")
MISCONCEPTION_WORDS = ("always", "never", "is it true", "does it exist", "prove that")
SEARCH_WORDS = ("search", "look up", "verify", "retrieve", "check online", "up-to-date")
REFUSE_WORDS = ("false premise", "cannot verify", "unjustified", "does not exist")
CALC_WORDS = ("calculate", "compute", "equation", "solve", "find the value")
CALC_EVIDENCE_PATTERN = re.compile(
    r"(\d|[%$]|\\frac|\\sqrt|[+\-*/=<>]| total | sum | difference | product | twice | half | percent)"
)


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


def infer_semantic_tags(example, reason_prefix: str = "", history: list[dict[str, Any]] | None = None) -> dict[str, bool]:
    metadata = dict(example.metadata or {})
    question = example.question.lower()
    combined = " ".join(part for part in [question, _normalize_text(reason_prefix), _normalize_text(history)] if part)
    time_sensitive = any(word in combined for word in TIME_WORDS) or bool(metadata.get("effective_year"))
    false_premise = bool(metadata.get("false_premise")) or any(word in combined for word in REFUSE_WORDS)
    missing_info = bool(metadata.get("vague")) or bool(metadata.get("missing_details"))
    new_or_tail = example.dataset == "mintqa" or bool(metadata.get("graph_preview")) or bool(metadata.get("source"))
    misconception_risk = false_premise or any(word in combined for word in MISCONCEPTION_WORDS)
    calc_required = any(word in combined for word in CALC_WORDS) or bool(CALC_EVIDENCE_PATTERN.search(f" {combined} "))
    tool_required = (
        (example.can_search and (time_sensitive or new_or_tail or any(word in combined for word in SEARCH_WORDS)))
        or (calc_required and example.can_calculate)
    )
    justified_refuse = false_premise or (missing_info and not example.can_clarify)
    return {
        "TIME_SENSITIVE": time_sensitive,
        "FALSE_PREMISE": false_premise,
        "MISSING_INFO": missing_info,
        "TOOL_REQUIRED": tool_required,
        "JUSTIFIED_REFUSE": justified_refuse,
        "NEW_OR_TAIL_KNOWLEDGE": new_or_tail,
        "MISCONCEPTION_RISK": misconception_risk,
        "CALCULATION_REQUIRED": calc_required,
    }


def active_semantic_tags(example, reason_prefix: str = "", history: list[dict[str, Any]] | None = None) -> list[str]:
    return [name for name, enabled in infer_semantic_tags(example, reason_prefix=reason_prefix, history=history).items() if enabled]
