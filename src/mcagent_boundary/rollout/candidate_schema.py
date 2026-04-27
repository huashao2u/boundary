from __future__ import annotations

import json
from typing import Any


ACTION_SET = {"ANSWER", "SEARCH", "CALCULATE", "CLARIFY", "REFUSE"}
REQUIRED_ACTION_INPUT_KEYS = {
    "ANSWER": "answer",
    "SEARCH": "query",
    "CALCULATE": "expression",
    "CLARIFY": "question",
    "REFUSE": "reason",
}

ANSWER_ALIASES = (
    "answer",
    "response",
    "result",
    "final",
    "final_answer",
    "output",
    "solution",
    "content",
    "text",
    "code",
    "courses",
    "books",
    "routine",
    "challenge",
    "plan",
)
SEARCH_ALIASES = ("query", "search_query", "question", "query_text", "keywords")
CALCULATE_ALIASES = ("expression", "expr", "formula", "equation", "calculation")
CLARIFY_ALIASES = ("question", "missing_info", "slots", "slots_to_fill", "missing_slot", "slot")
REFUSE_ALIASES = ("reason", "rationale", "explanation")


def _compact_string(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value).strip()
    try:
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    except (TypeError, ValueError):
        return str(value).strip()


def _first_nonempty(action_input: dict[str, Any], aliases: tuple[str, ...]) -> tuple[str, str] | tuple[None, str]:
    for key in aliases:
        value = _compact_string(action_input.get(key))
        if value:
            return key, value
    return None, ""


def _flatten_slot_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        parts = []
        for key in ("slot", "field", "name", "description", "inquiry", "question"):
            if value.get(key):
                parts.append(str(value[key]).strip())
        if parts:
            return ", ".join(dict.fromkeys(parts))
    if isinstance(value, (list, tuple, set)):
        parts = [_flatten_slot_text(item) for item in value]
        return ", ".join(part for part in parts if part)
    return _compact_string(value)


def _clarify_question_from_value(value: Any) -> str:
    text = _flatten_slot_text(value)
    if not text:
        return ""
    if text.endswith("?"):
        return text
    return f"Could you clarify {text}?"


def _fallback_answer_from_nonstandard_keys(action_input: dict[str, Any]) -> tuple[str | None, str]:
    ignored = {"action", "confidence", "brief_rationale"}
    payload = {
        key: value
        for key, value in action_input.items()
        if key not in ignored and _compact_string(value)
    }
    if not payload:
        return None, ""
    return "serialized_action_input", _compact_string(payload)


def canonicalize_action_input(
    action: str,
    action_input: Any,
    *,
    brief_rationale: str = "",
) -> tuple[dict[str, str], bool, list[dict[str, str]]]:
    """Return canonical action_input, validity, and diagnostics.

    The function never injects gold data. It only maps student-provided fields
    or, for REFUSE, the same candidate's rationale into the required field.
    """
    action = str(action or "").upper()
    diagnostics: list[dict[str, str]] = []
    if action not in ACTION_SET:
        return {}, False, [{"issue": "unknown_action", "detail": action}]

    if not isinstance(action_input, dict):
        diagnostics.append({"issue": "non_dict_action_input", "detail": type(action_input).__name__})
        action_input = {}

    canonical: dict[str, str] = {}
    source_key: str | None = None
    value = ""

    if action == "ANSWER":
        source_key, value = _first_nonempty(action_input, ANSWER_ALIASES)
        if not value:
            source_key, value = _fallback_answer_from_nonstandard_keys(action_input)
        canonical["answer"] = value
    elif action == "SEARCH":
        source_key, value = _first_nonempty(action_input, SEARCH_ALIASES)
        canonical["query"] = value
    elif action == "CALCULATE":
        source_key, value = _first_nonempty(action_input, CALCULATE_ALIASES)
        canonical["expression"] = value
    elif action == "CLARIFY":
        source_key, value = _first_nonempty(action_input, CLARIFY_ALIASES)
        if source_key != "question":
            value = _clarify_question_from_value(action_input.get(source_key)) if source_key else ""
        canonical["question"] = value
        slot_value = _compact_string(action_input.get("slot") or action_input.get("missing_slot"))
        if slot_value:
            canonical["slot"] = slot_value
    elif action == "REFUSE":
        source_key, value = _first_nonempty(action_input, REFUSE_ALIASES)
        if not value:
            value = _compact_string(brief_rationale)
            if value:
                source_key = "brief_rationale"
        canonical["reason"] = value

    required = REQUIRED_ACTION_INPUT_KEYS[action]
    if source_key and source_key != required:
        diagnostics.append({"issue": "mapped_nonstandard_key", "detail": f"{source_key}->{required}"})
    if not canonical.get(required, "").strip():
        diagnostics.append({"issue": "empty_required_action_input", "detail": required})
        return canonical, False, diagnostics
    return canonical, True, diagnostics


def canonicalize_candidate(candidate: dict[str, Any], *, rank: int | None = None) -> dict[str, Any]:
    action = str(candidate.get("action", "")).upper()
    canonical_input, valid, schema_diagnostics = canonicalize_action_input(
        action,
        candidate.get("action_input", {}),
        brief_rationale=str(candidate.get("brief_rationale", "")),
    )
    normalized = dict(candidate)
    normalized["rank"] = candidate.get("rank", rank)
    normalized["action"] = action
    normalized["raw_action_input"] = candidate.get("action_input", {})
    normalized["canonical_action_input"] = canonical_input
    normalized["action_input"] = canonical_input
    normalized["valid_candidate"] = bool(valid)
    normalized["schema_diagnostics"] = schema_diagnostics
    normalized["is_student_candidate"] = bool(candidate.get("is_student_candidate", True))
    normalized["is_debug_fallback"] = bool(candidate.get("is_debug_fallback", False))
    return normalized


def canonicalize_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        canonicalize_candidate(candidate, rank=index + 1)
        for index, candidate in enumerate(candidates)
        if isinstance(candidate, dict)
    ]


def is_valid_candidate(candidate: dict[str, Any]) -> bool:
    if not bool(candidate.get("valid_candidate")):
        return False
    action = str(candidate.get("action", "")).upper()
    required = REQUIRED_ACTION_INPUT_KEYS.get(action)
    if required is None:
        return False
    action_input = candidate.get("canonical_action_input") or candidate.get("action_input") or {}
    return isinstance(action_input, dict) and bool(_compact_string(action_input.get(required)))


def valid_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [candidate for candidate in candidates if is_valid_candidate(candidate)]


def required_input_value(candidate: dict[str, Any]) -> str:
    action = str(candidate.get("action", "")).upper()
    required = REQUIRED_ACTION_INPUT_KEYS.get(action)
    if required is None:
        return ""
    action_input = candidate.get("canonical_action_input") or candidate.get("action_input") or {}
    if not isinstance(action_input, dict):
        return ""
    return _compact_string(action_input.get(required))
