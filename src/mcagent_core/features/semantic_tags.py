from __future__ import annotations

import re
from typing import Any, TypedDict


class SemanticTagEvidence(TypedDict):
    source: str
    text: str
    confidence: float


PRIMARY_TAGS = (
    "TIME_SENSITIVE",
    "NEW_OR_TAIL_KNOWLEDGE",
    "FALSE_PREMISE",
    "MISCONCEPTION_RISK",
    "MISSING_INFO",
    "CALCULATION_REQUIRED",
    "JUSTIFIED_REFUSE",
)
DERIVED_TAGS = (
    "SEARCH_REQUIRED",
    "CALCULATE_REQUIRED",
    "CLARIFY_REQUIRED",
    "TOOL_REQUIRED",
)
ALL_TAGS = PRIMARY_TAGS + DERIVED_TAGS

CURRENT_TIME_RE = re.compile(
    r"\b(latest|currently|current|today|recent|recently|now|right now|this year|this month|this week|"
    r"up[- ]to[- ]date|newest|new version|latest version|current version|current price|current status)\b",
    re.IGNORECASE,
)
AS_OF_DYNAMIC_RE = re.compile(r"\bas of\s+(today|now|right now|this (?:year|month|week))\b", re.IGNORECASE)
FIXED_TIME_RE = re.compile(
    r"\b(as of|current through|through|in|during|for)\s+(?:the\s+year\s+)?(?:19|20)\d{2}\b",
    re.IGNORECASE,
)
FALSE_PREMISE_RE = re.compile(
    r"\b(false premise|premise is false|nonexistent|does not exist|never existed|never happened|no such)\b",
    re.IGNORECASE,
)
MISCONCEPTION_RE = re.compile(
    r"\b(common misconception|myth|urban legend|pseudoscience|folk belief|"
    r"is it true that|do vaccines cause autism|flat earth|homeopathy)\b",
    re.IGNORECASE,
)
CALC_EXPRESSION_RE = re.compile(
    r"(\d+(?:\s*[+\-*/^=<>]\s*\d+)+|\\frac|\\sqrt|\b\d+(?:\.\d+)?\s*%)",
    re.IGNORECASE,
)
CALC_WORD_RE = re.compile(
    r"\b(calculate|compute|evaluate|solve|simplify|find the value|how many|how much|"
    r"total|sum|difference|product|quotient|percent|percentage|twice|half|average|ratio)\b",
    re.IGNORECASE,
)
MATH_RELATION_RE = re.compile(r"\b(if|after|before|each|per|left|remain|altogether|combined)\b", re.IGNORECASE)
EXTERNAL_EVIDENCE_RE = re.compile(
    r"\b(search|look up|retrieve|external evidence|verify online|check online|check current)\b",
    re.IGNORECASE,
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


def _truthy_metadata(metadata: dict[str, Any], *keys: str) -> bool:
    for key in keys:
        value = metadata.get(key)
        if isinstance(value, str):
            if value.strip().lower() in {"true", "yes", "1", "y"}:
                return True
            if value.strip().lower() in {"false", "no", "0", "n", ""}:
                continue
        if bool(value):
            return True
    return False


def _metadata_text(metadata: dict[str, Any], *keys: str) -> str:
    return " ".join(_normalize_text(metadata.get(key)) for key in keys if key in metadata)


def _as_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1", "y"}:
            return True
        if lowered in {"false", "no", "0", "n", ""}:
            return False
    return bool(value)


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _add_evidence(
    evidence: dict[str, list[SemanticTagEvidence]],
    tag: str,
    source: str,
    text: Any,
    confidence: float,
) -> None:
    rendered = str(text).strip()
    if not rendered:
        return
    evidence.setdefault(tag, []).append(
        {
            "source": source,
            "text": rendered[:240],
            "confidence": round(max(0.0, min(1.0, float(confidence))), 3),
        }
    )


def _missing_info_is_critical(metadata: dict[str, Any]) -> bool:
    if _truthy_metadata(metadata, "critical_missing_info", "requires_clarification", "underspecified"):
        return True
    if _truthy_metadata(metadata, "minor_preference_missing", "optional_preference_missing"):
        return False
    missing_details = metadata.get("missing_details")
    if isinstance(missing_details, list):
        if not missing_details:
            return False
        for item in missing_details:
            if not isinstance(item, dict):
                return True
            if item.get("critical") is True:
                return True
            try:
                if int(item.get("importance", 1)) >= 2:
                    return True
            except (TypeError, ValueError):
                return True
        return False
    return bool(missing_details) or _truthy_metadata(metadata, "vague")


def infer_semantic_tag_details(
    *,
    question: str,
    metadata: dict[str, Any] | None = None,
    task_type: str | None = None,
    dataset: str | None = None,
    reason_prefix: str | None = None,
    history_prefix: list[dict[str, Any]] | None = None,
    candidate_action_input: dict[str, Any] | None = None,
    can_search: bool | None = None,
    can_calculate: bool | None = None,
    can_clarify: bool | None = None,
) -> dict[str, Any]:
    """Infer semantic action-calibration tags plus diagnostic evidence.

    The authoritative rules live here. They intentionally rely on the question
    and trusted metadata; student reasoning is recorded only as low-confidence
    evidence and is not allowed to turn a static task into a dynamic/search task.
    """
    metadata = dict(metadata or {})
    question_text = str(question or "")
    question_lower = question_text.lower()
    dataset_lower = str(dataset or metadata.get("legacy_dataset") or "").lower()
    task_type_lower = str(task_type or metadata.get("task_type") or "").lower()
    reason_text = _normalize_text(reason_prefix)
    candidate_text = _normalize_text(candidate_action_input)
    can_search = _as_bool(can_search, _as_bool(metadata.get("can_search"), False))
    can_calculate = _as_bool(can_calculate, _as_bool(metadata.get("can_calculate"), False))
    can_clarify = _as_bool(can_clarify, _as_bool(metadata.get("can_clarify"), False))

    tags = {tag: False for tag in ALL_TAGS}
    evidence: dict[str, list[SemanticTagEvidence]] = {tag: [] for tag in ALL_TAGS}

    fixed_time_anchor = bool(FIXED_TIME_RE.search(question_text))
    time_match = CURRENT_TIME_RE.search(question_text)
    as_of_dynamic_match = AS_OF_DYNAMIC_RE.search(question_text)
    math_context = task_type_lower == "math" or dataset_lower in {"gsm8k", "math", "competition_math"}
    dynamic_time = bool(time_match or as_of_dynamic_match)
    if dynamic_time and math_context and not as_of_dynamic_match:
        dynamic_time = False
        _add_evidence(evidence, "TIME_SENSITIVE", "question", "math context suppresses narrative time cue", 0.55)
    if dynamic_time and not fixed_time_anchor:
        tags["TIME_SENSITIVE"] = True
        _add_evidence(evidence, "TIME_SENSITIVE", "question", time_match.group(0) if time_match else "as of today/now", 0.9)
    if _truthy_metadata(metadata, "time_sensitive", "requires_current_info", "needs_current_info", "dynamic_fact"):
        tags["TIME_SENSITIVE"] = True
        _add_evidence(evidence, "TIME_SENSITIVE", "metadata", "metadata marks current/dynamic fact", 0.95)
    if fixed_time_anchor and not tags["TIME_SENSITIVE"]:
        _add_evidence(evidence, "TIME_SENSITIVE", "question", "fixed historical time anchor suppresses current-time tag", 0.6)

    knowledge_text = _metadata_text(metadata, "knowledge_type", "fact_type", "split", "category")
    if dataset_lower in {"mintqa", "mintqa-ti-v0.1"}:
        tags["NEW_OR_TAIL_KNOWLEDGE"] = True
        _add_evidence(evidence, "NEW_OR_TAIL_KNOWLEDGE", "metadata", f"dataset={dataset_lower}", 0.9)
    if any(term in knowledge_text for term in ("tail", "long-tail", "long tail", "new knowledge", "new", "multi-hop", "multihop")):
        tags["NEW_OR_TAIL_KNOWLEDGE"] = True
        _add_evidence(evidence, "NEW_OR_TAIL_KNOWLEDGE", "metadata", knowledge_text, 0.85)
    if metadata.get("graph_preview") or _as_int(metadata.get("graph_size")) > 0 or _as_int(metadata.get("num_hops")) > 1:
        tags["NEW_OR_TAIL_KNOWLEDGE"] = True
        _add_evidence(evidence, "NEW_OR_TAIL_KNOWLEDGE", "metadata", "graph/multi-hop metadata", 0.8)

    if _truthy_metadata(metadata, "false_premise"):
        tags["FALSE_PREMISE"] = True
        _add_evidence(evidence, "FALSE_PREMISE", "metadata", "false_premise=true", 0.95)
    false_match = FALSE_PREMISE_RE.search(question_text)
    if false_match:
        tags["FALSE_PREMISE"] = True
        _add_evidence(evidence, "FALSE_PREMISE", "question", false_match.group(0), 0.8)

    if dataset_lower == "truthfulqa" or _truthy_metadata(metadata, "misconception", "imitative_falsehood"):
        tags["MISCONCEPTION_RISK"] = True
        _add_evidence(evidence, "MISCONCEPTION_RISK", "metadata", "truthfulness/misconception metadata", 0.9)
    misconception_match = MISCONCEPTION_RE.search(question_text)
    if misconception_match and "prove that" not in question_lower:
        tags["MISCONCEPTION_RISK"] = True
        _add_evidence(evidence, "MISCONCEPTION_RISK", "question", misconception_match.group(0), 0.75)
    if tags["FALSE_PREMISE"]:
        _add_evidence(evidence, "MISCONCEPTION_RISK", "derived", "false premise can induce imitative correction risk", 0.4)

    if _missing_info_is_critical(metadata):
        tags["MISSING_INFO"] = True
        _add_evidence(evidence, "MISSING_INFO", "metadata", "critical missing detail/vague task metadata", 0.9)
    if re.search(r"\b(unspecified|underspecified|need more information|missing information|what .* should i)\b", question_text, re.IGNORECASE):
        tags["MISSING_INFO"] = True
        _add_evidence(evidence, "MISSING_INFO", "question", "explicit missing-information cue", 0.7)

    numeric_count = len(re.findall(r"\b\d+(?:\.\d+)?\b", question_text))
    has_expression = bool(CALC_EXPRESSION_RE.search(question_text) or CALC_EXPRESSION_RE.search(candidate_text))
    has_calc_word = bool(CALC_WORD_RE.search(question_text))
    has_relation = bool(MATH_RELATION_RE.search(question_text))
    metadata_calc = _truthy_metadata(metadata, "calculation_required", "requires_calculation", "needs_calculation")
    if metadata_calc or has_expression or (numeric_count >= 2 and (has_calc_word or has_relation)):
        tags["CALCULATION_REQUIRED"] = True
        source = "metadata" if metadata_calc else "question"
        detail = "metadata marks calculation required" if metadata_calc else "numeric relation/expression in question"
        _add_evidence(evidence, "CALCULATION_REQUIRED", source, detail, 0.85 if metadata_calc else 0.75)
    elif task_type_lower == "math" and reason_text and CALC_WORD_RE.search(reason_text) and numeric_count >= 2:
        tags["CALCULATION_REQUIRED"] = True
        _add_evidence(evidence, "CALCULATION_REQUIRED", "student_reasoning", "reasoning mentions concrete calculation", 0.35)

    cannot_verify_text = "cannot verify" in question_lower or _truthy_metadata(metadata, "cannot_verify", "unverifiable")
    tools_cannot_ground = (
        (tags["TIME_SENSITIVE"] or tags["NEW_OR_TAIL_KNOWLEDGE"] or bool(EXTERNAL_EVIDENCE_RE.search(question_text)))
        and not can_search
    )
    if tags["FALSE_PREMISE"] or _truthy_metadata(metadata, "unanswerable", "refuse", "should_refuse") or tools_cannot_ground:
        tags["JUSTIFIED_REFUSE"] = True
        _add_evidence(evidence, "JUSTIFIED_REFUSE", "derived", "false premise/unanswerable or no available grounding tool", 0.8)
    elif cannot_verify_text and not can_search:
        tags["JUSTIFIED_REFUSE"] = True
        _add_evidence(evidence, "JUSTIFIED_REFUSE", "metadata", "cannot verify and search unavailable", 0.65)

    external_needed = bool(EXTERNAL_EVIDENCE_RE.search(question_text)) or _truthy_metadata(
        metadata,
        "external_evidence_needed",
        "needs_search",
        "requires_search",
    )
    tags["SEARCH_REQUIRED"] = bool(tags["TIME_SENSITIVE"] or tags["NEW_OR_TAIL_KNOWLEDGE"] or external_needed)
    if tags["SEARCH_REQUIRED"]:
        _add_evidence(evidence, "SEARCH_REQUIRED", "derived", "fresh/tail knowledge or explicit external-evidence need", 0.9)

    tags["CALCULATE_REQUIRED"] = bool(tags["CALCULATION_REQUIRED"])
    if tags["CALCULATE_REQUIRED"]:
        _add_evidence(evidence, "CALCULATE_REQUIRED", "derived", "alias of CALCULATION_REQUIRED", 0.95)

    tags["CLARIFY_REQUIRED"] = bool(tags["MISSING_INFO"] and can_clarify)
    if tags["CLARIFY_REQUIRED"]:
        _add_evidence(evidence, "CLARIFY_REQUIRED", "derived", "critical missing info and clarify is available", 0.95)

    tags["TOOL_REQUIRED"] = bool(
        tags["SEARCH_REQUIRED"]
        or (tags["CALCULATION_REQUIRED"] and can_calculate)
        or tags["CLARIFY_REQUIRED"]
    )
    if tags["TOOL_REQUIRED"]:
        _add_evidence(evidence, "TOOL_REQUIRED", "derived", "summary of action-specific requirements", 0.8)

    active = [tag for tag in ALL_TAGS if tags.get(tag)]
    return {
        "semantic_tags": tags,
        "active_semantic_tags": active,
        "semantic_tag_evidence": {tag: values for tag, values in evidence.items() if values},
    }


def build_semantic_tag_details_from_state(
    *,
    question: str,
    metadata: dict[str, Any] | None = None,
    task_type: str | None = None,
    dataset: str | None = None,
    reason_prefix: str | None = None,
    history_prefix: list[dict[str, Any]] | None = None,
    candidate_action_input: dict[str, Any] | None = None,
    can_search: bool | None = None,
    can_calculate: bool | None = None,
    can_clarify: bool | None = None,
) -> dict[str, Any]:
    return infer_semantic_tag_details(
        question=question,
        metadata=metadata,
        task_type=task_type,
        dataset=dataset,
        reason_prefix=reason_prefix,
        history_prefix=history_prefix,
        candidate_action_input=candidate_action_input,
        can_search=can_search,
        can_calculate=can_calculate,
        can_clarify=can_clarify,
    )


def build_semantic_tags_from_state(
    *,
    question: str,
    metadata: dict[str, Any] | None = None,
    task_type: str | None = None,
    dataset: str | None = None,
    reason_prefix: str | None = None,
    history_prefix: list[dict[str, Any]] | None = None,
    candidate_action_input: dict[str, Any] | None = None,
    can_search: bool | None = None,
    can_calculate: bool | None = None,
    can_clarify: bool | None = None,
) -> dict[str, bool]:
    return build_semantic_tag_details_from_state(
        question=question,
        metadata=metadata,
        task_type=task_type,
        dataset=dataset,
        reason_prefix=reason_prefix,
        history_prefix=history_prefix,
        candidate_action_input=candidate_action_input,
        can_search=can_search,
        can_calculate=can_calculate,
        can_clarify=can_clarify,
    )["semantic_tags"]


def build_semantic_tag_details(sample, reason_prefix: str | None = None) -> dict[str, Any]:
    return build_semantic_tag_details_from_state(
        question=getattr(sample, "question", ""),
        metadata=getattr(sample, "metadata", {}) or {},
        task_type=getattr(sample, "task_type", None),
        dataset=getattr(sample, "dataset", None),
        reason_prefix=reason_prefix,
        can_search=getattr(sample, "can_search", None),
        can_calculate=getattr(sample, "can_calculate", None),
        can_clarify=getattr(sample, "can_clarify", None),
    )


def build_semantic_tags(sample) -> dict[str, bool]:
    return build_semantic_tag_details(sample)["semantic_tags"]


def active_semantic_tags(sample) -> list[str]:
    return build_semantic_tag_details(sample)["active_semantic_tags"]


def active_semantic_tags_from_state(
    *,
    question: str,
    metadata: dict[str, Any] | None = None,
    task_type: str | None = None,
    dataset: str | None = None,
    reason_prefix: str | None = None,
    history_prefix: list[dict[str, Any]] | None = None,
    can_search: bool | None = None,
    can_calculate: bool | None = None,
    can_clarify: bool | None = None,
) -> list[str]:
    tags = build_semantic_tag_details_from_state(
        question=question,
        metadata=metadata,
        task_type=task_type,
        dataset=dataset,
        reason_prefix=reason_prefix,
        history_prefix=history_prefix,
        can_search=can_search,
        can_calculate=can_calculate,
        can_clarify=can_clarify,
    )
    return tags["active_semantic_tags"]
