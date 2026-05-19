from __future__ import annotations

import argparse
import ast
import json
import logging
import re
import warnings
from typing import Any

from mcagent_core.data.loaders import UnifiedSample


ALLOWED_ACTIONS = ("ANSWER", "SEARCH", "CALCULATE", "CLARIFY", "REFUSE")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Diagnostics counters (module-level, per process lifetime)
# ---------------------------------------------------------------------------
_DIAG_COUNTS: dict[str, int] = {
    "parsed_ok": 0,
    "invalid_schema": 0,
    "invalid_missing_answer": 0,
    "invalid_single_action": 0,
}


def get_diagnostics_counts() -> dict[str, int]:
    """Return a copy of the module-level parse diagnostics counters."""
    return dict(_DIAG_COUNTS)


# DEPRECATED(mainline): the legacy single-decision prompt helpers below are not
# used by current boundary v0.2 rollouts. The active student prompt is
# mcagent_boundary.rollout.branch_actions.build_student_prompt plus
# prompts/student_rollout.md, and parse_candidate_output() is the active parser.


def build_system_prompt(enable_tool_schema: bool = True) -> str:
    schema_hint = (
        "Return valid JSON only. The `decision.action` must be one of "
        "`ANSWER`, `SEARCH`, `CALCULATE`, `CLARIFY`, `REFUSE`."
    )
    tool_hint = (
        "Use `SEARCH` for external evidence, `CALCULATE` for arithmetic or symbolic computation, "
        "`CLARIFY` for missing critical information, and `REFUSE` only when the request itself is unsafe, "
        "harmful, or disallowed. Do not refuse merely because evidence is missing, a premise is false, or a tool failed."
    )
    if not enable_tool_schema:
        tool_hint = "Still return the same JSON schema."
    return "\n".join(
        [
            "You are MCAgent, an action-calibrated assistant.",
            schema_hint,
            tool_hint,
            "JSON schema:",
            '{',
            '  "reason": "brief reasoning",',
            '  "decision": {',
            '    "action": "ANSWER|SEARCH|CALCULATE|CLARIFY|REFUSE",',
            '    "confidence": 0.0,',
            '    "action_input": {},',
            '    "brief_rationale": "why this action is appropriate"',
            "  }",
            "}",
        ]
    )


def build_user_prompt(
    sample: UnifiedSample,
    observations: list[dict[str, Any]] | None = None,
    state_tags: list[str] | None = None,
    reason_prefix: str | None = None,
) -> str:
    def _json_default(obj: Any):
        # Make pandas/numpy-derived metadata JSON-safe without leaking huge blobs into prompts.
        if hasattr(obj, "tolist"):
            try:
                return obj.tolist()
            except Exception:
                return str(obj)
        if hasattr(obj, "item"):
            try:
                return obj.item()
            except Exception:
                return str(obj)
        return str(obj)

    def _compact(value: Any, *, max_str: int = 160, max_list: int = 4, depth: int = 0) -> Any:
        if depth >= 2:
            return str(value)
        if value is None or isinstance(value, (bool, int, float)):
            return value
        if isinstance(value, str):
            return value if len(value) <= max_str else value[:max_str] + "...(truncated)"
        if isinstance(value, dict):
            compacted = {}
            for key, item in list(value.items())[:20]:
                if str(key) in {"graph", "source", "missing_details", "raw_answer"}:
                    compacted[str(key)] = _compact(str(item), max_str=max_str, max_list=max_list, depth=depth + 1)
                    continue
                compacted[str(key)] = _compact(item, max_str=max_str, max_list=max_list, depth=depth + 1)
            if len(value) > 20:
                compacted["_truncated_keys"] = len(value) - 20
            return compacted
        if isinstance(value, (list, tuple)):
            compacted = [_compact(item, max_str=max_str, max_list=max_list, depth=depth + 1) for item in list(value)[:max_list]]
            if len(value) > max_list:
                compacted.append(f"...(truncated {len(value) - max_list} items)")
            return compacted
        return value

    compact_metadata = _compact(sample.metadata or {})
    state_tag_block = "" if not state_tags else "\nState tags: " + json.dumps(state_tags, ensure_ascii=False)
    reason_prefix_block = "" if not reason_prefix else "\nReason prefix: " + reason_prefix.strip()
    observation_block = ""
    if observations:
        observation_lines = ["Tool observations:"]
        for index, obs in enumerate(observations, start=1):
            observation_lines.append(f"{index}. {json.dumps(_compact(obs), ensure_ascii=False, default=_json_default)}")
        observation_block = "\n" + "\n".join(observation_lines)
    return (
        f"Dataset: {sample.dataset}\n"
        f"Task type: {sample.task_type}\n"
        f"Question: {sample.question}\n"
        f"Metadata: {json.dumps(compact_metadata, ensure_ascii=False, default=_json_default)}"
        f"{state_tag_block}"
        f"{reason_prefix_block}"
        f"{observation_block}\n"
        "Choose the best next action and return JSON only."
    )


def build_prompt_text(
    sample: UnifiedSample,
    enable_tool_schema: bool = True,
    observations: list[dict[str, Any]] | None = None,
    state_tags: list[str] | None = None,
    reason_prefix: str | None = None,
) -> str:
    return build_system_prompt(enable_tool_schema=enable_tool_schema) + "\n\n" + build_user_prompt(
        sample,
        observations=observations,
        state_tags=state_tags,
        reason_prefix=reason_prefix,
    )


def build_state_prompt(prompt_text: str, reason_prefix: str) -> str:
    return (
        prompt_text
        + "\nResponse JSON prefix:\n"
        + json.dumps({"reason": reason_prefix}, ensure_ascii=False)[:-1]
        + ', "decision": {"action": "'
    )


def _keyword_fallback_action(raw_text: str) -> str:
    upper = raw_text.upper()
    for action in ALLOWED_ACTIONS:
        if action in upper:
            return action
    return "ANSWER"


def parse_decision_output(raw_text: str) -> dict[str, Any]:
    """Parse legacy single-decision output.

    DEPRECATED(mainline): retained only to preserve diagnostics when a model
    ignores the v0.2 top-k candidate schema. Such records are excluded from
    mining/pair construction as invalid_candidate_output.
    """
    parsed: dict[str, Any] | None = None
    try:
        from json_repair import repair_json

        candidate = repair_json(raw_text, return_objects=True)
        if isinstance(candidate, dict):
            parsed = candidate
    except Exception:
        parsed = None

    if parsed is None:
        match = re.search(r"\{.*\}", raw_text, flags=re.DOTALL)
        if match:
            try:
                parsed = json.loads(match.group(0))
            except json.JSONDecodeError:
                parsed = None

    if parsed is None:
        action = _keyword_fallback_action(raw_text)
        return {
            "reason": raw_text.strip(),
            "decision": {
                "action": action,
                "confidence": None,
                "action_input": {},
                "brief_rationale": "Fallback parser selected the most likely action keyword.",
            },
        }

    reasoning_raw = parsed.get("reasoning", {})
    if not isinstance(reasoning_raw, dict):
        reasoning_raw = {}
    decision = parsed.get("decision", parsed)
    action = str(decision.get("action", "ANSWER")).upper()
    if action not in ALLOWED_ACTIONS:
        action = _keyword_fallback_action(raw_text)
    action_input = decision.get("action_input", {}) or {}
    if not isinstance(action_input, dict):
        action_input = {}
    action_input = _minimal_repair_action_input(action, action_input)
    confidence = decision.get("confidence")
    try:
        confidence = None if confidence is None else max(0.0, min(1.0, float(confidence)))
    except (TypeError, ValueError):
        confidence = None
    reason = str(reasoning_raw.get("attempt") or parsed.get("reason", "")).strip()
    return {
        "reason": reason,
        "uncertainty_summary": str(reasoning_raw.get("uncertainty_summary", "")).strip(),
        "need_external_help": bool(reasoning_raw.get("need_external_help", False)),
        "decision": {
            "action": action,
            "confidence": confidence,
            "action_input": action_input,
            "brief_rationale": str(decision.get("brief_rationale", "")).strip(),
        },
    }


# ---------------------------------------------------------------------------
# v0.2 top-k candidate parser
# ---------------------------------------------------------------------------

_REQUIRED_ACTION_INPUT_KEYS = {
    "ANSWER": "answer",
    "SEARCH": "query",
    "CALCULATE": "expression",
    "CLARIFY": "question",
    "REFUSE": "reason",
}


def _minimal_repair_action_input(action: str, action_input: Any) -> dict[str, Any]:
    """Return a dict-shaped action_input without making empty fields trainable.

    Canonicalization and validity live in mcagent_boundary.rollout.candidate_schema
    after parsing. The parser only preserves what the student actually supplied.
    """
    if not isinstance(action_input, dict):
        return {}
    return dict(action_input)


def _parse_jsonish(value: str) -> dict[str, Any] | None:
    try:
        from json_repair import repair_json

        repaired = repair_json(value, return_objects=True)
        return repaired if isinstance(repaired, dict) else None
    except Exception:
        pass
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        pass
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", SyntaxWarning)
            parsed = ast.literal_eval(value)
        return parsed if isinstance(parsed, dict) else None
    except (SyntaxError, TypeError, ValueError):
        return None


def _balanced_object_spans(raw_text: str) -> list[str]:
    spans: list[str] = []
    depth = 0
    start: int | None = None
    in_string = False
    escape = False
    quote = ""
    for index, char in enumerate(raw_text):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == quote:
                in_string = False
            continue
        if char in {'"', "'"}:
            in_string = True
            quote = char
            continue
        if char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                spans.append(raw_text[start : index + 1])
                start = None
    return spans


def _candidate_payloads_from_text(raw_text: str, *, top_k: int = 3) -> dict[str, Any] | None:
    """Recover top-k candidate JSON from common model wrapper formats."""
    candidates: list[dict[str, Any]] = []
    # Free-form candidate snippets are only trusted when the model did not echo
    # the full prompt; otherwise the one-shot example can be mistaken for output.
    if "Hard rules" in raw_text or "Return JSON with the following schema" in raw_text:
        return None
    tail = raw_text
    marker = re.search(r"candidates?\s*:", raw_text, flags=re.IGNORECASE)
    if marker:
        tail = raw_text[marker.end() :]
    for span in _balanced_object_spans(tail):
        parsed = _parse_jsonish(span)
        if not isinstance(parsed, dict):
            continue
        if str(parsed.get("action", "")).upper() in ALLOWED_ACTIONS:
            candidates.append(parsed)
    required_min = min(2, max(1, int(top_k)))
    if len(candidates) < required_min:
        return None
    return {
        "reasoning": {
            "attempt": raw_text[: marker.start()].strip() if marker else "",
            "uncertainty_summary": "",
            "need_external_help": False,
        },
        "candidates": candidates,
    }


def _student_response_region(raw_text: str) -> str:
    marker = "Now respond for the user's question below"
    marker_index = raw_text.rfind(marker)
    if marker_index == -1:
        return raw_text
    return raw_text[marker_index + len(marker) :]


def _candidate_output_payload(raw_text: str, *, top_k: int = 3) -> dict[str, Any] | None:
    response_text = _student_response_region(raw_text)
    parsed = _parse_jsonish(response_text)
    if isinstance(parsed, dict) and isinstance(parsed.get("candidates"), list):
        return parsed

    fenced = re.findall(r"```(?:json)?\s*(.*?)```", response_text, flags=re.DOTALL | re.IGNORECASE)
    for block in reversed(fenced):
        parsed = _parse_jsonish(block)
        if isinstance(parsed, dict) and isinstance(parsed.get("candidates"), list):
            return parsed

    # Prefer complete objects with a candidates list, scanning from the end in
    # case an instruction example was echoed before the actual answer.
    for span in reversed(_balanced_object_spans(response_text)):
        parsed = _parse_jsonish(span)
        if isinstance(parsed, dict) and isinstance(parsed.get("candidates"), list):
            return parsed

    return _candidate_payloads_from_text(response_text, top_k=top_k)


def _enforce_answer_in_topk(candidates: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    if "ANSWER" not in [candidate["action"] for candidate in candidates]:
        raise ValueError("invalid_missing_answer")

    top = list(candidates[:top_k])
    if "ANSWER" not in [candidate["action"] for candidate in top]:
        answer_candidate = next(candidate for candidate in candidates if candidate["action"] == "ANSWER")
        if len(top) < top_k:
            top.append(answer_candidate)
        else:
            top[-1] = answer_candidate
    return [{**candidate, "rank": rank} for rank, candidate in enumerate(top, start=1)]


def parse_candidate_output(raw_text: str, top_k: int = 3) -> dict[str, Any] | None:
    """Parse a v0.2 top-k candidate response from the student model.

    Returns a dict with keys ``reasoning`` and ``candidates`` on success.
    Returns ``None`` and writes a diagnostics entry on parse failure.

    Rules enforced (§1 / §1a / §Schema):
    - Unique action types in candidates; list len 2–top_k.
    - ANSWER must be present.
    - Degenerate single-action-type → None + ``invalid_single_action``.
    - ANSWER missing           → None + ``invalid_missing_answer``.
    - JSON parse/schema error  → None + ``invalid_schema``.
    - action_input minimally repaired; gold never injected.
    - Ranks normalized to 1..top_k after dedup/sort.
    """
    global _DIAG_COUNTS
    top_k = max(1, int(top_k))
    required_min = min(2, top_k)
    parsed = _candidate_output_payload(raw_text, top_k=top_k)

    if parsed is None or not isinstance(parsed, dict):
        _DIAG_COUNTS["invalid_schema"] += 1
        logger.debug("parse_candidate_output: JSON parse failure. raw=%r", raw_text[:200])
        return None

    candidates_raw = parsed.get("candidates")
    if not isinstance(candidates_raw, list) or len(candidates_raw) == 0:
        _DIAG_COUNTS["invalid_schema"] += 1
        logger.debug("parse_candidate_output: missing/empty candidates list")
        return None

    seen_actions: dict[str, dict[str, Any]] = {}
    for item in candidates_raw:
        if not isinstance(item, dict):
            continue
        action = str(item.get("action", "")).strip().upper()
        if action not in ALLOWED_ACTIONS:
            continue
        if action in seen_actions:
            continue
        confidence = item.get("confidence")
        try:
            confidence = None if confidence is None else max(0.0, min(1.0, float(confidence)))
        except (TypeError, ValueError):
            confidence = None
        action_input = _minimal_repair_action_input(action, item.get("action_input", {}))
        seen_actions[action] = {
            "action": action,
            "confidence": confidence,
            "action_input": action_input,
            "brief_rationale": str(item.get("brief_rationale", "")).strip(),
        }

    if len(seen_actions) < required_min:
        _DIAG_COUNTS["invalid_single_action"] += 1
        logger.debug("parse_candidate_output: fewer than 2 unique action types: %s", list(seen_actions.keys()))
        return None

    if "ANSWER" not in seen_actions:
        _DIAG_COUNTS["invalid_missing_answer"] += 1
        logger.debug("parse_candidate_output: ANSWER missing from candidates")
        return None

    ordered = []
    for item in candidates_raw:
        if not isinstance(item, dict):
            continue
        action = str(item.get("action", "")).strip().upper()
        if action in seen_actions and action not in [c["action"] for c in ordered]:
            ordered.append(seen_actions[action])

    try:
        candidates = _enforce_answer_in_topk(ordered, top_k=top_k)
    except ValueError:
        _DIAG_COUNTS["invalid_missing_answer"] += 1
        logger.debug("parse_candidate_output: ANSWER missing from final top-k candidates")
        return None

    reasoning_raw = parsed.get("reasoning", {})
    if not isinstance(reasoning_raw, dict):
        reasoning_raw = {}
    reasoning = {
        "attempt": str(reasoning_raw.get("attempt", "")).strip(),
        "uncertainty_summary": str(reasoning_raw.get("uncertainty_summary", "")).strip(),
        "need_external_help": bool(reasoning_raw.get("need_external_help", False)),
    }

    _DIAG_COUNTS["parsed_ok"] += 1
    return {"reasoning": reasoning, "candidates": candidates}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build or parse MCAgent prompts.")
    parser.add_argument("--raw-text", help="Raw model text to parse.")
    args = parser.parse_args()
    if not args.raw_text:
        raise SystemExit("--raw-text is required for CLI parsing.")
    print(json.dumps(parse_decision_output(args.raw_text), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
