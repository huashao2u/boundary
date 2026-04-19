from __future__ import annotations

import argparse
import json
import re
from typing import Any

from mcagent_core.data.loaders import UnifiedSample


ALLOWED_ACTIONS = ("ANSWER", "SEARCH", "CALCULATE", "CLARIFY", "REFUSE")


def build_system_prompt(enable_tool_schema: bool = True) -> str:
    schema_hint = (
        "Return valid JSON only. The `decision.action` must be one of "
        "`ANSWER`, `SEARCH`, `CALCULATE`, `CLARIFY`, `REFUSE`."
    )
    tool_hint = (
        "Use `SEARCH` for external evidence, `CALCULATE` for arithmetic or symbolic computation, "
        "`CLARIFY` for missing critical information, and `REFUSE` when the premise is false or the request "
        "cannot be responsibly completed with the available tools."
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

    decision = parsed.get("decision", parsed)
    action = str(decision.get("action", "ANSWER")).upper()
    if action not in ALLOWED_ACTIONS:
        action = _keyword_fallback_action(raw_text)
    action_input = decision.get("action_input", {}) or {}
    if not isinstance(action_input, dict):
        action_input = {}
    if action == "ANSWER" and "answer" not in action_input:
        action_input = {"answer": ""}
    if action == "SEARCH" and "query" not in action_input:
        action_input = {"query": ""}
    if action == "CALCULATE" and "expression" not in action_input:
        action_input = {"expression": ""}
    if action == "CLARIFY" and "question" not in action_input:
        action_input = {"question": ""}
    if action == "REFUSE" and "reason" not in action_input:
        action_input = {"reason": ""}
    confidence = decision.get("confidence")
    try:
        confidence = None if confidence is None else max(0.0, min(1.0, float(confidence)))
    except (TypeError, ValueError):
        confidence = None
    return {
        "reason": str(parsed.get("reason", "")).strip(),
        "decision": {
            "action": action,
            "confidence": confidence,
            "action_input": action_input,
            "brief_rationale": str(decision.get("brief_rationale", "")).strip(),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build or parse MCAgent prompts.")
    parser.add_argument("--raw-text", help="Raw model text to parse.")
    args = parser.parse_args()
    if not args.raw_text:
        raise SystemExit("--raw-text is required for CLI parsing.")
    print(json.dumps(parse_decision_output(args.raw_text), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
