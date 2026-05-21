from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROMPT_ROOT = Path(__file__).resolve().parents[1] / "prompts"
ACTION_PAYLOAD_KEYS = {
    "ANSWER": "answer",
    "SEARCH": "query",
    "CALCULATE": "expression",
    "CLARIFY": "question",
    "REFUSE": "reason",
}


def _dataset_key(dataset: str | None) -> str:
    key = str(dataset or "").strip().lower().replace("-", "_")
    aliases = {
        "competition_math": "math",
        "orbench": "or_bench",
        "or-bench": "or_bench",
    }
    return aliases.get(key, key or "generic")


def _normalized_actions(allowed_actions: list[str]) -> list[str]:
    normalized_actions = []
    for action in allowed_actions:
        action = str(action).upper()
        if action and action not in normalized_actions:
            normalized_actions.append(action)
    if "ANSWER" not in normalized_actions:
        normalized_actions.insert(0, "ANSWER")
    return normalized_actions


def read_dataset_system_prompt(dataset: str | None) -> str:
    path = PROMPT_ROOT / f"student_system_{_dataset_key(dataset)}.md"
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    return read_action_decision_prompt()


def read_action_decision_prompt() -> str:
    return (PROMPT_ROOT / "student_action_decision.md").read_text(encoding="utf-8").strip()


def read_decision_window_prompt() -> str:
    return (PROMPT_ROOT / "student_decision_window.md").read_text(encoding="utf-8").strip()


def read_dataset_finalize_prompt(dataset: str | None) -> str:
    key = _dataset_key(dataset)
    path = PROMPT_ROOT / f"student_finalize_{key}.md"
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    if key == "gsm8k":
        math_path = PROMPT_ROOT / "student_finalize_math.md"
        if math_path.exists():
            return math_path.read_text(encoding="utf-8").strip()
    path = PROMPT_ROOT / "student_finalize_after_tool.md"
    return path.read_text(encoding="utf-8").strip()


def _payload_rules(allowed_actions: list[str]) -> str:
    lines = ["Action payload rules for the currently allowed actions:"]
    for action in _normalized_actions(allowed_actions):
        key = ACTION_PAYLOAD_KEYS.get(action)
        if not key:
            continue
        if action == "ANSWER":
            lines.append(f"- ANSWER: use non-empty `action_input.{key}` containing the complete final response.")
        elif action == "SEARCH":
            lines.append(f"- SEARCH: use non-empty `action_input.{key}` containing a specific query with key entities and relation.")
        elif action == "CALCULATE":
            lines.append(f"- CALCULATE: use non-empty `action_input.{key}` containing executable Python math, not natural language.")
        elif action == "CLARIFY":
            lines.append(f"- CLARIFY: use non-empty `action_input.{key}` containing one direct question for missing critical information.")
        elif action == "REFUSE":
            lines.append(f"- REFUSE: use non-empty `action_input.{key}` explaining the safety reason.")
    return "\n".join(lines)


def _allowed_action_union(allowed_actions: list[str]) -> str:
    return "|".join(_normalized_actions(allowed_actions))


def _schema_action_input_example(allowed_actions: list[str]) -> str:
    allowed = _normalized_actions(allowed_actions)
    if "ANSWER" in allowed:
        return '{"answer": "final answer text"}'
    action = allowed[0] if allowed else "ANSWER"
    key = ACTION_PAYLOAD_KEYS.get(action, "answer")
    return json.dumps({key: f"{key} text"}, ensure_ascii=False)


def build_rollout_system_prompt(
    *,
    dataset: str | None,
    allowed_actions: list[str],
    effective_top_k: int,
) -> str:
    allowed = _normalized_actions(allowed_actions)
    return "\n\n".join(
        [
            read_dataset_system_prompt(dataset),
            "Rollout mode: propose a ranked action set for boundary mining.",
            "\n".join(
                [
                    "Hard rules:",
                    f"1. Return exactly {effective_top_k} candidates with different action types.",
                    "2. One candidate must be ANSWER, even if it is ranked last or has low confidence.",
                    f"3. Candidate actions must come only from the current allowed actions: {', '.join(allowed)}.",
                    "4. Output valid JSON only. No prose before or after the JSON.",
                    "5. Every candidate must include `rank`, `action`, `confidence`, `brief_rationale`, and `action_input`.",
                    "6. Do not use REFUSE for missing evidence, tool failure, uncertainty, unsupported claims, or false premises unless the user asks for unsafe or harmful compliance.",
                ]
            ),
            _payload_rules(allowed),
            "\n".join(
                [
                    "Return JSON with this schema:",
                    "{",
                    '  "reasoning": {',
                    '    "attempt": "short task-specific reasoning",',
                    '    "uncertainty_summary": "what remains uncertain",',
                    '    "need_external_help": true',
                    "  },",
                    '  "candidates": [',
                    f'    {{"rank": 1, "action": "{_allowed_action_union(allowed)}", "confidence": 0.0, "brief_rationale": "", "action_input": {{}}}}',
                    "  ]",
                    "}",
                    f"The actual `candidates` array must contain exactly {effective_top_k} objects with different allowed actions.",
                ]
            ),
        ]
    )


def build_action_decision_system_prompt(
    *,
    dataset: str | None,
    allowed_actions: list[str],
) -> str:
    allowed = _normalized_actions(allowed_actions)
    return "\n\n".join(
        [
            read_dataset_system_prompt(dataset),
            "Single-action mode: choose exactly one action from the current allowed actions.",
            "\n".join(
                [
                    "Hard rules:",
                    f"1. The action must be one of: {', '.join(allowed)}.",
                    "2. Output valid JSON only. No prose before or after the JSON.",
                    "3. Include both `reasoning` and `decision`.",
                    "4. In `decision`, fields must appear as `action`, `confidence`, `brief_rationale`, `action_input`.",
                    "5. Do not use REFUSE for missing evidence, tool failure, uncertainty, unsupported claims, or false premises unless the user asks for unsafe or harmful compliance.",
                ]
            ),
            _payload_rules(allowed),
            "\n".join(
                [
                    "Return JSON with this schema:",
                    "{",
                    '  "reasoning": {',
                    '    "attempt": "short but substantive reasoning",',
                    '    "uncertainty_summary": "what remains uncertain",',
                    '    "need_external_help": true',
                    "  },",
                    '  "decision": {',
                    f'    "action": "{_allowed_action_union(allowed)}",',
                    '    "confidence": 0.0,',
                    '    "brief_rationale": "",',
                    '    "action_input": {}',
                    "  }",
                    "}",
                ]
            ),
        ]
    )


def build_decision_window_system_prompt(
    *,
    dataset: str | None,
    allowed_actions: list[str],
) -> str:
    allowed = _normalized_actions(allowed_actions)
    return "\n\n".join(
        [
            read_dataset_system_prompt(dataset),
            "Decision-window mode: treat the original student reasoning attempt as fixed state. Do not rewrite it.",
            "\n".join(
                [
                    "Hard rules:",
                    f"1. Choose exactly one action from the current allowed actions: {', '.join(allowed)}.",
                    "2. If an action is not listed as currently allowed, it is forbidden.",
                    "3. Output valid JSON only. No prose before or after the JSON.",
                    "4. The JSON root must contain exactly `action`, `confidence`, `brief_rationale`, and `action_input`.",
                    "5. Do not output `reasoning`, `decision`, `candidates`, markdown fences, or explanatory text.",
                    "6. Do not use REFUSE for missing evidence, tool failure, uncertainty, unsupported claims, or false premises unless the user asks for unsafe or harmful compliance.",
                ]
            ),
            _payload_rules(allowed),
            "\n".join(
                [
                    "Return exactly this root-only schema:",
                    "{",
                    f'  "action": "{_allowed_action_union(allowed)}",',
                    '  "confidence": 0.0,',
                    '  "brief_rationale": "",',
                    '  "action_input": {}',
                    "}",
                ]
            ),
        ]
    )


def build_finalize_after_tool_system_prompt(
    *,
    dataset: str | None,
    allowed_actions: list[str],
) -> str:
    allowed = _normalized_actions(allowed_actions)
    return "\n\n".join(
        [
            read_dataset_system_prompt(dataset),
            read_dataset_finalize_prompt(dataset),
            "Finalize-after-tool mode: use the prior action history and current tool observation to choose the final next action.",
            "\n".join(
                [
                    "Hard rules:",
                    "1. Use the previous history and current observation directly; do not invent additional facts.",
                    "2. If the observation is sufficient, choose ANSWER and put the final response in `action_input.answer`.",
                    f"3. Choose only from the allowed final actions: {', '.join(allowed)}.",
                    "4. If the observation shows a tool failure, missing evidence, or unsupported premise, do not use REFUSE for that reason.",
                    "5. REFUSE is only for unsafe, harmful, or disallowed user requests, and only when REFUSE is listed as an allowed final action.",
                    "6. Output valid JSON only. No markdown fences or prose before or after the JSON.",
                    "7. Include both `reasoning` and `final_decision`.",
                    "8. In `final_decision`, fields must appear as `action`, `confidence`, `brief_rationale`, `action_input`.",
                ]
            ),
            _payload_rules(allowed),
            "\n".join(
                [
                    "Return JSON with this schema:",
                    "{",
                    '  "reasoning": {',
                    '    "attempt": "brief reasoning over the prior history and current observation",',
                    '    "observation_summary": "what the observation establishes or fails to establish",',
                    '    "remaining_uncertainty": "what, if anything, remains uncertain"',
                    "  },",
                    '  "final_decision": {',
                    f'    "action": "{_allowed_action_union(allowed)}",',
                    '    "confidence": 0.0,',
                    '    "brief_rationale": "",',
                    f'    "action_input": {_schema_action_input_example(allowed)}',
                    "  }",
                    "}",
                ]
            ),
        ]
    )


def build_action_decision_user_block(
    *,
    question: str,
    allowed_actions: list[str],
    can_clarify: bool | None = None,
) -> str:
    normalized_actions = []
    for action in allowed_actions:
        action = str(action).upper()
        if action and action not in normalized_actions:
            normalized_actions.append(action)
    if "ANSWER" not in normalized_actions:
        normalized_actions.insert(0, "ANSWER")
    constraints = [
        f"Question: {question}",
        "",
        "Current allowed actions:",
        *[f"- {action}" for action in normalized_actions],
    ]
    if can_clarify is not None:
        constraints.extend(["", f"Clarify allowed: {str(bool(can_clarify))}"])
    return "\n".join(constraints)


def build_decision_window_user_block(
    *,
    question: str,
    allowed_actions: list[str],
    reasoning_attempt: str,
    can_clarify: bool | None = None,
) -> str:
    base = build_action_decision_user_block(
        question=question,
        allowed_actions=allowed_actions,
        can_clarify=can_clarify,
    )
    return "\n".join(
        [
            base,
            "",
            "Original student reasoning attempt (fixed state):",
            str(reasoning_attempt or "").strip(),
            "",
            "Decision-window mode: choose the next action from this fixed state.",
        ]
    )


def build_finalize_after_tool_user_block(
    *,
    question: str,
    allowed_actions: list[str],
    current_action: str,
    current_action_input: dict[str, Any],
    current_observation: dict[str, Any],
    history: list[dict[str, Any]] | None = None,
    trace: str | list[dict[str, Any]] | None = None,
) -> str:
    normalized_actions = _normalized_actions(allowed_actions)
    if isinstance(trace, str):
        trace_text = trace
    else:
        trace_text = json.dumps(trace or [], ensure_ascii=False)
    return "\n".join(
        [
            f"Original question: {question}",
            "",
            "Allowed final actions for this example:",
            *[f"- {action}" for action in normalized_actions],
            "",
            f"Previous action/observation history: {json.dumps(history or [], ensure_ascii=False)}",
            f"Prior finalize attempts: {trace_text}",
            f"Current action taken: {str(current_action).upper()}",
            f"Current action input: {json.dumps(current_action_input or {}, ensure_ascii=False)}",
            f"Current tool observation: {json.dumps(current_observation or {}, ensure_ascii=False)}",
            "",
            "Finalize-after-tool mode: return JSON with reasoning and final_decision only.",
        ]
    )


def build_action_decision_prompt_text(
    *,
    question: str,
    allowed_actions: list[str],
    can_clarify: bool | None = None,
    dataset: str | None = None,
) -> str:
    return (
        build_action_decision_system_prompt(dataset=dataset, allowed_actions=allowed_actions)
        + "\n\n"
        + build_action_decision_user_block(
            question=question,
            allowed_actions=allowed_actions,
            can_clarify=can_clarify,
        )
    )


def build_decision_window_prompt_text(
    *,
    question: str,
    allowed_actions: list[str],
    reasoning_attempt: str,
    can_clarify: bool | None = None,
    dataset: str | None = None,
) -> str:
    return (
        build_decision_window_system_prompt(dataset=dataset, allowed_actions=allowed_actions)
        + "\n\n"
        + build_decision_window_user_block(
            question=question,
            allowed_actions=allowed_actions,
            reasoning_attempt=reasoning_attempt,
            can_clarify=can_clarify,
        )
    )


def build_finalize_after_tool_prompt_text(
    *,
    question: str,
    allowed_actions: list[str],
    current_action: str,
    current_action_input: dict[str, Any],
    current_observation: dict[str, Any],
    history: list[dict[str, Any]] | None = None,
    trace: str | list[dict[str, Any]] | None = None,
    dataset: str | None = None,
) -> str:
    return (
        build_finalize_after_tool_system_prompt(dataset=dataset, allowed_actions=allowed_actions)
        + "\n\n"
        + build_finalize_after_tool_user_block(
            question=question,
            allowed_actions=allowed_actions,
            current_action=current_action,
            current_action_input=current_action_input,
            current_observation=current_observation,
            history=history,
            trace=trace,
        )
    )


def build_action_decision_prompt_messages(
    *,
    question: str,
    allowed_actions: list[str],
    can_clarify: bool | None = None,
    dataset: str | None = None,
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": build_action_decision_system_prompt(dataset=dataset, allowed_actions=allowed_actions)},
        {
            "role": "user",
            "content": build_action_decision_user_block(
                question=question,
                allowed_actions=allowed_actions,
                can_clarify=can_clarify,
            ),
        },
    ]


def build_decision_window_prompt_messages(
    *,
    question: str,
    allowed_actions: list[str],
    reasoning_attempt: str,
    can_clarify: bool | None = None,
    dataset: str | None = None,
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": build_decision_window_system_prompt(dataset=dataset, allowed_actions=allowed_actions)},
        {
            "role": "user",
            "content": build_decision_window_user_block(
                question=question,
                allowed_actions=allowed_actions,
                reasoning_attempt=reasoning_attempt,
                can_clarify=can_clarify,
            ),
        },
    ]


def build_decision_window_completion(
    *,
    action: str,
    action_input: dict[str, Any],
    confidence: float | None,
    brief_rationale: str = "",
) -> str:
    action = str(action).upper()
    try:
        confidence_value = None if confidence is None else max(0.0, min(1.0, float(confidence)))
    except (TypeError, ValueError):
        confidence_value = None
    payload = {
        "action": action,
        "confidence": 0.5 if confidence_value is None else confidence_value,
        "brief_rationale": str(brief_rationale or "").strip(),
        "action_input": action_input or {},
    }
    return json.dumps(payload, ensure_ascii=False)


def build_action_decision_completion(
    *,
    action: str,
    action_input: dict[str, Any],
    confidence: float | None,
    reasoning_attempt: str,
    uncertainty_summary: str = "",
    need_external_help: bool | None = None,
    brief_rationale: str = "",
) -> str:
    action = str(action).upper()
    if need_external_help is None:
        need_external_help = action in {"SEARCH", "CALCULATE", "CLARIFY"}
    try:
        confidence_value = None if confidence is None else max(0.0, min(1.0, float(confidence)))
    except (TypeError, ValueError):
        confidence_value = None
    payload = {
        "reasoning": {
            "attempt": str(reasoning_attempt or brief_rationale or "").strip(),
            "uncertainty_summary": str(uncertainty_summary or "").strip(),
            "need_external_help": bool(need_external_help),
        },
        "decision": {
            "action": action,
            "confidence": 0.5 if confidence_value is None else confidence_value,
            "brief_rationale": str(brief_rationale or reasoning_attempt or "").strip(),
            "action_input": action_input or {},
        },
    }
    return json.dumps(payload, ensure_ascii=False)
