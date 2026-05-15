from __future__ import annotations

import json
from pathlib import Path
from typing import Any


PROMPT_ROOT = Path(__file__).resolve().parents[1] / "prompts"


def read_action_decision_prompt() -> str:
    return (PROMPT_ROOT / "student_action_decision.md").read_text(encoding="utf-8").strip()


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
            "Decision-window mode:",
            "- Do not rewrite the reasoning attempt.",
            "- Complete only the decision JSON object.",
            "- The JSON root must contain: action, confidence, brief_rationale, action_input.",
            "- Do not wrap the output in a reasoning or decision field.",
        ]
    )


def build_action_decision_prompt_text(
    *,
    question: str,
    allowed_actions: list[str],
    can_clarify: bool | None = None,
) -> str:
    return (
        read_action_decision_prompt()
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
) -> str:
    return (
        read_action_decision_prompt()
        + "\n\n"
        + build_decision_window_user_block(
            question=question,
            allowed_actions=allowed_actions,
            reasoning_attempt=reasoning_attempt,
            can_clarify=can_clarify,
        )
    )


def build_action_decision_prompt_messages(
    *,
    question: str,
    allowed_actions: list[str],
    can_clarify: bool | None = None,
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": read_action_decision_prompt()},
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
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": read_action_decision_prompt()},
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
