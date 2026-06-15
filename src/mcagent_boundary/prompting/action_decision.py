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


def _read_stage_prompt(
    stage: str,
    dataset: str | None = None,
    *,
    legacy_dataset_pattern: str | None = None,
    legacy_base_name: str | None = None,
) -> str:
    key = _dataset_key(dataset)
    path = PROMPT_ROOT / stage / f"{key}.md"
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    base_path = PROMPT_ROOT / stage / "base.md"
    if base_path.exists():
        return base_path.read_text(encoding="utf-8").strip()
    if legacy_dataset_pattern:
        legacy_path = PROMPT_ROOT / legacy_dataset_pattern.format(dataset=key)
        if legacy_path.exists():
            return legacy_path.read_text(encoding="utf-8").strip()
        legacy_path = PROMPT_ROOT / "legacy" / legacy_dataset_pattern.format(dataset=key)
        if legacy_path.exists():
            return legacy_path.read_text(encoding="utf-8").strip()
    if legacy_base_name:
        legacy_path = PROMPT_ROOT / legacy_base_name
        if legacy_path.exists():
            return legacy_path.read_text(encoding="utf-8").strip()
        legacy_path = PROMPT_ROOT / "legacy" / legacy_base_name
        if legacy_path.exists():
            return legacy_path.read_text(encoding="utf-8").strip()
    raise FileNotFoundError(f"No prompt found for stage={stage!r}, dataset={key!r}")


def read_dataset_system_prompt(dataset: str | None) -> str:
    path = PROMPT_ROOT / "legacy" / f"student_system_{_dataset_key(dataset)}.md"
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    return read_action_decision_prompt()


def read_dataset_rollout_system_prompt(dataset: str | None) -> str:
    return _read_stage_prompt(
        "rollout",
        dataset,
        legacy_dataset_pattern="student_rollout_system_{dataset}.md",
        legacy_base_name="student_rollout.md",
    )


def read_action_decision_prompt() -> str:
    return _read_stage_prompt("action_decision", legacy_base_name="student_action_decision.md")


def read_decision_window_prompt() -> str:
    return _read_stage_prompt("decision_window", legacy_base_name="student_decision_window.md")


def read_dataset_action_decision_prompt(dataset: str | None) -> str:
    return _read_stage_prompt(
        "action_decision",
        dataset,
        legacy_dataset_pattern="student_action_decision_system_{dataset}.md",
        legacy_base_name="student_action_decision.md",
    )


def read_dataset_decision_window_prompt(dataset: str | None) -> str:
    return _read_stage_prompt(
        "decision_window",
        dataset,
        legacy_dataset_pattern="student_decision_window_system_{dataset}.md",
        legacy_base_name="student_decision_window.md",
    )


def read_dataset_finalize_system_prompt(dataset: str | None) -> str:
    return _read_stage_prompt(
        "finalize",
        dataset,
        legacy_dataset_pattern="student_finalize_system_{dataset}.md",
        legacy_base_name="student_finalize_after_tool.md",
    )


def read_dataset_finalize_prompt(dataset: str | None) -> str:
    key = _dataset_key(dataset)
    path = PROMPT_ROOT / "legacy" / f"student_finalize_{key}.md"
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    if key == "gsm8k":
        math_path = PROMPT_ROOT / "legacy" / "student_finalize_math.md"
        if math_path.exists():
            return math_path.read_text(encoding="utf-8").strip()
    return _read_stage_prompt("finalize", legacy_base_name="student_finalize_after_tool.md")


def _payload_rules(allowed_actions: list[str]) -> str:
    lines = ["Action payload rules for the currently allowed actions:"]
    for action in _normalized_actions(allowed_actions):
        if action == "ANSWER":
            lines.extend(
                [
                    "",
                    "For ANSWER:",
                    "- `action_input.answer` must be non-empty and must contain the actual final response to the user.",
                    "- Do not put the final answer only in `brief_rationale`.",
                    "- Do not ask a clarification question inside ANSWER.",
                    "- Do not refuse inside ANSWER.",
                    "- Do not say that a tool is unavailable inside ANSWER.",
                    "- Do not output meta-comments such as \"I cannot search\" or \"I do not have access to tools\" inside ANSWER.",
                    "- For mathematical problems, `action_input.answer` must be the final numeric or symbolic answer only.",
                    "- For arithmetic word problems, do not include units such as dollars, people, items, or dozen unless the question explicitly asks for that unit.",
                    "- Do not output an unevaluated expression as ANSWER. Use CALCULATE for expressions when CALCULATE is allowed.",
                    "- If the request is underspecified or unverifiable and no better action is appropriate, provide a caveated direct response or state the limitation directly.",
                ]
            )
        elif action == "SEARCH":
            lines.extend(
                [
                    "",
                    "For SEARCH:",
                    "- `action_input.query` must be non-empty.",
                    "- The query must be specific and contain the key entities and relation from the question.",
                    "- Do not use vague queries like \"search for the answer\".",
                    "- Prefer SEARCH only when external, current, obscure, or entity-specific evidence would materially improve correctness.",
                ]
            )
        elif action == "CALCULATE":
            lines.extend(
                [
                    "",
                    "For CALCULATE:",
                    "- `action_input.expression` must be non-empty.",
                    "- The expression must be a restricted Python math snippet whose printed output or final expression is the numeric or symbolic answer.",
                    "- For a single expression, you may omit `print`; the tool will evaluate it as if it were printed.",
                    "- For multi-step calculations, use simple Python-style assignments and make the last line either the final expression or `print(final_value)`.",
                    "- Use Python syntax: `*` for multiplication, `**` for exponentiation, `/` for division, `%` for modulo, and parentheses for grouping.",
                    "- Supported helpers include `sum`, `range`, `min`, `max`, `abs`, `round`, and Python math/sympy forms such as `import math`, `import sympy as sp`, `from sympy import symbols, Eq, solve, sqrt, simplify`.",
                    "- For algebra, write executable sympy code rather than natural-language instructions, e.g. `x = symbols(\"x\"); solve(Eq(4*x + 5, 9), x)[0]`.",
                    "- Short bounded loops such as `for i in range(6): ...` are allowed when they are the clearest calculation.",
                    "- Do not include units, comments, explanations, file/network/system calls, unsafe imports, or natural-language instructions such as \"solve for x\".",
                    "- Convert units before writing the expression, e.g. two dozen should be written as `2 * 12`.",
                    "- If a task is very complex, it is acceptable to use CALCULATE as the next computation step and continue after the observation when another tool step is allowed.",
                ]
            )
        elif action == "CLARIFY":
            lines.extend(
                [
                    "",
                    "For CLARIFY:",
                    "- `action_input.question` must be non-empty.",
                    "- The question must ask for missing critical information that blocks a useful answer.",
                    "- Ask for the specific missing slot; do not ask a generic clarification question.",
                    "- Do not put clarification questions inside ANSWER.",
                    "- Prefer CLARIFY only when the missing information materially changes what the assistant should do.",
                ]
            )
        elif action == "REFUSE":
            lines.extend(
                [
                    "",
                    "For REFUSE:",
                    "- `action_input.reason` must be non-empty.",
                    "- The reason must explain why the user request itself is unsafe, harmful, or disallowed.",
                    "- Do not put refusal language inside ANSWER.",
                    "- REFUSE is only for unsafe, harmful, or disallowed user requests.",
                    "- Do not use REFUSE for missing evidence, tool failure, uncertainty, unsupported claims, or false premises.",
                ]
            )
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


def _payload_examples(allowed_actions: list[str]) -> str:
    examples = {
        "ANSWER": '{"answer": "final answer text"}',
        "SEARCH": '{"query": "targeted search query"}',
        "CALCULATE": '{"expression": "executable_python_or_sympy_expression"}',
        "CLARIFY": '{"question": "specific missing information question"}',
        "REFUSE": '{"reason": "brief safety reason"}',
    }
    lines = ["Non-empty `action_input` examples by selected action:"]
    for action in _normalized_actions(allowed_actions):
        if action in examples:
            lines.append(f"- {action}: {examples[action]}")
    return "\n".join(lines)


def _field_contract(
    *,
    allowed_actions: list[str],
    root_name: str = "JSON object",
    include_reasoning: bool = False,
    final_decision: bool = False,
) -> str:
    action_field = _allowed_action_union(allowed_actions)
    if include_reasoning and final_decision:
        return "\n".join(
            [
                "Return a JSON object with these fields:",
                "- `reasoning`: object with `attempt`, `observation_summary`, and `remaining_uncertainty`.",
                "- `final_decision`: object with `action`, `confidence`, `brief_rationale`, and non-empty `action_input`.",
                f"- `final_decision.action`: one of {action_field}.",
                "- `final_decision.action_input`: the non-empty payload object required by the selected action.",
                _payload_examples(allowed_actions),
            ]
        )
    if include_reasoning:
        return "\n".join(
            [
                "Return a JSON object with these fields:",
                "- `reasoning`: object with `attempt`, `uncertainty_summary`, and `need_external_help`.",
                "- `decision`: object with `action`, `confidence`, `brief_rationale`, and non-empty `action_input`.",
                f"- `decision.action`: one of {action_field}.",
                "- `decision.action_input`: the non-empty payload object required by the selected action.",
                _payload_examples(allowed_actions),
            ]
        )
    return "\n".join(
        [
            f"Return exactly one root-only {root_name} with these fields:",
            f"- `action`: one of {action_field}.",
            "- `confidence`: number in [0, 1].",
            "- `brief_rationale`: short string explaining why this action is appropriate.",
            "- `action_input`: the non-empty payload object required by the selected action.",
            _payload_examples(allowed_actions),
        ]
    )


def build_rollout_system_prompt(
    *,
    dataset: str | None,
    allowed_actions: list[str],
    effective_top_k: int,
) -> str:
    allowed = _normalized_actions(allowed_actions)
    return "\n\n".join(
        [
            "You are a decision-aware assistant.",
            "Your goal is to solve the user's problem as far as possible using your current knowledge. Prefer a direct ANSWER when your own reasoning is sufficient; use an external action only when it would materially improve correctness, evidence, or safety.",
            "Rollout mode: first reasoning about the problem, then return a ranked candidate set for the next action.",
            "\n".join(
                [
                    "Hard rules:",
                    f"1. Return a TOP-K ranked candidate set with exactly {effective_top_k} candidates.",
                    "2. One candidate must be ANSWER, even if it is ranked last or has low confidence.",
                    f"3. Candidate actions must come only from the current allowed actions: {', '.join(allowed)}.",
                    "4. Output valid JSON only. No prose before or after the JSON.",
                    "5. Every candidate must include `rank`, `action`, `confidence`, `brief_rationale`, and `action_input`.",
                    "6. Do not use REFUSE for missing evidence, tool failure, uncertainty, unsupported claims, or false premises unless the user asks for unsafe or harmful compliance.",
                    "7. Do not output a root-only single-action object; rollout output must be a root object with `reasoning` and `candidates`.",
                ]
            ),
            "\n".join(
                [
                    "Current allowed actions for this example:",
                    *[f"- {action}" for action in allowed],
                ]
            ),
            _payload_rules(allowed),
            read_dataset_rollout_system_prompt(dataset),
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
            "You are a decision-aware assistant.",
            "Your goal is to solve the user's problem as far as possible using your own reasoning first, then choose exactly one next action. Prefer a direct ANSWER when your reasoning is sufficient; use an external action only when it would materially improve correctness, evidence, or safety.",
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
            _field_contract(allowed_actions=allowed, include_reasoning=True),
            read_dataset_action_decision_prompt(dataset),
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
            "You are a decision-window action selector.",
            "Your goal is to choose the next action from a fixed current state. Prefer a direct ANSWER when the fixed state already supports one; use an external action only when it would materially improve correctness, evidence, or safety.",
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
            _field_contract(allowed_actions=allowed, root_name="JSON object"),
            read_dataset_decision_window_prompt(dataset),
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
            "You are a decision-aware assistant.",
            "You previously selected a tool action and now have the tool observation.",
            "Your goal is to finish the user's problem using the previous history, current observation, and your own reasoning. Prefer a direct ANSWER when the observation and reasoning are sufficient; use another external action only when it would materially improve correctness, evidence, or safety.",
            "Finalize-after-tool mode: use the prior action history and current tool observation to choose the final next action.",
            "\n".join(
                [
                    "Hard rules:",
                    "1. Use the previous history and current observation directly; do not invent additional facts.",
                    "2. If the observation is sufficient, choose ANSWER and put the final response in `action_input.answer`.",
                    "3. Do not choose another external action just because it is available; choose it only when the final answer would otherwise be materially unreliable.",
                    f"4. Choose only from the allowed final actions: {', '.join(allowed)}.",
                    "5. If the observation shows a tool failure, missing evidence, or unsupported premise, do not use REFUSE for that reason.",
                    "6. REFUSE is only for unsafe, harmful, or disallowed user requests, and only when REFUSE is listed as an allowed final action.",
                    "7. Output valid JSON only. No markdown fences or prose before or after the JSON.",
                    "8. Include both `reasoning` and `final_decision`.",
                    "9. In `final_decision`, fields must appear as `action`, `confidence`, `brief_rationale`, `action_input`.",
                ]
            ),
            _payload_rules(allowed),
            _field_contract(allowed_actions=allowed, include_reasoning=True, final_decision=True),
            read_dataset_finalize_system_prompt(dataset),
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


def compact_observation(observation: dict[str, Any] | None, *, max_results: int = 3, max_chars: int = 600) -> dict[str, Any]:
    """Trim a tool observation for inclusion in a multi-turn finalize prompt.

    Multi-hop datasets accumulate one observation per hop; raw SEARCH payloads
    can carry many long snippets, so deep finalize loops overflow the context
    window. We keep status/answer/result plus the top `max_results` search
    snippets, each clipped to `max_chars`, and drop bulky raw fields.
    """
    if not isinstance(observation, dict):
        return {}
    compact: dict[str, Any] = {}
    for key in ("status", "answer", "result", "error_type", "message", "user_reply", "reason"):
        if observation.get(key) is not None:
            value = observation[key]
            if isinstance(value, str) and len(value) > max_chars:
                value = value[:max_chars] + "…"
            compact[key] = value
    results = observation.get("results")
    if isinstance(results, list) and results:
        trimmed = []
        for item in results[:max_results]:
            if isinstance(item, str):
                trimmed.append(item[:max_chars] + ("…" if len(item) > max_chars else ""))
            elif isinstance(item, dict):
                entry = {k: item.get(k) for k in ("title", "snippet", "link", "url") if item.get(k) is not None}
                snip = entry.get("snippet")
                if isinstance(snip, str) and len(snip) > max_chars:
                    entry["snippet"] = snip[:max_chars] + "…"
                trimmed.append(entry)
            else:
                trimmed.append(item)
        compact["results"] = trimmed
        if len(results) > max_results:
            compact["results_truncated"] = len(results) - max_results
    return compact


def build_finalize_after_tool_messages(
    *,
    question: str,
    allowed_actions: list[str],
    transcript: list[dict[str, Any]],
    dataset: str | None = None,
    max_results: int = 3,
    max_chars: int = 600,
    force_answer: bool = False,
) -> list[dict[str, str]]:
    """Build a native multi-turn finalize prompt.

    `transcript` is the accumulated per-hop record, each entry shaped as
    `{action, action_input, observation, reasoning?, brief_rationale?}` in
    chronological order. Instead of JSON-dumping the whole history into one
    user turn (the legacy 2-turn design), we emit role-alternating turns:
    system, the original question (user), then for each hop an assistant turn
    (the emitted action JSON) followed by a tool turn (the observation). The
    model then generates the next finalize decision. This matches the base
    model's native tool-use chat distribution and keeps multi-hop attention
    alignment clean.
    """
    allowed = _normalized_actions(allowed_actions)
    messages: list[dict[str, str]] = [
        {"role": "system", "content": build_finalize_after_tool_system_prompt(dataset=dataset, allowed_actions=allowed)},
        {
            "role": "user",
            "content": "\n".join(
                [
                    f"Original question: {question}",
                    "",
                    "Allowed final actions for this example:",
                    *[f"- {action}" for action in allowed],
                ]
            ),
        },
    ]
    for hop in transcript:
        action = str(hop.get("action") or "").upper()
        assistant_payload: dict[str, Any] = {"action": action}
        rationale = hop.get("brief_rationale") or hop.get("reasoning")
        if rationale:
            assistant_payload["brief_rationale"] = str(rationale)
        assistant_payload["action_input"] = hop.get("action_input") or {}
        messages.append({"role": "assistant", "content": json.dumps(assistant_payload, ensure_ascii=False)})
        obs = compact_observation(hop.get("observation"), max_results=max_results, max_chars=max_chars)
        messages.append({
            "role": "tool",
            "content": json.dumps({"observation": obs}, ensure_ascii=False),
        })
    if force_answer:
        final_user = (
            "You have already gathered tool evidence and a further tool call would "
            "repeat a previous one without new information. Do NOT request another "
            "tool action. Using only the action/observation history above and your "
            "own reasoning, give your best final ANSWER now. Return JSON with "
            "`reasoning` and `final_decision`, where `final_decision.action` is ANSWER."
        )
    else:
        final_user = (
            "Finalize-after-tool mode: using the action/observation history above, "
            "choose the next action. Return JSON with `reasoning` and `final_decision` only."
        )
    messages.append({"role": "user", "content": final_user})
    return messages



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
