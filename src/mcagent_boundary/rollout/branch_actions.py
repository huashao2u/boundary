from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from mcagent_boundary.envs.sandbox import BoundarySandbox
from mcagent_boundary.features.process_features import compute_process_features
from mcagent_boundary.features.semantic_tags import active_semantic_tags, infer_semantic_tags
from mcagent_boundary.rollout.state import build_state
from mcagent_boundary.scoring.correctness import evaluate_branch_correctness
from mcagent_boundary.scoring.utility import evaluate_branch_utilities


PROMPT_ROOT = Path(__file__).resolve().parents[1] / "prompts"
EXPRESSION_PATTERN = re.compile(r"(\d+(?:\s*[\+\-\*\/]\s*\d+)+)")


def _read_prompt(name: str) -> str:
    return (PROMPT_ROOT / name).read_text(encoding="utf-8")


def build_student_prompt(example) -> str:
    tool_list = ", ".join(action.lower() for action in example.allowed_actions() if action != "ANSWER")
    system_prompt = _read_prompt("student_rollout.md").strip()
    user_prompt = (
        f"Question: {example.question}\n\n"
        f"Constraints:\n"
        f"- Dataset: {example.dataset}\n"
        f"- Boundary type: {example.boundary_type}\n"
        f"- Tools allowed: {tool_list or 'none'}\n"
        f"- Clarify allowed: {str(example.can_clarify)}"
    )
    return system_prompt + "\n\n" + user_prompt


def _fallback_reason(example, semantic_tags: dict[str, bool]) -> str:
    if semantic_tags.get("MISSING_INFO"):
        return "The request looks underspecified, so I should avoid answering prematurely."
    if semantic_tags.get("FALSE_PREMISE"):
        return "The premise may be false, so I should not answer as if it were true."
    if semantic_tags.get("CALCULATION_REQUIRED"):
        return "A calculation step may be needed before trusting a direct answer."
    if semantic_tags.get("TOOL_REQUIRED"):
        return "External evidence may be needed before answering confidently."
    return "The task seems self-contained enough to consider a direct answer."


def _direct_answer_guess(example, semantic_tags: dict[str, bool]) -> str:
    if semantic_tags.get("FALSE_PREMISE"):
        return "The premise of the question appears false."
    if semantic_tags.get("MISSING_INFO"):
        return "I do not have enough information to answer directly."
    if semantic_tags.get("TIME_SENSITIVE") or semantic_tags.get("NEW_OR_TAIL_KNOWLEDGE"):
        return "I am not confident I can answer this directly without checking external evidence."
    gold = example.gold_answer
    if isinstance(gold, list):
        return str(gold[0]) if gold else "I am not sure."
    if isinstance(gold, str) and gold:
        return gold
    return "I need more context to answer precisely."


def _build_counterfactual_action_input(action: str, example, semantic_tags: dict[str, bool]) -> dict[str, Any]:
    if action == "ANSWER":
        return {"answer": _direct_answer_guess(example, semantic_tags)}
    if action == "SEARCH":
        return {"query": example.question}
    if action == "CALCULATE":
        expression_match = EXPRESSION_PATTERN.search(example.question)
        return {"expression": expression_match.group(1).replace(" ", "") if expression_match else "1+1"}
    if action == "CLARIFY":
        metadata = dict(example.metadata or {})
        missing_details = metadata.get("missing_details") or []
        slot = "missing_detail"
        question = metadata.get("gold_clarify_question") or "Could you clarify the missing detail?"
        if missing_details:
            top = max(missing_details, key=lambda item: int(item.get("importance", 0)))
            slot = str(top.get("slot") or top.get("field") or slot)
            question = str(top.get("inquiry") or question)
        return {"question": question, "slot": slot}
    return {"reason": "The request is unsupported, false-premise, or unjustified under current context."}


def _finalize_after_tool(action: str, example, observation: dict[str, Any], semantic_tags: dict[str, bool]) -> tuple[str | None, str]:
    if action == "SEARCH":
        results = observation.get("results") or []
        if semantic_tags.get("TIME_SENSITIVE") or semantic_tags.get("NEW_OR_TAIL_KNOWLEDGE"):
            gold = example.gold_answer
            if isinstance(gold, list):
                return (str(gold[0]) if gold else None), "answered_after_search"
            if isinstance(gold, str) and gold:
                return gold, "answered_after_search"
        return (results[0] if results else None), "answered_after_search"
    if action == "CALCULATE":
        return observation.get("result"), "answered_after_calculate"
    if action == "CLARIFY":
        user_reply = observation.get("user_reply")
        if user_reply:
            return f"Resolved after clarification: {user_reply}", "answered_after_clarify"
        return None, "clarify_failed"
    if action == "REFUSE":
        return observation.get("reason"), observation.get("status", "refused")
    return observation.get("answer"), observation.get("status", "answered")


def _branch_summary(branch: dict[str, Any]) -> str:
    return (
        f"{branch['action']}: utility={branch.get('utility')}, "
        f"outcome={branch.get('outcome_label')}, correctness={branch.get('correctness')}, "
        f"status={branch.get('final_status')}"
    )


REQUIRED_ACTION_FIELDS = {
    "ANSWER": "answer",
    "SEARCH": "query",
    "CALCULATE": "expression",
    "CLARIFY": "question",
    "REFUSE": "reason",
}


def _coerce_natural_action_input(
    action: str,
    student_input: dict[str, Any],
    example,
    semantic_tags: dict[str, bool],
) -> dict[str, Any]:
    """Ensure the student's action_input is executable.

    If the student emitted the right action but an empty/missing required field
    (e.g. ``CALCULATE`` with no ``expression``), fall back to the counterfactual
    builder for that one field so the branch is still executable. We keep the
    student's *decision* (the action) intact — only the input is repaired.
    """
    required = REQUIRED_ACTION_FIELDS.get(action)
    if required is None:
        return student_input
    value = student_input.get(required)
    if isinstance(value, str) and value.strip():
        return student_input
    if value not in (None, "", {}, []):
        return student_input
    counterfactual = _build_counterfactual_action_input(action, example, semantic_tags)
    repaired = dict(student_input)
    repaired[required] = counterfactual.get(required, "")
    return repaired


def rollout_one_example(example, config: dict[str, Any], phase: str, policy) -> dict[str, Any]:
    prompt_text = build_student_prompt(example)
    legacy_sample = example.to_legacy_sample()
    policy_output = policy.generate_decision(legacy_sample, prompt_text)
    base_semantic_tags = infer_semantic_tags(example, reason_prefix=policy_output.reason)
    reason_prefix = policy_output.reason or _fallback_reason(example, base_semantic_tags)
    process_features = compute_process_features(
        reason_prefix=reason_prefix,
        raw_text=policy_output.raw_text,
        token_threshold=int(config["features"]["long_reason_token_threshold"]),
    )
    natural_decision = dict(policy_output.decision)
    natural_action = str(natural_decision.get("action", "ANSWER")).upper()
    candidate_actions = [action for action in config["action_space"] if action in example.allowed_actions()]
    branches: list[dict[str, Any]] = []
    for action in candidate_actions:
        sandbox = BoundarySandbox(example=example, config=config, phase=phase)
        if action == natural_action:
            action_input = _coerce_natural_action_input(
                action,
                dict(natural_decision.get("action_input", {})),
                example,
                base_semantic_tags,
            )
        else:
            action_input = _build_counterfactual_action_input(action, example, base_semantic_tags)
        observation, done, info = sandbox.step(action, action_input)
        if action == "ANSWER":
            final_answer = action_input.get("answer")
            final_status = "answered"
        elif done:
            final_answer, final_status = _finalize_after_tool(action, example, observation, base_semantic_tags)
        else:
            final_answer, final_status = _finalize_after_tool(action, example, observation, base_semantic_tags)
        correctness = evaluate_branch_correctness(example, final_answer, action, base_semantic_tags)
        branches.append(
            {
                "action": action,
                "action_input": action_input,
                "observation": observation,
                "history": sandbox.history,
                "info": info,
                "final_answer": final_answer,
                "final_status": final_status,
                "correctness": correctness,
                "is_natural_action": action == natural_action,
            }
        )
    scored_branches = evaluate_branch_utilities(example, branches, base_semantic_tags, config)
    ranked = [
        {"action": branch["action"], "utility": float(branch["utility"]), "outcome_label": branch["outcome_label"]}
        for branch in sorted(scored_branches, key=lambda item: float(item["utility"]), reverse=True)
    ]
    candidate_utilities = {branch["action"]: float(branch["utility"]) for branch in scored_branches}
    best_action = ranked[0]["action"] if ranked else None
    natural_branch = next((branch for branch in scored_branches if branch["action"] == natural_action), None)
    resolved_active_tags = active_semantic_tags(example, reason_prefix=reason_prefix)
    state = build_state(
        example=example,
        reason_prefix=reason_prefix,
        history=None,
        process_features=process_features,
        semantic_tags=base_semantic_tags,
        active_semantic_tags=resolved_active_tags,
    )
    return {
        "state_id": state.state_key_hash(),
        "state": state.to_dict(),
        "example_id": example.example_id,
        "dataset": example.dataset,
        "split": example.split,
        "boundary_type": example.boundary_type,
        "question": example.question,
        "gold_answer": example.gold_answer,
        "metadata": example.metadata,
        "prompt": prompt_text,
        "reason_prefix": reason_prefix,
        "natural_action": natural_action,
        "natural_action_confidence": natural_decision.get("confidence"),
        "natural_action_input": natural_decision.get("action_input", {}),
        "process_features": process_features,
        "semantic_tags": base_semantic_tags,
        "active_semantic_tags": [name for name, enabled in base_semantic_tags.items() if enabled and name != "CALCULATION_REQUIRED"],
        "branches": scored_branches,
        "branch_summaries": [_branch_summary(branch) for branch in scored_branches],
        "candidate_utilities": candidate_utilities,
        "ranked_actions": ranked,
        "best_action": best_action,
        "best_utility": ranked[0]["utility"] if ranked else None,
        "natural_branch": natural_branch,
        "raw_policy_text": policy_output.raw_text,
    }
