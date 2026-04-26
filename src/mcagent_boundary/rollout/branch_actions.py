from __future__ import annotations

"""v0.2 branch_actions.py

Builds branches ONLY from student-proposed top-k candidates.

Annotation phase (annotation.execute_tools: false):
  - Record candidate + action_input; do NOT invoke tools or finalize.
  - Validate action_input schema; minimal repair without gold injection.
  - On invalid: write diagnostics, skip from training pool.

Eval phase (eval.execute_tools: true):
  - Execute tool, then run student finalize pass using student_finalize_after_tool.md.
  - Compute correctness against gold for U_real.

Gold is NEVER injected into student action_input.
Debug fallback branches never enter training pairs (use_fallback_branches_for_training: false).
"""

import json
import logging
import re
from pathlib import Path
from typing import Any

from mcagent_boundary.envs.sandbox import BoundarySandbox
from mcagent_boundary.features.process_features import compute_process_features
from mcagent_boundary.features.semantic_tags import active_semantic_tags, infer_semantic_tags
from mcagent_boundary.rollout.state import build_state
from mcagent_boundary.scoring.correctness import evaluate_branch_correctness


PROMPT_ROOT = Path(__file__).resolve().parents[1] / "prompts"
EXPRESSION_PATTERN = re.compile(r"(\d+(?:\s*[\+\-\*\/]\s*\d+)+)")
logger = logging.getLogger(__name__)

REQUIRED_ACTION_INPUT_KEYS = {
    "ANSWER": "answer",
    "SEARCH": "query",
    "CALCULATE": "expression",
    "CLARIFY": "question",
    "REFUSE": "reason",
}


def _read_prompt(name: str) -> str:
    return (PROMPT_ROOT / name).read_text(encoding="utf-8")


def build_student_prompt(example) -> str:
    """Build the student rollout prompt (v0.2: no dataset/boundary_type leak)."""
    system_prompt = _read_prompt("student_rollout.md").strip()
    tool_list = ", ".join(
        action.lower() for action in example.allowed_actions() if action != "ANSWER"
    )
    user_prompt = (
        f"Question: {example.question}\n\n"
        f"Constraints:\n"
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


def _validate_action_input(action: str, action_input: Any) -> tuple[dict[str, Any], bool]:
    """Validate and minimally repair action_input. Returns (repaired, is_valid).

    Repair only fills missing required keys with empty string.
    Gold is NEVER injected.
    """
    if not isinstance(action_input, dict):
        action_input = {}
    required = REQUIRED_ACTION_INPUT_KEYS.get(action)
    if required is None:
        return dict(action_input), True
    value = action_input.get(required)
    if isinstance(value, str) and value.strip():
        return dict(action_input), True
    if value not in (None, "", {}, []):
        return dict(action_input), True
    repaired = dict(action_input)
    repaired[required] = ""
    return repaired, False  # repaired but was invalid (empty required field)


def _finalize_after_tool_heuristic(action: str, observation: dict[str, Any]) -> tuple[str | None, str]:
    """Simple heuristic finalize used when policy.finalize_after_tool is unavailable."""
    if action == "SEARCH":
        results = observation.get("results") or []
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


def _run_finalize_pass(policy, sample, candidate: dict[str, Any], observation: dict[str, Any]) -> dict[str, str]:
    """Run student finalize pass after tool observation (eval only)."""
    try:
        finalize_prompt_text = _read_prompt("student_finalize_after_tool.md").strip()
        obs_text = json.dumps(observation, ensure_ascii=False)
        full_prompt = (
            finalize_prompt_text + "\n\n"
            f"Original question: {sample.question}\n"
            f"Action taken: {candidate['action']}\n"
            f"Action input: {json.dumps(candidate.get('action_input', {}), ensure_ascii=False)}\n"
            f"Tool observation: {obs_text}\n\n"
            "Return JSON with final_decision only."
        )
        if hasattr(policy, "finalize_after_tool"):
            return policy.finalize_after_tool(sample, candidate, observation)
        # Fallback: use heuristic finalize.
        final_answer, final_status = _finalize_after_tool_heuristic(candidate["action"], observation)
        return {"final_answer": final_answer or "", "final_status": final_status}
    except Exception as exc:
        logger.warning("finalize_after_tool failed for action %s: %s", candidate["action"], exc)
        final_answer, final_status = _finalize_after_tool_heuristic(candidate["action"], observation)
        return {"final_answer": final_answer or "", "final_status": final_status}


def _branch_summary(branch: dict[str, Any]) -> str:
    return (
        f"{branch['action']}: utility_rel={branch.get('utility_rel')}, "
        f"outcome={branch.get('outcome_label')}, correctness={branch.get('correctness')}, "
        f"status={branch.get('final_status')}"
    )


def _build_annotation_branches(
    candidates: list[dict[str, Any]],
    example,
    semantic_tags: dict[str, bool],
    diagnostics: list[dict[str, Any]],
    example_id: str,
) -> list[dict[str, Any]]:
    """Build branches for training annotation (no tool execution, no finalize)."""
    branches: list[dict[str, Any]] = []
    for candidate in candidates:
        action = candidate["action"]
        action_input, was_valid = _validate_action_input(action, candidate.get("action_input", {}))
        if not was_valid:
            logger.debug(
                "annotation branch %s for %s: action_input repaired (empty required field)",
                action, example_id,
            )
            diagnostics.append({
                "example_id": example_id,
                "action": action,
                "issue": "empty_required_field_repaired",
            })
        branches.append({
            "action": action,
            "action_input": action_input,
            "confidence": candidate.get("confidence"),
            "brief_rationale": candidate.get("brief_rationale", ""),
            "rank": candidate.get("rank"),
            # No observation, no final_answer in annotation phase.
            "observation": None,
            "history": [],
            "info": {},
            "final_answer": None,
            "final_status": "annotation_no_execution",
            "correctness": None,
            "is_student_candidate": True,
            "is_debug_fallback": False,
        })
    return branches


def _build_eval_branches(
    candidates: list[dict[str, Any]],
    example,
    semantic_tags: dict[str, bool],
    config: dict[str, Any],
    phase: str,
    policy,
    diagnostics: list[dict[str, Any]],
    example_id: str,
) -> list[dict[str, Any]]:
    """Build branches for eval (execute tools + finalize)."""
    branches: list[dict[str, Any]] = []
    legacy_sample = example.to_legacy_sample()
    for candidate in candidates:
        action = candidate["action"]
        action_input, was_valid = _validate_action_input(action, candidate.get("action_input", {}))
        if not was_valid:
            diagnostics.append({
                "example_id": example_id,
                "action": action,
                "issue": "empty_required_field_repaired",
            })
        sandbox = BoundarySandbox(example=example, config=config, phase=phase)
        if action == "ANSWER":
            observation, done, info = {"answer": action_input.get("answer", "")}, True, {}
            final_answer = action_input.get("answer", "")
            final_status = "answered"
            finalize_decision = None
        else:
            observation, done, info = sandbox.step(action, action_input)
            finalize_result = _run_finalize_pass(policy, legacy_sample, candidate, observation)
            final_answer = finalize_result.get("final_answer")
            final_status = finalize_result.get("final_status", f"completed_after_{action.lower()}")
            finalize_decision = finalize_result
        correctness = evaluate_branch_correctness(example, final_answer, action, semantic_tags)
        branches.append({
            "action": action,
            "action_input": action_input,
            "confidence": candidate.get("confidence"),
            "brief_rationale": candidate.get("brief_rationale", ""),
            "rank": candidate.get("rank"),
            "observation": observation,
            "history": sandbox.history if action != "ANSWER" else [],
            "info": info,
            "final_answer": final_answer,
            "final_status": final_status,
            "correctness": correctness,
            "finalize_decision": finalize_decision,
            "is_student_candidate": True,
            "is_debug_fallback": False,
        })
    return branches


def rollout_one_example(example, config: dict[str, Any], phase: str, policy) -> dict[str, Any]:
    """Rollout a single example using student-proposed top-k candidates.

    Returns a rollout record with candidates, state, process features, and
    semantic tags. Does NOT import utility functions here — utility_rel is
    computed downstream after teacher labeling.
    """
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

    candidates = policy_output.candidates
    diagnostics: list[dict[str, Any]] = []

    # Determine annotation vs eval mode.
    annotation_cfg = config.get("annotation", {})
    execute_tools = annotation_cfg.get("execute_tools", False)
    if phase in ("eval", "test"):
        eval_cfg = config.get("eval", {})
        execute_tools = eval_cfg.get("execute_tools", True)

    # If top-k parsing failed, keep the raw output for diagnostics and exclude
    # the record from mining/pair construction. v0.2 forbids synthesized debug
    # fallback candidates from entering the training pool.
    if candidates is None:
        logger.warning(
            "example %s: policy returned no valid top-k candidates; excluding from training pools.",
            example.example_id,
        )
        candidates = []
        diagnostics.append({
            "example_id": example.example_id,
            "issue": "invalid_candidate_output",
            "use_fallback_branches_for_training": False,
        })

    # Build branches (annotation or eval).
    if execute_tools:
        branches = _build_eval_branches(
            candidates, example, base_semantic_tags, config, phase, policy, diagnostics,
            example.example_id,
        )
    else:
        branches = _build_annotation_branches(
            candidates, example, base_semantic_tags, diagnostics, example.example_id
        )

    # Build ranked_actions placeholder (utility_rel computed downstream).
    ranked_actions = [
        {"action": c["action"], "rank": c.get("rank", i + 1), "confidence": c.get("confidence")}
        for i, c in enumerate(candidates)
    ]
    candidate_actions_map = {c["action"]: c for c in candidates}
    best_action = candidates[0]["action"] if candidates else None

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
        "reasoning_attempt": policy_output.reasoning_attempt,
        "uncertainty_summary": policy_output.uncertainty_summary,
        "process_features": process_features,
        "semantic_tags": base_semantic_tags,
        "active_semantic_tags": [name for name, enabled in base_semantic_tags.items() if enabled],
        "candidates": candidates,
        "branches": branches,
        "branch_summaries": [_branch_summary(b) for b in branches],
        "ranked_actions": ranked_actions,
        "best_action": best_action,
        "natural_action": candidates[0]["action"] if candidates else None,
        "natural_action_confidence": candidates[0].get("confidence") if candidates else None,
        "natural_action_input": candidates[0].get("action_input", {}) if candidates else {},
        "execute_tools": execute_tools,
        "diagnostics": diagnostics,
        "raw_policy_text": policy_output.raw_text,
    }
