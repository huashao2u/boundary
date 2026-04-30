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
from pathlib import Path
from typing import Any

from mcagent_boundary.envs.sandbox import BoundarySandbox
from mcagent_boundary.features.process_features import compute_process_feature_details
from mcagent_boundary.features.semantic_tags import infer_semantic_tag_details
from mcagent_boundary.rollout.candidate_schema import (
    canonicalize_candidates,
    valid_candidates,
)
from mcagent_boundary.rollout.state import build_state
from mcagent_boundary.scoring.correctness import evaluate_branch_correctness
from mcagent_boundary.scoring.helpfulness import outcome_helpfulness
from mcagent_boundary.scoring.utility import utility_real


PROMPT_ROOT = Path(__file__).resolve().parents[1] / "prompts"
logger = logging.getLogger(__name__)

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
    if semantic_tags.get("SEARCH_REQUIRED"):
        return "External evidence may be needed before answering confidently."
    return "The task seems self-contained enough to consider a direct answer."


def _finalize_after_tool_heuristic(action: str, observation: dict[str, Any]) -> tuple[str | None, str]:
    """Simple heuristic finalize used when policy.finalize_after_tool is unavailable.

    DEPRECATED(mainline): this eval-only fallback is not part of train-side
    boundary mining or pair construction.
    """
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
            return policy.finalize_after_tool(sample, candidate, observation, full_prompt)
        # DEPRECATED(mainline): eval-only fallback when a policy lacks a
        # finalize method; never used by train-side annotation.
        final_answer, final_status = _finalize_after_tool_heuristic(candidate["action"], observation)
        return {"final_answer": final_answer or "", "final_status": final_status}
    except Exception as exc:
        logger.warning("finalize_after_tool failed for action %s: %s", candidate["action"], exc)
        return {"final_answer": "", "final_status": "finalize_parse_failed", "finalize_error": str(exc)}


def _branch_summary(branch: dict[str, Any]) -> str:
    return (
        f"{branch['action']}: utility_rel={branch.get('utility_rel')}, "
        f"outcome={branch.get('outcome_label')}, correctness={branch.get('correctness')}, "
        f"status={branch.get('final_status')}"
    )


def _real_outcome_label(example, branch: dict[str, Any], branch_map: dict[str, dict[str, Any]], semantic_tags: dict[str, bool]) -> str:
    action = str(branch.get("action", "")).upper()
    helpful = outcome_helpfulness(example, branch, branch_map, semantic_tags)
    if action == "ANSWER":
        return "ANSWER_correct" if branch.get("correctness") is True else "ANSWER_wrong"
    if action == "REFUSE":
        return "REFUSE_justified" if helpful else "REFUSE_unjustified"
    return f"{action}_{'helpful' if helpful else 'unhelpful'}"


def _score_eval_branches_real(
    branches: list[dict[str, Any]],
    example,
    semantic_tags: dict[str, bool],
    config: dict[str, Any],
) -> dict[str, Any]:
    branch_map = {branch["action"]: branch for branch in branches}
    valid_for_scoring = [
        branch for branch in branches if branch.get("valid_for_eval_scoring", True)
    ]
    for branch in branches:
        if not branch.get("valid_for_eval_scoring", True):
            branch["utility_real"] = None
            branch["outcome_label_real"] = "invalid_for_eval_scoring"
            continue
        branch["utility_real"] = utility_real(branch, example, branch_map, semantic_tags, config)
        branch["outcome_label_real"] = _real_outcome_label(example, branch, branch_map, semantic_tags)
    best = max(valid_for_scoring, key=lambda branch: branch["utility_real"], default=None)
    natural = next((branch for branch in branches if branch.get("rank") == 1), None)
    return {
        "best_action_real": None if best is None else best["action"],
        "natural_branch_real": natural,
    }


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
        action_input = candidate.get("canonical_action_input") or candidate.get("action_input", {})
        was_valid = bool(candidate.get("valid_candidate"))
        if not was_valid:
            diagnostics.append({
                "example_id": example_id,
                "action": action,
                "rank": candidate.get("rank"),
                "issue": "invalid_action_input_after_canonicalization",
                "schema_diagnostics": candidate.get("schema_diagnostics", []),
            })
            continue
        branches.append({
            "action": action,
            "action_input": action_input,
            "canonical_action_input": action_input,
            "valid_candidate": True,
            "schema_diagnostics": candidate.get("schema_diagnostics", []),
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
        action_input = candidate.get("canonical_action_input") or candidate.get("action_input", {})
        was_valid = bool(candidate.get("valid_candidate"))
        if not was_valid:
            diagnostics.append({
                "example_id": example_id,
                "action": action,
                "rank": candidate.get("rank"),
                "issue": "invalid_action_input_after_canonicalization",
                "schema_diagnostics": candidate.get("schema_diagnostics", []),
            })
            continue
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
            valid_for_eval_scoring = final_status != "finalize_parse_failed"
            if not valid_for_eval_scoring:
                diagnostics.append({
                    "example_id": example_id,
                    "action": action,
                    "rank": candidate.get("rank"),
                    "issue": "finalize_parse_failed",
                })
        correctness = evaluate_branch_correctness(example, final_answer, action, semantic_tags)
        branches.append({
            "action": action,
            "action_input": action_input,
            "canonical_action_input": action_input,
            "valid_candidate": True,
            "schema_diagnostics": candidate.get("schema_diagnostics", []),
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
            "valid_for_eval_scoring": True if action == "ANSWER" else valid_for_eval_scoring,
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

    base_semantic_details = infer_semantic_tag_details(example, reason_prefix=policy_output.reason)
    base_semantic_tags = dict(base_semantic_details["semantic_tags"])
    semantic_tag_evidence = dict(base_semantic_details.get("semantic_tag_evidence") or {})
    reason_prefix = policy_output.reason or _fallback_reason(example, base_semantic_tags)

    raw_candidates = policy_output.candidates
    diagnostics: list[dict[str, Any]] = []
    top_k_actions = int(config.get("rollout", {}).get("top_k_actions", 3))
    diagnostics.append({
        "example_id": example.example_id,
        "issue": "candidate_parse_config",
        "top_k_actions": top_k_actions,
    })

    # Determine annotation vs eval mode.
    annotation_cfg = config.get("annotation", {})
    execute_tools = annotation_cfg.get("execute_tools", False)
    if phase in ("eval", "test"):
        eval_cfg = config.get("eval", {})
        execute_tools = eval_cfg.get("execute_tools", True)

    # If top-k parsing failed, keep the raw output for diagnostics and exclude
    # the record from mining/pair construction. v0.2 forbids synthesized debug
    # fallback candidates from entering the training pool.
    if raw_candidates is None:
        logger.warning(
            "example %s: policy returned no valid top-k candidates; excluding from training pools.",
            example.example_id,
        )
        raw_candidates = []
        diagnostics.append({
            "example_id": example.example_id,
            "issue": "invalid_candidate_output",
            "use_fallback_branches_for_training": False,
        })

    candidates = canonicalize_candidates(raw_candidates)
    allowed_actions = set(example.allowed_actions())
    filtered_candidates: list[dict[str, Any]] = []
    for candidate in candidates:
        action = str(candidate.get("action", "")).upper()
        if action not in allowed_actions:
            diagnostics.append({
                "example_id": example.example_id,
                "action": action,
                "rank": candidate.get("rank"),
                "issue": "candidate_action_not_allowed",
                "allowed_actions": sorted(allowed_actions),
            })
            continue
        filtered_candidates.append(candidate)
    candidates = [{**candidate, "rank": index + 1} for index, candidate in enumerate(filtered_candidates)]

    for candidate in candidates:
        if not candidate.get("valid_candidate"):
            diagnostics.append({
                "example_id": example.example_id,
                "action": candidate.get("action"),
                "rank": candidate.get("rank"),
                "issue": "dropped_invalid_candidate",
                "schema_diagnostics": candidate.get("schema_diagnostics", []),
            })

    valid_candidate_list = valid_candidates(candidates)
    valid_actions = {str(candidate.get("action", "")).upper() for candidate in valid_candidate_list}
    if "ANSWER" not in valid_actions:
        diagnostics.append({
            "example_id": example.example_id,
            "issue": "invalid_missing_answer",
            "detail": "ANSWER missing after allowed-action/schema filtering.",
        })
    if len(valid_candidate_list) < 2:
        diagnostics.append({
            "example_id": example.example_id,
            "issue": "invalid_single_action",
            "detail": "Fewer than two valid candidates after allowed-action/schema filtering.",
            "valid_candidate_count": len(valid_candidate_list),
        })
    empty_action_input_count = sum(
        1
        for candidate in candidates
        if not candidate.get("valid_candidate")
        and any(
            diag.get("issue") == "empty_required_action_input"
            for diag in (candidate.get("schema_diagnostics") or [])
        )
    )
    candidate_action_logprobs = [
        {
            key: candidate.get(key)
            for key in (
                "rank",
                "action",
                "action_json_logprob_sum",
                "action_json_logprob_mean",
                "action_json_num_tokens",
                "missing_logprob_positions",
            )
            if key in candidate
        }
        for candidate in valid_candidate_list
        if candidate.get("action_json_logprob_mean") is not None
    ]

    process_details = compute_process_feature_details(
        reason_prefix=reason_prefix,
        raw_text=policy_output.raw_text,
        token_threshold=int(config["features"]["long_reason_token_threshold"]),
        reasoning_attempt=policy_output.reasoning_attempt,
        candidate_confidences=[candidate.get("confidence") for candidate in valid_candidate_list],
        action_probabilities=policy_output.action_probabilities,
        action_scores=policy_output.action_scores,
        candidate_action_logprobs=candidate_action_logprobs,
        candidates=valid_candidate_list,
        candidate_logprob_config=dict(config.get("rollout", {}).get("candidate_logprob_scoring") or {}),
        confidence_source=policy_output.confidence_source,
    )
    process_features = dict(process_details["features"])
    process_feature_diagnostics = dict(process_details.get("diagnostics") or {})

    # Build branches (annotation or eval).
    if execute_tools:
        branches = _build_eval_branches(
            valid_candidate_list, example, base_semantic_tags, config, phase, policy, diagnostics,
            example.example_id,
        )
        eval_real = _score_eval_branches_real(branches, example, base_semantic_tags, config)
    else:
        branches = _build_annotation_branches(
            valid_candidate_list, example, base_semantic_tags, diagnostics, example.example_id
        )
        eval_real = {"best_action_real": None, "natural_branch_real": None}

    # Build ranked_actions placeholder (utility_rel computed downstream).
    ranked_actions = [
        {"action": c["action"], "rank": c.get("rank", i + 1), "confidence": c.get("confidence")}
        for i, c in enumerate(valid_candidate_list)
    ]
    best_action = valid_candidate_list[0]["action"] if valid_candidate_list else None

    resolved_semantic_details = infer_semantic_tag_details(example, reason_prefix=reason_prefix)
    base_semantic_tags = dict(resolved_semantic_details["semantic_tags"])
    resolved_active_tags = list(resolved_semantic_details["active_semantic_tags"])
    semantic_tag_evidence = dict(resolved_semantic_details.get("semantic_tag_evidence") or semantic_tag_evidence)
    state = build_state(
        example=example,
        reason_prefix=reason_prefix,
        history=None,
        process_features=process_features,
        semantic_tags=base_semantic_tags,
        active_semantic_tags=resolved_active_tags,
        semantic_tag_evidence=semantic_tag_evidence,
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
        "process_feature_diagnostics": process_feature_diagnostics,
        "semantic_tags": base_semantic_tags,
        "active_semantic_tags": resolved_active_tags,
        "semantic_tag_evidence": semantic_tag_evidence,
        "candidates": candidates,
        "candidate_action_logprobs": candidate_action_logprobs,
        "valid_candidate_count": len(valid_candidate_list),
        "invalid_candidate_count": len(candidates) - len(valid_candidate_list),
        "empty_action_input_count": empty_action_input_count,
        "branches": branches,
        "branch_summaries": [_branch_summary(b) for b in branches],
        "ranked_actions": ranked_actions,
        "best_action": best_action,
        "best_action_real": eval_real["best_action_real"],
        "natural_action": valid_candidate_list[0]["action"] if valid_candidate_list else None,
        "natural_branch": eval_real["natural_branch_real"],
        "natural_branch_real": eval_real["natural_branch_real"],
        "natural_action_confidence": valid_candidate_list[0].get("confidence") if valid_candidate_list else None,
        "natural_action_input": valid_candidate_list[0].get("action_input", {}) if valid_candidate_list else {},
        "action_scores": policy_output.action_scores,
        "action_probabilities": policy_output.action_probabilities,
        "confidence_source": policy_output.confidence_source,
        "execute_tools": execute_tools,
        "diagnostics": diagnostics,
        "raw_policy_text": policy_output.raw_text,
    }
