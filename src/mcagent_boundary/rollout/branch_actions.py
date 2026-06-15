from __future__ import annotations

"""v0.2 branch_actions.py

Builds branches ONLY from student-proposed top-k candidates.

Annotation phase (annotation.execute_tools: false):
  - Record candidate + action_input; do NOT invoke tools or finalize.
  - Validate action_input schema; minimal repair without gold injection.
  - On invalid: write diagnostics, skip from training pool.

Eval phase (eval.execute_tools: true):
  - Execute tool, then run student finalize pass with a dataset-specific prompt.
  - Compute correctness against gold for U_real.

Gold is NEVER injected into student action_input.
Debug fallback branches never enter training pairs (use_fallback_branches_for_training: false).
"""

import json
import logging
from typing import Any

from mcagent_boundary.envs.sandbox import BoundarySandbox
from mcagent_boundary.features.process_features import compute_process_feature_details
from mcagent_boundary.features.semantic_tags import infer_semantic_tag_details
from mcagent_boundary.prompting.action_decision import (
    build_action_decision_prompt_text,
    build_decision_window_prompt_text,
    build_finalize_after_tool_messages,
    build_finalize_after_tool_prompt_text,
    build_rollout_system_prompt,
)
from mcagent_boundary.rollout.candidate_schema import (
    canonicalize_candidates,
    valid_candidates,
)
from mcagent_boundary.rollout.state import build_state
from mcagent_boundary.scoring.correctness import evaluate_branch_correctness
from mcagent_boundary.scoring.helpfulness import outcome_helpfulness
from mcagent_boundary.scoring.utility import utility_real


logger = logging.getLogger(__name__)


def effective_top_k_for_example(example, config: dict[str, Any] | None = None) -> int:
    rollout_cfg = (config or {}).get("rollout", {})
    config_top_k = int(rollout_cfg.get("top_k_actions", 3))
    allowed_count = len(example.allowed_actions())
    if allowed_count <= 0:
        return 0
    return max(1, min(config_top_k, allowed_count))


def build_student_prompt(
    example,
    config: dict[str, Any] | None = None,
    *,
    reasoning_attempt: str | None = None,
) -> str:
    """Build the student prompt (v0.2: no dataset/boundary_type leak)."""
    prompt_mode = str((config or {}).get("rollout", {}).get("prompt_mode", "top_k")).lower()
    if prompt_mode in {"decision_window", "state_conditioned", "state_conditioned_action"}:
        return build_decision_window_prompt_text(
            question=example.question,
            allowed_actions=example.allowed_actions(),
            reasoning_attempt=str(reasoning_attempt or ""),
            can_clarify=bool(getattr(example, "can_clarify", False)),
            dataset=str(getattr(example, "dataset", "")),
        )
    if prompt_mode in {"single", "single_action", "action_decision"}:
        return build_action_decision_prompt_text(
            question=example.question,
            allowed_actions=example.allowed_actions(),
            can_clarify=bool(getattr(example, "can_clarify", False)),
            dataset=str(getattr(example, "dataset", "")),
        )

    allowed_actions = example.allowed_actions()
    effective_top_k = effective_top_k_for_example(example, config)
    system_prompt = build_rollout_system_prompt(
        dataset=str(getattr(example, "dataset", "")),
        allowed_actions=allowed_actions,
        effective_top_k=effective_top_k,
    )
    user_prompt = (
        f"Question:\n{example.question}\n\n"
        f"Available actions:\n"
        f"{json.dumps(allowed_actions, ensure_ascii=False)}\n\n"
        f"Return exactly {effective_top_k} candidates with different actions. ANSWER must appear."
    )
    return system_prompt + "\n\n" + user_prompt


def _dynamic_student_examples(allowed_actions: list[str]) -> str:
    actions = set(allowed_actions)
    examples: list[str] = []
    if "CALCULATE" in actions and actions <= {"ANSWER", "CALCULATE"}:
        examples.append(
            """

### Math calculation boundary example
```json
{
  "reasoning": {
    "attempt": "The arithmetic can be reasoned through, and a calculator action can independently verify the final value.",
    "uncertainty_summary": "The only uncertainty is arithmetic error.",
    "need_external_help": true
  },
  "candidates": [
    {"rank": 1, "action": "ANSWER", "confidence": 0.72, "brief_rationale": "The calculation is simple enough to answer directly.", "action_input": {"answer": "50"}},
    {"rank": 2, "action": "CALCULATE", "confidence": 0.58, "brief_rationale": "The executable expression can verify the numeric result.", "action_input": {"expression": "3*(4**2)+2"}}
  ]
}
```
""".rstrip()
        )
    if "CLARIFY" in actions:
        examples.append(
            """

### Clarify boundary example
```json
{
  "reasoning": {
    "attempt": "The request lacks a key slot: the city. I cannot make a concrete restaurant recommendation without it.",
    "uncertainty_summary": "The city is missing.",
    "need_external_help": true
  },
  "candidates": [
    {"rank": 1, "action": "CLARIFY", "confidence": 0.86, "brief_rationale": "The city is necessary for a useful recommendation.", "action_input": {"question": "Which city should I search in?"}},
    {"rank": 2, "action": "ANSWER", "confidence": 0.25, "brief_rationale": "A direct answer is possible only in a generic, less useful form.", "action_input": {"answer": "I can suggest general date-night restaurant criteria, but I need the city to recommend a specific place."}}
  ]
}
```
""".rstrip()
        )
    if "SEARCH" in actions:
        examples.append(
            """

### Search boundary example
```json
{
  "reasoning": {
    "attempt": "The question asks about a specific long-tail fact that I may not know reliably from memory.",
    "uncertainty_summary": "External evidence is needed to avoid hallucinating.",
    "need_external_help": true
  },
  "candidates": [
    {"rank": 1, "action": "SEARCH", "confidence": 0.82, "brief_rationale": "A targeted search is needed for reliable evidence.", "action_input": {"query": "specific entity relation in the question"}},
    {"rank": 2, "action": "ANSWER", "confidence": 0.18, "brief_rationale": "A direct answer would be uncertain, so it should be phrased cautiously if used.", "action_input": {"answer": "I cannot determine the answer reliably from internal knowledge alone."}}
  ]
}
```
""".rstrip()
        )
    if "REFUSE" in actions and actions <= {"ANSWER", "REFUSE"}:
        examples.append(
            """

### Answer-vs-refuse boundary example
```json
{
  "reasoning": {
    "attempt": "The prompt may be suspicious, but I should distinguish benign requests from requests that require refusal.",
    "uncertainty_summary": "The safe action depends on whether answering would enable harm.",
    "need_external_help": false
  },
  "candidates": [
    {"rank": 1, "action": "ANSWER", "confidence": 0.65, "brief_rationale": "Benign or safety-framed requests should not be over-refused.", "action_input": {"answer": "I can give a safe, high-level response that avoids harmful operational details."}},
    {"rank": 2, "action": "REFUSE", "confidence": 0.35, "brief_rationale": "Refusal is appropriate only when the request crosses a safety boundary.", "action_input": {"reason": "I should refuse if the request asks for harmful instructions or unsafe compliance."}}
  ]
}
```
""".rstrip()
        )
    if not examples:
        return ""
    return "\n\nRelevant formatting examples for this action space:" + "\n".join(examples)


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


def _observation_indicates_failure(observation: dict[str, Any]) -> bool:
    status = str(observation.get("status") or "").lower()
    if status in {"tool_error", "error", "failed", "failure", "timeout"}:
        return True
    return any(key in observation and observation.get(key) for key in ("error", "tool_error", "exception"))


def _allowed_finalize_actions(sample, observation: dict[str, Any] | None = None) -> list[str]:
    metadata = dict(getattr(sample, "metadata", {}) or {})
    dataset = str(metadata.get("dataset") or metadata.get("legacy_dataset") or getattr(sample, "dataset", "")).lower()
    actions = ["ANSWER"]
    for action, flag in (("SEARCH", "can_search"), ("CALCULATE", "can_calculate"), ("CLARIFY", "can_clarify")):
        if bool(metadata.get(flag)):
            actions.append(action)
    allow_refuse = bool(metadata.get("allow_refuse"))
    if dataset in {"in3", "mintqa"}:
        allow_refuse = False
    elif dataset == "or_bench":
        allow_refuse = True
    if allow_refuse and not _observation_indicates_failure(observation or {}):
        actions.append("REFUSE")
    return actions


def _format_finalize_trace(trace: list[dict[str, Any]] | None) -> str:
    compact: list[dict[str, Any]] = []
    for item in trace or []:
        result = dict(item.get("finalize_result") or {})
        compact.append({
            "step": item.get("step"),
            "action": item.get("action"),
            "action_input": item.get("action_input") or {},
            "observation": item.get("observation") or {},
            "finalize_action": result.get("action"),
            "finalize_action_input": result.get("action_input") or {},
            "finalize_status": result.get("final_status"),
            "brief_rationale": result.get("brief_rationale", ""),
        })
    return json.dumps(compact, ensure_ascii=False)


def _run_finalize_pass(
    policy,
    sample,
    candidate: dict[str, Any],
    observation: dict[str, Any],
    *,
    history: list[dict[str, Any]] | None = None,
    trace: list[dict[str, Any]] | None = None,
    transcript: list[dict[str, Any]] | None = None,
    force_answer: bool = False,
) -> dict[str, Any]:
    """Run student finalize pass after tool observation (eval only).

    When `transcript` is provided, build a native multi-turn message prompt
    (system + question + alternating assistant/tool turns) and pass it to the
    policy via `prompt_messages`. The legacy flat-text path (history/trace
    JSON-dumped into one user turn) remains as a fallback for policies that
    only accept `prompt_text`. When `force_answer` is set, the prompt instructs
    the model to answer from history instead of requesting another tool (used to
    break degenerate repeated-query loops).
    """
    try:
        allowed_actions = _allowed_finalize_actions(sample, observation)
        prompt_messages = None
        if transcript is not None:
            prompt_messages = build_finalize_after_tool_messages(
                question=sample.question,
                allowed_actions=allowed_actions,
                transcript=transcript,
                dataset=str(getattr(sample, "dataset", "")),
                force_answer=force_answer,
            )
        full_prompt = build_finalize_after_tool_prompt_text(
            question=sample.question,
            allowed_actions=allowed_actions,
            current_action=str(candidate["action"]),
            current_action_input=candidate.get("action_input", {}),
            current_observation=observation,
            history=history,
            trace=_format_finalize_trace(trace),
            dataset=str(getattr(sample, "dataset", "")),
        )
        if hasattr(policy, "finalize_after_tool"):
            try:
                result = policy.finalize_after_tool(
                    sample, candidate, observation, full_prompt, prompt_messages=prompt_messages
                )
            except TypeError:
                # Policy predates the prompt_messages kwarg; use flat text.
                result = policy.finalize_after_tool(sample, candidate, observation, full_prompt)
        else:
            # DEPRECATED(mainline): eval-only fallback when a policy lacks a
            # finalize method; never used by train-side annotation.
            final_answer, final_status = _finalize_after_tool_heuristic(candidate["action"], observation)
            result = {"final_answer": final_answer or "", "final_status": final_status}
        action = str(result.get("action") or "").upper()
        if action and action not in set(allowed_actions):
            result.update({
                "final_answer": "",
                "final_status": "finalize_disallowed_action",
                "disallowed_action": action,
                "allowed_actions": allowed_actions,
            })
        else:
            result.setdefault("allowed_actions", allowed_actions)
        return result
    except Exception as exc:
        logger.warning("finalize_after_tool failed for action %s: %s", candidate["action"], exc)
        return {"final_answer": "", "final_status": "finalize_parse_failed", "finalize_error": str(exc)}


def _eval_tool_finalize_depth(config: dict[str, Any], dataset: str | None = None) -> int:
    """Resolve the post-tool finalize depth, optionally per dataset.

    Multi-hop datasets (e.g. MintQA) need several SEARCH->observe->finalize
    passes to reach an answer; a single hop floors their EM at 0 and pollutes
    unnecessary_search_rate. `eval.tool_finalize_depth_by_dataset` overrides the
    scalar `eval.tool_finalize_depth` / `rollout.tool_finalize_depth` for the
    named dataset.
    """
    eval_cfg = config.get("eval", {}) or {}
    rollout_cfg = config.get("rollout", {}) or {}
    value = eval_cfg.get("tool_finalize_depth", rollout_cfg.get("tool_finalize_depth", 1))
    if dataset is not None:
        by_dataset = eval_cfg.get("tool_finalize_depth_by_dataset") or rollout_cfg.get(
            "tool_finalize_depth_by_dataset"
        ) or {}
        if dataset in by_dataset:
            value = by_dataset[dataset]
    try:
        return max(1, int(value))
    except (TypeError, ValueError):
        return 1


def _run_tool_finalize_loop(
    *,
    sandbox: BoundarySandbox,
    policy,
    sample,
    initial_candidate: dict[str, Any],
    initial_observation: dict[str, Any],
    max_depth: int,
) -> dict[str, Any]:
    """Run up to max_depth post-tool finalization passes.

    Depth 1 preserves the legacy behavior: one tool observation, one finalize
    pass. Larger depths allow finalize to emit another tool action, execute it,
    and finalize again.
    """
    candidate = dict(initial_candidate)
    observation = initial_observation
    trace: list[dict[str, Any]] = []
    # Accumulated multi-turn transcript: one entry per executed hop, in
    # chronological order. Each finalize pass sees the full transcript as
    # role-alternating assistant/tool turns rather than a JSON-dumped blob.
    transcript: list[dict[str, Any]] = []
    # Track (action, normalized-input) of every executed tool hop to detect
    # degenerate loops where the model keeps requesting the same query.
    executed_signatures: set[tuple[str, str]] = set()
    final_result: dict[str, Any] = {}

    def _signature(action: str, action_input: dict[str, Any]) -> tuple[str, str]:
        try:
            payload = json.dumps(action_input or {}, ensure_ascii=False, sort_keys=True)
        except (TypeError, ValueError):
            payload = str(action_input)
        return (str(action).upper(), payload.strip().lower())

    executed_signatures.add(_signature(candidate.get("action", ""), candidate.get("action_input") or {}))
    for step_index in range(max_depth):
        transcript.append({
            "action": candidate.get("action"),
            "action_input": candidate.get("action_input") or {},
            "brief_rationale": candidate.get("brief_rationale", ""),
            "observation": observation,
        })
        final_result = _run_finalize_pass(
            policy,
            sample,
            candidate,
            observation,
            history=sandbox.history,
            trace=trace,
            transcript=transcript,
        )
        trace.append({
            "step": step_index + 1,
            "action": candidate.get("action"),
            "action_input": candidate.get("action_input") or {},
            "observation": observation,
            "finalize_result": dict(final_result),
        })
        status = str(final_result.get("final_status") or "")
        next_action = str(final_result.get("action") or "").upper()
        next_input = final_result.get("action_input") or {}
        if status in {"finalize_parse_failed", "finalize_disallowed_action"} or next_action in {"ANSWER", "REFUSE", ""}:
            break
        if step_index + 1 >= max_depth:
            break
        allowed_next_actions = set(_allowed_finalize_actions(sample, observation))
        if next_action not in {"SEARCH", "CALCULATE", "CLARIFY"} or next_action not in allowed_next_actions or not isinstance(next_input, dict):
            break
        # Degenerate-loop guard: if the model proposes a tool call identical to
        # one already executed, do not run it again. Instead force a final
        # answer from the accumulated history and stop.
        if _signature(next_action, next_input) in executed_signatures:
            forced = _run_finalize_pass(
                policy,
                sample,
                candidate,
                observation,
                history=sandbox.history,
                trace=trace,
                transcript=transcript,
                force_answer=True,
            )
            forced.setdefault("final_status", "answered_after_repeat_break")
            forced["repeated_action_break"] = True
            forced["repeated_action"] = next_action
            trace.append({
                "step": step_index + 1,
                "action": next_action,
                "action_input": next_input,
                "observation": observation,
                "finalize_result": dict(forced),
                "note": "repeated_tool_call_forced_answer",
            })
            final_result = forced
            break
        observation, _done, _info = sandbox.step(next_action, next_input)
        executed_signatures.add(_signature(next_action, next_input))
        candidate = {
            "action": next_action,
            "action_input": next_input,
            "brief_rationale": final_result.get("brief_rationale", ""),
        }
    final_result["finalize_trace"] = trace
    final_result["finalize_depth_used"] = len(trace)
    return final_result


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
    # Branches are already ordered by the valid candidates after schema and
    # allowed-action filtering. The original rank-1 candidate may have been
    # dropped, so the natural branch must be the first valid/scored branch.
    natural = branches[0] if branches else None
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
            finalize_result = _run_tool_finalize_loop(
                sandbox=sandbox,
                policy=policy,
                sample=legacy_sample,
                initial_candidate=candidate,
                initial_observation=observation,
                max_depth=_eval_tool_finalize_depth(config, getattr(example, "dataset", None)),
            )
            final_answer = finalize_result.get("final_answer")
            final_status = finalize_result.get("final_status", f"completed_after_{action.lower()}")
            finalize_decision = finalize_result
            valid_for_eval_scoring = final_status not in {
                "finalize_parse_failed",
                "finalize_disallowed_action",
                "finalize_empty_answer",
            }
            if not valid_for_eval_scoring:
                diagnostics.append({
                    "example_id": example_id,
                    "action": action,
                    "rank": candidate.get("rank"),
                    "issue": final_status,
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


def rollout_one_example(
    example,
    config: dict[str, Any],
    phase: str,
    policy,
    *,
    reasoning_attempt: str | None = None,
) -> dict[str, Any]:
    """Rollout a single example using student-proposed top-k candidates.

    Returns a rollout record with candidates, state, process features, and
    semantic tags. Does NOT import utility functions here — utility_rel is
    computed downstream after teacher labeling.
    """
    rollout_cfg = config.get("rollout", {})
    prompt_mode = str(rollout_cfg.get("prompt_mode", "top_k")).lower()
    single_action_prompt = prompt_mode in {
        "single",
        "single_action",
        "action_decision",
        "decision_window",
        "state_conditioned",
        "state_conditioned_action",
    }
    prompt_text = build_student_prompt(example, config, reasoning_attempt=reasoning_attempt)
    legacy_sample = example.to_legacy_sample()
    policy_output = policy.generate_decision(legacy_sample, prompt_text)

    base_semantic_details = infer_semantic_tag_details(example, reason_prefix=policy_output.reason)
    base_semantic_tags = dict(base_semantic_details["semantic_tags"])
    semantic_tag_evidence = dict(base_semantic_details.get("semantic_tag_evidence") or {})
    reason_prefix = policy_output.reason or _fallback_reason(example, base_semantic_tags)

    raw_candidates = policy_output.candidates
    diagnostics: list[dict[str, Any]] = []
    allowed_actions_list = example.allowed_actions()
    top_k_actions = effective_top_k_for_example(example, config)
    required_min_candidates = min(2, len(allowed_actions_list))
    diagnostics.append({
        "example_id": example.example_id,
        "issue": "candidate_parse_config",
        "top_k_actions": top_k_actions,
        "effective_top_k": top_k_actions,
        "required_min_candidates": required_min_candidates,
        "allowed_actions": allowed_actions_list,
        "prompt_mode": prompt_mode,
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
    allowed_actions = set(allowed_actions_list)
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
    if not single_action_prompt and "ANSWER" not in valid_actions:
        diagnostics.append({
            "example_id": example.example_id,
            "issue": "invalid_missing_answer",
            "detail": "ANSWER missing after allowed-action/schema filtering.",
        })
    if not single_action_prompt and len(valid_candidate_list) < required_min_candidates:
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
        "allowed_actions": allowed_actions_list,
        "effective_top_k": top_k_actions,
        "required_min_candidates": required_min_candidates,
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
