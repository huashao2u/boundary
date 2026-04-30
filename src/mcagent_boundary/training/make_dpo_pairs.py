from __future__ import annotations

# NOTE(boundary v0.2): Pair construction now sources exclusively from student-proposed
# top-k candidates ranked by U_rel (not U_real/outcome). Pairs are truncated at action
# emission (no tool obs/finalize in training pairs). Chosen = argmax U_rel with
# rubric_degenerate=false; Rejected = lowest U_rel with different action type and
# gap >= min_utility_gap (0.2). Message format: prompt_messages/chosen_messages/rejected_messages.
# See §pair_construction in rollout.yaml.
#
# TODO(boundary): produce a *per-branch* teacher reflection so the rejected
# completion's reasoning text is also teacher-generated. Current interim
# behavior: `chosen` uses the teacher's single meta_reflection; `rejected`
# uses a deterministic rejected-role line so chosen/rejected pairs are not
# textually identical. The rejected action itself still comes only from
# student-proposed candidates. See CLAUDE.md §Change 4.

import json
import logging
from hashlib import sha1
from typing import Any, Literal

from mcagent_boundary.rollout.candidate_schema import is_valid_candidate, required_input_value
from mcagent_boundary.scoring.utility import utility_rel


logger = logging.getLogger(__name__)

_ACTION_SET = {"ANSWER", "SEARCH", "CALCULATE", "CLARIFY", "REFUSE"}
_BOUNDARY_TO_TASK = {
    "reasoning": "math",
    "factual": "factual_boundary",
    "intention": "intention_boundary",
}
_EXCLUDED_ROLLOUT_ISSUES = {
    "invalid_candidate_output",
    "legacy_single_decision_fallback",
    "invalid_schema",
    "invalid_missing_answer",
    "invalid_single_action",
}


# ---------------------------------------------------------------------------
# Prompt / completion rendering
# ---------------------------------------------------------------------------

def _build_prompt_messages(record: dict[str, Any]) -> list[dict[str, str]]:
    """Build prompt messages from student rollout record.

    v0.2: uses student's own reasoning prefix as assistant turn; dataset/boundary
    metadata is excluded from the prompt (§1 no gold/boundary leak to student).
    Prompt is truncated at the point of action emission.
    """
    candidates = list(record.get("candidates") or [])
    allowed_tools = [
        c["action"] for c in candidates if str(c.get("action", "")).upper() != "ANSWER"
    ]
    tool_list = ", ".join(dict.fromkeys(a.upper() for a in allowed_tools))  # preserve order, dedup

    return [
        {
            "role": "system",
            "content": "You are a decision-aware assistant. Choose the next action calibrated to the current boundary state.",
        },
        {
            "role": "user",
            "content": (
                f"Question: {record['question']}\n"
                f"Available actions: ANSWER{', ' + tool_list if tool_list else ''}"
            ),
        },
        {
            "role": "assistant",
            "content": f"Reasoning: {record.get('reason_prefix', '')}",
        },
    ]


def _reflection_for_role(
    branch: dict[str, Any],
    teacher_label: dict[str, Any] | None,
    role_label: Literal["chosen", "rejected"],
) -> str:
    if role_label == "chosen":
        if teacher_label is not None and teacher_label.get("meta_reflection"):
            return str(teacher_label["meta_reflection"])
        return f"I should prefer {branch['action']} under the current evidence and uncertainty."
    # rejected role — intentionally NOT the teacher's meta_reflection so that
    # chosen and rejected completions differ in surrounding text as well as the
    # final JSON action payload. See CLAUDE.md §Change 4.
    return (
        f"Alternative choice: picking {branch['action']} despite its lower local "
        f"utility under this boundary state."
    )


def _build_completion_messages(
    branch: dict[str, Any],
    teacher_label: dict[str, Any] | None,
    role_label: Literal["chosen", "rejected"] = "chosen",
) -> list[dict[str, str]]:
    """Build completion messages truncated at action emission (no tool obs/finalize)."""
    reflection = _reflection_for_role(branch, teacher_label, role_label)
    # action_input from the candidate — guaranteed no gold injection by branch_actions.py
    payload = {
        "action": branch["action"],
        "action_input": branch.get("canonical_action_input") or branch.get("action_input", {}),
    }
    return [
        {"role": "assistant", "content": f"Meta-Reflection: {reflection}"},
        {"role": "assistant", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def _render_prompt_text(record: dict[str, Any]) -> str:
    """Legacy flat-text prompt for backward compat / debugging."""
    candidates = list(record.get("candidates") or [])
    tool_list = ", ".join(
        dict.fromkeys(c["action"].upper() for c in candidates if c.get("action", "").upper() != "ANSWER")
    )
    return (
        "SYSTEM: You are a decision-aware assistant. Choose the next action calibrated to the current boundary state.\n"
        f"USER: Question: {record['question']}\n"
        f"Available actions: ANSWER{', ' + tool_list if tool_list else ''}\n"
        f"ASSISTANT: Reasoning: {record.get('reason_prefix', '')}\n"
        "ASSISTANT:"
    )


def _render_completion_text(
    branch: dict[str, Any],
    teacher_label: dict[str, Any] | None,
    role_label: Literal["chosen", "rejected"] = "chosen",
) -> str:
    """Legacy flat-text completion for backward compat / debugging."""
    reflection = _reflection_for_role(branch, teacher_label, role_label)
    payload = {
        "action": branch["action"],
        "action_input": branch.get("canonical_action_input") or branch.get("action_input", {}),
    }
    return f" Meta-Reflection: {reflection}\n{json.dumps(payload, ensure_ascii=False)}"


# ---------------------------------------------------------------------------
# U_rel computation helpers
# ---------------------------------------------------------------------------

class _CandidateExampleProxy:
    """Thin proxy so utility_rel can read example.question / gold_answer / metadata."""
    def __init__(self, record: dict[str, Any]) -> None:
        metadata = dict(record.get("metadata") or {})
        boundary_type = str(record.get("boundary_type", ""))
        self.question = record.get("question", "")
        self.gold_answer = record.get("gold_answer")
        self.task_type = metadata.get("task_type") or _BOUNDARY_TO_TASK.get(boundary_type, boundary_type)
        self.metadata = metadata


def _score_candidates(
    record: dict[str, Any],
    teacher_label: dict[str, Any] | None,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return candidates enriched with u_rel, sourced from record.candidates.

    Uses annotation branches (no tool execution) to build minimal branch dicts
    for utility_rel computation.
    """
    example = _CandidateExampleProxy(record)
    semantic_tags: dict[str, bool] = dict(record.get("semantic_tags") or {})
    candidates = [candidate for candidate in list(record.get("candidates") or []) if is_valid_candidate(candidate)]

    # Build a branch dict per candidate (annotation branch: no observation/correctness).
    # We pull action_input from the annotation branches if available, else from candidates.
    branch_by_key = {}
    for branch in (record.get("branches") or []):
        action = str(branch.get("action", "")).upper()
        branch_by_key[(branch.get("rank"), action)] = branch

    scored: list[dict[str, Any]] = []
    for cand in candidates:
        action = str(cand.get("action", "ANSWER")).upper()
        branch = branch_by_key.get((cand.get("rank"), action)) or {}
        # Compose a minimal branch dict for utility_rel
        action_input = (
            branch.get("canonical_action_input")
            or branch.get("action_input")
            or cand.get("canonical_action_input")
            or cand.get("action_input")
            or {}
        )
        branch_dict: dict[str, Any] = {
            "action": action,
            "action_input": action_input,
            "canonical_action_input": action_input,
            "rank": cand.get("rank"),
            "valid_candidate": True,
            "schema_diagnostics": cand.get("schema_diagnostics", []),
            "confidence": cand.get("confidence"),
            "brief_rationale": cand.get("brief_rationale", ""),
            # annotation branches have no observation/correctness
            "observation": branch.get("observation"),
            "correctness": branch.get("correctness"),
        }
        u_rel_val = utility_rel(branch_dict, teacher_label, semantic_tags, example, config)
        scored.append({
            **cand,
            "action": action,
            "action_input": branch_dict["action_input"],
            "canonical_action_input": branch_dict["canonical_action_input"],
            "valid_candidate": True,
            "u_rel": u_rel_val,
            "brief_rationale": cand.get("brief_rationale", ""),
        })
    return scored


# ---------------------------------------------------------------------------
# Pair selection
# ---------------------------------------------------------------------------

def _pick_chosen_rejected(
    scored_candidates: list[dict[str, Any]],
    teacher_label: dict[str, Any] | None,
    min_utility_gap: float,
    require_different_action_types: bool,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    """Pick (chosen, rejected) from U_rel-ranked candidates.

    Chosen: argmax U_rel, rubric_degenerate must be False (or teacher absent).
    Rejected: lowest U_rel among candidates with a different action type and
              gap >= min_utility_gap.
    """
    if len(scored_candidates) < 2:
        return None

    rubric_degenerate = bool(teacher_label and teacher_label.get("rubric_degenerate", False))
    if rubric_degenerate:
        return None

    ranked = sorted(scored_candidates, key=lambda c: c["u_rel"], reverse=True)
    chosen = ranked[0]

    for candidate in reversed(ranked):  # iterate from lowest U_rel upward
        if require_different_action_types and candidate["action"] == chosen["action"]:
            continue
        if chosen["u_rel"] - candidate["u_rel"] < min_utility_gap:
            continue
        return chosen, candidate

    return None


# ---------------------------------------------------------------------------
# Teacher lookup
# ---------------------------------------------------------------------------

def _teacher_lookup(teacher_labels: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {record["state_id"]: record for record in teacher_labels}


def _excluded_rollout_issues(record: dict[str, Any]) -> list[str]:
    issues = []
    for diagnostic in record.get("diagnostics") or []:
        issue = str(diagnostic.get("issue", ""))
        if issue in _EXCLUDED_ROLLOUT_ISSUES:
            issues.append(issue)
    return sorted(set(issues))


def _teacher_missing_scores(
    candidates: list[dict[str, Any]],
    teacher_label: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    if teacher_label is None:
        return []
    scores = teacher_label.get("candidate_utility") or teacher_label.get("candidate_helpfulness") or []
    keys: set[tuple[int | None, str]] = set()
    fallback_actions: set[str] = set()
    for entry in scores:
        if not isinstance(entry, dict):
            continue
        action = str(entry.get("action", "")).upper()
        rank = entry.get("rank")
        try:
            rank = None if rank is None else int(rank)
        except (TypeError, ValueError):
            rank = None
        if rank is None:
            fallback_actions.add(action)
        else:
            keys.add((rank, action))
    missing = []
    for candidate in candidates:
        action = str(candidate.get("action", "")).upper()
        rank = candidate.get("rank")
        try:
            rank = None if rank is None else int(rank)
        except (TypeError, ValueError):
            rank = None
        if (rank, action) not in keys and action not in fallback_actions:
            missing.append({"rank": rank, "action": action})
    return missing


def _pair_candidate_valid(candidate: dict[str, Any]) -> bool:
    return (
        is_valid_candidate(candidate)
        and bool(required_input_value(candidate))
        and not bool(candidate.get("is_debug_fallback", False))
        and bool(candidate.get("is_student_candidate", True))
    )


# ---------------------------------------------------------------------------
# Split assignment
# ---------------------------------------------------------------------------

def _assign_split(record: dict[str, Any], eval_datasets: set[str]) -> str:
    if record["dataset"] in eval_datasets:
        return "eval"
    bucket = int(sha1(record["state_id"].encode("utf-8")).hexdigest(), 16) % 10
    return "eval" if bucket == 0 else "train"


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_step_dpo_pairs(
    selected_records: list[dict[str, Any]],
    teacher_labels: list[dict[str, Any]],
    config: dict[str, Any],
    *,
    show_progress: bool = True,
    require_teacher_label: bool = False,
    require_poe_teacher: bool = False,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Build Step-DPO pairs from student top-k candidates ranked by U_rel.

    Chosen = argmax(U_rel) with rubric_degenerate=False.
    Rejected = min(U_rel) with different action type, gap >= min_utility_gap.
    Pairs are truncated at action emission (no tool obs/finalize).
    """
    teacher_by_state = _teacher_lookup(teacher_labels)
    pair_cfg = config.get("pair_construction", {})
    min_utility_gap = float(pair_cfg.get("min_utility_gap", 0.2))
    require_different_action_types = bool(pair_cfg.get("require_different_action_types", True))
    eval_datasets = set(config["datasets"]["eval"])

    train_pairs: list[dict[str, Any]] = []
    eval_pairs: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []

    from mcagent_boundary.progress import make_progress

    counters = {"produced": 0, "dropped": 0}
    progress = make_progress(
        selected_records,
        total=len(selected_records),
        desc="pair construction",
        unit="rec",
        disable=not show_progress,
    )
    for record in progress:
        state_id = record.get("state_id", "")
        diagnostic_base = {
            "state_id": state_id,
            "example_id": record.get("example_id"),
            "dataset": record.get("dataset"),
            "boundary_type": record.get("boundary_type"),
        }
        teacher_label = teacher_by_state.get(state_id)
        if require_teacher_label and teacher_label is None:
            diagnostics.append({
                **diagnostic_base,
                "reason": "missing_teacher_label",
            })
            counters["dropped"] = len(diagnostics)
            try:
                progress.set_postfix(**counters)
            except Exception:
                pass
            continue
        if require_poe_teacher and (teacher_label is None or teacher_label.get("source") != "poe_teacher"):
            diagnostics.append({
                **diagnostic_base,
                "reason": "non_poe_teacher_label",
                "teacher_source": None if teacher_label is None else teacher_label.get("source"),
            })
            counters["dropped"] = len(diagnostics)
            try:
                progress.set_postfix(**counters)
            except Exception:
                pass
            continue
        excluded_issues = _excluded_rollout_issues(record)
        if excluded_issues:
            diagnostics.append({
                **diagnostic_base,
                "reason": "excluded_invalid_rollout",
                "issues": excluded_issues,
            })
            counters["dropped"] = len(diagnostics)
            try:
                progress.set_postfix(**counters)
            except Exception:
                pass
            continue

        raw_candidates = list(record.get("candidates") or [])
        dropped_invalid = len([candidate for candidate in raw_candidates if not _pair_candidate_valid(candidate)])
        candidates = [candidate for candidate in raw_candidates if _pair_candidate_valid(candidate)]
        if dropped_invalid:
            diagnostics.append({
                **diagnostic_base,
                "reason": "dropped_invalid_candidate",
                "dropped": dropped_invalid,
            })
        if len(candidates) < 2:
            diagnostics.append({
                **diagnostic_base,
                "reason": "not_enough_valid_candidates",
                "n_candidates": len(candidates),
            })
            counters["dropped"] = len(diagnostics)
            try:
                progress.set_postfix(**counters)
            except Exception:
                pass
            continue

        missing_scores = _teacher_missing_scores(candidates, teacher_label)
        if missing_scores:
            diagnostics.append({
                **diagnostic_base,
                "reason": "teacher_missing_candidate_score",
                "missing_scores": missing_scores,
            })
            counters["dropped"] = len(diagnostics)
            try:
                progress.set_postfix(**counters)
            except Exception:
                pass
            continue

        scored = _score_candidates(record, teacher_label, config)
        scored = [candidate for candidate in scored if _pair_candidate_valid(candidate)]
        if len(scored) < 2:
            diagnostics.append({
                **diagnostic_base,
                "reason": "no_valid_pair_after_schema_filter",
                "n_candidates": len(scored),
            })
            counters["dropped"] = len(diagnostics)
            try:
                progress.set_postfix(**counters)
            except Exception:
                pass
            continue

        pair = _pick_chosen_rejected(
            scored,
            teacher_label,
            min_utility_gap=min_utility_gap,
            require_different_action_types=require_different_action_types,
        )
        if pair is None:
            diagnostics.append({
                **diagnostic_base,
                "reason": "no_valid_pair_after_schema_filter",
                "rubric_degenerate": bool(teacher_label and teacher_label.get("rubric_degenerate")),
                "n_candidates": len(scored),
                "actions": [c["action"] for c in scored],
                "u_rels": [c["u_rel"] for c in scored],
            })
            counters["dropped"] = len(diagnostics)
            try:
                progress.set_postfix(**counters)
            except Exception:
                pass
            continue

        chosen_cand, rejected_cand = pair
        prompt_messages = _build_prompt_messages(record)
        chosen_messages = _build_completion_messages(chosen_cand, teacher_label, role_label="chosen")
        rejected_messages = _build_completion_messages(rejected_cand, teacher_label, role_label="rejected")

        dpo_pair = {
            "state_id": state_id,
            "example_id": record["example_id"],
            "dataset": record["dataset"],
            "boundary_type": record["boundary_type"],
            "pool": record.get("pool", "boundary_critical"),
            "prompt_messages": prompt_messages,
            "chosen_messages": chosen_messages,
            "rejected_messages": rejected_messages,
            "prompt": _render_prompt_text(record),
            "chosen": _render_completion_text(chosen_cand, teacher_label, role_label="chosen"),
            "rejected": _render_completion_text(rejected_cand, teacher_label, role_label="rejected"),
            "chosen_action": chosen_cand["action"],
            "rejected_action": rejected_cand["action"],
            "metadata": {
                "dataset": record["dataset"],
                "boundary_type": record["boundary_type"],
                "delta_u_rel": round(chosen_cand["u_rel"] - rejected_cand["u_rel"], 6),
                "chosen_u_rel": chosen_cand["u_rel"],
                "rejected_u_rel": rejected_cand["u_rel"],
                "state_id": state_id,
                "teacher_source": None if teacher_label is None else teacher_label.get("source"),
                "chosen_valid_candidate": True,
                "rejected_valid_candidate": True,
                "chosen_schema_diagnostics": chosen_cand.get("schema_diagnostics", []),
                "rejected_schema_diagnostics": rejected_cand.get("schema_diagnostics", []),
                "active_semantic_tags": record.get("active_semantic_tags", []),
                "semantic_tag_evidence": record.get("semantic_tag_evidence", {}),
                "process_features": record.get("process_features", {}),
            },
        }
        if _assign_split(record, eval_datasets) == "eval":
            eval_pairs.append(dpo_pair)
        else:
            train_pairs.append(dpo_pair)
        counters["produced"] = len(train_pairs) + len(eval_pairs)
        counters["dropped"] = len(diagnostics)
        try:
            progress.set_postfix(**counters)
        except Exception:
            pass

    try:
        progress.close()
    except Exception:
        pass
    counters["produced"] = len(train_pairs) + len(eval_pairs)
    counters["dropped"] = len(diagnostics)
    logger.info(
        "build_step_dpo_pairs: train=%d eval=%d diagnostics=%d",
        len(train_pairs), len(eval_pairs), len(diagnostics),
    )
    return train_pairs, eval_pairs, diagnostics
