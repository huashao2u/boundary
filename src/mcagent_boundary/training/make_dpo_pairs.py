from __future__ import annotations

# NOTE(boundary v0.2): Pair selection still sources from student-proposed top-k
# candidates ranked by U_rel (not U_real/outcome), but DPO samples train the
# single-action decision format used by eval. Training pairs end at action
# emission: no tool execution, no query execution, and no finalize pass.
# Chosen = argmax U_rel with rubric_degenerate=false; Rejected = lowest U_rel
# with different action type and gap >= min_utility_gap (0.2).
# See §pair_construction in rollout.yaml.
#
import logging
import re
from hashlib import sha1
from typing import Any, Iterable, Literal

from mcagent_boundary.annotation.validate_teacher_evidence import apply_evidence_score_guardrail
from mcagent_boundary.prompting.action_decision import (
    build_action_decision_completion,
    build_action_decision_prompt_messages,
    build_action_decision_prompt_text,
    build_decision_window_completion,
    build_decision_window_prompt_messages,
    build_decision_window_prompt_text,
)
from mcagent_boundary.rollout.candidate_schema import (
    is_valid_candidate,
    is_empty_answer_rejected_only,
    required_input_value,
    valid_as_chosen,
    valid_as_rejected,
)
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
}
_REFLECTION_MARKER_REPLACEMENTS = {
    "lower utility": "less suitable",
    "rejected": "non-selected",
    "worse": "less suitable",
    "alternative despite": "another possible choice given",
}


# ---------------------------------------------------------------------------
# Prompt / completion rendering
# ---------------------------------------------------------------------------

def _pair_prompt_mode(config: dict[str, Any]) -> str:
    mode = str(config.get("pair_construction", {}).get("prompt_mode", "decision_window")).lower()
    if mode in {"end_to_end", "single_action", "full"}:
        return "end_to_end"
    return "decision_window"


def _record_reasoning_attempt(record: dict[str, Any]) -> str:
    return str(record.get("reasoning_attempt") or record.get("reason_prefix") or "").strip()


def _build_prompt_messages(record: dict[str, Any], config: dict[str, Any]) -> list[dict[str, str]]:
    """Build the same single-action prompt used by eval."""
    if _pair_prompt_mode(config) == "decision_window":
        return build_decision_window_prompt_messages(
            question=record["question"],
            allowed_actions=_allowed_actions_from_record(record),
            reasoning_attempt=_record_reasoning_attempt(record),
            can_clarify=_record_can_clarify(record),
        )
    return build_action_decision_prompt_messages(
        question=record["question"],
        allowed_actions=_allowed_actions_from_record(record),
        can_clarify=_record_can_clarify(record),
    )


def _record_can_clarify(record: dict[str, Any]) -> bool:
    metadata = dict(record.get("metadata") or {})
    if "can_clarify" in metadata:
        return bool(metadata.get("can_clarify"))
    return any(str(candidate.get("action", "")).upper() == "CLARIFY" for candidate in record.get("candidates") or [])


def _allowed_actions_from_record(record: dict[str, Any]) -> list[str]:
    actions = ["ANSWER"]
    metadata = dict(record.get("metadata") or {})
    metadata_action_flags = {
        "SEARCH": "can_search",
        "CALCULATE": "can_calculate",
        "CLARIFY": "can_clarify",
        "REFUSE": "allow_refuse",
    }
    for action, flag in metadata_action_flags.items():
        if bool(metadata.get(flag)):
            actions.append(action)
    for candidate in record.get("candidates") or []:
        action = str(candidate.get("action", "")).upper()
        if action in _ACTION_SET and action not in actions:
            actions.append(action)
    return actions


def _reflection_for_role(
    branch: dict[str, Any],
    teacher_label: dict[str, Any] | None,
    role_label: Literal["chosen", "rejected"],
) -> str:
    rank = branch.get("rank")
    try:
        rank = None if rank is None else int(rank)
    except (TypeError, ValueError):
        rank = None
    action = str(branch.get("action", "")).upper()
    if teacher_label is not None:
        for entry in teacher_label.get("candidate_reflection") or []:
            if not isinstance(entry, dict):
                continue
            entry_action = str(entry.get("action", "")).upper()
            entry_rank = entry.get("rank")
            try:
                entry_rank = None if entry_rank is None else int(entry_rank)
            except (TypeError, ValueError):
                entry_rank = None
            if entry_action == action and (rank is None or entry_rank == rank):
                reflection = str(entry.get("reflection", "")).strip()
                if reflection:
                    return _sanitize_reflection(reflection)
    if role_label == "chosen":
        if teacher_label is not None and teacher_label.get("meta_reflection"):
            return _sanitize_reflection(str(teacher_label["meta_reflection"]))
        return f"I should prefer {branch['action']} under the current evidence and uncertainty."
    return _sanitize_reflection(str(branch.get("brief_rationale") or "").strip()) or (
        f"I considered {branch['action']} under the current uncertainty."
    )


def _sanitize_reflection(text: str) -> str:
    cleaned = str(text or "").strip()
    lowered = cleaned.lower()
    for marker, replacement in _REFLECTION_MARKER_REPLACEMENTS.items():
        if marker in lowered:
            cleaned = re.sub(re.escape(marker), replacement, cleaned, flags=re.IGNORECASE)
            lowered = cleaned.lower()
    return cleaned


def _candidate_teacher_entry(
    teacher_label: dict[str, Any] | None,
    branch: dict[str, Any],
    *,
    field: str,
) -> dict[str, Any] | None:
    if teacher_label is None:
        return None
    rank = branch.get("rank")
    try:
        rank = None if rank is None else int(rank)
    except (TypeError, ValueError):
        rank = None
    action = str(branch.get("action", "")).upper()
    fallback = None
    for entry in teacher_label.get(field) or []:
        if not isinstance(entry, dict):
            continue
        if str(entry.get("action", "")).upper() != action:
            continue
        entry_rank = entry.get("rank")
        try:
            entry_rank = None if entry_rank is None else int(entry_rank)
        except (TypeError, ValueError):
            entry_rank = None
        if rank is not None and entry_rank == rank:
            return entry
        if fallback is None:
            fallback = entry
    return fallback


def _build_completion_messages(
    branch: dict[str, Any],
    teacher_label: dict[str, Any] | None,
    config: dict[str, Any],
    role_label: Literal["chosen", "rejected"] = "chosen",
) -> list[dict[str, str]]:
    """Build completion messages truncated at action emission (no tool obs/finalize)."""
    reflection = _reflection_for_role(branch, teacher_label, role_label)
    if _pair_prompt_mode(config) == "decision_window":
        completion = build_decision_window_completion(
            action=branch["action"],
            action_input=branch.get("canonical_action_input") or branch.get("action_input", {}),
            confidence=branch.get("confidence"),
            brief_rationale=reflection,
        )
        return [{"role": "assistant", "content": completion}]
    # action_input from the candidate — guaranteed no gold injection by branch_actions.py
    completion = build_action_decision_completion(
        action=branch["action"],
        action_input=branch.get("canonical_action_input") or branch.get("action_input", {}),
        confidence=branch.get("confidence"),
        reasoning_attempt=reflection,
        uncertainty_summary="",
        brief_rationale=reflection,
    )
    return [{"role": "assistant", "content": completion}]


def _render_prompt_text(record: dict[str, Any], config: dict[str, Any]) -> str:
    """Legacy flat-text prompt for backward compat / debugging."""
    if _pair_prompt_mode(config) == "decision_window":
        return build_decision_window_prompt_text(
            question=record["question"],
            allowed_actions=_allowed_actions_from_record(record),
            reasoning_attempt=_record_reasoning_attempt(record),
            can_clarify=_record_can_clarify(record),
        )
    return build_action_decision_prompt_text(
        question=record["question"],
        allowed_actions=_allowed_actions_from_record(record),
        can_clarify=_record_can_clarify(record),
    )


def _render_completion_text(
    branch: dict[str, Any],
    teacher_label: dict[str, Any] | None,
    config: dict[str, Any],
    role_label: Literal["chosen", "rejected"] = "chosen",
) -> str:
    """Legacy flat-text completion for backward compat / debugging."""
    reflection = _reflection_for_role(branch, teacher_label, role_label)
    if _pair_prompt_mode(config) == "decision_window":
        return build_decision_window_completion(
            action=branch["action"],
            action_input=branch.get("canonical_action_input") or branch.get("action_input", {}),
            confidence=branch.get("confidence"),
            brief_rationale=reflection,
        )
    return build_action_decision_completion(
        action=branch["action"],
        action_input=branch.get("canonical_action_input") or branch.get("action_input", {}),
        confidence=branch.get("confidence"),
        reasoning_attempt=reflection,
        uncertainty_summary="",
        brief_rationale=reflection,
    )


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
    candidates = [candidate for candidate in list(record.get("candidates") or []) if valid_as_rejected(candidate)]

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
            "candidate_status": cand.get("candidate_status", "valid" if is_valid_candidate(cand) else "empty_answer_input"),
            "valid_for_chosen": valid_as_chosen(cand),
            "valid_for_rejected": valid_as_rejected(cand),
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
            "candidate_status": branch_dict["candidate_status"],
            "valid_for_chosen": branch_dict["valid_for_chosen"],
            "valid_for_rejected": branch_dict["valid_for_rejected"],
            "u_rel": u_rel_val,
            "brief_rationale": cand.get("brief_rationale", ""),
        })
    return scored


# ---------------------------------------------------------------------------
# Pair selection
# ---------------------------------------------------------------------------

def _pair_augmentation_cfg(config: dict[str, Any]) -> dict[str, Any]:
    cfg = config.get("pair_construction", {}).get("augmentation", {})
    return cfg if isinstance(cfg, dict) else {}


def pair_augmentation_enabled(config: dict[str, Any]) -> bool:
    cfg = _pair_augmentation_cfg(config)
    return bool(cfg.get("enabled", False))


def _stable_sort_key(record: dict[str, Any], kind: str) -> str:
    state_id = str(record.get("state_id", ""))
    return sha1(f"{kind}:{state_id}".encode("utf-8")).hexdigest()


def _ranked_chosen_candidates(scored_candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        [candidate for candidate in scored_candidates if valid_as_chosen(candidate)],
        key=lambda candidate: candidate["u_rel"],
        reverse=True,
    )


def _select_augmented_candidates(
    candidates: list[dict[str, Any]],
    *,
    max_pairs: int,
    dataset_quota: dict[str, Any] | None = None,
    fill_remaining: bool = True,
) -> list[dict[str, Any]]:
    if max_pairs <= 0 or not candidates:
        return []
    selected: list[dict[str, Any]] = []
    selected_ids: set[int] = set()
    quotas: dict[str, int] = {}
    for dataset, value in (dataset_quota or {}).items():
        try:
            quotas[str(dataset)] = max(0, int(value))
        except (TypeError, ValueError):
            continue

    for dataset, quota in quotas.items():
        if quota <= 0:
            continue
        for idx, candidate in enumerate(candidates):
            if len(selected) >= max_pairs:
                break
            if idx in selected_ids or str(candidate["record"].get("dataset")) != dataset:
                continue
            selected.append(candidate)
            selected_ids.add(idx)
            quota -= 1
            if quota <= 0:
                break

    if fill_remaining:
        for idx, candidate in enumerate(candidates):
            if len(selected) >= max_pairs:
                break
            if idx in selected_ids:
                continue
            selected.append(candidate)
            selected_ids.add(idx)
    return selected


def _action_pair(pair: dict[str, Any]) -> str:
    chosen = str(pair.get("chosen_action", "UNKNOWN")).upper()
    rejected = str(pair.get("rejected_action", "UNKNOWN")).upper()
    return f"{chosen}>{rejected}"


def _post_filter_pair_quota(
    pairs: list[dict[str, Any]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    post_filter = config.get("pair_construction", {}).get("post_filter", {})
    if not isinstance(post_filter, dict) or not bool(post_filter.get("enabled", False)):
        return pairs
    quota_cfg = post_filter.get("action_pair_quota_by_dataset") or {}
    if not isinstance(quota_cfg, dict) or not quota_cfg:
        return pairs

    selected: list[dict[str, Any]] = []
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for pair in pairs:
        dataset = str(pair.get("dataset") or pair.get("metadata", {}).get("dataset") or "unknown")
        action_pair = _action_pair(pair)
        dataset_quota = quota_cfg.get(dataset) or {}
        if not isinstance(dataset_quota, dict) or action_pair not in dataset_quota:
            selected.append(pair)
            continue
        buckets.setdefault((dataset, action_pair), []).append(pair)

    for (dataset, action_pair), bucket_pairs in sorted(buckets.items()):
        try:
            limit = max(0, int((quota_cfg.get(dataset) or {}).get(action_pair, len(bucket_pairs))))
        except (TypeError, ValueError):
            limit = len(bucket_pairs)
        if limit >= len(bucket_pairs):
            selected.extend(bucket_pairs)
            continue
        bucket_pairs = sorted(
            bucket_pairs,
            key=lambda pair: (
                str(pair.get("pair_kind") or pair.get("metadata", {}).get("pair_kind") or "strong"),
                str(pair.get("pair_id") or pair.get("metadata", {}).get("pair_id") or pair.get("state_id", "")),
            ),
        )
        selected.extend(bucket_pairs[:limit])

    return sorted(
        selected,
        key=lambda pair: (
            str(pair.get("dataset") or pair.get("metadata", {}).get("dataset") or ""),
            str(pair.get("state_id") or pair.get("metadata", {}).get("state_id") or ""),
            str(pair.get("pair_kind") or pair.get("metadata", {}).get("pair_kind") or "strong"),
            str(pair.get("pair_id") or pair.get("metadata", {}).get("pair_id") or ""),
        ),
    )


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

    chosen_pool = [candidate for candidate in scored_candidates if valid_as_chosen(candidate)]
    rejected_pool = [candidate for candidate in scored_candidates if valid_as_rejected(candidate)]
    if not chosen_pool or len(rejected_pool) < 2:
        return None

    ranked_chosen = sorted(chosen_pool, key=lambda c: c["u_rel"], reverse=True)
    chosen = ranked_chosen[0]

    ranked_rejected = sorted(rejected_pool, key=lambda c: c["u_rel"], reverse=True)
    for candidate in reversed(ranked_rejected):  # iterate from lowest U_rel upward
        if require_different_action_types and candidate["action"] == chosen["action"]:
            continue
        if chosen["u_rel"] - candidate["u_rel"] < min_utility_gap:
            continue
        return chosen, candidate

    return None


def _near_tie_pair_candidate(
    record: dict[str, Any],
    scored: list[dict[str, Any]],
    teacher_label: dict[str, Any] | None,
    cfg: dict[str, Any],
) -> dict[str, Any] | None:
    if bool(teacher_label and teacher_label.get("rubric_degenerate", False)):
        return None
    ranked = _ranked_chosen_candidates(scored)
    if len(ranked) < 2:
        return None
    chosen, rejected = ranked[0], ranked[1]
    if bool(cfg.get("require_different_action_types", True)) and chosen["action"] == rejected["action"]:
        return None
    gap = float(chosen["u_rel"] - rejected["u_rel"])
    max_gap = float(cfg.get("max_gap", 0.20))
    primary_min_gap = float(cfg.get("primary_min_gap", cfg.get("min_gap", 0.10)))
    fallback_min_gap = float(cfg.get("fallback_min_gap", primary_min_gap))
    if gap < fallback_min_gap or gap > max_gap:
        return None
    tier = "primary" if gap >= primary_min_gap else "fallback"
    return {
        "record": record,
        "teacher_label": teacher_label,
        "chosen": chosen,
        "rejected": rejected,
        "gap": gap,
        "tier": tier,
        "sort_gap": gap,
        "sort_key": _stable_sort_key(record, "near_tie"),
    }


def _best_mid_pair_candidate(
    record: dict[str, Any],
    scored: list[dict[str, Any]],
    teacher_label: dict[str, Any] | None,
    cfg: dict[str, Any],
) -> dict[str, Any] | None:
    if bool(teacher_label and teacher_label.get("rubric_degenerate", False)):
        return None
    ranked = _ranked_chosen_candidates(scored)
    if len(ranked) < 3:
        return None
    chosen, rejected = ranked[0], ranked[1]
    if bool(cfg.get("require_different_action_types", True)) and chosen["action"] == rejected["action"]:
        return None
    gap = float(chosen["u_rel"] - rejected["u_rel"])
    if gap < float(cfg.get("min_gap", 0.10)):
        return None
    return {
        "record": record,
        "teacher_label": teacher_label,
        "chosen": chosen,
        "rejected": rejected,
        "gap": gap,
        "tier": "best_mid",
        "sort_gap": gap,
        "sort_key": _stable_sort_key(record, "best_mid"),
    }


def _make_dpo_pair(
    record: dict[str, Any],
    teacher_label: dict[str, Any] | None,
    config: dict[str, Any],
    chosen_cand: dict[str, Any],
    rejected_cand: dict[str, Any],
    *,
    min_utility_gap: float,
    pair_kind: str,
    sample_weight: float,
    augmentation_tier: str | None = None,
) -> dict[str, Any]:
    state_id = str(record.get("state_id", ""))
    prompt_mode = _pair_prompt_mode(config)
    prompt_messages = _build_prompt_messages(record, config)
    chosen_messages = _build_completion_messages(chosen_cand, teacher_label, config, role_label="chosen")
    rejected_messages = _build_completion_messages(rejected_cand, teacher_label, config, role_label="rejected")
    selection_gap = round(float(chosen_cand["u_rel"] - rejected_cand["u_rel"]), 6)
    pair_id = sha1(
        (
            f"{state_id}:{pair_kind}:"
            f"{chosen_cand.get('rank')}:{chosen_cand.get('action')}:"
            f"{rejected_cand.get('rank')}:{rejected_cand.get('action')}"
        ).encode("utf-8")
    ).hexdigest()
    return {
        "pair_id": pair_id,
        "state_id": state_id,
        "example_id": record["example_id"],
        "dataset": record["dataset"],
        "boundary_type": record["boundary_type"],
        "pool": record.get("pool", "boundary_critical"),
        "pair_kind": pair_kind,
        "sample_weight": float(sample_weight),
        "prompt_messages": prompt_messages,
        "chosen_messages": chosen_messages,
        "rejected_messages": rejected_messages,
        "prompt": _render_prompt_text(record, config),
        "chosen": _render_completion_text(chosen_cand, teacher_label, config, role_label="chosen"),
        "rejected": _render_completion_text(rejected_cand, teacher_label, config, role_label="rejected"),
        "chosen_action": chosen_cand["action"],
        "rejected_action": rejected_cand["action"],
        "chosen_candidate": {
            **chosen_cand,
            "teacher_evidence": _candidate_teacher_entry(
                teacher_label,
                chosen_cand,
                field="candidate_evidence",
            ),
        },
        "rejected_candidate": {
            **rejected_cand,
            "teacher_evidence": _candidate_teacher_entry(
                teacher_label,
                rejected_cand,
                field="candidate_evidence",
            ),
        },
        "metadata": {
            "dataset": record["dataset"],
            "boundary_type": record["boundary_type"],
            "delta_u_rel": selection_gap,
            "chosen_u_rel": chosen_cand["u_rel"],
            "rejected_u_rel": rejected_cand["u_rel"],
            "state_id": state_id,
            "pair_id": pair_id,
            "pair_kind": pair_kind,
            "sample_weight": float(sample_weight),
            "augmentation_tier": augmentation_tier,
            "teacher_source": None if teacher_label is None else teacher_label.get("source"),
            "prompt_mode": prompt_mode,
            "state_reasoning_attempt": _record_reasoning_attempt(record),
            "chosen_valid_candidate": True,
            "rejected_valid_candidate": bool(is_valid_candidate(rejected_cand)),
            "rejected_candidate_status": rejected_cand.get("candidate_status"),
            "rejected_valid_for_chosen": bool(valid_as_chosen(rejected_cand)),
            "rejected_valid_for_rejected": bool(valid_as_rejected(rejected_cand)),
            "min_utility_gap": min_utility_gap,
            "chosen_schema_diagnostics": chosen_cand.get("schema_diagnostics", []),
            "rejected_schema_diagnostics": rejected_cand.get("schema_diagnostics", []),
            "active_semantic_tags": record.get("active_semantic_tags", []),
            "semantic_tag_evidence": record.get("semantic_tag_evidence", {}),
            "process_features": record.get("process_features", {}),
            "teacher_guardrail_summary": None
            if teacher_label is None
            else teacher_label.get("guardrail_summary", {}),
            "chosen_candidate_evidence": _candidate_teacher_entry(
                teacher_label,
                chosen_cand,
                field="candidate_evidence",
            ),
            "rejected_candidate_evidence": _candidate_teacher_entry(
                teacher_label,
                rejected_cand,
                field="candidate_evidence",
            ),
        },
    }


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
        valid_as_rejected(candidate)
        and (bool(required_input_value(candidate)) or is_empty_answer_rejected_only(candidate))
        and not bool(candidate.get("is_debug_fallback", False))
        and bool(candidate.get("is_student_candidate", True))
    )


def _pair_candidate_valid_as_chosen(candidate: dict[str, Any]) -> bool:
    return valid_as_chosen(candidate) and bool(required_input_value(candidate))


def _min_utility_gap_for_record(
    record: dict[str, Any],
    chosen_action: str,
    pair_cfg: dict[str, Any],
) -> float:
    default = float(pair_cfg.get("min_utility_gap_default", pair_cfg.get("min_utility_gap", 0.2)))
    by_dataset = pair_cfg.get("min_utility_gap_by_dataset") or {}
    by_action = pair_cfg.get("min_utility_gap_by_chosen_action") or {}
    dataset_gap = by_dataset.get(record.get("dataset"))
    action_gap = by_action.get(chosen_action)
    gaps = [default]
    if dataset_gap is not None:
        gaps.append(float(dataset_gap))
    if action_gap is not None:
        gaps.append(float(action_gap))
    return min(gaps)


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
    required_teacher_sources: Iterable[str] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Build Step-DPO pairs from student top-k candidates ranked by U_rel.

    Chosen = argmax(U_rel) with rubric_degenerate=False.
    Rejected = min(U_rel) with different action type, gap >= min_utility_gap.
    Pairs are truncated at action emission (no tool obs/finalize).
    """
    teacher_by_state = _teacher_lookup(teacher_labels)
    pair_cfg = config.get("pair_construction", {})
    allowed_teacher_sources = {str(source) for source in (required_teacher_sources or []) if str(source)}
    if require_poe_teacher:
        allowed_teacher_sources.add(str(config.get("teacher", {}).get("source_label", "llm_teacher")))
    require_teacher_label = require_teacher_label or bool(allowed_teacher_sources)
    require_different_action_types = bool(pair_cfg.get("require_different_action_types", True))
    eval_datasets = set(config["datasets"]["eval"])

    train_pairs: list[dict[str, Any]] = []
    eval_pairs: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    augmentation_cfg = _pair_augmentation_cfg(config)
    near_tie_cfg = dict(augmentation_cfg.get("near_tie") or {})
    best_mid_cfg = dict(augmentation_cfg.get("best_mid") or {})
    near_tie_candidates: list[dict[str, Any]] = []
    best_mid_candidates: list[dict[str, Any]] = []

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
        if allowed_teacher_sources and (
            teacher_label is None or str(teacher_label.get("source")) not in allowed_teacher_sources
        ):
            diagnostics.append({
                **diagnostic_base,
                "reason": "teacher_source_not_allowed",
                "teacher_source": None if teacher_label is None else teacher_label.get("source"),
                "allowed_teacher_sources": sorted(allowed_teacher_sources),
            })
            counters["dropped"] = len(diagnostics)
            try:
                progress.set_postfix(**counters)
            except Exception:
                pass
            continue
        if teacher_label is not None:
            teacher_label = apply_evidence_score_guardrail(
                teacher_label,
                dataset=str(record.get("dataset", "")),
                record=record,
            )
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
        if len(candidates) < 2 or not any(_pair_candidate_valid_as_chosen(candidate) for candidate in candidates):
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
        if len(scored) < 2 or not any(_pair_candidate_valid_as_chosen(candidate) for candidate in scored):
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

        chosen_probe = max(
            [candidate for candidate in scored if _pair_candidate_valid_as_chosen(candidate)],
            key=lambda candidate: candidate["u_rel"],
        )
        min_utility_gap = _min_utility_gap_for_record(record, chosen_probe["action"], pair_cfg)
        pair = _pick_chosen_rejected(
            scored,
            teacher_label,
            min_utility_gap=min_utility_gap,
            require_different_action_types=require_different_action_types,
        )
        if pair is None:
            if pair_augmentation_enabled(config) and bool(near_tie_cfg.get("enabled", False)):
                near_tie = _near_tie_pair_candidate(record, scored, teacher_label, near_tie_cfg)
                if near_tie is not None:
                    near_tie_candidates.append(near_tie)
            diagnostics.append({
                **diagnostic_base,
                "reason": "no_valid_pair_after_schema_filter",
                "rubric_degenerate": bool(teacher_label and teacher_label.get("rubric_degenerate")),
                "n_candidates": len(scored),
                "actions": [c["action"] for c in scored],
                "u_rels": [c["u_rel"] for c in scored],
                "min_utility_gap": min_utility_gap,
            })
            counters["dropped"] = len(diagnostics)
            try:
                progress.set_postfix(**counters)
            except Exception:
                pass
            continue

        chosen_cand, rejected_cand = pair
        dpo_pair = _make_dpo_pair(
            record,
            teacher_label,
            config,
            chosen_cand,
            rejected_cand,
            min_utility_gap=min_utility_gap,
            pair_kind="strong",
            sample_weight=1.0,
        )
        if _assign_split(record, eval_datasets) == "eval":
            eval_pairs.append(dpo_pair)
        else:
            train_pairs.append(dpo_pair)
        if pair_augmentation_enabled(config) and bool(best_mid_cfg.get("enabled", False)):
            best_mid = _best_mid_pair_candidate(record, scored, teacher_label, best_mid_cfg)
            if best_mid is not None:
                best_mid_candidates.append(best_mid)
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

    if pair_augmentation_enabled(config):
        augmented_pairs: list[dict[str, Any]] = []
        if bool(near_tie_cfg.get("enabled", False)):
            near_tie_candidates.sort(
                key=lambda item: (
                    0 if item.get("tier") == "primary" else 1,
                    -float(item["sort_gap"]),
                    item["sort_key"],
                )
            )
            selected_near_tie = _select_augmented_candidates(
                near_tie_candidates,
                max_pairs=int(near_tie_cfg.get("max_pairs", 0)),
                dataset_quota=near_tie_cfg.get("dataset_quota") or {},
                fill_remaining=bool(near_tie_cfg.get("fill_remaining", True)),
            )
            for item in selected_near_tie:
                augmented_pairs.append(_make_dpo_pair(
                    item["record"],
                    item["teacher_label"],
                    config,
                    item["chosen"],
                    item["rejected"],
                    min_utility_gap=float(item["gap"]),
                    pair_kind="near_tie",
                    sample_weight=float(near_tie_cfg.get("sample_weight", 0.4)),
                    augmentation_tier=str(item.get("tier") or "near_tie"),
                ))
        if bool(best_mid_cfg.get("enabled", False)):
            best_mid_candidates.sort(
                key=lambda item: (
                    -float(item["sort_gap"]),
                    item["sort_key"],
                )
            )
            selected_best_mid = _select_augmented_candidates(
                best_mid_candidates,
                max_pairs=int(best_mid_cfg.get("max_pairs", 0)),
                dataset_quota=best_mid_cfg.get("dataset_quota") or {},
                fill_remaining=bool(best_mid_cfg.get("fill_remaining", True)),
            )
            for item in selected_best_mid:
                augmented_pairs.append(_make_dpo_pair(
                    item["record"],
                    item["teacher_label"],
                    config,
                    item["chosen"],
                    item["rejected"],
                    min_utility_gap=float(item["gap"]),
                    pair_kind="best_mid",
                    sample_weight=float(best_mid_cfg.get("sample_weight", 0.7)),
                    augmentation_tier=str(item.get("tier") or "best_mid"),
                ))
        for dpo_pair in augmented_pairs:
            if _assign_split(dpo_pair, eval_datasets) == "eval":
                eval_pairs.append(dpo_pair)
            else:
                train_pairs.append(dpo_pair)

    all_pairs = _post_filter_pair_quota(train_pairs + eval_pairs, config)
    train_pairs = []
    eval_pairs = []
    for dpo_pair in all_pairs:
        if _assign_split(dpo_pair, eval_datasets) == "eval":
            eval_pairs.append(dpo_pair)
        else:
            train_pairs.append(dpo_pair)

    counters["produced"] = len(train_pairs) + len(eval_pairs)
    counters["dropped"] = len(diagnostics)
    logger.info(
        "build_step_dpo_pairs: train=%d eval=%d diagnostics=%d",
        len(train_pairs), len(eval_pairs), len(diagnostics),
    )
    return train_pairs, eval_pairs, diagnostics
