from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from mcagent_boundary.annotation.poe_client import PoeChatClient
from mcagent_boundary.rollout.candidate_schema import is_valid_candidate


PROMPT_ROOT = Path(__file__).resolve().parents[1] / "prompts"
ACTION_SET = {"ANSWER", "SEARCH", "CALCULATE", "CLARIFY", "REFUSE"}
logger = logging.getLogger(__name__)


def _read_prompt(name: str) -> str:
    return (PROMPT_ROOT / name).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Rule-fallback helpers
# ---------------------------------------------------------------------------
# DEPRECATED(mainline): rule fallback labels are retained for smoke/offline
# debugging only. Main experiment runs must call 03_teacher_label_boundary.py
# with --strict-teacher and should only accept source == "poe_teacher".

_ACTION_FALLBACK_SCORES: dict[str, dict[str, float]] = {
    # base score per action (before semantic tag adjustment)
    "ANSWER": {"default": 0.6},
    "SEARCH": {"default": 0.4},
    "CALCULATE": {"default": 0.4},
    "CLARIFY": {"default": 0.4},
    "REFUSE": {"default": 0.3},
}

_PENALTY_TAGS = {
    "MISSING_INFO",
    "FALSE_PREMISE",
    "TIME_SENSITIVE",
    "NEW_OR_TAIL_KNOWLEDGE",
    "SEARCH_REQUIRED",
}
_BOOST_MAP: dict[str, str] = {
    "SEARCH": "SEARCH_REQUIRED",
    "CALCULATE": "CALCULATION_REQUIRED",
    "CLARIFY": "CLARIFY_REQUIRED",
    "REFUSE": "FALSE_PREMISE",
}


def _rule_helpfulness_score(action: str, active_tags: set[str]) -> tuple[float, str]:
    """Return (score, reason) for one candidate action from semantic tags only."""
    action = action.upper()
    if action == "ANSWER":
        penalized = _PENALTY_TAGS & active_tags
        if "FALSE_PREMISE" in active_tags:
            return 0.1, "FALSE_PREMISE tag present — direct answer likely wrong"
        if "MISSING_INFO" in active_tags:
            return 0.2, "MISSING_INFO tag present — answer without clarification is risky"
        if "TIME_SENSITIVE" in active_tags or "NEW_OR_TAIL_KNOWLEDGE" in active_tags:
            return 0.3, "External knowledge tag present — direct answer may be stale"
        if penalized:
            return 0.3, f"Conflicting tags present: {', '.join(penalized)}"
        return 0.7, "No blocking tags — direct answer appears feasible"
    if action == "SEARCH":
        if "SEARCH_REQUIRED" in active_tags or "TIME_SENSITIVE" in active_tags or "NEW_OR_TAIL_KNOWLEDGE" in active_tags:
            return 0.8, "External evidence tag present — search is appropriate"
        return 0.3, "No external evidence tag — search may be unnecessary"
    if action == "CALCULATE":
        if "CALCULATION_REQUIRED" in active_tags:
            return 0.8, "CALCULATION_REQUIRED tag present — calculation is appropriate"
        return 0.2, "No CALCULATION_REQUIRED tag — calculation likely unnecessary"
    if action == "CLARIFY":
        if "CLARIFY_REQUIRED" in active_tags or "MISSING_INFO" in active_tags:
            return 0.8, "MISSING_INFO tag present — clarification is appropriate"
        return 0.2, "No MISSING_INFO tag — clarification likely unnecessary"
    if action == "REFUSE":
        if "FALSE_PREMISE" in active_tags or "JUSTIFIED_REFUSE" in active_tags:
            return 0.8, "FALSE_PREMISE or JUSTIFIED_REFUSE tag present — refusal is appropriate"
        if "MISSING_INFO" in active_tags:
            return 0.5, "MISSING_INFO present — refusal may be warranted if clarification unavailable"
        return 0.2, "No refusal-justifying tag — refusal likely inappropriate"
    return 0.5, "unknown action"


def _rule_fallback_candidate_helpfulness(
    candidates: list[dict[str, Any]],
    active_tags: set[str],
) -> list[dict[str, Any]]:
    """Deterministically populate candidate_helpfulness from semantic tags."""
    result = []
    for cand in candidates:
        action = str(cand.get("action", "ANSWER")).upper()
        score, reason = _rule_helpfulness_score(action, active_tags)
        result.append({
            "rank": cand.get("rank"),
            "action": action,
            "score": round(score, 1),
            "reason": reason,
        })
    return result


def _best_action_from_tags(active_tags: set[str], candidates: list[dict[str, Any]]) -> str:
    """Pick best action from semantic tags, restricted to candidate actions."""
    candidate_actions = {str(c.get("action", "")).upper() for c in candidates}
    if "FALSE_PREMISE" in active_tags and "REFUSE" in candidate_actions:
        return "REFUSE"
    if "JUSTIFIED_REFUSE" in active_tags and "REFUSE" in candidate_actions:
        return "REFUSE"
    if ("CLARIFY_REQUIRED" in active_tags or "MISSING_INFO" in active_tags) and "CLARIFY" in candidate_actions:
        return "CLARIFY"
    if (
        "SEARCH_REQUIRED" in active_tags
        or "TIME_SENSITIVE" in active_tags
        or "NEW_OR_TAIL_KNOWLEDGE" in active_tags
    ) and "SEARCH" in candidate_actions:
        return "SEARCH"
    if "CALCULATION_REQUIRED" in active_tags and "CALCULATE" in candidate_actions:
        return "CALCULATE"
    if "ANSWER" in candidate_actions:
        return "ANSWER"
    return (candidates[0]["action"].upper() if candidates else "ANSWER")


# ---------------------------------------------------------------------------
# Teacher payload validation
# ---------------------------------------------------------------------------

def _validate_teacher_payload(
    payload: dict[str, Any],
    fallback_action: str,
    candidates: list[dict[str, Any]],
    *,
    allow_score_fallback: bool = True,
) -> dict[str, Any]:
    semantic_tags = payload.get("semantic_tags")
    if not isinstance(semantic_tags, list):
        semantic_tags = []

    meta_reflection = str(payload.get("meta_reflection", "")).strip()[:200]
    if not meta_reflection:
        meta_reflection = "I should calibrate my next action to the current evidence and uncertainty."

    candidate_actions = {str(candidate.get("action", "")).upper() for candidate in candidates}
    recommended_action = str(payload.get("recommended_action", fallback_action)).upper()
    if recommended_action not in ACTION_SET or (candidate_actions and recommended_action not in candidate_actions):
        recommended_action = fallback_action

    rationale = str(payload.get("rationale", "")).strip()
    if not rationale:
        rationale = f"The current state favors {recommended_action} over weaker alternatives."

    # candidate_helpfulness validation
    raw_helpfulness = payload.get("candidate_helpfulness") or []
    candidate_helpfulness_by_key: dict[tuple[int | None, str], dict[str, Any]] = {}
    unmatched_by_action: dict[str, list[dict[str, Any]]] = {}
    if isinstance(raw_helpfulness, list) and raw_helpfulness:
        for entry in raw_helpfulness:
            if not isinstance(entry, dict):
                continue
            action = str(entry.get("action", "")).upper()
            if action not in ACTION_SET:
                continue
            rank = entry.get("rank")
            try:
                rank = None if rank is None else int(rank)
            except (TypeError, ValueError):
                rank = None
            try:
                score = max(0.0, min(1.0, float(entry.get("score", 0.5))))
            except (TypeError, ValueError):
                score = 0.5
            reason = str(entry.get("reason", "")).strip()[:80]
            normalized = {"rank": rank, "action": action, "score": round(score, 1), "reason": reason}
            if rank is None:
                unmatched_by_action.setdefault(action, []).append(normalized)
            else:
                candidate_helpfulness_by_key[(rank, action)] = normalized

    active_tags = set(semantic_tags)
    fallback_scores = {
        (entry.get("rank"), entry["action"]): entry
        for entry in _rule_fallback_candidate_helpfulness(candidates, active_tags)
    }
    candidate_helpfulness: list[dict[str, Any]] = []
    for candidate in candidates:
        action = str(candidate.get("action", "")).upper()
        rank_raw = candidate.get("rank")
        try:
            rank = None if rank_raw is None else int(rank_raw)
        except (TypeError, ValueError):
            rank = None
        key = (rank, action)
        entry = candidate_helpfulness_by_key.get(key)
        if entry is None and unmatched_by_action.get(action):
            entry = unmatched_by_action[action].pop(0)
            entry = {**entry, "rank": rank}
        if entry is None:
            if not allow_score_fallback:
                raise ValueError(f"teacher_missing_candidate_score rank={rank} action={action}")
            entry = fallback_scores.get(key) or {
                "rank": rank,
                "action": action,
                "score": 0.5,
                "reason": "teacher_missing_candidate_score",
            }
        candidate_helpfulness.append(entry)

    # preferred_over validation
    raw_preferred = payload.get("preferred_over") or []
    preferred_over = [str(a).upper() for a in raw_preferred if str(a).upper() in candidate_actions]

    rubric_degenerate = bool(payload.get("rubric_degenerate", False))
    # Auto-detect degenerate: all scores identical
    scores = [e["score"] for e in candidate_helpfulness]
    if len(scores) > 1 and len(set(scores)) == 1:
        rubric_degenerate = True

    return {
        "semantic_tags": [str(tag) for tag in semantic_tags],
        "meta_reflection": meta_reflection,
        "recommended_action": recommended_action,
        "rationale": rationale,
        "candidate_helpfulness": candidate_helpfulness,
        "preferred_over": preferred_over,
        "rubric_degenerate": rubric_degenerate,
    }


# ---------------------------------------------------------------------------
# Fallback label (no teacher available)
# ---------------------------------------------------------------------------

def _fallback_label(record: dict[str, Any]) -> dict[str, Any]:
    best_action = str(record.get("best_action", "ANSWER")).upper()
    tags = list(record.get("active_semantic_tags", []))
    active_tags = set(tags)
    candidates = [c for c in list(record.get("candidates", [])) if is_valid_candidate(c)]

    # Restrict best_action to actual candidates
    cand_actions = {str(c.get("action", "")).upper() for c in candidates}
    if best_action not in cand_actions:
        best_action = _best_action_from_tags(active_tags, candidates)

    candidate_helpfulness = _rule_fallback_candidate_helpfulness(candidates, active_tags)

    # preferred_over: actions that best_action outranks (all other candidate actions)
    preferred_over = [a for a in cand_actions if a != best_action]

    return {
        "semantic_tags": tags,
        "meta_reflection": f"I should choose {best_action} because it best matches the current uncertainty.",
        "recommended_action": best_action,
        "rationale": f"The highest local utility branch for this state is {best_action}.",
        "candidate_helpfulness": candidate_helpfulness,
        "preferred_over": preferred_over,
        "rubric_degenerate": False,
    }


# ---------------------------------------------------------------------------
# Main labeling function
# ---------------------------------------------------------------------------

def label_boundary_records(
    records: list[dict[str, Any]],
    config: dict[str, Any],
    *,
    show_progress: bool = True,
    allow_rule_fallback: bool = True,
) -> list[dict[str, Any]]:
    from mcagent_boundary.progress import make_progress

    system_prompt = _read_prompt("teacher_tag_reflect.md")
    user_prompt_template = _read_prompt("teacher_action_recommend.md")
    client = PoeChatClient(config)
    labeled: list[dict[str, Any]] = []
    counters = {"success": 0, "fallback": 0, "failure": 0, "skipped": 0}
    progress = make_progress(
        records,
        total=len(records),
        desc="teacher labeling",
        unit="rec",
        disable=not show_progress,
    )
    for record in progress:
        candidates = [c for c in list(record.get("candidates", [])) if is_valid_candidate(c)]
        fallback = _fallback_label(record)

        # Build student_candidates JSON for prompt. Invalid candidates are excluded.
        student_candidates_for_prompt = [
            {
                "rank": c.get("rank", i + 1),
                "action": str(c.get("action", "")).upper(),
                "brief_rationale": str(c.get("brief_rationale", "")),
                "confidence": c.get("confidence"),
                "action_input": c.get("canonical_action_input") or c.get("action_input") or {},
                "valid_candidate": bool(c.get("valid_candidate")),
                "schema_diagnostics": c.get("schema_diagnostics", []),
            }
            for i, c in enumerate(candidates)
        ]

        dataset_boundary = json.dumps(
            {
                "dataset": record.get("dataset", ""),
                "boundary_type": record.get("boundary_type", ""),
            },
            ensure_ascii=False,
        )

        user_prompt = user_prompt_template.format(
            question=record["question"],
            reason_attempt=record.get("reasoning_attempt", record.get("reason_prefix", "")),
            uncertainty_summary=record.get("uncertainty_summary", ""),
            student_candidates=json.dumps(student_candidates_for_prompt, ensure_ascii=False, indent=2),
            process_features=json.dumps(record.get("process_features", {}), ensure_ascii=False),
            semantic_tag_evidence=json.dumps(record.get("semantic_tag_evidence", {}), ensure_ascii=False, indent=2),
            semantic_hints=json.dumps(record.get("active_semantic_tags", []), ensure_ascii=False),
            dataset_boundary=dataset_boundary,
        )

        payload = fallback
        source = "rule_fallback"
        if not client.is_ready() and not allow_rule_fallback:
            raise RuntimeError("Teacher credentials are not configured and rule fallback is disabled.")
        if client.is_ready():
            try:
                raw = client.complete_json(system_prompt=system_prompt, user_prompt=user_prompt)
                payload = _validate_teacher_payload(
                    raw,
                    fallback_action=fallback["recommended_action"],
                    candidates=candidates,
                    allow_score_fallback=allow_rule_fallback,
                )
                source = "poe_teacher"
                counters["success"] += 1
            except Exception as exc:
                logger.warning(
                    "Teacher call failed for example %s: %s", record.get("example_id"), exc
                )
                if not allow_rule_fallback:
                    raise RuntimeError(
                        f"Teacher call failed for example {record.get('example_id')}; "
                        "rule fallback is disabled."
                    ) from exc
                payload = {
                    **fallback,
                    "rationale": f"{fallback['rationale']} Teacher call failed: {exc}",
                }
                source = "rule_fallback_after_failure"
                counters["failure"] += 1
        else:
            counters["fallback"] += 1

        labeled.append(
            {
                "state_id": record["state_id"],
                "example_id": record["example_id"],
                "dataset": record["dataset"],
                "boundary_type": record["boundary_type"],
                "question": record["question"],
                "source": source,
                "num_valid_candidates": len(candidates),
                **payload,
            }
        )
        try:
            progress.set_postfix(**counters)
        except Exception:
            pass
    try:
        progress.close()
    except Exception:
        pass
    logger.info("Teacher labeling counters: %s", counters)
    return labeled
