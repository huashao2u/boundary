from __future__ import annotations

import concurrent.futures
from collections import deque
import json
import logging
from pathlib import Path
import threading
import time
from typing import Any, Callable

from mcagent_boundary.annotation.openai_compatible_client import OpenAICompatibleChatClient
from mcagent_boundary.annotation.validate_teacher_evidence import (
    apply_evidence_score_guardrail,
    attach_auto_evidence_to_candidates,
    build_auto_candidate_evidence,
)
from mcagent_boundary.rollout.candidate_schema import is_valid_candidate, valid_as_rejected


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
# with --strict-teacher and should require the configured teacher source.

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


def _rule_fallback_candidate_evidence(
    candidates: list[dict[str, Any]],
    record: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Best-effort evidence for rule fallback / smoke paths."""
    record = record or {}
    evidence_items = []
    for candidate in candidates:
        action = str(candidate.get("action", "")).upper()
        action_input = candidate.get("canonical_action_input") or candidate.get("action_input") or {}
        auto_evidence = build_auto_candidate_evidence(
            candidate,
            gold_answer=record.get("gold_answer"),
            dataset=str(record.get("dataset", "")),
            metadata=dict((record.get("metadata") or {})),
        )
        evidence = {
            "rank": candidate.get("rank"),
            "action": action,
            "evidence_summary": "Rule fallback evidence; use only for smoke/debug labels.",
            **auto_evidence,
        }
        if action == "ANSWER":
            answer = str(action_input.get("answer", ""))
            evidence.update({
                "payload_answer": answer,
                "rationale_answer": "",
                "payload_answer_type": "empty" if not answer.strip() else "unknown",
                "payload_answer_correct": auto_evidence.get("auto_payload_answer_correct"),
                "rationale_answer_correct": None,
                "payload_rationale_conflict": False,
            })
        elif action == "CALCULATE":
            evidence.update({
                "expression": str(action_input.get("expression", "")),
                "expression_relevance": "direct_final"
                if auto_evidence.get("auto_expression_matches_gold")
                else "unknown",
                "expression_matches_gold": auto_evidence.get("auto_expression_matches_gold"),
            })
        elif action == "SEARCH":
            evidence["query_specific_and_relevant"] = None
        elif action == "CLARIFY":
            evidence["targets_critical_slot"] = None
        elif action == "REFUSE":
            evidence["should_refuse"] = (record.get("metadata") or {}).get("should_refuse")
        evidence_items.append(evidence)
    return evidence_items


def _rule_fallback_candidate_reflection(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "rank": candidate.get("rank"),
            "action": str(candidate.get("action", "")).upper(),
            "reflection": str(candidate.get("brief_rationale") or "").strip()
            or f"I considered {str(candidate.get('action', 'ACTION')).upper()} under the current uncertainty.",
        }
        for candidate in candidates
    ]


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
    record: dict[str, Any] | None = None,
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

    # candidate_utility validation. candidate_helpfulness is accepted only as a
    # backward-compatible alias for older teacher outputs.
    raw_helpfulness = payload.get("candidate_utility") or payload.get("candidate_helpfulness") or []
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
            failure_mode = str(entry.get("failure_mode", "")).strip()[:80]
            normalized = {
                "rank": rank,
                "action": action,
                "score": round(score, 1),
                "reason": reason,
                "failure_mode": failure_mode,
            }
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

    raw_evidence = payload.get("candidate_evidence") or []
    evidence_by_key: dict[tuple[int | None, str], dict[str, Any]] = {}
    unmatched_evidence_by_action: dict[str, list[dict[str, Any]]] = {}
    if isinstance(raw_evidence, list) and raw_evidence:
        for entry in raw_evidence:
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
            normalized = {
                **entry,
                "rank": rank,
                "action": action,
                "evidence_summary": str(entry.get("evidence_summary", "")).strip()[:240],
            }
            if rank is None:
                unmatched_evidence_by_action.setdefault(action, []).append(normalized)
            else:
                evidence_by_key[(rank, action)] = normalized

    fallback_evidence = {
        (entry.get("rank"), entry["action"]): entry
        for entry in _rule_fallback_candidate_evidence(candidates, record)
    }
    candidate_evidence: list[dict[str, Any]] = []
    for candidate in candidates:
        action = str(candidate.get("action", "")).upper()
        rank_raw = candidate.get("rank")
        try:
            rank = None if rank_raw is None else int(rank_raw)
        except (TypeError, ValueError):
            rank = None
        key = (rank, action)
        entry = evidence_by_key.get(key)
        if entry is None and unmatched_evidence_by_action.get(action):
            entry = unmatched_evidence_by_action[action].pop(0)
            entry = {**entry, "rank": rank}
        if entry is None:
            if not allow_score_fallback:
                raise ValueError(f"teacher_missing_candidate_evidence rank={rank} action={action}")
            entry = fallback_evidence.get(key) or {"rank": rank, "action": action, "evidence_summary": ""}
        else:
            entry = {**(fallback_evidence.get(key) or {}), **entry}
        candidate_evidence.append(entry)

    raw_reflection = payload.get("candidate_reflection") or []
    reflection_by_key: dict[tuple[int | None, str], dict[str, Any]] = {}
    unmatched_reflection_by_action: dict[str, list[dict[str, Any]]] = {}
    if isinstance(raw_reflection, list) and raw_reflection:
        for entry in raw_reflection:
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
            normalized = {
                "rank": rank,
                "action": action,
                "reflection": str(entry.get("reflection", "")).strip()[:240],
            }
            if rank is None:
                unmatched_reflection_by_action.setdefault(action, []).append(normalized)
            else:
                reflection_by_key[(rank, action)] = normalized
    fallback_reflection = {
        (entry.get("rank"), entry["action"]): entry
        for entry in _rule_fallback_candidate_reflection(candidates)
    }
    candidate_reflection: list[dict[str, Any]] = []
    for candidate in candidates:
        action = str(candidate.get("action", "")).upper()
        rank_raw = candidate.get("rank")
        try:
            rank = None if rank_raw is None else int(rank_raw)
        except (TypeError, ValueError):
            rank = None
        key = (rank, action)
        entry = reflection_by_key.get(key)
        if entry is None and unmatched_reflection_by_action.get(action):
            entry = unmatched_reflection_by_action[action].pop(0)
            entry = {**entry, "rank": rank}
        if entry is None:
            entry = fallback_reflection.get(key) or {
                "rank": rank,
                "action": action,
                "reflection": f"I considered {action} under the current uncertainty.",
            }
        if not entry.get("reflection"):
            entry["reflection"] = f"I considered {action} under the current uncertainty."
        candidate_reflection.append(entry)

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
        "candidate_evidence": candidate_evidence,
        "recommended_action": recommended_action,
        "rationale": rationale,
        "candidate_utility": candidate_helpfulness,
        "candidate_helpfulness": candidate_helpfulness,
        "candidate_reflection": candidate_reflection,
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
    candidates = [c for c in list(record.get("candidates", [])) if valid_as_rejected(c)]

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
        "candidate_evidence": _rule_fallback_candidate_evidence(candidates, record),
        "recommended_action": best_action,
        "rationale": f"The highest local utility branch for this state is {best_action}.",
        "candidate_utility": candidate_helpfulness,
        "candidate_helpfulness": candidate_helpfulness,
        "candidate_reflection": _rule_fallback_candidate_reflection(candidates),
        "preferred_over": preferred_over,
        "rubric_degenerate": False,
    }


class _RequestRateLimiter:
    def __init__(self, requests_per_minute: int | None) -> None:
        self.requests_per_minute = int(requests_per_minute or 0)
        self._window_seconds = 60.0
        self._timestamps: deque[float] = deque()
        self._lock = threading.Lock()

    def acquire(self) -> None:
        if self.requests_per_minute <= 0:
            return
        while True:
            with self._lock:
                now = time.monotonic()
                while self._timestamps and now - self._timestamps[0] >= self._window_seconds:
                    self._timestamps.popleft()
                if len(self._timestamps) < self.requests_per_minute:
                    self._timestamps.append(now)
                    return
                sleep_for = self._window_seconds - (now - self._timestamps[0])
            time.sleep(max(0.05, sleep_for))


def _label_one_record(
    record: dict[str, Any],
    *,
    system_prompt: str,
    user_prompt_template: str,
    client: OpenAICompatibleChatClient,
    allow_rule_fallback: bool,
    rate_limiter: _RequestRateLimiter,
    validation_retries: int,
) -> tuple[dict[str, Any], str]:
    candidates = [c for c in list(record.get("candidates", [])) if valid_as_rejected(c)]
    fallback = _fallback_label(record)

    # Build student_candidates JSON for prompt. Invalid candidates are excluded.
    student_candidates_for_prompt = [
        {
            "rank": c.get("rank", i + 1),
            "action": str(c.get("action", "")).upper(),
            "brief_rationale": str(c.get("brief_rationale", "")),
            "confidence": c.get("confidence"),
            "action_input": c.get("canonical_action_input") or c.get("action_input") or {},
            "action_json_logprob_mean": c.get("action_json_logprob_mean"),
            "action_json_logprob_sum": c.get("action_json_logprob_sum"),
            "action_json_num_tokens": c.get("action_json_num_tokens"),
            "missing_logprob_positions": c.get("missing_logprob_positions"),
            "valid_candidate": bool(c.get("valid_candidate")),
            "candidate_status": c.get("candidate_status"),
            "valid_for_chosen": bool(c.get("valid_for_chosen", is_valid_candidate(c))),
            "valid_for_rejected": bool(c.get("valid_for_rejected", valid_as_rejected(c))),
            "schema_diagnostics": c.get("schema_diagnostics", []),
        }
        for i, c in enumerate(candidates)
    ]
    student_candidates_for_prompt = attach_auto_evidence_to_candidates(
        student_candidates_for_prompt,
        gold_answer=record.get("gold_answer"),
        dataset=str(record.get("dataset", "")),
        metadata=dict((record.get("metadata") or {})),
    )

    dataset_boundary = json.dumps(
        {
            "dataset": record.get("dataset", ""),
            "boundary_type": record.get("boundary_type", ""),
        },
        ensure_ascii=False,
    )

    user_prompt = user_prompt_template.format(
        question=record["question"],
        gold_reference=json.dumps(
            {
                "gold_answer": record.get("gold_answer"),
                "metadata_refusal_label": {
                    "should_refuse": (record.get("metadata") or {}).get("should_refuse"),
                    "or_bench_label": (record.get("metadata") or {}).get("or_bench_label"),
                },
            },
            ensure_ascii=False,
            indent=2,
        ),
        reason_attempt=record.get("reasoning_attempt", record.get("reason_prefix", "")),
        uncertainty_summary=record.get("uncertainty_summary", ""),
        student_candidates=json.dumps(student_candidates_for_prompt, ensure_ascii=False, indent=2),
        process_features=json.dumps(record.get("process_features", {}), ensure_ascii=False),
        semantic_tag_evidence=json.dumps(record.get("semantic_tag_evidence", {}), ensure_ascii=False, indent=2),
        semantic_hints=json.dumps(record.get("active_semantic_tags", []), ensure_ascii=False),
        dataset_boundary=dataset_boundary,
    )
    required_candidates_for_repair = [
        {
            "rank": item["rank"],
            "action": item["action"],
            "action_input": item["action_input"],
            "brief_rationale": item["brief_rationale"],
        }
        for item in student_candidates_for_prompt
    ]

    payload = fallback
    source = "rule_fallback"
    counter_key = "fallback"
    if not client.is_ready() and not allow_rule_fallback:
        raise RuntimeError("Teacher credentials are not configured and rule fallback is disabled.")
    if client.is_ready():
        try:
            validation_prompt = user_prompt
            for attempt in range(validation_retries + 1):
                rate_limiter.acquire()
                raw = client.complete_json(system_prompt=system_prompt, user_prompt=validation_prompt)
                try:
                    payload = _validate_teacher_payload(
                        raw,
                        fallback_action=fallback["recommended_action"],
                        candidates=candidates,
                        allow_score_fallback=allow_rule_fallback,
                        record=record,
                    )
                    payload = apply_evidence_score_guardrail(
                        payload,
                        dataset=str(record.get("dataset", "")),
                        record=record,
                    )
                    break
                except Exception as validation_exc:
                    if attempt >= validation_retries:
                        raise
                    validation_prompt = (
                        "Repair the previous annotation JSON. Return corrected JSON only.\n"
                        f"Validation error: {validation_exc}\n"
                        "The candidate_evidence and candidate_utility arrays must each contain "
                        "exactly one entry for every required candidate below, matching both rank "
                        "and action. candidate_helpfulness and candidate_reflection should also "
                        "cover every required candidate. Do not omit any candidate, even if its "
                        "action looks wrong or redundant.\n"
                        "For candidate_evidence, include evidence_summary plus the relevant "
                        "action-specific fields when available; if the candidate is bad, still "
                        "write evidence explaining why it is bad.\n"
                        "For candidate_utility, include score, reason, and failure_mode.\n"
                        "Required candidates:\n"
                        f"{json.dumps(required_candidates_for_repair, ensure_ascii=False, indent=2)}\n"
                        "Question:\n"
                        f"{record['question']}\n"
                        "Gold/reference for offline judging only:\n"
                        f"{record.get('gold_answer')}\n"
                        "Previous invalid response:\n"
                        f"{json.dumps(raw, ensure_ascii=False)}\n"
                        "Return the full JSON object with semantic_tags, meta_reflection, "
                        "candidate_evidence, candidate_utility, candidate_helpfulness, "
                        "candidate_reflection, recommended_action, rationale, preferred_over, "
                        "and rubric_degenerate."
                    )
            source = client.source_label
            counter_key = "success"
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
            payload = apply_evidence_score_guardrail(
                payload,
                dataset=str(record.get("dataset", "")),
                record=record,
            )
            source = "rule_fallback_after_failure"
            counter_key = "failure"

    if "guardrail_summary" not in payload:
        payload = apply_evidence_score_guardrail(
            payload,
            dataset=str(record.get("dataset", "")),
            record=record,
        )

    return (
        {
            "state_id": record["state_id"],
            "example_id": record["example_id"],
            "dataset": record["dataset"],
            "boundary_type": record["boundary_type"],
            "question": record["question"],
            "gold_answer": record.get("gold_answer"),
            "metadata": record.get("metadata") or {},
            "gold_reference": {
                "gold_answer": record.get("gold_answer"),
                "should_refuse": (record.get("metadata") or {}).get("should_refuse"),
                "or_bench_label": (record.get("metadata") or {}).get("or_bench_label"),
            },
            "source": source,
            "teacher_provider": client.provider,
            "teacher_model": client.model,
            "num_valid_candidates": len(candidates),
            **payload,
        },
        counter_key,
    )


# ---------------------------------------------------------------------------
# Main labeling function
# ---------------------------------------------------------------------------

def label_boundary_records(
    records: list[dict[str, Any]],
    config: dict[str, Any],
    *,
    show_progress: bool = True,
    allow_rule_fallback: bool = True,
    teacher_workers: int | None = None,
    rpm_limit: int | None = None,
    on_label: Callable[[dict[str, Any]], None] | None = None,
    skip_failed: bool = False,
    on_failure: Callable[[dict[str, Any], Exception], None] | None = None,
) -> list[dict[str, Any]]:
    from mcagent_boundary.progress import make_progress

    system_prompt = _read_prompt("teacher_tag_reflect.md")
    user_prompt_template = _read_prompt("teacher_action_recommend.md")
    client = OpenAICompatibleChatClient(config)
    counters = {"success": 0, "fallback": 0, "failure": 0, "skipped": 0}
    teacher_cfg = config.get("teacher", {})
    workers = int(teacher_workers or teacher_cfg.get("workers") or 1)
    workers = max(1, workers)
    rate_limiter = _RequestRateLimiter(rpm_limit or teacher_cfg.get("rpm_limit"))
    validation_retries = max(0, int(teacher_cfg.get("validation_retries", 2)))
    labeled: list[dict[str, Any] | None] = [None] * len(records)

    def _set_postfix(progress: Any) -> None:
        try:
            progress.set_postfix(**counters)
        except Exception:
            pass

    if workers == 1 or len(records) <= 1:
        progress = make_progress(
            list(enumerate(records)),
            total=len(records),
            desc="teacher labeling",
            unit="rec",
            disable=not show_progress,
        )
        try:
            for index, record in progress:
                try:
                    label, counter_key = _label_one_record(
                        record,
                        system_prompt=system_prompt,
                        user_prompt_template=user_prompt_template,
                        client=client,
                        allow_rule_fallback=allow_rule_fallback,
                        rate_limiter=rate_limiter,
                        validation_retries=validation_retries,
                    )
                except Exception as exc:
                    if not skip_failed:
                        raise
                    counters["failure"] += 1
                    if on_failure:
                        on_failure(record, exc)
                    _set_postfix(progress)
                    continue
                labeled[index] = label
                if on_label:
                    on_label(label)
                counters[counter_key] += 1
                _set_postfix(progress)
        finally:
            try:
                progress.close()
            except Exception:
                pass
    else:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_index = {
                executor.submit(
                    _label_one_record,
                    record,
                    system_prompt=system_prompt,
                    user_prompt_template=user_prompt_template,
                    client=client,
                    allow_rule_fallback=allow_rule_fallback,
                    rate_limiter=rate_limiter,
                    validation_retries=validation_retries,
                ): index
                for index, record in enumerate(records)
            }
            progress = make_progress(
                concurrent.futures.as_completed(future_to_index),
                total=len(future_to_index),
                desc=f"teacher labeling ({workers} workers)",
                unit="rec",
                disable=not show_progress,
            )
            try:
                for future in progress:
                    index = future_to_index[future]
                    try:
                        label, counter_key = future.result()
                    except Exception:
                        if not skip_failed:
                            for pending in future_to_index:
                                pending.cancel()
                            raise
                        record = records[index]
                        counters["failure"] += 1
                        if on_failure:
                            on_failure(record, future.exception() or RuntimeError("unknown teacher failure"))
                        _set_postfix(progress)
                        continue
                    labeled[index] = label
                    if on_label:
                        on_label(label)
                    counters[counter_key] += 1
                    _set_postfix(progress)
            finally:
                try:
                    progress.close()
                except Exception:
                    pass
    logger.info("Teacher labeling counters: %s", counters)
    return [label for label in labeled if label is not None]
