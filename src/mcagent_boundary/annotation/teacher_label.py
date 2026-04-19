from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from mcagent_boundary.annotation.poe_client import PoeChatClient


PROMPT_ROOT = Path(__file__).resolve().parents[1] / "prompts"
ACTION_SET = {"ANSWER", "SEARCH", "CALCULATE", "CLARIFY", "REFUSE"}


def _read_prompt(name: str) -> str:
    return (PROMPT_ROOT / name).read_text(encoding="utf-8")


def _validate_teacher_payload(payload: dict[str, Any], fallback_action: str) -> dict[str, Any]:
    semantic_tags = payload.get("semantic_tags")
    if not isinstance(semantic_tags, list):
        semantic_tags = []
    meta_reflection = str(payload.get("meta_reflection", "")).strip()[:200]
    if not meta_reflection:
        meta_reflection = "I should calibrate my next action to the current evidence and uncertainty."
    recommended_action = str(payload.get("recommended_action", fallback_action)).upper()
    if recommended_action not in ACTION_SET:
        recommended_action = fallback_action
    rationale = str(payload.get("rationale", "")).strip()
    if not rationale:
        rationale = f"The current state favors {recommended_action} over weaker alternatives."
    return {
        "semantic_tags": [str(tag) for tag in semantic_tags],
        "meta_reflection": meta_reflection,
        "recommended_action": recommended_action,
        "rationale": rationale,
    }


def _fallback_label(record: dict[str, Any]) -> dict[str, Any]:
    best_action = str(record.get("best_action", "ANSWER")).upper()
    tags = list(record.get("active_semantic_tags", []))
    return {
        "semantic_tags": tags,
        "meta_reflection": f"I should choose {best_action} because it best matches the current uncertainty.",
        "recommended_action": best_action,
        "rationale": f"The highest local utility branch for this state is {best_action}.",
    }


def label_boundary_records(records: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    system_prompt = _read_prompt("teacher_tag_reflect.md")
    user_prompt_template = _read_prompt("teacher_action_recommend.md")
    client = PoeChatClient(config)
    labeled: list[dict[str, Any]] = []
    for record in records:
        branch_summaries = "\n".join(record.get("branch_summaries", []))
        fallback = _fallback_label(record)
        user_prompt = user_prompt_template.format(
            question=record["question"],
            dataset=record["dataset"],
            boundary_type=record["boundary_type"],
            reason_prefix=record.get("reason_prefix", ""),
            process_features=json.dumps(record.get("process_features", {}), ensure_ascii=False),
            semantic_hints=json.dumps(record.get("active_semantic_tags", []), ensure_ascii=False),
            branch_summaries=branch_summaries,
        )
        payload = fallback
        source = "rule_fallback"
        if client.is_ready():
            try:
                payload = _validate_teacher_payload(
                    client.complete_json(system_prompt=system_prompt, user_prompt=user_prompt),
                    fallback_action=fallback["recommended_action"],
                )
                source = "poe_teacher"
            except Exception as exc:
                payload = {**fallback, "rationale": f"{fallback['rationale']} Teacher call failed: {exc}"}
        labeled.append(
            {
                "state_id": record["state_id"],
                "example_id": record["example_id"],
                "dataset": record["dataset"],
                "boundary_type": record["boundary_type"],
                "question": record["question"],
                "source": source,
                **payload,
            }
        )
    return labeled

