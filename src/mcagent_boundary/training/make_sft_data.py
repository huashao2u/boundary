from __future__ import annotations

import json
from typing import Any


def _flatten_messages(messages: list[dict[str, str]]) -> str:
    return "\n".join(f"{message['role'].upper()}: {message['content']}" for message in messages)


def _render_text(prompt_messages: list[dict[str, str]], completion_messages: list[dict[str, str]]) -> str:
    prompt_text = (
        f"SYSTEM: {prompt_messages[0]['content']}\n"
        f"USER: {prompt_messages[1]['content']}\n"
        "ASSISTANT:"
    )
    completion_text = (
        f" {completion_messages[0]['content']}\n"
        f"{completion_messages[1]['content']}"
    )
    return prompt_text + completion_text


def _teacher_lookup(teacher_labels: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {record["state_id"]: record for record in teacher_labels}


def build_warmup_sft_dataset(
    selected_records: list[dict[str, Any]],
    teacher_labels: list[dict[str, Any]],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    teacher_by_state = _teacher_lookup(teacher_labels)
    warmup_cfg = config["warmup"]
    candidates = [
        record
        for record in selected_records
        if record.get("pool") in {"boundary_critical", "clear_external_anchor"}
    ]
    target_limit = int(warmup_cfg["max_examples"])
    records = sorted(candidates, key=lambda item: float(item.get("utility_gap", 0.0)), reverse=True)[:target_limit]
    dataset: list[dict[str, Any]] = []
    for record in records:
        teacher = teacher_by_state.get(record["state_id"])
        if teacher is None:
            continue
        branch = next((item for item in record["branches"] if item["action"] == teacher["recommended_action"]), None)
        if branch is None:
            branch = max(record["branches"], key=lambda item: float(item["utility"]))
        prompt_messages = [
            {"role": "system", "content": "You are a decision-aware assistant. Predict the best next action."},
            {
                "role": "user",
                "content": (
                    f"Question: {record['question']}\n"
                    f"Dataset: {record['dataset']}\n"
                    f"Boundary type: {record['boundary_type']}\n"
                    f"Reasoning Prefix: {record.get('reason_prefix', '')}"
                ),
            },
        ]
        completion_messages = [
            {"role": "assistant", "content": f"Meta-Reflection: {teacher['meta_reflection']}"},
            {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "action": teacher["recommended_action"],
                        "action_input": branch.get("action_input", {}),
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        dataset.append(
            {
                "state_id": record["state_id"],
                "dataset": record["dataset"],
                "prompt_messages": prompt_messages,
                "completion_messages": completion_messages,
                "messages": prompt_messages + completion_messages,
                "text": _render_text(prompt_messages, completion_messages),
            }
        )
    return dataset
