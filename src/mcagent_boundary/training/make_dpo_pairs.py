from __future__ import annotations

import json
from hashlib import sha1
from typing import Any


def _flatten_messages(messages: list[dict[str, str]]) -> str:
    return "\n".join(f"{message['role'].upper()}: {message['content']}" for message in messages)


def _render_prompt_text(record: dict[str, Any]) -> str:
    tool_list = ", ".join(action for action in [branch["action"] for branch in record["branches"] if branch["action"] != "ANSWER"])
    return (
        "SYSTEM: You are a decision-aware assistant. Choose the next action calibrated to the current boundary state.\n"
        f"USER: Question: {record['question']}\n"
        f"Dataset: {record['dataset']}\n"
        f"Boundary type: {record['boundary_type']}\n"
        f"Available actions: ANSWER{', ' + tool_list if tool_list else ''}\n"
        f"ASSISTANT: Reasoning Prefix: {record.get('reason_prefix', '')}\n"
        "ASSISTANT:"
    )


def _render_completion_text(branch: dict[str, Any], teacher_label: dict[str, Any] | None) -> str:
    reflection = (
        teacher_label.get("meta_reflection")
        if teacher_label is not None
        else f"I should prefer {branch['action']} under the current evidence and uncertainty."
    )
    payload = {
        "action": branch["action"],
        "action_input": branch.get("action_input", {}),
        "outcome_label": branch.get("outcome_label"),
    }
    return f" Meta-Reflection: {reflection}\n{json.dumps(payload, ensure_ascii=False)}"


def _teacher_lookup(teacher_labels: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {record["state_id"]: record for record in teacher_labels}


def _build_prompt_messages(record: dict[str, Any]) -> list[dict[str, str]]:
    tool_list = ", ".join(action for action in [branch["action"] for branch in record["branches"] if branch["action"] != "ANSWER"])
    return [
        {
            "role": "system",
            "content": "You are a decision-aware assistant. Choose the next action calibrated to the current boundary state.",
        },
        {
            "role": "user",
            "content": (
                f"Question: {record['question']}\n"
                f"Dataset: {record['dataset']}\n"
                f"Boundary type: {record['boundary_type']}\n"
                f"Available actions: ANSWER{', ' + tool_list if tool_list else ''}"
            ),
        },
        {
            "role": "assistant",
            "content": f"Reasoning Prefix: {record.get('reason_prefix', '')}",
        },
    ]


def _build_completion_messages(branch: dict[str, Any], teacher_label: dict[str, Any] | None) -> list[dict[str, str]]:
    reflection = (
        teacher_label.get("meta_reflection")
        if teacher_label is not None
        else f"I should prefer {branch['action']} under the current evidence and uncertainty."
    )
    payload = {
        "action": branch["action"],
        "action_input": branch.get("action_input", {}),
        "outcome_label": branch.get("outcome_label"),
    }
    return [
        {"role": "assistant", "content": f"Meta-Reflection: {reflection}"},
        {"role": "assistant", "content": json.dumps(payload, ensure_ascii=False)},
    ]


def _pick_rejected_branch(record: dict[str, Any], chosen_branch: dict[str, Any], min_delta: float) -> dict[str, Any] | None:
    answer_branch = next((branch for branch in record["branches"] if branch["action"] == "ANSWER"), None)
    if (
        answer_branch is not None
        and answer_branch["action"] != chosen_branch["action"]
        and float(chosen_branch["utility"]) - float(answer_branch["utility"]) >= min_delta
    ):
        return answer_branch
    ranked = sorted(record["branches"], key=lambda item: float(item["utility"]))
    rejected = ranked[0] if ranked else None
    if rejected is None or rejected["action"] == chosen_branch["action"]:
        return None
    if float(chosen_branch["utility"]) - float(rejected["utility"]) < min_delta:
        return None
    return rejected


def _assign_split(record: dict[str, Any], eval_datasets: set[str]) -> str:
    if record["dataset"] in eval_datasets:
        return "eval"
    bucket = int(sha1(record["state_id"].encode("utf-8")).hexdigest(), 16) % 10
    return "eval" if bucket == 0 else "train"


def build_step_dpo_pairs(
    selected_records: list[dict[str, Any]],
    teacher_labels: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    teacher_by_state = _teacher_lookup(teacher_labels)
    min_delta = float(config["mining"]["min_delta_u"])
    eval_datasets = set(config["datasets"]["eval"])
    train_pairs: list[dict[str, Any]] = []
    eval_pairs: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []

    for record in selected_records:
        ranked = sorted(record["branches"], key=lambda item: float(item["utility"]), reverse=True)
        if len(ranked) < 2:
            diagnostics.append({"state_id": record["state_id"], "reason": "not_enough_ranked_branches"})
            continue
        chosen_branch = ranked[0]
        rejected_branch = _pick_rejected_branch(record, chosen_branch, min_delta=min_delta)
        if rejected_branch is None:
            diagnostics.append({"state_id": record["state_id"], "reason": "no_clean_rejected_branch"})
            continue
        teacher_label = teacher_by_state.get(record["state_id"])
        prompt_messages = _build_prompt_messages(record)
        chosen_messages = _build_completion_messages(chosen_branch, teacher_label)
        rejected_messages = _build_completion_messages(rejected_branch, teacher_label)
        pair = {
            "state_id": record["state_id"],
            "example_id": record["example_id"],
            "dataset": record["dataset"],
            "boundary_type": record["boundary_type"],
            "pool": record.get("pool", "boundary_critical"),
            "prompt_messages": prompt_messages,
            "chosen_messages": chosen_messages,
            "rejected_messages": rejected_messages,
            "prompt": _render_prompt_text(record),
            "chosen": _render_completion_text(chosen_branch, teacher_label),
            "rejected": _render_completion_text(rejected_branch, teacher_label),
            "chosen_action": chosen_branch["action"],
            "rejected_action": rejected_branch["action"],
            "metadata": {
                "dataset": record["dataset"],
                "boundary_type": record["boundary_type"],
                "delta_u": float(chosen_branch["utility"]) - float(rejected_branch["utility"]),
                "chosen_u": float(chosen_branch["utility"]),
                "rejected_u": float(rejected_branch["utility"]),
                "state_id": record["state_id"],
                "teacher_source": None if teacher_label is None else teacher_label.get("source"),
            },
        }
        if _assign_split(record, eval_datasets) == "eval":
            eval_pairs.append(pair)
        else:
            train_pairs.append(pair)
    return train_pairs, eval_pairs, diagnostics
