from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mcagent_boundary.io import read_jsonl, write_jsonl


ERROR_TYPES = [
    "none",
    "wrong_answer_payload_high_score",
    "correct_calculate_low_score",
    "answer_is_expression",
    "payload_rationale_conflict",
    "teacher_ignored_payload",
    "over_refuse",
    "bad_search_query_high_score",
    "bad_clarify_question_high_score",
    "unsafe_answer_high_score",
    "pairwise_disagreement",
]


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=False)


def _index_by_state(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    return {str(record.get("state_id")): record for record in records if record.get("state_id")}


def _candidate_key(candidate: dict[str, Any] | None) -> tuple[int | None, str]:
    if not candidate:
        return None, ""
    rank = candidate.get("rank")
    try:
        rank = None if rank is None else int(rank)
    except (TypeError, ValueError):
        rank = None
    return rank, str(candidate.get("action", "")).upper()


def _candidate_by_pair_role(
    rollout: dict[str, Any] | None,
    evidence: dict[str, Any] | None,
    action: str,
) -> dict[str, Any] | None:
    candidates = list((rollout or {}).get("candidates") or [])
    evidence_rank, evidence_action = _candidate_key(evidence)
    action = str(action).upper()
    fallback = None
    for candidate in candidates:
        candidate_rank, candidate_action = _candidate_key(candidate)
        if candidate_action != action:
            continue
        if fallback is None:
            fallback = candidate
        if evidence_rank is not None and candidate_rank == evidence_rank:
            return candidate
        if evidence_action and candidate_action == evidence_action:
            fallback = candidate
    return fallback


def _payload_conflict(evidence: dict[str, Any] | None) -> bool:
    return bool((evidence or {}).get("payload_rationale_conflict"))


def _answer_is_expression(evidence: dict[str, Any] | None) -> bool:
    return str((evidence or {}).get("payload_answer_type", "")).lower() == "expression"


def _review_focus(pair: dict[str, Any], teacher_label: dict[str, Any] | None) -> list[str]:
    metadata = pair.get("metadata") or {}
    dataset = str(pair.get("dataset", ""))
    chosen_action = str(pair.get("chosen_action", "")).upper()
    rejected_action = str(pair.get("rejected_action", "")).upper()
    chosen_ev = metadata.get("chosen_candidate_evidence") or {}
    rejected_ev = metadata.get("rejected_candidate_evidence") or {}
    flags: list[str] = []

    if dataset in {"gsm8k", "math"} and chosen_action == "ANSWER":
        flags.append("gsm8k/math ANSWER chosen")
    if rejected_action == "CALCULATE":
        flags.append("CALCULATE rejected")
    if _payload_conflict(chosen_ev) or _payload_conflict(rejected_ev):
        flags.append("payload_rationale_conflict")
    if _answer_is_expression(chosen_ev) or _answer_is_expression(rejected_ev):
        flags.append("answer_is_expression")
    if chosen_action in {"SEARCH", "CLARIFY", "REFUSE"}:
        flags.append(f"{chosen_action} chosen")
    if dataset == "or_bench":
        semantic_tags = set(teacher_label.get("semantic_tags") or []) if teacher_label else set()
        if chosen_action == "REFUSE" and "BENIGN" in semantic_tags:
            flags.append("OR-Bench benign REFUSE")
        if chosen_action == "ANSWER" and "UNSAFE" in semantic_tags:
            flags.append("OR-Bench toxic ANSWER")
    gap = metadata.get("delta_u_rel")
    try:
        if gap is not None and float(gap) <= 0.2:
            flags.append("low_utility_gap")
    except (TypeError, ValueError):
        pass
    return flags


def _label_config_xml() -> str:
    error_choices = "\n".join(f'    <Choice value="{value}"/>' for value in ERROR_TYPES)
    return f"""<View>
  <Style>
    .block {{ margin: 12px 0; padding: 10px; border: 1px solid #ddd; border-radius: 4px; }}
    .meta {{ color: #555; font-size: 13px; }}
    .warn {{ color: #8a4b00; font-weight: 600; }}
    pre {{ white-space: pre-wrap; }}
  </Style>

  <Header value="v0.2.5 Pair Review"/>
  <View className="block">
    <Text name="meta" value="Pair: $pair_id | Dataset: $dataset | Boundary: $boundary_type | Pool: $pool"/>
    <Text name="ids" value="Example: $example_id | State: $state_id"/>
    <Text name="focus" value="Review focus: $review_focus"/>
  </View>

  <View className="block">
    <Header value="Question"/>
    <Text name="question" value="$question"/>
    <Header value="Gold / Reference"/>
    <Text name="gold_answer" value="$gold_answer"/>
  </View>

  <View className="block">
    <Header value="Student Context"/>
    <Text name="student_reasoning_attempt" value="$student_reasoning_attempt"/>
    <Text name="uncertainty_summary" value="$uncertainty_summary"/>
    <Header value="Student Candidates JSON"/>
    <Text name="student_candidates_json" value="$student_candidates_json"/>
  </View>

  <View className="block">
    <Header value="Teacher Self-Evidence Label"/>
    <Text name="teacher_action" value="Recommended action: $teacher_recommended_action"/>
    <Text name="teacher_rationale" value="$teacher_rationale"/>
    <Header value="Candidate Evidence"/>
    <Text name="teacher_candidate_evidence_json" value="$teacher_candidate_evidence_json"/>
    <Header value="Candidate Utility"/>
    <Text name="teacher_candidate_utility_json" value="$teacher_candidate_utility_json"/>
    <Header value="Guardrail Summary"/>
    <Text name="teacher_guardrail_summary_json" value="$teacher_guardrail_summary_json"/>
  </View>

  <View className="block">
    <Header value="Constructed Pair"/>
    <Text name="pair_actions" value="Chosen: $chosen_action | Rejected: $rejected_action | Gap: $delta_u_rel"/>
    <Header value="Chosen Candidate"/>
    <Text name="chosen_candidate_json" value="$chosen_candidate_json"/>
    <Header value="Chosen Completion"/>
    <Text name="chosen_text" value="$chosen_text"/>
    <Header value="Rejected Candidate"/>
    <Text name="rejected_candidate_json" value="$rejected_candidate_json"/>
    <Header value="Rejected Completion"/>
    <Text name="rejected_text" value="$rejected_text"/>
    <Header value="Pair Metadata"/>
    <Text name="pair_metadata_json" value="$pair_metadata_json"/>
  </View>

  <Header value="Human Review"/>
  <Choices name="human_pair_accept" toName="pair_actions" choice="single" showInLine="true" required="true">
    <Choice value="true"/>
    <Choice value="false"/>
    <Choice value="unclear"/>
  </Choices>
  <Choices name="teacher_error" toName="teacher_candidate_utility_json" choice="single" showInLine="true" required="true">
    <Choice value="false"/>
    <Choice value="true"/>
    <Choice value="unclear"/>
  </Choices>
  <Choices name="error_type" toName="pair_actions" choice="single" showInLine="false" required="true">
{error_choices}
  </Choices>
  <TextArea name="notes" toName="pair_actions" rows="4" placeholder="Optional notes"/>
</View>
"""


def _task_from_pair(
    pair: dict[str, Any],
    *,
    index: int,
    rollout: dict[str, Any] | None,
    teacher_label: dict[str, Any] | None,
) -> dict[str, Any]:
    metadata = pair.get("metadata") or {}
    chosen_ev = metadata.get("chosen_candidate_evidence") or {}
    rejected_ev = metadata.get("rejected_candidate_evidence") or {}
    chosen_candidate = _candidate_by_pair_role(rollout, chosen_ev, pair.get("chosen_action", ""))
    rejected_candidate = _candidate_by_pair_role(rollout, rejected_ev, pair.get("rejected_action", ""))
    pair_id = f"{pair.get('dataset', 'unknown')}-{index:04d}"
    focus = _review_focus(pair, teacher_label)

    data = {
        "pair_id": pair_id,
        "review_id": pair_id,
        "state_id": pair.get("state_id", ""),
        "example_id": pair.get("example_id", ""),
        "dataset": pair.get("dataset", ""),
        "boundary_type": pair.get("boundary_type", ""),
        "pool": pair.get("pool", ""),
        "review_focus": ", ".join(focus) if focus else "general",
        "question": (rollout or {}).get("question") or _question_from_prompt(pair.get("prompt", "")),
        "gold_answer": (rollout or {}).get("gold_answer", ""),
        "student_reasoning_attempt": (rollout or {}).get("reasoning_attempt")
        or (rollout or {}).get("reason_prefix", ""),
        "uncertainty_summary": (rollout or {}).get("uncertainty_summary", ""),
        "student_candidates_json": _json_dumps((rollout or {}).get("candidates") or []),
        "teacher_recommended_action": (teacher_label or {}).get("recommended_action", ""),
        "teacher_rationale": (teacher_label or {}).get("rationale", ""),
        "teacher_candidate_evidence_json": _json_dumps((teacher_label or {}).get("candidate_evidence") or []),
        "teacher_candidate_utility_json": _json_dumps((teacher_label or {}).get("candidate_utility") or []),
        "teacher_candidate_reflection_json": _json_dumps((teacher_label or {}).get("candidate_reflection") or []),
        "teacher_guardrail_summary_json": _json_dumps((teacher_label or {}).get("guardrail_summary") or metadata.get("teacher_guardrail_summary") or {}),
        "teacher_semantic_tags_json": _json_dumps((teacher_label or {}).get("semantic_tags") or metadata.get("active_semantic_tags") or []),
        "chosen_action": pair.get("chosen_action", ""),
        "rejected_action": pair.get("rejected_action", ""),
        "delta_u_rel": metadata.get("delta_u_rel", ""),
        "chosen_text": pair.get("chosen", ""),
        "rejected_text": pair.get("rejected", ""),
        "chosen_candidate_json": _json_dumps(chosen_candidate or {"evidence": chosen_ev}),
        "rejected_candidate_json": _json_dumps(rejected_candidate or {"evidence": rejected_ev}),
        "pair_metadata_json": _json_dumps(metadata),
    }
    return {"data": data}


def _question_from_prompt(prompt: str) -> str:
    marker = "USER: Question: "
    if marker not in prompt:
        return ""
    tail = prompt.split(marker, 1)[1]
    return tail.split("\nAvailable actions:", 1)[0].strip()


def export_label_studio_tasks(
    *,
    input_dir: Path,
    output_dir: Path,
    pair_file: str,
    rollout_file: str,
    teacher_label_file: str,
    basename: str,
) -> dict[str, Any]:
    pairs = read_jsonl(input_dir / pair_file)
    rollouts = _index_by_state(read_jsonl(input_dir / rollout_file))
    teacher_labels = _index_by_state(read_jsonl(input_dir / teacher_label_file))

    tasks = [
        _task_from_pair(
            pair,
            index=i,
            rollout=rollouts.get(str(pair.get("state_id"))),
            teacher_label=teacher_labels.get(str(pair.get("state_id"))),
        )
        for i, pair in enumerate(pairs, start=1)
    ]

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / f"{basename}.json"
    jsonl_path = output_dir / f"{basename}.jsonl"
    config_path = output_dir / f"{basename}_config.xml"
    summary_path = output_dir / f"{basename}_summary.json"

    json_path.write_text(_json_dumps(tasks) + "\n", encoding="utf-8")
    write_jsonl(jsonl_path, tasks)
    config_path.write_text(_label_config_xml(), encoding="utf-8")

    by_dataset = Counter(str((task.get("data") or {}).get("dataset", "")) for task in tasks)
    chosen = Counter(str((task.get("data") or {}).get("chosen_action", "")) for task in tasks)
    rejected = Counter(str((task.get("data") or {}).get("rejected_action", "")) for task in tasks)
    focus = Counter()
    for task in tasks:
        value = str((task.get("data") or {}).get("review_focus", "general"))
        for part in [p.strip() for p in value.split(",") if p.strip()]:
            focus[part] += 1

    summary = {
        "num_tasks": len(tasks),
        "input_dir": str(input_dir),
        "pair_file": pair_file,
        "rollout_file": rollout_file,
        "teacher_label_file": teacher_label_file,
        "label_studio_json": str(json_path),
        "label_studio_jsonl": str(jsonl_path),
        "label_config": str(config_path),
        "summary": str(summary_path),
        "by_dataset": dict(sorted(by_dataset.items())),
        "chosen_action": dict(sorted(chosen.items())),
        "rejected_action": dict(sorted(rejected.items())),
        "review_focus": dict(sorted(focus.items())),
        "annotation_fields": {
            "human_pair_accept": ["true", "false", "unclear"],
            "teacher_error": ["false", "true", "unclear"],
            "error_type": ERROR_TYPES,
            "notes": "free text",
        },
    }
    summary_path.write_text(_json_dumps(summary) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Export DPO pairs to Label Studio pair-review tasks.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--pair-file", default="train_step_dpo_pairs.jsonl")
    parser.add_argument("--rollout-file", default="all_rollouts.jsonl")
    parser.add_argument("--teacher-label-file", default="teacher_labels_self_evidence.jsonl")
    parser.add_argument("--basename", default="label_studio_v025_pair_review")
    args = parser.parse_args()

    output_dir = args.output_dir or (args.input_dir / "human_review")
    summary = export_label_studio_tasks(
        input_dir=args.input_dir,
        output_dir=output_dir,
        pair_file=args.pair_file,
        rollout_file=args.rollout_file,
        teacher_label_file=args.teacher_label_file,
        basename=args.basename,
    )
    print(_json_dumps(summary))


if __name__ == "__main__":
    main()
