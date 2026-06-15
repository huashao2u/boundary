from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from html import escape
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mcagent_boundary.io import read_jsonl, write_jsonl


TEACHER_ERROR_TYPES = [
    "none",
    "wrong_answer_payload_high_score",
    "correct_calculate_low_score",
    "teacher_ignored_payload",
    "teacher_rationale_score_conflict",
    "pairwise_disagreement",
    "unclear",
]

MODEL_OUTPUT_ERROR_TYPES = [
    "none",
    "answer_contains_clarify",
    "answer_contains_refusal",
    "answer_mentions_tool_unavailable",
    "answer_has_unit",
    "payload_action_mismatch",
    "answer_is_expression",
    "payload_rationale_conflict",
    "over_refuse",
    "bad_search_query",
    "bad_clarify_question",
    "unsafe_answer_high_score",
    "answer_shell_refusal",
    "other_model_output_error",
    "unclear",
]

CHOICE_LABELS = {
    "accept": "Accept：接受",
    "reject": "Reject：拒绝",
    "unclear": "Unclear：无法判断",
    "no_teacher_error": "No Error：基本合理",
    "teacher_error": "Error：明显错误",
    "none": "None：无此类错误 / 不需要归因",
    "answer_contains_clarify": "答案包含澄清：ANSWER 输出像是在提问或澄清，未直接回答",
    "answer_contains_refusal": "答案包含拒答：ANSWER 输出拒绝或逃避回答",
    "answer_mentions_tool_unavailable": "工具不可用借口：候选声称无法使用工具或能力受限",
    "answer_has_unit": "答案单位/格式问题：数值或单位格式导致答案质量下降",
    "payload_action_mismatch": "Action-payload 不匹配：动作类型和实际输出内容不一致",
    "wrong_answer_payload_high_score": "错误答案被高分：候选的 answer / action_input 明显错误，但 teacher 给了较高分",
    "correct_calculate_low_score": "正确计算/答案被低分：候选基本正确，但 teacher 给了过低分",
    "answer_is_expression": "把表达式当作最终答案：只给公式/表达式/中间结果，没有真正回答",
    "payload_rationale_conflict": "Payload-rationale 冲突：实际输出和 rationale / reflection 不一致",
    "teacher_ignored_payload": "Teacher 忽略实际输出：teacher 没有正确检查 action_input / answer 内容",
    "teacher_rationale_score_conflict": "Teacher 分数-理由冲突：utility 分数、推荐动作和 rationale 互相矛盾",
    "over_refuse": "过度拒答：候选不必要地拒绝或逃避当前任务",
    "bad_search_query": "低质量搜索请求：搜索 query 不具体、不相关或遗漏关键约束",
    "bad_clarify_question": "低质量澄清问题：澄清问题没有命中关键缺失信息",
    "unsafe_answer_high_score": "不安全答案：候选有安全、隐私、合规等风险",
    "answer_shell_refusal": "模板化拒答：候选只有空泛安全壳，没有处理当前任务",
    "pairwise_disagreement": "Pair 方向/相对偏好不成立：chosen 并不明显优于 rejected，甚至方向反了",
    "other_model_output_error": "其他模型输出错误：以上类型都不准确，但候选输出本身有明显问题",
}

HUMAN_PAIR_ACCEPT_VALUES = ["accept", "reject", "unclear"]
TEACHER_ERROR_VALUES = ["no_teacher_error", "teacher_error", "unclear"]


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=False)


def _display_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        return _json_dumps(value)
    return str(value)


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


def _choice_xml(value: str, indent: str = "          ") -> str:
    label = CHOICE_LABELS.get(value, value)
    return f'{indent}<Choice value="{escape(label)}" alias="{escape(value)}"/>'


def _label_config_xml() -> str:
    teacher_error_choices = "\n".join(_choice_xml(value) for value in TEACHER_ERROR_TYPES)
    model_output_error_choices = "\n".join(_choice_xml(value) for value in MODEL_OUTPUT_ERROR_TYPES)
    return f"""<View>
  <Style>
    .page {{
      display: flex;
      gap: 16px;
      align-items: flex-start;
      width: 100%;
    }}

    .left-panel {{
      width: 64%;
      max-height: calc(100vh - 120px);
      overflow-y: auto;
      padding-right: 12px;
      border-right: 1px solid #ddd;
    }}

    .right-panel {{
      width: 36%;
      position: sticky;
      top: 12px;
      align-self: flex-start;
      padding: 12px;
      border: 1px solid #ddd;
      border-radius: 6px;
      background: #fff;
    }}

    .block {{
      margin: 12px 0;
      padding: 10px;
      border: 1px solid #ddd;
      border-radius: 4px;
    }}

    .meta {{
      color: #555;
      font-size: 13px;
    }}

    .warn {{
      color: #8a4b00;
      font-weight: 600;
    }}

    pre {{
      white-space: pre-wrap;
    }}

    .instruction {{
      color: #555;
      font-size: 13px;
      line-height: 1.45;
      margin-bottom: 8px;
    }}

    .review-section {{
      margin-bottom: 16px;
      padding-bottom: 12px;
      border-bottom: 1px solid #eee;
    }}
  </Style>

  <Header value="v0.2.6 Pair Review"/>

  <View className="page">
    <View className="left-panel">
      <View className="block">
        <Text name="meta" value="Pair: $pair_id | Split: $pair_split | Dataset: $dataset | Boundary: $boundary_type | Pool: $pool"/>
        <Text name="ids" value="Example: $example_id | State: $state_id"/>
        <Text name="focus" value="Review focus: $review_focus"/>
      </View>

      <View className="block">
        <Header value="Question"/>
        <Text name="question" value="$question"/>
        <Header value="Gold / Reference"/>
        <Text name="gold_answer" value="$gold_answer"/>
        <Text name="allowed_actions" value="Allowed actions: $allowed_actions_json | effective_top_k: $effective_top_k"/>
        <Header value="Metadata"/>
        <Text name="metadata_json" value="$metadata_json"/>
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
    </View>

    <View className="right-panel">
      <Header value="Human Review"/>

      <View className="review-section">
        <Header value="1. 是否接受这个 chosen/rejected pair？"/>
        <Text name="human_pair_accept_instruction"
              className="instruction"
              value="判断标准：如果 chosen 明显优于 rejected，且二者都符合当前任务格式和语义边界，则选择 Accept；如果 rejected 不差于 chosen、chosen 有明显问题、pair 方向反了，或样本本身不适合作为偏好对，则选择 Reject；如果信息不足或难以判断，则选择 Unclear。"/>
        <Choices name="human_pair_accept"
                 toName="pair_actions"
                 choice="single"
                 showInLine="true"
                 required="true"
                 requiredMessage="请选择是否接受当前 chosen/rejected pair。">
          <Choice value="Accept：接受" alias="accept"/>
          <Choice value="Reject：拒绝" alias="reject"/>
          <Choice value="Unclear：无法判断" alias="unclear"/>
        </Choices>
      </View>

      <View className="review-section">
        <Header value="2. teacher 判断是否有明显错误？"/>
        <Text name="teacher_error_instruction"
              className="instruction"
              value="判断标准：如果 teacher 的推荐动作、utility 分数、错误分析或理由与样本事实明显不符，则选择 Teacher Error；如果 teacher 判断基本合理，即使措辞不完美，也选择 No Teacher Error；如果无法确认 teacher 是否错误，则选择 Unclear。"/>
        <Choices name="teacher_error"
                 toName="teacher_candidate_utility_json"
                 choice="single"
                 showInLine="true"
                 required="true"
                 requiredMessage="请选择 teacher 判断是否存在明显错误。">
          <Choice value="No Error：基本合理" alias="no_teacher_error"/>
          <Choice value="Error：明显错误" alias="teacher_error"/>
          <Choice value="Unclear：无法确认" alias="unclear"/>
        </Choices>
      </View>

      <View className="review-section">
        <Header value="3. teacher 标注错误类型"/>
        <Text name="teacher_error_type_instruction"
              className="instruction"
              value="只归因 teacher 的判断或标注问题：例如分数给错、忽略实际 payload、理由和分数矛盾、或 pair 相对偏好不成立。如果 teacher 基本合理，选择 none。"/>
        <Choices name="teacher_error_type"
                 toName="teacher_candidate_utility_json"
                 choice="single"
                 layout="select"
                 required="true"
                 requiredMessage="请选择 teacher 错误类型；如果没有明显错误，请选择 none。">
{teacher_error_choices}
        </Choices>
      </View>

      <View className="review-section">
        <Header value="4. 模型输出错误类型"/>
        <Text name="model_output_error_type_instruction"
              className="instruction"
              value="只归因 student/model 候选输出本身的问题：例如 ANSWER 像澄清/拒答、payload 和 action 不匹配、表达式冒充答案、搜索或澄清请求质量低、安全风险等。如果候选输出本身没有明显问题，选择 none。"/>
        <Choices name="model_output_error_type"
                 toName="pair_actions"
                 choice="single"
                 layout="select"
                 required="true"
                 requiredMessage="请选择模型输出错误类型；如果没有明显错误，请选择 none。">
{model_output_error_choices}
        </Choices>
      </View>

      <TextArea name="notes"
                toName="pair_actions"
                rows="4"
                placeholder="Optional notes"/>
    </View>
  </View>
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
    display_metadata = (rollout or {}).get("metadata") or (teacher_label or {}).get("metadata") or {}
    gold_answer = (rollout or {}).get("gold_answer")
    if gold_answer is None and isinstance(display_metadata, dict):
        clarify_question = display_metadata.get("gold_clarify_question")
        clarify_reply = display_metadata.get("gold_clarify_reply")
        if clarify_question or clarify_reply:
            gold_answer = {
                "gold_clarify_question": clarify_question,
                "gold_clarify_reply": clarify_reply,
            }

    data = {
        "pair_id": pair_id,
        "review_id": pair_id,
        "source_pair_id": pair.get("pair_id", ""),
        "pair_split": pair.get("_review_split", ""),
        "state_id": pair.get("state_id", ""),
        "example_id": pair.get("example_id", ""),
        "dataset": pair.get("dataset", ""),
        "boundary_type": pair.get("boundary_type", ""),
        "pool": pair.get("pool", ""),
        "review_focus": ", ".join(focus) if focus else "general",
        "question": (rollout or {}).get("question") or _question_from_prompt(pair.get("prompt", "")),
        "gold_answer": _display_text(gold_answer),
        "metadata": display_metadata,
        "metadata_json": _json_dumps(display_metadata),
        "allowed_actions": (rollout or {}).get("allowed_actions") or [],
        "allowed_actions_json": _json_dumps((rollout or {}).get("allowed_actions") or []),
        "effective_top_k": (rollout or {}).get("effective_top_k", ""),
        "student_reasoning_attempt": (rollout or {}).get("reasoning_attempt")
        or (rollout or {}).get("reason_prefix", ""),
        "uncertainty_summary": (rollout or {}).get("uncertainty_summary", ""),
        "student_candidates_json": _json_dumps((rollout or {}).get("candidates") or []),
        "teacher_recommended_action": (teacher_label or {}).get("recommended_action", ""),
        "teacher_rationale": (teacher_label or {}).get("rationale", ""),
        "teacher_candidate_evidence_json": _json_dumps((teacher_label or {}).get("candidate_evidence") or []),
        "candidate_evidence": (teacher_label or {}).get("candidate_evidence") or [],
        "teacher_candidate_utility_json": _json_dumps((teacher_label or {}).get("candidate_utility") or []),
        "teacher_utility": (teacher_label or {}).get("candidate_utility") or [],
        "teacher_candidate_reflection_json": _json_dumps((teacher_label or {}).get("candidate_reflection") or []),
        "teacher_guardrail_summary_json": _json_dumps((teacher_label or {}).get("guardrail_summary") or metadata.get("teacher_guardrail_summary") or {}),
        "guardrail_summary": (teacher_label or {}).get("guardrail_summary") or metadata.get("teacher_guardrail_summary") or {},
        "teacher_semantic_tags_json": _json_dumps((teacher_label or {}).get("semantic_tags") or metadata.get("active_semantic_tags") or []),
        "chosen_action": pair.get("chosen_action", ""),
        "rejected_action": pair.get("rejected_action", ""),
        "delta_u_rel": metadata.get("delta_u_rel", ""),
        "chosen_text": pair.get("chosen", ""),
        "rejected_text": pair.get("rejected", ""),
        "chosen_candidate_json": _json_dumps(chosen_candidate or {"evidence": chosen_ev}),
        "chosen": chosen_candidate or {"evidence": chosen_ev},
        "rejected_candidate_json": _json_dumps(rejected_candidate or {"evidence": rejected_ev}),
        "rejected": rejected_candidate or {"evidence": rejected_ev},
        "pair_metadata_json": _json_dumps(metadata),
    }
    return {"data": data}


def _question_from_prompt(prompt: str) -> str:
    marker = "USER: Question: "
    if marker not in prompt:
        return ""
    tail = prompt.split(marker, 1)[1]
    return tail.split("\nAvailable actions:", 1)[0].strip()


def _load_pairs_for_review(input_dir: Path, pair_files: list[tuple[str, str]]) -> list[dict[str, Any]]:
    pairs: list[dict[str, Any]] = []
    for split, pair_file in pair_files:
        for row in read_jsonl(input_dir / pair_file):
            item = dict(row)
            item["_review_split"] = split
            item["_review_pair_file"] = pair_file
            pairs.append(item)
    return pairs


def _stratum_value(pair: dict[str, Any], field: str) -> str:
    metadata = pair.get("metadata") or {}
    if field == "split":
        return str(pair.get("_review_split") or pair.get("split") or metadata.get("split") or "")
    return str(pair.get(field) or metadata.get(field) or "")


def _stratified_sample_pairs(
    pairs: list[dict[str, Any]],
    *,
    sample_size: int | None,
    sample_seed: int,
    stratify_by: list[str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    population = len(pairs)
    if sample_size is None or sample_size <= 0 or sample_size >= population:
        return list(pairs), {
            "enabled": False,
            "population": population,
            "sample_size": population,
            "sample_seed": sample_seed,
            "stratify_by": stratify_by,
            "reason": "sample_size omitted, non-positive, or >= population",
        }

    rng = random.Random(sample_seed)
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for pair in pairs:
        key = tuple(_stratum_value(pair, field) for field in stratify_by)
        groups[key].append(pair)

    ideals = {key: (len(rows) / population) * sample_size for key, rows in groups.items()}
    min_per_stratum = 1 if len(groups) <= sample_size else 0
    allocations: dict[tuple[str, ...], int] = {
        key: min(len(groups[key]), max(min_per_stratum, int(ideal)))
        for key, ideal in ideals.items()
    }

    while sum(allocations.values()) > sample_size:
        removable = [key for key, count in allocations.items() if count > min_per_stratum]
        if not removable:
            break
        key = min(
            removable,
            key=lambda item: (
                ideals[item] - allocations[item],
                allocations[item],
                tuple(reversed(item)),
            ),
        )
        allocations[key] -= 1

    remainders = sorted(
        ((ideals[key] - allocations[key], key) for key in groups),
        key=lambda item: (item[0], item[1]),
        reverse=True,
    )
    while sum(allocations.values()) < sample_size:
        progressed = False
        for _, key in remainders:
            if sum(allocations.values()) >= sample_size:
                break
            if allocations[key] >= len(groups[key]):
                continue
            allocations[key] += 1
            progressed = True
        if not progressed:
            break

    sampled: list[dict[str, Any]] = []
    stratum_summary: dict[str, dict[str, int]] = {}
    for key in sorted(groups):
        rows = list(groups[key])
        rows.sort(key=lambda row: str(row.get("pair_id") or row.get("state_id") or row.get("example_id") or ""))
        count = allocations.get(key, 0)
        chosen = rng.sample(rows, count) if count < len(rows) else rows
        chosen.sort(key=lambda row: str(row.get("pair_id") or row.get("state_id") or row.get("example_id") or ""))
        sampled.extend(chosen)
        stratum_summary["|".join(key)] = {"population": len(rows), "sampled": len(chosen)}

    sampled.sort(
        key=lambda row: (
            str(row.get("_review_split") or ""),
            str(row.get("dataset") or ""),
            str(row.get("pair_id") or row.get("state_id") or row.get("example_id") or ""),
        )
    )
    return sampled, {
        "enabled": True,
        "population": population,
        "sample_size": len(sampled),
        "requested_sample_size": sample_size,
        "sample_seed": sample_seed,
        "stratify_by": stratify_by,
        "num_strata": len(groups),
        "strata": stratum_summary,
        "sampled_pair_ids": [str(pair.get("pair_id") or "") for pair in sampled],
    }


def export_label_studio_tasks(
    *,
    input_dir: Path,
    output_dir: Path,
    pair_files: list[tuple[str, str]],
    rollout_file: str,
    teacher_label_file: str,
    basename: str,
    sample_size: int | None = None,
    sample_seed: int = 20260615,
    stratify_by: list[str] | None = None,
) -> dict[str, Any]:
    stratify_by = stratify_by or ["split", "dataset", "pair_kind", "chosen_action", "rejected_action"]
    pairs_all = _load_pairs_for_review(input_dir, pair_files)
    pairs, sampling = _stratified_sample_pairs(
        pairs_all,
        sample_size=sample_size,
        sample_seed=sample_seed,
        stratify_by=stratify_by,
    )
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
    import_json_path = output_dir / f"{basename}.import.json"
    jsonl_path = output_dir / f"{basename}.jsonl"
    config_path = output_dir / f"{basename}_config.xml"
    summary_path = output_dir / f"{basename}_summary.json"
    sampling_path = output_dir / f"{basename}_sampling_manifest.json"

    json_path.write_text(_json_dumps(tasks) + "\n", encoding="utf-8")
    import_json_path.write_text(
        json.dumps(tasks, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    write_jsonl(jsonl_path, tasks)
    config_path.write_text(_label_config_xml(), encoding="utf-8")
    sampling_path.write_text(_json_dumps(sampling) + "\n", encoding="utf-8")

    by_split = Counter(str((task.get("data") or {}).get("pair_split", "")) for task in tasks)
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
        "num_population_pairs": len(pairs_all),
        "input_dir": str(input_dir),
        "pair_files": [{"split": split, "path": pair_file} for split, pair_file in pair_files],
        "rollout_file": rollout_file,
        "teacher_label_file": teacher_label_file,
        "label_studio_json": str(json_path),
        "label_studio_import_json": str(import_json_path),
        "label_studio_jsonl": str(jsonl_path),
        "label_config": str(config_path),
        "summary": str(summary_path),
        "sampling_manifest": str(sampling_path),
        "sampling": {key: value for key, value in sampling.items() if key not in {"strata", "sampled_pair_ids"}},
        "by_split": dict(sorted(by_split.items())),
        "by_dataset": dict(sorted(by_dataset.items())),
        "chosen_action": dict(sorted(chosen.items())),
        "rejected_action": dict(sorted(rejected.items())),
        "review_focus": dict(sorted(focus.items())),
        "annotation_fields": {
            "human_pair_accept": HUMAN_PAIR_ACCEPT_VALUES,
            "teacher_error": TEACHER_ERROR_VALUES,
            "teacher_error_type": TEACHER_ERROR_TYPES,
            "model_output_error_type": MODEL_OUTPUT_ERROR_TYPES,
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
    parser.add_argument("--eval-pair-file", default=None)
    parser.add_argument("--rollout-file", default="all_rollouts.jsonl")
    parser.add_argument("--teacher-label-file", default="teacher_labels_self_evidence.jsonl")
    parser.add_argument("--basename", default="label_studio_v025_pair_review")
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--sample-seed", type=int, default=20260615)
    parser.add_argument("--stratify-by", default="split,dataset,pair_kind,chosen_action,rejected_action")
    args = parser.parse_args()

    output_dir = args.output_dir or (args.input_dir / "human_review")
    pair_files = [("train", args.pair_file)]
    if args.eval_pair_file:
        pair_files.append(("eval", args.eval_pair_file))
    stratify_by = [part.strip() for part in args.stratify_by.split(",") if part.strip()]
    summary = export_label_studio_tasks(
        input_dir=args.input_dir,
        output_dir=output_dir,
        pair_files=pair_files,
        rollout_file=args.rollout_file,
        teacher_label_file=args.teacher_label_file,
        basename=args.basename,
        sample_size=args.sample_size,
        sample_seed=args.sample_seed,
        stratify_by=stratify_by,
    )
    print(_json_dumps(summary))


if __name__ == "__main__":
    main()
