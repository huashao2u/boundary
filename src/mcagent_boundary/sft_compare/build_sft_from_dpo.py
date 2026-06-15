from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mcagent_boundary.io import read_jsonl, write_json, write_jsonl


DEFAULT_PAIR_DIR = REPO_ROOT / "artifacts/pairs_v026_20260527T_boundary_from_rollout171000"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts/sft_compare/pairs_v026_20260527T_boundary_from_rollout171000"


def _sample_weight(pair: dict[str, Any]) -> float:
    weight = pair.get("sample_weight")
    if weight is None:
        weight = (pair.get("metadata") or {}).get("sample_weight", 1.0)
    try:
        return float(weight)
    except (TypeError, ValueError):
        return 1.0


def _convert_pair(pair: dict[str, Any]) -> dict[str, Any]:
    prompt_messages = list(pair["prompt_messages"])
    completion_messages = list(pair["chosen_messages"])
    return {
        "sft_id": pair.get("pair_id"),
        "pair_id": pair.get("pair_id"),
        "state_id": pair.get("state_id"),
        "example_id": pair.get("example_id"),
        "dataset": pair.get("dataset"),
        "boundary_type": pair.get("boundary_type"),
        "pool": pair.get("pool"),
        "pair_kind": pair.get("pair_kind", "strong"),
        "sample_weight": _sample_weight(pair),
        "chosen_action": pair.get("chosen_action"),
        "rejected_action": pair.get("rejected_action"),
        "prompt_messages": prompt_messages,
        "completion_messages": completion_messages,
        "messages": prompt_messages + completion_messages,
        "metadata": {
            **dict(pair.get("metadata") or {}),
            "sft_source": "dpo_chosen_only",
            "dpo_pair_id": pair.get("pair_id"),
        },
    }


def build_sft_records(
    pairs: list[dict[str, Any]],
    *,
    min_sample_weight: float = 0.0,
    pair_kind: str = "all",
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for pair in pairs:
        if pair_kind != "all" and str(pair.get("pair_kind", "")) != pair_kind:
            continue
        if _sample_weight(pair) < min_sample_weight:
            continue
        pair_id = str(pair.get("pair_id") or "")
        if pair_id and pair_id in seen:
            continue
        if pair_id:
            seen.add(pair_id)
        if not pair.get("prompt_messages") or not pair.get("chosen_messages"):
            continue
        records.append(_convert_pair(pair))
    return records


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "num_records": len(records),
        "by_dataset": dict(Counter(str(row.get("dataset")) for row in records).most_common()),
        "by_pair_kind": dict(Counter(str(row.get("pair_kind")) for row in records).most_common()),
        "by_chosen_action": dict(Counter(str(row.get("chosen_action")) for row in records).most_common()),
        "by_action_pair": dict(
            Counter(f"{row.get('chosen_action')}>{row.get('rejected_action')}" for row in records).most_common()
        ),
        "sample_weight_sum": round(sum(_sample_weight(row) for row in records), 6),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert boundary DPO pairs into chosen-only SFT records.")
    parser.add_argument("--pair-dir", type=str, default=str(DEFAULT_PAIR_DIR))
    parser.add_argument("--train-pair-file", type=str, default=None)
    parser.add_argument("--eval-pair-file", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument(
        "--min-sample-weight",
        type=float,
        default=0.0,
        help="Filter weak augmented pairs. Use 1.0 for strong-only SFT.",
    )
    parser.add_argument(
        "--pair-kind",
        type=str,
        default="all",
        choices=["all", "strong", "near_tie", "best_mid", "mintqa_answer_search"],
    )
    args = parser.parse_args()

    pair_dir = Path(args.pair_dir)
    train_pair_file = Path(args.train_pair_file) if args.train_pair_file else pair_dir / "train_step_dpo_pairs.jsonl"
    eval_pair_file = Path(args.eval_pair_file) if args.eval_pair_file else pair_dir / "eval_step_dpo_pairs.jsonl"
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_records = build_sft_records(
        read_jsonl(train_pair_file),
        min_sample_weight=float(args.min_sample_weight),
        pair_kind=str(args.pair_kind),
    )
    eval_records = build_sft_records(
        read_jsonl(eval_pair_file),
        min_sample_weight=float(args.min_sample_weight),
        pair_kind=str(args.pair_kind),
    )

    train_output = output_dir / "train_sft_chosen.jsonl"
    eval_output = output_dir / "eval_sft_chosen.jsonl"
    summary = {
        "source_train_pair_file": str(train_pair_file),
        "source_eval_pair_file": str(eval_pair_file),
        "train_output": str(train_output),
        "eval_output": str(eval_output),
        "min_sample_weight": float(args.min_sample_weight),
        "pair_kind": str(args.pair_kind),
        "train": summarize(train_records),
        "eval": summarize(eval_records),
    }
    write_jsonl(train_output, train_records)
    write_jsonl(eval_output, eval_records)
    write_json(output_dir / "sft_conversion_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

