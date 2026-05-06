from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mcagent_boundary.config import load_boundary_config, resolve_repo_path
from mcagent_boundary.io import read_jsonl, write_jsonl, write_jsonl_by_dataset
from mcagent_boundary.training.make_dpo_pairs import build_step_dpo_pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Step-DPO pairs from rollout + teacher labels.")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output directory.")
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Override input directory for boundary/anchor/teacher JSONL files.",
    )
    parser.add_argument(
        "--require-teacher-labels",
        action="store_true",
        help="Drop records that do not have a teacher label.",
    )
    parser.add_argument(
        "--require-poe-teacher",
        action="store_true",
        help=(
            "Deprecated alias for --require-configured-teacher-source. "
            "Use --require-teacher-source poe_teacher for legacy Poe-only labels."
        ),
    )
    parser.add_argument(
        "--require-configured-teacher-source",
        action="store_true",
        help="Drop records whose teacher label source is not teacher.source_label.",
    )
    parser.add_argument(
        "--require-teacher-source",
        action="append",
        default=[],
        metavar="SOURCE",
        help="Allowed teacher label source. May be passed multiple times.",
    )
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bars.")
    args = parser.parse_args()

    config = load_boundary_config()

    def _in(name: str) -> Path:
        base = resolve_repo_path(config["paths"][name], config)
        if args.input_dir:
            return Path(args.input_dir) / base.name
        return base

    def _out(name: str) -> Path:
        base = resolve_repo_path(config["paths"][name], config)
        if args.output_dir:
            return Path(args.output_dir) / base.name
        return base

    boundary_records = read_jsonl(_in("boundary_candidates_output"))
    clear_answer = read_jsonl(_in("clear_answer_output"))
    clear_external = read_jsonl(_in("clear_external_output"))
    teacher_labels = read_jsonl(_in("teacher_label_output"))
    selected_records = boundary_records + clear_answer + clear_external
    configured_teacher_source = str(config.get("teacher", {}).get("source_label", "llm_teacher"))
    required_teacher_sources = set(args.require_teacher_source or [])
    if args.require_configured_teacher_source or args.require_poe_teacher:
        required_teacher_sources.add(configured_teacher_source)
    train_pairs, eval_pairs, diagnostics = build_step_dpo_pairs(
        selected_records,
        teacher_labels,
        config,
        show_progress=not args.no_progress,
        require_teacher_label=args.require_teacher_labels or bool(required_teacher_sources),
        required_teacher_sources=required_teacher_sources,
    )
    train_path = _out("train_pair_output")
    eval_path = _out("eval_pair_output")
    write_jsonl(train_path, train_pairs)
    write_jsonl(eval_path, eval_pairs)
    train_by_dataset = write_jsonl_by_dataset(train_path, train_pairs)
    eval_by_dataset = write_jsonl_by_dataset(eval_path, eval_pairs)
    diagnostics_path = train_path.with_name("pair_diagnostics.jsonl")
    write_jsonl(diagnostics_path, diagnostics)
    diagnostics_by_dataset = write_jsonl_by_dataset(diagnostics_path, diagnostics)
    diagnostic_reasons: dict[str, int] = {}
    for item in diagnostics:
        reason = str(item.get("reason", "unknown"))
        diagnostic_reasons[reason] = diagnostic_reasons.get(reason, 0) + 1
    print(json.dumps({
        "train_pairs": len(train_pairs),
        "eval_pairs": len(eval_pairs),
        "diagnostics": len(diagnostics),
        "diagnostic_reasons": diagnostic_reasons,
        "train_output": str(train_path),
        "eval_output": str(eval_path),
        "required_teacher_sources": sorted(required_teacher_sources),
        "train_by_dataset": train_by_dataset,
        "eval_by_dataset": eval_by_dataset,
        "diagnostics_by_dataset": diagnostics_by_dataset,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
