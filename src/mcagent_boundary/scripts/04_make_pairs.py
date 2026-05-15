from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mcagent_boundary.config import load_boundary_config, resolve_repo_path
from mcagent_boundary.io import read_jsonl, to_jsonable, write_jsonl, write_jsonl_by_dataset
from mcagent_boundary.training.make_dpo_pairs import build_step_dpo_pairs, pair_augmentation_enabled


def _iter_jsonl(path: Path) -> Iterable[dict]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _append_jsonl(path: Path, records: Iterable[dict]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(to_jsonable(record), ensure_ascii=False) + "\n")
            count += 1
    return count


def _count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def _write_by_dataset_streaming(path: Path, *, dataset_key: str = "dataset") -> dict[str, str]:
    outputs: dict[str, str] = {}
    handles: dict[str, object] = {}
    try:
        for row in _iter_jsonl(path):
            dataset = str(row.get(dataset_key) or "unknown")
            dataset_path = path.parent / "by_dataset" / dataset / path.name
            if dataset not in handles:
                dataset_path.parent.mkdir(parents=True, exist_ok=True)
                handles[dataset] = dataset_path.open("w", encoding="utf-8")
                outputs[dataset] = str(dataset_path)
            handles[dataset].write(json.dumps(to_jsonable(row), ensure_ascii=False) + "\n")
    finally:
        for handle in handles.values():
            handle.close()
    return outputs


def _load_teacher_by_state(path: Path) -> dict[str, dict]:
    labels: dict[str, dict] = {}
    for label in _iter_jsonl(path):
        state_id = label.get("state_id")
        if state_id:
            labels[str(state_id)] = label
    return labels


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
    parser.add_argument(
        "--teacher-label-file",
        type=str,
        default=None,
        help=(
            "Override teacher label JSONL. Relative paths are resolved under --input-dir "
            "when provided, otherwise under the repo root."
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help="Stream selected records in batches and append outputs to reduce peak memory.",
    )
    parser.add_argument(
        "--prompt-mode",
        choices=["decision_window", "end_to_end"],
        default=None,
        help="DPO prompt/completion format. Defaults to pair_construction.prompt_mode.",
    )
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bars.")
    args = parser.parse_args()

    config = load_boundary_config()
    if args.prompt_mode is not None:
        config.setdefault("pair_construction", {})["prompt_mode"] = args.prompt_mode

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

    if args.teacher_label_file:
        teacher_label_path = Path(args.teacher_label_file)
        if not teacher_label_path.is_absolute():
            teacher_label_path = Path(args.input_dir or ".") / teacher_label_path
    else:
        teacher_label_path = _in("teacher_label_output")
    configured_teacher_source = str(config.get("teacher", {}).get("source_label", "llm_teacher"))
    required_teacher_sources = set(args.require_teacher_source or [])
    if args.require_configured_teacher_source or args.require_poe_teacher:
        required_teacher_sources.add(configured_teacher_source)

    if args.batch_size and args.batch_size > 0 and not pair_augmentation_enabled(config):
        teacher_by_state = _load_teacher_by_state(teacher_label_path)
        input_paths = [
            _in("boundary_candidates_output"),
            _in("clear_answer_output"),
            _in("clear_external_output"),
        ]
        train_path = _out("train_pair_output")
        eval_path = _out("eval_pair_output")
        diagnostics_path = train_path.with_name("pair_diagnostics.jsonl")
        for path in [train_path, eval_path, diagnostics_path]:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("", encoding="utf-8")

        batch_size = max(1, int(args.batch_size))
        batch: list[dict] = []
        total_seen = 0
        train_count = 0
        eval_count = 0
        diagnostics_count = 0
        diagnostic_reasons: dict[str, int] = {}

        def _run_batch(items: list[dict]) -> None:
            nonlocal train_count, eval_count, diagnostics_count
            if not items:
                return
            labels = [
                teacher_by_state[str(record.get("state_id"))]
                for record in items
                if str(record.get("state_id")) in teacher_by_state
            ]
            train_pairs, eval_pairs, diagnostics = build_step_dpo_pairs(
                items,
                labels,
                config,
                show_progress=False,
                require_teacher_label=args.require_teacher_labels or bool(required_teacher_sources),
                required_teacher_sources=required_teacher_sources,
            )
            train_count += _append_jsonl(train_path, train_pairs)
            eval_count += _append_jsonl(eval_path, eval_pairs)
            diagnostics_count += _append_jsonl(diagnostics_path, diagnostics)
            for item in diagnostics:
                reason = str(item.get("reason", "unknown"))
                diagnostic_reasons[reason] = diagnostic_reasons.get(reason, 0) + 1

        for input_path in input_paths:
            for record in _iter_jsonl(input_path):
                total_seen += 1
                batch.append(record)
                if len(batch) >= batch_size:
                    _run_batch(batch)
                    batch.clear()
        _run_batch(batch)
        train_by_dataset = _write_by_dataset_streaming(train_path)
        eval_by_dataset = _write_by_dataset_streaming(eval_path)
        diagnostics_by_dataset = _write_by_dataset_streaming(diagnostics_path)
        print(json.dumps({
            "train_pairs": train_count,
            "eval_pairs": eval_count,
            "diagnostics": diagnostics_count,
            "diagnostic_reasons": diagnostic_reasons,
            "train_output": str(train_path),
            "eval_output": str(eval_path),
            "teacher_label_file": str(teacher_label_path),
            "required_teacher_sources": sorted(required_teacher_sources),
            "train_by_dataset": train_by_dataset,
            "eval_by_dataset": eval_by_dataset,
            "diagnostics_by_dataset": diagnostics_by_dataset,
            "input_records_seen": total_seen,
            "streaming_batch_size": batch_size,
            "output_line_counts": {
                "train_pairs": _count_jsonl(train_path),
                "eval_pairs": _count_jsonl(eval_path),
                "diagnostics": _count_jsonl(diagnostics_path),
            },
        }, ensure_ascii=False, indent=2))
        return

    if args.batch_size and args.batch_size > 0 and pair_augmentation_enabled(config):
        print(json.dumps({
            "streaming_batch_size_requested": int(args.batch_size),
            "streaming_batch_size_effective": 0,
            "reason": (
                "pair_construction.augmentation is enabled; global near-tie/best-mid quotas "
                "require a full-corpus pass."
            ),
        }, ensure_ascii=False, indent=2))

    boundary_records = read_jsonl(_in("boundary_candidates_output"))
    clear_answer = read_jsonl(_in("clear_answer_output"))
    clear_external = read_jsonl(_in("clear_external_output"))
    teacher_labels = read_jsonl(teacher_label_path)
    selected_records = boundary_records + clear_answer + clear_external
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
        "teacher_label_file": str(teacher_label_path),
        "required_teacher_sources": sorted(required_teacher_sources),
        "train_by_dataset": train_by_dataset,
        "eval_by_dataset": eval_by_dataset,
        "diagnostics_by_dataset": diagnostics_by_dataset,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
