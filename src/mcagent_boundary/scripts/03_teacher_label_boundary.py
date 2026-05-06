from __future__ import annotations

import argparse
import json
import sys
import threading
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mcagent_boundary.annotation.teacher_label import label_boundary_records
from mcagent_boundary.config import load_boundary_config, resolve_repo_path
from mcagent_boundary.io import read_jsonl, write_jsonl, write_jsonl_by_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Run teacher labeling on boundary records.")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output directory.")
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Override input directory for boundary/clear_external JSONL files.",
    )
    parser.add_argument(
        "--include-clear-answer",
        action="store_true",
        help="Also label clear-answer anchors so pair construction never has unlabeled anchors.",
    )
    parser.add_argument(
        "--strict-teacher",
        action="store_true",
        help="Disable rule fallback; fail immediately if an LLM teacher call fails.",
    )
    parser.add_argument(
        "--resume-existing",
        action="store_true",
        help="Reuse labels already present in the output teacher_labels.jsonl and only label missing states.",
    )
    parser.add_argument(
        "--teacher-workers",
        type=int,
        default=None,
        help="Number of concurrent LLM teacher calls. Defaults to teacher.workers or 1.",
    )
    parser.add_argument(
        "--teacher-rpm-limit",
        type=int,
        default=None,
        help="Client-side requests-per-minute limit for teacher calls. Defaults to teacher.rpm_limit.",
    )
    parser.add_argument(
        "--checkpoint-flush-every",
        type=int,
        default=None,
        help="Append completed teacher labels every N records. Defaults to teacher.checkpoint_flush_every or 100.",
    )
    parser.add_argument(
        "--skip-failed-teacher",
        action="store_true",
        help="Record irrecoverable teacher failures and continue without emitting fallback labels.",
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
    clear_external = read_jsonl(_in("clear_external_output"))
    clear_answer = read_jsonl(_in("clear_answer_output")) if args.include_clear_answer else []
    records = boundary_records + clear_answer + clear_external
    output_path = _out("teacher_label_output")
    failure_path = output_path.with_name("teacher_label_failures.jsonl")
    if not args.resume_existing and output_path.exists():
        output_path.write_text("", encoding="utf-8")
        if failure_path.exists():
            failure_path.write_text("", encoding="utf-8")

    existing_labels = read_jsonl(output_path) if args.resume_existing and output_path.exists() else []
    existing_by_state = {
        str(label.get("state_id")): label
        for label in existing_labels
        if label.get("state_id")
    }
    existing_failures = read_jsonl(failure_path) if args.skip_failed_teacher and failure_path.exists() else []
    failed_by_state = {
        str(item.get("state_id")): item
        for item in existing_failures
        if item.get("state_id")
    }
    records_to_label = [
        record
        for record in records
        if str(record.get("state_id")) not in existing_by_state
        and str(record.get("state_id")) not in failed_by_state
    ]

    checkpoint_flush_every = int(
        args.checkpoint_flush_every
        or config.get("teacher", {}).get("checkpoint_flush_every")
        or 100
    )
    checkpoint_flush_every = max(1, checkpoint_flush_every)
    append_lock = threading.Lock()
    pending_checkpoint_labels: list[dict] = []

    def _flush_checkpoints() -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with append_lock:
            if not pending_checkpoint_labels:
                return
            with output_path.open("a", encoding="utf-8") as handle:
                for item in pending_checkpoint_labels:
                    handle.write(json.dumps(item, ensure_ascii=False) + "\n")
            pending_checkpoint_labels.clear()

    def _append_checkpoint(label: dict) -> None:
        with append_lock:
            pending_checkpoint_labels.append(label)
            should_flush = len(pending_checkpoint_labels) >= checkpoint_flush_every
        if should_flush:
            _flush_checkpoints()

    def _append_failure(record: dict, exc: Exception) -> None:
        failure_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "state_id": record.get("state_id"),
            "example_id": record.get("example_id"),
            "dataset": record.get("dataset"),
            "boundary_type": record.get("boundary_type"),
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        with append_lock:
            with failure_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    try:
        new_labels = label_boundary_records(
            records_to_label,
            config,
            show_progress=not args.no_progress,
            allow_rule_fallback=not args.strict_teacher,
            teacher_workers=args.teacher_workers,
            rpm_limit=args.teacher_rpm_limit,
            on_label=_append_checkpoint,
            skip_failed=args.skip_failed_teacher,
            on_failure=_append_failure if args.skip_failed_teacher else None,
        )
    finally:
        _flush_checkpoints()
    new_by_state = {
        str(label.get("state_id")): label
        for label in new_labels
        if label.get("state_id")
    }
    labels = [
        new_by_state.get(str(record.get("state_id")))
        or existing_by_state.get(str(record.get("state_id")))
        for record in records
    ]
    labels = [label for label in labels if label is not None]
    write_jsonl(output_path, labels)
    labels_by_dataset = write_jsonl_by_dataset(output_path, labels)
    source_counts: dict[str, int] = {}
    for label in labels:
        source = str(label.get("source", "unknown"))
        source_counts[source] = source_counts.get(source, 0) + 1
    print(json.dumps({
        "output": str(output_path),
        "by_dataset": labels_by_dataset,
        "num_labels": len(labels),
        "reused_existing": len(existing_by_state),
        "skipped_existing_failures": len(failed_by_state),
        "new_labels": len(new_labels),
        "source_counts": source_counts,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
