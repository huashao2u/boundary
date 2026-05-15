from __future__ import annotations

import argparse
import json
import sys
import threading
from pathlib import Path
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mcagent_boundary.annotation.teacher_label import label_boundary_records
from mcagent_boundary.config import load_boundary_config, resolve_repo_path
from mcagent_boundary.io import read_jsonl, write_jsonl, write_jsonl_by_dataset


def _iter_jsonl(path: Path) -> Iterable[dict]:
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _state_ids(path: Path) -> set[str]:
    ids: set[str] = set()
    for row in _iter_jsonl(path):
        if row.get("state_id"):
            ids.add(str(row["state_id"]))
    return ids


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
            handles[dataset].write(json.dumps(row, ensure_ascii=False) + "\n")
    finally:
        for handle in handles.values():
            handle.close()
    return outputs


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
    parser.add_argument(
        "--batch-size",
        type=int,
        default=0,
        help=(
            "Stream input records in batches of this size instead of loading all records into memory. "
            "Recommended for full runs under tight cgroup memory limits."
        ),
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

    output_path = _out("teacher_label_output")
    failure_path = output_path.with_name("teacher_label_failures.jsonl")
    if not args.resume_existing and output_path.exists():
        output_path.write_text("", encoding="utf-8")
        if failure_path.exists():
            failure_path.write_text("", encoding="utf-8")

    existing_by_state = _state_ids(output_path) if args.resume_existing and output_path.exists() else set()
    failed_by_state = _state_ids(failure_path) if args.skip_failed_teacher and failure_path.exists() else set()

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
        cause = getattr(exc, "__cause__", None)
        payload = {
            "state_id": record.get("state_id"),
            "example_id": record.get("example_id"),
            "dataset": record.get("dataset"),
            "boundary_type": record.get("boundary_type"),
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        if cause is not None:
            payload["cause_type"] = type(cause).__name__
            payload["cause"] = str(cause)
        with append_lock:
            with failure_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")

    if args.batch_size and args.batch_size > 0:
        input_paths = [_in("boundary_candidates_output")]
        if args.include_clear_answer:
            input_paths.append(_in("clear_answer_output"))
        input_paths.append(_in("clear_external_output"))
        batch_size = max(1, int(args.batch_size))
        total_seen = 0
        skipped_existing = 0
        new_labels_total = 0
        batch: list[dict] = []

        def _run_batch(items: list[dict]) -> None:
            nonlocal new_labels_total
            if not items:
                return
            try:
                labels = label_boundary_records(
                    items,
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
            new_labels_total += len(labels)

        for input_path in input_paths:
            for record in _iter_jsonl(input_path):
                total_seen += 1
                state_id = str(record.get("state_id"))
                if state_id in existing_by_state or state_id in failed_by_state:
                    skipped_existing += 1
                    continue
                batch.append(record)
                if len(batch) >= batch_size:
                    _run_batch(batch)
                    batch.clear()
        _run_batch(batch)
        _flush_checkpoints()
        labels_by_dataset = _write_by_dataset_streaming(output_path)
        source_counts: dict[str, int] = {}
        for label in _iter_jsonl(output_path):
            source = str(label.get("source", "unknown"))
            source_counts[source] = source_counts.get(source, 0) + 1
        print(json.dumps({
            "output": str(output_path),
            "by_dataset": labels_by_dataset,
            "num_labels": _count_jsonl(output_path),
            "num_failures": _count_jsonl(failure_path),
            "input_records_seen": total_seen,
            "reused_existing_or_failed": skipped_existing,
            "new_labels": new_labels_total,
            "source_counts": source_counts,
            "streaming_batch_size": batch_size,
        }, ensure_ascii=False, indent=2))
        return

    boundary_records = read_jsonl(_in("boundary_candidates_output"))
    clear_external = read_jsonl(_in("clear_external_output"))
    clear_answer = read_jsonl(_in("clear_answer_output")) if args.include_clear_answer else []
    records = boundary_records + clear_answer + clear_external
    existing_labels = read_jsonl(output_path) if args.resume_existing and output_path.exists() else []
    existing_label_by_state = {
        str(label.get("state_id")): label
        for label in existing_labels
        if label.get("state_id")
    }
    failed_by_state = _state_ids(failure_path) if args.skip_failed_teacher and failure_path.exists() else set()
    records_to_label = [
        record
        for record in records
        if str(record.get("state_id")) not in existing_label_by_state
        and str(record.get("state_id")) not in failed_by_state
    ]

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
        or existing_label_by_state.get(str(record.get("state_id")))
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
        "reused_existing": len(existing_label_by_state),
        "skipped_existing_failures": len(failed_by_state),
        "new_labels": len(new_labels),
        "source_counts": source_counts,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
