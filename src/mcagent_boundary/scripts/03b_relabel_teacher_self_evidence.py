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
from mcagent_boundary.io import read_jsonl, write_json, write_jsonl, write_jsonl_by_dataset


def _default_self_evidence_path(config: dict, *, output_dir: str | None) -> Path:
    configured = config.get("paths", {}).get("teacher_self_evidence_label_output")
    if configured:
        base = resolve_repo_path(configured, config)
    else:
        base = resolve_repo_path(config["paths"]["teacher_label_output"], config).with_name(
            "teacher_labels_self_evidence.jsonl"
        )
    if output_dir:
        return Path(output_dir) / base.name
    return base


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Relabel existing rollout/mined records with self-evidence teacher labels."
    )
    parser.add_argument("--output-dir", type=str, default=None, help="Override output directory.")
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Override input directory for boundary/anchor JSONL files.",
    )
    parser.add_argument(
        "--include-clear-answer",
        action="store_true",
        help="Also relabel clear-answer anchors.",
    )
    parser.add_argument(
        "--strict-teacher",
        action="store_true",
        help="Compatibility flag; relabel is strict by default unless --allow-rule-fallback is set.",
    )
    parser.add_argument(
        "--allow-rule-fallback",
        action="store_true",
        help="Allow rule fallback labels for smoke/debug runs.",
    )
    parser.add_argument("--resume-existing", action="store_true", help="Only relabel missing states.")
    parser.add_argument("--teacher-workers", type=int, default=None, help="Concurrent teacher calls.")
    parser.add_argument("--teacher-rpm-limit", type=int, default=None, help="Client-side RPM limit.")
    parser.add_argument(
        "--checkpoint-flush-every",
        type=int,
        default=None,
        help="Append completed labels every N records.",
    )
    parser.add_argument(
        "--skip-failed-teacher",
        action="store_true",
        help="Record teacher failures and continue without fallback labels.",
    )
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bars.")
    args = parser.parse_args()

    config = load_boundary_config()

    def _in(name: str) -> Path:
        base = resolve_repo_path(config["paths"][name], config)
        if args.input_dir:
            return Path(args.input_dir) / base.name
        return base

    boundary_records = read_jsonl(_in("boundary_candidates_output"))
    clear_external = read_jsonl(_in("clear_external_output"))
    clear_answer = read_jsonl(_in("clear_answer_output")) if args.include_clear_answer else []
    records = boundary_records + clear_answer + clear_external

    output_path = _default_self_evidence_path(config, output_dir=args.output_dir)
    failure_path = output_path.with_name("teacher_self_evidence_failures.jsonl")
    report_path = output_path.with_name("relabel_report.json")
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

    allow_rule_fallback = bool(args.allow_rule_fallback)
    try:
        new_labels = label_boundary_records(
            records_to_label,
            config,
            show_progress=not args.no_progress,
            allow_rule_fallback=allow_rule_fallback,
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
    dataset_counts: dict[str, int] = {}
    guardrail_reasons: dict[str, int] = {}
    guardrail_changed = 0
    for label in labels:
        source = str(label.get("source", "unknown"))
        dataset = str(label.get("dataset", "unknown"))
        source_counts[source] = source_counts.get(source, 0) + 1
        dataset_counts[dataset] = dataset_counts.get(dataset, 0) + 1
        summary = label.get("guardrail_summary") or {}
        if summary.get("guardrail_applied"):
            guardrail_changed += 1
        for reason in summary.get("guardrail_reasons") or []:
            guardrail_reasons[str(reason)] = guardrail_reasons.get(str(reason), 0) + 1

    report = {
        "output": str(output_path),
        "by_dataset": labels_by_dataset,
        "num_input_records": len(records),
        "num_labels": len(labels),
        "reused_existing": len(existing_by_state),
        "skipped_existing_failures": len(failed_by_state),
        "new_labels": len(new_labels),
        "source_counts": source_counts,
        "dataset_counts": dataset_counts,
        "guardrail_changed_labels": guardrail_changed,
        "guardrail_reasons": guardrail_reasons,
        "failure_output": str(failure_path),
    }
    write_json(report_path, report)
    print(json.dumps({**report, "report": str(report_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
