#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_RECIPIENTS = ["mario2zxy1234@gmail.com", "1563615421@qq.com"]
DEFAULT_INPUTS = [
    ("boundary", "boundary_candidates.jsonl"),
    ("clear_answer", "clear_answer_anchors.jsonl"),
    ("clear_external", "clear_external_anchors.jsonl"),
]


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                rows.append({"__json_decode_error__": True})
    return rows


def get_state_id(row: dict[str, Any]) -> str:
    return str(row.get("state_id") or row.get("id") or row.get("example_id") or "")


def get_dataset(row: dict[str, Any]) -> str:
    dataset = row.get("dataset")
    if dataset:
        return str(dataset)
    state = row.get("state")
    if isinstance(state, dict):
        z_t = state.get("z_t")
        if isinstance(z_t, dict):
            metadata = z_t.get("metadata")
            if isinstance(metadata, dict) and metadata.get("dataset"):
                return str(metadata["dataset"])
    metadata = row.get("metadata")
    if isinstance(metadata, dict) and metadata.get("dataset"):
        return str(metadata["dataset"])
    return "unknown"


def build_input_index(run_dir: Path) -> tuple[dict[str, dict[str, str]], Counter[str], Counter[str]]:
    index: dict[str, dict[str, str]] = {}
    by_dataset: Counter[str] = Counter()
    by_source: Counter[str] = Counter()
    for source, filename in DEFAULT_INPUTS:
        for row in load_jsonl(run_dir / filename):
            if row.get("__json_decode_error__"):
                by_source[f"{source}:json_decode_error"] += 1
                continue
            state_id = get_state_id(row)
            dataset = get_dataset(row)
            if state_id:
                index[state_id] = {"dataset": dataset, "input_source": source}
            by_dataset[dataset] += 1
            by_source[source] += 1
    return index, by_dataset, by_source


def count_outputs(
    rows: list[dict[str, Any]],
    input_index: dict[str, dict[str, str]],
) -> tuple[Counter[str], Counter[str], Counter[str]]:
    by_dataset: Counter[str] = Counter()
    by_source: Counter[str] = Counter()
    errors: Counter[str] = Counter()
    for row in rows:
        if row.get("__json_decode_error__"):
            errors["json_decode_error"] += 1
            continue
        state_id = get_state_id(row)
        indexed = input_index.get(state_id, {})
        dataset = str(row.get("dataset") or indexed.get("dataset") or "unknown")
        input_source = str(indexed.get("input_source") or "unknown")
        by_dataset[dataset] += 1
        by_source[input_source] += 1
        if row.get("error"):
            errors[str(row.get("error", "unknown_error"))] += 1
        elif row.get("error_type"):
            errors[str(row.get("error_type", "unknown_error"))] += 1
    return by_dataset, by_source, errors


def process_status(pid: int | None) -> str:
    if not pid:
        return "not checked"
    result = subprocess.run(
        ["ps", "-p", str(pid), "-o", "pid=,stat=,etime=,pcpu=,rss="],
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() or "not running"


def format_counter(counter: Counter[str], expected_keys: list[str] | None = None) -> list[str]:
    keys = expected_keys or sorted(counter)
    lines = []
    for key in keys:
        if key in counter or expected_keys:
            lines.append(f"  {key}: {counter.get(key, 0)}")
    for key in sorted(set(counter) - set(keys)):
        lines.append(f"  {key}: {counter[key]}")
    return lines


def format_snapshot(run_dir: Path, pid: int | None) -> tuple[str, bool]:
    input_index, input_by_dataset, input_by_source = build_input_index(run_dir)
    expected_total = sum(input_by_source.values())
    labels = load_jsonl(run_dir / "teacher_labels.jsonl")
    failures = load_jsonl(run_dir / "teacher_label_failures.jsonl")
    label_by_dataset, label_by_source, label_errors = count_outputs(labels, input_index)
    failure_by_dataset, failure_by_source, failure_errors = count_outputs(failures, input_index)

    done_total = len(labels) + len(failures)
    pct = done_total / expected_total * 100.0 if expected_total else 0.0
    done = expected_total > 0 and done_total >= expected_total
    subject_status = "finished" if done else "running"
    process = process_status(pid)

    lines = [
        f"time: {utc_now()}",
        f"run_dir: {run_dir}",
        f"stage: teacher_label_{subject_status}",
        f"process: {process}",
        f"progress: {done_total}/{expected_total} ({pct:.2f}%)",
        f"success: {len(labels)}",
        f"failures: {len(failures)}",
        "",
        "expected_by_source:",
        *format_counter(input_by_source, [source for source, _ in DEFAULT_INPUTS]),
        "",
        "success_by_source:",
        *format_counter(label_by_source, [source for source, _ in DEFAULT_INPUTS]),
        "",
        "failure_by_source:",
        *format_counter(failure_by_source, [source for source, _ in DEFAULT_INPUTS]),
        "",
        "expected_by_dataset:",
        *format_counter(input_by_dataset),
        "",
        "success_by_dataset:",
        *format_counter(label_by_dataset, sorted(input_by_dataset)),
        "",
        "failure_by_dataset:",
        *format_counter(failure_by_dataset, sorted(input_by_dataset)),
    ]
    combined_errors = failure_errors + label_errors
    if combined_errors:
        lines.extend(["", "top_failure_reasons:"])
        for reason, count in combined_errors.most_common(8):
            lines.append(f"  {count}: {reason}")
    return "\n".join(lines), done


def send_email(send_email_path: Path, recipients: list[str], subject: str, body: str) -> None:
    for recipient in recipients:
        subprocess.run(
            [
                "python3",
                str(send_email_path),
                "--to",
                recipient,
                "--subject",
                subject,
                "--body",
                body,
            ],
            check=False,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Email teacher-labeling progress snapshots.")
    parser.add_argument("--run-dir", required=True, type=Path, help="Artifact directory for the teacher labeling run.")
    parser.add_argument("--interval-minutes", type=float, default=20.0, help="Polling/email interval in minutes.")
    parser.add_argument("--pid", type=int, default=None, help="Teacher labeling process id to include in snapshots.")
    parser.add_argument(
        "--to",
        action="append",
        default=[],
        help="Recipient email. Can be repeated. Defaults to the project Gmail and QQ recipients.",
    )
    parser.add_argument("--send-email", type=Path, default=Path("send_email.py"), help="Path to send_email.py.")
    parser.add_argument("--once", action="store_true", help="Send one snapshot and exit.")
    parser.add_argument("--stop-when-complete", action="store_true", help="Exit after sending the completed snapshot.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    recipients = args.to or DEFAULT_RECIPIENTS
    interval_seconds = max(1.0, args.interval_minutes * 60.0)
    while True:
        body, done = format_snapshot(args.run_dir, args.pid)
        subject_state = "finished" if done else "progress"
        subject = f"[boundary] teacher {subject_state} {args.run_dir.name}"
        print(body, flush=True)
        send_email(args.send_email, recipients, subject, body)
        if args.once or (done and args.stop_when_complete):
            break
        time.sleep(interval_seconds)


if __name__ == "__main__":
    main()
