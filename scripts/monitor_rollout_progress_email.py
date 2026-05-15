#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_EXPECTED = {
    "gsm8k": 3000,
    "math": 3000,
    "in3": 1261,
    "mintqa": 7888,
    "or_bench": 5974,
}
DEFAULT_ORDER = ["gsm8k", "math", "in3", "mintqa", "or_bench"]
DEFAULT_RECIPIENTS = ["mario2zxy1234@gmail.com", "1563615421@qq.com"]


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def count_rollouts(path: Path) -> tuple[int, Counter[str]]:
    counts: Counter[str] = Counter()
    total = 0
    if not path.exists():
        return total, counts
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                counts["__json_decode_error__"] += 1
                continue
            counts[str(row.get("dataset", "unknown"))] += 1
    return total, counts


def format_snapshot(run_dir: Path, expected: dict[str, int], order: list[str]) -> tuple[str, bool]:
    output_path = run_dir / "all_rollouts.jsonl"
    total, counts = count_rollouts(output_path)
    expected_total = sum(expected.values())
    done = total >= expected_total and all(counts.get(name, 0) >= expected[name] for name in order)
    current = "finished"
    for name in order:
        if counts.get(name, 0) < expected[name]:
            current = name
            break

    pct = (total / expected_total * 100.0) if expected_total else 0.0
    lines = [
        f"time: {utc_now()}",
        f"run_dir: {run_dir}",
        f"output: {output_path}",
        f"stage: {current}",
        f"total: {total}/{expected_total} ({pct:.2f}%)",
        "",
        "by_dataset:",
    ]
    for name in order:
        value = counts.get(name, 0)
        target = expected[name]
        dataset_pct = value / target * 100.0 if target else 0.0
        marker = " <== current" if name == current else ""
        lines.append(f"  {name}: {value}/{target} ({dataset_pct:.2f}%){marker}")
    extras = {key: value for key, value in counts.items() if key not in expected}
    if extras:
        lines.extend(["", f"extra_counts: {extras}"])
    return "\n".join(lines), done


def send_email(send_email_path: Path, recipients: list[str], subject: str, body: str) -> None:
    for recipient in recipients:
        cmd = [
            "python3",
            str(send_email_path),
            "--to",
            recipient,
            "--subject",
            subject,
            "--body",
            body,
        ]
        subprocess.run(cmd, check=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Email rollout progress snapshots at a fixed interval.")
    parser.add_argument("--run-dir", required=True, type=Path, help="Artifact directory containing all_rollouts.jsonl.")
    parser.add_argument("--interval-minutes", type=float, default=20.0, help="Polling/email interval in minutes.")
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
    run_dir = args.run_dir
    send_email_path = args.send_email
    while True:
        body, done = format_snapshot(run_dir, DEFAULT_EXPECTED, DEFAULT_ORDER)
        subject = f"[boundary] rollout progress {run_dir.name}"
        if done:
            subject = f"[boundary] rollout finished {run_dir.name}"
        print(body, flush=True)
        send_email(send_email_path, recipients, subject, body)
        if args.once or (done and args.stop_when_complete):
            break
        time.sleep(interval_seconds)


if __name__ == "__main__":
    main()
