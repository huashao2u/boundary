from __future__ import annotations

import argparse
import json
import sys
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
        help="Disable rule fallback; fail immediately if a Poe teacher call fails.",
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
    labels = label_boundary_records(
        records,
        config,
        show_progress=not args.no_progress,
        allow_rule_fallback=not args.strict_teacher,
    )
    output_path = _out("teacher_label_output")
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
        "source_counts": source_counts,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
