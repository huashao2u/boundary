from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mcagent_boundary.annotation.teacher_label import label_boundary_records
from mcagent_boundary.config import load_boundary_config, resolve_repo_path
from mcagent_boundary.io import read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Run teacher labeling on boundary records.")
    parser.add_argument("--output-dir", type=str, default=None, help="Override output directory.")
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Override input directory for boundary/clear_external JSONL files.",
    )
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
    records = boundary_records + clear_external
    labels = label_boundary_records(records, config)
    output_path = _out("teacher_label_output")
    write_jsonl(output_path, labels)
    print(json.dumps({"output": str(output_path), "num_labels": len(labels)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
