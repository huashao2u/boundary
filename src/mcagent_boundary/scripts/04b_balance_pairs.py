from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mcagent_boundary.config import load_boundary_config, resolve_repo_path
from mcagent_boundary.io import read_jsonl, write_json, write_jsonl, write_jsonl_by_dataset
from mcagent_boundary.training.balance_pairs import balance_pairs


def main() -> None:
    parser = argparse.ArgumentParser(description="Balance Step-DPO train pairs by dataset and chosen action.")
    parser.add_argument("--input-dir", type=str, default=None, help="Directory containing train_step_dpo_pairs.jsonl.")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory for balanced outputs.")
    args = parser.parse_args()

    config = load_boundary_config()

    def _path(name: str, override_dir: str | None) -> Path:
        base = resolve_repo_path(config["paths"][name], config)
        if override_dir:
            return Path(override_dir) / base.name
        return base

    input_path = _path("train_pair_output", args.input_dir)
    output_base = _path("train_pair_output", args.output_dir)
    output_path = output_base.with_name(output_base.stem + "_balanced" + output_base.suffix)
    report_path = output_base.with_name("pair_balance_report.json")

    pairs = read_jsonl(input_path)
    balanced, report = balance_pairs(pairs, config)
    write_jsonl(output_path, balanced)
    by_dataset = write_jsonl_by_dataset(output_path, balanced)
    report = {**report, "input": str(input_path), "output": str(output_path), "by_dataset": by_dataset}
    write_json(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
