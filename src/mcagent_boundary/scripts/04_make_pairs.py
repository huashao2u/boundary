from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mcagent_boundary.config import load_boundary_config, resolve_repo_path
from mcagent_boundary.io import read_jsonl, write_jsonl
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
    train_pairs, eval_pairs, diagnostics = build_step_dpo_pairs(selected_records, teacher_labels, config)
    train_path = _out("train_pair_output")
    eval_path = _out("eval_pair_output")
    write_jsonl(train_path, train_pairs)
    write_jsonl(eval_path, eval_pairs)
    diagnostics_path = train_path.with_name("pair_diagnostics.jsonl")
    write_jsonl(diagnostics_path, diagnostics)
    print(
        json.dumps(
            {
                "train_pairs": len(train_pairs),
                "eval_pairs": len(eval_pairs),
                "diagnostics": len(diagnostics),
                "train_output": str(train_path),
                "eval_output": str(eval_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
