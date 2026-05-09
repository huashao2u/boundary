from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mcagent_boundary.config import load_boundary_config, resolve_repo_path
from mcagent_boundary.io import read_jsonl
from mcagent_boundary.training.run_step_dpo import run_step_dpo


def main() -> None:
    parser = argparse.ArgumentParser(description="Run boundary-focused Step-DPO training.")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--train-pair-file",
        type=str,
        default=None,
        help="Override train pair JSONL path. Defaults to paths.train_pair_output.",
    )
    parser.add_argument(
        "--eval-pair-file",
        type=str,
        default=None,
        help="Override eval/validation pair JSONL path. Defaults to paths.eval_pair_output.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override DPO checkpoint output dir. Defaults to paths.dpo_output_dir.",
    )
    args = parser.parse_args()

    config = load_boundary_config()
    train_pair_path = (
        Path(args.train_pair_file).resolve()
        if args.train_pair_file
        else resolve_repo_path(config["paths"]["train_pair_output"], config)
    )
    eval_pair_path = (
        Path(args.eval_pair_file).resolve()
        if args.eval_pair_file
        else resolve_repo_path(config["paths"]["eval_pair_output"], config)
    )
    output_dir = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else resolve_repo_path(config["paths"]["dpo_output_dir"], config)
    )
    train_pairs = read_jsonl(train_pair_path)
    eval_pairs = read_jsonl(eval_pair_path)
    metrics = run_step_dpo(
        train_pairs=train_pairs,
        eval_pairs=eval_pairs,
        config=config,
        output_dir=output_dir,
        smoke=args.smoke,
    )
    metrics["train_pair_file"] = str(train_pair_path)
    metrics["eval_pair_file"] = str(eval_pair_path)
    metrics["output_dir"] = str(output_dir)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
