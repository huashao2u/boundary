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
    args = parser.parse_args()

    config = load_boundary_config()
    train_pairs = read_jsonl(resolve_repo_path(config["paths"]["train_pair_output"], config))
    eval_pairs = read_jsonl(resolve_repo_path(config["paths"]["eval_pair_output"], config))
    metrics = run_step_dpo(
        train_pairs=train_pairs,
        eval_pairs=eval_pairs,
        config=config,
        output_dir=resolve_repo_path(config["paths"]["dpo_output_dir"], config),
        smoke=args.smoke,
    )
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
