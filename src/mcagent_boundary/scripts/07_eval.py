from __future__ import annotations

# TODO(boundary eval extension — CLAUDE.md §Change 6):
#   Future eval datasets will include `truthfulqa` and `realtimeqa`. Stub
#   adapters already live at:
#     - src/mcagent_boundary/adapters/truthfulqa.py
#     - src/mcagent_boundary/adapters/realtimeqa.py
#   Once raw assets exist under `dataset/truthfulqa` and `dataset/realtimeqa`,
#   (1) implement the adapters' `_convert` methods,
#   (2) uncomment the corresponding entries in `configs/rollout.yaml`,
#   (3) this script will pick them up via `config["datasets"]["eval"]`.

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mcagent_boundary.config import load_boundary_config, resolve_repo_path
from mcagent_boundary.evaluation.eval_actions import evaluate_actions
from mcagent_boundary.evaluation.eval_calibration import evaluate_calibration
from mcagent_boundary.evaluation.eval_task_metrics import evaluate_task_metrics
from mcagent_boundary.io import write_json, write_jsonl
from mcagent_boundary.rollout.generate_rollouts import generate_rollouts


def main() -> None:
    parser = argparse.ArgumentParser(description="Run eval-side rollouts and boundary metrics.")
    parser.add_argument("--limit-per-dataset", type=int, default=None)
    args = parser.parse_args()

    config = load_boundary_config()
    eval_rollouts = generate_rollouts(
        config,
        dataset_names=list(config["datasets"]["eval"]),
        phase=str(config["rollout"]["search_mode_eval"]),
        limit_per_dataset=args.limit_per_dataset if args.limit_per_dataset is not None else config["rollout"]["limit_per_dataset"],
    )
    eval_rollout_path = resolve_repo_path(config["paths"]["eval_rollout_output"], config)
    write_jsonl(eval_rollout_path, eval_rollouts)
    action_metrics = evaluate_actions(eval_rollouts)
    task_metrics = evaluate_task_metrics(eval_rollouts)
    calibration_metrics = evaluate_calibration(
        eval_rollouts,
        default_confidence=float(config["evaluation"]["confidence_default"]),
    )
    aggregate = {
        "rollout_output": str(eval_rollout_path),
        "action_metrics": action_metrics,
        "task_metrics": task_metrics,
        "calibration_metrics": calibration_metrics,
    }
    write_json(resolve_repo_path(config["paths"]["action_eval_output"], config), action_metrics)
    write_json(resolve_repo_path(config["paths"]["task_eval_output"], config), task_metrics)
    write_json(resolve_repo_path(config["paths"]["calibration_eval_output"], config), calibration_metrics)
    write_json(resolve_repo_path(config["paths"]["aggregate_eval_output"], config), aggregate)
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
