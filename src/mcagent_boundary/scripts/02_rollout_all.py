from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mcagent_boundary.config import load_boundary_config, resolve_repo_path
from mcagent_boundary.io import write_json, write_jsonl
from mcagent_boundary.mining.anchor_sampling import sample_anchor_pools
from mcagent_boundary.mining.boundary_mining import mine_boundary_states
from mcagent_boundary.rollout.generate_rollouts import generate_rollouts


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full train-side boundary rollouts and mining.")
    parser.add_argument("--limit-per-dataset", type=int, default=None)
    args = parser.parse_args()

    config = load_boundary_config()
    rollouts = generate_rollouts(
        config,
        dataset_names=list(config["datasets"]["train"]),
        phase=str(config["rollout"]["search_mode_train"]),
        limit_per_dataset=args.limit_per_dataset if args.limit_per_dataset is not None else config["rollout"]["limit_per_dataset"],
    )
    rollout_output = resolve_repo_path(config["paths"]["rollout_output"], config)
    write_jsonl(rollout_output, rollouts)

    mined = mine_boundary_states(rollouts, config)
    sampled = sample_anchor_pools(mined, config)
    write_json(resolve_repo_path(config["paths"]["mining_output"], config), mined["summary"])
    write_jsonl(resolve_repo_path(config["paths"]["boundary_candidates_output"], config), sampled["boundary_candidates"])
    write_jsonl(resolve_repo_path(config["paths"]["clear_answer_output"], config), sampled["clear_answer_anchors"])
    write_jsonl(resolve_repo_path(config["paths"]["clear_external_output"], config), sampled["clear_external_anchors"])
    print(
        json.dumps(
            {
                "rollout_output": str(rollout_output),
                "summary": mined["summary"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
