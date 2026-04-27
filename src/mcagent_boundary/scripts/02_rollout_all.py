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
    parser.add_argument(
        "--backend",
        choices=["hf", "vllm", "heuristic", "auto"],
        default=None,
        help="Override rollout backend from config.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Override rollout.max_new_tokens for this run.",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help="Comma-separated dataset subset. Defaults to configured train datasets.",
    )
    parser.add_argument("--output-dir", type=str, default=None, help="Override output directory for all artifacts.")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bars.")
    args = parser.parse_args()

    config = load_boundary_config()
    if args.backend is not None:
        config["rollout"]["backend"] = args.backend
    if args.max_new_tokens is not None:
        config["rollout"]["max_new_tokens"] = args.max_new_tokens
    dataset_names = (
        [item.strip() for item in args.datasets.split(",") if item.strip()]
        if args.datasets
        else list(config["datasets"]["train"])
    )

    rollouts = generate_rollouts(
        config,
        dataset_names=dataset_names,
        phase=str(config["rollout"]["search_mode_train"]),
        limit_per_dataset=args.limit_per_dataset if args.limit_per_dataset is not None else config["rollout"]["limit_per_dataset"],
        show_progress=not args.no_progress,
    )

    def _out(name: str) -> Path:
        base = resolve_repo_path(config["paths"][name], config)
        if args.output_dir:
            return Path(args.output_dir) / base.name
        return base

    rollout_output = _out("rollout_output")
    write_jsonl(rollout_output, rollouts)

    mined = mine_boundary_states(rollouts, config)
    sampled = sample_anchor_pools(mined, config)
    write_json(_out("mining_output"), mined["summary"])
    write_jsonl(_out("boundary_candidates_output"), sampled["boundary_candidates"])
    write_jsonl(_out("clear_answer_output"), sampled["clear_answer_anchors"])
    write_jsonl(_out("clear_external_output"), sampled["clear_external_anchors"])
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
