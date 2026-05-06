from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mcagent_boundary.config import load_boundary_config, resolve_repo_path
from mcagent_boundary.io import read_jsonl, write_json, write_jsonl, write_jsonl_by_dataset
from mcagent_boundary.mining.anchor_sampling import sample_anchor_pools
from mcagent_boundary.mining.boundary_mining import mine_boundary_states


def main() -> None:
    parser = argparse.ArgumentParser(description="Mine boundary states from existing student rollouts.")
    parser.add_argument("--input-dir", type=str, default=None, help="Directory containing all_rollouts.jsonl.")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory for mining outputs.")
    args = parser.parse_args()

    config = load_boundary_config()

    def _path(name: str, override_dir: str | None) -> Path:
        base = resolve_repo_path(config["paths"][name], config)
        if override_dir:
            return Path(override_dir) / base.name
        return base

    rollout_path = _path("rollout_output", args.input_dir)
    rollouts = read_jsonl(rollout_path)
    mined = mine_boundary_states(rollouts, config)
    sampled = sample_anchor_pools(mined, config)

    summary = {
        **mined["summary"],
        "sampling_report": sampled.get("sampling_report", {}),
    }
    mining_output = _path("mining_output", args.output_dir)
    write_json(mining_output, summary)

    boundary_path = _path("boundary_candidates_output", args.output_dir)
    clear_answer_path = _path("clear_answer_output", args.output_dir)
    clear_external_path = _path("clear_external_output", args.output_dir)
    write_jsonl(boundary_path, sampled["boundary_candidates"])
    write_jsonl(clear_answer_path, sampled["clear_answer_anchors"])
    write_jsonl(clear_external_path, sampled["clear_external_anchors"])
    boundary_by_dataset = write_jsonl_by_dataset(boundary_path, sampled["boundary_candidates"])
    clear_answer_by_dataset = write_jsonl_by_dataset(clear_answer_path, sampled["clear_answer_anchors"])
    clear_external_by_dataset = write_jsonl_by_dataset(clear_external_path, sampled["clear_external_anchors"])

    print(
        json.dumps(
            {
                "rollout_input": str(rollout_path),
                "mining_output": str(mining_output),
                "boundary_output": str(boundary_path),
                "clear_answer_output": str(clear_answer_path),
                "clear_external_output": str(clear_external_path),
                "boundary_by_dataset": boundary_by_dataset,
                "clear_answer_by_dataset": clear_answer_by_dataset,
                "clear_external_by_dataset": clear_external_by_dataset,
                "summary": summary,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
