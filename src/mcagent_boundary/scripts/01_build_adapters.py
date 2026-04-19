from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mcagent_boundary.config import load_boundary_config, resolve_repo_path
from mcagent_boundary.io import write_json, write_jsonl
from mcagent_boundary.rollout.generate_rollouts import load_standardized_examples


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize standardized adapter outputs.")
    parser.add_argument("--limit-per-dataset", type=int, default=None)
    args = parser.parse_args()

    config = load_boundary_config()
    adapter_cache_dir = resolve_repo_path(config["paths"]["adapter_cache_dir"], config)
    adapter_cache_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict] = {}
    for split_name in ("train", "eval"):
        datasets = list(config["datasets"][split_name])
        examples = load_standardized_examples(
            config,
            dataset_names=datasets,
            limit_per_dataset=args.limit_per_dataset if args.limit_per_dataset is not None else config["rollout"]["limit_per_dataset"],
        )
        for dataset in datasets:
            subset = [example.to_record() for example in examples if example.dataset == dataset]
            output_path = adapter_cache_dir / f"{split_name}_{dataset}.jsonl"
            write_jsonl(output_path, subset)
            summary[f"{split_name}:{dataset}"] = {"num_examples": len(subset), "output": str(output_path)}
    summary_path = adapter_cache_dir / "summary.json"
    write_json(summary_path, summary)
    print(json.dumps({"output": str(summary_path), "summary": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
