from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mcagent_boundary.config import load_boundary_config, resolve_repo_path
from mcagent_boundary.io import write_json, write_jsonl
from mcagent_boundary.progress import make_progress
from mcagent_boundary.rollout.dataset_selection import (
    apply_selection_preset,
    filter_by_example_ids,
    load_fixed_example_ids,
    selection_summary,
)
from mcagent_boundary.rollout.generate_rollouts import load_standardized_examples


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize standardized adapter outputs.")
    parser.add_argument("--limit-per-dataset", type=int, default=None)
    parser.add_argument(
        "--full-dataset",
        action="store_true",
        help="Ignore rollout.limit_per_dataset and load all available examples.",
    )
    parser.add_argument(
        "--selection-preset",
        default="none",
        choices=["none", "v023_full_rollout", "v026_full_rollout"],
        help="Apply a named post-load dataset selection plan.",
    )
    parser.add_argument(
        "--fixed-example-ids",
        type=str,
        default=None,
        help="Optional txt/JSON/JSONL file of example_id values for fixed small experiments.",
    )
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bars.")
    args = parser.parse_args()

    config = load_boundary_config()
    fixed_example_ids = load_fixed_example_ids(args.fixed_example_ids) if args.fixed_example_ids else set()
    adapter_cache_dir = resolve_repo_path(config["paths"]["adapter_cache_dir"], config)
    adapter_cache_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, dict] = {}
    split_iter = make_progress(
        ("train", "eval"),
        total=2,
        desc="adapter splits",
        unit="split",
        disable=args.no_progress,
    )
    for split_name in split_iter:
        datasets = list(config["datasets"][split_name])
        limit_per_dataset = (
            None
            if args.full_dataset
            else args.limit_per_dataset if args.limit_per_dataset is not None else config["rollout"]["limit_per_dataset"]
        )
        examples = load_standardized_examples(
            config,
            dataset_names=datasets,
            limit_per_dataset=limit_per_dataset,
            show_progress=not args.no_progress,
        )
        examples = apply_selection_preset(examples, args.selection_preset)
        examples = filter_by_example_ids(examples, fixed_example_ids)
        dataset_iter = make_progress(
            datasets,
            total=len(datasets),
            desc=f"write {split_name} adapters",
            unit="dataset",
            disable=args.no_progress,
        )
        for dataset in dataset_iter:
            subset = [example.to_record() for example in examples if example.dataset == dataset]
            output_path = adapter_cache_dir / f"{split_name}_{dataset}.jsonl"
            write_jsonl(output_path, subset)
            summary[f"{split_name}:{dataset}"] = {"num_examples": len(subset), "output": str(output_path)}
        summary[f"{split_name}:selection"] = selection_summary(examples)
    summary_path = adapter_cache_dir / "summary.json"
    write_json(summary_path, summary)
    print(json.dumps({"output": str(summary_path), "summary": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
