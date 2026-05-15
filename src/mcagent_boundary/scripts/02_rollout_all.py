from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mcagent_boundary.config import load_boundary_config, resolve_repo_path
from mcagent_boundary.io import ensure_parent, read_jsonl, to_jsonable, write_jsonl_by_dataset
from mcagent_boundary.rollout.dataset_selection import load_fixed_example_ids, selection_summary
from mcagent_boundary.rollout.generate_rollouts import generate_rollouts


def _record_key(record: dict) -> tuple[str, str]:
    return (str(record.get("dataset", "")), str(record.get("example_id", "")))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run train-side student rollouts only.")
    parser.add_argument("--limit-per-dataset", type=int, default=None)
    parser.add_argument(
        "--full-dataset",
        action="store_true",
        help="Ignore rollout.limit_per_dataset and rollout every available example.",
    )
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
    parser.add_argument("--output-dir", type=str, default=None, help="Override output directory for all artifacts.")
    parser.add_argument(
        "--resume-existing",
        action="store_true",
        help="Append to an existing all_rollouts.jsonl and skip completed (dataset, example_id) records.",
    )
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

    fixed_example_ids = load_fixed_example_ids(args.fixed_example_ids) if args.fixed_example_ids else set()

    def _out(name: str) -> Path:
        base = resolve_repo_path(config["paths"][name], config)
        if args.output_dir:
            return Path(args.output_dir) / base.name
        return base

    rollout_output = _out("rollout_output")
    ensure_parent(rollout_output)
    existing_rollouts = read_jsonl(rollout_output) if args.resume_existing and rollout_output.exists() else []
    completed_keys = {_record_key(record) for record in existing_rollouts}
    output_mode = "a" if args.resume_existing else "w"

    with rollout_output.open(output_mode, encoding="utf-8") as handle:

        def append_record(record: dict) -> None:
            handle.write(json.dumps(to_jsonable(record), ensure_ascii=False) + "\n")
            handle.flush()

        rollouts = generate_rollouts(
            config,
            dataset_names=dataset_names,
            phase=str(config["rollout"]["search_mode_train"]),
            limit_per_dataset=(
                None
                if args.full_dataset
                else args.limit_per_dataset if args.limit_per_dataset is not None else config["rollout"]["limit_per_dataset"]
            ),
            selection_preset=args.selection_preset,
            fixed_example_ids=fixed_example_ids,
            skip_example_keys=completed_keys,
            show_progress=not args.no_progress,
            record_callback=append_record,
        )

    all_rollouts = read_jsonl(rollout_output)
    rollout_by_dataset = write_jsonl_by_dataset(rollout_output, all_rollouts)
    print(
        json.dumps(
            {
                "rollout_output": str(rollout_output),
                "rollout_by_dataset": rollout_by_dataset,
                "selection": selection_summary(all_rollouts),
                "resume": {
                    "enabled": args.resume_existing,
                    "existing_records": len(existing_rollouts),
                    "new_records": len(rollouts),
                    "total_records": len(all_rollouts),
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
