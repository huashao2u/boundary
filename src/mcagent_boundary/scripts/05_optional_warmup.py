from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mcagent_boundary.config import load_boundary_config, resolve_repo_path
from mcagent_boundary.io import read_jsonl, write_json, write_jsonl
from mcagent_boundary.training.make_sft_data import build_warmup_sft_dataset
from mcagent_boundary.training.run_optional_warmup_sft import maybe_run_warmup_sft


def main() -> None:
    parser = argparse.ArgumentParser(description="Build optional warm-up SFT data and optionally train.")
    parser.add_argument("--run-training", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    config = load_boundary_config()
    rollouts = read_jsonl(resolve_repo_path(config["paths"]["rollout_output"], config))
    teacher_labels = read_jsonl(resolve_repo_path(config["paths"]["teacher_label_output"], config))
    boundary_records = read_jsonl(resolve_repo_path(config["paths"]["boundary_candidates_output"], config))
    clear_external = read_jsonl(resolve_repo_path(config["paths"]["clear_external_output"], config))
    selected_records = boundary_records + clear_external
    sft_records = build_warmup_sft_dataset(selected_records, teacher_labels, config)
    sft_output = resolve_repo_path(config["paths"]["sft_output"], config)
    write_jsonl(sft_output, sft_records)

    answer_count = sum(1 for rollout in rollouts if rollout.get("natural_action") == "ANSWER")
    non_answer_rate = None if not rollouts else 1 - (answer_count / len(rollouts))
    summary = maybe_run_warmup_sft(
        sft_records=sft_records,
        non_answer_rate=non_answer_rate,
        config=config,
        output_dir=resolve_repo_path(config["paths"]["warmup_output_dir"], config),
        run_training=args.run_training,
        smoke=args.smoke,
    )
    summary["sft_output"] = str(sft_output)
    write_json(resolve_repo_path(config["paths"]["warmup_output_dir"], config) / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
