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
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Override training.max_steps for this run.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Override training.max_length for this run.",
    )
    parser.add_argument(
        "--rpo-alpha",
        type=float,
        default=None,
        help=(
            "Add a chosen-NLL anchor (rRPO/RPO style) to the DPO loss. The total loss "
            "becomes sigmoid_dpo + rpo_alpha * mean_token_NLL(chosen). 0 disables it. "
            "Recommended 0.5-1.0 to mitigate chosen logp drift in long-epoch DPO."
        ),
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=None,
        help=(
            "Override training.beta (DPO inverse-temperature / implicit KL strength). "
            "Larger beta = sharper preference gradient AND stronger KL pull toward the "
            "reference, so the policy drifts less from base."
        ),
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=None,
        help=(
            "Override training.gradient_accumulation_steps. Use this to keep the global "
            "batch (per_device * grad_accum * nproc_per_node) constant when changing the "
            "number of GPUs, e.g. 2-card runs should halve it from the single-card value."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Override training.seed (passed to DPOConfig). Changes data shuffling order and "
            "init RNG, so different seeds give genuinely different runs for multi-seed variance."
        ),
    )
    parser.add_argument(
        "--no-sample-weights",
        action="store_true",
        help=(
            "Ablation: force training.use_sample_weights=false (uniform weight). Proves the gain "
            "is not from hand-tuned strong/near_tie/best_mid sample weights. Does not edit the yaml."
        ),
    )
    args = parser.parse_args()

    config = load_boundary_config()
    if args.max_steps is not None:
        config.setdefault("training", {})["max_steps"] = int(args.max_steps)
    if args.max_length is not None:
        config.setdefault("training", {})["max_length"] = int(args.max_length)
    if args.rpo_alpha is not None:
        config.setdefault("training", {})["rpo_alpha"] = float(args.rpo_alpha)
    if args.beta is not None:
        config.setdefault("training", {})["beta"] = float(args.beta)
    if args.gradient_accumulation_steps is not None:
        config.setdefault("training", {})["gradient_accumulation_steps"] = int(args.gradient_accumulation_steps)
    if args.seed is not None:
        config.setdefault("training", {})["seed"] = int(args.seed)
    if args.no_sample_weights:
        config.setdefault("training", {})["use_sample_weights"] = False
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
