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
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mcagent_boundary.config import load_boundary_config, resolve_repo_path
from mcagent_boundary.evaluation.eval_actions import evaluate_actions
from mcagent_boundary.evaluation.eval_calibration import evaluate_calibration
from mcagent_boundary.evaluation.eval_task_metrics import evaluate_task_metrics
from mcagent_boundary.io import write_json, write_jsonl, write_jsonl_by_dataset
from mcagent_boundary.rollout.dataset_selection import load_fixed_example_ids, selection_summary
from mcagent_boundary.rollout.generate_rollouts import generate_rollouts


def main() -> None:
    parser = argparse.ArgumentParser(description="Run eval-side rollouts and boundary metrics.")
    parser.add_argument("--limit-per-dataset", type=int, default=None)
    parser.add_argument(
        "--datasets",
        type=str,
        default=None,
        help="Comma-separated eval dataset subset. Defaults to configured eval datasets.",
    )
    parser.add_argument(
        "--fixed-example-ids",
        type=str,
        default=None,
        help="Optional txt/JSON/JSONL file of example_id values for fixed eval sets.",
    )
    parser.add_argument(
        "--backend",
        choices=["hf", "vllm", "heuristic", "auto"],
        default=None,
        help="Override rollout backend from config.",
    )
    parser.add_argument(
        "--adapter-path",
        type=str,
        default=None,
        help="Optional PEFT/LoRA adapter path. Currently requires --backend hf.",
    )
    parser.add_argument(
        "--candidate-temperature",
        type=float,
        default=None,
        help="Override rollout.candidate_temperature for deterministic paired evals.",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Override rollout.max_new_tokens for this eval run.",
    )
    parser.add_argument("--vllm-tensor-parallel-size", type=int, default=None)
    parser.add_argument("--vllm-pipeline-parallel-size", type=int, default=None)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=None)
    parser.add_argument("--vllm-max-model-len", type=int, default=None)
    parser.add_argument(
        "--prompt-mode",
        choices=["single_action", "top_k"],
        default="single_action",
        help="Eval prompt format. Defaults to the DPO single-action decision prompt.",
    )
    parser.add_argument("--output-dir", type=str, default=None, help="Override output directory for eval artifacts.")
    parser.add_argument("--no-progress", action="store_true", help="Disable progress bars.")
    parser.add_argument(
        "--progress-every",
        type=int,
        default=0,
        help="Write a lightweight progress.json and print one line every N completed rollouts.",
    )
    args = parser.parse_args()

    config = load_boundary_config()
    if args.backend is not None:
        config["rollout"]["backend"] = args.backend
    if args.adapter_path is not None:
        config.setdefault("student", {})["adapter_path"] = args.adapter_path
    if args.candidate_temperature is not None:
        config["rollout"]["candidate_temperature"] = args.candidate_temperature
    if args.max_new_tokens is not None:
        config["rollout"]["max_new_tokens"] = args.max_new_tokens
    if args.vllm_tensor_parallel_size is not None:
        config["rollout"]["vllm_tensor_parallel_size"] = args.vllm_tensor_parallel_size
    if args.vllm_pipeline_parallel_size is not None:
        config["rollout"]["vllm_pipeline_parallel_size"] = args.vllm_pipeline_parallel_size
    if args.vllm_gpu_memory_utilization is not None:
        config["rollout"]["vllm_gpu_memory_utilization"] = args.vllm_gpu_memory_utilization
    if args.vllm_max_model_len is not None:
        config["rollout"]["vllm_max_model_len"] = args.vllm_max_model_len
    config.setdefault("rollout", {})["prompt_mode"] = args.prompt_mode
    dataset_names = (
        [item.strip() for item in args.datasets.split(",") if item.strip()]
        if args.datasets
        else list(config["datasets"]["eval"])
    )
    fixed_example_ids = load_fixed_example_ids(args.fixed_example_ids) if args.fixed_example_ids else set()

    def _out(path_key: str) -> Path:
        base = resolve_repo_path(config["paths"][path_key], config)
        if args.output_dir:
            return Path(args.output_dir) / base.name
        return base

    started_at = time.time()
    progress_path = Path(args.output_dir) / "progress.json" if args.output_dir and args.progress_every > 0 else None

    def _progress_callback(state: dict) -> None:
        if args.progress_every <= 0:
            return
        records = int(state.get("records") or 0)
        if records % args.progress_every != 0:
            return
        payload = {
            **state,
            "elapsed_seconds": round(time.time() - started_at, 2),
            "dataset_names": dataset_names,
            "fixed_eval_size": len(fixed_example_ids) if fixed_example_ids else None,
        }
        if progress_path is not None:
            progress_path.parent.mkdir(parents=True, exist_ok=True)
            progress_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(json.dumps({"progress": payload}, ensure_ascii=False), flush=True)

    eval_rollouts = generate_rollouts(
        config,
        dataset_names=dataset_names,
        phase=str(config["rollout"]["search_mode_eval"]),
        limit_per_dataset=(
            args.limit_per_dataset
            if args.limit_per_dataset is not None
            else None
            if fixed_example_ids
            else config["rollout"]["limit_per_dataset"]
        ),
        fixed_example_ids=fixed_example_ids,
        show_progress=not args.no_progress,
        progress_callback=_progress_callback,
    )

    eval_rollout_path = _out("eval_rollout_output")
    write_jsonl(eval_rollout_path, eval_rollouts)
    eval_rollouts_by_dataset = write_jsonl_by_dataset(eval_rollout_path, eval_rollouts)
    action_metrics = evaluate_actions(eval_rollouts)
    task_metrics = evaluate_task_metrics(eval_rollouts)
    calibration_metrics = evaluate_calibration(
        eval_rollouts,
        default_confidence=float(config["evaluation"]["confidence_default"]),
    )
    aggregate = {
        "rollout_output": str(eval_rollout_path),
        "rollouts_by_dataset": eval_rollouts_by_dataset,
        "selection": selection_summary(eval_rollouts),
        "action_metrics": action_metrics,
        "task_metrics": task_metrics,
        "calibration_metrics": calibration_metrics,
    }
    write_json(_out("action_eval_output"), action_metrics)
    write_json(_out("task_eval_output"), task_metrics)
    write_json(_out("calibration_eval_output"), calibration_metrics)
    write_json(_out("aggregate_eval_output"), aggregate)
    print(json.dumps(aggregate, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
