from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mcagent_boundary.adapters.base import StandardizedExample
from mcagent_boundary.config import load_boundary_config
from mcagent_boundary.io import read_jsonl, write_json, write_jsonl, write_jsonl_by_dataset
from mcagent_boundary.progress import make_progress
from mcagent_boundary.rollout.branch_actions import rollout_one_example

_PAIR_EVAL_PATH = Path(__file__).resolve().with_name("07b_eval_dpo_pairs.py")
_PAIR_EVAL_SPEC = importlib.util.spec_from_file_location("mcagent_boundary_07b_eval_dpo_pairs", _PAIR_EVAL_PATH)
if _PAIR_EVAL_SPEC is None or _PAIR_EVAL_SPEC.loader is None:
    raise RuntimeError(f"Could not load pair eval helpers from {_PAIR_EVAL_PATH}")
pair_eval = importlib.util.module_from_spec(_PAIR_EVAL_SPEC)
_PAIR_EVAL_SPEC.loader.exec_module(pair_eval)


def _example_from_record(record: dict[str, Any]) -> StandardizedExample:
    return StandardizedExample(
        example_id=str(record["example_id"]),
        dataset=str(record["dataset"]),
        split=str(record.get("split") or (record.get("metadata") or {}).get("split") or "test"),
        question=str(record["question"]),
        gold_answer=record.get("gold_answer"),
        metadata=dict(record.get("metadata") or {}),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate end-to-end student behavior on standardized test examples.")
    parser.add_argument("--examples-file", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--backend", choices=["hf", "vllm", "heuristic", "auto"], default="vllm")
    parser.add_argument("--adapter-path", default=None)
    parser.add_argument("--candidate-temperature", type=float, default=0.0)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--vllm-tensor-parallel-size", type=int, default=None)
    parser.add_argument("--vllm-pipeline-parallel-size", type=int, default=None)
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=None)
    parser.add_argument("--vllm-max-model-len", type=int, default=None)
    parser.set_defaults(prompt_mode="end_to_end")
    parser.add_argument("--tool-finalize-depth", type=int, default=2)
    parser.add_argument("--teacher-judge-workers", type=int, default=8)
    parser.add_argument("--skip-mintqa-teacher-judge", action="store_true")
    parser.add_argument("--skip-in3-teacher-judge", action="store_true")
    parser.add_argument("--skip-or-bench-teacher-judge", action="store_true")
    parser.add_argument("--resume-existing", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    config = load_boundary_config()
    config.setdefault("rollout", {})["backend"] = args.backend
    config["rollout"]["prompt_mode"] = "single_action"
    config["rollout"]["candidate_temperature"] = args.candidate_temperature
    config["rollout"]["max_new_tokens"] = args.max_new_tokens
    if args.vllm_tensor_parallel_size is not None:
        config["rollout"]["vllm_tensor_parallel_size"] = args.vllm_tensor_parallel_size
    if args.vllm_pipeline_parallel_size is not None:
        config["rollout"]["vllm_pipeline_parallel_size"] = args.vllm_pipeline_parallel_size
    if args.vllm_gpu_memory_utilization is not None:
        config["rollout"]["vllm_gpu_memory_utilization"] = args.vllm_gpu_memory_utilization
    if args.vllm_max_model_len is not None:
        config["rollout"]["vllm_max_model_len"] = args.vllm_max_model_len
    config.setdefault("eval", {})["tool_finalize_depth"] = max(1, int(args.tool_finalize_depth))
    if args.adapter_path:
        config.setdefault("student", {})["adapter_path"] = args.adapter_path

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rollout_path = output_dir / "eval_test_end_to_end_rollouts.jsonl"

    records: list[dict[str, Any]] = []
    completed_ids: set[str] = set()
    if args.resume_existing and rollout_path.exists():
        records = read_jsonl(rollout_path)
        completed_ids = {str(record.get("example_id")) for record in records}

    examples = [_example_from_record(record) for record in read_jsonl(args.examples_file)]
    todo = [example for example in examples if example.example_id not in completed_ids]
    policy = pair_eval._build_raw_policy(config, args)

    mode = "a" if args.resume_existing and rollout_path.exists() else "w"
    with rollout_path.open(mode, encoding="utf-8") as handle:
        iterator = make_progress(todo, total=len(todo), desc="eval test examples", unit="sample", disable=args.no_progress)
        for example in iterator:
            record = rollout_one_example(example, config=config, phase="eval", policy=policy)
            record["eval_prompt_mode"] = "end_to_end"
            record["actual_rollout_prompt_mode"] = "single_action"
            records.append(record)
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            try:
                iterator.set_postfix(dataset=example.dataset, records=len(records))
            except Exception:
                pass

    by_dataset_paths = write_jsonl_by_dataset(rollout_path, records)
    datasets = {str(record.get("dataset")) for record in records}

    mintqa_judgments: list[dict[str, Any]] = []
    if not args.skip_mintqa_teacher_judge and "mintqa" in datasets:
        mintqa_judgments = pair_eval._attach_teacher_judgments(
            records,
            config,
            dataset="mintqa",
            judge_fn=pair_eval._teacher_judge_mintqa,
            workers=max(1, int(args.teacher_judge_workers)),
            show_progress=not args.no_progress,
        )
        write_jsonl(output_dir / "mintqa_teacher_judgments.jsonl", mintqa_judgments)

    in3_judgments: list[dict[str, Any]] = []
    if not args.skip_in3_teacher_judge and "in3" in datasets:
        in3_judgments = pair_eval._attach_teacher_judgments(
            records,
            config,
            dataset="in3",
            judge_fn=pair_eval._teacher_judge_in3_clarify,
            workers=max(1, int(args.teacher_judge_workers)),
            show_progress=not args.no_progress,
        )
        write_jsonl(output_dir / "in3_clarify_teacher_judgments.jsonl", in3_judgments)

    or_bench_judgments: list[dict[str, Any]] = []
    if not args.skip_or_bench_teacher_judge and "or_bench" in datasets:
        or_bench_judgments = pair_eval._attach_teacher_judgments(
            records,
            config,
            dataset="or_bench",
            judge_fn=pair_eval._teacher_judge_or_bench,
            workers=max(1, int(args.teacher_judge_workers)),
            show_progress=not args.no_progress,
        )
        write_jsonl(output_dir / "or_bench_teacher_judgments.jsonl", or_bench_judgments)

    metrics = pair_eval._summarize_records(
        records,
        mintqa_judgments=mintqa_judgments,
        in3_judgments=in3_judgments,
        or_bench_judgments=or_bench_judgments,
    )
    metrics.update(
        {
            "examples_file": args.examples_file,
            "num_examples": len(examples),
            "num_evaluated": len(records),
            "rollout_output": str(rollout_path),
            "rollouts_by_dataset": by_dataset_paths,
            "search_backend": config.get("tools", {}).get("search", {}).get("eval_backend"),
            "serper_api_key_available": pair_eval._serper_api_key_available(config),
            "tool_finalize_depth": int(config.get("eval", {}).get("tool_finalize_depth", 1)),
            "eval_prompt_mode": "end_to_end",
            "actual_rollout_prompt_mode": "single_action",
            "calculator_backend": (
                (config.get("tools", {}).get("calculator", {}) or {}).get(
                    "backend_eval",
                    (config.get("tools", {}).get("calculator", {}) or {}).get("backend", "python_sandbox"),
                )
            ),
            "mintqa_teacher_judgments": str(output_dir / "mintqa_teacher_judgments.jsonl") if mintqa_judgments else None,
            "in3_clarify_teacher_judgments": str(output_dir / "in3_clarify_teacher_judgments.jsonl") if in3_judgments else None,
            "or_bench_teacher_judgments": str(output_dir / "or_bench_teacher_judgments.jsonl") if or_bench_judgments else None,
        }
    )
    write_json(output_dir / "eval_test_end_to_end_metrics.json", metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
