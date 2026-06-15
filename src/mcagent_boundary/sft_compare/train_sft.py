from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from transformers import Trainer, TrainingArguments

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mcagent_boundary.config import load_boundary_config
from mcagent_boundary.io import read_jsonl, write_json
from mcagent_boundary.training.common import load_model_and_tokenizer, resolve_model_path


DEFAULT_SFT_DIR = REPO_ROOT / "artifacts/sft_compare/pairs_v026_20260527T_boundary_from_rollout171000"
logger = logging.getLogger(__name__)


@dataclass
class CompletionOnlyChatCollator:
    tokenizer: Any
    max_length: int
    truncation_mode: str = "keep_end"
    prefix_alignment_tolerance: int = 8
    num_examples: int = 0
    num_truncated: int = 0
    num_zero_loss: int = 0
    num_prefix_alignment_warnings: int = 0
    max_prefix_delta: int = 0

    def _chat_ids(self, messages: list[dict[str, str]], *, add_generation_prompt: bool) -> list[int]:
        return list(
            self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=add_generation_prompt,
            )
        )

    @staticmethod
    def _common_prefix_len(left: list[int], right: list[int]) -> int:
        limit = min(len(left), len(right))
        index = 0
        while index < limit and left[index] == right[index]:
            index += 1
        return index

    def _truncate(self, input_ids: list[int], labels: list[int]) -> tuple[list[int], list[int]]:
        if len(input_ids) <= self.max_length:
            return input_ids, labels
        self.num_truncated += 1
        if self.truncation_mode == "keep_start":
            return input_ids[: self.max_length], labels[: self.max_length]
        return input_ids[-self.max_length :], labels[-self.max_length :]

    def _encode_one(self, row: dict[str, Any]) -> dict[str, list[int]]:
        prompt_messages = list(row["prompt_messages"])
        completion_messages = list(row.get("completion_messages") or row.get("chosen_messages") or [])
        if not completion_messages:
            raise ValueError("SFT row is missing completion_messages/chosen_messages.")
        prompt_ids = self._chat_ids(prompt_messages, add_generation_prompt=True)
        full_ids = self._chat_ids(prompt_messages + completion_messages, add_generation_prompt=False)
        prompt_len = self._common_prefix_len(prompt_ids, full_ids)
        if prompt_len == len(full_ids):
            prompt_len = min(len(prompt_ids), len(full_ids))
        prefix_delta = abs(prompt_len - len(prompt_ids))
        self.max_prefix_delta = max(self.max_prefix_delta, prefix_delta)
        if prefix_delta > self.prefix_alignment_tolerance:
            self.num_prefix_alignment_warnings += 1
            if self.num_prefix_alignment_warnings <= 5:
                logger.warning(
                    "Large SFT chat-template prefix mismatch: prompt_len=%s len(prompt_ids)=%s delta=%s",
                    prompt_len,
                    len(prompt_ids),
                    prefix_delta,
                )
        labels = [-100] * prompt_len + full_ids[prompt_len:]
        input_ids, labels = self._truncate(full_ids, labels)
        self.num_examples += 1
        if not any(label != -100 for label in labels):
            self.num_zero_loss += 1
        attention_mask = [1] * len(input_ids)
        return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        encoded = [self._encode_one(feature) for feature in features]
        max_len = max(len(item["input_ids"]) for item in encoded)
        pad_id = self.tokenizer.pad_token_id
        batch = {"input_ids": [], "attention_mask": [], "labels": []}
        for item in encoded:
            pad_len = max_len - len(item["input_ids"])
            batch["input_ids"].append(item["input_ids"] + [pad_id] * pad_len)
            batch["attention_mask"].append(item["attention_mask"] + [0] * pad_len)
            batch["labels"].append(item["labels"] + [-100] * pad_len)
        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}

    def diagnostics(self) -> dict[str, Any]:
        return {
            "num_collated_examples": self.num_examples,
            "num_truncated_examples": self.num_truncated,
            "truncated_example_rate": None if self.num_examples == 0 else self.num_truncated / self.num_examples,
            "num_zero_loss_examples": self.num_zero_loss,
            "num_prefix_alignment_warnings": self.num_prefix_alignment_warnings,
            "max_prefix_delta_tokens": self.max_prefix_delta,
            "prefix_alignment_tolerance": self.prefix_alignment_tolerance,
            "truncation_mode": self.truncation_mode,
        }


def _serializable_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in metrics.items()
        if isinstance(value, (int, float, str, bool)) or value is None
    }


def _trainer_row(row: dict[str, Any]) -> dict[str, Any]:
    completion_messages = list(row.get("completion_messages") or row.get("chosen_messages") or [])
    if not completion_messages:
        raise ValueError("SFT row is missing completion_messages/chosen_messages.")
    return {
        "sft_id": str(row.get("sft_id") or row.get("pair_id") or ""),
        "dataset": str(row.get("dataset") or ""),
        "prompt_messages": list(row["prompt_messages"]),
        "completion_messages": completion_messages,
    }


def run_sft(
    train_records: list[dict[str, Any]],
    eval_records: list[dict[str, Any]],
    config: dict[str, Any],
    output_dir: str | Path,
    *,
    smoke: bool = False,
) -> dict[str, Any]:
    if not train_records:
        raise ValueError("No SFT train records were provided.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    training_cfg = config["training"]
    raw_train = train_records[: min(len(train_records), 64)] if smoke else train_records
    raw_eval = eval_records[: min(len(eval_records), 16)] if smoke else eval_records
    train_rows = [_trainer_row(row) for row in raw_train]
    eval_rows = [_trainer_row(row) for row in raw_eval]

    model, tokenizer = load_model_and_tokenizer(config)
    collator = CompletionOnlyChatCollator(
        tokenizer=tokenizer,
        max_length=int(training_cfg["max_length"]),
        truncation_mode=str(training_cfg.get("truncation_mode", "keep_end")),
    )
    args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_train_batch_size=int(training_cfg["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(training_cfg.get("per_device_eval_batch_size", 1)),
        gradient_accumulation_steps=int(training_cfg["gradient_accumulation_steps"]),
        learning_rate=float(training_cfg["learning_rate"]),
        max_steps=int(training_cfg["smoke_max_steps"] if smoke else training_cfg["max_steps"]),
        warmup_steps=int(training_cfg["warmup_steps"]),
        logging_steps=int(training_cfg["logging_steps"]),
        eval_steps=int(training_cfg["eval_steps"]),
        save_steps=int(training_cfg["save_steps"]),
        save_total_limit=(
            int(training_cfg["save_total_limit"]) if training_cfg.get("save_total_limit") is not None else None
        ),
        fp16=bool(training_cfg.get("fp16", False)),
        bf16=bool(training_cfg.get("bf16", False)),
        gradient_checkpointing=bool(training_cfg.get("gradient_checkpointing", True)),
        remove_unused_columns=False,
        report_to=list(training_cfg.get("report_to", [])),
        eval_strategy="steps" if raw_eval else "no",
        ddp_find_unused_parameters=(
            bool(training_cfg["ddp_find_unused_parameters"])
            if training_cfg.get("ddp_find_unused_parameters") is not None
            else None
        ),
        seed=int(training_cfg.get("seed", 42)),
        data_seed=int(training_cfg.get("data_seed", training_cfg.get("seed", 42))),
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=Dataset.from_list(train_rows),
        eval_dataset=Dataset.from_list(eval_rows) if eval_rows else None,
        data_collator=collator,
        tokenizer=tokenizer,
    )
    train_result = trainer.train()
    eval_metrics = trainer.evaluate() if raw_eval else {}
    trainer.save_model(str(output_dir))
    metrics = {
        "num_train_records": len(train_records),
        "num_eval_records": len(eval_records),
        "num_actual_train_records": len(raw_train),
        "num_actual_eval_records": len(raw_eval),
        "sft_format": "chat_completion_only",
        "smoke": smoke,
        "train_metrics": _serializable_metrics(train_result.metrics),
        "eval_metrics": _serializable_metrics(eval_metrics),
        "collator_diagnostics": collator.diagnostics(),
        "training_params": {
            "learning_rate": float(training_cfg["learning_rate"]),
            "max_steps": int(training_cfg["smoke_max_steps"] if smoke else training_cfg["max_steps"]),
            "max_length": int(training_cfg["max_length"]),
            "global_batch_per_update": (
                int(training_cfg["per_device_train_batch_size"])
                * int(training_cfg["gradient_accumulation_steps"])
            ),
            "lora": dict(training_cfg.get("lora") or {}),
        },
    }
    write_json(output_dir / "sft_metrics.json", metrics)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a base-model SFT baseline on chosen DPO completions.")
    parser.add_argument("--train-sft-file", type=str, default=str(DEFAULT_SFT_DIR / "train_sft_chosen.jsonl"))
    parser.add_argument("--eval-sft-file", type=str, default=str(DEFAULT_SFT_DIR / "eval_sft_chosen.jsonl"))
    parser.add_argument("--output-dir", type=str, default=str(REPO_ROOT / "artifacts/checkpoints/sft_compare_chosen"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--allow-student-model-env",
        action="store_true",
        help="Allow STUDENT_MODEL_PATH/model_path_env to override the configured base model.",
    )
    args = parser.parse_args()

    config = load_boundary_config()
    student_cfg = config.get("student", {})
    env_name = str(student_cfg.get("model_path_env", "STUDENT_MODEL_PATH"))
    if os.environ.get(env_name) and not args.allow_student_model_env:
        raise RuntimeError(
            f"{env_name} is set to {os.environ[env_name]!r}. Unset it before SFT base-model comparisons, "
            "or pass --allow-student-model-env if this override is intentional."
        )
    if args.max_steps is not None:
        config.setdefault("training", {})["max_steps"] = int(args.max_steps)
    if args.max_length is not None:
        config.setdefault("training", {})["max_length"] = int(args.max_length)
    if args.learning_rate is not None:
        config.setdefault("training", {})["learning_rate"] = float(args.learning_rate)
    if args.gradient_accumulation_steps is not None:
        config.setdefault("training", {})["gradient_accumulation_steps"] = int(args.gradient_accumulation_steps)
    if args.seed is not None:
        config.setdefault("training", {})["seed"] = int(args.seed)
        config.setdefault("training", {})["data_seed"] = int(args.seed)

    train_records = read_jsonl(Path(args.train_sft_file))
    eval_records = read_jsonl(Path(args.eval_sft_file))
    metrics = run_sft(
        train_records=train_records,
        eval_records=eval_records,
        config=config,
        output_dir=Path(args.output_dir),
        smoke=args.smoke,
    )
    metrics["train_sft_file"] = str(Path(args.train_sft_file).resolve())
    metrics["eval_sft_file"] = str(Path(args.eval_sft_file).resolve())
    metrics["output_dir"] = str(Path(args.output_dir).resolve())
    metrics["resolved_model_path"] = resolve_model_path(config)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
