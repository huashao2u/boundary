from __future__ import annotations

import json
from pathlib import Path

from datasets import Dataset

from mcagent_boundary.io import write_json
from mcagent_boundary.training.common import load_model_and_tokenizer


def maybe_run_warmup_sft(
    sft_records: list[dict],
    non_answer_rate: float | None,
    config: dict,
    output_dir: str | Path,
    run_training: bool = False,
    smoke: bool = False,
) -> dict:
    threshold = float(config["warmup"]["enable_if_non_answer_rate_below"])
    should_run = non_answer_rate is not None and non_answer_rate < threshold and bool(sft_records)
    summary = {
        "non_answer_rate": non_answer_rate,
        "threshold": threshold,
        "num_sft_records": len(sft_records),
        "should_run_warmup": should_run,
        "trained": False,
    }
    if not should_run or not run_training:
        return summary

    from trl import SFTConfig, SFTTrainer

    records = sft_records[: min(len(sft_records), 32)] if smoke else sft_records
    model, tokenizer = load_model_and_tokenizer(config)
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=Dataset.from_list(records),
        args=SFTConfig(
            output_dir=str(output_dir),
            per_device_train_batch_size=int(config["training"]["per_device_train_batch_size"]),
            gradient_accumulation_steps=int(config["training"]["gradient_accumulation_steps"]),
            learning_rate=float(config["training"]["learning_rate"]),
            max_steps=1 if smoke else int(config["training"]["max_steps"]),
            logging_steps=int(config["training"]["logging_steps"]),
            save_steps=int(config["training"]["save_steps"]),
            dataset_text_field="text",
            bf16=bool(config["training"].get("bf16", False)),
            fp16=bool(config["training"].get("fp16", False)),
            ddp_find_unused_parameters=(
                bool(config["training"]["ddp_find_unused_parameters"])
                if config["training"].get("ddp_find_unused_parameters") is not None
                else None
            ),
            report_to=list(config["training"].get("report_to", [])),
        ),
    )
    train_result = trainer.train()
    trainer.save_model(str(output_dir))
    summary.update(
        {
            "trained": True,
            "output_dir": str(output_dir),
            "training_metrics": {
                key: value
                for key, value in train_result.metrics.items()
                if isinstance(value, (int, float, str, bool)) or value is None
            },
        }
    )
    write_json(Path(output_dir) / "warmup_summary.json", summary)
    return summary
