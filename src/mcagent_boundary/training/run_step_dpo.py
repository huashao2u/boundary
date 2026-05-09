from __future__ import annotations

from pathlib import Path

from datasets import Dataset

from mcagent_boundary.io import write_json
from mcagent_boundary.training.common import load_model_and_tokenizer


def to_message_dpo_row(pair: dict) -> dict:
    try:
        return {
            "prompt": pair["prompt_messages"],
            "chosen": pair["chosen_messages"],
            "rejected": pair["rejected_messages"],
        }
    except KeyError as exc:
        raise ValueError("Step-DPO pair is missing message DPO fields.") from exc


def run_step_dpo(
    train_pairs: list[dict],
    eval_pairs: list[dict],
    config: dict,
    output_dir: str | Path,
    smoke: bool = False,
) -> dict:
    from trl import DPOConfig, DPOTrainer

    if not train_pairs:
        raise ValueError("No Step-DPO train pairs were provided.")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_train_data = train_pairs[: min(len(train_pairs), 64)] if smoke else train_pairs
    raw_eval_data = eval_pairs[: min(len(eval_pairs), 16)] if smoke else eval_pairs
    train_data = [to_message_dpo_row(pair) for pair in raw_train_data]
    eval_data = [to_message_dpo_row(pair) for pair in raw_eval_data]
    model, tokenizer = load_model_and_tokenizer(config)
    training_cfg = config["training"]
    dpo_args = DPOConfig(
        output_dir=str(output_dir),
        per_device_train_batch_size=int(training_cfg["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(training_cfg["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(training_cfg["gradient_accumulation_steps"]),
        learning_rate=float(training_cfg["learning_rate"]),
        beta=float(training_cfg["beta"]),
        num_train_epochs=float(training_cfg["num_train_epochs"]),
        max_steps=int(training_cfg["smoke_max_steps"] if smoke else training_cfg["max_steps"]),
        warmup_steps=int(training_cfg["warmup_steps"]),
        logging_steps=int(training_cfg["logging_steps"]),
        eval_steps=int(training_cfg["eval_steps"]),
        save_steps=int(training_cfg["save_steps"]),
        max_length=int(training_cfg["max_length"]),
        remove_unused_columns=False,
        report_to=list(training_cfg.get("report_to", [])),
        do_eval=bool(eval_data),
        eval_strategy="steps" if eval_data else "no",
        bf16=bool(training_cfg.get("bf16", False)),
        fp16=bool(training_cfg.get("fp16", False)),
        gradient_checkpointing=bool(training_cfg.get("gradient_checkpointing", True)),
    )
    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_args,
        train_dataset=Dataset.from_list(train_data),
        eval_dataset=Dataset.from_list(eval_data) if eval_data else None,
        processing_class=tokenizer,
    )
    train_result = trainer.train()
    eval_metrics = trainer.evaluate() if eval_data else {}
    trainer.save_model(str(output_dir))
    metrics = {
        "num_train_pairs": len(train_pairs),
        "num_eval_pairs": len(eval_pairs),
        "dpo_format": "message",
        "smoke": smoke,
        "train_metrics": {
            key: value
            for key, value in train_result.metrics.items()
            if isinstance(value, (int, float, str, bool)) or value is None
        },
        "eval_metrics": {
            key: value
            for key, value in eval_metrics.items()
            if isinstance(value, (int, float, str, bool)) or value is None
        },
    }
    write_json(output_dir / "metrics.json", metrics)
    return metrics
