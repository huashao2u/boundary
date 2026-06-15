from __future__ import annotations

from pathlib import Path

from datasets import Dataset

from mcagent_boundary.io import write_json
from mcagent_boundary.training.common import load_model_and_tokenizer


def _pair_sample_weight(pair: dict) -> float:
    weight = pair.get("sample_weight")
    if weight is None:
        weight = pair.get("metadata", {}).get("sample_weight", 1.0)
    try:
        weight = float(weight)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid DPO sample_weight: {weight!r}") from exc
    if weight < 0:
        raise ValueError(f"DPO sample_weight must be non-negative: {weight!r}")
    return weight


def to_message_dpo_row(pair: dict) -> dict:
    try:
        return {
            "prompt": pair["prompt_messages"],
            "chosen": pair["chosen_messages"],
            "rejected": pair["rejected_messages"],
            "sample_weight": _pair_sample_weight(pair),
        }
    except KeyError as exc:
        raise ValueError("Step-DPO pair is missing message DPO fields.") from exc


def _weighted_sequence_loss_mean(per_sequence_loss, sample_weight):
    if sample_weight is None:
        return per_sequence_loss.mean()
    import torch

    weights = torch.as_tensor(sample_weight, device=per_sequence_loss.device, dtype=per_sequence_loss.dtype).view(-1)
    if weights.numel() != per_sequence_loss.numel():
        raise ValueError(
            f"sample_weight size {weights.numel()} does not match DPO batch size {per_sequence_loss.numel()}."
        )
    weights = weights.clamp_min(0)
    weight_sum = weights.sum()
    if weight_sum.item() <= 0:
        return per_sequence_loss.mean()
    normalized_weights = weights * (weights.numel() / weight_sum.clamp_min(1e-12))
    return (per_sequence_loss.view(-1) * normalized_weights).mean()


def _make_weighted_dpo_trainer(base_cls):
    class WeightedDPOTrainer(base_cls):
        def _compute_loss(self, model, inputs, return_outputs):
            sample_weight = inputs.get("sample_weight")
            if sample_weight is None:
                clean_inputs = {key: value for key, value in inputs.items() if key != "sample_weight"}
                return super()._compute_loss(model, clean_inputs, return_outputs)

            unsupported = []
            if list(getattr(self, "loss_types", ["sigmoid"])) != ["sigmoid"]:
                unsupported.append(f"loss_types={getattr(self, 'loss_types', None)!r}")
            if getattr(self, "ld_alpha", None) is not None:
                unsupported.append("ld_alpha")
            if getattr(self, "f_divergence_type", "reverse_kl") != "reverse_kl":
                unsupported.append(f"f_divergence_type={getattr(self, 'f_divergence_type', None)!r}")
            if getattr(self, "use_weighting", False):
                unsupported.append("trl_use_weighting")
            if unsupported:
                raise ValueError(
                    "sample_weight currently supports the standard sigmoid reverse-KL DPO loss only; "
                    f"unsupported settings: {', '.join(unsupported)}"
                )

            from trl.trainer import dpo_trainer as trl_dpo

            torch = trl_dpo.torch
            F = trl_dpo.F
            mode = "train" if self.model.training else "eval"
            device = self.accelerator.device

            non_model_keys = {"completion_mask", "ref_chosen_logps", "ref_rejected_logps", "sample_weight"}
            model_kwargs = {key: value for key, value in inputs.items() if key not in non_model_keys}
            model_kwargs["use_cache"] = False
            outputs = model(**model_kwargs)

            input_ids = inputs["input_ids"]
            completion_mask = inputs["completion_mask"]
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = input_ids[..., 1:].contiguous()
            shift_completion_mask = completion_mask[..., 1:].contiguous()
            per_token_logps = trl_dpo.selective_log_softmax(shift_logits, shift_labels)
            per_token_logps[shift_completion_mask == 0] = 0.0
            logps = per_token_logps.sum(dim=1)
            chosen_logps, rejected_logps = logps.chunk(2, dim=0)

            if self.precompute_ref_logps:
                ref_chosen_logps, ref_rejected_logps = inputs["ref_chosen_logps"], inputs["ref_rejected_logps"]
            else:
                with torch.no_grad(), trl_dpo.disable_gradient_checkpointing(
                    self.model, self.args.gradient_checkpointing_kwargs
                ):
                    if trl_dpo.is_peft_model(model) and self.ref_model is None:
                        unwrapped_model = self.accelerator.unwrap_model(model)
                        adapter_name = "ref" if "ref" in unwrapped_model.peft_config else None
                        with trl_dpo.use_adapter(unwrapped_model, adapter_name=adapter_name):
                            ref_outputs = self.model(**model_kwargs)
                    else:
                        ref_outputs = self.ref_model(**model_kwargs)

                ref_shift_logits = ref_outputs.logits[..., :-1, :].contiguous()
                ref_per_token_logps = trl_dpo.selective_log_softmax(ref_shift_logits, shift_labels)
                ref_per_token_logps[shift_completion_mask == 0] = 0.0
                ref_logps = ref_per_token_logps.sum(dim=1)
                ref_chosen_logps, ref_rejected_logps = ref_logps.chunk(2, dim=0)

            chosen_logratios = chosen_logps - ref_chosen_logps
            rejected_logratios = rejected_logps - ref_rejected_logps
            delta_score = chosen_logratios - rejected_logratios
            per_sequence_loss = -F.logsigmoid(self.beta * delta_score)

            rpo_alpha = float(getattr(self, "rpo_alpha", 0.0) or 0.0)
            chosen_completion_mask = shift_completion_mask[: shift_completion_mask.shape[0] // 2]
            chosen_token_counts = chosen_completion_mask.sum(dim=1).clamp_min(1).to(chosen_logps.dtype)
            chosen_nll_per_seq = -chosen_logps / chosen_token_counts
            if rpo_alpha > 0:
                per_sequence_loss = per_sequence_loss + rpo_alpha * chosen_nll_per_seq
            self._metrics[mode]["nll/chosen"].append(
                self.accelerator.gather(chosen_nll_per_seq.detach()).mean().item()
            )

            loss_weight = float(getattr(self, "loss_weights", [1.0])[0])
            loss = _weighted_sequence_loss_mean(per_sequence_loss, sample_weight) * loss_weight

            per_token_entropy = trl_dpo.entropy_from_logits(shift_logits.detach())
            entropy = per_token_entropy[shift_completion_mask.bool()].mean()
            entropy = self.accelerator.gather_for_metrics(entropy).mean().item()
            self._metrics[mode]["entropy"].append(entropy)

            if mode == "train":
                num_tokens_in_batch = self.accelerator.gather_for_metrics(inputs["attention_mask"].sum()).sum().item()
                self._total_train_tokens += num_tokens_in_batch
            self._metrics[mode]["num_tokens"] = [self._total_train_tokens]

            chosen_logits, rejected_logits = shift_logits.detach().chunk(2, dim=0)
            chosen_mask, rejected_mask = shift_completion_mask.chunk(2, dim=0)
            total_chosen_logits = chosen_logits[chosen_mask.bool()].mean(-1).sum()
            total_chosen_tokens = chosen_mask.sum()
            total_rejected_logits = rejected_logits[rejected_mask.bool()].mean(-1).sum()
            total_rejected_tokens = rejected_mask.sum()
            total_chosen_logits = self.accelerator.gather_for_metrics(total_chosen_logits).sum().item()
            total_chosen_tokens = self.accelerator.gather_for_metrics(total_chosen_tokens).sum().item()
            total_rejected_logits = self.accelerator.gather_for_metrics(total_rejected_logits).sum().item()
            total_rejected_tokens = self.accelerator.gather_for_metrics(total_rejected_tokens).sum().item()
            self._metrics[mode]["logits/chosen"].append(
                total_chosen_logits / total_chosen_tokens if total_chosen_tokens > 0 else 0.0
            )
            self._metrics[mode]["logits/rejected"].append(
                total_rejected_logits / total_rejected_tokens if total_rejected_tokens > 0 else 0.0
            )

            predictions = chosen_logits.argmax(dim=-1)
            chosen_mask_bool = shift_completion_mask[: len(shift_completion_mask) // 2].bool()
            chosen_labels = shift_labels[: len(shift_labels) // 2]
            correct_predictions = (predictions == chosen_labels) & chosen_mask_bool
            correct_tokens = self.accelerator.gather_for_metrics(correct_predictions.sum())
            total_tokens = self.accelerator.gather_for_metrics(chosen_mask_bool.sum())
            total_sum = total_tokens.sum()
            accuracy = (correct_tokens.sum() / total_sum).item() if total_sum > 0 else 0.0
            self._metrics[mode]["mean_token_accuracy"].append(accuracy)

            chosen_rewards = self.beta * chosen_logratios.detach()
            rejected_rewards = self.beta * rejected_logratios.detach()
            agg_chosen_rewards = self.accelerator.gather(chosen_rewards)
            agg_rejected_rewards = self.accelerator.gather(rejected_rewards)
            self._metrics[mode]["rewards/chosen"].append(agg_chosen_rewards.mean().item())
            self._metrics[mode]["rewards/rejected"].append(agg_rejected_rewards.mean().item())

            reward_accuracies = (chosen_rewards > rejected_rewards).float()
            self._metrics[mode]["rewards/accuracies"].append(
                self.accelerator.gather(reward_accuracies).mean().item()
            )
            margins = chosen_rewards - rejected_rewards
            self._metrics[mode]["rewards/margins"].append(self.accelerator.gather(margins).mean().item())
            self._metrics[mode]["logps/chosen"].append(self.accelerator.gather(chosen_logps).mean().item())
            self._metrics[mode]["logps/rejected"].append(self.accelerator.gather(rejected_logps).mean().item())

            gathered_weights = self.accelerator.gather_for_metrics(
                torch.as_tensor(sample_weight, device=device, dtype=torch.float32).view(-1)
            )
            self._metrics[mode]["sample_weight/mean"].append(gathered_weights.mean().item())

            return (loss, outputs) if return_outputs else loss

    return WeightedDPOTrainer


def _make_weighted_preference_collator(dpo_args, tokenizer):
    from trl.trainer.dpo_trainer import DataCollatorForPreference
    import torch

    class WeightedPreferenceCollator(DataCollatorForPreference):
        def torch_call(self, examples):
            output = super().torch_call(examples)
            if examples and "sample_weight" in examples[0]:
                output["sample_weight"] = torch.tensor(
                    [float(example.get("sample_weight", 1.0)) for example in examples],
                    dtype=torch.float32,
                )
            return output

    pad_token = dpo_args.pad_token or tokenizer.pad_token or tokenizer.eos_token
    if pad_token not in tokenizer.get_vocab():
        raise ValueError(f"The configured pad token is not in the tokenizer vocabulary: {pad_token!r}")
    tokenizer.pad_token = pad_token
    return WeightedPreferenceCollator(
        pad_token_id=tokenizer.pad_token_id,
        max_length=dpo_args.max_length,
        truncation_mode=dpo_args.truncation_mode,
        pad_to_multiple_of=dpo_args.pad_to_multiple_of,
    )


def _make_min_save_step_callback(min_save_steps: int):
    from transformers import TrainerCallback

    class MinSaveStepCallback(TrainerCallback):
        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step < min_save_steps:
                control.should_save = False
            return control

        def on_save(self, args, state, control, **kwargs):
            if state.global_step < min_save_steps:
                control.should_save = False
            return control

    return MinSaveStepCallback()


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
    training_cfg = config["training"]
    use_sample_weights = bool(training_cfg.get("use_sample_weights", True))
    train_data = [to_message_dpo_row(pair) for pair in raw_train_data]
    eval_data = [to_message_dpo_row(pair) for pair in raw_eval_data]
    if not use_sample_weights:
        for row in train_data + eval_data:
            row.pop("sample_weight", None)
    model, tokenizer = load_model_and_tokenizer(config)
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
        save_total_limit=(
            int(training_cfg["save_total_limit"]) if training_cfg.get("save_total_limit") is not None else None
        ),
        max_length=int(training_cfg["max_length"]),
        seed=int(training_cfg.get("seed", 42)),
        truncation_mode=str(training_cfg.get("truncation_mode", "keep_end")),
        remove_unused_columns=False,
        report_to=list(training_cfg.get("report_to", [])),
        do_eval=bool(eval_data),
        eval_strategy="steps" if eval_data else "no",
        bf16=bool(training_cfg.get("bf16", False)),
        fp16=bool(training_cfg.get("fp16", False)),
        gradient_checkpointing=bool(training_cfg.get("gradient_checkpointing", True)),
        precompute_ref_log_probs=bool(training_cfg.get("precompute_ref_log_probs", False)),
        precompute_ref_batch_size=(
            int(training_cfg["precompute_ref_batch_size"])
            if training_cfg.get("precompute_ref_batch_size") is not None
            else None
        ),
        ddp_find_unused_parameters=(
            bool(training_cfg["ddp_find_unused_parameters"])
            if training_cfg.get("ddp_find_unused_parameters") is not None
            else None
        ),
    )
    trainer_cls = _make_weighted_dpo_trainer(DPOTrainer) if use_sample_weights else DPOTrainer
    data_collator = _make_weighted_preference_collator(dpo_args, tokenizer) if use_sample_weights else None
    callbacks = []
    min_save_steps = int(training_cfg.get("min_save_steps", 0) or 0)
    if min_save_steps > 0 and not smoke:
        callbacks.append(_make_min_save_step_callback(min_save_steps))
    trainer = trainer_cls(
        model=model,
        ref_model=None,
        args=dpo_args,
        callbacks=callbacks,
        data_collator=data_collator,
        train_dataset=Dataset.from_list(train_data),
        eval_dataset=Dataset.from_list(eval_data) if eval_data else None,
        processing_class=tokenizer,
    )
    rpo_alpha = float(training_cfg.get("rpo_alpha", 0.0) or 0.0)
    if rpo_alpha < 0:
        raise ValueError(f"rpo_alpha must be >= 0, got {rpo_alpha}")
    trainer.rpo_alpha = rpo_alpha
    train_result = trainer.train()
    eval_metrics = trainer.evaluate() if eval_data else {}
    trainer.save_model(str(output_dir))
    metrics = {
        "num_train_pairs": len(train_pairs),
        "num_eval_pairs": len(eval_pairs),
        "dpo_format": "message",
        "use_sample_weights": use_sample_weights,
        "rpo_alpha": rpo_alpha,
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
