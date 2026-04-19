from __future__ import annotations

import os
from pathlib import Path

from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


def resolve_model_path(config: dict) -> str:
    student_cfg = config["student"]
    env_name = str(student_cfg["model_path_env"])
    return os.environ.get(env_name, str(config["paths"]["model_root"]))


def load_model_and_tokenizer(config: dict):
    model_path = str(Path(resolve_model_path(config)).resolve())
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = "auto"
    if bool(config["training"].get("bf16", False)):
        import torch

        dtype = torch.bfloat16
    elif bool(config["training"].get("fp16", False)):
        import torch

        dtype = torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        dtype=dtype,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    lora_cfg = config["training"]["lora"]
    if bool(lora_cfg.get("enabled", False)):
        peft_config = LoraConfig(
            r=int(lora_cfg["r"]),
            lora_alpha=int(lora_cfg["alpha"]),
            lora_dropout=float(lora_cfg["dropout"]),
            target_modules=list(lora_cfg["target_modules"]),
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, peft_config)
    return model, tokenizer
