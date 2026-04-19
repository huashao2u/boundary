from __future__ import annotations

from mcagent_boundary.adapters.base import DatasetAdapter, StandardizedExample
from mcagent_boundary.adapters.freshqa_eval import FreshQAEvalAdapter
from mcagent_boundary.adapters.gsm8k import GSM8KAdapter
from mcagent_boundary.adapters.in3 import IN3Adapter
from mcagent_boundary.adapters.math import MathAdapter
from mcagent_boundary.adapters.mintqa import MintQAAdapter


def build_adapter_registry() -> dict[str, DatasetAdapter]:
    adapters = [
        GSM8KAdapter(),
        MathAdapter(),
        IN3Adapter(),
        MintQAAdapter(),
        FreshQAEvalAdapter(),
    ]
    return {adapter.dataset_name: adapter for adapter in adapters}
