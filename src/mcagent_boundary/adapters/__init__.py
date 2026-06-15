from __future__ import annotations

from mcagent_boundary.adapters.base import DatasetAdapter, StandardizedExample
from mcagent_boundary.adapters.commonsenseqa import CommonsenseQAAdapter
from mcagent_boundary.adapters.freshqa_eval import FreshQAEvalAdapter
from mcagent_boundary.adapters.gsm8k import GSM8KAdapter
from mcagent_boundary.adapters.in3 import IN3Adapter
from mcagent_boundary.adapters.math import MathAdapter
from mcagent_boundary.adapters.mintqa import MintQAAdapter
from mcagent_boundary.adapters.or_bench import ORBenchAdapter
from mcagent_boundary.adapters.realtimeqa import RealTimeQAEvalAdapter
from mcagent_boundary.adapters.truthfulqa import TruthfulQAEvalAdapter


def build_adapter_registry() -> dict[str, DatasetAdapter]:
    # The two stub adapters (truthfulqa, realtimeqa) register here so code
    # imports stay stable. They raise FileNotFoundError at ``load()`` time if
    # the underlying dataset is not yet present. See CLAUDE.md §Change 6.
    adapters = [
        GSM8KAdapter(),
        MathAdapter(),
        IN3Adapter(),
        MintQAAdapter(),
        CommonsenseQAAdapter(),
        ORBenchAdapter(),
        FreshQAEvalAdapter(),
        TruthfulQAEvalAdapter(),
        RealTimeQAEvalAdapter(),
    ]
    return {adapter.dataset_name: adapter for adapter in adapters}
