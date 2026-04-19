from __future__ import annotations

from pathlib import Path

from mcagent_core.rollout.policy import build_policy

from mcagent_boundary.adapters import build_adapter_registry
from mcagent_boundary.rollout.branch_actions import rollout_one_example


def load_standardized_examples(config: dict, dataset_names: list[str], limit_per_dataset: int | None = None) -> list:
    registry = build_adapter_registry()
    dataset_root = Path(config["paths"]["dataset_root"]).resolve()
    examples = []
    default_splits = config["datasets"]["default_splits"]
    for dataset_name in dataset_names:
        adapter = registry[dataset_name]
        examples.extend(
            adapter.load(
                dataset_root=dataset_root,
                limit=limit_per_dataset,
                split=str(default_splits[dataset_name]),
            )
        )
    return examples


def generate_rollouts(config: dict, dataset_names: list[str], phase: str, limit_per_dataset: int | None = None) -> list[dict]:
    examples = load_standardized_examples(config, dataset_names=dataset_names, limit_per_dataset=limit_per_dataset)
    policy = build_policy(
        backend=str(config["rollout"]["backend"]),
        model_path=str(Path(config["paths"]["model_root"]).resolve()),
        exploration_rate=float(config["rollout"]["exploration_rate"]),
        max_new_tokens=int(config["rollout"]["max_new_tokens"]),
    )
    return [rollout_one_example(example, config=config, phase=phase, policy=policy) for example in examples]
