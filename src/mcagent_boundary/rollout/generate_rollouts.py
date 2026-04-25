from __future__ import annotations

import logging
from pathlib import Path

from mcagent_core.rollout.policy import _model_assets_available, build_policy

from mcagent_boundary.adapters import build_adapter_registry
from mcagent_boundary.rollout.branch_actions import rollout_one_example


logger = logging.getLogger(__name__)


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


def _resolve_backend(requested: str, model_path: str) -> str:
    """Resolve the rollout backend, demoting ``hf`` to ``heuristic`` if model assets
    are missing (so smoke tests still run) but leaving ``heuristic`` / ``auto`` alone.

    See CLAUDE.md §Change 1.
    """
    if requested == "hf" and not _model_assets_available(model_path):
        logger.warning(
            "rollout.backend=hf requested but student model assets not found at %r — "
            "falling back to HeuristicPolicy for this run. This is smoke-only behavior; "
            "mainline experiment rollouts require a real student model.",
            model_path,
        )
        return "heuristic"
    return requested


def generate_rollouts(config: dict, dataset_names: list[str], phase: str, limit_per_dataset: int | None = None) -> list[dict]:
    examples = load_standardized_examples(config, dataset_names=dataset_names, limit_per_dataset=limit_per_dataset)
    model_path = str(Path(config["paths"]["model_root"]).resolve())
    backend = _resolve_backend(str(config["rollout"]["backend"]), model_path)
    rollout_cfg = config.get("rollout", {})
    policy = build_policy(
        backend=backend,
        model_path=model_path,
        exploration_rate=float(rollout_cfg.get("heuristic_exploration_rate", rollout_cfg.get("exploration_rate", 0.2))),
        max_new_tokens=int(rollout_cfg.get("max_new_tokens", 256)),
        candidate_temperature=float(rollout_cfg.get("candidate_temperature", 0.7)),
        candidate_top_p=float(rollout_cfg.get("candidate_top_p", 0.95)),
    )
    return [rollout_one_example(example, config=config, phase=phase, policy=policy) for example in examples]
