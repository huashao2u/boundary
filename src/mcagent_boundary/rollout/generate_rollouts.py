from __future__ import annotations

import logging
import os
from collections import Counter
from pathlib import Path

from mcagent_core.rollout.policy import _model_assets_available, build_policy

from mcagent_boundary.adapters import build_adapter_registry
from mcagent_boundary.progress import make_progress
from mcagent_boundary.rollout.branch_actions import rollout_one_example
from mcagent_boundary.rollout.dataset_selection import apply_selection_preset, filter_by_example_ids


logger = logging.getLogger(__name__)


def _resolve_path(config: dict, path_value: str) -> Path:
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (Path(config.get("_repo_root", ".")) / path).resolve()


def _resolve_model_path(config: dict) -> Path:
    student_cfg = config.get("student", {})
    env_name = str(student_cfg.get("model_path_env", "STUDENT_MODEL_PATH"))
    if os.environ.get(env_name):
        return _resolve_path(config, os.environ[env_name])
    if student_cfg.get("fallback_model_path"):
        fallback = _resolve_path(config, str(student_cfg["fallback_model_path"]))
        if _model_assets_available(str(fallback)):
            return fallback
    return _resolve_path(config, str(config["paths"]["model_root"]))


def load_standardized_examples(
    config: dict,
    dataset_names: list[str],
    limit_per_dataset: int | None = None,
    show_progress: bool = True,
) -> list:
    registry = build_adapter_registry()
    dataset_root = _resolve_path(config, str(config["paths"]["dataset_root"]))
    examples = []
    default_splits = config["datasets"]["default_splits"]
    iterator = make_progress(
        dataset_names,
        total=len(dataset_names),
        desc="load datasets",
        unit="dataset",
        disable=not show_progress,
    )
    for dataset_name in iterator:
        adapter = registry[dataset_name]
        examples.extend(
            adapter.load(
                dataset_root=dataset_root,
                limit=limit_per_dataset,
                split=str(default_splits[dataset_name]),
            )
        )
    return examples


def _resolve_backend(requested: str, model_path: str, *, allow_heuristic_fallback: bool) -> str:
    """Resolve the rollout backend, demoting ``hf`` to ``heuristic`` if model assets
    are missing (so smoke tests still run) but leaving ``heuristic`` / ``auto`` alone.

    DEPRECATED(mainline): this missing-model demotion is a smoke-test escape
    hatch. Main experiment artifacts must use an actual student backend (hf or
    vllm) and must not rely on heuristic fallback.

    See CLAUDE.md §Change 1.
    """
    assets_available = _model_assets_available(model_path)
    if requested in {"hf", "vllm"} and not assets_available and not allow_heuristic_fallback:
        raise RuntimeError("Student model assets missing; heuristic fallback disabled.")
    if requested in {"hf", "vllm"} and not assets_available:
        logger.warning(
            "rollout.backend=%s requested but student model assets not found at %r — "
            "falling back to HeuristicPolicy for this run. This is smoke-only behavior; "
            "mainline experiment rollouts require a real student model.",
            requested,
            model_path,
        )
        return "heuristic"
    if requested == "auto" and not assets_available and not allow_heuristic_fallback:
        raise RuntimeError("Student model assets missing for rollout.backend=auto; heuristic fallback disabled.")
    return requested


def generate_rollouts(
    config: dict,
    dataset_names: list[str],
    phase: str,
    limit_per_dataset: int | None = None,
    selection_preset: str | None = None,
    fixed_example_ids: set[str] | None = None,
    show_progress: bool = True,
) -> list[dict]:
    examples = load_standardized_examples(
        config,
        dataset_names=dataset_names,
        limit_per_dataset=limit_per_dataset,
        show_progress=show_progress,
    )
    examples = apply_selection_preset(examples, selection_preset)
    examples = filter_by_example_ids(examples, fixed_example_ids or set())
    model_path = str(_resolve_model_path(config))
    backend = _resolve_backend(
        str(config["rollout"]["backend"]),
        model_path,
        allow_heuristic_fallback=bool(config.get("rollout", {}).get("allow_heuristic_fallback", False)),
    )
    rollout_cfg = config.get("rollout", {})
    policy = build_policy(
        backend=backend,
        model_path=model_path,
        exploration_rate=float(rollout_cfg.get("heuristic_exploration_rate", rollout_cfg.get("exploration_rate", 0.2))),
        max_new_tokens=int(rollout_cfg.get("max_new_tokens", 256)),
        candidate_temperature=float(rollout_cfg.get("candidate_temperature", 0.7)),
        candidate_top_p=float(rollout_cfg.get("candidate_top_p", 0.95)),
        candidate_top_k=(
            int(rollout_cfg["candidate_top_k"])
            if rollout_cfg.get("candidate_top_k") is not None
            else None
        ),
        vllm_gpu_memory_utilization=float(rollout_cfg.get("vllm_gpu_memory_utilization", 0.85)),
        vllm_max_model_len=(
            int(rollout_cfg["vllm_max_model_len"])
            if rollout_cfg.get("vllm_max_model_len") is not None
            else None
        ),
        top_k_actions=int(rollout_cfg.get("top_k_actions", 3)),
        candidate_logprob_scoring=dict(rollout_cfg.get("candidate_logprob_scoring") or {}),
    )
    rollouts: list[dict] = []
    counters: Counter[str] = Counter()
    for dataset_name in dataset_names:
        dataset_examples = [example for example in examples if example.dataset == dataset_name]
        progress = make_progress(
            dataset_examples,
            total=len(dataset_examples),
            desc=f"rollout {dataset_name}",
            unit="example",
            disable=not show_progress,
        )
        for example in progress:
            record = rollout_one_example(example, config=config, phase=phase, policy=policy)
            rollouts.append(record)
            counters["records"] += 1
            counters["valid_candidates"] += int(record.get("valid_candidate_count") or 0)
            counters["invalid_candidates"] += int(record.get("invalid_candidate_count") or 0)
            counters["empty_action_input"] += int(record.get("empty_action_input_count") or 0)
            if record.get("diagnostics"):
                counters["diagnostic_records"] += 1
            try:
                progress.set_postfix(
                    total=counters["records"],
                    valid=counters["valid_candidates"],
                    invalid=counters["invalid_candidates"],
                    empty=counters["empty_action_input"],
                )
            except Exception:
                pass
        try:
            progress.close()
        except Exception:
            pass
    return rollouts
