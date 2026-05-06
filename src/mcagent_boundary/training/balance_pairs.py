from __future__ import annotations

import math
from collections import Counter
from typing import Any


def _target_counts(total: int, quotas: dict[str, float]) -> dict[str, int]:
    raw = {key: total * float(value) for key, value in quotas.items()}
    targets = {key: int(math.floor(value)) for key, value in raw.items()}
    remainder = total - sum(targets.values())
    ranked = sorted(raw, key=lambda key: raw[key] - math.floor(raw[key]), reverse=True)
    for key in ranked[:remainder]:
        targets[key] += 1
    return targets


def balance_pairs(
    pairs: list[dict[str, Any]],
    config: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Downsample train pairs toward dataset and chosen-action quotas.

    This function does not oversample scarce classes and does not fill missing
    quota with abundant math pairs. Shortages are reported explicitly.
    """
    pair_cfg = config.get("pair_construction", {})
    balance_cfg = pair_cfg.get("balance", {})
    dataset_quota = {str(k): float(v) for k, v in (balance_cfg.get("dataset_quota") or {}).items()}
    action_quota = {str(k): float(v) for k, v in (balance_cfg.get("chosen_action_quota") or {}).items()}
    if not pairs or not dataset_quota or not action_quota:
        return list(pairs), {
            "enabled": False,
            "input_pairs": len(pairs),
            "output_pairs": len(pairs),
            "reason": "missing_pairs_or_quotas",
        }

    total = len(pairs)
    dataset_targets = _target_counts(total, dataset_quota)
    action_targets = _target_counts(total, action_quota)
    available_by_dataset = Counter(str(pair.get("dataset")) for pair in pairs)
    available_by_action = Counter(str(pair.get("chosen_action")) for pair in pairs)

    selected: list[dict[str, Any]] = []
    selected_ids: set[tuple[str, str, str]] = set()
    dataset_counts: Counter[str] = Counter()
    action_counts: Counter[str] = Counter()

    def _pair_id(pair: dict[str, Any]) -> tuple[str, str, str]:
        return (
            str(pair.get("state_id")),
            str(pair.get("chosen_action")),
            str(pair.get("rejected_action")),
        )

    ranked_pairs = sorted(
        pairs,
        key=lambda pair: (
            str(pair.get("dataset")),
            str(pair.get("chosen_action")),
            -float((pair.get("metadata") or {}).get("delta_u_rel", 0.0)),
            str(pair.get("state_id")),
        ),
    )

    for pair in ranked_pairs:
        dataset = str(pair.get("dataset"))
        action = str(pair.get("chosen_action"))
        if dataset_counts[dataset] >= dataset_targets.get(dataset, 0):
            continue
        if action_counts[action] >= action_targets.get(action, 0):
            continue
        selected.append(pair)
        selected_ids.add(_pair_id(pair))
        dataset_counts[dataset] += 1
        action_counts[action] += 1

    # Second pass: allow remaining pairs to fill their own under-target dataset
    # or action buckets, but still never exceed any configured target.
    for pair in ranked_pairs:
        pid = _pair_id(pair)
        if pid in selected_ids:
            continue
        dataset = str(pair.get("dataset"))
        action = str(pair.get("chosen_action"))
        if dataset_counts[dataset] >= dataset_targets.get(dataset, 0):
            continue
        if action_counts[action] >= action_targets.get(action, 0):
            continue
        selected.append(pair)
        selected_ids.add(pid)
        dataset_counts[dataset] += 1
        action_counts[action] += 1

    dataset_shortage = {
        key: {
            "target": target,
            "available": available_by_dataset.get(key, 0),
            "selected": dataset_counts.get(key, 0),
            "short_by": max(0, target - dataset_counts.get(key, 0)),
        }
        for key, target in dataset_targets.items()
        if dataset_counts.get(key, 0) < target
    }
    action_shortage = {
        key: {
            "target": target,
            "available": available_by_action.get(key, 0),
            "selected": action_counts.get(key, 0),
            "short_by": max(0, target - action_counts.get(key, 0)),
        }
        for key, target in action_targets.items()
        if action_counts.get(key, 0) < target
    }
    report = {
        "enabled": True,
        "input_pairs": len(pairs),
        "output_pairs": len(selected),
        "dataset_targets": dataset_targets,
        "chosen_action_targets": action_targets,
        "available_by_dataset": dict(available_by_dataset),
        "available_by_chosen_action": dict(available_by_action),
        "selected_by_dataset": dict(dataset_counts),
        "selected_by_chosen_action": dict(action_counts),
        "dataset_shortage": dataset_shortage,
        "chosen_action_shortage": action_shortage,
    }
    return selected, report
