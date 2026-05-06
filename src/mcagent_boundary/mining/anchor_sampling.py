from __future__ import annotations

from collections import Counter
from typing import Any


def _truncate(records: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    ranked = sorted(records, key=lambda item: float(item.get("utility_gap", 0.0)), reverse=True)
    return ranked[:limit]


def _threshold_for(record: dict[str, Any], thresholds: dict[str, Any], default: float) -> float:
    dataset = str(record.get("dataset", ""))
    if dataset in thresholds:
        return float(thresholds[dataset])
    return float(default)


def _ranked(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(
        records,
        key=lambda item: (
            float(item.get("utility_gap", item.get("boundary_score", 0.0)) or 0.0),
            str(item.get("state_id", "")),
        ),
        reverse=True,
    )


def _with_soft_boundary(records: list[dict[str, Any]], mining_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    soft_cfg = mining_cfg.get("soft_boundary_from_other") or {}
    if not bool(soft_cfg.get("enabled", False)):
        return []
    default_threshold = float(soft_cfg.get("threshold_default", mining_cfg.get("min_delta_u", 0.5)))
    thresholds = soft_cfg.get("threshold_by_dataset") or {}
    soft: list[dict[str, Any]] = []
    for record in records:
        if record.get("pool_reason") != "low_boundary_score":
            continue
        if not bool(record.get("has_real_action_competition", False)):
            continue
        actions = set(str(action).upper() for action in record.get("candidate_actions") or [])
        if "ANSWER" not in actions or not any(action != "ANSWER" for action in actions):
            continue
        score = float(record.get("boundary_score", record.get("utility_gap", 0.0)) or 0.0)
        if score < _threshold_for(record, thresholds, default_threshold):
            continue
        soft.append(
            {
                **record,
                "pool": "soft_boundary",
                "soft_boundary": True,
                "soft_boundary_threshold": _threshold_for(record, thresholds, default_threshold),
            }
        )
    return soft


def _quota_sample(
    records: list[dict[str, Any]],
    *,
    total_limit: int,
    quota_by_dataset: dict[str, Any] | None,
    redistribute_unused: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if total_limit <= 0:
        return [], {
            "input": len(records),
            "output": 0,
            "quota_by_dataset": dict(quota_by_dataset or {}),
            "selected_by_dataset": {},
            "available_by_dataset": {},
        }
    if not quota_by_dataset:
        sampled = _truncate(records, total_limit)
        return sampled, {
            "input": len(records),
            "output": len(sampled),
            "quota_by_dataset": {},
            "selected_by_dataset": dict(Counter(str(item.get("dataset")) for item in sampled)),
            "available_by_dataset": dict(Counter(str(item.get("dataset")) for item in records)),
        }

    buckets: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        buckets.setdefault(str(record.get("dataset", "unknown")), []).append(record)
    for dataset, dataset_records in list(buckets.items()):
        buckets[dataset] = _ranked(dataset_records)

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    selected_by_dataset: Counter[str] = Counter()
    quota = {str(key): int(value) for key, value in quota_by_dataset.items()}
    for dataset, limit in quota.items():
        for record in buckets.get(dataset, [])[: max(0, limit)]:
            selected.append(record)
            selected_ids.add(str(record.get("state_id")))
            selected_by_dataset[dataset] += 1

    if redistribute_unused and len(selected) < total_limit:
        for record in _ranked(records):
            if len(selected) >= total_limit:
                break
            state_id = str(record.get("state_id"))
            if state_id in selected_ids:
                continue
            selected.append(record)
            selected_ids.add(state_id)
            selected_by_dataset[str(record.get("dataset", "unknown"))] += 1

    if len(selected) > total_limit:
        selected = _ranked(selected)[:total_limit]
        selected_by_dataset = Counter(str(item.get("dataset", "unknown")) for item in selected)

    available_by_dataset = Counter(str(item.get("dataset", "unknown")) for item in records)
    shortages = {
        dataset: {
            "quota": target,
            "available": available_by_dataset.get(dataset, 0),
            "selected": selected_by_dataset.get(dataset, 0),
            "short_by": max(0, target - selected_by_dataset.get(dataset, 0)),
        }
        for dataset, target in quota.items()
        if selected_by_dataset.get(dataset, 0) < target
    }
    return selected, {
        "input": len(records),
        "output": len(selected),
        "quota_by_dataset": quota,
        "redistribute_unused": redistribute_unused,
        "available_by_dataset": dict(available_by_dataset),
        "selected_by_dataset": dict(selected_by_dataset),
        "shortages": shortages,
    }


def sample_anchor_pools(mined: dict[str, Any], config: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    mining_cfg = config["mining"]
    sampling_cfg = mining_cfg.get("sampling") or {}
    redistribute_unused = bool(sampling_cfg.get("redistribute_unused", False))
    boundary_records = list(mined.get("boundary_candidates", []))
    boundary_records.extend(_with_soft_boundary(list(mined.get("other", [])), mining_cfg))

    boundary, boundary_report = _quota_sample(
        boundary_records,
        total_limit=int(mining_cfg["max_boundary_candidates"]),
        quota_by_dataset=sampling_cfg.get("boundary_quota_by_dataset"),
        redistribute_unused=redistribute_unused,
    )
    clear_answer, clear_answer_report = _quota_sample(
        list(mined.get("clear_answer_anchors", [])),
        total_limit=int(mining_cfg["max_clear_answer_anchors"]),
        quota_by_dataset=sampling_cfg.get("clear_answer_quota_by_dataset"),
        redistribute_unused=redistribute_unused,
    )
    clear_external, clear_external_report = _quota_sample(
        list(mined.get("clear_external_anchors", [])),
        total_limit=int(mining_cfg["max_clear_external_anchors"]),
        quota_by_dataset=sampling_cfg.get("clear_external_quota_by_dataset"),
        redistribute_unused=redistribute_unused,
    )
    return {
        "boundary_candidates": boundary,
        "clear_answer_anchors": clear_answer,
        "clear_external_anchors": clear_external,
        "sampling_report": {
            "boundary_candidates": boundary_report,
            "clear_answer_anchors": clear_answer_report,
            "clear_external_anchors": clear_external_report,
            "soft_boundary_candidates": sum(1 for item in boundary if item.get("soft_boundary")),
        },
    }
