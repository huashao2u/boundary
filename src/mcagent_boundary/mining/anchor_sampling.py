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


def _sampling_label(record: dict[str, Any]) -> str:
    metadata = dict(record.get("metadata") or {})
    dataset = str(record.get("dataset") or metadata.get("dataset") or metadata.get("legacy_dataset") or "")
    if dataset == "or_bench":
        label = metadata.get("or_bench_label")
        if label:
            return str(label)
        example_id = str(record.get("example_id") or "")
        prefix = "or_bench-"
        if example_id.startswith(prefix):
            return example_id[len(prefix) :].split("-", 1)[0]
    if dataset == "in3":
        vague = metadata.get("vague")
        if vague is True:
            return "vague_true"
        if vague is False:
            return "vague_false"
    return "default"


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
    quota_by_dataset_label: dict[str, Any] | None = None,
    redistribute_unused: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if total_limit <= 0:
        return [], {
            "input": len(records),
            "output": 0,
            "quota_by_dataset": dict(quota_by_dataset or {}),
            "quota_by_dataset_label": dict(quota_by_dataset_label or {}),
            "selected_by_dataset": {},
            "selected_by_dataset_label": {},
            "available_by_dataset": {},
            "available_by_dataset_label": {},
        }
    if not quota_by_dataset and not quota_by_dataset_label:
        sampled = _truncate(records, total_limit)
        return sampled, {
            "input": len(records),
            "output": len(sampled),
            "quota_by_dataset": {},
            "quota_by_dataset_label": {},
            "selected_by_dataset": dict(Counter(str(item.get("dataset")) for item in sampled)),
            "selected_by_dataset_label": dict(Counter(
                f"{str(item.get('dataset'))}:{_sampling_label(item)}" for item in sampled
            )),
            "available_by_dataset": dict(Counter(str(item.get("dataset")) for item in records)),
            "available_by_dataset_label": dict(Counter(
                f"{str(item.get('dataset'))}:{_sampling_label(item)}" for item in records
            )),
        }

    buckets: dict[str, list[dict[str, Any]]] = {}
    label_buckets: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for record in records:
        dataset = str(record.get("dataset", "unknown"))
        label = _sampling_label(record)
        buckets.setdefault(dataset, []).append(record)
        label_buckets.setdefault((dataset, label), []).append(record)
    for dataset, dataset_records in list(buckets.items()):
        buckets[dataset] = _ranked(dataset_records)
    for key, bucket_records in list(label_buckets.items()):
        label_buckets[key] = _ranked(bucket_records)

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    selected_by_dataset: Counter[str] = Counter()
    selected_by_dataset_label: Counter[str] = Counter()
    quota: dict[str, int] = {}
    quota_report: dict[str, Any] = {}
    label_quota_report: dict[str, dict[str, Any]] = {}

    def add_record(record: dict[str, Any]) -> None:
        state_id = str(record.get("state_id"))
        if state_id in selected_ids:
            return
        selected.append(record)
        selected_ids.add(state_id)
        dataset = str(record.get("dataset", "unknown"))
        selected_by_dataset[dataset] += 1
        selected_by_dataset_label[f"{dataset}:{_sampling_label(record)}"] += 1

    for dataset_key, label_quotas in (quota_by_dataset_label or {}).items():
        dataset = str(dataset_key)
        if not isinstance(label_quotas, dict):
            continue
        label_quota_report[dataset] = {}
        for label_key, value in label_quotas.items():
            label = str(label_key)
            bucket = label_buckets.get((dataset, label), [])
            if str(value).strip().lower() == "all":
                limit = len(bucket)
                label_quota_report[dataset][label] = "all"
            else:
                limit = int(value)
                label_quota_report[dataset][label] = int(value)
            for record in bucket[: max(0, limit)]:
                add_record(record)

    for key, value in (quota_by_dataset or {}).items():
        dataset = str(key)
        if str(value).strip().lower() == "all":
            quota[dataset] = len(buckets.get(dataset, []))
            quota_report[dataset] = "all"
        else:
            quota[dataset] = int(value)
            quota_report[dataset] = int(value)
    for dataset, limit in quota.items():
        if selected_by_dataset.get(dataset, 0) >= max(0, limit):
            continue
        for record in buckets.get(dataset, []):
            if selected_by_dataset.get(dataset, 0) >= max(0, limit):
                break
            add_record(record)

    if redistribute_unused and len(selected) < total_limit:
        for record in _ranked(records):
            if len(selected) >= total_limit:
                break
            add_record(record)

    if len(selected) > total_limit:
        selected = _ranked(selected)[:total_limit]
        selected_by_dataset = Counter(str(item.get("dataset", "unknown")) for item in selected)
        selected_by_dataset_label = Counter(f"{str(item.get('dataset', 'unknown'))}:{_sampling_label(item)}" for item in selected)

    available_by_dataset = Counter(str(item.get("dataset", "unknown")) for item in records)
    available_by_dataset_label = Counter(f"{str(item.get('dataset', 'unknown'))}:{_sampling_label(item)}" for item in records)
    shortages = {
        dataset: {
            "quota": quota_report.get(dataset, target),
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
        "quota_by_dataset": quota_report,
        "quota_by_dataset_label": label_quota_report,
        "redistribute_unused": redistribute_unused,
        "available_by_dataset": dict(available_by_dataset),
        "available_by_dataset_label": dict(available_by_dataset_label),
        "selected_by_dataset": dict(selected_by_dataset),
        "selected_by_dataset_label": dict(selected_by_dataset_label),
        "shortages": shortages,
    }


def sample_anchor_pools(mined: dict[str, Any], config: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    mining_cfg = config["mining"]
    sampling_cfg = mining_cfg.get("sampling") or {}
    redistribute_unused = bool(sampling_cfg.get("redistribute_unused", False))
    boundary_redistribute_unused = bool(sampling_cfg.get("boundary_redistribute_unused", redistribute_unused))
    boundary_records = list(mined.get("boundary_candidates", []))
    boundary_records.extend(_with_soft_boundary(list(mined.get("other", [])), mining_cfg))

    boundary, boundary_report = _quota_sample(
        boundary_records,
        total_limit=int(mining_cfg["max_boundary_candidates"]),
        quota_by_dataset=sampling_cfg.get("boundary_quota_by_dataset"),
        quota_by_dataset_label=sampling_cfg.get("boundary_quota_by_dataset_label"),
        redistribute_unused=boundary_redistribute_unused,
    )
    clear_answer, clear_answer_report = _quota_sample(
        list(mined.get("clear_answer_anchors", [])),
        total_limit=int(mining_cfg["max_clear_answer_anchors"]),
        quota_by_dataset=sampling_cfg.get("clear_answer_quota_by_dataset"),
        quota_by_dataset_label=sampling_cfg.get("clear_answer_quota_by_dataset_label"),
        redistribute_unused=redistribute_unused,
    )
    clear_external, clear_external_report = _quota_sample(
        list(mined.get("clear_external_anchors", [])),
        total_limit=int(mining_cfg["max_clear_external_anchors"]),
        quota_by_dataset=sampling_cfg.get("clear_external_quota_by_dataset"),
        quota_by_dataset_label=sampling_cfg.get("clear_external_quota_by_dataset_label"),
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
