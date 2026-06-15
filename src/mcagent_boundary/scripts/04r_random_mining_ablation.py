from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from hashlib import sha1
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mcagent_boundary.config import load_boundary_config, resolve_repo_path
from mcagent_boundary.io import read_jsonl, write_json, write_jsonl, write_jsonl_by_dataset
from mcagent_boundary.rollout.candidate_schema import required_input_value, valid_as_chosen, valid_as_rejected
from mcagent_boundary.training.make_dpo_pairs import build_step_dpo_pairs


DEFAULT_ROLLOUT_FILE = REPO_ROOT / "artifacts/rollout_v026_20260526T171000Z/all_rollouts.jsonl"
DEFAULT_REFERENCE_DIR = REPO_ROOT / "artifacts/pairs_v026_20260527T_boundary_from_rollout171000"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts/random_mining_ablation_v026_from_rollout171000_seed0"
EXCLUDED_ROLLOUT_ISSUES = {
    "invalid_candidate_output",
    "legacy_single_decision_fallback",
    "invalid_schema",
}


def _resolve_path(value: str | Path | None, config: dict[str, Any], *, default: Path | None = None) -> Path | None:
    if value is None:
        return default
    path = Path(value)
    if path.is_absolute():
        return path
    return resolve_repo_path(str(path), config)


def _dataset_of(record: dict[str, Any]) -> str:
    metadata = record.get("metadata") or {}
    return str(record.get("dataset") or metadata.get("dataset") or metadata.get("legacy_dataset") or "unknown")


def _state_id_of(record: dict[str, Any]) -> str:
    metadata = record.get("metadata") or {}
    return str(record.get("state_id") or metadata.get("state_id") or "")


def _action_pair_of(pair: dict[str, Any]) -> str:
    return f"{str(pair.get('chosen_action') or 'UNKNOWN').upper()}>{str(pair.get('rejected_action') or 'UNKNOWN').upper()}"


def _chosen_action_of(pair: dict[str, Any]) -> str:
    return str(pair.get("chosen_action") or "UNKNOWN").upper()


def _rejected_action_of(pair: dict[str, Any]) -> str:
    return str(pair.get("rejected_action") or "UNKNOWN").upper()


def _pair_kind_of(pair: dict[str, Any]) -> str:
    metadata = pair.get("metadata") or {}
    return str(pair.get("pair_kind") or metadata.get("pair_kind") or "unknown")


def _quota_key(record: dict[str, Any], fields: tuple[str, ...]) -> tuple[str, ...]:
    parts: list[str] = []
    for field in fields:
        if field == "dataset":
            parts.append(_dataset_of(record))
        elif field == "chosen_action":
            parts.append(_chosen_action_of(record))
        elif field == "rejected_action":
            parts.append(_rejected_action_of(record))
        elif field == "action_pair":
            parts.append(_action_pair_of(record))
        elif field == "pair_kind":
            parts.append(_pair_kind_of(record))
        else:
            raise ValueError(f"Unsupported quota field: {field}")
    return tuple(parts)


def _quota_counter(records: Iterable[dict[str, Any]], fields: tuple[str, ...]) -> Counter[tuple[str, ...]]:
    counts: Counter[tuple[str, ...]] = Counter()
    for record in records:
        counts[_quota_key(record, fields)] += 1
    return counts


def _largest_remainder_counts(distribution: Counter[str], total: int) -> Counter[str]:
    if total <= 0 or not distribution:
        return Counter()
    source_total = sum(distribution.values())
    if source_total <= 0:
        return Counter()
    raw = [(key, total * (value / source_total)) for key, value in sorted(distribution.items())]
    counts: Counter[str] = Counter({key: int(value) for key, value in raw})
    remainder = total - sum(counts.values())
    ranked = sorted(raw, key=lambda item: (item[1] - int(item[1]), item[0]), reverse=True)
    for key, _ in ranked[: max(0, remainder)]:
        counts[key] += 1
    return counts


def _excluded_issues(rollout: dict[str, Any]) -> list[str]:
    issues = []
    for diagnostic in rollout.get("diagnostics") or []:
        issue = str(diagnostic.get("issue", ""))
        if issue in EXCLUDED_ROLLOUT_ISSUES:
            issues.append(issue)
    return sorted(set(issues))


def _candidate_actions(candidates: Iterable[dict[str, Any]]) -> list[str]:
    actions: list[str] = []
    for candidate in candidates:
        action = str(candidate.get("action") or "").upper()
        if action and action not in actions:
            actions.append(action)
    return actions


def _valid_random_candidates(rollout: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        candidate
        for candidate in list(rollout.get("candidates") or [])
        if valid_as_rejected(candidate)
        and not bool(candidate.get("is_debug_fallback", False))
        and bool(candidate.get("is_student_candidate", True))
    ]


def _random_record(
    rollout: dict[str, Any],
    *,
    min_valid_candidates: int,
    require_action_diversity: bool,
) -> dict[str, Any] | None:
    if _excluded_issues(rollout):
        return None
    candidates = _valid_random_candidates(rollout)
    if len(candidates) < min_valid_candidates:
        return None
    if not any(valid_as_chosen(candidate) and required_input_value(candidate) for candidate in candidates):
        return None
    actions = _candidate_actions(candidates)
    if require_action_diversity and len(actions) < 2:
        return None
    return {
        **rollout,
        "candidates": candidates,
        "pool": "random_candidate",
        "pool_reason": "random_mining_ablation",
        "mining_mode": "random",
        "random_sample": True,
        "candidate_actions": actions,
        "num_valid_candidates": len(candidates),
        "boundary_score": None,
        "boundary_threshold": None,
        "utility_gap": None,
        "boundary_reasons": ["random_mining_ablation"],
        "has_real_action_competition": len(actions) >= 2,
    }


def _eligible_random_records(
    rollouts: list[dict[str, Any]],
    *,
    min_valid_candidates: int,
    require_action_diversity: bool,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    eligible: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    seen: set[str] = set()
    for rollout in rollouts:
        state_id = _state_id_of(rollout)
        if not state_id:
            skipped["missing_state_id"] += 1
            continue
        if state_id in seen:
            skipped["duplicate_state_id"] += 1
            continue
        seen.add(state_id)
        record = _random_record(
            rollout,
            min_valid_candidates=min_valid_candidates,
            require_action_diversity=require_action_diversity,
        )
        if record is None:
            skipped["ineligible_candidates_or_diagnostics"] += 1
            continue
        eligible.append(record)
    return eligible, skipped


def _target_counts_from_reference(records: list[dict[str, Any]], fields: tuple[str, ...]) -> Counter[tuple[str, ...]]:
    counts = _quota_counter(records, fields)
    if not counts:
        raise ValueError("Reference pairs are empty; cannot derive final quotas.")
    return counts


def _sample_records(
    eligible: list[dict[str, Any]],
    *,
    target_total: int,
    seed: int,
    target_by_dataset: Counter[str] | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rng = random.Random(seed)
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in eligible:
        buckets[_dataset_of(record)].append(record)
    for records in buckets.values():
        rng.shuffle(records)

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    shortages: dict[str, dict[str, int]] = {}
    if target_by_dataset:
        for dataset, target in sorted(target_by_dataset.items()):
            bucket = buckets.get(dataset, [])
            take = min(int(target), len(bucket))
            selected.extend(bucket[:take])
            selected_ids.update(_state_id_of(record) for record in bucket[:take])
            if take < int(target):
                shortages[dataset] = {"target": int(target), "available": len(bucket), "selected": take}

    if len(selected) < target_total:
        leftovers = [record for record in eligible if _state_id_of(record) not in selected_ids]
        rng.shuffle(leftovers)
        selected.extend(leftovers[: max(0, target_total - len(selected))])

    rng.shuffle(selected)
    report = {
        "target_total": target_total,
        "selected_total": len(selected),
        "target_by_dataset": dict(target_by_dataset or {}),
        "selected_by_dataset": dict(Counter(_dataset_of(record) for record in selected)),
        "available_by_dataset": dict(Counter(_dataset_of(record) for record in eligible)),
        "shortages": shortages,
    }
    return selected, report


def _stable_pair_sort_key(pair: dict[str, Any], seed: int) -> str:
    pair_id = str(pair.get("pair_id") or "")
    state_id = _state_id_of(pair)
    return sha1(f"{seed}:{pair_id}:{state_id}:{_action_pair_of(pair)}:{_pair_kind_of(pair)}".encode("utf-8")).hexdigest()


def _match_pairs_to_reference(
    pairs: list[dict[str, Any]],
    reference_pairs: list[dict[str, Any]],
    *,
    fields: tuple[str, ...],
    seed: int,
    fill_shortages: bool,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    target_counts = _target_counts_from_reference(reference_pairs, fields)
    target_total = sum(target_counts.values())
    buckets: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for pair in pairs:
        buckets[_quota_key(pair, fields)].append(pair)
    for key in list(buckets):
        buckets[key].sort(key=lambda pair: _stable_pair_sort_key(pair, seed))

    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()
    shortages: dict[str, dict[str, Any]] = {}
    for key, target in sorted(target_counts.items()):
        bucket = buckets.get(key, [])
        take = min(target, len(bucket))
        selected.extend(bucket[:take])
        selected_ids.update(str(pair.get("pair_id") or id(pair)) for pair in bucket[:take])
        if take < target:
            shortages["|".join(key)] = {"target": target, "available": len(bucket), "selected": take}

    if fill_shortages and len(selected) < target_total:
        leftovers = [
            pair
            for pair in sorted(pairs, key=lambda item: _stable_pair_sort_key(item, seed + 17))
            if str(pair.get("pair_id") or id(pair)) not in selected_ids
        ]
        selected.extend(leftovers[: target_total - len(selected)])

    selected.sort(key=lambda pair: _stable_pair_sort_key(pair, seed + 31))
    output_counts = _quota_counter(selected, fields)
    report = {
        "target_total": target_total,
        "available_total": len(pairs),
        "selected_total": len(selected),
        "match_fields": list(fields),
        "target_counts": {"|".join(key): value for key, value in sorted(target_counts.items())},
        "selected_counts": {"|".join(key): value for key, value in sorted(output_counts.items())},
        "shortages": shortages,
        "filled_shortages_globally": bool(fill_shortages),
        "exact_total_match": len(selected) == target_total,
        "exact_quota_match": not shortages,
    }
    return selected, report


def _teacher_labels_for_records(teacher_labels: list[dict[str, Any]], records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    wanted = {_state_id_of(record) for record in records}
    return [label for label in teacher_labels if _state_id_of(label) in wanted]


def _write_pool_outputs(output_dir: Path, records: list[dict[str, Any]], summary: dict[str, Any]) -> dict[str, Any]:
    mining_path = output_dir / "mined_random.json"
    boundary_path = output_dir / "boundary_candidates.jsonl"
    clear_answer_path = output_dir / "clear_answer_anchors.jsonl"
    clear_external_path = output_dir / "clear_external_anchors.jsonl"
    write_json(mining_path, summary)
    write_jsonl(boundary_path, records)
    write_jsonl(clear_answer_path, [])
    write_jsonl(clear_external_path, [])
    return {
        "mining_output": str(mining_path),
        "boundary_output": str(boundary_path),
        "clear_answer_output": str(clear_answer_path),
        "clear_external_output": str(clear_external_path),
        "boundary_by_dataset": write_jsonl_by_dataset(boundary_path, records),
    }


def _write_pair_outputs(
    output_dir: Path,
    train_pairs: list[dict[str, Any]],
    fixed_eval_pairs: list[dict[str, Any]],
    diagnostics: list[dict[str, Any]],
) -> dict[str, Any]:
    train_path = output_dir / "train_step_dpo_pairs.jsonl"
    eval_path = output_dir / "eval_step_dpo_pairs.jsonl"
    diagnostics_path = output_dir / "pair_diagnostics.jsonl"
    write_jsonl(train_path, train_pairs)
    write_jsonl(eval_path, fixed_eval_pairs)
    write_jsonl(diagnostics_path, diagnostics)
    return {
        "train_output": str(train_path),
        "eval_output": str(eval_path),
        "diagnostics_output": str(diagnostics_path),
        "train_by_dataset": write_jsonl_by_dataset(train_path, train_pairs),
        "eval_by_dataset": write_jsonl_by_dataset(eval_path, fixed_eval_pairs),
        "diagnostics_by_dataset": write_jsonl_by_dataset(diagnostics_path, diagnostics),
    }


def _parse_match_fields(value: str) -> tuple[str, ...]:
    aliases = {
        "dataset": ("dataset",),
        "chosen_action": ("chosen_action",),
        "dataset_chosen_action": ("dataset", "chosen_action"),
        "dataset_action_pair": ("dataset", "action_pair"),
        "dataset_pair_kind": ("dataset", "pair_kind"),
        "dataset_action_pair_kind": ("dataset", "action_pair", "pair_kind"),
    }
    if value in aliases:
        return aliases[value]
    fields = tuple(part.strip() for part in value.split(",") if part.strip())
    if not fields:
        raise ValueError("--match-fields cannot be empty")
    for field in fields:
        if field not in {"dataset", "chosen_action", "rejected_action", "action_pair", "pair_kind"}:
            raise ValueError(f"Unsupported --match-fields item: {field}")
    return fields


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build a random-mining ablation from an existing rollout without changing the main "
            "boundary mining or pair construction scripts."
        )
    )
    parser.add_argument("--rollout-file", type=str, default=str(DEFAULT_ROLLOUT_FILE))
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--reference-train-pairs", type=str, default=str(DEFAULT_REFERENCE_DIR / "train_step_dpo_pairs.jsonl"))
    parser.add_argument("--reference-eval-pairs", type=str, default=str(DEFAULT_REFERENCE_DIR / "eval_step_dpo_pairs.jsonl"))
    parser.add_argument(
        "--teacher-label-file",
        type=str,
        default=None,
        help=(
            "Teacher labels for random records. If omitted, the script only writes the random "
            "candidate pool. For a clean random-mining ablation, generate labels for this pool "
            "and rerun with that label file."
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--random-total-records", type=int, default=None)
    parser.add_argument("--candidate-multiplier", type=float, default=2.2)
    parser.add_argument("--random-min-valid-candidates", type=int, default=2)
    parser.add_argument("--random-allow-single-action", action="store_true")
    parser.add_argument(
        "--match-fields",
        default="dataset_action_pair",
        help=(
            "Final train quota matching fields. Defaults to dataset_action_pair to preserve "
            "the mainline-like per-dataset action-pair balance. Supported aliases: dataset, "
            "chosen_action, dataset_chosen_action, dataset_action_pair, dataset_pair_kind, "
            "dataset_action_pair_kind, or a comma-separated subset of "
            "dataset/chosen_action/rejected_action/action_pair/pair_kind."
        ),
    )
    parser.add_argument(
        "--fill-shortages",
        action="store_true",
        help="Keep the exact total by filling quota shortages from other available random pairs.",
    )
    parser.add_argument(
        "--allow-shortage",
        action="store_true",
        help="Write outputs even if final random pairs cannot match reference totals.",
    )
    parser.add_argument("--require-teacher-source", action="append", default=[], metavar="SOURCE")
    parser.add_argument("--require-configured-teacher-source", action="store_true")
    parser.add_argument("--prompt-mode", choices=["decision_window", "end_to_end"], default=None)
    parser.add_argument("--no-progress", action="store_true")
    args = parser.parse_args()

    config = load_boundary_config()
    if args.prompt_mode is not None:
        config.setdefault("pair_construction", {})["prompt_mode"] = args.prompt_mode

    output_dir = _resolve_path(args.output_dir, config)
    assert output_dir is not None
    output_dir.mkdir(parents=True, exist_ok=True)
    rollout_file = _resolve_path(args.rollout_file, config)
    reference_train_file = _resolve_path(args.reference_train_pairs, config)
    reference_eval_file = _resolve_path(args.reference_eval_pairs, config)
    teacher_label_file = _resolve_path(args.teacher_label_file, config, default=None)
    assert rollout_file is not None and reference_train_file is not None and reference_eval_file is not None

    reference_train = read_jsonl(reference_train_file)
    fixed_reference_eval = read_jsonl(reference_eval_file)
    reference_total = len(reference_train)
    train_dataset_targets = Counter(_dataset_of(pair) for pair in reference_train)
    if args.random_total_records is not None:
        random_total = int(args.random_total_records)
    else:
        random_total = max(reference_total, int(round(reference_total * max(1.0, float(args.candidate_multiplier)))))
    random_dataset_targets = _largest_remainder_counts(train_dataset_targets, random_total)

    rollouts = read_jsonl(rollout_file)
    eligible, skipped = _eligible_random_records(
        rollouts,
        min_valid_candidates=max(1, int(args.random_min_valid_candidates)),
        require_action_diversity=not args.random_allow_single_action,
    )
    sampled_records, sampling_report = _sample_records(
        eligible,
        target_total=random_total,
        seed=int(args.seed),
        target_by_dataset=random_dataset_targets,
    )
    summary: dict[str, Any] = {
        "mode": "random_mining_ablation",
        "rollout_file": str(rollout_file),
        "output_dir": str(output_dir),
        "reference_train_pairs": str(reference_train_file),
        "fixed_reference_eval_pairs": str(reference_eval_file),
        "reference_train_count": len(reference_train),
        "fixed_reference_eval_count": len(fixed_reference_eval),
        "reference_train_dataset_distribution": dict(train_dataset_targets),
        "seed": int(args.seed),
        "random_total_records_requested": random_total,
        "random_min_valid_candidates": max(1, int(args.random_min_valid_candidates)),
        "random_require_action_diversity": not args.random_allow_single_action,
        "num_rollouts": len(rollouts),
        "num_eligible_random_records": len(eligible),
        "skipped_random_records": dict(skipped),
        "sampling_report": sampling_report,
        "design_note": (
            "Final comparability is enforced after teacher-label/u_rel pair construction, "
            "not at the mining layer. Random mining over-samples states; final random train "
            "pairs are downsampled to the reference train quotas. Eval pairs are not resampled: "
            "the output eval file is copied from the fixed reference eval set."
        ),
    }
    pool_outputs = _write_pool_outputs(output_dir, sampled_records, summary)
    summary.update(pool_outputs)

    labels_for_pairs: list[dict[str, Any]] = []
    if teacher_label_file is not None:
        labels_for_pairs = read_jsonl(teacher_label_file)

    if not labels_for_pairs:
        summary["pair_construction"] = {
            "status": "skipped_missing_teacher_labels",
            "next_step": (
                "Run teacher labeling on boundary_candidates.jsonl in this output directory, "
                "then rerun this script with --teacher-label-file <random teacher_labels.jsonl>."
            ),
        }
        write_json(output_dir / "random_ablation_summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return

    selected_labels = _teacher_labels_for_records(labels_for_pairs, sampled_records)
    configured_teacher_source = str(config.get("teacher", {}).get("source_label", "llm_teacher"))
    required_teacher_sources = set(args.require_teacher_source or [])
    if args.require_configured_teacher_source:
        required_teacher_sources.add(configured_teacher_source)

    train_pairs_raw, heldout_pairs_raw, diagnostics = build_step_dpo_pairs(
        sampled_records,
        selected_labels,
        config,
        show_progress=not args.no_progress,
        require_teacher_label=True,
        required_teacher_sources=required_teacher_sources,
    )
    train_candidate_pairs_raw = train_pairs_raw + heldout_pairs_raw
    match_fields = _parse_match_fields(args.match_fields)
    train_pairs, train_match_report = _match_pairs_to_reference(
        train_candidate_pairs_raw,
        reference_train,
        fields=match_fields,
        seed=int(args.seed),
        fill_shortages=bool(args.fill_shortages),
    )
    pair_outputs = _write_pair_outputs(output_dir, train_pairs, fixed_reference_eval, diagnostics)
    exact_train = train_match_report["exact_total_match"]
    if not args.allow_shortage and not exact_train:
        summary["pair_construction"] = {
            "status": "failed_shortage",
            "raw_train_pairs": len(train_pairs_raw),
            "raw_heldout_pairs_folded_into_train_candidates": len(heldout_pairs_raw),
            "raw_train_candidate_pairs": len(train_candidate_pairs_raw),
            "fixed_eval_pairs": len(fixed_reference_eval),
            "train_match_report": train_match_report,
            "diagnostics_count": len(diagnostics),
            **pair_outputs,
        }
        write_json(output_dir / "random_ablation_summary.json", summary)
        raise SystemExit(
            "Random pairs did not match reference totals. Increase --candidate-multiplier, "
            "label more random records, or use --fill-shortages/--allow-shortage for diagnostics."
        )

    summary["pair_construction"] = {
        "status": "ok",
        "teacher_label_file": str(teacher_label_file),
        "teacher_labels_loaded": len(labels_for_pairs),
        "teacher_labels_used": len(selected_labels),
        "required_teacher_sources": sorted(required_teacher_sources),
        "raw_train_pairs": len(train_pairs_raw),
        "raw_heldout_pairs_folded_into_train_candidates": len(heldout_pairs_raw),
        "raw_train_candidate_pairs": len(train_candidate_pairs_raw),
        "final_train_pairs": len(train_pairs),
        "fixed_eval_pairs": len(fixed_reference_eval),
        "eval_mode": "fixed_reference_eval_copied",
        "diagnostics_count": len(diagnostics),
        "train_match_report": train_match_report,
        **pair_outputs,
    }
    write_json(output_dir / "random_ablation_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
