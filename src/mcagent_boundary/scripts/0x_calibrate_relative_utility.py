from __future__ import annotations

"""0x_calibrate_relative_utility.py — v0.2 calibration script.

Auxiliary only: this script is not part of the main artifact-building path.
It may synthesize rule-fallback teacher labels for calibration diagnostics, so
its outputs must not be mixed into strict training pair construction.

Loads held-out rollout records, computes U_rel (relative, training estimator)
and U_real (outcome-based, eval estimator), then reports calibration metrics:
  - Spearman ρ and Kendall τ between U_rel and U_real
  - Argmax hit rate (argmax(U_rel) == argmax(U_real) per example)
  - Per-action AUC (U_rel separates helpful vs unhelpful per action type)

Writes:
  {output_dir}/relative_utility_calibration.jsonl  — per-example records
  {output_dir}/summary.json                         — aggregated metrics

Calibration thresholds (from rollout.yaml calibration block):
  min_spearman:        0.5
  min_argmax_hit_rate: 0.6
  min_per_action_auc:  0.6
"""

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mcagent_boundary.annotation.teacher_label import label_boundary_records
from mcagent_boundary.config import load_boundary_config, resolve_repo_path
from mcagent_boundary.io import read_jsonl, write_json, write_jsonl
from mcagent_boundary.scoring.utility import utility_real, utility_rel


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Stats helpers (no numpy required)
# ---------------------------------------------------------------------------

def _rank_list(values: list[float]) -> list[float]:
    """Return fractional ranks for a list of floats (ascending)."""
    n = len(values)
    indexed = sorted(range(n), key=lambda i: values[i])
    ranks: list[float] = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n - 1 and values[indexed[j + 1]] == values[indexed[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1
        for k in range(i, j + 1):
            ranks[indexed[k]] = avg_rank
        i = j + 1
    return ranks


def spearman_r(x: list[float], y: list[float]) -> float:
    if len(x) < 2:
        return float("nan")
    rx = _rank_list(x)
    ry = _rank_list(y)
    n = len(x)
    d2 = sum((rx[i] - ry[i]) ** 2 for i in range(n))
    return 1.0 - 6.0 * d2 / (n * (n * n - 1))


def kendall_tau(x: list[float], y: list[float]) -> float:
    n = len(x)
    if n < 2:
        return float("nan")
    concordant = discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            if dx * dy > 0:
                concordant += 1
            elif dx * dy < 0:
                discordant += 1
    denom = n * (n - 1) / 2
    return (concordant - discordant) / denom if denom > 0 else float("nan")


def roc_auc(scores: list[float], labels: list[int]) -> float:
    """Compute ROC-AUC via Wilcoxon-Mann-Whitney statistic."""
    pos = [s for s, l in zip(scores, labels) if l == 1]
    neg = [s for s, l in zip(scores, labels) if l == 0]
    if not pos or not neg:
        return float("nan")
    concordant = sum(1 for p in pos for n in neg if p > n)
    ties = sum(0.5 for p in pos for n in neg if p == n)
    return (concordant + ties) / (len(pos) * len(neg))


# ---------------------------------------------------------------------------
# Thin example proxy (for utility functions that need .attribute access)
# ---------------------------------------------------------------------------

class _ExampleProxy:
    """Minimal proxy so utility functions can read example attributes."""
    def __init__(self, record: dict[str, Any]) -> None:
        metadata = dict(record.get("metadata") or {})
        boundary_type = str(record.get("boundary_type", ""))
        boundary_to_task = {
            "reasoning": "math",
            "factual": "factual_boundary",
            "intention": "intention_boundary",
        }
        self.question = record.get("question", "")
        self.gold_answer = record.get("gold_answer")
        self.task_type = metadata.get("task_type") or boundary_to_task.get(boundary_type, boundary_type)
        self.metadata = metadata
        self.can_clarify = bool(metadata.get("can_clarify", True))
        self.can_search = bool(metadata.get("can_search", True))


# ---------------------------------------------------------------------------
# Calibration core
# ---------------------------------------------------------------------------

def _compute_per_example(
    record: dict[str, Any],
    teacher_label: dict[str, Any] | None,
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return a list of {action, u_rel, u_real} dicts for each branch."""
    example = _ExampleProxy(record)
    semantic_tags: dict[str, bool] = dict(record.get("semantic_tags") or {})
    branches = list(record.get("branches") or [])
    if not record.get("execute_tools"):
        raise ValueError(
            f"Record {record.get('example_id')} is not an eval-style executed rollout; "
            "calibration requires real tool execution and finalize outputs."
        )
    for branch in branches:
        action = str(branch.get("action", "")).upper()
        if action != "ANSWER" and branch.get("observation") is None:
            raise ValueError(
                f"Record {record.get('example_id')} branch {action} has no observation; "
                "refusing to compute U_real on annotation rollouts."
            )
    branch_map = {b["action"]: b for b in branches}
    rows = []
    for branch in branches:
        action = str(branch.get("action", "ANSWER")).upper()
        u_rel_val = utility_rel(branch, teacher_label, semantic_tags, example, config)
        u_real_val = utility_real(branch, example, branch_map, semantic_tags, config)
        rows.append({
            "state_id": record.get("state_id", ""),
            "example_id": record.get("example_id", ""),
            "dataset": record.get("dataset", ""),
            "action": action,
            "u_rel": u_rel_val,
            "u_real": u_real_val,
        })
    return rows


def run_calibration(
    rollout_records: list[dict[str, Any]],
    teacher_labels: list[dict[str, Any]],
    config: dict[str, Any],
    output_dir: Path,
    held_out_size: int,
) -> dict[str, Any]:
    teacher_by_state = {r["state_id"]: r for r in teacher_labels}

    # Sample held-out subset
    records = list(rollout_records)
    random.shuffle(records)
    sample = records[:held_out_size]

    all_rows: list[dict[str, Any]] = []
    for record in sample:
        teacher_label = teacher_by_state.get(record.get("state_id", ""))
        rows = _compute_per_example(record, teacher_label, config)
        all_rows.extend(rows)

    # Global Spearman / Kendall
    u_rels = [r["u_rel"] for r in all_rows]
    u_reals = [r["u_real"] for r in all_rows]
    global_spearman = spearman_r(u_rels, u_reals)
    global_kendall = kendall_tau(u_rels, u_reals)

    # Argmax hit rate per example
    examples_seen: dict[str, list[dict[str, Any]]] = {}
    for row in all_rows:
        examples_seen.setdefault(row["example_id"], []).append(row)
    argmax_hits = 0
    argmax_total = 0
    for rows_for_ex in examples_seen.values():
        if len(rows_for_ex) < 2:
            continue
        best_rel = max(rows_for_ex, key=lambda r: r["u_rel"])["action"]
        best_real = max(rows_for_ex, key=lambda r: r["u_real"])["action"]
        argmax_hits += int(best_rel == best_real)
        argmax_total += 1
    argmax_hit_rate = argmax_hits / argmax_total if argmax_total > 0 else float("nan")

    # Per-action AUC (U_rel separates helpful vs unhelpful decisions)
    cal_cfg = config.get("calibration", {})
    breakdown_actions = cal_cfg.get("per_action_breakdown") or []
    per_action_auc: dict[str, float] = {}

    # helpful label: u_real > 0 => 1, else 0
    action_rows: dict[str, list[dict[str, Any]]] = {}
    for row in all_rows:
        action_rows.setdefault(row["action"], []).append(row)

    for action in breakdown_actions:
        if "_vs_" in action:
            # e.g. ANSWER_vs_non_ANSWER: 1 for ANSWER branches, 0 for rest
            pos_action = action.split("_vs_")[0]
            pos_rows = [r for r in all_rows if r["action"] == pos_action]
            neg_rows = [r for r in all_rows if r["action"] != pos_action]
            scores = [r["u_rel"] for r in pos_rows + neg_rows]
            labels = [1] * len(pos_rows) + [0] * len(neg_rows)
        else:
            rows = action_rows.get(action, [])
            if not rows:
                per_action_auc[action] = float("nan")
                continue
            scores = [r["u_rel"] for r in rows]
            labels = [1 if r["u_real"] > 0.0 else 0 for r in rows]
        per_action_auc[action] = roc_auc(scores, labels)

    # Threshold checks
    min_spearman = float(cal_cfg.get("min_spearman", 0.5))
    min_argmax = float(cal_cfg.get("min_argmax_hit_rate", 0.6))
    min_auc = float(cal_cfg.get("min_per_action_auc", 0.6))
    warnings = []
    if global_spearman < min_spearman:
        warnings.append(f"Spearman {global_spearman:.3f} < threshold {min_spearman}")
    if argmax_hit_rate < min_argmax:
        warnings.append(f"Argmax hit rate {argmax_hit_rate:.3f} < threshold {min_argmax}")
    for action, auc in per_action_auc.items():
        import math
        if not math.isnan(auc) and auc < min_auc:
            warnings.append(f"AUC[{action}] {auc:.3f} < threshold {min_auc}")

    summary = {
        "n_records_sampled": len(sample),
        "n_branch_rows": len(all_rows),
        "n_examples_for_argmax": argmax_total,
        "global_spearman": round(global_spearman, 4) if global_spearman == global_spearman else None,
        "global_kendall": round(global_kendall, 4) if global_kendall == global_kendall else None,
        "argmax_hit_rate": round(argmax_hit_rate, 4) if argmax_hit_rate == argmax_hit_rate else None,
        "per_action_auc": {k: (round(v, 4) if v == v else None) for k, v in per_action_auc.items()},
        "thresholds": {
            "min_spearman": min_spearman,
            "min_argmax_hit_rate": min_argmax,
            "min_per_action_auc": min_auc,
        },
        "warnings": warnings,
        "pass": len(warnings) == 0,
    }

    # Write outputs
    output_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_dir / "relative_utility_calibration.jsonl", all_rows)
    write_json(output_dir / "summary.json", summary)

    return summary


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate U_rel vs U_real on a held-out sample.")
    parser.add_argument(
        "--rollout-path",
        type=str,
        default=None,
        help="Path to rollout JSONL (defaults to config paths.rollout_output).",
    )
    parser.add_argument(
        "--teacher-label-path",
        type=str,
        default=None,
        help="Path to teacher labels JSONL (defaults to config paths.teacher_label_output).",
    )
    parser.add_argument(
        "--held-out-size",
        type=int,
        default=None,
        help="Number of records to use (default: calibration.sample_size from config).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: calibration.output_dir from config).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling.",
    )
    args = parser.parse_args()

    config = load_boundary_config()
    cal_cfg = config.get("calibration", {})

    rollout_path = (
        Path(args.rollout_path)
        if args.rollout_path
        else resolve_repo_path(config["paths"]["rollout_output"], config)
    )
    teacher_label_path_default = config.get("paths", {}).get("teacher_label_output")
    teacher_label_path = (
        Path(args.teacher_label_path)
        if args.teacher_label_path
        else (resolve_repo_path(teacher_label_path_default, config) if teacher_label_path_default else None)
    )
    held_out_size = args.held_out_size or int(cal_cfg.get("sample_size", 300))
    output_dir_cfg = str(cal_cfg.get("output_dir", "artifacts_v02/calibration"))
    output_dir = (
        Path(args.output_dir)
        if args.output_dir
        else resolve_repo_path(output_dir_cfg, config)
    )

    logger.info("Loading rollouts from %s", rollout_path)
    rollout_records = read_jsonl(rollout_path)
    logger.info("Loaded %d rollout records", len(rollout_records))

    teacher_labels: list[dict[str, Any]] = []
    if teacher_label_path and teacher_label_path.exists():
        logger.info("Loading teacher labels from %s", teacher_label_path)
        teacher_labels = read_jsonl(teacher_label_path)
        logger.info("Loaded %d teacher label records", len(teacher_labels))
    else:
        logger.info("No teacher labels found — rule fallback will be used for all records")
        # Generate rule-fallback labels on-the-fly for calibration
        random.seed(args.seed)
        sample_for_labels = random.sample(rollout_records, min(held_out_size, len(rollout_records)))
        teacher_labels = label_boundary_records(sample_for_labels, config)

    random.seed(args.seed)
    summary = run_calibration(rollout_records, teacher_labels, config, output_dir, held_out_size)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if not summary["pass"]:
        logger.warning("Calibration FAILED — see warnings above")
    else:
        logger.info("Calibration PASSED")


if __name__ == "__main__":
    main()
