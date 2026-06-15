"""Aggregate planA 4-way eval + old baselines into a comparison table."""
import json
from pathlib import Path

ROOT = Path("/media/songyl/boundary/artifacts")

RUNS = [
    ("base",      ROOT / "eval_v026_ckpt500_compare_20260527/base_action_decision"),
    ("ckpt225",   ROOT / "eval_v026_early_ckpt_compare_20260527/ckpt225_action_decision"),
    ("ckpt275",   ROOT / "eval_v026_early_ckpt_compare_20260527/ckpt275_action_decision"),
    ("ckpt500",   ROOT / "eval_v026_ckpt500_compare_20260527/ckpt500_action_decision"),
    ("a0-s240",   ROOT / "eval_v026_planA_compare_20260528/a0_s240_decision"),
    ("a05-s200",  ROOT / "eval_v026_planA_compare_20260528/a05_s200_decision"),
    ("a05-s240",  ROOT / "eval_v026_planA_compare_20260528/a05_s240_decision"),
    ("a05-s300",  ROOT / "eval_v026_planA_compare_20260528/a05_s300_decision"),
]

DATASETS = ["gsm8k", "math", "in3", "mintqa", "or_bench"]


def load(d):
    p = d / "eval_dpo_raw_single_action_metrics.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


metrics = {}
for name, d in RUNS:
    m = load(d)
    if m is None:
        print(f"MISSING: {name} -> {d}")
        continue
    metrics[name] = m


def cell(m, ds, key):
    bd = m["by_dataset"].get(ds, {})
    v = bd.get(key)
    return v if isinstance(v, (int, float)) else None


def fmt(v):
    return f"{v:.3f}" if v is not None else "  -  "


print("\n" + "=" * 110)
print("planA 4-way vs baselines — final_correct_rate (decision_window, single-action eval)")
print("=" * 110)
header = f"{'run':<11} | " + " | ".join(f"{ds:>8}" for ds in DATASETS) + " |  avg"
print(header)
print("-" * len(header))
for name, _ in RUNS:
    if name not in metrics:
        continue
    row_vals = [cell(metrics[name], ds, "final_correct_rate") for ds in DATASETS]
    valid = [v for v in row_vals if v is not None]
    avg = sum(valid) / len(valid) if valid else None
    print(f"{name:<11} | " + " | ".join(f"{fmt(v):>8}" for v in row_vals) + f" | {fmt(avg)}")

print("\n" + "=" * 110)
print("non_answer_action_rate  (= rate model picked SEARCH/CALCULATE/CLARIFY/REFUSE; should be high on math/in3/mintqa)")
print("=" * 110)
print(header)
print("-" * len(header))
for name, _ in RUNS:
    if name not in metrics:
        continue
    row_vals = [cell(metrics[name], ds, "non_answer_action_rate") for ds in DATASETS]
    valid = [v for v in row_vals if v is not None]
    avg = sum(valid) / len(valid) if valid else None
    print(f"{name:<11} | " + " | ".join(f"{fmt(v):>8}" for v in row_vals) + f" | {fmt(avg)}")

print("\n" + "=" * 110)
print("tool_error_rate  (lower is better — calculator / search syntax breakage)")
print("=" * 110)
print(header)
print("-" * len(header))
for name, _ in RUNS:
    if name not in metrics:
        continue
    row_vals = [cell(metrics[name], ds, "tool_error_rate") for ds in DATASETS]
    valid = [v for v in row_vals if v is not None]
    avg = sum(valid) / len(valid) if valid else None
    print(f"{name:<11} | " + " | ".join(f"{fmt(v):>8}" for v in row_vals) + f" | {fmt(avg)}")

print("\n" + "=" * 110)
print("Action-rate breakdown per run  (ANSWER / SEARCH / CALCULATE / CLARIFY / REFUSE)")
print("=" * 110)
for name, _ in RUNS:
    if name not in metrics:
        continue
    print(f"\n--- {name} ---")
    print(f"{'ds':<10}  ANSWER  SEARCH  CALC.   CLAR.   REFUSE")
    for ds in DATASETS:
        bd = metrics[name]["by_dataset"].get(ds, {})
        rates = bd.get("action_rates", {}) or {}
        cells = [rates.get(a, 0.0) for a in ("ANSWER", "SEARCH", "CALCULATE", "CLARIFY", "REFUSE")]
        print(f"{ds:<10} " + " ".join(f"{c:6.3f}" for c in cells))


# Δ vs base (the most important view)
print("\n" + "=" * 110)
print("Δ final_correct_rate vs BASE  (positive = improvement)")
print("=" * 110)
print(header)
print("-" * len(header))
base = metrics["base"]
for name, _ in RUNS:
    if name == "base" or name not in metrics:
        continue
    row = []
    for ds in DATASETS:
        b = cell(base, ds, "final_correct_rate")
        v = cell(metrics[name], ds, "final_correct_rate")
        row.append(None if (b is None or v is None) else v - b)
    valid = [v for v in row if v is not None]
    avg = sum(valid) / len(valid) if valid else None
    cells = ["  -  " if v is None else (f"{v:+.3f}") for v in row]
    avg_s = "  -  " if avg is None else f"{avg:+.3f}"
    print(f"{name:<11} | " + " | ".join(f"{c:>8}" for c in cells) + f" | {avg_s}")
