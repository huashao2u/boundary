"""Aggregate planA results: use rerun mintqa metrics + original 4-dataset metrics."""
import json
from pathlib import Path

ROOT = Path("/media/songyl/boundary/artifacts")
RUN_OLD = ROOT / "eval_v026_planA_compare_20260528"
RUN_NEW = RUN_OLD / "mintqa_rerun_20260529"

BASELINES = [
    ("base",    ROOT / "eval_v026_ckpt500_compare_20260527/base_action_decision"),
    ("ckpt225", ROOT / "eval_v026_early_ckpt_compare_20260527/ckpt225_action_decision"),
    ("ckpt275", ROOT / "eval_v026_early_ckpt_compare_20260527/ckpt275_action_decision"),
    ("ckpt500", ROOT / "eval_v026_ckpt500_compare_20260527/ckpt500_action_decision"),
]
PLAN_A_TAGS = ["a0_s240", "a05_s200", "a05_s240", "a05_s300"]
DATASETS = ["gsm8k", "math", "in3", "mintqa", "or_bench"]


def load(d):
    p = d / "eval_dpo_raw_single_action_metrics.json"
    return json.loads(p.read_text()) if p.exists() else None


def cell(m, ds, key):
    bd = m["by_dataset"].get(ds, {}) if m else {}
    v = bd.get(key)
    return v if isinstance(v, (int, float)) else None


def fmt(v):
    return f"{v:.3f}" if v is not None else "  -  "


metrics = {}
for name, d in BASELINES:
    metrics[name] = load(d)

for tag in PLAN_A_TAGS:
    m_old = load(RUN_OLD / f"{tag}_decision")
    m_new = load(RUN_NEW / f"{tag}_decision")
    if m_old is None:
        continue
    if m_new is not None and "mintqa" in m_new["by_dataset"]:
        m_old["by_dataset"]["mintqa"] = m_new["by_dataset"]["mintqa"]
        m_old["mintqa_source"] = str(RUN_NEW / f"{tag}_decision")
    pretty = tag.replace("_", "-")
    metrics[pretty] = m_old

ORDER = ["base", "ckpt225", "ckpt275", "ckpt500", "a0-s240", "a05-s200", "a05-s240", "a05-s300"]

print("\n" + "=" * 110)
print("planA 4-way (mintqa rerun merged) vs baselines — final_correct_rate")
print("=" * 110)
header = f"{'run':<11} | " + " | ".join(f"{ds:>8}" for ds in DATASETS) + " |  avg"
print(header); print("-" * len(header))
for name in ORDER:
    if name not in metrics or metrics[name] is None: continue
    row = [cell(metrics[name], ds, "final_correct_rate") for ds in DATASETS]
    valid = [v for v in row if v is not None]
    avg = sum(valid)/len(valid) if valid else None
    print(f"{name:<11} | " + " | ".join(f"{fmt(v):>8}" for v in row) + f" | {fmt(avg)}")

print("\n" + "=" * 110)
print("Δ vs base (positive = improvement)")
print("=" * 110)
print(header); print("-" * len(header))
base = metrics["base"]
for name in ORDER:
    if name == "base" or name not in metrics or metrics[name] is None: continue
    row = []
    for ds in DATASETS:
        b = cell(base, ds, "final_correct_rate")
        v = cell(metrics[name], ds, "final_correct_rate")
        row.append(None if (b is None or v is None) else v - b)
    valid = [v for v in row if v is not None]
    avg = sum(valid)/len(valid) if valid else None
    cells = [f"{v:+.3f}" if v is not None else "  -  " for v in row]
    avg_s = f"{avg:+.3f}" if avg is not None else "  -  "
    print(f"{name:<11} | " + " | ".join(f"{c:>8}" for c in cells) + f" | {avg_s}")

print("\n" + "=" * 110)
print("mintqa tool_error_count  (URL/network — should be near 0 after rerun)")
print("=" * 110)
for name in ORDER:
    if name not in metrics or metrics[name] is None: continue
    bd = metrics[name]["by_dataset"].get("mintqa", {})
    tec = bd.get("tool_error_count")
    exec_ct = bd.get("executed_tool_call_count")
    print(f"  {name:<11}  tool_err={tec}  exec={exec_ct}")
