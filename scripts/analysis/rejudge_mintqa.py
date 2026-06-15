"""Re-judge MintQA records that failed teacher scoring (RuntimeError, etc.).

Reads existing rollouts and existing teacher_judgments.jsonl; for any judgment
with teacher_correct=None *and* a teacher_error, re-runs only that record's
judge call. Merges fixed judgments back into the file in-place, and recomputes
the mintqa block of eval_dpo_raw_single_action_metrics.json.

Usage:
  python rejudge_mintqa.py --run-dir <run_dir> [--workers 4] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path("/media/songyl/boundary")
sys.path.insert(0, str(REPO_ROOT / "src"))

from mcagent_boundary.config import load_boundary_config
# Reuse the exact judge fn + helpers from 07b
import importlib
import importlib.util
spec = importlib.util.spec_from_file_location(
    "_mod_07b", str(REPO_ROOT / "src/mcagent_boundary/scripts/07b_eval_dpo_pairs.py")
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
from mcagent_boundary.annotation.openai_compatible_client import OpenAICompatibleChatClient


def needs_rejudge(j: dict) -> bool:
    if j.get("teacher_correct") is not None:
        return False
    # null_response skips are intentional, leave them alone
    if j.get("teacher_skipped"):
        return False
    return bool(j.get("teacher_error") or j.get("teacher_error_type"))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", required=True, help="Eval run directory containing rollouts + teacher_judgments")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    run_dir = Path(args.run_dir)
    rollouts_path = run_dir / "eval_dpo_raw_single_action_rollouts.jsonl"
    judg_path = run_dir / "mintqa_teacher_judgments.jsonl"
    metrics_path = run_dir / "eval_dpo_raw_single_action_metrics.json"
    if not rollouts_path.exists() or not judg_path.exists() or not metrics_path.exists():
        raise SystemExit(f"missing one of: {rollouts_path}, {judg_path}, {metrics_path}")

    config = load_boundary_config()
    records = [json.loads(l) for l in rollouts_path.read_text().splitlines() if l.strip()]
    judgments = [json.loads(l) for l in judg_path.read_text().splitlines() if l.strip()]

    j_by_key = {j.get("eval_key"): j for j in judgments}
    rec_by_key = {mod._record_eval_key(r): r for r in records if r.get("dataset") == "mintqa"}

    pending = []
    for j in judgments:
        if j.get("dataset") != "mintqa":
            continue
        if needs_rejudge(j):
            rec = rec_by_key.get(j.get("eval_key"))
            if rec is not None:
                pending.append((j, rec))

    print(f"[{run_dir.name}] mintqa records={len(rec_by_key)}, judgments={sum(1 for j in judgments if j.get('dataset')=='mintqa')}, pending rejudge={len(pending)}")
    if args.dry_run or not pending:
        return

    client = OpenAICompatibleChatClient(config)
    if not client.is_ready():
        raise SystemExit("teacher client not configured (POE_API_KEY?)")

    import concurrent.futures
    fixed = 0
    still_failed = 0
    def task(rec):
        try:
            return mod._teacher_judge_mintqa(client, rec), None
        except Exception as e:
            return None, e
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(task, rec): (j, rec) for (j, rec) in pending}
        for fut in concurrent.futures.as_completed(futs):
            j, rec = futs[fut]
            new_j, err = fut.result()
            if new_j is not None and new_j.get("teacher_correct") is not None:
                new_j.setdefault("eval_key", mod._record_eval_key(rec))
                new_j.setdefault("eval_pair_id", rec.get("eval_pair_id"))
                j_by_key[new_j["eval_key"]] = new_j
                fixed += 1
            else:
                still_failed += 1

    # merge back, preserve order
    new_judgments = []
    for j in judgments:
        key = j.get("eval_key")
        new_judgments.append(j_by_key.get(key, j))

    judg_path.write_text("\n".join(json.dumps(j, ensure_ascii=False) for j in new_judgments) + "\n")

    # recompute mintqa block in metrics.json
    metrics = json.loads(metrics_path.read_text())
    teacher_correct_by_key = mod._judgment_value_by_eval_key(
        [j for j in new_judgments if j.get("dataset") == "mintqa"], "teacher_correct"
    )
    subset = [r for r in records if r.get("dataset") == "mintqa"]
    correct = 0
    correct_denom = 0
    for r in subset:
        k = mod._record_eval_key(r)
        if k in teacher_correct_by_key:
            correct_denom += 1
            correct += int(teacher_correct_by_key[k])

    block = metrics["by_dataset"].get("mintqa", {})
    total = block.get("num_samples") or len(subset)
    block["final_correct_count"] = correct
    block["final_correct_denom"] = correct_denom
    block["final_correct_rate"] = (correct / correct_denom) if correct_denom > 0 else None
    block["final_correct_valid_count"] = correct
    block["final_correct_valid_denom"] = correct_denom
    block["final_correct_valid_rate"] = (correct / correct_denom) if correct_denom > 0 else None
    block["final_correct_all_count"] = correct
    block["final_correct_all_denom"] = total
    block["final_correct_all_rate"] = (correct / total) if total > 0 else None
    metrics["by_dataset"]["mintqa"] = block
    metrics_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2))

    print(f"[{run_dir.name}] fixed={fixed} still_failed={still_failed} -> mintqa rate={block['final_correct_rate']!s} ({correct}/{correct_denom})")


if __name__ == "__main__":
    main()
