"""Look at concrete CALCULATE→CALCULATE regressions on GSM8K/MATH.
The hypothesis: the action TYPE is the same; what regressed is the action_input
contents (the math expression / final composition).
"""
import json
from pathlib import Path

ROOT = Path("/media/songyl/boundary/artifacts/eval_v026_ckpt500_compare_20260527")


def load(p):
    with open(p) as f:
        return [json.loads(l) for l in f]


def gather(ds):
    bp = ROOT / "base_action_decision" / "by_dataset" / ds / "eval_dpo_raw_single_action_rollouts.jsonl"
    cp = ROOT / "ckpt500_action_decision" / "by_dataset" / ds / "eval_dpo_raw_single_action_rollouts.jsonl"
    return {r["state_id"]: r for r in load(bp)}, {r["state_id"]: r for r in load(cp)}


def correct(rec):
    nb = rec.get("natural_branch_real") or rec.get("natural_branch") or {}
    return bool(nb.get("correctness", False)), nb


for ds in ("gsm8k", "math"):
    base, ckpt = gather(ds)
    common = sorted(base.keys() & ckpt.keys())
    print(f"\n{'=' * 90}\n{ds.upper()} — CALCULATE→CALCULATE regressions (was correct, now wrong)\n{'=' * 90}")
    samples = []
    for sid in common:
        b, c = base[sid], ckpt[sid]
        bc, bn = correct(b)
        cc, cn = correct(c)
        ba = b.get("best_action"); ca = c.get("best_action")
        if bc and not cc and ba == "CALCULATE" and ca == "CALCULATE":
            samples.append((sid, b, c, bn, cn))
    print(f"({len(samples)} cases)")
    for i, (sid, b, c, bn, cn) in enumerate(samples[:5]):
        print(f"\n--- {ds} #{i}  example_id={b['example_id']}  pair_chosen={b.get('pair_chosen_action')}/{b.get('pair_rejected_action')}")
        print(f"  Q: {b['question'][:150]}…")
        print(f"  gold: {b['gold_answer'][-150:]}")
        print(f"  base CALCULATE input: {json.dumps(b['natural_branch'].get('action_input',{}), ensure_ascii=False)[:200]}")
        print(f"  ckpt CALCULATE input: {json.dumps(c['natural_branch'].get('action_input',{}), ensure_ascii=False)[:200]}")
        print(f"  base outcome: status={bn.get('status')}  pred={str(bn.get('predicted_answer',''))[:80]}  err={bn.get('error','')[:60]}")
        print(f"  ckpt outcome: status={cn.get('status')}  pred={str(cn.get('predicted_answer',''))[:80]}  err={cn.get('error','')[:60]}")
