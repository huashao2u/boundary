from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mcagent_boundary.config import load_boundary_config, resolve_repo_path
from mcagent_boundary.io import read_jsonl, write_jsonl
from mcagent_boundary.training.make_dpo_pairs import build_step_dpo_pairs


def main() -> None:
    config = load_boundary_config()
    boundary_records = read_jsonl(resolve_repo_path(config["paths"]["boundary_candidates_output"], config))
    clear_answer = read_jsonl(resolve_repo_path(config["paths"]["clear_answer_output"], config))
    clear_external = read_jsonl(resolve_repo_path(config["paths"]["clear_external_output"], config))
    teacher_labels = read_jsonl(resolve_repo_path(config["paths"]["teacher_label_output"], config))
    selected_records = boundary_records + clear_answer + clear_external
    train_pairs, eval_pairs, diagnostics = build_step_dpo_pairs(selected_records, teacher_labels, config)
    train_path = resolve_repo_path(config["paths"]["train_pair_output"], config)
    eval_path = resolve_repo_path(config["paths"]["eval_pair_output"], config)
    write_jsonl(train_path, train_pairs)
    write_jsonl(eval_path, eval_pairs)
    diagnostics_path = train_path.with_name("pair_diagnostics.jsonl")
    write_jsonl(diagnostics_path, diagnostics)
    print(
        json.dumps(
            {
                "train_pairs": len(train_pairs),
                "eval_pairs": len(eval_pairs),
                "diagnostics": len(diagnostics),
                "train_output": str(train_path),
                "eval_output": str(eval_path),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
