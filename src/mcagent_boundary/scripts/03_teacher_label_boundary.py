from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mcagent_boundary.annotation.teacher_label import label_boundary_records
from mcagent_boundary.config import load_boundary_config, resolve_repo_path
from mcagent_boundary.io import read_jsonl, write_jsonl


def main() -> None:
    config = load_boundary_config()
    boundary_records = read_jsonl(resolve_repo_path(config["paths"]["boundary_candidates_output"], config))
    clear_external = read_jsonl(resolve_repo_path(config["paths"]["clear_external_output"], config))
    records = boundary_records + clear_external
    labels = label_boundary_records(records, config)
    output_path = resolve_repo_path(config["paths"]["teacher_label_output"], config)
    write_jsonl(output_path, labels)
    print(json.dumps({"output": str(output_path), "num_labels": len(labels)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
