from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from mcagent_boundary.config import load_boundary_config, resolve_repo_path
from mcagent_boundary.io import write_json


def main() -> None:
    config = load_boundary_config()
    core_root = resolve_repo_path(config["paths"]["shared_core_root"], config)

    def _display_path(path: Path) -> str:
        try:
            return str(path.relative_to(REPO_ROOT))
        except ValueError:
            return str(path)

    files_to_check = {
        "tools": [
            core_root / "tools" / "search_tool.py",
            core_root / "tools" / "calculator_tool.py",
            core_root / "tools" / "clarify_tool.py",
            core_root / "tools" / "refuse_tool.py",
        ],
        "scoring": [
            core_root / "scoring" / "action_oracle.py",
        ],
        "policy_data": [
            core_root / "rollout" / "policy.py",
            core_root / "data" / "loaders.py",
        ],
        "evaluation": [
            core_root / "eval" / "evaluate_answers.py",
            core_root / "eval" / "calibration_utils.py",
        ],
    }
    summary = {}
    for section, paths in files_to_check.items():
        summary[section] = {
            "existing_files": [_display_path(path) for path in paths if path.exists()],
            "missing_files": [_display_path(path) for path in paths if not path.exists()],
        }

    summary["answers"] = {
        "tool_wrappers_exist": not summary["tools"]["missing_files"],
        "scoring_helpers_exist": not summary["scoring"]["missing_files"],
        "core_policy_exists": not summary["policy_data"]["missing_files"],
        "evaluation_scripts_adaptable": not summary["evaluation"]["missing_files"],
        "notes": [
            "Shared tools, policy, loaders, and eval helpers now live in the in-repo package src/mcagent_core.",
            "Boundary-specific utility, mining, teacher annotation, and training remain inside mcagent_boundary.",
            "The merged repository no longer imports mcagent/src directly during normal execution.",
        ],
    }
    output_path = resolve_repo_path(config["paths"]["legacy_report_output"], config)
    write_json(output_path, summary)
    print(json.dumps({"output": str(output_path), "summary": summary["answers"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
