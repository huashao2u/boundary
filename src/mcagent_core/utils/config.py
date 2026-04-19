from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml

from mcagent_core.utils.paths import PROJECT_ROOT


def _deep_update(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_update(result[key], value)
        else:
            result[key] = value
    return result


def load_config(config_path: str | Path | None = None) -> dict[str, Any]:
    config_file = PROJECT_ROOT / "configs" / "default.yaml" if config_path is None else Path(config_path)
    with config_file.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    for sibling_name in ("tools.yaml", "scoring.yaml", "teacher.yaml"):
        sibling = config_file.parent / sibling_name
        if sibling.exists() and sibling != config_file:
            with sibling.open("r", encoding="utf-8") as handle:
                patch = yaml.safe_load(handle) or {}
            config = merge_config(config, patch)
    config["_config_path"] = str(config_file)
    return config


def merge_config(base: dict[str, Any], override: dict[str, Any] | None) -> dict[str, Any]:
    if not override:
        return base
    return _deep_update(base, override)
