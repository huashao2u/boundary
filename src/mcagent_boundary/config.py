from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parents[1]
ROOT_DEFAULT_CONFIG = REPO_ROOT / "configs" / "default.yaml"
ROOT_LOCAL_CONFIG = REPO_ROOT / "configs" / "default.local.yaml"
DEFAULT_CONFIG_FILES = ("paths.yaml", "models.yaml", "rollout.yaml", "dpo.yaml")


def _deep_update(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in patch.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_boundary_config(config_dir: str | Path | None = None) -> dict[str, Any]:
    directory = PACKAGE_ROOT / "configs" if config_dir is None else Path(config_dir)
    config: dict[str, Any] = {}
    if ROOT_DEFAULT_CONFIG.exists():
        with ROOT_DEFAULT_CONFIG.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        config = _deep_update(config, payload)
    if ROOT_LOCAL_CONFIG.exists():
        with ROOT_LOCAL_CONFIG.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        config = _deep_update(config, payload)
    for name in DEFAULT_CONFIG_FILES:
        path = directory / name
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
        config = _deep_update(config, payload)
    config["_config_dir"] = str(directory)
    config["_repo_root"] = str(REPO_ROOT)
    return config


def resolve_repo_path(path_value: str | Path, config: dict[str, Any] | None = None) -> Path:
    base = REPO_ROOT if config is None else Path(config.get("_repo_root", REPO_ROOT))
    path = Path(path_value)
    return path if path.is_absolute() else (base / path).resolve()
