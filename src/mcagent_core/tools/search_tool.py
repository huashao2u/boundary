from __future__ import annotations

from typing import Any

from mcagent_core.tools.search.dispatcher import build_search_backend
from mcagent_core.utils.config import load_config


class SearchTool:
    name = "SEARCH"

    def __init__(self, config: dict[str, Any] | None = None, phase: str = "train"):
        runtime_config = load_config() if config is None else config
        self.backend = build_search_backend(runtime_config, phase=phase)

    def run(self, action_input: dict[str, Any], sample, history: list[dict[str, Any]]) -> tuple[dict[str, Any], bool, dict[str, Any]]:
        return self.backend.run(action_input, sample, history)
