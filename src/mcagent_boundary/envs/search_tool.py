from __future__ import annotations

from mcagent_core.tools.search_tool import SearchTool as LegacySearchTool


class SearchTool:
    def __init__(self, config: dict, phase: str):
        self._tool = LegacySearchTool(config=config, phase=phase)

    def run(self, action_input: dict, sample, history: list[dict]) -> tuple[dict, bool, dict]:
        return self._tool.run(action_input, sample, history)

