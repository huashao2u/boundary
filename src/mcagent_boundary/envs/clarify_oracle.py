from __future__ import annotations

from mcagent_core.tools.clarify_tool import ClarifyTool as LegacyClarifyTool


class ClarifyOracle:
    def __init__(self):
        self._tool = LegacyClarifyTool()

    def run(self, action_input: dict, sample, history: list[dict]) -> tuple[dict, bool, dict]:
        return self._tool.run(action_input, sample, history)

