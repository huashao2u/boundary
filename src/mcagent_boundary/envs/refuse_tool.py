from __future__ import annotations

from mcagent_core.tools.refuse_tool import RefuseTool as LegacyRefuseTool


class RefuseTool:
    def __init__(self):
        self._tool = LegacyRefuseTool()

    def run(self, action_input: dict, sample, history: list[dict]) -> tuple[dict, bool, dict]:
        return self._tool.run(action_input, sample, history)

