from __future__ import annotations

from mcagent_core.tools.calculator_tool import CalculatorTool as LegacyCalculatorTool


class CalculatorTool:
    def __init__(self):
        self._tool = LegacyCalculatorTool()

    def run(self, action_input: dict, sample, history: list[dict]) -> tuple[dict, bool, dict]:
        return self._tool.run(action_input, sample, history)

