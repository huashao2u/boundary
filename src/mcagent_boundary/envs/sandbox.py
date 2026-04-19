from __future__ import annotations

from typing import Any

from mcagent_boundary.envs.calculator_tool import CalculatorTool
from mcagent_boundary.envs.clarify_oracle import ClarifyOracle
from mcagent_boundary.envs.refuse_tool import RefuseTool
from mcagent_boundary.envs.search_tool import SearchTool


class BoundarySandbox:
    def __init__(self, example, config: dict[str, Any], phase: str):
        self.example = example
        self.phase = phase
        self.history: list[dict[str, Any]] = []
        self.tools = {
            "SEARCH": SearchTool(config=config, phase=phase),
            "CALCULATE": CalculatorTool(),
            "CLARIFY": ClarifyOracle(),
            "REFUSE": RefuseTool(),
        }

    def step(self, action_name: str, action_input: dict[str, Any]) -> tuple[dict[str, Any], bool, dict[str, Any]]:
        normalized = action_name.upper()
        if normalized == "ANSWER":
            observation = {"status": "answered", "answer": action_input.get("answer")}
            info = {"terminal": True, "helpful": True}
            self.history.append({"action": normalized, "action_input": action_input, "observation": observation})
            return observation, True, info
        tool = self.tools.get(normalized)
        if tool is None:
            raise KeyError(f"Unsupported boundary action: {action_name}")
        observation, done, info = tool.run(action_input, self.example.to_legacy_sample(), self.history)
        self.history.append({"action": normalized, "action_input": action_input, "observation": observation})
        return observation, done, info

