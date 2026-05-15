from __future__ import annotations

from typing import Any

from mcagent_core.tools.calculator_tool import CalculatorTool as LegacyCalculatorTool


class CalculatorTool:
    def __init__(self, config: dict[str, Any] | None = None, phase: str = "train"):
        config = config or {}
        tools_cfg = config.get("tools", {}) or {}
        calculator_cfg = tools_cfg.get("calculator", {}) or {}
        default_backend = "python_sandbox" if phase in {"eval", "test"} else "safe_eval"
        backend = str(
            calculator_cfg.get(
                f"backend_{phase}",
                tools_cfg.get(f"calculator_backend_{phase}", calculator_cfg.get("backend", tools_cfg.get("calculator_backend", default_backend))),
            )
        )
        self._tool = LegacyCalculatorTool(
            backend=backend,
            timeout_sec=float(calculator_cfg.get("timeout_sec", tools_cfg.get("calculator_timeout_sec", 3.0))),
            memory_mb=int(calculator_cfg.get("memory_mb", tools_cfg.get("calculator_memory_mb", 512))),
            output_max_chars=int(calculator_cfg.get("output_max_chars", tools_cfg.get("calculator_output_max_chars", 512))),
        )

    def run(self, action_input: dict, sample, history: list[dict]) -> tuple[dict, bool, dict]:
        return self._tool.run(action_input, sample, history)
