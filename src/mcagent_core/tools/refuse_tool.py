from __future__ import annotations

from typing import Any


class RefuseTool:
    name = "REFUSE"

    def run(self, action_input: dict[str, Any], sample, history: list[dict[str, Any]]) -> tuple[dict[str, Any], bool, dict[str, Any]]:
        reason = action_input.get("reason") or "The request is not answerable under the current assumptions."
        observation = {"status": "refused", "reason": reason}
        return observation, True, {"helpful": True}
