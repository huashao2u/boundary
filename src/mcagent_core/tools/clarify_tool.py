from __future__ import annotations

from typing import Any


class ClarifyTool:
    name = "CLARIFY"

    def run(self, action_input: dict[str, Any], sample, history: list[dict[str, Any]]) -> tuple[dict[str, Any], bool, dict[str, Any]]:
        metadata = sample.metadata or {}
        missing_details = metadata.get("missing_details") or []
        requested_question = action_input.get("question")
        selected = None
        for detail in missing_details:
            if requested_question and detail.get("inquiry") == requested_question:
                selected = detail
                break
        if selected is None and missing_details:
            selected = max(missing_details, key=lambda item: int(item.get("importance", 0)))
        user_reply = metadata.get("gold_clarify_reply") or "I prefer the default option."
        if selected and selected.get("options"):
            user_reply = selected["options"][0]
        observation = {
            "question": requested_question or metadata.get("gold_clarify_question") or "Could you clarify your intent?",
            "user_reply": user_reply,
        }
        return observation, False, {"helpful": bool(selected)}
