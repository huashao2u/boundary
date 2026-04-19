from __future__ import annotations

from typing import Any, Protocol


class SearchBackend(Protocol):
    name: str

    def run(self, action_input: dict[str, Any], sample, history: list[dict[str, Any]]) -> tuple[dict[str, Any], bool, dict[str, Any]]:
        ...
