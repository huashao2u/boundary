from __future__ import annotations

from typing import Any


class MockRetriever:
    name = "mock_retriever"

    def __init__(self, top_k: int = 3, phase: str = "train"):
        self.top_k = top_k
        self.phase = phase

    def run(self, action_input: dict[str, Any], sample, history: list[dict[str, Any]]) -> tuple[dict[str, Any], bool, dict[str, Any]]:
        query = action_input.get("query") or sample.question
        evidence = [
            f"Smoke-test placeholder result {index + 1} for query: {query}"
            for index in range(self.top_k)
        ]
        observation = {
            "query": query,
            "results": evidence,
            "doc_ids": [f"mock:{index}" for index in range(self.top_k)],
            "scores": [1.0 for _ in range(self.top_k)],
            "metadata": {
                "tool": self.name,
                "mode": "smoke_proxy",
                "phase": self.phase,
                "retrieval_type": "placeholder",
            },
        }
        return observation, False, {"helpful": True}
