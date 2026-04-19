from __future__ import annotations

from typing import Any

from mcagent_core.tools.search.indexing import load_or_build_local_index, search_lexical_index


class LocalRetriever:
    name = "local_retriever"

    def __init__(self, config: dict[str, Any], top_k: int = 3, phase: str = "train"):
        self.config = config
        self.top_k = top_k
        self.phase = phase
        self.search_config = config.get("tools", {}).get("search", {})
        self.index_payload = load_or_build_local_index(config)

    def run(self, action_input: dict[str, Any], sample, history: list[dict[str, Any]]) -> tuple[dict[str, Any], bool, dict[str, Any]]:
        query = action_input.get("query") or sample.question
        ranked = search_lexical_index(
            self.index_payload,
            query,
            top_k=self.top_k,
            excluded_doc_ids={sample.id},
            min_score=float(self.search_config.get("min_score", 0.0)),
        )
        evidence = [item["result"] for item in ranked]
        if not ranked:
            evidence = [f"No indexed local evidence found for query: {query}"]
        observation = {
            "query": query,
            "results": evidence[: self.top_k],
            "doc_ids": [item["doc_id"] for item in ranked],
            "scores": [item["score"] for item in ranked],
            "metadata": {
                "tool": self.name,
                "mode": "train_proxy" if self.phase == "train" else "real_eval_search",
                "phase": self.phase,
                "retrieval_type": "local_index",
                "num_index_docs": self.index_payload.get("num_docs", 0),
                "retrieved_doc_metadata": [item.get("metadata", {}) for item in ranked],
            },
        }
        return observation, False, {"helpful": bool(ranked)}
