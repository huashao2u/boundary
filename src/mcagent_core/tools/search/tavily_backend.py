from __future__ import annotations

from typing import Any

from mcagent_core.tools.search.online_backend_utils import build_online_observation, http_post_json, require_api_key


class TavilyBackend:
    name = "tavily"

    def __init__(self, config: dict[str, Any], top_k: int = 3, phase: str = "eval"):
        self.config = config
        self.top_k = top_k
        self.phase = phase
        self.endpoint = str(config.get("tavily_base_url", "https://api.tavily.com/search"))
        self.api_key_env = str(config.get("tavily_api_key_env", "TAVILY_API_KEY"))
        self.timeout_seconds = float(config.get("online_timeout_seconds", 30.0))

    def run(self, action_input: dict[str, Any], sample, history: list[dict[str, Any]]) -> tuple[dict[str, Any], bool, dict[str, Any]]:
        query = action_input.get("query") or sample.question
        response = http_post_json(
            self.endpoint,
            {
                "query": query,
                "search_depth": str(self.config.get("tavily_search_depth", "basic")),
                "max_results": self.top_k,
                "topic": str(self.config.get("tavily_topic", "general")),
                "include_answer": False,
                "include_raw_content": False,
            },
            headers={"Authorization": f"Bearer {require_api_key(self.api_key_env)}"},
            timeout_seconds=self.timeout_seconds,
        )
        rows = []
        for item in response.get("results", [])[: self.top_k]:
            rows.append(
                {
                    "title": item.get("title"),
                    "content": item.get("content"),
                    "url": item.get("url"),
                    "score": item.get("score"),
                    "source": "tavily",
                }
            )
        return build_online_observation(
            provider=self.name,
            phase=self.phase,
            query=query,
            rows=rows,
            request_metadata={"request_id": response.get("request_id"), "response_time": response.get("response_time")},
        )
