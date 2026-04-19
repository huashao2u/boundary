from __future__ import annotations

from typing import Any

from mcagent_core.tools.search.online_backend_utils import build_online_observation, http_get_json, require_api_key


class BraveBackend:
    name = "brave"

    def __init__(self, config: dict[str, Any], top_k: int = 3, phase: str = "eval"):
        self.config = config
        self.top_k = top_k
        self.phase = phase
        self.endpoint = str(config.get("brave_base_url", "https://api.search.brave.com/res/v1/web/search"))
        self.api_key_env = str(config.get("brave_api_key_env", "BRAVE_SEARCH_API_KEY"))
        self.timeout_seconds = float(config.get("online_timeout_seconds", 30.0))

    def run(self, action_input: dict[str, Any], sample, history: list[dict[str, Any]]) -> tuple[dict[str, Any], bool, dict[str, Any]]:
        query = action_input.get("query") or sample.question
        response = http_get_json(
            self.endpoint,
            {
                "q": query,
                "count": self.top_k,
                "country": str(self.config.get("country_code", "us")),
                "search_lang": str(self.config.get("language_code", "en")),
            },
            headers={"X-Subscription-Token": require_api_key(self.api_key_env)},
            timeout_seconds=self.timeout_seconds,
        )
        rows = []
        for item in ((response.get("web") or {}).get("results") or [])[: self.top_k]:
            rows.append(
                {
                    "title": item.get("title"),
                    "snippet": item.get("description") or item.get("snippet"),
                    "url": item.get("url"),
                    "score": item.get("score"),
                    "source": "brave",
                }
            )
        return build_online_observation(
            provider=self.name,
            phase=self.phase,
            query=query,
            rows=rows,
            request_metadata={"query": response.get("query", {})},
        )
