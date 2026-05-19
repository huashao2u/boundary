from __future__ import annotations

import re
from typing import Any

from mcagent_core.tools.search.online_backend_utils import build_online_observation, http_post_json, require_api_key


_ENV_NAME_RE = re.compile(r"^[A-Z_][A-Z0-9_]*$")

class SerperBackend:
    name = "serper"

    def __init__(self, config: dict[str, Any], top_k: int = 3, phase: str = "eval"):
        self.config = config
        self.top_k = top_k
        self.phase = phase
        self.endpoint = str(config.get("serper_base_url", "https://google.serper.dev/search"))
        self.api_key_env = str(config.get("serper_api_key_env", "SERPER_API_KEY"))
        self.api_key = str(config.get("serper_api_key") or config.get("api_key") or "").strip()
        if not self.api_key and self.api_key_env and not _ENV_NAME_RE.fullmatch(self.api_key_env):
            self.api_key = self.api_key_env
            self.api_key_env = "SERPER_API_KEY"
        self.timeout_seconds = float(config.get("online_timeout_seconds", 30.0))

    def run(self, action_input: dict[str, Any], sample, history: list[dict[str, Any]]) -> tuple[dict[str, Any], bool, dict[str, Any]]:
        query = action_input.get("query") or sample.question
        payload = {
            "q": query,
            "num": self.top_k,
            "gl": str(self.config.get("country_code", "us")),
            "hl": str(self.config.get("language_code", "en")),
        }
        response = http_post_json(
            self.endpoint,
            payload,
            headers={"X-API-KEY": self.api_key or require_api_key(self.api_key_env)},
            timeout_seconds=self.timeout_seconds,
        )
        rows = []
        for item in response.get("organic", [])[: self.top_k]:
            rows.append(
                {
                    "title": item.get("title"),
                    "snippet": item.get("snippet"),
                    "url": item.get("link"),
                    "score": None,
                    "source": "serper",
                }
            )
        return build_online_observation(
            provider=self.name,
            phase=self.phase,
            query=query,
            rows=rows,
            request_metadata={"request_credits": response.get("credits"), "search_parameters": response.get("searchParameters", {})},
        )
