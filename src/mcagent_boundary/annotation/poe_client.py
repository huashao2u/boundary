from __future__ import annotations

import json
import os
import random
import time
import urllib.request
from typing import Any


class PoeChatClient:
    def __init__(self, config: dict[str, Any]):
        teacher_cfg = config["teacher"]
        api_key_env = str(teacher_cfg.get("api_key_env", "POE_API_KEY"))
        model_env = str(teacher_cfg.get("model_env", "POE_TEACHER_MODEL"))
        base_url_env = str(teacher_cfg.get("base_url_env", "POE_BASE_URL"))
        chat_url_env = str(teacher_cfg.get("chat_completions_env", "POE_CHAT_COMPLETIONS_URL"))

        self.api_key = os.environ.get(api_key_env, str(teacher_cfg.get("api_key", ""))).strip()
        self.model = os.environ.get(model_env, str(teacher_cfg.get("default_model", teacher_cfg.get("model", ""))))
        self.base_url = os.environ.get(base_url_env, str(teacher_cfg.get("default_base_url", teacher_cfg.get("base_url", ""))))
        self.chat_url = os.environ.get(
            chat_url_env,
            str(teacher_cfg.get("default_chat_completions_url", teacher_cfg.get("chat_completions_url", ""))),
        ).strip()
        self.temperature = float(teacher_cfg["temperature"])
        self.max_retries = int(teacher_cfg["max_retries"])
        self.timeout_seconds = int(teacher_cfg["timeout_seconds"])
        self.retry_backoff_seconds = float(teacher_cfg.get("retry_backoff_seconds", 2.0))
        self.retry_backoff_max_seconds = float(teacher_cfg.get("retry_backoff_max_seconds", 30.0))

    def is_ready(self) -> bool:
        return bool(self.api_key)

    def complete_json(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        if not self.is_ready():
            raise RuntimeError("Poe credentials are not configured.")
        endpoint = self.chat_url or self._chat_endpoint()
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
            "response_format": {"type": "json_object"},
        }
        last_error = None
        for attempt in range(self.max_retries):
            try:
                raw = self._post_json(endpoint, payload)
                content = (((raw.get("choices") or [{}])[0].get("message") or {}).get("content")) or "{}"
                return json.loads(content)
            except Exception as exc:
                last_error = exc
                if attempt < self.max_retries - 1:
                    delay = min(
                        self.retry_backoff_max_seconds,
                        self.retry_backoff_seconds * (2 ** attempt),
                    )
                    time.sleep(delay + random.uniform(0.0, min(1.0, delay * 0.25)))
        raise RuntimeError(f"Poe completion failed after retries: {last_error}")

    def _chat_endpoint(self) -> str:
        base_url = self.base_url.rstrip("/")
        if base_url.endswith("/chat/completions"):
            return base_url
        return base_url + "/chat/completions"

    def _post_json(self, endpoint: str, payload: dict[str, Any]) -> dict[str, Any]:
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
            return json.loads(response.read().decode("utf-8"))
