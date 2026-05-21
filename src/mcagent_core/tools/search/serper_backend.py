from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
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
        self.fallback_api_key = str(config.get("serper_fallback_api_key") or "").strip()
        self.fallback_api_key_env = str(config.get("serper_fallback_api_key_env") or "").strip()
        self.quota_tracker_path = str(config.get("serper_quota_tracker_path") or "").strip()
        self.primary_initial_remaining = config.get("serper_quota_initial_remaining")
        self.fallback_initial_remaining = config.get("serper_fallback_quota_initial_remaining")
        self._primary_disabled = False
        self.timeout_seconds = float(config.get("online_timeout_seconds", 30.0))

    def _key_id(self, api_key: str) -> str:
        digest = hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:12]
        return f"sha256:{digest}"

    def _read_tracker(self) -> dict[str, Any]:
        if not self.quota_tracker_path:
            return {}
        path = Path(self.quota_tracker_path)
        if not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            return payload if isinstance(payload, dict) else {}
        except Exception:
            return {}

    def _write_tracker(self, payload: dict[str, Any]) -> None:
        if not self.quota_tracker_path:
            return
        path = Path(self.quota_tracker_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    def _record_key_event(
        self,
        *,
        api_key: str,
        label: str,
        success: bool,
        credits: Any = None,
        error: str | None = None,
        initial_remaining: Any = None,
    ) -> None:
        if not self.quota_tracker_path or not api_key:
            return
        now = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        payload = self._read_tracker()
        payload.setdefault("note", "Serper quota ledger. API keys are not stored here; key_id is a sha256 fingerprint.")
        payload.setdefault("created_at", now)
        payload["updated_at"] = now
        keys = payload.setdefault("keys", {})
        key_id = self._key_id(api_key)
        entry = keys.setdefault(
            key_id,
            {
                "label": label,
                "configured_initial_remaining": initial_remaining,
                "successful_calls_observed": 0,
                "failed_calls_observed": 0,
                "request_credits_total": 0,
                "estimated_remaining": initial_remaining if isinstance(initial_remaining, int) else None,
            },
        )
        entry["label"] = label
        if initial_remaining is not None and entry.get("configured_initial_remaining") is None:
            entry["configured_initial_remaining"] = initial_remaining
        entry["last_used_at"] = now
        if success:
            entry["successful_calls_observed"] = int(entry.get("successful_calls_observed") or 0) + 1
            try:
                credit_value = int(credits)
            except (TypeError, ValueError):
                credit_value = 1
            entry["request_credits_total"] = int(entry.get("request_credits_total") or 0) + max(credit_value, 0)
            initial = entry.get("configured_initial_remaining")
            if isinstance(initial, int):
                entry["estimated_remaining"] = max(0, initial - int(entry.get("request_credits_total") or 0))
            entry["last_status"] = "success"
            entry.pop("last_error", None)
        else:
            entry["failed_calls_observed"] = int(entry.get("failed_calls_observed") or 0) + 1
            entry["last_status"] = "failed"
            entry["last_error"] = str(error or "")[:500]
        payload["active_key_id"] = key_id
        self._write_tracker(payload)

    def _fallback_key(self) -> str:
        if self.fallback_api_key:
            return self.fallback_api_key
        if self.fallback_api_key_env:
            return require_api_key(self.fallback_api_key_env)
        return ""

    def _primary_key(self) -> str:
        return self.api_key or require_api_key(self.api_key_env)

    def _key_marked_unusable(self, api_key: str) -> bool:
        if not self.quota_tracker_path or not api_key:
            return False
        entry = (self._read_tracker().get("keys") or {}).get(self._key_id(api_key)) or {}
        if entry.get("estimated_remaining") == 0:
            return True
        last_error = str(entry.get("last_error") or "").lower()
        if entry.get("last_status") == "failed" and (
            "bad request" in last_error
            or "quota" in last_error
            or "credit" in last_error
            or "limit" in last_error
            or "unauthorized" in last_error
            or "forbidden" in last_error
        ):
            return True
        return False

    def _post_serper(self, payload: dict[str, Any]) -> dict[str, Any]:
        primary_key = self._primary_key()
        fallback_key = self._fallback_key()
        if self._key_marked_unusable(primary_key):
            self._primary_disabled = True
        if self._primary_disabled and fallback_key:
            response = http_post_json(
                self.endpoint,
                payload,
                headers={"X-API-KEY": fallback_key},
                timeout_seconds=self.timeout_seconds,
            )
            self._record_key_event(
                api_key=fallback_key,
                label="fallback",
                success=True,
                credits=response.get("credits"),
                initial_remaining=self.fallback_initial_remaining,
            )
            return response
        try:
            response = http_post_json(
                self.endpoint,
                payload,
                headers={"X-API-KEY": primary_key},
                timeout_seconds=self.timeout_seconds,
            )
            self._record_key_event(
                api_key=primary_key,
                label="primary",
                success=True,
                credits=response.get("credits"),
                initial_remaining=self.primary_initial_remaining,
            )
            return response
        except Exception as exc:
            self._record_key_event(
                api_key=primary_key,
                label="primary",
                success=False,
                error=repr(exc),
                initial_remaining=self.primary_initial_remaining,
            )
            if not fallback_key or fallback_key == primary_key:
                raise
            self._primary_disabled = True
            try:
                response = http_post_json(
                    self.endpoint,
                    payload,
                    headers={"X-API-KEY": fallback_key},
                    timeout_seconds=self.timeout_seconds,
                )
            except Exception as fallback_exc:
                self._record_key_event(
                    api_key=fallback_key,
                    label="fallback",
                    success=False,
                    error=repr(fallback_exc),
                    initial_remaining=self.fallback_initial_remaining,
                )
                raise
            self._record_key_event(
                api_key=fallback_key,
                label="fallback",
                success=True,
                credits=response.get("credits"),
                initial_remaining=self.fallback_initial_remaining,
            )
            return response

    def run(self, action_input: dict[str, Any], sample, history: list[dict[str, Any]]) -> tuple[dict[str, Any], bool, dict[str, Any]]:
        query = action_input.get("query") or sample.question
        payload = {
            "q": query,
            "num": self.top_k,
            "gl": str(self.config.get("country_code", "us")),
            "hl": str(self.config.get("language_code", "en")),
        }
        response = self._post_serper(payload)
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
