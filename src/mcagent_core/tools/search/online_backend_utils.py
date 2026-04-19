from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from typing import Any


def _stringify_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (list, tuple)):
        return " ".join(_stringify_text(item) for item in value if _stringify_text(item))
    return str(value).strip()


def require_api_key(env_name: str) -> str:
    api_key = os.environ.get(env_name, "").strip()
    if not api_key:
        raise RuntimeError(f"Missing API key in environment variable `{env_name}`.")
    return api_key


def http_post_json(
    endpoint: str,
    payload: dict[str, Any],
    *,
    headers: dict[str, str] | None = None,
    timeout_seconds: float = 30.0,
) -> dict[str, Any]:
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **(headers or {})},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        return json.loads(response.read().decode("utf-8"))


def http_get_json(
    endpoint: str,
    params: dict[str, Any],
    *,
    headers: dict[str, str] | None = None,
    timeout_seconds: float = 30.0,
) -> dict[str, Any]:
    encoded = urllib.parse.urlencode({key: value for key, value in params.items() if value not in (None, "")})
    request = urllib.request.Request(
        endpoint + ("?" + encoded if encoded else ""),
        headers={"Accept": "application/json", **(headers or {})},
        method="GET",
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        return json.loads(response.read().decode("utf-8"))


def rank_to_score(index: int) -> float:
    return round(1.0 / float(index + 1), 6)


def build_online_observation(
    *,
    provider: str,
    phase: str,
    query: str,
    rows: list[dict[str, Any]],
    request_metadata: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], bool, dict[str, Any]]:
    results: list[str] = []
    doc_ids: list[str] = []
    scores: list[float] = []
    retrieved_doc_metadata: list[dict[str, Any]] = []

    for index, row in enumerate(rows):
        title = _stringify_text(row.get("title"))
        snippet = _stringify_text(row.get("snippet") or row.get("content") or row.get("text"))
        url = _stringify_text(row.get("url") or row.get("link") or row.get("id"))
        score_value = row.get("score")
        try:
            score = rank_to_score(index) if score_value in (None, "") else round(float(score_value), 6)
        except (TypeError, ValueError):
            score = rank_to_score(index)
        line = " - ".join(part for part in (title, snippet) if part)
        results.append(line or url or f"{provider} result {index + 1}")
        doc_ids.append(url or f"{provider}:{index + 1}")
        scores.append(score)
        retrieved_doc_metadata.append(
            {
                "title": title,
                "url": url or None,
                "published_date": row.get("published_date") or row.get("publishedDate"),
                "source": row.get("source"),
            }
        )

    observation = {
        "query": query,
        "results": results,
        "doc_ids": doc_ids,
        "scores": scores,
        "metadata": {
            "tool": provider,
            "mode": "real_eval_search" if phase == "eval" else "train_proxy",
            "phase": phase,
            "retrieval_type": "online_api",
            "retrieved_doc_metadata": retrieved_doc_metadata,
            **(request_metadata or {}),
        },
    }
    return observation, False, {"helpful": bool(results)}
