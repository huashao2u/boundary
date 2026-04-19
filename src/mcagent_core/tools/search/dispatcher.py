from __future__ import annotations

from typing import Any

from mcagent_core.tools.search.brave_backend import BraveBackend
from mcagent_core.tools.search.exa_backend import ExaBackend
from mcagent_core.tools.search.local_retriever import LocalRetriever
from mcagent_core.tools.search.mock_retriever import MockRetriever
from mcagent_core.tools.search.serper_backend import SerperBackend
from mcagent_core.tools.search.tavily_backend import TavilyBackend


ONLINE_BACKENDS = {
    "serper": SerperBackend,
    "brave": BraveBackend,
    "tavily": TavilyBackend,
    "exa": ExaBackend,
}


def _resolve_search_config(runtime_config: dict[str, Any] | None) -> dict[str, Any]:
    if runtime_config is None:
        return {}
    if "tools" in runtime_config:
        return runtime_config.get("tools", {}).get("search", {})
    return runtime_config


def build_search_backend(runtime_config: dict[str, Any] | None = None, *, phase: str = "train"):
    cfg = _resolve_search_config(runtime_config)
    backend_name = str(
        cfg.get("train_backend" if phase == "train" else "eval_backend")
        or cfg.get("backend", "mock_retriever")
    )
    top_k = int(cfg.get("top_k", 3))
    if backend_name == "mock_retriever":
        return MockRetriever(top_k=top_k, phase=phase)
    if backend_name == "local_retriever":
        root_config = runtime_config if runtime_config and "tools" in runtime_config else {"tools": {"search": cfg}}
        return LocalRetriever(config=root_config, top_k=top_k, phase=phase)
    if backend_name in ONLINE_BACKENDS:
        if not bool(cfg.get("enable_online_backend", False)):
            raise RuntimeError(f"Online backend `{backend_name}` is disabled by config.")
        return ONLINE_BACKENDS[backend_name](config=cfg, top_k=top_k, phase=phase)
    raise KeyError(f"Unsupported search backend: {backend_name}")
