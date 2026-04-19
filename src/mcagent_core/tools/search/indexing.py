from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

from mcagent_core.tools.search.corpus_builder import build_search_corpus
from mcagent_core.utils.io import ensure_parent, write_json


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9_]+")


def tokenize_text(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_PATTERN.finditer(text or "")]


def build_lexical_index(corpus: list[dict[str, Any]]) -> dict[str, Any]:
    tokenized_docs: list[list[str]] = []
    doc_freq: Counter[str] = Counter()
    doc_lengths: list[int] = []
    for doc in corpus:
        tokens = tokenize_text(doc.get("text", ""))
        tokenized_docs.append(tokens)
        doc_lengths.append(len(tokens))
        doc_freq.update(set(tokens))
    avg_doc_len = sum(doc_lengths) / len(doc_lengths) if doc_lengths else 0.0
    return {
        "docs": corpus,
        "tokenized_docs": tokenized_docs,
        "doc_freq": dict(doc_freq),
        "doc_lengths": doc_lengths,
        "avg_doc_len": avg_doc_len,
        "num_docs": len(corpus),
    }


def _bm25_score(query_tokens: list[str], doc_tokens: list[str], doc_freq: dict[str, int], num_docs: int, avg_doc_len: float, *, k1: float = 1.5, b: float = 0.75) -> float:
    if not query_tokens or not doc_tokens or num_docs == 0:
        return 0.0
    tf = Counter(doc_tokens)
    score = 0.0
    doc_len = max(len(doc_tokens), 1)
    for token in query_tokens:
        freq = tf.get(token, 0)
        if freq == 0:
            continue
        df = doc_freq.get(token, 0)
        idf = math.log(1 + (num_docs - df + 0.5) / (df + 0.5))
        norm = freq + k1 * (1 - b + b * (doc_len / max(avg_doc_len, 1.0)))
        score += idf * ((freq * (k1 + 1)) / max(norm, 1e-8))
    return score


def search_lexical_index(index_payload: dict[str, Any], query: str, *, top_k: int = 3, excluded_doc_ids: set[str] | None = None, min_score: float = 0.0) -> list[dict[str, Any]]:
    query_tokens = tokenize_text(query)
    excluded = excluded_doc_ids or set()
    scored: list[dict[str, Any]] = []
    for doc, doc_tokens in zip(index_payload.get("docs", []), index_payload.get("tokenized_docs", [])):
        if doc.get("id") in excluded:
            continue
        score = _bm25_score(
            query_tokens,
            doc_tokens,
            index_payload.get("doc_freq", {}),
            int(index_payload.get("num_docs", 0)),
            float(index_payload.get("avg_doc_len", 0.0)),
        )
        if score <= min_score:
            continue
        scored.append(
            {
                "doc_id": doc.get("id"),
                "score": round(score, 6),
                "result": doc.get("snippet") or doc.get("text", "")[:240],
                "metadata": doc.get("metadata", {}),
            }
        )
    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:top_k]


def load_or_build_local_index(config: dict[str, Any]) -> dict[str, Any]:
    search_cfg = config.get("tools", {}).get("search", {})
    corpus_path = Path(search_cfg.get("corpus_path") or config["paths"]["search_corpus_output"])
    index_path = Path(search_cfg.get("index_path") or config["paths"]["search_index_output"])
    if index_path.exists():
        return json.loads(index_path.read_text(encoding="utf-8"))
    corpus = build_search_corpus(config, split_name=str(search_cfg.get("index_split", "all")))
    ensure_parent(corpus_path)
    ensure_parent(index_path)
    write_json(corpus_path, {"num_docs": len(corpus), "docs": corpus})
    index_payload = build_lexical_index(corpus)
    write_json(index_path, index_payload)
    return index_payload
