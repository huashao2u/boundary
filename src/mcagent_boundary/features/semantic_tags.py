from __future__ import annotations

from typing import Any

from mcagent_core.features.semantic_tags import (
    build_semantic_tag_details_from_state,
    build_semantic_tags_from_state,
)


def infer_semantic_tag_details(
    example,
    reason_prefix: str = "",
    history: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Boundary-side compatibility wrapper around the core tagger."""
    return build_semantic_tag_details_from_state(
        question=getattr(example, "question", ""),
        metadata=getattr(example, "metadata", {}) or {},
        task_type=getattr(example, "task_type", None),
        dataset=getattr(example, "dataset", None),
        reason_prefix=reason_prefix,
        history_prefix=history,
        can_search=getattr(example, "can_search", None),
        can_calculate=getattr(example, "can_calculate", None),
        can_clarify=getattr(example, "can_clarify", None),
    )


def infer_semantic_tags(
    example,
    reason_prefix: str = "",
    history: list[dict[str, Any]] | None = None,
) -> dict[str, bool]:
    return build_semantic_tags_from_state(
        question=getattr(example, "question", ""),
        metadata=getattr(example, "metadata", {}) or {},
        task_type=getattr(example, "task_type", None),
        dataset=getattr(example, "dataset", None),
        reason_prefix=reason_prefix,
        history_prefix=history,
        can_search=getattr(example, "can_search", None),
        can_calculate=getattr(example, "can_calculate", None),
        can_clarify=getattr(example, "can_clarify", None),
    )


def active_semantic_tags(
    example,
    reason_prefix: str = "",
    history: list[dict[str, Any]] | None = None,
) -> list[str]:
    return infer_semantic_tag_details(
        example,
        reason_prefix=reason_prefix,
        history=history,
    )["active_semantic_tags"]
