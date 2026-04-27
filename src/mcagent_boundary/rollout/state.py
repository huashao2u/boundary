"""Explicit representation of the boundary decision state ``s_t``.

CLAUDE.md defines the boundary state as::

    s_t = (x, h_<t, r_t, z_t)

where:

* ``x``    : the original question,
* ``h_<t`` : prior interaction / tool history up to the decision point,
* ``r_t``  : the reasoning prefix currently available to the student,
* ``z_t``  : exogenous signals — process features, semantic tags, and dataset /
             boundary metadata.

The rest of the pipeline has always carried these components implicitly inside
rollout records; this helper just makes the mapping inspectable in code so that
a reviewer can read one boundary record and understand every part of ``s_t``.

See CLAUDE.md §"State definition" → "Required implementation rule".
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from hashlib import sha1
from typing import Any


@dataclass
class BoundaryState:
    """Structured form of ``s_t`` for one decision point.

    Field names map 1:1 onto the mathematical definition:
      * ``x``      -> original question text
      * ``h_prev`` -> prior interaction / tool history (``h_<t``), possibly empty
      * ``r_t``    -> reasoning prefix (the student's current partial thought)
      * ``z_t``    -> exogenous signals, split into ``process_features``,
                      ``semantic_tags``, ``active_semantic_tags`` and
                      ``metadata`` (dataset, boundary type, example id, etc.)
    """

    x: str
    h_prev: list[dict[str, Any]] = field(default_factory=list)
    r_t: str = ""
    z_t: dict[str, Any] = field(default_factory=dict)

    def state_key_hash(self) -> str:
        """Return a stable SHA1 over the identifying components of the state.

        This preserves the exact hash the rollout pipeline has always used for
        ``state_id`` so boundary mining and teacher-label joins remain stable
        across the refactor.
        """
        active_tags = list(self.z_t.get("active_semantic_tags", []))
        example_id = self.z_t.get("metadata", {}).get("example_id", "")
        state_key = json.dumps(
            {
                "example_id": example_id,
                "question": self.x,
                "reason_prefix": self.r_t,
                "active_semantic_tags": active_tags,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        return sha1(state_key.encode("utf-8")).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        """Return a plain-dict form safe for JSONL serialization."""
        return asdict(self)


def build_state(
    example,
    reason_prefix: str,
    history: list[dict[str, Any]] | None,
    process_features: dict[str, bool],
    semantic_tags: dict[str, bool],
    active_semantic_tags: list[str],
    semantic_tag_evidence: dict[str, Any] | None = None,
) -> BoundaryState:
    """Construct a :class:`BoundaryState` from the pieces a rollout already computes.

    This does not recompute features or tags — it only packs them into a single
    explicit object so downstream consumers (mining, pair construction, eval)
    can read ``s_t`` without reassembling it by hand.
    """
    metadata = {
        "example_id": getattr(example, "example_id", None),
        "dataset": getattr(example, "dataset", None),
        "split": getattr(example, "split", None),
        "boundary_type": getattr(example, "boundary_type", None),
        "task_type": getattr(example, "task_type", None),
    }
    z_t: dict[str, Any] = {
        "process_features": dict(process_features),
        "semantic_tags": dict(semantic_tags),
        "active_semantic_tags": list(active_semantic_tags),
        "semantic_tag_evidence": dict(semantic_tag_evidence or {}),
        "metadata": metadata,
    }
    return BoundaryState(
        x=getattr(example, "question", ""),
        h_prev=list(history or []),
        r_t=reason_prefix,
        z_t=z_t,
    )
