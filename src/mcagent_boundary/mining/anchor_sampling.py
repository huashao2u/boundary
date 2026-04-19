from __future__ import annotations

from typing import Any


def _truncate(records: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    ranked = sorted(records, key=lambda item: float(item.get("utility_gap", 0.0)), reverse=True)
    return ranked[:limit]


def sample_anchor_pools(mined: dict[str, Any], config: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    mining_cfg = config["mining"]
    return {
        "boundary_candidates": _truncate(
            mined.get("boundary_candidates", []),
            int(mining_cfg["max_boundary_candidates"]),
        ),
        "clear_answer_anchors": _truncate(
            mined.get("clear_answer_anchors", []),
            int(mining_cfg["max_clear_answer_anchors"]),
        ),
        "clear_external_anchors": _truncate(
            mined.get("clear_external_anchors", []),
            int(mining_cfg["max_clear_external_anchors"]),
        ),
    }

