from __future__ import annotations

import re
from typing import Any


SELF_REPAIR_PATTERN = re.compile(r"\b(wait|however|actually|let me revise|on second thought|correction)\b", re.IGNORECASE)
BRANCHING_PATTERN = re.compile(r"\b(maybe|perhaps|or|alternatively|possibly)\b", re.IGNORECASE)


def extract_process_features(
    reason: str,
    raw_text: str,
    tool_observation: dict[str, Any] | None = None,
    token_threshold: int = 80,
) -> dict[str, bool]:
    text = " ".join(part for part in [reason, raw_text, "" if tool_observation is None else str(tool_observation)] if part)
    approx_tokens = len(text.split())
    return {
        "STRUGGLE_LONG": approx_tokens >= token_threshold,
        "HAS_SELF_REPAIR": bool(SELF_REPAIR_PATTERN.search(text)),
        "LOW_LOGIT_MARGIN": "not sure" in text.lower() or "uncertain" in text.lower(),
        "HIGH_BRANCHING": len(BRANCHING_PATTERN.findall(text)) >= 2,
    }
