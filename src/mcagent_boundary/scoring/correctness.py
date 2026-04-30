from __future__ import annotations

import re
from typing import Any

from mcagent_core.eval.evaluate_answers import is_answer_correct


EXPRESSION_PATTERN = re.compile(r"\d+(?:\s*[\+\-\*\/]\s*\d+)+")


def needs_calculation(example, semantic_tags: dict[str, bool] | None = None, reason_text: str | None = None) -> bool:
    """Return True only when calculation is genuinely needed (v0.2 §Problem C).

    Conditions (any one sufficient):
    1. The question contains an explicit arithmetic expression.
    2. semantic_tags contains CALCULATION_REQUIRED.
    3. reasoning text explicitly mentions computation.

    Does NOT return True just because boundary_type == "reasoning".
    """
    if EXPRESSION_PATTERN.search(example.question):
        return True
    if semantic_tags and semantic_tags.get("CALCULATION_REQUIRED"):
        return True
    if reason_text:
        lower = reason_text.lower()
        compute_keywords = ("calculat", "comput", "arithmetic", "expression", "formula", "equation", "math")
        if any(kw in lower for kw in compute_keywords):
            return True
    return False


def evaluate_branch_correctness(example, final_answer: str | None, action: str, semantic_tags: dict[str, bool]) -> bool | None:
    metadata = dict(getattr(example, "metadata", {}) or {})
    if metadata.get("task_type") == "refusal_boundary":
        should_refuse = bool(metadata.get("should_refuse", False))
        if action == "REFUSE":
            return should_refuse
        if action == "ANSWER":
            return not should_refuse and bool(final_answer)
        return None
    if action == "CLARIFY":
        return True if semantic_tags.get("MISSING_INFO") and final_answer else False
    if action == "REFUSE":
        return None
    if example.gold_answer is None:
        if action == "ANSWER":
            return not semantic_tags.get("MISSING_INFO", False)
        return None
    sample = {
        "gold_answer": example.gold_answer,
        "task_type": example.task_type,
    }
    return is_answer_correct(sample, final_answer)
