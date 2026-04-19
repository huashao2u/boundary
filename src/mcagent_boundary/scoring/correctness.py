from __future__ import annotations

import re
from typing import Any

from mcagent_core.eval.evaluate_answers import is_answer_correct


EXPRESSION_PATTERN = re.compile(r"\d+(?:\s*[\+\-\*\/]\s*\d+)+")


def needs_calculation(example) -> bool:
    if example.boundary_type == "reasoning":
        return True
    return bool(EXPRESSION_PATTERN.search(example.question))


def evaluate_branch_correctness(example, final_answer: str | None, action: str, semantic_tags: dict[str, bool]) -> bool | None:
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

