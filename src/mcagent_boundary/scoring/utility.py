from __future__ import annotations

"""utility.py — v0.2 split utility functions.

Training annotation:  utility_rel(branch, teacher_label, semantic_tags, example, cfg) -> float
Eval:                 utility_real(branch, example, branch_map, semantic_tags, cfg) -> float

Process features MUST NOT appear in utility_rel (§5a, §5d).
Gold is only used for offline score computation in utility_rel (ANSWER correctness, CLARIFY slot),
never injected into student action_input.
"""

import ast
import logging
import re
from typing import Any

from mcagent_core.eval.evaluate_answers import is_answer_correct
from mcagent_boundary.scoring.helpfulness import (
    outcome_helpfulness,
    teacher_helpfulness,
)


logger = logging.getLogger(__name__)
_EXPRESSION_RE = re.compile(r"(\d+(?:\s*[\+\-\*\/\^\(\)]\s*\d+)+)")
_NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\b")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _expression_validity(expression: str) -> float:
    """Return 1.0 if the expression is parseable, 0.0 otherwise."""
    if not expression or not isinstance(expression, str):
        return 0.0
    expr = expression.strip()
    if _EXPRESSION_RE.search(expr):
        try:
            ast.parse(expr, mode="eval")
            return 1.0
        except SyntaxError:
            pass
        return 0.8  # regex matched but ast parse failed — partial credit
    return 0.0


def _goal_alignment(expression: str, question: str) -> float:
    """Return jaccard overlap of numbers/variables in expression vs question."""
    if not expression or not question:
        return 0.0
    q_numbers = set(_NUMBER_RE.findall(question))
    e_numbers = set(_NUMBER_RE.findall(expression))
    if not q_numbers and not e_numbers:
        return 0.5  # neutral
    if not q_numbers or not e_numbers:
        return 0.0
    intersection = q_numbers & e_numbers
    union = q_numbers | e_numbers
    return len(intersection) / len(union) if union else 0.0


def _slot_hit_rate(clarify_question: str, question: str, gold_slots: list[str] | None) -> float:
    """Return keyword jaccard between clarify question and gold slots / question."""
    if not clarify_question:
        return 0.0
    cq_words = set(clarify_question.lower().split())
    if gold_slots:
        slot_words: set[str] = set()
        for slot in gold_slots:
            slot_words.update(slot.lower().split())
        if not slot_words:
            return 0.0
        intersection = cq_words & slot_words
        union = cq_words | slot_words
        return len(intersection) / len(union) if union else 0.0
    # Fall back to question word overlap.
    q_words = set(question.lower().split())
    if not q_words:
        return 0.0
    intersection = cq_words & q_words
    union = cq_words | q_words
    return len(intersection) / len(union) if union else 0.0


def _auto_answer_score(branch: dict[str, Any], example) -> float | None:
    """Return 1.0/0.0 correctness when gold_answer is available; else None."""
    gold = getattr(example, "gold_answer", None)
    if gold is None:
        return None
    final_answer = branch.get("action_input", {}).get("answer") or branch.get("final_answer")
    sample = {"gold_answer": gold, "task_type": getattr(example, "task_type", "")}
    try:
        correct = is_answer_correct(sample, final_answer)
        return 1.0 if correct else 0.0
    except Exception:
        return None


def _auto_calculate_score(branch: dict[str, Any], example) -> float | None:
    """Return expression_validity × goal_alignment or None if unavailable."""
    expression = branch.get("action_input", {}).get("expression") or ""
    if not expression:
        return None
    validity = _expression_validity(expression)
    alignment = _goal_alignment(expression, getattr(example, "question", ""))
    return validity * alignment


def _auto_clarify_score(branch: dict[str, Any], example) -> float | None:
    """Return slot_hit_rate or None if no gold slot info available."""
    clarify_question = branch.get("action_input", {}).get("question") or ""
    if not clarify_question:
        return None
    metadata = dict(getattr(example, "metadata", {}) or {})
    gold_slots: list[str] | None = None
    if "gold_clarify_question" in metadata and metadata["gold_clarify_question"]:
        gold_slots = [str(metadata["gold_clarify_question"])]
    elif "missing_details" in metadata and metadata["missing_details"]:
        gold_slots = [
            str(item.get("slot") or item.get("field") or "")
            for item in (metadata["missing_details"] or [])
            if isinstance(item, dict)
        ]
    if gold_slots:
        return _slot_hit_rate(clarify_question, getattr(example, "question", ""), gold_slots)
    return None  # no gold slot info; caller falls back to teacher


def _semantic_bonus(action: str, semantic_tags: dict[str, bool], cfg: dict[str, Any]) -> float:
    """Compute semantic_bonus from semantic tags only (never process features).

    §5a: each ε ≤ 0.05; total per candidate capped at 0.15.
    Process feature keys are explicitly forbidden.
    """
    bonus_cfg = cfg.get("scoring", {}).get("semantic_bonus", {})
    cap = float(bonus_cfg.get("cap_per_candidate", 0.15))
    bonuses = {
        "SEARCH_REQUIRED_SEARCH": ("SEARCH_REQUIRED", "SEARCH"),
        "TIME_SENSITIVE_SEARCH": ("TIME_SENSITIVE", "SEARCH"),
        "NEW_OR_TAIL_KNOWLEDGE_SEARCH": ("NEW_OR_TAIL_KNOWLEDGE", "SEARCH"),
        "CALCULATION_REQUIRED_CALCULATE": ("CALCULATION_REQUIRED", "CALCULATE"),
        "CLARIFY_REQUIRED_CLARIFY": ("CLARIFY_REQUIRED", "CLARIFY"),
        "FALSE_PREMISE_REFUSE": ("FALSE_PREMISE", "REFUSE"),
        "JUSTIFIED_REFUSE_REFUSE": ("JUSTIFIED_REFUSE", "REFUSE"),
    }
    total = 0.0
    for key, (tag, required_action) in bonuses.items():
        if action != required_action:
            continue
        if not semantic_tags.get(tag):
            continue
        eps = float(bonus_cfg.get(key, 0.05))
        eps = min(eps, 0.05)  # enforce ε ≤ 0.05
        total += eps
    return min(total, cap)


# ---------------------------------------------------------------------------
# Training annotation: U_rel (mixed estimator)
# Process features MUST NOT appear in this function.
# ---------------------------------------------------------------------------

def utility_rel(
    branch: dict[str, Any],
    teacher_label: dict[str, Any] | None,
    semantic_tags: dict[str, bool],
    example,
    cfg: dict[str, Any],
) -> float:
    """Relative utility estimate for training annotation (§5a).

    U_rel = base_value(a) × score(a|s) − action_cost(a) + semantic_bonus(a, z^sem)

    score(a|s) depends on action type:
      ANSWER    -> auto_correctness (gold hard label) or teacher fallback
      CALCULATE -> expression_validity × goal_alignment or teacher fallback
      CLARIFY   -> slot_hit_rate or teacher fallback
      SEARCH    -> teacher_helpfulness (main)
      REFUSE    -> teacher_helpfulness (main)

    Process features DO NOT appear in this function.
    """
    action = str(branch.get("action", "ANSWER")).upper()
    scoring_cfg = cfg.get("scoring", {})
    base_values = scoring_cfg.get("base_value", {})
    cost_map = scoring_cfg.get("action_cost", {})

    base_value = float(base_values.get(action, 0.5))
    action_cost = float(cost_map.get(action, 0.0))

    # Determine score by action type.
    if action == "ANSWER":
        auto = _auto_answer_score(branch, example)
        score = auto if auto is not None else teacher_helpfulness(branch, teacher_label)
    elif action == "CALCULATE":
        auto = _auto_calculate_score(branch, example)
        score = auto if auto is not None else teacher_helpfulness(branch, teacher_label)
    elif action == "CLARIFY":
        auto = _auto_clarify_score(branch, example)
        score = auto if auto is not None else teacher_helpfulness(branch, teacher_label)
    elif action == "SEARCH":
        score = teacher_helpfulness(branch, teacher_label)
    elif action == "REFUSE":
        score = teacher_helpfulness(branch, teacher_label)
    else:
        score = teacher_helpfulness(branch, teacher_label)

    bonus = _semantic_bonus(action, semantic_tags, cfg)
    u = base_value * score - action_cost + bonus
    return round(u, 6)


# ---------------------------------------------------------------------------
# Eval: U_real (outcome-based)
# ---------------------------------------------------------------------------

def utility_real(
    branch: dict[str, Any],
    example,
    branch_map: dict[str, dict[str, Any]],
    semantic_tags: dict[str, bool],
    cfg: dict[str, Any],
) -> float:
    """Outcome-based utility for eval (§5b).

    U_real = outcome_value(a, finalize_result) − action_cost(a)

    Only used for eval reporting and calibration; NOT for training pair construction.
    """
    action = str(branch.get("action", "ANSWER")).upper()
    scoring_cfg = cfg.get("scoring", {})
    cost_map = scoring_cfg.get("action_cost", {})
    action_cost = float(cost_map.get(action, 0.0))

    helpful = outcome_helpfulness(example, branch, branch_map, semantic_tags)

    outcome_values: dict[str, tuple[float, float]] = {
        "ANSWER": (1.0, -1.0),
        "SEARCH": (0.6, -0.1),
        "CALCULATE": (0.6, -0.1),
        "CLARIFY": (0.5, -0.1),
        "REFUSE": (0.4, -0.6),
    }
    helpful_val, unhelpful_val = outcome_values.get(action, (0.5, -0.5))
    outcome_value = helpful_val if helpful else unhelpful_val
    u = outcome_value - action_cost
    return round(u, 6)


# ---------------------------------------------------------------------------
# Backward-compat: evaluate_branch_utilities (still called by old code paths)
# ---------------------------------------------------------------------------

def load_utility_config(config: dict[str, Any]) -> dict[str, float]:
    scoring = config["scoring"]
    return {
        "lambda_tool": float(scoring.get("lambda_tool", 0.10)),
        "lambda_calc": float(scoring.get("lambda_calc", 0.05)),
        "lambda_clar": float(scoring.get("lambda_clar", 0.10)),
        "answer_correct": float(scoring.get("answer_correct", 1.0)),
        "answer_wrong": float(scoring.get("answer_wrong", -1.0)),
        "refuse_justified": float(scoring.get("refuse_justified", 0.4)),
        "refuse_unjustified": float(scoring.get("refuse_unjustified", -0.6)),
        "search_helpful": float(scoring.get("search_helpful", 0.6)),
        "search_unhelpful": float(scoring.get("search_unhelpful", -0.1)),
        "calculate_helpful": float(scoring.get("calculate_helpful", 0.6)),
        "calculate_unhelpful": float(scoring.get("calculate_unhelpful", -0.1)),
        "clarify_helpful": float(scoring.get("clarify_helpful", 0.5)),
        "clarify_unhelpful": float(scoring.get("clarify_unhelpful", -0.1)),
    }


def evaluate_branch_utilities(
    example,
    branches: list[dict[str, Any]],
    semantic_tags: dict[str, bool],
    config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Legacy entry point: compute outcome-based utility for each branch.

    Used by old rollout code and eval paths. For training annotation, prefer
    utility_rel() which is called after teacher labeling.
    """
    from mcagent_boundary.scoring.helpfulness import (
        is_calculate_helpful,
        is_clarify_helpful,
        is_refuse_justified,
        is_search_helpful,
    )
    utility_cfg = load_utility_config(config)
    branch_map = {branch["action"]: branch for branch in branches}
    scored: list[dict[str, Any]] = []
    for branch in branches:
        branch = dict(branch)
        action = branch["action"]
        if action == "ANSWER":
            correct = branch.get("correctness") is True
            branch["utility"] = utility_cfg["answer_correct"] if correct else utility_cfg["answer_wrong"]
            branch["outcome_label"] = "ANSWER_correct" if correct else "ANSWER_wrong"
        elif action == "SEARCH":
            helpful = is_search_helpful(example, branch, branch_map, semantic_tags)
            base = utility_cfg["search_helpful"] if helpful else utility_cfg["search_unhelpful"]
            branch["utility"] = base - utility_cfg["lambda_tool"]
            branch["outcome_label"] = "SEARCH_helpful" if helpful else "SEARCH_unhelpful"
        elif action == "CALCULATE":
            helpful = is_calculate_helpful(example, branch, branch_map, semantic_tags)
            base = utility_cfg["calculate_helpful"] if helpful else utility_cfg["calculate_unhelpful"]
            branch["utility"] = base - utility_cfg["lambda_calc"]
            branch["outcome_label"] = "CALCULATE_helpful" if helpful else "CALCULATE_unhelpful"
        elif action == "CLARIFY":
            helpful = is_clarify_helpful(example, branch, branch_map, semantic_tags)
            base = utility_cfg["clarify_helpful"] if helpful else utility_cfg["clarify_unhelpful"]
            branch["utility"] = base - utility_cfg["lambda_clar"]
            branch["outcome_label"] = "CLARIFY_helpful" if helpful else "CLARIFY_unhelpful"
        else:
            justified = is_refuse_justified(example, branch, branch_map, semantic_tags)
            branch["utility"] = utility_cfg["refuse_justified"] if justified else utility_cfg["refuse_unjustified"]
            branch["outcome_label"] = "REFUSE_justified" if justified else "REFUSE_unjustified"
        scored.append(branch)
    return scored
