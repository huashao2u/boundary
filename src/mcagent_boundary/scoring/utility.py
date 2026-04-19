from __future__ import annotations

from typing import Any

from mcagent_boundary.scoring.helpfulness import (
    is_calculate_helpful,
    is_clarify_helpful,
    is_refuse_justified,
    is_search_helpful,
)


def load_utility_config(config: dict[str, Any]) -> dict[str, float]:
    scoring = config["scoring"]
    return {
        "lambda_tool": float(scoring["lambda_tool"]),
        "lambda_calc": float(scoring["lambda_calc"]),
        "lambda_clar": float(scoring["lambda_clar"]),
        "answer_correct": float(scoring["answer_correct"]),
        "answer_wrong": float(scoring["answer_wrong"]),
        "refuse_justified": float(scoring["refuse_justified"]),
        "refuse_unjustified": float(scoring["refuse_unjustified"]),
        "search_helpful": float(scoring["search_helpful"]),
        "search_unhelpful": float(scoring["search_unhelpful"]),
        "calculate_helpful": float(scoring["calculate_helpful"]),
        "calculate_unhelpful": float(scoring["calculate_unhelpful"]),
        "clarify_helpful": float(scoring["clarify_helpful"]),
        "clarify_unhelpful": float(scoring["clarify_unhelpful"]),
    }


def evaluate_branch_utilities(example, branches: list[dict[str, Any]], semantic_tags: dict[str, bool], config: dict[str, Any]) -> list[dict[str, Any]]:
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

