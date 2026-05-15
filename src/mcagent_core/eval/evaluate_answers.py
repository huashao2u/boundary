from __future__ import annotations

import argparse
import json
import re
from decimal import Decimal, InvalidOperation
from fractions import Fraction
from pathlib import Path
from typing import Any

from mcagent_core.utils.io import read_jsonl, write_json


NUMBER_PATTERN = re.compile(r"[-+]?\$?\d[\d,]*(?:\.\d+)?(?:/\d[\d,]*)?")


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _normalize_numeric(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = str(value).strip().replace("$", "").replace(",", "")
    if not cleaned:
        return None
    try:
        if "/" in cleaned and not cleaned.startswith(("http://", "https://")):
            fraction = Fraction(cleaned)
            decimal = Decimal(fraction.numerator) / Decimal(fraction.denominator)
        else:
            decimal = Decimal(cleaned)
    except (InvalidOperation, ValueError, ZeroDivisionError):
        return cleaned.lower()
    normalized = decimal.normalize()
    if normalized == normalized.to_integral_value():
        return str(normalized.quantize(Decimal(1)))
    return format(normalized, "f").rstrip("0").rstrip(".")


def extract_math_final_answer(text: str) -> str:
    if not text:
        return ""
    if "####" in text:
        text = text.split("####")[-1]
    boxed = re.findall(r"\\boxed\{([^{}]+)\}", text)
    if boxed:
        text = boxed[-1]
    numbers = NUMBER_PATTERN.findall(str(text))
    if numbers:
        return _normalize_numeric(numbers[-1]) or numbers[-1]
    return normalize_text(text)


def is_answer_correct(sample: dict[str, Any], predicted_answer: str | None) -> bool | None:
    if predicted_answer is None:
        return False
    gold = sample.get("gold_answer")
    task_type = sample.get("task_type")
    if gold is None:
        return None
    if task_type == "math":
        return extract_math_final_answer(str(predicted_answer)) == extract_math_final_answer(str(gold))
    if isinstance(gold, list):
        normalized_pred = normalize_text(str(predicted_answer))
        return any(
            normalize_text(answer) == normalized_pred or normalize_text(answer) in normalized_pred
            for answer in gold
            if answer
        )
    return normalize_text(str(gold)) == normalize_text(str(predicted_answer))


def evaluate_answers(rollouts: list[dict[str, Any]]) -> dict[str, Any]:
    math_records = [record for record in rollouts if record["task_type"] == "math"]
    factual_records = [record for record in rollouts if record["task_type"] == "factual_boundary"]
    math_correct = [record["correctness"] for record in math_records if record.get("correctness") is not None]
    factual_correct = [record["correctness"] for record in factual_records if record.get("correctness") is not None]
    return {
        "math_accuracy": (sum(bool(value) for value in math_correct) / len(math_correct)) if math_correct else None,
        "factual_correctness": (sum(bool(value) for value in factual_correct) / len(factual_correct)) if factual_correct else None,
        "num_math_samples": len(math_records),
        "num_factual_samples": len(factual_records),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate answer-level metrics from rollout logs.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    metrics = evaluate_answers(read_jsonl(args.input))
    write_json(args.output, metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
