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


def _extract_commonsenseqa_labels(answer: Any) -> list[str]:
    text = str(answer or "").strip()
    if not text:
        return []
    labels: list[str] = []
    if re.fullmatch(r"[A-Ea-e]", text):
        labels.append(text.upper())
    patterns = [
        r"(?:^|[\s\(\[])([A-Ea-e])[\)\].:]",
        r"\b(?:answer|option|choice)\s*(?:is|:)?\s*[\(\[]?([A-Ea-e])\b",
        r"\b(?:choose|select|selected|picks?)\s*(?:option|choice)?\s*[\(\[]?([A-Ea-e])\b",
    ]
    for pattern in patterns:
        for match in re.finditer(pattern, text):
            labels.append(match.group(1).upper())
    deduped: list[str] = []
    for label in labels:
        if label not in deduped:
            deduped.append(label)
    return deduped


def _infer_commonsenseqa_gold(gold: Any, metadata: dict[str, Any]) -> tuple[str, list[str]]:
    gold_label = str(metadata.get("answer_label") or "").strip().upper()
    text_values: list[str] = []
    if metadata.get("answer_text") not in (None, ""):
        text_values.append(str(metadata["answer_text"]))
    choices = metadata.get("choices")
    if gold_label and isinstance(choices, list):
        for choice in choices:
            if not isinstance(choice, dict):
                continue
            if str(choice.get("label") or "").strip().upper() == gold_label and choice.get("text") not in (None, ""):
                text_values.append(str(choice["text"]))

    values = gold if isinstance(gold, list) else [gold]
    for value in values:
        if value in (None, ""):
            continue
        normalized = normalize_text(str(value))
        if not gold_label and re.fullmatch(r"[a-e]", normalized):
            gold_label = normalized.upper()
            continue
        labeled = re.match(r"^([A-Ea-e])[\).:]\s*(.+)$", str(value).strip())
        if labeled:
            if not gold_label:
                gold_label = labeled.group(1).upper()
            text_values.append(labeled.group(2))
            continue
        if normalized.upper() != gold_label and not re.fullmatch(r"[a-e]", normalized):
            text_values.append(str(value))

    deduped_texts: list[str] = []
    for value in text_values:
        if value not in deduped_texts:
            deduped_texts.append(value)
    return gold_label, deduped_texts


def _looks_like_commonsenseqa_sample(sample: dict[str, Any], gold: Any) -> bool:
    dataset = str(sample.get("dataset") or sample.get("legacy_dataset") or "").lower()
    if dataset in {"commonsenseqa", "commonsense_qa", "csqa"}:
        return True
    metadata = sample.get("metadata") if isinstance(sample.get("metadata"), dict) else {}
    if metadata.get("answer_label") or metadata.get("answer_text"):
        return True
    if isinstance(gold, list):
        return any(re.fullmatch(r"[A-Ea-e]", str(item).strip()) for item in gold if item not in (None, ""))
    return False


def _is_commonsenseqa_answer_correct(sample: dict[str, Any], predicted_answer: str | None, gold: Any) -> bool | None:
    normalized_pred = normalize_text(str(predicted_answer or ""))
    if not normalized_pred:
        return False
    metadata = sample.get("metadata") if isinstance(sample.get("metadata"), dict) else {}
    gold_label, text_values = _infer_commonsenseqa_gold(gold, dict(metadata))
    labels = _extract_commonsenseqa_labels(predicted_answer)
    if labels:
        if len(labels) > 1:
            return False
        if gold_label:
            return labels[0] == gold_label
        return None
    for value in text_values:
        normalized_gold = normalize_text(str(value))
        if normalized_gold and (normalized_pred == normalized_gold or normalized_gold in normalized_pred):
            return True
    return False


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
    try:
        normalized = decimal.normalize()
        if normalized == normalized.to_integral_value():
            return format(normalized, "f").split(".", 1)[0]
        return format(normalized, "f").rstrip("0").rstrip(".")
    except (InvalidOperation, ValueError, OverflowError):
        return cleaned.lower()


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
    if _looks_like_commonsenseqa_sample(sample, gold):
        return _is_commonsenseqa_answer_correct(sample, predicted_answer, gold)
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
