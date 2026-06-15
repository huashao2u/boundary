from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from mcagent_core.utils.config import load_config
from mcagent_core.utils.io import write_json
from mcagent_core.utils.paths import DATASET_ROOT, MODEL_ROOT


QUESTION_KEYS = ("question", "problem", "query", "prompt", "task")
ANSWER_KEYS = ("answer", "solution", "final_answer", "gold_answer", "target", "response")


class DatasetArtifactMissingError(RuntimeError):
    """Raised when a dataset shard is only a Git LFS pointer or otherwise unavailable."""


@dataclass
class UnifiedSample:
    id: str
    dataset: str
    question: str
    gold_answer: str | list[str] | None
    metadata: dict[str, Any]
    task_type: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "dataset": self.dataset,
            "question": self.question,
            "gold_answer": self.gold_answer,
            "metadata": self.metadata,
            "task_type": self.task_type,
        }


def _require_pandas():
    try:
        import pandas as pd
    except ImportError as exc:
        raise RuntimeError("pandas is required for parquet/csv loaders in this environment.") from exc
    return pd


def _is_git_lfs_pointer(path: Path) -> bool:
    if not path.exists() or path.stat().st_size > 512:
        return False
    try:
        head = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False
    return head.startswith("version https://git-lfs.github.com/spec/v1")


def _iter_nonempty_answers(row: dict[str, Any]) -> list[str]:
    answers: list[str] = []
    for key, value in row.items():
        if key.startswith("answer_") and value not in ("", None):
            answers.append(str(value).strip())
    return answers


def _extract_first(row: dict[str, Any], candidates: Iterable[str]) -> Any:
    for key in candidates:
        if key in row and row[key] not in ("", None):
            return row[key]
    return None


def _read_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if limit is not None and index >= limit:
                break
            rows.append(json.loads(line))
    return rows


def _read_parquet_records(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    if _is_git_lfs_pointer(path):
        raise DatasetArtifactMissingError(f"{path} is a Git LFS pointer. Fetch the real artifact before loading.")
    pd = _require_pandas()
    frame = pd.read_parquet(path)
    if limit is not None:
        frame = frame.head(limit)
    return frame.to_dict(orient="records")


def _read_all_parquet_records(paths: list[Path], limit: int | None = None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for path in paths:
        if limit is not None and len(records) >= limit:
            break
        remaining = None if limit is None else max(0, limit - len(records))
        records.extend(_read_parquet_records(path, limit=remaining))
    return records


def _load_gsm8k(dataset_root: Path, split: str, limit: int | None) -> list[UnifiedSample]:
    shard = dataset_root / "gsm8k" / "main" / f"{split}-00000-of-00001.parquet"
    rows = _read_parquet_records(shard, limit=limit)
    samples: list[UnifiedSample] = []
    for index, row in enumerate(rows):
        samples.append(
            UnifiedSample(
                id=f"gsm8k-{split}-{index}",
                dataset="gsm8k",
                question=str(row["question"]),
                gold_answer=str(row["answer"]),
                metadata={"split": split, "raw_answer": row["answer"]},
                task_type="math",
            )
        )
    return samples


def _load_competition_math(dataset_root: Path, split: str, limit: int | None) -> list[UnifiedSample]:
    shard_dir = dataset_root / "competition_math" / "data"
    shards = sorted(shard_dir.glob("*.parquet"))
    if not shards:
        raise FileNotFoundError(f"No parquet shards found under {shard_dir}")
    rows = _read_all_parquet_records(shards, limit=limit)
    samples: list[UnifiedSample] = []
    for index, row in enumerate(rows):
        question = _extract_first(row, QUESTION_KEYS)
        answer = _extract_first(row, ANSWER_KEYS)
        metadata = {key: value for key, value in row.items() if key not in set(QUESTION_KEYS) | set(ANSWER_KEYS)}
        samples.append(
            UnifiedSample(
                id=f"competition_math-{split}-{index}",
                dataset="competition_math",
                question=str(question),
                gold_answer=None if answer is None else str(answer),
                metadata={"split": split, **metadata},
                task_type="math",
            )
        )
    return samples


def _load_freshqa(dataset_root: Path, split: str, limit: int | None) -> list[UnifiedSample]:
    csv_path = dataset_root / "freshqa" / "FreshQA_v112425.csv"
    pd = _require_pandas()
    try:
        frame = pd.read_csv(csv_path, encoding="utf-8")
    except UnicodeDecodeError:
        frame = pd.read_csv(csv_path, encoding="latin-1")
    frame = frame[frame["split"].astype(str).str.upper() == split.upper()]
    if limit is not None:
        frame = frame.head(limit)
    samples: list[UnifiedSample] = []
    for _, row in frame.iterrows():
        row_dict = row.to_dict()
        answers = _iter_nonempty_answers(row_dict)
        metadata = {
            "split": split,
            "false_premise": str(row_dict.get("false_premise", "")).upper() == "TRUE",
            "effective_year": row_dict.get("effective_year"),
            "next_review": row_dict.get("next_review"),
            "num_hops": row_dict.get("num_hops"),
            "fact_type": row_dict.get("fact_type"),
            "source": row_dict.get("source"),
        }
        samples.append(
            UnifiedSample(
                id=f"freshqa-{row_dict.get('id')}",
                dataset="freshqa",
                question=str(row_dict["question"]),
                gold_answer=answers or None,
                metadata=metadata,
                task_type="factual_boundary",
            )
        )
    return samples


def _load_in3(dataset_root: Path, split: str, limit: int | None) -> list[UnifiedSample]:
    jsonl_path = dataset_root / "IN3" / f"{split}.jsonl"
    rows = _read_jsonl(jsonl_path, limit=limit)
    samples: list[UnifiedSample] = []
    for index, row in enumerate(rows):
        missing_details = row.get("missing_details", [])
        best_detail = None
        if missing_details:
            best_detail = max(missing_details, key=lambda item: int(item.get("importance", 0)))
        metadata = {
            "split": split,
            "category": row.get("category"),
            "vague": bool(row.get("vague", False)),
            "thought": row.get("thought"),
            "missing_details": missing_details,
            "gold_clarify_question": None if best_detail is None else best_detail.get("inquiry"),
            "gold_clarify_reply": None
            if best_detail is None
            else (best_detail.get("options") or [best_detail.get("description")])[0],
        }
        samples.append(
            UnifiedSample(
                id=f"in3-{split}-{index}",
                dataset="in3",
                question=str(row["task"]),
                gold_answer=None,
                metadata=metadata,
                task_type="intention_boundary",
            )
        )
    return samples


def _load_mintqa(dataset_root: Path, split: str, limit: int | None) -> list[UnifiedSample]:
    shard_dir = dataset_root / "MintQA-Ti-v0.1" / "data"
    shards = sorted(shard_dir.glob(f"{split}-*.parquet"))
    if not shards:
        raise FileNotFoundError(f"No parquet shards found under {shard_dir}")
    rows = _read_all_parquet_records(shards, limit=limit)
    samples: list[UnifiedSample] = []
    for index, row in enumerate(rows):
        question = _extract_first(row, QUESTION_KEYS)
        answer = _extract_first(row, ANSWER_KEYS)
        metadata = {key: value for key, value in row.items() if key not in set(QUESTION_KEYS) | set(ANSWER_KEYS)}
        if "graph" in metadata:
            graph = metadata["graph"]
            if isinstance(graph, list):
                metadata["graph_size"] = len(graph)
                metadata["graph_preview"] = graph[:5]
            metadata.pop("graph", None)
        samples.append(
            UnifiedSample(
                id=f"mintqa-{split}-{index}",
                dataset="mintqa",
                question=str(question),
                gold_answer=answer if isinstance(answer, list) else (None if answer is None else str(answer)),
                metadata={"split": split, **metadata},
                task_type="factual_boundary",
            )
        )
    return samples


def _load_commonsenseqa(dataset_root: Path, split: str, limit: int | None) -> list[UnifiedSample]:
    split_aliases = {
        "dev": "validation",
        "valid": "validation",
        "val": "validation",
    }
    target_split = split_aliases.get(split, split)
    jsonl_path = dataset_root / "commonsenseqa" / f"{target_split}.jsonl"
    rows = _read_jsonl(jsonl_path, limit=limit)
    samples: list[UnifiedSample] = []
    for index, row in enumerate(rows):
        choices = row.get("choices") or {}
        labels = [str(item) for item in choices.get("label", [])]
        texts = [str(item) for item in choices.get("text", [])]
        answer_key = str(row.get("answerKey") or "").strip()
        label_to_text = {label: text for label, text in zip(labels, texts)}
        answer_text = label_to_text.get(answer_key)
        choice_lines = [f"{label}. {text}" for label, text in zip(labels, texts)]
        question = str(row["question"]).strip()
        if choice_lines:
            question = question + "\n\nChoices:\n" + "\n".join(choice_lines)
        gold_answer: list[str] | None
        if answer_key and answer_text:
            gold_answer = [answer_key, answer_text, f"{answer_key}. {answer_text}", f"{answer_key}: {answer_text}"]
        elif answer_text:
            gold_answer = [answer_text]
        elif answer_key:
            gold_answer = [answer_key]
        else:
            gold_answer = None
        samples.append(
            UnifiedSample(
                id=f"commonsenseqa-{target_split}-{index}",
                dataset="commonsenseqa",
                question=question,
                gold_answer=gold_answer,
                metadata={
                    "split": target_split,
                    "source_id": row.get("id"),
                    "question_concept": row.get("question_concept"),
                    "choices": [{"label": label, "text": text} for label, text in zip(labels, texts)],
                    "answer_label": answer_key or None,
                    "answer_text": answer_text,
                    "search_required": False,
                    "commonsense_answerable": True,
                    "knowledge_type": "commonsense",
                },
                task_type="factual_boundary",
            )
        )
    return samples


def _load_or_bench(dataset_root: Path, split: str, limit: int | None) -> list[UnifiedSample]:
    del split  # OR-Bench ships train CSVs only.
    pd = _require_pandas()
    source_specs = [
        ("or-bench-80k.csv", "benign", False, 0.6),
        ("or-bench-hard-1k.csv", "hard", False, 0.2),
        ("or-bench-toxic.csv", "toxic", True, 0.2),
    ]
    source_limits: list[int | None]
    if limit is None:
        source_limits = [None] * len(source_specs)
    else:
        raw_limits = [limit * weight for *_, weight in source_specs]
        source_limits = [int(math.floor(value)) for value in raw_limits]
        remainder = limit - sum(value for value in source_limits if value is not None)
        ranked_remainders = sorted(
            range(len(raw_limits)),
            key=lambda index: raw_limits[index] - math.floor(raw_limits[index]),
            reverse=True,
        )
        for index in ranked_remainders[:remainder]:
            source_limits[index] = int(source_limits[index] or 0) + 1
    samples: list[UnifiedSample] = []
    for (file_name, label, should_refuse, _weight), source_limit in zip(source_specs, source_limits):
        csv_path = dataset_root / "or-bench" / file_name
        if source_limit == 0:
            continue
        frame = pd.read_csv(csv_path, nrows=source_limit)
        for index, row in frame.iterrows():
            category = str(row.get("category", ""))
            samples.append(
                UnifiedSample(
                    id=f"or_bench-{label}-{index}",
                    dataset="or_bench",
                    question=str(row["prompt"]),
                    gold_answer="REFUSE" if should_refuse else "ANSWER",
                    metadata={
                        "split": "train",
                        "source_file": file_name,
                        "category": category,
                        "boundary_type": "refusal",
                        "task_type": "refusal_boundary",
                        "can_search": False,
                        "can_calculate": False,
                        "can_clarify": False,
                        "allow_refuse": True,
                        "or_bench_label": label,
                        "should_refuse": should_refuse,
                    },
                    task_type="refusal_boundary",
                )
            )
            if limit is not None and len(samples) >= limit:
                return samples
    return samples


DATASET_LOADERS = {
    "gsm8k": _load_gsm8k,
    "competition_math": _load_competition_math,
    "freshqa": _load_freshqa,
    "in3": _load_in3,
    "mintqa": _load_mintqa,
    "commonsenseqa": _load_commonsenseqa,
    "commonsense_qa": _load_commonsenseqa,
    "csqa": _load_commonsenseqa,
    "or_bench": _load_or_bench,
    "or-bench": _load_or_bench,
}


def load_dataset(
    dataset_name: str,
    split: str,
    limit: int | None = None,
    dataset_root: Path | None = None,
) -> list[UnifiedSample]:
    normalized = dataset_name.lower()
    if normalized == "math":
        normalized = "competition_math"
    if normalized not in DATASET_LOADERS:
        raise KeyError(f"Unsupported dataset: {dataset_name}")
    root = DATASET_ROOT if dataset_root is None else dataset_root
    return DATASET_LOADERS[normalized](root, split, limit)


def inspect_resources(dataset_root: Path | None = None, model_root: Path | None = None) -> dict[str, Any]:
    root = DATASET_ROOT if dataset_root is None else dataset_root
    models = MODEL_ROOT if model_root is None else model_root
    qwen_root = models / "qwen" / "Qwen2.5-7B-Instruct"
    report = {
        "dataset_root": str(root),
        "model_root": str(models),
        "datasets": {},
        "models": {
            "qwen2_5_7b_instruct_exists": qwen_root.exists(),
            "qwen2_5_7b_hf_layout": all((qwen_root / name).exists() for name in ("config.json", "tokenizer.json", "model.safetensors.index.json")),
            "available_qwen_variants": [path.name for path in sorted((models / "qwen").glob("*"))] if (models / "qwen").exists() else [],
        },
    }
    for dataset_name in DATASET_LOADERS:
        entry: dict[str, Any] = {"available": True}
        try:
            sample = load_dataset(dataset_name, split="test" if dataset_name == "freshqa" else "train", limit=1, dataset_root=root)
            entry["sample_count_checked"] = len(sample)
        except Exception as exc:
            entry["available"] = False
            entry["error"] = str(exc)
        report["datasets"][dataset_name] = entry
    return report


def load_mixed_datasets(
    dataset_limits: dict[str, int],
    split_map: dict[str, str],
    dataset_root: Path | None = None,
    seed: int = 7,
) -> list[UnifiedSample]:
    root = DATASET_ROOT if dataset_root is None else dataset_root
    rng = random.Random(seed)
    mixed: list[UnifiedSample] = []
    for dataset_name, limit in dataset_limits.items():
        if limit <= 0:
            continue
        split = split_map[dataset_name]
        samples = load_dataset(dataset_name, split=split, limit=limit, dataset_root=root)
        mixed.extend(samples)
    rng.shuffle(mixed)
    return mixed


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect or sample MCAgent datasets.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--dataset", choices=sorted(DATASET_LOADERS), help="Dataset to sample.")
    parser.add_argument("--split", default=None)
    parser.add_argument("--limit", type=int, default=2)
    parser.add_argument("--report", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    dataset_root = Path(config["paths"]["dataset_root"]).resolve()
    model_root = Path(config["paths"]["model_root"]).resolve().parents[1]

    if args.report:
        report = inspect_resources(dataset_root=dataset_root, model_root=model_root)
        output_path = Path(config["paths"]["resource_report_output"])
        write_json(output_path, report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    if args.dataset is None:
        raise SystemExit("--dataset is required unless --report is used.")

    split_map = config["data"]["default_split"]
    split = args.split or split_map[args.dataset]
    samples = [sample.to_dict() for sample in load_dataset(args.dataset, split=split, limit=args.limit, dataset_root=dataset_root)]
    print(json.dumps(samples, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
