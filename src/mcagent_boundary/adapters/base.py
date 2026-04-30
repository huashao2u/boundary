from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

from mcagent_core.data.loaders import load_dataset


BOUNDARY_TO_TASK = {
    "reasoning": "math",
    "factual": "factual_boundary",
    "intention": "intention_boundary",
    "refusal": "refusal_boundary",
}


@dataclass
class StandardizedExample:
    example_id: str
    dataset: str
    split: str
    question: str
    gold_answer: str | list[str] | None
    metadata: dict[str, Any]

    @property
    def boundary_type(self) -> str:
        return str(self.metadata.get("boundary_type", "factual"))

    @property
    def task_type(self) -> str:
        return str(self.metadata.get("task_type") or BOUNDARY_TO_TASK.get(self.boundary_type, "factual_boundary"))

    @property
    def can_search(self) -> bool:
        return bool(self.metadata.get("can_search", False))

    @property
    def can_calculate(self) -> bool:
        return bool(self.metadata.get("can_calculate", False))

    @property
    def can_clarify(self) -> bool:
        return bool(self.metadata.get("can_clarify", False))

    @property
    def allow_refuse(self) -> bool:
        return bool(self.metadata.get("allow_refuse", True))

    def allowed_actions(self) -> list[str]:
        actions = ["ANSWER"]
        if self.can_search:
            actions.append("SEARCH")
        if self.can_calculate:
            actions.append("CALCULATE")
        if self.can_clarify:
            actions.append("CLARIFY")
        if self.allow_refuse:
            actions.append("REFUSE")
        return actions

    def to_record(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "dataset": self.dataset,
            "split": self.split,
            "question": self.question,
            "gold_answer": self.gold_answer,
            "metadata": self.metadata,
        }

    def to_legacy_sample(self):
        return SimpleNamespace(
            id=self.example_id,
            dataset=str(self.metadata.get("legacy_dataset", self.dataset)),
            question=self.question,
            gold_answer=self.gold_answer,
            metadata=self.metadata,
            task_type=self.task_type,
        )


class DatasetAdapter(ABC):
    dataset_name = ""
    legacy_dataset_name = ""
    default_split = "train"
    boundary_type = "factual"
    can_search = False
    can_calculate = False
    can_clarify = False
    allow_refuse = True

    def load(self, dataset_root, limit: int | None = None, split: str | None = None) -> list[StandardizedExample]:
        target_split = split or self.default_split
        raw_samples = load_dataset(self.legacy_dataset_name, split=target_split, limit=limit, dataset_root=dataset_root)
        return [self._convert(sample, split=target_split) for sample in raw_samples]

    def _base_metadata(self, sample, split: str) -> dict[str, Any]:
        original = dict(sample.metadata or {})
        return {
            **original,
            "boundary_type": self.boundary_type,
            "task_type": BOUNDARY_TO_TASK.get(self.boundary_type, "factual_boundary"),
            "can_search": self.can_search,
            "can_calculate": self.can_calculate,
            "can_clarify": self.can_clarify,
            "allow_refuse": self.allow_refuse,
            "legacy_dataset": sample.dataset,
            "legacy_id": sample.id,
            "split": split,
        }

    @abstractmethod
    def _convert(self, sample, split: str) -> StandardizedExample:
        raise NotImplementedError
