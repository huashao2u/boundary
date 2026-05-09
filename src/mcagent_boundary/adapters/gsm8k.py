from __future__ import annotations

from mcagent_boundary.adapters.base import DatasetAdapter, StandardizedExample


class GSM8KAdapter(DatasetAdapter):
    dataset_name = "gsm8k"
    legacy_dataset_name = "gsm8k"
    default_split = "train"
    boundary_type = "reasoning"
    can_search = False
    can_calculate = True
    can_clarify = False
    allow_refuse = False

    def _convert(self, sample, split: str) -> StandardizedExample:
        return StandardizedExample(
            example_id=sample.id,
            dataset=self.dataset_name,
            split=split,
            question=sample.question,
            gold_answer=sample.gold_answer,
            metadata=self._base_metadata(sample, split),
        )
