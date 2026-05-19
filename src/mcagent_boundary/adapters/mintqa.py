from __future__ import annotations

from mcagent_boundary.adapters.base import DatasetAdapter, StandardizedExample


class MintQAAdapter(DatasetAdapter):
    dataset_name = "mintqa"
    legacy_dataset_name = "mintqa"
    default_split = "train"
    boundary_type = "factual"
    can_search = True
    can_calculate = False
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

