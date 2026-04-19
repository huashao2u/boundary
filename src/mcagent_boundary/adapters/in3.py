from __future__ import annotations

from mcagent_boundary.adapters.base import DatasetAdapter, StandardizedExample


class IN3Adapter(DatasetAdapter):
    dataset_name = "in3"
    legacy_dataset_name = "in3"
    default_split = "train"
    boundary_type = "intention"
    can_search = False
    can_calculate = False
    can_clarify = True
    allow_refuse = True

    def _convert(self, sample, split: str) -> StandardizedExample:
        return StandardizedExample(
            example_id=sample.id,
            dataset=self.dataset_name,
            split=split,
            question=sample.question,
            gold_answer=sample.gold_answer,
            metadata=self._base_metadata(sample, split),
        )

