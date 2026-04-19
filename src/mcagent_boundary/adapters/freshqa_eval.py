from __future__ import annotations

from mcagent_boundary.adapters.base import DatasetAdapter, StandardizedExample


class FreshQAEvalAdapter(DatasetAdapter):
    dataset_name = "freshqa"
    legacy_dataset_name = "freshqa"
    default_split = "test"
    boundary_type = "factual"
    can_search = True
    can_calculate = False
    can_clarify = False
    allow_refuse = True

    def _convert(self, sample, split: str) -> StandardizedExample:
        metadata = self._base_metadata(sample, split)
        metadata["eval_only"] = True
        return StandardizedExample(
            example_id=sample.id,
            dataset=self.dataset_name,
            split=split,
            question=sample.question,
            gold_answer=sample.gold_answer,
            metadata=metadata,
        )

