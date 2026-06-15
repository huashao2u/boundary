from __future__ import annotations

from mcagent_boundary.adapters.base import DatasetAdapter, StandardizedExample


class CommonsenseQAAdapter(DatasetAdapter):
    dataset_name = "commonsenseqa"
    legacy_dataset_name = "commonsenseqa"
    default_split = "train"
    boundary_type = "factual"
    can_search = True
    can_calculate = False
    can_clarify = False
    allow_refuse = False

    def _convert(self, sample, split: str) -> StandardizedExample:
        metadata = self._base_metadata(sample, split)
        metadata.update(
            {
                "search_required": False,
                "commonsense_answerable": True,
                "knowledge_type": "commonsense",
            }
        )
        return StandardizedExample(
            example_id=sample.id,
            dataset=self.dataset_name,
            split=split,
            question=sample.question,
            gold_answer=sample.gold_answer,
            metadata=metadata,
        )
