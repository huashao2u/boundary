from __future__ import annotations

from mcagent_boundary.adapters.base import DatasetAdapter, StandardizedExample


class MathAdapter(DatasetAdapter):
    dataset_name = "math"
    legacy_dataset_name = "competition_math"
    default_split = "train"
    boundary_type = "reasoning"
    can_search = False
    can_calculate = True
    can_clarify = False
    allow_refuse = True

    def _convert(self, sample, split: str) -> StandardizedExample:
        metadata = self._base_metadata(sample, split)
        if sample.metadata.get("level") is not None:
            metadata["math_level"] = sample.metadata.get("level")
        return StandardizedExample(
            example_id=sample.id.replace("competition_math", "math"),
            dataset=self.dataset_name,
            split=split,
            question=sample.question,
            gold_answer=sample.gold_answer,
            metadata=metadata,
        )
