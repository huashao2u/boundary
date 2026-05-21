from __future__ import annotations

from mcagent_boundary.adapters.base import DatasetAdapter, StandardizedExample


class ORBenchAdapter(DatasetAdapter):
    dataset_name = "or_bench"
    legacy_dataset_name = "or_bench"
    default_split = "train"
    boundary_type = "refusal"
    can_search = False
    can_calculate = False
    can_clarify = False
    allow_refuse = True

    def _convert(self, sample, split: str) -> StandardizedExample:
        metadata = self._base_metadata(sample, split)
        should_refuse = bool(sample.metadata.get("should_refuse", False))
        metadata.update(
            {
                "boundary_type": "refusal",
                "task_type": "refusal_boundary",
                "can_search": False,
                "can_calculate": False,
                "can_clarify": False,
                "allow_refuse": True,
                "or_bench_label": sample.metadata.get("or_bench_label"),
                "should_refuse": should_refuse,
                "category": sample.metadata.get("category"),
                "source_file": sample.metadata.get("source_file"),
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
