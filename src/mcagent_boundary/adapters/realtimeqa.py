"""Stub adapter for RealTimeQA (future eval extension).

See CLAUDE.md §Change 6: this file is a clean extension hook — it registers
in the adapter registry so code imports stay stable, but ``load()`` fails
loudly until RealTimeQA assets land under ``<dataset_root>/realtimeqa``.

Wire-up, once assets exist:
  1. drop the ``FileNotFoundError`` guard in ``load()``,
  2. implement ``_convert`` to map the raw sample onto ``StandardizedExample``,
  3. register the legacy loader under ``legacy_dataset_name = "realtimeqa"``
     in ``mcagent_core.data.loaders``,
  4. uncomment the ``realtimeqa`` line in ``configs/rollout.yaml``.
"""

from __future__ import annotations

from pathlib import Path

from mcagent_boundary.adapters.base import DatasetAdapter, StandardizedExample


class RealTimeQAEvalAdapter(DatasetAdapter):
    dataset_name = "realtimeqa"
    legacy_dataset_name = "realtimeqa"
    default_split = "test"
    boundary_type = "factual"
    can_search = True
    can_calculate = False
    can_clarify = False
    allow_refuse = True

    def load(self, dataset_root, limit: int | None = None, split: str | None = None) -> list[StandardizedExample]:
        candidate = Path(dataset_root) / self.dataset_name
        if not candidate.exists():
            raise FileNotFoundError(
                f"realtimeqa assets not present at {candidate!s} — see CLAUDE.md §Change 6 "
                "(extension hooks) for how to enable this evaluation target."
            )
        return super().load(dataset_root=dataset_root, limit=limit, split=split)

    def _convert(self, sample, split: str) -> StandardizedExample:  # pragma: no cover - stub
        raise NotImplementedError(
            "RealTimeQAEvalAdapter._convert is a stub. Implement when realtimeqa "
            "assets and legacy loader are available. See CLAUDE.md §Change 6."
        )
