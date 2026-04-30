from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable


def ensure_parent(path: str | Path) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path


def to_jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]
    if hasattr(value, "tolist"):
        try:
            return to_jsonable(value.tolist())
        except Exception:
            pass
    if hasattr(value, "item"):
        try:
            return to_jsonable(value.item())
        except Exception:
            pass
    return str(value)


def read_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    output_path = ensure_parent(path)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(to_jsonable(payload), handle, ensure_ascii=False, indent=2)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path: str | Path, records: Iterable[dict[str, Any]]) -> None:
    output_path = ensure_parent(path)
    with output_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(to_jsonable(record), ensure_ascii=False) + "\n")


def write_jsonl_by_dataset(
    path: str | Path,
    records: Iterable[dict[str, Any]],
    *,
    dataset_key: str = "dataset",
) -> dict[str, str]:
    output_path = Path(path)
    materialized = list(records)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in materialized:
        dataset = str(record.get(dataset_key) or "unknown")
        grouped.setdefault(dataset, []).append(record)
    outputs: dict[str, str] = {}
    for dataset, dataset_records in grouped.items():
        dataset_path = output_path.parent / "by_dataset" / dataset / output_path.name
        write_jsonl(dataset_path, dataset_records)
        outputs[dataset] = str(dataset_path)
    return outputs
