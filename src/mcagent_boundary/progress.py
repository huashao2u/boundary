from __future__ import annotations

import sys
from typing import Any, Iterable, Iterator


class _NoProgress:
    def __init__(self, iterable: Iterable[Any]) -> None:
        self._iterable = iterable

    def __iter__(self) -> Iterator[Any]:
        return iter(self._iterable)

    def set_postfix(self, **_: Any) -> None:
        return None

    def close(self) -> None:
        return None


class _FallbackProgress:
    def __init__(
        self,
        iterable: Iterable[Any],
        *,
        total: int | None = None,
        desc: str = "",
        unit: str = "it",
    ) -> None:
        self._iterable = iterable
        self.total = total
        self.desc = desc or "progress"
        self.unit = unit
        self.count = 0
        self._last_percent = -1
        self._postfix = ""

    def __iter__(self) -> Iterator[Any]:
        self._emit(force=True)
        for item in self._iterable:
            yield item
            self.count += 1
            self._emit()
        self._emit(force=True, done=True)

    def set_postfix(self, **kwargs: Any) -> None:
        if not kwargs:
            self._postfix = ""
            return
        self._postfix = " " + " ".join(f"{key}={value}" for key, value in kwargs.items())

    def close(self) -> None:
        self._emit(force=True, done=True)

    def _emit(self, *, force: bool = False, done: bool = False) -> None:
        if self.total:
            percent = int((self.count / max(self.total, 1)) * 100)
            should_emit = force or done or percent >= self._last_percent + 10
            if not should_emit:
                return
            self._last_percent = percent
            sys.stderr.write(
                f"\r{self.desc}: {self.count}/{self.total} {self.unit} ({percent:3d}%){self._postfix}"
            )
        else:
            sys.stderr.write(f"\r{self.desc}: {self.count} {self.unit}{self._postfix}")
        if done:
            sys.stderr.write("\n")
        sys.stderr.flush()


def make_progress(
    iterable: Iterable[Any],
    *,
    total: int | None = None,
    desc: str = "",
    unit: str = "it",
    disable: bool = False,
):
    if disable:
        return _NoProgress(iterable)
    try:
        from tqdm.auto import tqdm

        return tqdm(iterable, total=total, desc=desc, unit=unit)
    except Exception:
        return _FallbackProgress(iterable, total=total, desc=desc, unit=unit)
