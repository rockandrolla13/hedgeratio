"""Domain-agnostic parallel map over the (dgp, path, config) grid."""
from __future__ import annotations
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Callable, TypeVar

T = TypeVar("T")
R = TypeVar("R")
_log = logging.getLogger(__name__)


def parallel_map(
    fn: Callable[[T], R],
    items: list[T],
    max_workers: int | None = None,
    desc: str = "grid",
) -> list[R]:
    """Map fn over items in separate processes.

    Failed cells are logged and dropped (the runner records a failed-cell marker); one
    bad path never kills the sweep. fn and items must be picklable.
    """
    results: list[R] = []
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(fn, it): it for it in items}
        for fut in as_completed(futs):
            try:
                results.append(fut.result())
            except Exception as exc:  # noqa: BLE001
                _log.error("%s failed for %r: %s", desc, futs[fut], exc)
    return results
