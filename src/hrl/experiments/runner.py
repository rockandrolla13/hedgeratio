"""AblationRunner: runs the (dgp x path x config) grid in parallel and scores results.

Each grid cell builds its OWN fresh Pipeline/stages/providers (no cross-path shared mutable
state) from a config + seed, filters one sample, and returns scored metrics. Parallelised via
core.parallel.parallel_map. The inherently sequential part is the per-path recursion in t.
"""
from __future__ import annotations
from dataclasses import dataclass

from ..config import ExperimentConfig
from ..core.parallel import parallel_map


@dataclass
class GridCell:
    """One unit of work: a DGP name, MC path index (seed), and a pipeline config."""
    dgp: str
    path_idx: int
    seed: int
    config: ExperimentConfig


def run_cell(cell: GridCell) -> dict:
    """Filter one sample and return scored metrics. Picklable top-level fn for the pool."""
    # TODO: generate/load sample; Pipeline.from_config; run; score via eval.metrics
    raise NotImplementedError("run_cell")


class AblationRunner:
    """Expands a config into the factorial grid and evaluates every cell."""

    def __init__(self, config: ExperimentConfig, max_workers: int | None = None) -> None:
        self.config = config
        self.max_workers = max_workers

    def expand(self) -> list[GridCell]:
        """Expand DGPs x MC paths x ablation configs into GridCells."""
        # TODO: build the factorial (robust x noise x anchor x break) x dgps x paths
        raise NotImplementedError("AblationRunner.expand")

    def run(self) -> list[dict]:
        """Run the whole grid in parallel and return per-cell scored metrics."""
        return parallel_map(run_cell, self.expand(),
                            max_workers=self.max_workers, desc="ablation")
