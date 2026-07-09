"""Report generation: per-DGP summary tables (with MC standard errors), the blame-assignment
diagnostic plot, and an auto-generated results/REPORT.md.

The blame-assignment plot overlays beta_true and beta_hat from {vanilla, WoLF-only, full
composite} on the hard-mode S5+S3 path, with outlier and break times marked.
"""
from __future__ import annotations
from pathlib import Path


def summary_table(results: dict, out_dir: Path) -> Path:
    """Write one summary table per DGP (rows = configs, cols = metrics, +/- MC std errors)."""
    # TODO: aggregate FilterResult metrics into a parquet/markdown table
    raise NotImplementedError("summary_table")


def blame_assignment_plot(paths: dict, out_dir: Path) -> Path:
    """Overlay beta_true vs {vanilla, WoLF-only, composite} on S5+S3, mark outliers/breaks."""
    # TODO: matplotlib figure -> PNG
    raise NotImplementedError("blame_assignment_plot")


def write_report(results: dict, out_dir: Path) -> Path:
    """Assemble tables + figures into results/REPORT.md with findings vs the acceptance criteria."""
    # TODO: render REPORT.md referencing generated tables/figures
    raise NotImplementedError("write_report")
