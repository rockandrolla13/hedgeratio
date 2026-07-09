"""Evaluation: estimation/stability/spread/coverage metrics, backtest, and reports.

Fat-tail robustness layer (spec docs/FAT_TAIL_SPEC.md) adds:
  * stats            -- paired inference (Wilcoxon/HL, Holm-Bonferroni, Kupiec, Christoffersen)
  * conformal_bands  -- adaptive conformal (ACI) calibration of z-score bands (Mechanism 4)
  * fat_tail_report  -- auto-assembled FAT_TAIL_REPORT.md
"""
from . import conformal_bands, fat_tail_report, stats
from .backtest import Backtest

__all__ = ["Backtest", "stats", "conformal_bands", "fat_tail_report"]
