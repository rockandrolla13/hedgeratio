"""Data: synthetic ground-truth DGPs (S1..S6) and real ETF-pair loaders."""
from .synthetic import SyntheticSample, generate
from .loaders import load_pair

__all__ = ["SyntheticSample", "generate", "load_pair"]
