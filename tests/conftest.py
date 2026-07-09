"""Shared pytest fixtures."""
from __future__ import annotations

import numpy as np
import pytest

from hrl.config import ExperimentConfig, PipelineConfig
from hrl.core.context import StepContext


@pytest.fixture
def rng() -> np.random.Generator:
    """Deterministic RNG for reproducible tests."""
    return np.random.default_rng(0)


@pytest.fixture
def default_config() -> ExperimentConfig:
    """A minimal, valid ExperimentConfig."""
    return ExperimentConfig()


@pytest.fixture
def pipeline_config() -> PipelineConfig:
    """Vanilla (predict + update) pipeline config."""
    return PipelineConfig(stages=["predict", "update"], weight_fn="none", noise_model="fixed")


@pytest.fixture
def blank_context() -> StepContext:
    """A hand-built StepContext for exercising a single stage in isolation."""
    return StepContext(t=1, y=1.0, x=2.0,
                       theta=np.array([0.0, 1.0]),
                       P=np.eye(2), R=1e-3)
