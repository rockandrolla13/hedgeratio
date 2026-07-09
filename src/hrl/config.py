"""Experiment configuration. One ExperimentConfig == one reproducible run."""
from __future__ import annotations
from pathlib import Path

from pydantic import BaseModel, Field


class DGPConfig(BaseModel):
    """Synthetic data-generating-process selection and Monte Carlo size."""
    name: str = "S1"          # S1..S6
    n_steps: int = 2500
    n_paths: int = 200


class PipelineConfig(BaseModel):
    """Which stages run, in canonical order, plus their policy choices and hyperparameters.

    The ablation grid is expressed entirely through this object.
    """
    stages: list[str] = Field(default_factory=lambda: ["predict", "update"])
    weight_fn: str = "imq"            # imq | huber | student_t | none
    noise_model: str = "fixed"        # fixed | ewma | garch | vbakf
    detector: str = "none"            # none | cusum | bocpd
    anchor: str = "none"              # none | johansen | tls
    c: float = 3.0                    # WoLF hyperparameter
    lam: float = 0.94                 # EWMA decay
    phi: float = 0.99                 # AR(1) anchor pull
    q_alpha: float = 1e-7             # process noise on alpha
    q_beta: float = 1e-5              # process noise on beta
    r: float = 1e-3                   # fixed measurement noise (noise_model == "fixed")
    r_floor: float = 1e-8             # measurement-noise floor (prevents S_t -> 0)
    p0: float = 1.0                   # initial state-covariance scale
    hazard: float = 1.0 / 500.0       # BOCPD hazard rate


class BacktestConfig(BaseModel):
    """z-band pairs strategy parameters (secondary metric only)."""
    enter_z: float = 2.0
    exit_z: float = 0.5
    cost_bps: float = 5.0


class ExperimentConfig(BaseModel):
    """Root config. Serialised to/from YAML; carries the seed for reproducibility."""
    seed: int = 0
    dgp: DGPConfig = DGPConfig()
    pipeline: PipelineConfig = PipelineConfig()
    backtest: BacktestConfig = BacktestConfig()
    output_dir: Path = Path("results")

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ExperimentConfig":
        """Load an ExperimentConfig from a YAML file."""
        import yaml
        with open(path) as fh:
            return cls.model_validate(yaml.safe_load(fh))
