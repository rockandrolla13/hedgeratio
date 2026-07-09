"""Phase 1 tests for the Gaussian predict/update stages and the Pipeline recursion."""
import numpy as np

from hrl.config import PipelineConfig
from hrl.data.synthetic import generate
from hrl.eval import metrics
from hrl.filters.baselines import VanillaKalmanRW
from hrl.filters.pipeline import Pipeline
from hrl.filters.stages.gaussian import GaussianUpdateStage, PredictStage
from hrl.models.linear_ssm import LinearGaussianSSM


def _vanilla_pipeline() -> Pipeline:
    model = LinearGaussianSSM(q_alpha=1e-7, q_beta=1e-5)
    cfg = PipelineConfig(stages=["predict", "update"], weight_fn="none",
                         noise_model="fixed", r=4e-4, p0=10.0)
    return Pipeline.from_config(cfg, model)


def test_import_and_instantiate():
    """Smoke: stages and model construct without error."""
    assert PredictStage().name == "predict"
    assert GaussianUpdateStage().name == "update"
    assert LinearGaussianSSM().dim == 2


def test_pipeline_rejects_no_transition():
    """Composition validation: a pipeline with no transition stage is rejected."""
    import pytest
    with pytest.raises(ValueError):
        Pipeline([GaussianUpdateStage()], LinearGaussianSSM())


def test_s1_beta_converges_to_truth():
    """On the static-beta Gaussian DGP, KF beta converges to the true beta."""
    sample = generate("S1", n_steps=2500, seed=0)
    result = _vanilla_pipeline().run(sample.y, sample.x)

    true_beta = sample.beta_true[-1]
    tail = slice(-250, None)
    assert abs(np.mean(result.beta[tail]) - true_beta) < 0.05
    # Second half is a much better fit than the first (genuine convergence).
    n = len(result.beta)
    rmse_first = metrics.beta_rmse(result.beta[: n // 2], sample.beta_true[: n // 2])
    rmse_second = metrics.beta_rmse(result.beta[n // 2:], sample.beta_true[n // 2:])
    assert rmse_second < rmse_first


def test_pipeline_matches_independent_kf():
    """The composed vanilla Pipeline equals the standalone VanillaKalmanRW to 1e-10."""
    sample = generate("S1", n_steps=1000, seed=1)
    pipe = _vanilla_pipeline()
    kf = VanillaKalmanRW(q_alpha=1e-7, q_beta=1e-5, r=4e-4, p0=10.0)
    rp = pipe.run(sample.y, sample.x)
    rk = kf.run(sample.y, sample.x)
    assert np.allclose(rp.beta, rk.beta, atol=1e-10)
    assert np.allclose(rp.alpha, rk.alpha, atol=1e-10)
    assert np.allclose(rp.S, rk.S, atol=1e-10)


def test_innovation_variance_stabilises():
    """S_t reaches an approximate steady state (EWRLS-equivalent gain)."""
    sample = generate("S1", n_steps=2500, seed=2)
    result = _vanilla_pipeline().run(sample.y, sample.x)
    late = result.S[-500:]
    assert np.std(late) / np.mean(late) < 0.5  # roughly stationary innovation variance
