"""Phase 1 tests for the estimation + stability metrics."""
import numpy as np

from hrl.eval import metrics


def test_metric_callables_present():
    """The metric surface is importable."""
    for name in ("beta_rmse", "delta_beta_var", "turnover", "sign_flips",
                 "reconverge_time", "spread_adf", "ou_half_life", "pit_coverage"):
        assert callable(getattr(metrics, name))


def test_beta_rmse_zero_on_identity():
    """RMSE of a series against itself is zero."""
    b = np.array([1.0, 1.1, 0.9, 1.2])
    assert metrics.beta_rmse(b, b) == 0.0


def test_beta_rmse_known_value():
    """RMSE matches a hand-computed value."""
    hat = np.array([1.0, 2.0, 3.0])
    true = np.array([1.0, 2.0, 5.0])   # errors 0, 0, 2 -> rmse = sqrt(4/3)
    assert abs(metrics.beta_rmse(hat, true) - np.sqrt(4.0 / 3.0)) < 1e-12


def test_delta_beta_var_flags_churn():
    """A churning path has higher delta-beta variance than a smooth one."""
    smooth = np.linspace(1.0, 1.1, 100)
    churny = 1.0 + 0.1 * ((-1.0) ** np.arange(100))
    assert metrics.delta_beta_var(churny) > metrics.delta_beta_var(smooth)


def test_sign_flips_counts_reversals():
    """A zig-zag path counts one reversal per direction change."""
    zig = np.array([0.0, 1.0, 0.0, 1.0, 0.0])   # diffs +,-,+,- -> 3 flips
    assert metrics.sign_flips(zig) == 3
    assert metrics.sign_flips(np.arange(5.0)) == 0
