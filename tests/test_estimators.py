"""
tests/test_estimators.py
========================
Tests for the core SignalDecayEstimator.

Run with: pytest tests/ -v
"""

import numpy as np
import pandas as pd
import pytest

from halflife.estimators import SignalDecayEstimator, DecayResult
from halflife.utils import forward_returns, winsorise, zscore


# ---------------------------------------------------------------------------
# Fixtures: synthetic signals with known properties
# ---------------------------------------------------------------------------

def make_decaying_signal(n: int = 500, true_half_life: float = 50.0, noise: float = 0.3):
    """
    Generate a synthetic signal whose IC decays exponentially.

    The true IC starts at 0.10 and halves every `true_half_life` periods.
    """
    rng = np.random.default_rng(42)
    t = np.arange(n)
    lam = np.log(2) / true_half_life
    ic_t = 0.10 * np.exp(-lam * t)  # true IC at each time step

    # Generate returns that have the right correlation with the signal
    signal = rng.standard_normal(n)
    returns = ic_t * signal + np.sqrt(1 - ic_t**2) * rng.standard_normal(n)

    sig = pd.Series(signal, name="signal")
    ret = pd.Series(returns, name="returns")
    return sig, ret, true_half_life


def make_null_signal(n: int = 500):
    """Signal with zero IC — should return half-life close to NaN or very small."""
    rng = np.random.default_rng(99)
    sig = pd.Series(rng.standard_normal(n))
    ret = pd.Series(rng.standard_normal(n))
    return sig, ret


# ---------------------------------------------------------------------------
# Tests: input validation
# ---------------------------------------------------------------------------

class TestInputValidation:
    def test_rejects_non_series_signal(self):
        with pytest.raises(TypeError):
            SignalDecayEstimator(signal=[1, 2, 3], returns=pd.Series([1, 2, 3]))

    def test_rejects_mismatched_lengths(self):
        with pytest.raises(ValueError):
            SignalDecayEstimator(
                signal=pd.Series([1, 2, 3]),
                returns=pd.Series([1, 2])
            )

    def test_rejects_window_too_large(self):
        with pytest.raises(ValueError):
            SignalDecayEstimator(
                signal=pd.Series(range(10)),
                returns=pd.Series(range(10)),
                window=15,
            )


# ---------------------------------------------------------------------------
# Tests: rolling IC computation
# ---------------------------------------------------------------------------

class TestRollingIC:
    def test_ic_series_length(self):
        sig, ret, _ = make_decaying_signal()
        est = SignalDecayEstimator(sig, ret, window=63)
        ic = est.rolling_ic()
        assert len(ic) == len(sig) - 63

    def test_ic_values_in_range(self):
        sig, ret, _ = make_decaying_signal()
        est = SignalDecayEstimator(sig, ret, window=63)
        ic = est.rolling_ic().dropna()
        assert (ic >= -1.0).all() and (ic <= 1.0).all()

    def test_positive_signal_has_positive_mean_ic(self):
        """A signal that genuinely predicts returns should have positive mean IC."""
        sig, ret, _ = make_decaying_signal(n=500, true_half_life=200.0)  # slow decay
        est = SignalDecayEstimator(sig, ret, window=63)
        ic = est.rolling_ic().dropna()
        assert ic.mean() > 0, f"Expected positive IC, got {ic.mean():.4f}"

    def test_null_signal_has_near_zero_ic(self):
        sig, ret = make_null_signal(n=500)
        est = SignalDecayEstimator(sig, ret, window=63)
        ic = est.rolling_ic().dropna()
        assert abs(ic.mean()) < 0.15, f"Null signal IC too large: {ic.mean():.4f}"


# ---------------------------------------------------------------------------
# Tests: DecayResult output
# ---------------------------------------------------------------------------

class TestDecayResult:
    def test_fit_returns_decay_result(self):
        sig, ret, _ = make_decaying_signal()
        est = SignalDecayEstimator(sig, ret, window=63)
        result = est.fit()
        assert isinstance(result, DecayResult)

    def test_result_has_ic_curve(self):
        sig, ret, _ = make_decaying_signal()
        result = SignalDecayEstimator(sig, ret).fit()
        assert isinstance(result.ic_curve, pd.Series)
        assert len(result.ic_curve) > 0

    def test_result_has_decay_type(self):
        sig, ret, _ = make_decaying_signal()
        result = SignalDecayEstimator(sig, ret).fit()
        assert result.decay_type in ("exponential", "power_law")

    def test_summary_returns_string(self):
        sig, ret, _ = make_decaying_signal()
        result = SignalDecayEstimator(sig, ret).fit()
        s = result.summary()
        assert isinstance(s, str)
        assert "Half-life" in s


# ---------------------------------------------------------------------------
# Tests: utilities
# ---------------------------------------------------------------------------

class TestUtils:
    def test_forward_returns_no_lookahead(self):
        """The last `horizon` values should be NaN."""
        prices = pd.Series(np.cumsum(np.random.randn(100)) + 100)
        fwd = forward_returns(prices, horizon=5)
        assert fwd.iloc[-5:].isna().all()
        assert fwd.iloc[:-5].notna().all()

    def test_winsorise_clips_extremes(self):
        s = pd.Series([-1000.0] + list(range(98)) + [1000.0], dtype=float)
        w = winsorise(s, lower=0.01, upper=0.99)
        assert w.iloc[0] > -1000
        assert w.iloc[-1] < 1000

    def test_zscore_mean_zero(self):
        s = pd.Series(np.random.randn(200) * 5 + 10.0)
        z = zscore(s)
        assert abs(z.mean()) < 1e-10
        assert abs(z.std() - 1.0) < 1e-10
