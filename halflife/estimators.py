"""
estimators.py
=============
Core classes for measuring signal decay.

The main deliverable here is SignalDecayEstimator. It takes a signal series
and a forward-returns series, computes rolling Information Coefficient (IC),
fits a decay model, and returns an estimated half-life with confidence interval.

TODO (Phase 2 work items):
    - Implement power-law decay fitting alongside exponential
    - Add bootstrap CI for half-life estimate
    - Add multi-signal comparison (list of signals → ranked decay table)
    - Handle non-overlapping forward returns correctly (shift by horizon)
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
from scipy.stats import spearmanr
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class DecayResult:
    """
    Output of SignalDecayEstimator.fit().

    Attributes
    ----------
    half_life : float
        Estimated number of periods until IC halves. NaN if fit failed.
    decay_type : str
        'exponential' or 'power_law'.
    ic_curve : pd.Series
        Rolling Spearman IC indexed by time.
    ic_mean : float
        Mean IC over the full sample.
    ic_ir : float
        IC information ratio (mean / std). Rough measure of signal consistency.
    half_life_ci : tuple[float, float]
        95% confidence interval on half_life. (NaN, NaN) if not computed.
    fit_params : dict
        Raw parameters from the decay curve fit.
    """
    half_life: float
    decay_type: str
    ic_curve: pd.Series
    ic_mean: float
    ic_ir: float
    half_life_ci: tuple[float, float] = field(default=(float("nan"), float("nan")))
    fit_params: dict = field(default_factory=dict)

    def plot(self, ax=None, title: str | None = None) -> plt.Axes:
        """
        Plot the IC decay curve with half-life annotation.

        Parameters
        ----------
        ax : matplotlib Axes, optional
        title : str, optional

        Returns
        -------
        matplotlib Axes
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 4))

        ax.plot(self.ic_curve.values, color="#378ADD", lw=1.5, label="Rolling IC")
        ax.axhline(0, color="#888780", lw=0.8, ls="--")
        ax.axhline(self.ic_mean, color="#1D9E75", lw=1, ls=":", label=f"Mean IC = {self.ic_mean:.4f}")

        if not np.isnan(self.half_life):
            ax.axvline(self.half_life, color="#D85A30", lw=1.2, ls="--",
                       label=f"Half-life ≈ {self.half_life:.1f} periods")

        ax.set_xlabel("Periods")
        ax.set_ylabel("Spearman IC")
        ax.set_title(title or "IC decay curve")
        ax.legend(fontsize=9)
        plt.tight_layout()
        return ax

    def summary(self) -> str:
        lines = [
            f"Signal decay summary",
            f"--------------------",
            f"Decay type    : {self.decay_type}",
            f"Half-life     : {self.half_life:.1f} periods",
            f"IC mean       : {self.ic_mean:.4f}",
            f"IC IR         : {self.ic_ir:.3f}",
            f"CI (95%)      : ({self.half_life_ci[0]:.1f}, {self.half_life_ci[1]:.1f})",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main estimator
# ---------------------------------------------------------------------------

class SignalDecayEstimator:
    """
    Estimate the half-life of a trading signal's Information Coefficient.

    Parameters
    ----------
    signal : pd.Series
        The trading signal, indexed by time. Higher value = stronger long.
    returns : pd.Series
        Forward returns, same index as signal. Should already be shifted
        (i.e. returns.iloc[t] is the return realised *after* signal.iloc[t]).
    window : int
        Rolling window for IC calculation. Default 63 (approx 3 months daily).
    decay_type : 'exponential' | 'power_law' | 'auto'
        Decay model to fit. 'auto' fits both and picks lower AIC.

    Notes
    -----
    This is the skeleton for Phase 2. The _fit_exponential and _fit_power_law
    methods need to be implemented. The rolling IC computation is complete.

    Example
    -------
    >>> est = SignalDecayEstimator(signal=sig, returns=ret, window=63)
    >>> result = est.fit()
    >>> print(result.half_life)
    >>> result.plot()
    """

    def __init__(
        self,
        signal: pd.Series,
        returns: pd.Series,
        window: int = 63,
        decay_type: Literal["exponential", "power_law", "auto"] = "exponential",
    ) -> None:
        if not isinstance(signal, pd.Series):
            raise TypeError("signal must be a pd.Series")
        if not isinstance(returns, pd.Series):
            raise TypeError("returns must be a pd.Series")
        if len(signal) != len(returns):
            raise ValueError("signal and returns must have the same length")
        if window >= len(signal):
            raise ValueError("window must be smaller than signal length")

        self.signal = signal.copy()
        self.returns = returns.copy()
        self.window = window
        self.decay_type = decay_type

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def fit(self) -> DecayResult:
        """
        Fit the decay model and return a DecayResult.

        Returns
        -------
        DecayResult
        """
        ic_curve = self._rolling_ic()
        ic_mean = float(ic_curve.mean())
        ic_std = float(ic_curve.std())
        ic_ir = ic_mean / ic_std if ic_std > 0 else float("nan")

        dtype = self.decay_type
        if dtype == "auto":
            # TODO (Phase 2): fit both, compare AIC, pick winner
            dtype = "exponential"

        if dtype == "exponential":
            half_life, params = self._fit_exponential(ic_curve)
        else:
            half_life, params = self._fit_power_law(ic_curve)

        return DecayResult(
            half_life=half_life,
            decay_type=dtype,
            ic_curve=ic_curve,
            ic_mean=ic_mean,
            ic_ir=ic_ir,
            fit_params=params,
        )

    def rolling_ic(self) -> pd.Series:
        """Return the rolling Spearman IC series directly."""
        return self._rolling_ic()

    # ------------------------------------------------------------------
    # Internal: rolling IC
    # ------------------------------------------------------------------

    def _rolling_ic(self) -> pd.Series:
        """
        Compute rolling Spearman rank correlation between signal and returns.

        Spearman is preferred over Pearson because signal distributions are
        often fat-tailed and we care about rank, not magnitude.
        """
        ic_values = []
        idx = []

        sig_arr = self.signal.values
        ret_arr = self.returns.values

        for i in range(self.window, len(sig_arr)):
            s_window = sig_arr[i - self.window : i]
            r_window = ret_arr[i - self.window : i]

            # Drop NaNs within the window
            mask = ~(np.isnan(s_window) | np.isnan(r_window))
            if mask.sum() < 10:
                ic_values.append(float("nan"))
            else:
                rho, _ = spearmanr(s_window[mask], r_window[mask])
                ic_values.append(float(rho))

            idx.append(self.signal.index[i])

        return pd.Series(ic_values, index=idx, name="rolling_ic")

    # ------------------------------------------------------------------
    # Internal: decay model fitting
    # ------------------------------------------------------------------

    def _fit_exponential(self, ic_curve: pd.Series) -> tuple[float, dict]:
        """
        Fit IC(t) = A * exp(-lambda * t) to the rolling IC series.

        Returns (half_life, params_dict).
        half_life = ln(2) / lambda

        TODO (Phase 2):
            - Use non-linear least squares via scipy.optimize.curve_fit
            - Handle cases where IC is negative or non-monotone
            - Add bootstrap for CI
            - Return AIC for model comparison
        """
        # Placeholder: compute naive half-life as number of periods until
        # IC drops below 50% of its peak. Replace with curve fitting in Phase 2.
        try:
            clean = ic_curve.dropna()
            if len(clean) < 5:
                return float("nan"), {}

            peak = clean.iloc[0]
            if abs(peak) < 1e-6:
                return float("nan"), {}

            half_target = peak * 0.5
            crossings = clean[
                (clean - half_target).abs() == (clean - half_target).abs().min()
            ]
            naive_hl = float(crossings.index[0]) if not crossings.empty else float("nan")

            # TODO: replace naive_hl with proper curve_fit below
            # def exp_decay(t, A, lam): return A * np.exp(-lam * t)
            # t = np.arange(len(clean))
            # popt, pcov = curve_fit(exp_decay, t, clean.values, p0=[peak, 0.01], maxfev=5000)
            # lam = popt[1]
            # half_life = np.log(2) / lam if lam > 0 else float("nan")

            return naive_hl, {"method": "naive_peak_crossing", "peak_ic": peak}

        except Exception as e:
            warnings.warn(f"Exponential fit failed: {e}")
            return float("nan"), {}

    def _fit_power_law(self, ic_curve: pd.Series) -> tuple[float, dict]:
        """
        Fit IC(t) = A * t^(-alpha) to the rolling IC series.

        Power-law decay is theoretically motivated for signals where
        information diffuses slowly rather than being immediately arbitraged.

        TODO (Phase 2): implement via log-log regression or scipy curve_fit.
        """
        # Stub — returns placeholder until Phase 2 implements this
        warnings.warn("Power-law fitting not yet implemented. Falling back to exponential.")
        return self._fit_exponential(ic_curve)
