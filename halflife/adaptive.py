"""
adaptive.py
===========
Online signal weighting — down-weight a dying signal in real time.

The core problem: by the time your backtest tells you a signal has decayed,
you've been trading on a dead signal for weeks. These classes implement
online methods that update signal weights as new IC observations arrive.

Classes
-------
AdaptiveWeighter : Kalman filter over rolling IC → adaptive weight
ExponentialForgetter : Simple EMA-based IC tracker

TODO (Phase 2 work items):
    - AdaptiveWeighter: tune process noise Q and observation noise R
      from data rather than hardcoding (EM algorithm or grid search)
    - Add multi-signal version: weight a portfolio of signals jointly
    - Add regime-conditioned weighting (use CrowdingDetector to switch)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class AdaptiveWeighter:
    """
    Track a signal's IC over time using a Kalman filter, and output
    a weight in [0, 1] proportional to the filtered IC estimate.

    The state model is:
        IC_t = IC_{t-1} + w_t,    w_t ~ N(0, Q)   (random walk)
        y_t  = IC_t + v_t,        v_t ~ N(0, R)   (noisy observation)

    Parameters
    ----------
    process_noise : float
        Q — how much the true IC is allowed to drift per period.
        Higher Q → filter tracks faster but is noisier. Default 1e-4.
    obs_noise : float
        R — assumed variance of a single IC observation.
        Higher R → filter trusts observations less. Default 1e-2.
    min_weight : float
        Floor weight — never fully zero out a signal. Default 0.0.

    TODO (Phase 2): tune Q and R from data via EM algorithm.
    """

    def __init__(
        self,
        process_noise: float = 1e-4,
        obs_noise: float = 1e-2,
        min_weight: float = 0.0,
    ) -> None:
        self.Q = process_noise
        self.R = obs_noise
        self.min_weight = min_weight

        # Kalman state
        self._x = 0.0   # filtered IC estimate
        self._P = 1.0   # estimate variance

    def update(self, ic_observation: float) -> float:
        """
        Update the Kalman filter with a new IC observation.

        Parameters
        ----------
        ic_observation : float
            The most recent rolling IC value.

        Returns
        -------
        float
            Recommended signal weight in [min_weight, 1].
        """
        if np.isnan(ic_observation):
            return self._ic_to_weight(self._x)

        # Predict
        P_pred = self._P + self.Q

        # Update
        K = P_pred / (P_pred + self.R)       # Kalman gain
        self._x = self._x + K * (ic_observation - self._x)
        self._P = (1 - K) * P_pred

        return self._ic_to_weight(self._x)

    def fit_series(self, ic_series: pd.Series) -> pd.Series:
        """
        Apply the Kalman filter to a full IC series, return a weight series.

        Useful for backtesting: what weight would the filter have assigned
        at each point in time?

        Parameters
        ----------
        ic_series : pd.Series
            Rolling IC values, indexed by time.

        Returns
        -------
        pd.Series of weights, same index.
        """
        weights = []
        for obs in ic_series.values:
            w = self.update(obs)
            weights.append(w)
        return pd.Series(weights, index=ic_series.index, name="adaptive_weight")

    def _ic_to_weight(self, ic: float) -> float:
        """Map filtered IC to a weight. Clips negatives to min_weight."""
        raw = max(ic, 0.0)  # only trade when IC is positive
        w = min(raw / 0.1, 1.0)  # normalise: IC of 0.10 → full weight
        return max(w, self.min_weight)

    @property
    def current_ic_estimate(self) -> float:
        return self._x

    @property
    def current_uncertainty(self) -> float:
        return self._P


class ExponentialForgetter:
    """
    Simple exponential moving average of IC as a weight tracker.

    Simpler than the Kalman filter but often good enough. Equivalent to
    AdaptiveWeighter in the limit of very high observation noise.

    Parameters
    ----------
    alpha : float
        EMA smoothing factor. Higher → faster forgetting. Default 0.05.
    """

    def __init__(self, alpha: float = 0.05) -> None:
        if not 0 < alpha < 1:
            raise ValueError("alpha must be in (0, 1)")
        self.alpha = alpha
        self._ema: float | None = None

    def update(self, ic_observation: float) -> float:
        if np.isnan(ic_observation):
            return max(self._ema or 0.0, 0.0)
        if self._ema is None:
            self._ema = ic_observation
        else:
            self._ema = self.alpha * ic_observation + (1 - self.alpha) * self._ema
        return max(self._ema, 0.0)

    def fit_series(self, ic_series: pd.Series) -> pd.Series:
        weights = [self.update(v) for v in ic_series.values]
        return pd.Series(weights, index=ic_series.index, name="ema_weight")
