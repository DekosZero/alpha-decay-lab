"""
utils.py
========
Shared utility functions used across the halflife package.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def forward_returns(prices: pd.Series, horizon: int = 1) -> pd.Series:
    """
    Compute forward log returns, properly shifted to avoid lookahead bias.

    forward_returns.iloc[t] = log(prices.iloc[t + horizon] / prices.iloc[t])

    The last `horizon` values will be NaN (no forward price available).

    Parameters
    ----------
    prices : pd.Series
    horizon : int

    Returns
    -------
    pd.Series
    """
    log_ret = np.log(prices / prices.shift(horizon))
    return log_ret.shift(-horizon)


def winsorise(series: pd.Series, lower: float = 0.01, upper: float = 0.99) -> pd.Series:
    """
    Clip a series at its lower and upper quantiles. Reduces IC contamination
    from outlier returns around events, earnings, or data errors.
    """
    lo = series.quantile(lower)
    hi = series.quantile(upper)
    return series.clip(lo, hi)


def zscore(series: pd.Series, window: int | None = None) -> pd.Series:
    """
    Cross-sectional or rolling z-score normalisation.

    If window is None, use the full-sample mean/std (cross-sectional style).
    If window is given, use rolling mean/std.
    """
    if window is None:
        return (series - series.mean()) / series.std()
    else:
        mu = series.rolling(window).mean()
        sigma = series.rolling(window).std()
        return (series - mu) / sigma.replace(0, np.nan)


def ic_summary(signal: pd.Series, returns: pd.Series) -> dict:
    """
    Quick full-sample IC diagnostics without a rolling window.

    Useful for a first-pass sanity check on a new signal.

    Returns
    -------
    dict with keys: ic_mean, ic_std, ic_ir, ic_t_stat, n_obs
    """
    from scipy.stats import spearmanr, ttest_1samp

    mask = ~(signal.isna() | returns.isna())
    s = signal[mask].values
    r = returns[mask].values

    if len(s) < 10:
        return {"error": "Too few observations"}

    rho, _ = spearmanr(s, r)
    n = len(s)

    return {
        "ic_mean": rho,
        "n_obs": n,
        "ic_t_stat": rho * np.sqrt(n - 2) / np.sqrt(1 - rho**2 + 1e-12),
    }
