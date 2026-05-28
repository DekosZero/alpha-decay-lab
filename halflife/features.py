"""
features.py
===========
Canonical microstructure signal features.

These are the signals whose decay we benchmark in Phase 3.
Each class follows the same interface: __init__ takes raw trade/quote data,
.compute() returns a pd.Series of signal values aligned to a time index.

Data format assumed (from Binance or LOBSTER):
    trades : pd.DataFrame with columns [timestamp, price, qty, side]
              side = 1 for buyer-initiated, -1 for seller-initiated
    quotes : pd.DataFrame with columns [timestamp, bid, ask, bid_size, ask_size]

TODO (Phase 2 work items for each class):
    - OrderFlowImbalance: add volume-weighted variant (VWAP-OFI)
    - TradeSigns: implement Lee-Ready algorithm for TAQ data (no explicit side)
    - ShortTermReversal: experiment with different lookback windows
    - Add: DepthImbalance, CancellationRate, TradeIntensity
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class OrderFlowImbalance:
    """
    Order Flow Imbalance (OFI) — the canonical microstructure signal.

    OFI measures the excess of buyer-initiated volume over seller-initiated
    volume in a rolling window. Positive OFI → price pressure upward.

    Reference: Cont, Kukanov & Stoikov (2014), "The Price Impact of Order
    Book Events", Journal of Financial Econometrics.

    Parameters
    ----------
    trades : pd.DataFrame
        Columns: [timestamp, price, qty, side]. side = +1 buy, -1 sell.
    window : int
        Number of trades in the rolling OFI window.

    TODO (Phase 2):
        - Implement multi-level LOB version (OFI across top N levels)
        - Normalise by ADV to make cross-asset comparable
    """

    def __init__(self, trades: pd.DataFrame, window: int = 100) -> None:
        required = {"timestamp", "price", "qty", "side"}
        if not required.issubset(trades.columns):
            raise ValueError(f"trades must have columns: {required}")
        self.trades = trades.copy()
        self.window = window

    def compute(self) -> pd.Series:
        """
        Compute rolling OFI.

        Returns
        -------
        pd.Series indexed by timestamp, values in [-1, 1] (normalised).
        """
        df = self.trades.set_index("timestamp").sort_index()
        signed_vol = df["qty"] * df["side"]

        # Rolling sum of signed volume, normalised by total volume
        buy_vol = signed_vol.clip(lower=0).rolling(self.window).sum()
        sell_vol = signed_vol.clip(upper=0).abs().rolling(self.window).sum()
        total_vol = buy_vol + sell_vol

        ofi = (buy_vol - sell_vol) / total_vol.replace(0, np.nan)
        ofi.name = f"ofi_w{self.window}"
        return ofi


class TradeSigns:
    """
    Trade sign autocorrelation signal.

    Persistent buyer-initiated or seller-initiated activity predicts
    short-horizon price continuation. A run of buy trades → positive signal.

    Parameters
    ----------
    trades : pd.DataFrame
        Columns: [timestamp, price, qty, side].
    window : int
        Rolling window for sign autocorrelation.

    TODO (Phase 2):
        - Implement Lee-Ready algorithm to infer side from quote midpoint
          for datasets (like TAQ) that don't provide explicit aggressor side
    """

    def __init__(self, trades: pd.DataFrame, window: int = 50) -> None:
        required = {"timestamp", "side"}
        if not required.issubset(trades.columns):
            raise ValueError(f"trades must have columns: {required}")
        self.trades = trades.copy()
        self.window = window

    def compute(self) -> pd.Series:
        """
        Compute rolling mean trade sign as the signal.

        A positive value means recent trades were predominantly buyer-initiated.
        """
        df = self.trades.set_index("timestamp").sort_index()
        sign_ma = df["side"].rolling(self.window).mean()
        sign_ma.name = f"trade_sign_w{self.window}"
        return sign_ma


class ShortTermReversal:
    """
    Short-term price reversal signal.

    Over-reaction to information causes transient price dislocations that
    mean-revert. Negative lagged return → positive expected forward return.

    Parameters
    ----------
    prices : pd.Series
        Mid-price or last-trade price, indexed by time.
    lookback : int
        Number of periods for the return used as signal.

    Notes
    -----
    This is the simplest possible reversal signal. In practice you would
    condition on volume, time-of-day, and exclude earnings/event windows.
    """

    def __init__(self, prices: pd.Series, lookback: int = 5) -> None:
        if not isinstance(prices, pd.Series):
            raise TypeError("prices must be a pd.Series")
        self.prices = prices.copy()
        self.lookback = lookback

    def compute(self) -> pd.Series:
        """
        Return the negative of the lookback-period log return.

        Signal = -log(P_t / P_{t-lookback})
        Positive signal → expect reversion upward.
        """
        log_ret = np.log(self.prices / self.prices.shift(self.lookback))
        signal = -log_ret  # negative: we expect reversal
        signal.name = f"reversal_lb{self.lookback}"
        return signal
