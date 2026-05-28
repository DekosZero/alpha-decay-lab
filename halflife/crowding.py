"""
crowding.py
===========
Separate crowding-driven alpha decay from regime change.

This is the hardest research problem in the project. A signal's IC can
drop for two fundamentally different reasons:
    1. Crowding: too many people trade it, arbitraging away the edge
    2. Regime change: the underlying microstructure changed (new market
       structure, different participant mix, new regulations)

Why it matters: if the decay is crowding-driven, rotating to a correlated
signal is useless. If it's regime-driven, the signal may revive.

This module is a skeleton. Phase 2 work involves:
    - Implementing the Granger causality test
    - Building the double/debiased ML estimator for causal effect
    - Finding a usable crowding proxy (AUM flows, put/call ratios,
      position data from SEC 13F filings)

References
----------
- Granger, C.W.J. (1969). Investigating causal relations by econometric
  models and cross-spectral methods. Econometrica, 37(3), 424–438.
- Chernozhukov et al. (2018). Double/debiased machine learning for
  treatment and structural parameters. Econometrics Journal, 21(1), C1–C68.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class CrowdingDiagnostic:
    """
    Output of CrowdingDetector.diagnose().

    Attributes
    ----------
    likely_cause : str
        'crowding', 'regime_change', or 'ambiguous'.
    crowding_score : float
        In [0, 1]. Higher → more consistent with crowding hypothesis.
    granger_p_value : float
        p-value from Granger causality test (crowding proxy → IC decay).
        Low p-value supports crowding hypothesis.
    notes : str
        Human-readable interpretation.
    """
    likely_cause: str
    crowding_score: float
    granger_p_value: float
    notes: str


class CrowdingDetector:
    """
    Diagnose whether IC decay is caused by crowding or regime change.

    Parameters
    ----------
    ic_series : pd.Series
        Rolling IC over time.
    crowding_proxy : pd.Series, optional
        A time series that proxies crowding intensity — e.g. net flows into
        factor ETFs, put/call ratio, or estimated AUM in similar strategies.
        If None, the detector runs only structural tests (no causal test).

    Usage
    -----
    detector = CrowdingDetector(ic_series=result.ic_curve, crowding_proxy=flows)
    diagnostic = detector.diagnose()
    print(diagnostic.likely_cause)
    print(diagnostic.notes)

    TODO (Phase 2):
        - Implement _granger_test properly using statsmodels VAR
        - Implement _regime_change_test using CUSUM or Bai-Perron breakpoints
        - Implement double/debiased ML estimator for causal IC → crowding
        - Source a crowding proxy: SEC 13F flows or ETF flow data (free via ETF.com)
    """

    def __init__(
        self,
        ic_series: pd.Series,
        crowding_proxy: pd.Series | None = None,
    ) -> None:
        self.ic_series = ic_series.dropna()
        self.crowding_proxy = crowding_proxy

    def diagnose(self) -> CrowdingDiagnostic:
        """
        Run crowding vs regime-change diagnostics.

        Returns
        -------
        CrowdingDiagnostic
        """
        regime_score = self._regime_change_score()
        granger_p = self._granger_test() if self.crowding_proxy is not None else float("nan")

        # Heuristic scoring (replace with proper model in Phase 2)
        if not np.isnan(granger_p) and granger_p < 0.05:
            crowding_score = 0.8
            cause = "crowding"
            notes = (
                f"Granger test rejects non-causality (p={granger_p:.3f}). "
                "IC decay is temporally preceded by crowding proxy increases. "
                "Suggests arbitrage crowding is the primary driver."
            )
        elif regime_score > 0.6:
            crowding_score = 0.2
            cause = "regime_change"
            notes = (
                f"Structural break score = {regime_score:.2f}. "
                "IC dropped sharply around a detectable break point. "
                "Consider whether a market structure change coincides."
            )
        else:
            crowding_score = 0.5
            cause = "ambiguous"
            notes = (
                "Cannot distinguish crowding from regime change with available data. "
                "Consider adding a crowding proxy series (ETF flows, 13F data)."
            )

        return CrowdingDiagnostic(
            likely_cause=cause,
            crowding_score=crowding_score,
            granger_p_value=granger_p,
            notes=notes,
        )

    def _regime_change_score(self) -> float:
        """
        Simple CUSUM-based structural break score.

        Returns a value in [0, 1]. High value = evidence of structural break.

        TODO (Phase 2): replace with Bai-Perron multiple breakpoint test
        from statsmodels.
        """
        try:
            ic = self.ic_series.values
            n = len(ic)
            if n < 20:
                return 0.0
            cumsum = np.cumsum(ic - ic.mean())
            cusum_range = cumsum.max() - cumsum.min()
            # Normalise by std * sqrt(n) — rough CUSUM significance scaling
            sigma = ic.std()
            if sigma < 1e-9:
                return 0.0
            score = cusum_range / (sigma * np.sqrt(n))
            # Clip to [0, 1] heuristically — values > 1.36 are significant at 5%
            return float(min(score / 1.36, 1.0))
        except Exception as e:
            warnings.warn(f"Regime change score failed: {e}")
            return 0.0

    def _granger_test(self) -> float:
        """
        Granger causality: does crowding_proxy Granger-cause IC decay?

        TODO (Phase 2): implement using statsmodels.tsa.stattools.grangercausalitytests
        Currently returns a placeholder NaN.

        Sketch of implementation:
            from statsmodels.tsa.stattools import grangercausalitytests
            combined = pd.concat([self.ic_series, self.crowding_proxy], axis=1).dropna()
            result = grangercausalitytests(combined, maxlag=5, verbose=False)
            # Extract min p-value across lags
            p_values = [result[lag][0]['ssr_ftest'][1] for lag in result]
            return min(p_values)
        """
        warnings.warn(
            "Granger test not yet implemented. "
            "See _granger_test docstring for Phase 2 implementation sketch."
        )
        return float("nan")
