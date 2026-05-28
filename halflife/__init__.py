"""
halflife — signal half-life estimation for microstructure research.

Core classes:
    SignalDecayEstimator  — fits IC decay curve, estimates half-life
    AdaptiveWeighter      — online signal weighting via Kalman filter
    CrowdingDetector      — separates regime change from crowding decay

Quick example::

    from halflife import SignalDecayEstimator

    est = SignalDecayEstimator(signal=sig, returns=ret)
    result = est.fit()
    print(result.half_life)
    result.plot()
"""

from halflife.estimators import SignalDecayEstimator, DecayResult
from halflife.adaptive import AdaptiveWeighter
from halflife.crowding import CrowdingDetector
from halflife.features import OrderFlowImbalance, TradeSigns, ShortTermReversal

__version__ = "0.1.0"
__all__ = [
    "SignalDecayEstimator",
    "DecayResult",
    "AdaptiveWeighter",
    "CrowdingDetector",
    "OrderFlowImbalance",
    "TradeSigns",
    "ShortTermReversal",
]
