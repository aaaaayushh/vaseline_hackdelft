"""Insight detectors. Import ``DEFAULT_DETECTORS`` to get the standard set."""

from .subscription_radar import SubscriptionRadar
from .fx_fee_leakage import FxFeeLeakage
from .cashflow_forecast import CashflowForecast
from .peer_benchmarking import PeerBenchmarking


def default_detectors():
    """The four detectors that power the dashboard, in a sensible order."""
    return [
        SubscriptionRadar(),
        FxFeeLeakage(),
        CashflowForecast(),
        PeerBenchmarking(),
    ]


DEFAULT_DETECTORS = default_detectors

__all__ = [
    "SubscriptionRadar",
    "FxFeeLeakage",
    "CashflowForecast",
    "PeerBenchmarking",
    "default_detectors",
    "DEFAULT_DETECTORS",
]
