"""Insight detectors. Import ``DEFAULT_DETECTORS`` to get the standard set."""

from .decline_shield import DeclineShield
from .overspend_alert import OverspendAlert
from .subscription_radar import SubscriptionRadar
from .fx_fee_leakage import FxFeeLeakage
from .cashflow_forecast import CashflowForecast
from .peer_benchmarking import PeerBenchmarking


def default_detectors():
    """The detectors that power the dashboard, in a sensible order.

    Decline Shield and Overspend Alert lead: both are grounded in the
    strongest *real* signals in the data and are preventive/proactive. The
    rest are supporting context tiles.
    """
    return [
        DeclineShield(),
        OverspendAlert(),
        SubscriptionRadar(),
        CashflowForecast(),
        PeerBenchmarking(),
        FxFeeLeakage(),
    ]


DEFAULT_DETECTORS = default_detectors

__all__ = [
    "DeclineShield",
    "OverspendAlert",
    "SubscriptionRadar",
    "FxFeeLeakage",
    "CashflowForecast",
    "PeerBenchmarking",
    "default_detectors",
    "DEFAULT_DETECTORS",
]
