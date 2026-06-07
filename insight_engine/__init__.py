"""Spending IQ — Insight Engine.

A modular library of insight detectors over the categorized transaction table.
Each detector emits typed `Insight` objects; a ranking layer surfaces the
top, non-spammy insights; a dashboard builder packages them as JSON for the
React Native app.

Public API::

    from insight_engine import InsightEngine, load_enriched

    df = load_enriched("output/df_clean_clean.parquet")
    engine = InsightEngine(df)
    dashboard = engine.dashboard(user_id)        # dict, ready to JSON-serialize
"""

from .contract import Insight, Severity, load_enriched, ENRICHED_COLUMNS
from .base import InsightDetector, EngineContext
from .analytics import SpendingAnalytics
from .engine import InsightEngine

__all__ = [
    "Insight",
    "Severity",
    "load_enriched",
    "ENRICHED_COLUMNS",
    "InsightDetector",
    "EngineContext",
    "SpendingAnalytics",
    "InsightEngine",
]
