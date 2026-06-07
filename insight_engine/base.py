"""Detector interface and shared engine context.

Every insight is a pluggable detector implementing one interface
(``InsightDetector``). Population-level work that is expensive and shared
across users (e.g. cohort baselines for benchmarking) is computed once in
``EngineContext.fit`` and handed to every detector.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from .contract import Insight


class EngineContext:
    """Population-level context shared by all detectors, computed once.

    Holds the full enriched frame and the "as of" reference date (defaults to
    the last transaction in the data, since detectors reason about
    month-to-date run rates and upcoming charges).
    """

    def __init__(self, df: pd.DataFrame, as_of: pd.Timestamp | None = None):
        self.df = df
        self.as_of: pd.Timestamp = (
            pd.to_datetime(as_of) if as_of is not None else df["created_date"].max()
        )
        # Per-detector caches populated during fit().
        self.cache: dict[str, object] = {}

    def fit(self, detectors: "list[InsightDetector]") -> "EngineContext":
        for det in detectors:
            det.fit(self)
        return self


class InsightDetector(ABC):
    """Base class for all insight detectors.

    Subclasses set ``type`` and implement ``detect``. ``fit`` is optional and
    used for population-level precomputation (e.g. cohort baselines).
    """

    type: str = "base"

    def fit(self, ctx: EngineContext) -> None:  # noqa: B027 - intentional no-op
        """Optional population-level precomputation. Default: nothing."""

    @abstractmethod
    def detect(self, user_df: pd.DataFrame, ctx: EngineContext) -> list[Insight]:
        """Return zero or more insights for a single user's transactions."""
        raise NotImplementedError
