"""InsightEngine — orchestrates detectors, ranking, and dashboard assembly.

Typical use::

    df = load_enriched("output/df_clean_clean.parquet")
    engine = InsightEngine(df)              # fits population context once
    payload = engine.dashboard(user_id)     # JSON-ready dict for the RN app
"""

from __future__ import annotations

import pandas as pd

from .base import EngineContext, InsightDetector
from .contract import Insight
from .detectors import default_detectors
from .analytics import SpendingAnalytics
from . import ranking


class InsightEngine:
    def __init__(self, df: pd.DataFrame,
                 detectors: list[InsightDetector] | None = None,
                 as_of: pd.Timestamp | None = None):
        self.df = df
        self.detectors = detectors if detectors is not None else default_detectors()
        self.ctx = EngineContext(df, as_of=as_of)
        self.ctx.fit(self.detectors)
        # group index for fast per-user slicing
        self._groups = df.groupby("owner_id")
        # deterministic stats query layer for the dashboard charts
        self.analytics = SpendingAnalytics(df, as_of=self.ctx.as_of)

    # -- core ------------------------------------------------------------- #

    def run_user(self, user_id: str) -> list[Insight]:
        """Run every detector for one user. Detectors that raise are skipped so
        one bad detector can't take down the dashboard."""
        try:
            user_df = self._groups.get_group(user_id)
        except KeyError:
            return []
        out: list[Insight] = []
        for det in self.detectors:
            try:
                out.extend(det.detect(user_df, self.ctx))
            except Exception as exc:  # pragma: no cover - defensive
                import logging
                logging.getLogger(__name__).warning(
                    "detector %s failed for user %s: %s", det.type, user_id, exc)
        return out

    # -- dashboard -------------------------------------------------------- #

    def dashboard(self, user_id: str, *, top_n: int | None = None,
                  chart_months: int = 6) -> dict:
        """Assemble the per-user dashboard payload for the React Native app.

        Structure::

            {
              "user_id": ...,
              "generated_at": ...,           # = engine as_of
              "hero": <top insight or null>, # drives the push notification
              "insights": [ <insight cards in priority order> ],
              "sections": { <type>: <insight card> },  # for fixed dashboard tiles
              "charts": {
                "spending_history": {...},   # history bars + forecast ghost bar
                "category_momentum": {...},  # MoM rate of change per category
              }
            }
        """
        insights = ranking.rank(self.run_user(user_id), top_n=top_n)
        cards = [i.to_dict() for i in insights]
        sections = {c["type"]: c for c in cards}  # one card per detector type
        return {
            "user_id": user_id,
            "generated_at": self.ctx.as_of.isoformat(),
            "hero": cards[0] if cards else None,
            "insights": cards,
            "sections": sections,
            "charts": {
                "spending_history": self.analytics.spending_history(user_id, months=chart_months),
                "category_momentum": self.analytics.category_momentum(user_id, months=chart_months),
            },
        }

    # -- analytics accessors (deterministic chart queries) ---------------- #

    def spending_history(self, user_id: str, months: int = 6) -> dict:
        """History bars + forecast ghost bar for the spending chart page."""
        return self.analytics.spending_history(user_id, months=months)

    def category_momentum(self, user_id: str, months: int = 6) -> dict:
        """Month-on-month rate of change per category (goal tracking)."""
        return self.analytics.category_momentum(user_id, months=months)

    def run_all(self, user_ids: list[str] | None = None) -> dict[str, list[Insight]]:
        ids = user_ids if user_ids is not None else list(self._groups.groups.keys())
        return {uid: self.run_user(uid) for uid in ids}
