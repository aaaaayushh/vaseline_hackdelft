"""Spending analytics — deterministic stats for the dashboard charts.

This is the *query layer* the React Native app calls to draw charts, distinct
from the insight detectors (which find noteworthy events). Everything here is a
plain, explainable computation over the enriched table — no ML, no LLM — so the
numbers are auditable.

Primary surfaces:

* ``monthly_spend_by_category`` — stacked bar chart: last N months, one stack
  segment per category.
* ``forecast_next_month``       — the "ghost bar": projected next-month spend,
  per category, with a per-category forecast.
* ``spending_history``          — history bars + ghost bar in one payload (the
  whole chart page in a single call).
* ``category_momentum``         — month-on-month rate of change per category
  (the goal-tracking signal).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .taxonomy import CATEGORIES

# Money-in isn't expenditure; everything else (incl. fees, ATM) is spend.
_NON_SPEND = {"Income & Refunds"}


def _month_window(as_of: pd.Timestamp, months: int) -> pd.PeriodIndex:
    """The last ``months`` calendar months ending at ``as_of`` (inclusive)."""
    end = as_of.to_period("M")
    return pd.period_range(end - (months - 1), end, freq="M")


def _forecast_series(values: np.ndarray) -> float:
    """Forecast the next value of a short monthly series.

    Robust + explainable: least-squares linear trend over the available months
    (damped to half the slope so a single spike can't blow up the projection),
    falling back to the recent mean when there's too little history. Clipped at
    zero — you can't spend negative.
    """
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return 0.0
    if len(v) <= 2:
        return float(max(0.0, v.mean()))
    x = np.arange(len(v))
    slope, intercept = np.polyfit(x, v, 1)
    trend = intercept + slope * len(v)
    # damp toward the recent mean to stay robust on noisy synthetic data
    recent_mean = float(v[-3:].mean())
    forecast = 0.5 * trend + 0.5 * recent_mean
    return float(max(0.0, forecast))


class SpendingAnalytics:
    """Per-user spending statistics for the dashboard.

    Stateless apart from the dataframe + reference date; cheap to construct.
    Reuse one instance across requests.
    """

    def __init__(self, df: pd.DataFrame, as_of: pd.Timestamp | None = None):
        self.df = df
        self.as_of = pd.to_datetime(as_of) if as_of is not None else df["created_date"].max()
        self._groups = df.groupby("owner_id")
        # canonical color/order key for the frontend
        self.category_order = [c for c in CATEGORIES if c not in _NON_SPEND]

    # ------------------------------------------------------------------ #
    def _user_spend(self, user_id: str) -> pd.DataFrame | None:
        try:
            u = self._groups.get_group(user_id)
        except KeyError:
            return None
        spend = u[(u["amount_signed_gbp"] > 0) & (~u["category"].isin(_NON_SPEND))].copy()
        if spend.empty:
            return None
        spend["ym"] = spend["created_date"].dt.to_period("M")
        return spend

    def _pivot(self, spend: pd.DataFrame, window: pd.PeriodIndex) -> pd.DataFrame:
        """months × categories matrix of summed spend, aligned to ``window``."""
        piv = (
            spend.groupby(["ym", "category"])["amount_signed_gbp"].sum().unstack(fill_value=0.0)
        )
        piv = piv.reindex(index=window, fill_value=0.0)
        # order columns by canonical taxonomy, keep only those the user has
        cols = [c for c in self.category_order if c in piv.columns]
        return piv[cols]

    # ------------------------------------------------------------------ #
    def monthly_spend_by_category(self, user_id: str, months: int = 6) -> dict:
        """Stacked-bar data: per-month spend split by category.

        Returns::

            {
              "user_id", "months": ["2025-07", ...],
              "categories": ["Groceries", ...],         # in stack order
              "series": {"Groceries": [m0, m1, ...], ...},
              "totals": [per-month total],
              "category_order": [...]                    # canonical, for colors
            }
        """
        spend = self._user_spend(user_id)
        window = _month_window(self.as_of, months)
        if spend is None:
            return {"user_id": user_id, "months": [str(m) for m in window],
                    "categories": [], "series": {}, "totals": [0.0] * len(window),
                    "category_order": self.category_order}
        piv = self._pivot(spend, window)
        return {
            "user_id": user_id,
            "months": [str(m) for m in window],
            "categories": list(piv.columns),
            "series": {c: [round(float(v), 2) for v in piv[c].values] for c in piv.columns},
            "totals": [round(float(v), 2) for v in piv.sum(axis=1).values],
            "category_order": self.category_order,
        }

    # ------------------------------------------------------------------ #
    def forecast_next_month(self, user_id: str, lookback: int = 6) -> dict:
        """The ghost bar: forecast next month's spend, per category.

        Forecasts each category independently from its recent monthly series,
        then sums for the headline total.
        """
        spend = self._user_spend(user_id)
        next_month = (self.as_of.to_period("M") + 1)
        if spend is None:
            return {"user_id": user_id, "month": str(next_month), "is_forecast": True,
                    "by_category": {}, "total": 0.0, "category_order": self.category_order}
        window = _month_window(self.as_of, lookback)
        piv = self._pivot(spend, window)
        by_cat = {c: round(_forecast_series(piv[c].values), 2) for c in piv.columns}
        by_cat = {c: v for c, v in by_cat.items() if v > 0}
        return {
            "user_id": user_id,
            "month": str(next_month),
            "is_forecast": True,
            "by_category": by_cat,
            "total": round(sum(by_cat.values()), 2),
            "method": "damped linear trend over recent months",
            "category_order": self.category_order,
        }

    # ------------------------------------------------------------------ #
    def spending_history(self, user_id: str, months: int = 6) -> dict:
        """The whole chart page in one call: history bars + the forecast ghost
        bar, sharing one category/color ordering."""
        history = self.monthly_spend_by_category(user_id, months=months)
        forecast = self.forecast_next_month(user_id, lookback=months)
        # union of categories so the stack/colors line up across all bars
        cats = list(dict.fromkeys(history["categories"] + list(forecast["by_category"])))
        cats = [c for c in self.category_order if c in cats]
        return {
            "user_id": user_id,
            "category_order": self.category_order,
            "categories": cats,
            "history": {
                "months": history["months"],
                "series": history["series"],
                "totals": history["totals"],
            },
            "forecast": {
                "month": forecast["month"],
                "by_category": forecast["by_category"],
                "total": forecast["total"],
                "is_forecast": True,
            },
        }

    # ------------------------------------------------------------------ #
    def category_momentum(self, user_id: str, months: int = 6) -> dict:
        """Month-on-month rate of change per category — the goal-tracking signal.

        For each category reports last vs previous month (% change) and the
        trend over the window (avg monthly % change via slope/mean), plus a
        direction label.
        """
        spend = self._user_spend(user_id)
        window = _month_window(self.as_of, months)
        empty = {"user_id": user_id, "window": [str(m) for m in window], "categories": []}
        if spend is None:
            return empty
        piv = self._pivot(spend, window)
        if piv.shape[0] < 2:
            return empty

        out = []
        for c in piv.columns:
            v = piv[c].values.astype(float)
            last, prev = float(v[-1]), float(v[-2])
            # month-on-month change; distinguish a brand-new category (prev=0)
            # from a percentage move so the UI doesn't show a fake "+100%".
            if prev > 0:
                mom = round((last / prev - 1) * 100, 1)
                direction = "up" if mom > 10 else "down" if mom < -10 else "flat"
            elif last > 0:
                mom = None
                direction = "new"
            else:
                mom = None
                direction = "flat"
            # window trend: slope as % of the window mean
            mean = v.mean()
            slope = float(np.polyfit(np.arange(len(v)), v, 1)[0]) if len(v) >= 2 else 0.0
            trend_pct = (slope / mean * 100) if mean > 0 else 0.0
            out.append({
                "category": c,
                "last_month": round(last, 2),
                "prev_month": round(prev, 2),
                "mom_pct_change": mom,
                "trend_pct_per_month": round(trend_pct, 1),
                "direction": direction,
                "monthly": [round(float(x), 2) for x in v],
            })
        # biggest movers first (by absolute MoM change; new categories after)
        out.sort(key=lambda r: abs(r["mom_pct_change"] or 0), reverse=True)
        return {
            "user_id": user_id,
            "window": [str(m) for m in window],
            "categories": out,
        }
