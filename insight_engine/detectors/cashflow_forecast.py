"""Cashflow Forecast — upcoming charges and projected net flow.

We don't have account balances, so we forecast *net cashflow* over a horizon
(default 30 days), not a balance. Three ingredients:

1. **Upcoming known charges** — recurring outflows (subscriptions / bills)
   whose next due date falls inside the horizon, predicted from cadence.
2. **Recurring income** — regular inflows (credits) projected forward.
3. **Discretionary burn** — a daily run-rate from recent history, applied to
   the remaining days.

Net = projected inflow − (known charges + discretionary burn).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import EngineContext, InsightDetector
from ..contract import Insight

_CADENCE_TOL = 0.4  # interval CV must be below this to count as recurring


def _recurring_streams(df: pd.DataFrame, min_charges: int = 3):
    """Yield (merchant, cadence_days, typical_amount, last_date) for regular
    streams in ``df`` (already filtered to one direction of flow)."""
    for merchant, g in df.groupby("merchant_id"):
        if len(g) < min_charges:
            continue
        g = g.sort_values("created_date")
        days = g["created_date"].values.astype("datetime64[D]")
        gaps = np.diff(days).astype("timedelta64[D]").astype(float)
        gaps = gaps[gaps > 0]
        if len(gaps) < min_charges - 1:
            continue
        median_gap = float(np.median(gaps))
        if median_gap <= 0 or (np.std(gaps) / median_gap) > _CADENCE_TOL:
            continue
        if not (5 <= median_gap <= 400):
            continue
        yield (
            str(merchant),
            median_gap,
            float(g["txn_amount_gbp"].abs().median()),
            g["created_date"].iloc[-1],
            str(g["category"].mode().iloc[0]),
        )


class CashflowForecast(InsightDetector):
    type = "cashflow_forecast"

    def __init__(self, horizon_days: int = 30, recent_window_days: int = 60):
        self.horizon_days = horizon_days
        self.recent_window_days = recent_window_days

    def detect(self, user_df: pd.DataFrame, ctx: EngineContext) -> list[Insight]:
        as_of = ctx.as_of
        horizon_end = as_of + pd.Timedelta(days=self.horizon_days)

        payments = user_df[user_df["type"].eq("CARD_PAYMENT")]
        income = user_df[user_df["type"].isin(["CARD_CREDIT", "CARD_REFUND"])]
        if payments.empty:
            return []

        # 1) upcoming known charges from recurring outflow streams.
        upcoming = []
        for merchant, cad, amt, last, cat in _recurring_streams(payments):
            due = last + pd.Timedelta(days=cad)
            while due < as_of:
                due += pd.Timedelta(days=cad)
            while due <= horizon_end:
                upcoming.append({"merchant": merchant, "date": due,
                                 "amount": round(amt, 2), "category": cat})
                due += pd.Timedelta(days=cad)
        upcoming.sort(key=lambda c: c["date"])
        known_outflow = round(sum(c["amount"] for c in upcoming), 2)

        # 2) projected recurring income.
        upcoming_income = []
        for merchant, cad, amt, last, _cat in _recurring_streams(income):
            due = last + pd.Timedelta(days=cad)
            while due < as_of:
                due += pd.Timedelta(days=cad)
            while due <= horizon_end:
                upcoming_income.append({"source": merchant, "date": due,
                                        "amount": round(amt, 2)})
                due += pd.Timedelta(days=cad)
        projected_inflow = round(sum(c["amount"] for c in upcoming_income), 2)

        # 3) discretionary burn = recent non-recurring daily spend run-rate.
        recent = payments[payments["created_date"] >= as_of - pd.Timedelta(days=self.recent_window_days)]
        recurring_merchants = {m for m, *_ in _recurring_streams(payments)}
        discretionary = recent[~recent["merchant_id"].isin(recurring_merchants)]
        window = max((as_of - recent["created_date"].min()).days, 1) if not recent.empty else 1
        daily_burn = float(discretionary["txn_amount_gbp"].abs().sum()) / window
        projected_discretionary = round(daily_burn * self.horizon_days, 2)

        projected_outflow = round(known_outflow + projected_discretionary, 2)
        projected_net = round(projected_inflow - projected_outflow, 2)

        # Severity: a projected deficit is the thing to flag, scaled by size.
        recent_monthly = float(payments["txn_amount_gbp"].abs().sum()) / max(
            (as_of - payments["created_date"].min()).days, 1) * 30.4
        sev = 0.2
        if projected_net < 0 and recent_monthly > 0:
            sev = min(1.0, 0.5 + min(0.5, abs(projected_net) / recent_monthly))

        title = (f"Next {self.horizon_days} days: "
                 f"{'-' if projected_net < 0 else '+'}£{abs(projected_net):.0f} projected")
        explanation = (
            f"Over the next {self.horizon_days} days we expect about "
            f"£{projected_outflow:.0f} out"
            + (f" and £{projected_inflow:.0f} in" if projected_inflow else "")
            + f", including {len(upcoming)} known recurring charge"
            f"{'s' if len(upcoming) != 1 else ''} (£{known_outflow:.0f})."
        )

        return [Insight(
            type=self.type,
            user_id=str(user_df["owner_id"].iloc[0]),
            title=title,
            explanation=explanation,
            severity=sev,
            payload={
                "as_of": as_of,
                "horizon_days": self.horizon_days,
                "projected_outflow": projected_outflow,
                "projected_inflow": projected_inflow,
                "projected_net": projected_net,
                "known_recurring_outflow": known_outflow,
                "projected_discretionary": projected_discretionary,
                "daily_burn_rate": round(daily_burn, 2),
                "recent_monthly_spend": round(recent_monthly, 2),
                "upcoming_charges": upcoming,
                "upcoming_income": upcoming_income,
            },
        )]
