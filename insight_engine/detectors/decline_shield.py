"""Decline Shield — predict & prevent the next insufficient-balance decline.

This is the **preventive** hero. Where every other detector describes money
that already moved, Decline Shield sees a problem coming and offers to stop it.

It fuses two signals:

* **Decline history (real).** The strongest, most universal signal in the
  data: 19% of transactions decline and 75% of those are *insufficient
  balance*; ~77% of users hit at least one. We count a user's
  insufficient-balance declines, how recent they are, and whether they recur
  across months (chronic).
* **Upcoming recurring charge (predicted).** The next due date of a recurring
  outflow (subscription / bill), predicted from cadence — the same machinery
  the cashflow detector uses. A known charge landing while the user is in a
  decline-prone stretch is exactly the moment to nudge a top-up.

We have **no account balance** in the data, so we never invent a shortfall
figure. The insight is honest: "you've been declined N times for insufficient
balance, and £X is due on <date> — top up to avoid another one."
"""

from __future__ import annotations

import pandas as pd

from ..base import EngineContext, InsightDetector
from ..contract import Insight
from .cashflow_forecast import _recurring_streams

_INSUFFICIENT = "INSUFFICIENT BALANCE"


class DeclineShield(InsightDetector):
    type = "decline_shield"

    def __init__(self, horizon_days: int = 14, recent_window_days: int = 30,
                 chronic_min: int = 3):
        self.horizon_days = horizon_days
        self.recent_window_days = recent_window_days
        self.chronic_min = chronic_min  # declines that make the pattern "chronic"

    def detect(self, user_df: pd.DataFrame, ctx: EngineContext) -> list[Insight]:
        # Gracefully no-op if the substrate lacks the decline columns.
        if "state" not in user_df.columns:
            return []

        as_of = ctx.as_of
        state = user_df["state"].astype(str).str.upper()
        declined = user_df[state.eq("DECLINED")]
        if declined.empty:
            return []

        reason = declined.get("declined_reason_category")
        if reason is not None:
            insufficient = declined[reason.astype(str).str.upper().eq(_INSUFFICIENT)]
        else:
            insufficient = declined.iloc[0:0]

        n_declines = int(len(declined))
        n_insufficient = int(len(insufficient))
        if n_insufficient == 0:
            return []

        # Recency + chronicity of the insufficient-balance declines.
        recent_cut = as_of - pd.Timedelta(days=self.recent_window_days)
        recent = insufficient[insufficient["created_date"] >= recent_cut]
        n_recent = int(len(recent))
        months_hit = int(insufficient["created_date"].dt.to_period("M").nunique())
        chronic = n_insufficient >= self.chronic_min or months_hit >= 3
        last_decline = insufficient["created_date"].max()

        # The merchants the user actually got declined at (for the explanation).
        top_merchants = (
            insufficient["transaction_merchants_name"].astype(str)
            .value_counts().head(3).index.tolist()
        )

        # --- predicted next recurring charge (the thing to protect) -------- #
        completed = user_df[
            user_df["type"].eq("CARD_PAYMENT") & state.eq("COMPLETED")
        ]
        horizon_end = as_of + pd.Timedelta(days=self.horizon_days)
        next_charge = None
        for _mid, name, cad, amt, last, cat in _recurring_streams(completed):
            due = last + pd.Timedelta(days=cad)
            while due < as_of:
                due += pd.Timedelta(days=cad)
            if due <= horizon_end:
                cand = {"merchant": name, "amount": round(amt, 2),
                        "date": due, "category": cat,
                        "days_until": int((due - as_of).days)}
                if next_charge is None or cand["date"] < next_charge["date"]:
                    next_charge = cand

        # --- severity ------------------------------------------------------ #
        # Real, universal, actionable -> ranks high. Chronic + an imminent
        # charge is the worst case.
        sev = min(1.0,
                  0.35
                  + 0.1 * min(n_insufficient, 4)
                  + (0.15 if chronic else 0.0)
                  + (0.1 if n_recent > 0 else 0.0)
                  + (0.15 if next_charge is not None else 0.0))

        decline_rate = round(n_declines / max(len(user_df), 1) * 100, 1)

        # --- copy ---------------------------------------------------------- #
        when = "this month" if n_recent else "recently"
        title = f"Avoid your next decline · {n_insufficient} so far"
        bits = [
            f"You've had {n_insufficient} payment"
            f"{'s' if n_insufficient != 1 else ''} declined for insufficient "
            f"balance ({months_hit} month{'s' if months_hit != 1 else ''} running)"
            if chronic else
            f"You've had {n_insufficient} payment"
            f"{'s' if n_insufficient != 1 else ''} declined for insufficient balance"
        ]
        if next_charge is not None:
            day = "tomorrow" if next_charge["days_until"] <= 1 else \
                  f"in {next_charge['days_until']} days"
            bits.append(
                f"and {next_charge['merchant']} (£{next_charge['amount']:.2f}) "
                f"is due {day}"
            )
        explanation = " ".join(bits) + ". Top up to avoid another one."

        actions = [{
            "action": "top_up_reminder",
            "reason": "prevent_insufficient_balance_decline",
            "before_date": next_charge["date"] if next_charge else None,
            "merchant": next_charge["merchant"] if next_charge else None,
            "amount_due": next_charge["amount"] if next_charge else None,
        }]

        return [Insight(
            type=self.type,
            user_id=str(user_df["owner_id"].iloc[0]),
            title=title,
            explanation=explanation,
            severity=sev,
            payload={
                "n_declines": n_declines,
                "n_insufficient_balance": n_insufficient,
                "n_recent": n_recent,
                "months_with_declines": months_hit,
                "chronic": chronic,
                "decline_rate_pct": decline_rate,
                "last_decline": last_decline,
                "declined_at_merchants": top_merchants,
                "next_recurring_charge": next_charge,
            },
            actions=actions,
        )]
