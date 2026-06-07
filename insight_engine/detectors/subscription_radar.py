"""Subscription Radar — new / forgotten / price-hiked recurring charges.

DESIGN.md owns that synthetic recurrence is noisy (even merchants literally
named ``abonnement`` have jittery timestamps), so we accept a subscription via
*either* of two signals:

* **cadence path** — a regular interval (weekly / monthly / yearly) with low
  interval variability. High confidence.
* **merchant path** — the merchant is categorized *Digital & Subscriptions* or
  named like a subscription (``abonnement``), charges repeatedly, and the
  amount is stable. Catches real subscriptions whose timing is noisy.

Detected subscriptions are classified ``new`` / ``price_hike`` / ``forgotten``
/ ``active``. Each carries a ``confidence`` so the dashboard/agent can hedge.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import EngineContext, InsightDetector
from ..contract import Insight

# Known cadences in days and a tolerance for matching the median interval.
_CADENCES = [
    ("weekly", 7, 2.5),
    ("biweekly", 14, 4),
    ("monthly", 30.4, 7),
    ("quarterly", 91.3, 15),
    ("yearly", 365, 40),
]


def _classify_cadence(median_gap: float) -> tuple[str, float] | None:
    best = None
    for name, days, tol in _CADENCES:
        if abs(median_gap - days) <= tol:
            score = abs(median_gap - days) / tol
            if best is None or score < best[2]:
                best = (name, days, score)
    if best is None:
        return None
    return best[0], best[1]


class SubscriptionRadar(InsightDetector):
    type = "subscription_radar"

    def __init__(self, min_charges: int = 3, max_cv: float = 0.6,
                 max_amount_cv: float = 0.4,
                 hike_threshold: float = 0.15, forgotten_max_amount: float = 15.0):
        self.min_charges = min_charges
        self.max_cv = max_cv                 # max coeff. of variation of intervals
        self.max_amount_cv = max_amount_cv   # amount stability for the merchant path
        self.hike_threshold = hike_threshold
        self.forgotten_max_amount = forgotten_max_amount

    @staticmethod
    def _is_subscription_merchant(g: pd.DataFrame, name: str) -> bool:
        from ..taxonomy import SUBSCRIPTION_CATEGORIES
        cat = str(g["category"].mode().iloc[0])
        return cat in SUBSCRIPTION_CATEGORIES or "abonnement" in name.lower()

    def detect(self, user_df: pd.DataFrame, ctx: EngineContext) -> list[Insight]:
        pay = user_df[user_df["type"].eq("CARD_PAYMENT")]
        if pay.empty:
            return []

        subs: list[dict] = []
        for merchant_id, g in pay.groupby("merchant_id"):
            if len(g) < self.min_charges:
                continue
            g = g.sort_values("created_date")
            name = str(g["transaction_merchants_name"].mode().iloc[0])
            days = g["created_date"].values.astype("datetime64[D]")
            gaps = np.diff(days).astype("timedelta64[D]").astype(float)
            gaps = gaps[gaps > 0]
            if len(gaps) < self.min_charges - 1:
                continue
            median_gap = float(np.median(gaps))
            cv = float(np.std(gaps) / median_gap) if median_gap else np.inf

            amounts = g["txn_amount_gbp"].abs()
            typical = float(amounts.median())
            amount_cv = float(amounts.std() / typical) if typical else np.inf
            first_seen = g["created_date"].iloc[0]
            last_seen = g["created_date"].iloc[-1]
            span_days = max((last_seen - first_seen).days, 1)

            # --- three acceptance paths ------------------------------------
            cad = _classify_cadence(median_gap)
            cadence_ok = cad is not None and cv <= self.max_cv
            merchant_ok = (
                self._is_subscription_merchant(g, name) and amount_cv <= self.max_amount_cv
            )
            # the Categorization Engine ships its own learned recurrence flag —
            # trust it as a signal when the amount is stable.
            flag_ok = (
                "is_recurring" in g.columns
                and bool(g["is_recurring"].mean() >= 0.5)
                and amount_cv <= self.max_amount_cv
            )
            if not (cadence_ok or merchant_ok or flag_ok):
                continue

            if cadence_ok:
                cadence_name, cadence_days = cad
                confidence = round(max(0.5, 1 - cv), 2)
                monthly_cost = typical * (30.4 / cadence_days)
            else:
                # noisy timing: estimate cadence/cost from observed frequency
                cadence_days = span_days / max(len(g) - 1, 1)
                named = _classify_cadence(cadence_days)
                cadence_name = named[0] if named else "irregular"
                # categorizer agreement lifts confidence
                base = 0.7 if flag_ok else 0.6
                confidence = round(max(0.4, base - amount_cv), 2)
                monthly_cost = float(amounts.sum()) / max(span_days / 30.4, 1)

            latest = float(amounts.iloc[-1])
            prior_typical = float(amounts.iloc[:-1].median())
            next_expected = last_seen + pd.Timedelta(days=cadence_days)

            status = "active"
            if prior_typical > 0 and latest >= prior_typical * (1 + self.hike_threshold):
                status = "price_hike"
            elif (ctx.as_of - first_seen).days <= cadence_days * 1.4:
                status = "new"
            elif typical <= self.forgotten_max_amount and len(g) >= 4:
                status = "forgotten"

            subs.append({
                "merchant": name,
                "merchant_id": str(merchant_id),
                "category": str(g["category"].mode().iloc[0]),
                "cadence": cadence_name,
                "cadence_days": round(cadence_days, 1),
                "amount": round(typical, 2),
                "latest_amount": round(latest, 2),
                "monthly_cost": round(monthly_cost, 2),
                "n_charges": int(len(g)),
                "first_seen": first_seen,
                "last_seen": last_seen,
                "next_expected": next_expected,
                "status": status,
                "confidence": confidence,
                "price_increase_pct": (
                    round((latest / prior_typical - 1) * 100, 1)
                    if status == "price_hike" and prior_typical else None
                ),
            })

        if not subs:
            return []

        subs.sort(key=lambda s: s["monthly_cost"], reverse=True)
        monthly_total = round(sum(s["monthly_cost"] for s in subs), 2)
        counts = {k: sum(s["status"] == k for s in subs)
                  for k in ("new", "price_hike", "forgotten", "active")}

        # Severity: price hikes and forgotten subs matter most; scale also with
        # how big the subscription bill is.
        sev = min(1.0,
                  0.4 * counts["price_hike"]
                  + 0.25 * counts["forgotten"]
                  + 0.15 * counts["new"]
                  + min(0.3, monthly_total / 100.0))

        headline_bits = []
        if counts["price_hike"]:
            headline_bits.append(f"{counts['price_hike']} price increase"
                                 f"{'s' if counts['price_hike'] > 1 else ''}")
        if counts["forgotten"]:
            headline_bits.append(f"{counts['forgotten']} you may have forgotten")
        if counts["new"]:
            headline_bits.append(f"{counts['new']} new")
        detail = "; ".join(headline_bits) if headline_bits else "all steady"

        title = f"{len(subs)} subscriptions · £{monthly_total:.0f}/mo"
        explanation = (
            f"We found {len(subs)} recurring charges totalling about "
            f"£{monthly_total:.0f} a month ({detail})."
        )

        actions = []
        for s in subs:
            if s["status"] in ("forgotten", "price_hike"):
                actions.append({
                    "action": "cancel_subscription",
                    "merchant": s["merchant"],
                    "monthly_cost": s["monthly_cost"],
                    "reason": s["status"],
                })

        return [Insight(
            type=self.type,
            user_id=str(user_df["owner_id"].iloc[0]),
            title=title,
            explanation=explanation,
            severity=sev,
            payload={
                "monthly_total": monthly_total,
                "annual_total": round(monthly_total * 12, 2),
                "count": len(subs),
                "status_counts": counts,
                "subscriptions": subs,
            },
            actions=actions,
        )]
