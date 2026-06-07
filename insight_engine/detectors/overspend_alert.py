"""Overspend Alert — the insight the brief names, and DESIGN's hero (§5.2).

For each user × category we learn a **personal baseline** from prior complete
months (median spend) and a **robust dispersion** (median absolute deviation).
We then take the *current month* and project it to a full-month run rate from
how much of the month has elapsed, and fire when that projection lands
materially above the user's own baseline — *early enough to act* ("you're 40%
over your usual Dining, and it's only the 18th").

Why this is the honest hero:

* It needs clean categories to exist at all — you can't alert on "Dining"
  until it's unified from Eating Places / Fast Food / Drinking Places.
* It's *personal* (your baseline, not a population average), so it doesn't
  punish someone for whom high Dining is normal.
* Monthly per-category spend is volatile on this data (median CV ≈ 0.68), so
  we only fire on **well-sampled, stable** categories with a robust threshold —
  no false alarms on naturally spiky categories.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import EngineContext, InsightDetector
from ..contract import Insight

# Categories that aren't discretionary "overspend" in a budgeting sense.
_EXCLUDE = {"Income & Refunds", "Fees & Charges", "Cash & ATM"}


class OverspendAlert(InsightDetector):
    type = "overspend_alert"

    def __init__(self, min_history_months: int = 3, min_baseline_gbp: float = 25.0,
                 overshoot: float = 0.30, min_robust_z: float = 1.0,
                 min_month_progress: float = 0.2):
        self.min_history_months = min_history_months
        self.min_baseline_gbp = min_baseline_gbp      # ignore trivial categories
        self.overshoot = overshoot                    # +30% over baseline to fire
        self.min_robust_z = min_robust_z              # and clear of normal noise
        self.min_month_progress = min_month_progress  # need enough of the month

    def detect(self, user_df: pd.DataFrame, ctx: EngineContext) -> list[Insight]:
        as_of = ctx.as_of

        # Real spend only: positive signed amount, discretionary categories, and
        # — crucially — *completed* (a declined charge never spent money).
        spend = user_df[(user_df["amount_signed_gbp"] > 0)
                        & (~user_df["category"].isin(_EXCLUDE))]
        if "state" in spend.columns:
            spend = spend[spend["state"].astype(str).str.upper().eq("COMPLETED")]
        if spend.empty:
            return []

        spend = spend.assign(ym=spend["created_date"].dt.to_period("M"))
        cur_month = as_of.to_period("M")

        # How far through the current month are we? (drives the projection)
        days_in_month = as_of.days_in_month
        progress = min(as_of.day / days_in_month, 1.0)
        if progress < self.min_month_progress:
            return []  # too early in the month to say anything useful

        piv = spend.groupby(["ym", "category"])["amount_signed_gbp"].sum().unstack(fill_value=0.0)
        if cur_month not in piv.index:
            return []
        prior = piv.loc[piv.index < cur_month]
        if len(prior) < self.min_history_months:
            return []

        candidates = []
        for cat in piv.columns:
            hist = prior[cat].values.astype(float)
            hist = hist[hist > 0]  # months the user actually spent in this cat
            if len(hist) < self.min_history_months:
                continue
            baseline = float(np.median(hist))
            if baseline < self.min_baseline_gbp:
                continue
            mad = float(np.median(np.abs(hist - baseline))) or (0.1 * baseline)

            mtd = float(piv.loc[cur_month, cat])
            projected = mtd / progress
            overshoot = projected / baseline - 1.0
            robust_z = (projected - baseline) / (1.4826 * mad)

            if overshoot >= self.overshoot and robust_z >= self.min_robust_z:
                candidates.append({
                    "category": cat,
                    "baseline": round(baseline, 2),
                    "month_to_date": round(mtd, 2),
                    "projected": round(projected, 2),
                    "overshoot_pct": round(overshoot * 100, 0),
                    "robust_z": round(robust_z, 2),
                    "month_progress_pct": round(progress * 100, 0),
                    "history_months": int(len(hist)),
                })

        if not candidates:
            return []

        candidates.sort(key=lambda c: c["overshoot_pct"], reverse=True)
        top = candidates[0]

        # severity scales with how far over baseline the projection is
        sev = min(1.0, 0.45 + min(0.5, (top["overshoot_pct"] / 100.0) * 0.6))

        partial = progress < 0.95
        day_clause = (f", and it's only the {as_of.day}{_ordinal(as_of.day)}"
                      if partial else "")
        title = f"{int(top['overshoot_pct'])}% over your usual {top['category']}"
        if partial:
            explanation = (
                f"You're on track to spend about £{top['projected']:.0f} on "
                f"{top['category']} this month — roughly {int(top['overshoot_pct'])}% "
                f"above your usual £{top['baseline']:.0f}{day_clause}. "
                f"You've spent £{top['month_to_date']:.0f} so far."
            )
        else:
            explanation = (
                f"You spent £{top['month_to_date']:.0f} on {top['category']} this "
                f"month — about {int(top['overshoot_pct'])}% above your usual "
                f"£{top['baseline']:.0f}."
            )

        actions = [{
            "action": "set_budget",
            "category": top["category"],
            "suggested_monthly": top["baseline"],
            "reason": "projected_overspend",
        }]

        return [Insight(
            type=self.type,
            user_id=str(user_df["owner_id"].iloc[0]),
            title=title,
            explanation=explanation,
            severity=sev,
            payload={
                "as_of": as_of,
                "month": str(cur_month),
                "month_progress_pct": top["month_progress_pct"],
                "top_category": top["category"],
                "overspending_categories": candidates,
            },
            actions=actions,
        )]


def _ordinal(n: int) -> str:
    if 11 <= (n % 100) <= 13:
        return "th"
    return {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
