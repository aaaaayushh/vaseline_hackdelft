"""Peer Benchmarking — how a user's spend compares to similar users.

Cohorts are defined by demographics (``age_group`` × ``region``, falling back
to ``age_group`` alone when a cell is too small). For each cohort × category we
build the distribution of *per-user monthly spend* and report where the user
sits (ratio to the cohort median + percentile rank). We surface the categories
where the user is most above their peers.

All cohort baselines are computed once in ``fit`` over the full population, so
``detect`` is a cheap lookup per user.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..base import EngineContext, InsightDetector
from ..contract import Insight

_EXCLUDE = {"Income & Refunds", "Fees & Charges", "Cash & ATM"}


class PeerBenchmarking(InsightDetector):
    type = "peer_benchmarking"

    def __init__(self, min_cohort: int = 30, surface_ratio: float = 1.3,
                 surface_percentile: float = 0.75, max_ratio: float = 5.0,
                 min_cohort_median: float = 20.0, min_user_monthly: float = 30.0):
        self.min_cohort = min_cohort
        self.surface_ratio = surface_ratio
        self.surface_percentile = surface_percentile
        # Guard against the small-denominator artifact: a near-zero cohort
        # median turns a normal user into "38× peers". Require a meaningful
        # cohort median and user spend, and cap the headline ratio.
        self.max_ratio = max_ratio
        self.min_cohort_median = min_cohort_median
        self.min_user_monthly = min_user_monthly
        self._bench: pd.DataFrame | None = None  # indexed by owner_id

    def fit(self, ctx: EngineContext) -> None:
        df = ctx.df
        spend = df[df["amount_signed_gbp"] > 0]
        spend = spend[~spend["category"].isin(_EXCLUDE)]

        # per-user active months -> monthly spend per (user, category)
        active_months = df.groupby("owner_id")["month"].nunique().rename("active_months")
        per_uc = (
            spend.groupby(["owner_id", "category"])["amount_signed_gbp"]
            .sum().rename("total").reset_index()
            .merge(active_months, on="owner_id")
        )
        per_uc["user_monthly"] = per_uc["total"] / per_uc["active_months"].clip(lower=1)

        # attach cohort keys (one row per user)
        demo = (
            df.groupby("owner_id")[["age_group", "region"]].first().reset_index()
        )
        per_uc = per_uc.merge(demo, on="owner_id")

        # choose cohort granularity per (age_group, region) cell, falling back
        # to age_group alone for small cells.
        cell_sizes = demo.groupby(["age_group", "region"]).size().rename("n").reset_index()
        big = cell_sizes[cell_sizes["n"] >= self.min_cohort][["age_group", "region"]]
        big_set = set(map(tuple, big.itertuples(index=False)))

        def cohort_label(row):
            if (row["age_group"], row["region"]) in big_set:
                return f"{row['age_group']} · {row['region']}"
            return f"{row['age_group']} · all regions"

        per_uc["cohort"] = per_uc.apply(cohort_label, axis=1)

        # cohort × category stats + per-user percentile within the cohort
        grp = per_uc.groupby(["cohort", "category"])["user_monthly"]
        per_uc["cohort_median"] = grp.transform("median")
        per_uc["cohort_p75"] = grp.transform(lambda s: s.quantile(0.75))
        per_uc["cohort_size"] = grp.transform("size")
        per_uc["percentile"] = grp.rank(pct=True)
        per_uc["ratio"] = per_uc["user_monthly"] / per_uc["cohort_median"].replace(0, np.nan)

        self._bench = per_uc.set_index("owner_id").sort_index()

    def detect(self, user_df: pd.DataFrame, ctx: EngineContext) -> list[Insight]:
        if self._bench is None:
            raise RuntimeError("PeerBenchmarking.fit must run before detect")
        user_id = str(user_df["owner_id"].iloc[0])
        if user_id not in self._bench.index:
            return []
        rows = self._bench.loc[[user_id]]

        cats = []
        for r in rows.itertuples():
            cats.append({
                "category": r.category,
                "user_monthly": round(float(r.user_monthly), 2),
                "cohort_median": round(float(r.cohort_median), 2),
                "ratio": round(float(r.ratio), 2) if pd.notna(r.ratio) else None,
                "percentile": round(float(r.percentile), 3),
            })
        cats.sort(key=lambda c: (c["ratio"] or 0), reverse=True)

        cohort = str(rows["cohort"].iloc[0])
        cohort_size = int(rows["cohort_size"].max())

        # categories where the user clearly outspends peers — but only on a
        # statistically meaningful base (real cohort median + real user spend),
        # so we don't surface a divide-by-near-zero artifact.
        flagged = [c for c in cats
                   if (c["ratio"] or 0) >= self.surface_ratio
                   and (c["ratio"] or 0) <= self.max_ratio
                   and c["percentile"] >= self.surface_percentile
                   and c["cohort_median"] >= self.min_cohort_median
                   and c["user_monthly"] >= self.min_user_monthly]

        if not flagged:
            # still emit an info card so the dashboard can show the comparison
            top = cats[0] if cats else None
            sev = 0.15
            title = "You're in line with your peers"
            explanation = (
                f"Your category spend tracks the typical {cohort} user."
                if top else "Not enough spend to benchmark yet."
            )
        else:
            top = flagged[0]
            sev = min(1.0, 0.4 + min(0.6, (top["ratio"] - 1) * 0.5))
            others = (f" (and {len(flagged) - 1} other categor"
                      f"{'ies' if len(flagged) > 2 else 'y'})") if len(flagged) > 1 else ""
            title = (f"{top['ratio']:.1f}× peers on {top['category']}")
            explanation = (
                f"You spend about £{top['user_monthly']:.0f}/mo on "
                f"{top['category']} — roughly {top['ratio']:.1f}× the £"
                f"{top['cohort_median']:.0f} typical for {cohort} users"
                f" (top {round((1 - top['percentile']) * 100)}%){others}."
            )

        return [Insight(
            type=self.type,
            user_id=user_id,
            title=title,
            explanation=explanation,
            severity=sev,
            payload={
                "cohort": cohort,
                "cohort_size": cohort_size,
                "categories": cats,
                "flagged": flagged,
            },
        )]
