"""FX & Fee Leakage — money lost to fees and foreign-currency exposure.

On this synthetic data the txn-vs-bill spread is symmetric noise (it nets to
~zero), so the robust, honest leakage signal is the explicit
``fee_amount_gbp``. We treat any fee as leakage and flag fees charged on
home-currency (GBP) transactions as *avoidable* — there's no FX reason to pay
them. We also report foreign-currency exposure (share of spend and number of
currencies) as context, which is the uniquely-Revolut angle.
"""

from __future__ import annotations

import pandas as pd

from ..base import EngineContext, InsightDetector
from ..contract import Insight

HOME_CURRENCY = "GBP"


class FxFeeLeakage(InsightDetector):
    type = "fx_fee_leakage"

    def detect(self, user_df: pd.DataFrame, ctx: EngineContext) -> list[Insight]:
        fees = user_df[user_df["fee_amount_gbp"].fillna(0) != 0].copy()
        total_fees = round(float(user_df["fee_amount_gbp"].fillna(0).sum()), 2)

        # Foreign-currency exposure on actual spend.
        spend = user_df[user_df["type"].eq("CARD_PAYMENT")]
        total_spend = float(spend["txn_amount_gbp"].abs().sum())
        foreign = spend[spend["txn_currency"].ne(HOME_CURRENCY)]
        foreign_spend = float(foreign["txn_amount_gbp"].abs().sum())
        foreign_share = (foreign_spend / total_spend) if total_spend else 0.0
        currency_count = int(spend["txn_currency"].nunique())

        if total_fees == 0 and currency_count <= 1:
            return []

        # Avoidable: fees on home-currency transactions.
        home_fees = round(
            float(fees.loc[fees["txn_currency"].eq(HOME_CURRENCY), "fee_amount_gbp"].sum()),
            2,
        )

        # Annualise from the observed window.
        span_days = max((ctx.as_of - user_df["created_date"].min()).days, 1)
        annual_fees = round(total_fees * 365 / span_days, 2)

        # Per-month fee series (for a small bar chart on the dashboard).
        monthly = (
            user_df.assign(ym=user_df["created_date"].dt.to_period("M").astype(str))
            .groupby("ym")["fee_amount_gbp"]
            .sum()
            .round(2)
        )
        monthly_series = [{"month": m, "fees": float(v)} for m, v in monthly.items()]

        # Worst offenders.
        top = (
            fees.sort_values("fee_amount_gbp", ascending=False)
            .head(5)[["transaction_merchants_name", "fee_amount_gbp",
                      "txn_currency", "created_date"]]
        )
        top_fees = [{
            "merchant": str(r.transaction_merchants_name),
            "fee": round(float(r.fee_amount_gbp), 2),
            "currency": str(r.txn_currency),
            "date": r.created_date,
        } for r in top.itertuples()]

        # Severity: annual fees relative to spend, with a bump for avoidable
        # home-currency fees.
        fee_rate = (total_fees / total_spend) if total_spend else 0.0
        sev = min(1.0, fee_rate * 15 + (0.3 if home_fees > 0 else 0.0)
                  + min(0.3, annual_fees / 100.0))

        title = f"£{total_fees:.2f} in fees · ~£{annual_fees:.0f}/yr"
        bits = [f"You've paid £{total_fees:.2f} in fees"]
        if home_fees > 0:
            bits.append(f"£{home_fees:.2f} of it on GBP transactions that "
                        f"shouldn't carry FX fees")
        if currency_count > 1:
            bits.append(f"{foreign_share*100:.0f}% of your spend is in "
                        f"{currency_count} foreign currencies")
        explanation = "; ".join(bits) + "."

        actions = []
        if home_fees > 0:
            actions.append({"action": "review_card_plan",
                            "reason": "avoidable_home_currency_fees",
                            "amount": home_fees})

        return [Insight(
            type=self.type,
            user_id=str(user_df["owner_id"].iloc[0]),
            title=title,
            explanation=explanation,
            severity=sev,
            payload={
                "total_fees": total_fees,
                "annual_projection": annual_fees,
                "n_fee_txns": int(len(fees)),
                "avg_fee": round(float(fees["fee_amount_gbp"].mean()), 2) if len(fees) else 0.0,
                "avoidable_home_currency_fees": home_fees,
                "foreign_spend_share": round(foreign_share, 4),
                "currency_count": currency_count,
                "monthly_fees": monthly_series,
                "top_fees": top_fees,
            },
            actions=actions,
        )]
