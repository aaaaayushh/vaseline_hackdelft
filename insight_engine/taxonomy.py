"""Placeholder MCC -> 12-category mapper.

This is a *stand-in* for the Categorization Engine, which is built separately.
It lets the Insight Engine run end-to-end on the cleaned parquet today. When
the real categorizer ships, pass its output column to ``load_enriched`` and
this module is bypassed entirely.

The 12-category budgeting taxonomy (from DESIGN.md §4.1):
    Groceries, Dining & Takeaway, Transport, Shopping, Digital & Subscriptions,
    Bills & Utilities, Entertainment, Health, Travel, Cash & ATM,
    Fees & Charges, Income & Refunds
"""

from __future__ import annotations

import numpy as np
import pandas as pd

GROCERIES = "Groceries"
DINING = "Dining & Takeaway"
TRANSPORT = "Transport"
SHOPPING = "Shopping"
DIGITAL = "Digital & Subscriptions"
BILLS = "Bills & Utilities"
ENTERTAINMENT = "Entertainment"
HEALTH = "Health"
TRAVEL = "Travel"
CASH = "Cash & ATM"
FEES = "Fees & Charges"
INCOME = "Income & Refunds"

CATEGORIES = [
    GROCERIES, DINING, TRANSPORT, SHOPPING, DIGITAL, BILLS,
    ENTERTAINMENT, HEALTH, TRAVEL, CASH, FEES, INCOME,
]

# Type -> category (Layer-0 structural rules; bypasses MCC).
_TYPE_MAP = {
    "ATM": CASH,
    "FEE": FEES,
    "CHARGE": FEES,
    "CARD_REFUND": INCOME,
    "CARD_CREDIT": INCOME,
    "CARD_CHARGEBACK": INCOME,
}

# Exact-MCC overrides where range logic is too coarse.
_MCC_EXACT = {
    "5411": GROCERIES, "5412": GROCERIES, "5422": GROCERIES, "5441": GROCERIES,
    "5451": GROCERIES, "5462": GROCERIES, "5499": GROCERIES,
    "5812": DINING, "5811": DINING, "5814": DINING, "5499 ": DINING,
    "5813": ENTERTAINMENT, "5921": GROCERIES,
    "4111": TRANSPORT, "4121": TRANSPORT, "4131": TRANSPORT, "4789": TRANSPORT,
    "5541": TRANSPORT, "5542": TRANSPORT, "7523": TRANSPORT, "4112": TRANSPORT,
    "4899": BILLS, "4814": BILLS, "4812": BILLS, "4900": BILLS, "4816": BILLS,
    "5735": DIGITAL, "5734": DIGITAL, "5816": DIGITAL, "5815": DIGITAL,
    "5817": DIGITAL, "5818": DIGITAL,
    "7995": ENTERTAINMENT, "7832": ENTERTAINMENT, "7996": ENTERTAINMENT,
    "7922": ENTERTAINMENT, "7993": ENTERTAINMENT, "7994": ENTERTAINMENT,
    "5912": HEALTH, "8011": HEALTH, "8021": HEALTH, "8042": HEALTH,
    "8062": HEALTH, "8099": HEALTH, "8043": HEALTH,
    "4722": TRAVEL, "4511": TRAVEL, "7011": TRAVEL, "4411": TRAVEL,
    "7512": TRAVEL, "7011 ": TRAVEL,
}

# Merchant-name substrings that signal digital/subscription spend regardless of
# MCC (this is exactly the ~26% misfiled digital spend DESIGN.md calls out).
_DIGITAL_NAME_HINTS = ("abonnement", ".app", ".io", "gracht", "tulp")


def _mcc_range(mcc: str) -> str:
    """Fallback by MCC numeric range (standard ISO 18245 banding)."""
    try:
        n = int(mcc)
    except (TypeError, ValueError):
        return SHOPPING
    if 3000 <= n <= 3999:
        return TRAVEL          # airlines / car rental / hotels
    if 4000 <= n <= 4799:
        return TRANSPORT
    if 4800 <= n <= 4999:
        return BILLS
    if 5300 <= n <= 5499:
        return GROCERIES
    if 5500 <= n <= 5599:
        return TRANSPORT       # automotive / fuel
    if 5600 <= n <= 5699:
        return SHOPPING        # apparel
    if 5700 <= n <= 5799:
        return SHOPPING
    if 5800 <= n <= 5899:
        return DINING
    if 5900 <= n <= 5999:
        return SHOPPING        # retail
    if 7000 <= n <= 7299:
        return TRAVEL          # lodging / personal services
    if 7800 <= n <= 7999:
        return ENTERTAINMENT
    if 8000 <= n <= 8099:
        return HEALTH
    return SHOPPING


def derive_category(df: pd.DataFrame) -> pd.Series:
    """Vectorized MCC + merchant-name -> 12-category mapping for a frame."""
    types = df["type"].astype(str)
    mcc = df["mcc"].astype(str) if "mcc" in df.columns else pd.Series("", index=df.index)
    names = (
        df["transaction_merchants_name"].astype(str).str.lower()
        if "transaction_merchants_name" in df.columns
        else pd.Series("", index=df.index)
    )

    # 1) structural type rules
    out = types.map(_TYPE_MAP)

    # 2) exact MCC, then range fallback (only where type rule didn't fire)
    need = out.isna()
    exact = mcc.map(_MCC_EXACT)
    out = out.where(~need, exact)
    still = out.isna()
    out.loc[still] = mcc.loc[still].map(_mcc_range)

    # 3) digital-merchant name override (the misfiled-digital fix), but never
    #    override structural income/fee/cash rows.
    name_hit = pd.Series(False, index=df.index)
    for h in _DIGITAL_NAME_HINTS:
        name_hit |= names.str.contains(h, regex=False, na=False)
    overridable = ~types.isin(_TYPE_MAP)
    out = out.mask(name_hit & overridable, DIGITAL)

    return out.fillna(SHOPPING)
