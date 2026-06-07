"""
Label assignment for the categorization engine.

On synthetic data MCC is the ground truth (DESIGN.md s8), so the training/eval label is:
  * structural TYPE rows (ATM/FEE/CHARGE/CARD_CREDIT/CARD_CHARGEBACK) -> Layer-0 TYPE_RULES
  * merchant rows (CARD_PAYMENT/CARD_REFUND, which carry a real MCC) -> mcc_to_pfc(mcc)

`label_source` records which path produced the label so eval can separate the
deterministic Layer-0 rows (trivially correct) from the ML-relevant merchant rows.
"""

from __future__ import annotations
import pandas as pd

from .taxonomy import mcc_to_pfc, TYPE_RULES, INCOME

MISSING_MCC = "None"  # how missing MCC is encoded in df_clean.parquet
MERCHANT_TYPES = ("CARD_PAYMENT", "CARD_REFUND")


def assign_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Return df with added columns: `pfc_label` (str) and `label_source` ('type'|'mcc')."""
    out = df.copy()
    is_merchant = out["type"].isin(MERCHANT_TYPES) & (out["mcc"] != MISSING_MCC)

    label = pd.Series(index=out.index, dtype="object")
    source = pd.Series(index=out.index, dtype="object")

    # Merchant rows -> MCC-derived PFC (the ML target)
    label.loc[is_merchant] = out.loc[is_merchant, "mcc"].map(mcc_to_pfc)
    source.loc[is_merchant] = "mcc"

    # Structural rows -> TYPE rule
    struct = ~is_merchant
    label.loc[struct] = out.loc[struct, "type"].map(TYPE_RULES)
    source.loc[struct] = "type"

    # Any leftover (e.g. a merchant type whose mcc was 'None') -> fall back on TYPE rule,
    # and if the type has no rule, treat as INCOME if money-in else SHOPPING-safe SHOPPING.
    missing = label.isna()
    if missing.any():
        label.loc[missing] = out.loc[missing, "type"].map(TYPE_RULES)
        source.loc[missing] = "type"
        still = label.isna()
        if still.any():
            label.loc[still] = INCOME  # safe: only money-in types reach here
            source.loc[still] = "type"

    out["pfc_label"] = label.astype(str)
    out["label_source"] = source.astype(str)
    return out
