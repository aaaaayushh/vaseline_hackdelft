"""
Load and clean the synthetic transaction dataset.

The data is synthetic and, per the brief, "can and does contain mistakes, weird
values, impossible combinations, and artifacts." This module loads it, repairs
or flags the issues that matter for categorisation and spend analytics, and
returns both the cleaned frame and a human-readable cleaning report so nothing
is silently dropped.

Design choice: we do NOT delete rows. A categorisation engine has to assign
*something* to every transaction (declined, reverted, zero-amount included), so
we keep all rows and instead add boolean flags (`is_spend`, `is_completed`)
that downstream analytics can use to scope what counts as real spending.
"""
from __future__ import annotations

import pandas as pd

# Transaction types that represent an outgoing card purchase at a merchant.
PURCHASE_TYPES = {"CARD_PAYMENT"}


def load_and_clean(path: str = "dataset.parquet") -> tuple[pd.DataFrame, dict]:
    df = pd.read_parquet(path)
    report: dict[str, object] = {"rows_in": len(df)}

    # --- 1. De-duplicate on transaction_id (defensive; synthetic UUIDs) ------
    dupes = int(df["transaction_id"].duplicated().sum())
    if dupes:
        df = df.drop_duplicates(subset="transaction_id")
    report["duplicate_txn_ids_removed"] = dupes

    # --- 2. Normalise string columns: strip whitespace, empty -> NA ----------
    str_cols = [
        "type", "state", "mcc", "category", "transaction_merchants_name",
        "transaction_merchants_code", "txn_currency",
    ]
    for c in str_cols:
        df[c] = df[c].astype("string").str.strip()
        df.loc[df[c].eq(""), c] = pd.NA

    # --- 3. Flag impossible / non-spend rows (keep them, don't drop) ---------
    # Zero-amount transactions carry no spend signal.
    report["zero_amount_rows"] = int((df["txn_amount_gbp"] == 0).sum())
    # Date ordering sanity (completed before created would be impossible).
    bad_dates = int(
        (df["completed_date"].notna()
         & (df["completed_date"] < df["created_date"])).sum()
    )
    report["impossible_date_order_rows"] = bad_dates

    df["is_completed"] = df["state"].eq("COMPLETED")
    # "Real spend" = a completed outgoing purchase or ATM withdrawal with a
    # positive amount. This is the slice budgeting insights should sum over.
    df["is_spend"] = (
        df["is_completed"]
        & df["type"].isin(PURCHASE_TYPES | {"ATM"})
        & (df["txn_amount_gbp"] > 0)
    )
    report["spend_rows"] = int(df["is_spend"].sum())

    # --- 4. Time features used by the categoriser / insight engine -----------
    df["hour"] = df["created_date"].dt.hour
    df["dow"] = df["created_date"].dt.dayofweek
    df["month"] = df["created_date"].dt.to_period("M").astype("string")

    # --- 5. Missingness snapshot for the report ------------------------------
    report["missing_merchant_code_rows"] = int(
        df["transaction_merchants_code"].isna().sum()
    )
    report["rows_out"] = len(df)
    report["date_range"] = (
        str(df["created_date"].min()), str(df["created_date"].max())
    )
    return df, report


def build_merchant_catalog(df: pd.DataFrame) -> pd.DataFrame:
    """
    Collapse the 1M transactions to the unique-merchant level.

    `transaction_merchants_code` is the true merchant key: each code maps to
    exactly one name and one MCC. Categorising ~4.6k merchants once (then
    joining back to transactions) is far cheaper than classifying 1M rows and
    is how a production system would cache merchant categories.
    """
    merch = df[df["transaction_merchants_code"].notna()].copy()
    catalog = (
        merch.groupby("transaction_merchants_code")
        .agg(
            name=("transaction_merchants_name", "first"),
            mcc=("mcc", "first"),
            mcc_category=("category", "first"),
            n_txns=("transaction_id", "size"),
            median_amount=("txn_amount_gbp", "median"),
            ecommerce_rate=("app_is_ecommerce",
                            lambda s: pd.to_numeric(
                                s.astype("string").map(
                                    {"True": 1, "False": 0}), errors="coerce"
                            ).mean()),
        )
        .reset_index()
    )
    return catalog
