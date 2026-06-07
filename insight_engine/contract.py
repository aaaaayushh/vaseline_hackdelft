"""The data contract between the Categorization Engine and the Insight Engine.

The categorizer is expected to stamp every transaction with
``{category, subcategory, confidence, merchant_id, is_recurring}`` and write an
enriched parquet. That engine is built separately and may not exist yet, so
``load_enriched`` is tolerant: any missing contract column is synthesised from
the cleaned substrate using a *placeholder* mapper (see ``taxonomy.py``). When
the real categorizer lands, point ``category_col`` at its output column and the
placeholder is bypassed — no detector code changes.

It also defines the output side of the contract: the typed ``Insight`` object
every detector emits and the dashboard renders.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any
import hashlib

import numpy as np
import pandas as pd

from . import taxonomy


# --------------------------------------------------------------------------- #
# Input contract
# --------------------------------------------------------------------------- #

# Columns the Insight Engine relies on. Everything else is passed through.
ENRICHED_COLUMNS = [
    # identity / partitioning
    "transaction_id",
    "owner_id",
    "type",
    # categorizer stamps (synthesised by load_enriched if absent)
    "category",        # 12-category budgeting taxonomy
    "merchant_id",     # stable merchant key
    "is_recurring",    # bool — subscription-like cadence
    # money
    "amount_signed_gbp",  # debits +, credits -  (spend is positive)
    "txn_amount_gbp",
    "bill_amount_gbp",
    "fee_amount_gbp",
    "txn_currency",
    # merchant / context
    "transaction_merchants_name",
    # demographics (cohort benchmarking)
    "age_group",
    "gender",
    "region",
    # time
    "created_date",
    "date",
    "month",
    "week",
]

# Transaction types that represent money leaving the account as discretionary
# spend (used by most detectors). Credits/refunds/chargebacks are money in.
SPEND_TYPES = {"CARD_PAYMENT", "ATM", "FEE", "CHARGE"}
INCOME_TYPES = {"CARD_REFUND", "CARD_CREDIT", "CARD_CHARGEBACK"}


def load_enriched(
    path: str,
    *,
    category_col: str | None = None,
    merchant_col: str = "transaction_merchants_name",
) -> pd.DataFrame:
    """Load the enriched transaction table the Insight Engine consumes.

    Parameters
    ----------
    path:
        Parquet written by the cleaning notebook (or, later, the categorizer).
    category_col:
        Name of the clean 12-category column to use. If ``None`` (default) or
        the column is absent, a placeholder category is derived from MCC +
        merchant name via :mod:`insight_engine.taxonomy`. Set this to the
        categorizer's output column once it exists.
    merchant_col:
        Column to use as the stable merchant key when ``merchant_id`` is absent.
    """
    df = pd.read_parquet(path)

    # --- category -------------------------------------------------------- #
    if category_col and category_col in df.columns:
        df["category"] = df[category_col]
    elif "category" not in df.columns or _looks_like_raw_mcc_labels(df):
        df["category"] = taxonomy.derive_category(df)
    # Collapse the categorizer's richer 16-category labels onto the canonical
    # 12-category budgeting taxonomy so either engine's output uses one contract.
    df["category"] = taxonomy.normalize_category(df["category"])

    # --- merchant_id ----------------------------------------------------- #
    if "merchant_id" not in df.columns:
        src = merchant_col if merchant_col in df.columns else "transaction_merchants_name"
        df["merchant_id"] = df[src].fillna("UNKNOWN").astype(str)

    # --- is_recurring (computed here if categorizer hasn't) ------------- #
    if "is_recurring" not in df.columns:
        df["is_recurring"] = _infer_recurring(df)

    # --- amount_signed_gbp ---------------------------------------------- #
    if "amount_signed_gbp" not in df.columns:
        sign = np.where(df["type"].isin(INCOME_TYPES), -1.0, 1.0)
        df["amount_signed_gbp"] = sign * df["txn_amount_gbp"].abs()

    # --- normalise time columns ----------------------------------------- #
    df["created_date"] = pd.to_datetime(df["created_date"])
    if "date" not in df.columns:
        df["date"] = df["created_date"].dt.date
    if "month" not in df.columns:
        df["month"] = df["created_date"].dt.month
    if "week" not in df.columns:
        df["week"] = df["created_date"].dt.isocalendar().week.astype(int)

    return df


def _looks_like_raw_mcc_labels(df: pd.DataFrame) -> bool:
    """The shipped `category` is a 1:1 copy of the messy MCC labels (~396 of
    them). If we see that many distinct values it is not a budgeting taxonomy,
    so we replace it with the placeholder mapper."""
    if "category" not in df.columns:
        return True
    return df["category"].nunique(dropna=True) > 40


def _infer_recurring(df: pd.DataFrame, min_hits: int = 3) -> pd.Series:
    """Mark transactions whose (user, merchant) pair recurs on a roughly
    regular cadence. This is a stand-in for the categorizer's `is_recurring`
    flag; the Subscription Radar does its own, richer cadence analysis."""
    is_card = df["type"].eq("CARD_PAYMENT")
    grp = df.loc[is_card].groupby(["owner_id", "merchant_id"])
    counts = grp["transaction_id"].transform("count")
    flag = pd.Series(False, index=df.index)
    flag.loc[is_card] = counts.ge(min_hits).values
    return flag


# --------------------------------------------------------------------------- #
# Output contract
# --------------------------------------------------------------------------- #

class Severity(str, Enum):
    """Coarse severity band, derived from the numeric score. Drives card
    styling on the dashboard (color / icon)."""
    INFO = "info"
    NOTICE = "notice"
    WARNING = "warning"
    ALERT = "alert"

    @classmethod
    def from_score(cls, score: float) -> "Severity":
        if score >= 0.75:
            return cls.ALERT
        if score >= 0.5:
            return cls.WARNING
        if score >= 0.25:
            return cls.NOTICE
        return cls.INFO


@dataclass
class Insight:
    """A typed insight object — the unit the Insight Engine emits and the
    dashboard renders as a card.

    Attributes
    ----------
    type:        detector type, e.g. ``"subscription_radar"``.
    user_id:     the user this insight is for.
    title:       short headline for the card.
    explanation: one or two human-readable sentences (no invented numbers).
    severity:    numeric score in [0, 1] used for ranking.
    payload:     structured, JSON-serializable data for rich rendering
                 (lists, series for charts, amounts, dates...).
    actions:     optional suggested actions the agent/app can offer.
    """
    type: str
    user_id: str
    title: str
    explanation: str
    severity: float = 0.0
    payload: dict[str, Any] = field(default_factory=dict)
    actions: list[dict[str, Any]] = field(default_factory=list)

    @property
    def level(self) -> Severity:
        return Severity.from_score(self.severity)

    @property
    def insight_id(self) -> str:
        """Stable id for dedup / cooldown across runs."""
        key = f"{self.user_id}|{self.type}|{self.title}"
        return hashlib.sha1(key.encode()).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["level"] = self.level.value
        d["insight_id"] = self.insight_id
        d["severity"] = round(float(self.severity), 4)
        return _jsonable(d)


def _jsonable(obj: Any) -> Any:
    """Recursively coerce numpy / pandas / datetime types to JSON-native."""
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if np.isnan(obj) else round(float(obj), 4)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp,)):
        return obj.isoformat()
    if isinstance(obj, float):
        return None if np.isnan(obj) else round(obj, 4)
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    return obj
