"""
The categorization cascade + data-contract writer (DESIGN.md s3, s4.2).

Stamps EVERY transaction with the enriched contract the downstream engines read:
    {category, subcategory, confidence, source, merchant_id, is_recurring}

Cascade:
  Layer 0  STRUCTURAL RULES   transaction TYPE -> category (deterministic, conf=1.0).
                              Covers the 3.3% no-MCC rows + money-in. Bypasses ML.
  Layer 1  ML CLASSIFIER      LightGBM (MCC prior + merchant hash + user history +
                              persona) for every merchant txn. confidence = max proba.
  Layer 2  LLM FALLBACK       when Layer-1 confidence < threshold OR cold-start.
                              Stubbed here as `needs_llm=True` + a deterministic
                              merchant-modal backstop, cached by merchant_id (we do NOT
                              spend API calls during the bulk build -- DESIGN.md s4.2:
                              only the uncertain ~3% would hit the batched/cached LLM).

For the delivered enriched table we train Layer-1 on all merchant rows (category
assignment is a labeling task, not forecasting); temporal integrity is enforced in the
separate eval (train.py). User-history features are temporal-safe regardless.
"""

from __future__ import annotations
import time
import numpy as np
import pandas as pd

from .taxonomy import TAXONOMY, TYPE_RULES, mcc_to_pfc, INCOME
from .labels import assign_labels, MERCHANT_TYPES, MISSING_MCC
from .features import build_features, to_lgb_matrix
from .train import _fit_lgb, LABEL2ID, ID2LABEL

LLM_THRESHOLD = 0.55  # Layer-1 confidence below this -> Layer-2 (LLM) fallback


def _merchant_id(names: pd.Series) -> pd.Series:
    codes, _ = pd.factorize(names.astype(str))
    return pd.Series([f"m_{c:05d}" for c in codes], index=names.index)


def _is_recurring(df: pd.DataFrame) -> pd.Series:
    """Subscription radar (DESIGN.md s5.3): a (user, merchant) pair seen in >=3 distinct
    months with a near-fixed amount (low coefficient of variation). Conservative -- the
    synthetic cadence is noisy, so we only flag stable repeats."""
    g = df.groupby(["owner_id", "transaction_merchants_name"])
    months = g["month"].nunique()
    amt = df.groupby(["owner_id", "transaction_merchants_name"])["txn_amount_gbp"]
    cv = (amt.std() / amt.mean().abs().replace(0, np.nan)).fillna(1.0)
    recurring_pairs = (months >= 3) & (cv <= 0.15)
    key = list(zip(df["owner_id"], df["transaction_merchants_name"]))
    flag = recurring_pairs.reindex(key).astype("boolean").fillna(False).to_numpy(dtype=bool)
    return pd.Series(flag, index=df.index)


def run(parquet="output/df_clean.parquet", out="output/df_enriched.parquet",
        sample=None, seed=0):
    t0 = time.time()
    df = pd.read_parquet(parquet)
    if sample:
        df = df.sample(sample, random_state=seed).reset_index(drop=True)
    df = assign_labels(df)
    n = len(df)
    print(f"loaded {n:,} rows in {time.time()-t0:.1f}s")

    category = pd.Series(index=df.index, dtype="object")
    subcategory = df["irs_description"].astype("object").where(
        df["irs_description"].notna(), df["type"])
    confidence = pd.Series(np.nan, index=df.index, dtype="float64")
    source = pd.Series(index=df.index, dtype="object")
    needs_llm = pd.Series(False, index=df.index)

    # ---------- Layer 0: structural TYPE rules ----------
    struct_mask = ~(df["type"].isin(MERCHANT_TYPES) & (df["mcc"] != MISSING_MCC))
    category.loc[struct_mask] = df.loc[struct_mask, "type"].map(TYPE_RULES)
    confidence.loc[struct_mask] = 1.0
    source.loc[struct_mask] = "rule"
    # CARD_REFUND keeps its MCC category but is money-in -> still flows through ML below.
    print(f"Layer 0 (structural rules): {struct_mask.sum():,} rows")

    # ---------- Layer 1: ML classifier on merchant rows ----------
    merch_mask = ~struct_mask
    merch = df[merch_mask].copy()
    train_mask = pd.Series(True, index=merch.index)  # label-task: fit on all merchant rows
    dense, mhash, art = build_features(merch, merch["pfc_label"], train_mask, include_mcc=True)
    X, names, cat_idx = to_lgb_matrix(dense, mhash)
    y = merch["pfc_label"].map(LABEL2ID).to_numpy()
    print(f"Layer 1 (ML): training on {X.shape[0]:,} merchant rows, {X.shape[1]:,} features...")
    model = _fit_lgb(X, y, [], len(TAXONOMY), seed)  # numeric encoding (mcc_prior deterministic)
    proba = model.predict(X)
    pred_id = proba.argmax(axis=1)
    conf = proba.max(axis=1)
    category.loc[merch_mask] = [ID2LABEL[i] for i in pred_id]
    confidence.loc[merch_mask] = conf
    source.loc[merch_mask] = "ml"

    # ---------- Layer 2: LLM fallback (stubbed, cached by merchant) ----------
    low = merch_mask & (confidence < LLM_THRESHOLD)
    needs_llm.loc[low] = True
    # deterministic backstop instead of a live API call: per-merchant modal MCC->PFC prior
    if low.any():
        merch_prior = df.loc[merch_mask].assign(p=df.loc[merch_mask, "mcc"].map(mcc_to_pfc))
        modal = merch_prior.groupby("transaction_merchants_name")["p"].agg(
            lambda s: s.value_counts().idxmax())
        backstop = df.loc[low, "transaction_merchants_name"].map(modal)
        category.loc[low] = backstop.fillna(category.loc[low])
        source.loc[low] = "llm_fallback(stub)"
    print(f"Layer 2 (LLM fallback): {low.sum():,} low-confidence rows flagged "
          f"({low.sum()/n*100:.2f}% of all)")

    # ---------- contract extras ----------
    df["category"] = category.astype(str)
    df["subcategory"] = subcategory.astype(str)
    df["confidence"] = confidence.astype(float)
    df["cat_source"] = source.astype(str)
    df["needs_llm"] = needs_llm.to_numpy()
    df["merchant_id"] = _merchant_id(df["transaction_merchants_name"])
    df["is_recurring"] = _is_recurring(df).to_numpy()
    # clean signed amount for cashflow math (credits negative, debits positive)
    money_in = df["type"].isin(["CARD_CREDIT", "CARD_CHARGEBACK", "CARD_REFUND"])
    df["amount_signed_gbp"] = np.where(money_in, -df["txn_amount_gbp"].abs(),
                                       df["txn_amount_gbp"].abs())

    assert df["category"].isin(TAXONOMY).all(), "uncategorized rows remain!"
    df.to_parquet(out, index=False)
    print(f"\nwrote {out}  ({len(df):,} rows, every row categorized)")
    print(f"category distribution:\n{df['category'].value_counts().to_string()}")
    print(f"\nrecurring merchant-txns: {df['is_recurring'].mean()*100:.1f}%  | "
          f"distinct merchants: {df['merchant_id'].nunique():,}")
    print(f"total wall time: {time.time()-t0:.1f}s")
    return df


if __name__ == "__main__":
    import sys
    sample = int(sys.argv[1]) if len(sys.argv) > 1 else None
    run(sample=sample)
