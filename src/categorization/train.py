"""
Train + evaluate the Layer-1 LightGBM categorizer (DESIGN.md s4, s8).

Two evaluation regimes, both honest about synthetic data:

  PRODUCTION (time split):   train Jul-Oct, test Nov-Dec, MCC prior + history INCLUDED.
                             The realistic deployed accuracy on a user's continuing life.

  CAPABILITY (held-out merch): hold out a random 20% of MERCHANTS; train on the rest;
                             MCC WITHHELD. Tests whether the model recovers the budgeting
                             category from behaviour + merchant structure when MCC is
                             absent -- the production reality (3.3% have no MCC here; often
                             missing entirely in the wild). A random row split would be
                             trivial (only 0.03% of Nov-Dec txns are at new merchants), so
                             held-out merchants is the real generalization test (s8).
"""

from __future__ import annotations
import json
import time
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import accuracy_score, f1_score, classification_report

from .taxonomy import TAXONOMY
from .labels import assign_labels, MERCHANT_TYPES, MISSING_MCC
from .features import build_features, to_lgb_matrix

LABEL2ID = {c: i for i, c in enumerate(TAXONOMY)}
ID2LABEL = {i: c for c, i in LABEL2ID.items()}


def _fit_lgb(X, y, cat_idx, n_class, seed=0, n_estimators=300):
    """Fit a multiclass LightGBM.

    Pass cat_idx=[] for NUMERIC/ordinal encoding (the right choice for the production /
    enrichment model: the near-deterministic mcc_prior is isolated cleanly by numeric
    splits, and LightGBM's categorical-split smoothing -- which over-regularizes when
    many categoricals coexist at scale -- is avoided). Pass real cat_idx (with default
    regularization) for the CAPABILITY model, where genuine categorical grouping of
    merchant_country / persona / prev-category generalizes to unseen merchants.
    """
    cats = list(cat_idx) if cat_idx else []  # [] => all-numeric encoding
    train = lgb.Dataset(X, label=y, categorical_feature=cats, free_raw_data=False)
    params = dict(
        objective="multiclass", num_class=n_class, learning_rate=0.1,
        num_leaves=63, min_data_in_leaf=100, feature_fraction=0.8,
        bagging_fraction=0.8, bagging_freq=1, max_depth=-1,
        verbose=-1, seed=seed, num_threads=0,
    )
    return lgb.train(params, train, num_boost_round=n_estimators)


def _eval(model, X, y_true_ids, name):
    proba = model.predict(X)
    pred = proba.argmax(axis=1)
    conf = proba.max(axis=1)
    acc = accuracy_score(y_true_ids, pred)
    f1m = f1_score(y_true_ids, pred, average="macro")
    print(f"\n[{name}]  n={len(y_true_ids):,}  accuracy={acc:.4f}  macro-F1={f1m:.4f}  "
          f"mean-confidence={conf.mean():.3f}")
    return {"name": name, "n": int(len(y_true_ids)), "accuracy": float(acc),
            "macro_f1": float(f1m), "mean_confidence": float(conf.mean())}


def run(parquet="output/df_clean.parquet", sample=None, seed=0):
    t0 = time.time()
    cols = None
    df = pd.read_parquet(parquet)
    if sample:
        df = df.sample(sample, random_state=seed).reset_index(drop=True)
    df = assign_labels(df)
    print(f"loaded {len(df):,} rows in {time.time()-t0:.1f}s | merchant rows: "
          f"{df['type'].isin(MERCHANT_TYPES).sum():,}")

    # ML operates on MERCHANT rows only (structural rows are Layer-0 deterministic).
    merch = df[df["type"].isin(MERCHANT_TYPES) & (df["mcc"] != MISSING_MCC)].copy()
    merch = merch.reset_index(drop=True)
    y = merch["pfc_label"].map(LABEL2ID).to_numpy()

    results = {}

    # ---------- PRODUCTION: time split, MCC included ----------
    tmask = merch["created_date"].dt.month <= 10  # train Jul-Oct
    dense, mhash, art = build_features(merch, merch["pfc_label"], tmask, include_mcc=True)
    X, names, cat_idx = to_lgb_matrix(dense, mhash)
    Xtr, ytr = X[tmask.to_numpy()], y[tmask.to_numpy()]
    Xte, yte = X[~tmask.to_numpy()], y[~tmask.to_numpy()]
    print(f"\n=== PRODUCTION (time split, MCC prior + history) ===\n"
          f"train {Xtr.shape[0]:,} (Jul-Oct) | test {Xte.shape[0]:,} (Nov-Dec) | "
          f"features {X.shape[1]:,}")
    m_prod = _fit_lgb(Xtr, ytr, [], len(TAXONOMY), seed)  # numeric: mcc_prior is deterministic
    results["production"] = _eval(m_prod, Xte, yte, "PRODUCTION time-split (Nov-Dec)")

    # known vs cold-start merchant breakdown within the test window
    te_idx = (~tmask).to_numpy()
    seen_merch = set(merch.loc[tmask, "transaction_merchants_name"])
    is_new = ~merch.loc[te_idx, "transaction_merchants_name"].isin(seen_merch)
    if is_new.any():
        results["production_new_merch"] = _eval(
            m_prod, Xte[is_new.to_numpy()], yte[is_new.to_numpy()],
            "  -> cold-start merchants in test")

    # ---------- CAPABILITY: held-out merchants, MCC withheld ----------
    rng = np.random.default_rng(seed)
    all_merch = merch["transaction_merchants_name"].astype(str).unique()
    holdout = set(rng.choice(all_merch, size=int(0.2 * len(all_merch)), replace=False))
    held_mask = merch["transaction_merchants_name"].astype(str).isin(holdout).to_numpy()
    cap_train_mask = pd.Series(~held_mask, index=merch.index)
    dense2, mhash2, _ = build_features(merch, merch["pfc_label"], cap_train_mask,
                                       include_mcc=False)
    X2, names2, cat_idx2 = to_lgb_matrix(dense2, mhash2)
    print(f"\n=== CAPABILITY (held-out merchants, MCC WITHHELD) ===\n"
          f"train {(~held_mask).sum():,} on {len(all_merch)-len(holdout):,} merchants | "
          f"test {held_mask.sum():,} on {len(holdout):,} unseen merchants | "
          f"features {X2.shape[1]:,}")
    m_cap = _fit_lgb(X2[~held_mask], y[~held_mask], cat_idx2, len(TAXONOMY), seed)
    results["capability"] = _eval(m_cap, X2[held_mask], y[held_mask],
                                  "CAPABILITY held-out merchants (no MCC)")

    # majority-class & MCC-prior baselines for context
    maj = np.bincount(y[~held_mask]).argmax()
    base_acc = accuracy_score(y[held_mask], np.full(held_mask.sum(), maj))
    print(f"  baseline (majority class) accuracy on held-out: {base_acc:.4f}")
    results["capability_majority_baseline"] = float(base_acc)

    print(f"\ntotal wall time: {time.time()-t0:.1f}s")
    with open("output/categorizer_eval.json", "w") as f:
        json.dump(results, f, indent=2)
    print("wrote output/categorizer_eval.json")
    return results


if __name__ == "__main__":
    import sys
    sample = int(sys.argv[1]) if len(sys.argv) > 1 else None
    run(sample=sample)
