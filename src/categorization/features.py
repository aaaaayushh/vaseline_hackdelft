"""
Feature engineering for the Layer-1 LightGBM classifier (DESIGN.md s4.2-4.4).

Three feature families:
  1. STATIC transaction features  (amount, channel, country, time, demographics)
  2. MERCHANT-NAME char n-gram hashing  (generalizes to unseen merchants: gracht.app ~ gracht.io)
  3. USER-HISTORY features  (temporal-safe: computed only from strictly-prior rows)
     + BEHAVIORAL-PERSONA prior (GMM cluster modal category) for cold-start.

Temporal integrity (DESIGN.md s4.4): every user-history feature is built with a
within-group time-ordered shift / expanding aggregate, so a row only ever sees that
user's PAST. No future leakage.

`mcc_prior` (the deterministic MCC->PFC category) is included as a categorical feature
but can be WITHHELD via include_mcc=False -- that is the capability eval (s8): recover
the budgeting category from behaviour + merchant structure when MCC is absent.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer

from .taxonomy import mcc_to_pfc, TAXONOMY

# pandas-categorical columns LightGBM will treat as native categoricals
_CAT_COLS = [
    "entry_method", "card_location", "merchant_country", "card_type", "card_brand",
    "region", "age_group", "gender", "txn_currency",
]
_BOOL_COLS = ["is_ecommerce", "is_domestic", "is_weekend"]
_NUM_COLS = ["log_amount", "is_credit", "hour", "day_of_week", "month"]

# user-history (added below), all temporal-safe
_HIST_NUM = ["u_prior_n", "um_prior_n", "um_prior_mean_amt", "amt_vs_um_mean"]
_HIST_CAT = ["u_prev_pfc", "um_prev_pfc", "persona", "persona_modal_pfc"]

N_HASH = 512  # merchant-name hashing dimensionality


def _basic(df: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=df.index)
    amt = df["txn_amount_gbp"].astype(float)
    f["log_amount"] = np.log1p(amt.abs())
    f["is_credit"] = (amt < 0).astype("int8")
    f["hour"] = df["hour"].astype("int16")
    f["day_of_week"] = df["day_of_week"].astype("int16")
    f["month"] = df["month"].astype("int16")
    for c in _BOOL_COLS:
        f[c] = df[c].astype("int8")
    for c in _CAT_COLS:
        f[c] = df[c].astype("category")
    return f


def _history_features(df: pd.DataFrame, pfc_label: pd.Series) -> pd.DataFrame:
    """Temporal-safe per-user and per-(user,merchant) history features.

    Sort by created_date and use within-group shifts / expanding aggregates so each
    row sees only strictly-earlier transactions of the same user/merchant.
    """
    work = pd.DataFrame({
        "owner_id": df["owner_id"].values,
        "merchant": df["transaction_merchants_name"].astype(str).values,
        "amt": df["txn_amount_gbp"].astype(float).abs().values,
        "pfc": pfc_label.values,
        "t": df["created_date"].values,
    }, index=df.index)
    work = work.sort_values("t", kind="stable")

    g_u = work.groupby("owner_id", sort=False)
    g_um = work.groupby(["owner_id", "merchant"], sort=False)

    h = pd.DataFrame(index=work.index)
    # counts of strictly-prior txns
    h["u_prior_n"] = g_u.cumcount().astype("int32")
    h["um_prior_n"] = g_um.cumcount().astype("int32")
    # previous category (leak-free: shifted within group) -- "history categorizes the future"
    h["u_prev_pfc"] = g_u["pfc"].shift(1)
    h["um_prev_pfc"] = g_um["pfc"].shift(1)
    # prior mean amount at this (user,merchant): vectorized expanding mean of strictly-prior
    # rows  ->  (cumsum - current) / prior_count
    um_cumsum = g_um["amt"].cumsum()
    prior_count = h["um_prior_n"].to_numpy()
    prior_sum = um_cumsum.to_numpy() - work["amt"].to_numpy()
    with np.errstate(invalid="ignore", divide="ignore"):
        prior_mean = np.where(prior_count > 0, prior_sum / prior_count, np.nan)
    h["um_prior_mean_amt"] = prior_mean
    h["amt_vs_um_mean"] = work["amt"].to_numpy() / (np.nan_to_num(prior_mean, nan=0.0) + 1e-6)

    return h.reindex(df.index)


def _persona_features(df: pd.DataFrame, pfc_label: pd.Series, train_mask: pd.Series,
                      n_personas: int = 6, random_state: int = 0):
    """Behavioral-persona prior via GMM on per-user category spend-mix.

    Mix vectors are built from TRAIN-period rows only (no leakage). Users unseen in
    train get assigned by the fitted GMM at predict time; their persona supplies a
    cold-start prior category. Returns (persona_id Series, persona_modal_pfc Series,
    fitted artifacts dict).
    """
    from sklearn.mixture import GaussianMixture

    cat_index = {c: i for i, c in enumerate(TAXONOMY)}
    tr = pd.DataFrame({
        "owner_id": df.loc[train_mask, "owner_id"].values,
        "pfc": pfc_label.loc[train_mask].values,
        "amt": df.loc[train_mask, "txn_amount_gbp"].astype(float).abs().values,
    })
    # per-user spend share per category
    piv = (tr.assign(ci=tr["pfc"].map(cat_index))
             .pivot_table(index="owner_id", columns="ci", values="amt",
                          aggfunc="sum", fill_value=0.0))
    piv = piv.reindex(columns=range(len(TAXONOMY)), fill_value=0.0)
    mix = piv.div(piv.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)

    gmm = GaussianMixture(n_components=n_personas, covariance_type="diag",
                          random_state=random_state, max_iter=100)
    gmm.fit(mix.values)
    user_persona = pd.Series(gmm.predict(mix.values), index=mix.index)

    # persona -> modal category (by total spend share across its users)
    persona_modal = {}
    for p in range(n_personas):
        members = user_persona[user_persona == p].index
        share = mix.loc[members].sum(axis=0)
        persona_modal[p] = TAXONOMY[int(share.values.argmax())] if len(members) else TAXONOMY[0]

    # map every row's user to a persona; unseen users -> global modal persona
    global_persona = int(user_persona.value_counts().idxmax())
    persona_id = df["owner_id"].map(user_persona).fillna(global_persona).astype(int)
    persona_modal_pfc = persona_id.map(persona_modal)
    artifacts = {"gmm": gmm, "user_persona": user_persona, "persona_modal": persona_modal,
                 "global_persona": global_persona}
    return persona_id, persona_modal_pfc, artifacts


def build_features(df: pd.DataFrame, pfc_label: pd.Series, train_mask: pd.Series,
                   include_mcc: bool = True):
    """Assemble the full design matrix.

    Returns (X_sparse, feature_names, dense_df, artifacts).
      * dense_df  -- numeric+categorical block (LightGBM-native categoricals)
      * X_sparse  -- horizontally stacked [dense one-hot-less codes] is NOT used;
                     instead we return dense_df for LightGBM + a separate sparse
                     merchant-hash block; caller hstacks via to_lgb_matrix().
    """
    dense = _basic(df)

    # mcc prior (categorical) -- withheld for capability eval
    if include_mcc:
        dense["mcc_prior"] = df["mcc"].map(mcc_to_pfc).astype("category")

    # history
    hist = _history_features(df, pfc_label)
    for c in _HIST_NUM:
        dense[c] = hist[c].astype("float32")
    for c in ["u_prev_pfc", "um_prev_pfc"]:
        dense[c] = hist[c].astype("category")

    # persona
    persona_id, persona_modal_pfc, persona_art = _persona_features(df, pfc_label, train_mask)
    dense["persona"] = persona_id.astype("category")
    dense["persona_modal_pfc"] = persona_modal_pfc.astype("category")

    # merchant-name char hashing (sparse) -- generalizes to unseen merchants
    hv = HashingVectorizer(analyzer="char_wb", ngram_range=(3, 5),
                           n_features=N_HASH, alternate_sign=False, norm=None)
    merch_hash = hv.transform(df["transaction_merchants_name"].astype(str).values)

    artifacts = {"persona": persona_art, "hashing_vectorizer": hv,
                 "include_mcc": include_mcc, "cat_cols": _cat_feature_names(dense)}
    return dense, merch_hash, artifacts


def _cat_feature_names(dense: pd.DataFrame) -> list[str]:
    return [c for c in dense.columns if str(dense[c].dtype) == "category"]


def to_lgb_matrix(dense: pd.DataFrame, merch_hash):
    """Combine the dense block + sparse merchant-hash into one scipy sparse matrix,
    with categorical columns integer-coded. Returns (X, feature_names, cat_idx)."""
    cols = list(dense.columns)
    blocks, names = [], []
    cat_idx = []
    for j, c in enumerate(cols):
        s = dense[c]
        if str(s.dtype) == "category":
            codes = s.cat.codes.to_numpy().astype(np.float32)  # -1 = NaN
            blocks.append(sparse.csr_matrix(codes.reshape(-1, 1)))
            cat_idx.append(j)
        else:
            blocks.append(sparse.csr_matrix(s.to_numpy(dtype=np.float32).reshape(-1, 1)))
        names.append(c)
    hash_names = [f"mh_{i}" for i in range(merch_hash.shape[1])]
    X = sparse.hstack(blocks + [merch_hash.tocsr().astype(np.float32)], format="csr")
    return X, names + hash_names, cat_idx
