"""
The categorisation cascade (Option 4: embeddings + LLM).

Per merchant, cheapest-confident-tier-first:

  Tier A  name keyword        deterministic Dutch head-word -> category
  Tier B  MCC embedding       embed the MCC description, nearest taxonomy anchor
  Tier C  LLM (Claude)        for Unknown-MCC / low-confidence merchants
  Tier C' name embedding      deterministic fallback when the LLM tier is off

A merchant's final category comes from the highest-confidence applicable tier.
Where a confident MCC mapping disagrees with a strong name signal, we don't
silently override — we keep the category but raise a `correction_flag`, which is
exactly the "supermarket tagged as Shopping" case the brief asks us to fix.

Transaction-level non-purchases (FEE/ATM/credits/chargebacks, ~3.3%, no MCC)
are handled by the deterministic TYPE_TO_CATEGORY rule when we join back.
"""
from __future__ import annotations

import pandas as pd

from . import embed, llm
from .taxonomy import (MCC_OVERRIDES, NAME_KEYWORDS, TYPE_TO_CATEGORY,
                       UNKNOWN_MCCS)

# Cosine-similarity floor below which an MCC->anchor mapping is "unsure" and
# gets escalated to the LLM / name tier.
LOW_CONF = 0.30


def _name_keyword(name: str | None) -> str | None:
    if not name:
        return None
    low = name.lower()
    for kw, cat in NAME_KEYWORDS.items():
        if kw in low:
            return cat
    return None


def categorise_catalog(catalog: pd.DataFrame) -> pd.DataFrame:
    """Add final_category / source / confidence / correction_flag per merchant."""
    cat = catalog.copy()

    # --- Tier A: name keyword (cheap, deterministic) -------------------------
    cat["name_category"] = cat["name"].map(_name_keyword)

    # --- Tier 0: explicit ISO-MCC pins (override the embedding where its
    # wording is known to mislead, e.g. fuel "Service Stations") -------------
    cat["mcc_override"] = cat["mcc"].map(MCC_OVERRIDES)

    # --- Tier B: MCC-description embedding -> taxonomy ------------------------
    # Map each *unique* MCC description once, then broadcast.
    has_mcc = ~cat["mcc"].isin(UNKNOWN_MCCS) & cat["mcc_category"].notna()
    uniq_desc = (
        cat.loc[has_mcc, "mcc_category"].dropna().astype(str).unique().tolist()
    )
    desc_cat, desc_conf = embed.map_texts_to_taxonomy(uniq_desc)
    desc_map = {d: (c, float(s))
                for d, c, s in zip(uniq_desc, desc_cat, desc_conf)}
    cat["mcc_emb_category"] = cat["mcc_category"].map(
        lambda d: desc_map.get(str(d), (None, 0.0))[0])
    cat["mcc_emb_conf"] = cat["mcc_category"].map(
        lambda d: desc_map.get(str(d), (None, 0.0))[1])

    # --- Resolve each merchant through the cascade ---------------------------
    final_cat, source, conf, correction = [], [], [], []
    # Collect the hard tail for the LLM tier in one batch.
    tail_idx, tail_rows = [], []

    for i, r in cat.iterrows():
        nm_cat = r["name_category"]
        # Misses come back as NaN under string dtype; NaN is truthy, so coerce
        # anything that isn't a real category string to None.
        if not isinstance(nm_cat, str):
            nm_cat = None
        mcc_ok = bool(has_mcc.loc[i])
        emb_cat, emb_conf = r["mcc_emb_category"], r["mcc_emb_conf"]
        ovr = r["mcc_override"]
        ovr = ovr if isinstance(ovr, str) else None

        if ovr is not None:
            # Deterministic ISO pin wins.
            final_cat.append(ovr); source.append("mcc_rule")
            conf.append(1.0)
            correction.append(bool(nm_cat and nm_cat != ovr))
        elif mcc_ok and emb_conf >= LOW_CONF:
            # Confident MCC mapping is the backbone.
            final_cat.append(emb_cat); source.append("mcc_embedding")
            conf.append(emb_conf)
            # Strong name signal disagreeing => candidate miscategorisation.
            correction.append(bool(nm_cat and nm_cat != emb_cat))
        elif nm_cat is not None:
            # No/weak MCC, but an informative name -> trust the name (68-100%
            # pure on this data).
            final_cat.append(nm_cat); source.append("name_rule")
            conf.append(0.90); correction.append(False)
        else:
            # Hard tail: Unknown MCC and uninformative name. Defer to LLM tier;
            # remember the slot so we can fill it after the batch call.
            final_cat.append(None); source.append("pending")
            conf.append(0.0); correction.append(False)
            tail_idx.append(i)
            tail_rows.append({
                "name": r["name"],
                "median_amount": r.get("median_amount", 0) or 0,
                "ecommerce_rate": r.get("ecommerce_rate", 0) or 0,
            })

    cat["final_category"] = final_cat
    cat["source"] = source
    cat["confidence"] = conf
    cat["correction_flag"] = correction

    # --- Tier C / C': resolve the pending tail -------------------------------
    if tail_rows:
        llm_out = llm.classify_merchants(tail_rows)        # {} if no API key
        if llm_out:
            for idx, row in zip(tail_idx, tail_rows):
                res = llm_out.get(row["name"])
                if res:
                    cat.at[idx, "final_category"] = res["category"]
                    cat.at[idx, "confidence"] = res["confidence"]
                    cat.at[idx, "source"] = "llm"
        # Anything still unresolved (LLM off, or name missed) -> embed the name.
        still = cat.index[cat["source"] == "pending"].tolist()
        if still:
            names = cat.loc[still, "name"].fillna("unknown").astype(str).tolist()
            ncat, nconf = embed.map_texts_to_taxonomy(names)
            for idx, c, s in zip(still, ncat, nconf):
                cat.at[idx, "final_category"] = c
                cat.at[idx, "confidence"] = float(s)
                cat.at[idx, "source"] = "name_embedding"

    return cat


def apply_to_transactions(df: pd.DataFrame,
                          catalog: pd.DataFrame) -> pd.DataFrame:
    """
    Produce a per-transaction `clean_category` for all 1M rows:
      - non-merchant types (FEE/ATM/...) via the deterministic type rule,
      - merchant purchases via the merchant catalog category.
    """
    code_to_cat = dict(zip(catalog["transaction_merchants_code"],
                           catalog["final_category"]))
    code_to_src = dict(zip(catalog["transaction_merchants_code"],
                           catalog["source"]))

    type_cat = df["type"].map(TYPE_TO_CATEGORY)
    merch_cat = df["transaction_merchants_code"].map(code_to_cat)

    out = df.copy()
    # Merchant category wins for purchases; type rule covers the rest.
    out["clean_category"] = merch_cat.where(merch_cat.notna(), type_cat)
    out["clean_category"] = out["clean_category"].fillna("Other")
    out["category_source"] = df["transaction_merchants_code"].map(code_to_src)
    out.loc[out["category_source"].isna()
            & type_cat.notna(), "category_source"] = "type_rule"
    out["category_source"] = out["category_source"].fillna("default")
    return out
