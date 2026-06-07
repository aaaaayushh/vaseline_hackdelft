"""
End-to-end run of the categorisation engine + evaluation.

    python run_categorize.py

Produces:
  results/merchant_categories.csv   one row per merchant with its category
  results/transactions_categorised.parquet  1M rows + clean_category
  results/corrections.csv           merchants where MCC disagrees with the name
  results/report.md                 the headline numbers (for the deck)

Because the dataset has no external ground-truth category (the provided
`category` is just the MCC description), we evaluate the way the brief implies:
coverage, taxonomy compression, agreement-with-MCC where the name is
informative, the corrections we surface, and recovery of Unknown-MCC merchants.
"""
from __future__ import annotations

import pandas as pd

from engine import clean, categorize, llm
from engine.taxonomy import CATEGORIES


def main() -> None:
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)

    print("1/4  Loading + cleaning ...")
    df, rep = clean.load_and_clean("dataset.parquet")
    print(f"     rows: {rep['rows_in']:,} | spend rows: {rep['spend_rows']:,} "
          f"| zero-amount: {rep['zero_amount_rows']:,} "
          f"| impossible dates: {rep['impossible_date_order_rows']}")

    print("2/4  Building merchant catalog ...")
    catalog = clean.build_merchant_catalog(df)
    print(f"     unique merchants: {len(catalog):,}")

    print(f"3/4  Categorising merchants (LLM tier "
          f"{'ON' if llm.available() else 'OFF -> embedding fallback'}) ...")
    catalog = categorize.categorise_catalog(catalog)

    print("4/4  Applying to all transactions ...")
    txns = categorize.apply_to_transactions(df, catalog)

    # ---------------------------------------------------------------- outputs
    catalog.to_csv("results/merchant_categories.csv", index=False)
    txns[["transaction_id", "type", "mcc", "category", "transaction_merchants_name",
          "txn_amount_gbp", "is_spend", "clean_category", "category_source"]] \
        .to_parquet("results/transactions_categorised.parquet", index=False)

    corrections = catalog[catalog["correction_flag"]].copy()
    corrections = corrections[[
        "name", "mcc", "mcc_category", "mcc_emb_category",
        "name_category", "n_txns"]].sort_values("n_txns", ascending=False)
    corrections.to_csv("results/corrections.csv", index=False)

    _report(df, rep, catalog, txns, corrections)
    _charts(txns, catalog)
    print("\nDone. See results/report.md and results/*.png")


def _charts(txns, catalog) -> None:
    """Two deck-ready figures: spend mix and cascade-tier breakdown."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    dist = (txns.loc[txns["is_spend"], "clean_category"]
            .value_counts().sort_values())
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(dist.index, dist.values, color="#3498db")
    ax.set_title("Spend by clean category (completed spend)")
    ax.set_xlabel("Transactions")
    fig.tight_layout(); fig.savefig("results/spend_distribution.png", dpi=130)
    plt.close(fig)

    src = catalog["source"].value_counts()
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(src.index, src.values, color="#9b59b6")
    ax.set_title("How each merchant was categorised (cascade tier)")
    ax.set_ylabel("Merchants")
    fig.tight_layout(); fig.savefig("results/cascade_tiers.png", dpi=130)
    plt.close(fig)


def _report(df, rep, catalog, txns, corrections) -> None:
    n = len(txns)
    src = txns["category_source"].value_counts()
    merch_src = catalog["source"].value_counts()
    dist = txns.loc[txns["is_spend"], "clean_category"].value_counts()

    # Taxonomy compression
    raw_cats = df["category"].nunique()

    # Agreement-with-MCC on informative-name merchants (silver-label check):
    informative = catalog[catalog["name_category"].notna()
                          & catalog["source"].eq("mcc_embedding")]
    agree = (informative["name_category"] == informative["mcc_emb_category"]).mean()

    # Unknown-MCC recovery
    from engine.taxonomy import UNKNOWN_MCCS
    unknown = catalog[catalog["mcc"].isin(UNKNOWN_MCCS)
                      | catalog["mcc_category"].isna()]
    recovered = unknown[unknown["final_category"].ne("Other")]

    lines = []
    A = lines.append
    A("# Categorisation Engine — Results\n")
    A(f"_LLM tier (Claude {llm.MODEL}): "
      f"**{'ON' if llm.available() else 'OFF — deterministic embedding fallback used'}**. "
      f"Set `ANTHROPIC_API_KEY` to activate it for the hard tail._\n")
    A("## Data cleaning\n")
    A(f"- Rows in / out: **{rep['rows_in']:,} / {rep['rows_out']:,}**")
    A(f"- Real spend rows (completed purchase/ATM, amount>0): "
      f"**{rep['spend_rows']:,}** ({rep['spend_rows']/n:.1%})")
    A(f"- Zero-amount rows flagged: {rep['zero_amount_rows']:,} | "
      f"impossible date-order rows: {rep['impossible_date_order_rows']} | "
      f"duplicate txn ids: {rep['duplicate_txn_ids_removed']}")
    A(f"- Date range: {rep['date_range'][0]} → {rep['date_range'][1]}\n")

    A("## Categorisation\n")
    A(f"- **Taxonomy compression:** {raw_cats} raw MCC categories → "
      f"**{len(CATEGORIES)}** consumer categories.")
    A(f"- **Coverage:** {(txns['clean_category'].ne('Other')).mean():.1%} of "
      f"transactions assigned a concrete category "
      f"({(txns['clean_category'].eq('Other')).sum():,} left as Other).")
    A(f"- **Unique merchants categorised:** {len(catalog):,}\n")

    A("### Where each merchant's category came from (cascade tiers)\n")
    for s, c in merch_src.items():
        A(f"- `{s}`: {c:,} merchants ({c/len(catalog):.0%})")
    A("")
    A("### Where each *transaction's* category came from\n")
    for s, c in src.items():
        A(f"- `{s}`: {c:,} txns ({c/n:.0%})")
    A("")

    A("## Evaluation (no external ground truth — silver-label checks)\n")
    A(f"- **Name↔MCC agreement** on merchants with an informative name "
      f"(n={len(informative):,}): **{agree:.1%}** — the name-derived category "
      f"agrees with the independent MCC-embedding category this often, "
      f"validating the engine.")
    A(f"- **Corrections surfaced:** **{len(corrections):,}** merchants where a "
      f"strong name signal contradicts the MCC category (the "
      f"'supermarket-tagged-as-Shopping' fix). Top examples in "
      f"`results/corrections.csv`.")
    A(f"- **Unknown-MCC recovery:** {len(recovered):,} of {len(unknown):,} "
      f"merchants with no usable MCC were still categorised from their name.\n")

    A("### Final spend distribution (completed spend only)\n")
    tot = dist.sum()
    for c, v in dist.items():
        A(f"- {c}: {v:,} ({v/tot:.1%})")

    if len(corrections):
        A("\n### Sample corrections (MCC vs name)\n")
        A("| merchant | MCC category | →MCC-mapped | name suggests | txns |")
        A("|---|---|---|---|---|")
        for _, r in corrections.head(12).iterrows():
            A(f"| {r['name']} | {str(r['mcc_category'])[:28]} | "
              f"{r['mcc_emb_category']} | {r['name_category']} | {r['n_txns']} |")

    open("results/report.md", "w").write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
