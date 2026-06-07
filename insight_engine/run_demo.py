"""Demo / smoke-test runner for the Insight Engine.

    uv run python -m insight_engine.run_demo                 # random active user
    uv run python -m insight_engine.run_demo --user <uuid>   # specific user
    uv run python -m insight_engine.run_demo --out demo.json # write payload

Picks an active user, runs all four detectors, and prints the dashboard JSON.
"""

from __future__ import annotations

import argparse
import json

from .contract import load_enriched
from .engine import InsightEngine


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--parquet", default="output/df_clean_clean.parquet")
    ap.add_argument("--category-col", default=None,
                    help="clean category column from the categorizer, if available")
    ap.add_argument("--user", default=None, help="owner_id to inspect")
    ap.add_argument("--out", default=None, help="write dashboard JSON here")
    args = ap.parse_args()

    print(f"Loading {args.parquet} ...")
    df = load_enriched(args.parquet, category_col=args.category_col)
    print(f"  {len(df):,} txns · {df['owner_id'].nunique():,} users")

    print("Fitting engine (cohort baselines) ...")
    engine = InsightEngine(df)

    if args.user:
        user_id = args.user
    else:
        # most-active user makes for the richest demo
        user_id = df["owner_id"].value_counts().index[0]
    print(f"Building dashboard for user {user_id}\n")

    payload = engine.dashboard(user_id)
    text = json.dumps(payload, indent=2)
    print(text)

    if args.out:
        with open(args.out, "w") as f:
            f.write(text)
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
