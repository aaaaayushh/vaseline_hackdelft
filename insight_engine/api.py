"""FastAPI stub the React Native dashboard will call.

This is a thin HTTP layer over ``InsightEngine`` — the app is built later, so
this exists to (a) pin the contract the front-end codes against and (b) be
runnable today for local development::

    uv run uvicorn insight_engine.api:app --reload

Env vars:
    INSIGHT_PARQUET   path to the enriched parquet (default: output/df_clean_clean.parquet)
    INSIGHT_CATEGORY  name of the clean category column once the categorizer ships
"""

from __future__ import annotations

import os
from functools import lru_cache

from .contract import load_enriched
from .engine import InsightEngine

try:
    from fastapi import FastAPI, HTTPException
except ImportError:  # pragma: no cover - fastapi is optional until the app is built
    FastAPI = None  # type: ignore


@lru_cache(maxsize=1)
def get_engine() -> InsightEngine:
    path = os.environ.get("INSIGHT_PARQUET", "output/df_clean_clean.parquet")
    category_col = os.environ.get("INSIGHT_CATEGORY") or None
    df = load_enriched(path, category_col=category_col)
    return InsightEngine(df)


def create_app():
    if FastAPI is None:
        raise RuntimeError("fastapi is not installed; add it to run the API")
    app = FastAPI(title="Spending IQ — Insight Engine", version="0.1.0")

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/users")
    def users(limit: int = 50):
        eng = get_engine()
        ids = list(eng._groups.groups.keys())[:limit]
        return {"count": len(ids), "user_ids": ids}

    @app.get("/dashboard/{user_id}")
    def dashboard(user_id: str, top_n: int | None = None):
        eng = get_engine()
        if user_id not in eng._groups.groups:
            raise HTTPException(status_code=404, detail="unknown user_id")
        return eng.dashboard(user_id, top_n=top_n)

    @app.get("/insights/{user_id}")
    def insights(user_id: str):
        eng = get_engine()
        if user_id not in eng._groups.groups:
            raise HTTPException(status_code=404, detail="unknown user_id")
        return {"user_id": user_id,
                "insights": [i.to_dict() for i in eng.run_user(user_id)]}

    return app


# Module-level app for `uvicorn insight_engine.api:app`.
app = create_app() if FastAPI is not None else None
