# Insight Engine

Modular insight detectors over the categorized transaction table. Emits typed
`Insight` objects, ranks them, and packages a per-user **dashboard JSON** for
the React Native app (built later). See `DESIGN.md` §5.

## Layout
```
insight_engine/
├─ contract.py        # data contract in/out: load_enriched(), Insight, Severity
├─ taxonomy.py        # placeholder MCC→12-category mapper (until categorizer ships)
├─ base.py            # InsightDetector ABC, EngineContext
├─ ranking.py         # severity ranking + dedup → non-spammy feed
├─ engine.py          # InsightEngine: fit context once, build dashboard per user
├─ api.py             # FastAPI stub the RN app calls
├─ run_demo.py        # CLI smoke test / demo
└─ detectors/
   ├─ subscription_radar.py   # new / forgotten / price-hiked recurring charges
   ├─ fx_fee_leakage.py       # fees + foreign-currency exposure (Revolut angle)
   ├─ cashflow_forecast.py    # upcoming charges + projected 30-day net flow
   └─ peer_benchmarking.py    # spend vs demographic cohort
```

## Quick start
```bash
# CLI demo (most-active user)
uv run python -m insight_engine.run_demo --out output/demo_dashboard.json

# specific user
uv run python -m insight_engine.run_demo --user <owner_id>

# HTTP API for the dashboard
uv run uvicorn insight_engine.api:app --reload
#   GET /dashboard/{user_id}   GET /insights/{user_id}   GET /users   GET /health
```

```python
from insight_engine import load_enriched, InsightEngine

df = load_enriched("output/df_clean_clean.parquet")
engine = InsightEngine(df)                 # fits cohort baselines once
payload = engine.dashboard(user_id)        # JSON-ready dict for React Native
```

## Dashboard payload
```jsonc
{
  "user_id": "...",
  "generated_at": "2025-12-31T23:59:...",
  "hero": { ...top insight... },        // drives the push notification
  "insights": [ { ...card... }, ... ],  // priority order
  "sections": { "subscription_radar": {...}, "fx_fee_leakage": {...},
                "cashflow_forecast": {...}, "peer_benchmarking": {...} }
}
```
Each card: `{ type, user_id, title, explanation, severity, level, payload, actions, insight_id }`.

## Categorizer handoff
The categorizer is built separately. Until it ships, `load_enriched` synthesizes
`category` / `merchant_id` / `is_recurring` from the cleaned data via a
placeholder mapper. When it's ready:
```python
df = load_enriched("enriched.parquet", category_col="category_pfc")
```
No detector code changes.

## Add a detector
Subclass `InsightDetector`, set `type`, implement `detect` (and optionally
`fit`), then add it to `detectors/__init__.py:default_detectors()`.
```python
class MyDetector(InsightDetector):
    type = "my_detector"
    def detect(self, user_df, ctx) -> list[Insight]:
        ...
        return [Insight(type=self.type, user_id=..., title=..., explanation=...,
                        severity=..., payload={...})]
```
