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
├─ analytics.py       # SpendingAnalytics: deterministic chart queries (stats)
├─ engine.py          # InsightEngine: fit context once, build dashboard per user
├─ agent.py           # Financial Agent: Claude tool-calling chat over the engine
├─ api.py             # FastAPI stub the RN app calls
├─ run_demo.py        # CLI smoke test / demo
└─ detectors/
   ├─ decline_shield.py       # ★ predict & prevent the next insufficient-balance decline
   ├─ overspend_alert.py      # ★ personal per-category baseline + month-to-date projection
   ├─ subscription_radar.py   # new / forgotten / price-hiked recurring charges
   ├─ cashflow_forecast.py    # upcoming known charges (context tile)
   ├─ peer_benchmarking.py    # spend vs demographic cohort (guarded against tiny-cohort artifacts)
   └─ fx_fee_leakage.py       # fees + foreign-currency exposure (capped — ~absent in data)
```

★ = the two preventive heroes, grounded in the strongest *real* signals in the
data (77% of users hit an insufficient-balance decline; personal overspend is
the brief-named insight). The other three are supporting context tiles —
ranking (`ranking.py`) puts the heroes first.

## Financial Agent (chat layer)
A Claude (`claude-opus-4-8`) tool-calling agent over the engine. It explains
insights and takes simulated actions; every number comes from a tool, never the
model. Read-only tools auto-run; `set_budget` / `cancel_subscription` are gated
behind a confirmation. Runs with `ANTHROPIC_API_KEY` set; degrades cleanly
without it (the detectors/dashboard are unaffected).
```bash
uv pip install anthropic && export ANTHROPIC_API_KEY=...
uv run python -m insight_engine.agent --user <owner_id>
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

df = load_enriched("output/df_enriched.parquet")
engine = InsightEngine(df)                 # fits cohort baselines once
payload = engine.dashboard(user_id)        # JSON-ready dict for React Native
```

## Spending analytics (chart queries)
Deterministic stats for the dashboard's spending-history page — separate from
the insight detectors. Available via `engine.analytics` / `engine.<method>` or
the `/charts/...` endpoints.

```python
engine.spending_history(user_id, months=6)
# -> { categories[],                         # canonical stack/color order
#      history: { months[], series{cat:[..]}, totals[] },   # one bar per month
#      forecast: { month, by_category{}, total, is_forecast } }  # ghost bar

engine.category_momentum(user_id, months=6)
# -> { window[], categories: [ { category, last_month, prev_month,
#        mom_pct_change, trend_pct_per_month, direction: up|down|flat|new,
#        monthly[] } ] }                      # month-on-month rate of change
```
`spending_history` gives the stacked monthly bars **plus** a damped-linear-trend
forecast "ghost bar" for next month, all sharing one category ordering so the
front-end can color segments consistently. `category_momentum` is the
goal-tracking signal (per-category MoM % change + window trend). Both are also
embedded in `dashboard()` under `charts`.

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

## Categorizer handoff (now wired)
The Categorization Engine ships `output/df_enriched.parquet` with the full
contract already stamped: a clean 12-category `category`, stable `merchant_id`
(`m_00000…`), `subcategory`, `confidence`, and a learned `is_recurring` flag.
The engine reads it directly — no config needed:
```python
df = load_enriched("output/df_enriched.parquet")   # contract columns kept as-is
```
Detectors group on the stable `merchant_id` but display
`transaction_merchants_name`. Subscription Radar also trusts the categorizer's
`is_recurring` flag as an acceptance signal.

`load_enriched` stays backward-compatible: if any contract column is missing
(e.g. running on the raw cleaned parquet), it falls back to the placeholder MCC
mapper in `taxonomy.py`. To point at a differently-named clean category column:
```python
df = load_enriched("enriched.parquet", category_col="pfc_label")
```

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
