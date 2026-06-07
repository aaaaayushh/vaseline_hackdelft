# Spending IQ — Findings & Approach

**Revolut AI Challenge · HackDelft 2026**

A framework and working proof-of-concept that turns raw card transactions into financial
intelligence: a categorisation engine that produces a clean, budget-ready category for
every transaction, an insight engine that surfaces the single most important thing a user
should act on, and a conversational agent that explains it and takes the action. This
document records what we found in the data and the approach we built on top of it. All
figures below are measured from the code and data in this repository, not estimated.

---

## 1. Summary

Revolut's spending analytics are only as good as the categories underneath them, and the
dataset makes the problem concrete: the shipped `category` column is a one-to-one copy of
the raw MCC code — **396 granular labels** ("Record Shops", "Drinking Places", "Misc. Food
Stores") that nobody budgets by — and MCC is **absent on 3.3%** of rows and routinely stale
for new merchants in production. Unusable categories produce useless budgets.

We built three components over a single data contract:

1. **Categorisation Engine** — a cheapest-confident-tier-first cascade (deterministic
   transaction-type rules and ISO-MCC pins → Dutch merchant-name keywords → MCC-description
   embeddings mapped to taxonomy anchors → a Claude LLM tier for the hard tail) that
   collapses the 396 raw MCC labels onto a **16-category consumer taxonomy** and assigns a
   concrete category to **98.3%** of transactions. It runs at the merchant level (4,605
   unique merchants) and joins back to all 1,000,000 rows, the way a production system
   caches merchant categories.

2. **Insight Engine** — a modular library of detectors over the categorised table, with a
   ranking layer that emits one non-spammy feed. The lead insight, **Decline Shield**, is
   preventive: it fuses a user's real insufficient-balance decline history with the
   predicted date of their next recurring charge.

3. **Financial Agent** — a Claude (`claude-opus-4-8`) tool-calling advisor that explains
   insights and takes gated, simulated actions. Every number it quotes comes from a tool
   over the data; the model reasons and explains but never computes the figures itself.

The throughline is honesty. The categoriser **never silently overrides** the MCC — where a
strong name signal disagrees with the MCC category it raises a `correction_flag` rather than
guessing, because on synthetic data MCC is the ground truth. And on the insight side, we
mined every candidate against the full dataset and kept only what the data actually
supports, capping or cutting the rest rather than faking a signal that isn't there.

---

## 2. The data, and what we found in it

The data is loaded and cleaned by `engine_cat/clean.py`; the categorised output is
`results_cat/transactions_categorised.parquet`. Measured directly:

| Property | Value |
|---|---|
| Transactions | 1,000,000 |
| Distinct users (`owner_id`) | 17,841 |
| Distinct merchants (`transaction_merchants_code`) | 4,605 |
| Date range | 2025-07-01 → 2025-12-31 (6 months) |
| Real spend rows (completed purchase/ATM, amount > 0) | 772,961 (77.3%) |
| Rows with no MCC (structural: ATM/FEE/credits) | 3.3% |
| Data quality | 16,268 zero-amount rows; 0 impossible date orders; 0 duplicate IDs — flagged, not dropped |

**Finding 1 — the category column is the MCC, not a budgeting taxonomy.** `category` is a
deterministic copy of the MCC description: 396 distinct values. Because the merchants are
synthetic, **MCC is effectively the ground truth** — there are no "wrong" labels to correct.
The problem is not accuracy; it is that 396 labels are the wrong *vocabulary* for budgeting,
and MCC is not always present. Our job is normalisation and MCC-independent categorisation.

**Finding 2 — one budgeting concept is scattered across many MCCs, and the engine surfaces
it as corrections.** The dataset's single largest merchant is the digital-domain
`gracht.app` (98,059 transactions), and merchants like it (`molen.app`, `brug.app`,
`stoep.app`, …) carry MCCs such as "Record Shops", "Computer Software Stores", and
"Business Services" — which map to **Shopping / Services**, not a digital-subscription
concept. Rather than silently re-map them, the engine keeps the MCC category and raises a
`correction_flag`: **34 merchants flagged**, led by exactly these `.app` merchants. This is
the "supermarket-tagged-as-Shopping" scenario the brief names, made concrete and auditable.

**Finding 3 — the strongest, most universal signal in the data is failed payments.** This
is the finding that shaped our hero insight:

- **19.2%** of transactions are `DECLINED`.
- **75.1%** of those declines are for **insufficient balance** (14.4% of all transactions).
- **73.4%** of users (13,094 of 17,841) hit at least one insufficient-balance decline.

No other behavioural signal in this dataset is close to that universal *and* that
actionable.

**Finding 4 — some "classic" fintech insights are simply absent here.** We measured them
and they do not hold up: foreign-exchange markup nets to roughly nothing, fees are
negligible, and there is no recurring-income signal to forecast a balance against (the
dataset has no salary cadence). We treat those as context tiles or cut them entirely rather
than dramatise a number that isn't real (see §4.4 and §7).

**Finding 5 — spending is merchant-concentrated.** Categorising at the merchant level (4,605
merchants → joined to 1M rows) is both cheaper and the right unit, and it means a user's own
history is strongly predictive of how an ambiguous merchant should be categorised for *that*
user — headroom a production system would exploit with per-user history features.

---

## 3. Component 1 — Categorisation Engine

Implemented in `engine_cat/`. It rebuilds categories from the signals a user actually sees —
merchant name, MCC description, transaction type, amount — onto a taxonomy designed for
budgeting, and validates the result with silver-label checks since there is no external
ground truth.

### 3.1 Taxonomy (`taxonomy.py`)

A **16-category consumer taxonomy**: Groceries, Restaurants & Takeaway, Transport, Travel,
Shopping, Bills & Utilities, Entertainment & Digital, Health & Pharmacy, Personal Care,
Cash & ATM, Fees & Charges, Income, Refunds & Reversals, Gambling, Services, Other. Each
category carries a densely-written natural-language **anchor sentence** describing what
belongs in it (e.g. Groceries → "Supermarkets, grocery stores, convenience stores, food
markets, bakeries, butchers …"). The anchors are what the embedding tier maps into, so the
mapping is semantic rather than a hand-maintained lookup, and it generalises to MCCs never
seen before.

The module also defines three rule layers used by the cascade: `TYPE_TO_CATEGORY`
(structural transaction types → category, for the 3.3% no-MCC rows), `NAME_KEYWORDS`
(high-precision Dutch head-words like *Supermarkt*, *Tankstation*, *Apotheek* → category),
and `MCC_OVERRIDES` (≈30 unambiguous ISO-18245 codes pinned deterministically to correct
known embedding word-traps — e.g. fuel "Service Stations" being pulled toward "Services" by
the word *service*).

### 3.2 The cascade (`categorize.py`, `embed.py`, `llm.py`)

Categorisation runs once per merchant and is resolved cheapest-confident-tier-first:

```
Tier 0  MCC pin          ~30 unambiguous ISO codes pinned deterministically       (confidence 1.0)
Tier A  Name keyword     high-precision Dutch head-words                          (confidence 0.90)
Tier B  MCC embedding    embed the MCC description, take the nearest taxonomy       (confidence = cosine
                         anchor by cosine similarity                                similarity; floor 0.30)
Tier C  LLM (Claude)     classify the residual hard tail from the merchant name +  (confidence from model)
                         light context, structured output
Tier C' Name embedding   deterministic fallback when the LLM tier is off (no key)
```

Each merchant takes the category from the highest-confidence applicable tier. The embedding
tier (`embed.py`) uses `sentence-transformers/all-MiniLM-L6-v2`: it embeds each taxonomy
anchor once, embeds the input text (an MCC description or a merchant name), and assigns the
nearest anchor by cosine similarity — and that **similarity doubles as a confidence score**,
so low-similarity assignments (below 0.30) are exactly the ones escalated to the LLM tier.

The LLM tier (`llm.py`) is Claude `claude-opus-4-8` with **structured outputs** (a Pydantic
schema), batched at 40 merchants per request, classifying only the residual tail. If no
`ANTHROPIC_API_KEY` is configured it reports itself unavailable and the cascade falls back
to deterministic name-embedding (Tier C′), so the whole pipeline runs end-to-end offline.

**Where each category actually came from** (measured, LLM tier off → name-embedding fallback):

| Tier | Merchants (of 4,605) | Transactions (of 1,000,000) |
|---|---|---|
| MCC embedding | 3,577 (78%) | 883,225 (88%) |
| MCC pin (`mcc_rule`) | 552 (12%) | 70,733 (7%) |
| Type rule (structural) | — | 33,303 (3%) |
| Name embedding (fallback) | 476 (10%) | 12,739 (1%) |

### 3.3 Correction detection — flag, don't silently override

The defining design choice: where a confident MCC mapping disagrees with a strong name
signal, the engine **keeps the MCC category and raises a `correction_flag`** rather than
overriding it. On synthetic data MCC is the ground truth, so silently "fixing" it would be
dishonest; instead the disagreements become the interesting, auditable output. **34
merchants** are flagged, dominated by the `.app` digital-domain merchants whose MCCs
("Record Shops", "Computer Software Stores", "Business Services") map to Shopping/Services
while their names clearly read as Entertainment & Digital. This is precisely the
miscategorisation the brief describes, surfaced for human/agent review.

### 3.4 The handoff to the rest of the system

`apply_to_transactions` produces a per-transaction `clean_category` (merchant category for
purchases; the deterministic type rule for the non-merchant rows) plus a `category_source`.
The Insight Engine's loader (`insight_engine/contract.py`) consumes the category column and
**canonicalises** the 16-category labels onto the 12 budgeting categories its detectors use
(e.g. *Restaurants & Takeaway* → *Dining & Takeaway*, *Entertainment & Digital* → *Digital &
Subscriptions*), and synthesises the remaining contract fields it needs
(`merchant_id`, `is_recurring`, `amount_signed_gbp`) if they are absent — so the two engines
stay decoupled across one stable contract.

### 3.5 Evaluation — honest about synthetic data

There is **no external ground-truth category** (the provided `category` is just the MCC
description), so we do not run a supervised accuracy contest and we never claim to "beat" or
"fix" MCC. We validate with silver-label checks (`run_categorize.py`, results in
`results_cat/report.md`):

- **Taxonomy compression:** 396 raw MCC categories → **16** consumer categories.
- **Coverage:** **98.3%** of transactions assigned a concrete category (17,212 left as
  *Other*) — including the 3.3% no-MCC rows a pure lookup cannot touch.
- **Name↔MCC agreement:** on the 552 merchants with an informative name, the name-derived
  category independently agrees with the MCC-embedding category **95.3%** of the time. Two
  independent signals agreeing this often is the strongest validation available without
  ground truth.
- **Corrections surfaced:** **34** merchants where the name contradicts the MCC (§3.3) — the
  disagreements are the point.
- **Unknown-MCC recovery:** 32 of 39 merchants with no usable MCC were still categorised from
  their name.

The final spend distribution (completed spend) is led by **Groceries 26.9%, Restaurants &
Takeaway 17.1%, Shopping 17.0%, Transport 15.5%, Entertainment & Digital 9.3%** — a
budget-ready money map, where the raw data offered only 396 MCC descriptions.

---

## 4. Component 2 — Insight Engine

Implemented as a standalone package in `insight_engine/`. It consumes the categorised table
and emits dashboard-ready JSON.

### 4.1 Framework

Every insight is a pluggable **detector** implementing one interface:

```python
class InsightDetector:
    type: str
    def fit(self, ctx: EngineContext) -> None: ...          # optional population-level precompute
    def detect(self, user_df, ctx) -> list[Insight]: ...     # per-user

Insight = { type, user_id, title, explanation, severity (0..1), payload, actions }
        + derived: level (info/notice/warning/alert), insight_id (stable, for dedup)
```

`EngineContext` holds the full frame and the reference date, and runs each detector's
`fit()` once (e.g. peer-benchmarking cohort baselines computed across the whole population).
`InsightEngine.dashboard(user_id)` runs every detector for a user, ranks the results, and
packages a payload with a `hero` (the one insight that drives the push notification), the
ranked `insights` feed, per-type `sections`, and `charts`. A detector that raises is
skipped, so one failure cannot take down the dashboard.

### 4.2 Ranking — one feed, not notification spam

`ranking.py` scores each candidate by `severity × type_prior`, drops empties, dedupes by a
stable `insight_id`, and returns priority order. The type priors deliberately put the two
preventive, real-signal detectors first:

```
decline_shield 1.25 · overspend_alert 1.15 · subscription_radar 1.05
cashflow_forecast 0.90 · peer_benchmarking 0.85 · fx_fee_leakage 0.80
```

### 4.3 The detectors

**Decline Shield** (`decline_shield.py`) — *the hero.* The only preventive detector. It
combines two signals: the user's real insufficient-balance decline history (count, recency,
whether it recurs across months) and the predicted date of their next recurring charge
(reusing the cashflow detector's cadence logic). We have **no account balance** in the data,
so it deliberately never fabricates a "you are £X short" figure — it states the measured
decline history and the predicted charge, and recommends a top-up before that date. Example:
*"You've had 8 payments declined for insufficient balance (6 months running) and Eetcafe
Blauwe (£16.55) is due in 3 days. Top up to avoid another one."*

**Overspend Alert** (`overspend_alert.py`) — the insight the brief names. For each user ×
category it learns a **personal** baseline (median of prior complete months) and a robust
dispersion (median absolute deviation), projects the current month from how much of it has
elapsed, and fires only when the projection is materially above *that user's own* baseline
(default +30% and robust-z ≥ 1.0) on a well-sampled, stable category. Comparing to the
user's own history — not a population average — means a naturally heavy spender is not
nagged. It also excludes declined transactions, which never actually spent money.

**Subscription Radar** (`subscription_radar.py`) — detects recurring charges via either a
regular cadence (low interval variability) or a digital/subscription merchant with stable
amounts, then classifies each as new / price-hiked / forgotten / active. Built to be
conservative because synthetic cadence is noisy.

**Cashflow Forecast** (`cashflow_forecast.py`) — projects known recurring charges due in the
next 30 days from cadence. Presented as a **context tile**, not an alarm: because the data
has no recurring income, we explicitly do **not** headline a "projected deficit" (there is no
inflow to net against). It feeds the upcoming-charge date into Decline Shield.

**Peer Benchmarking** (`peer_benchmarking.py`) — compares a user's per-category monthly
spend to their demographic cohort (age × region, with fallback). To avoid the
small-denominator artefact where a near-zero cohort median manufactures a fake "40× your
peers," it requires a meaningful cohort median and user spend before flagging and caps the
reported ratio.

**FX & Fee Leakage** (`fx_fee_leakage.py`) — reports fees and foreign-currency exposure, with
severity **capped at 0.5** so it can never be the hero. This reflects Finding 4: fees and FX
markup are negligible in this dataset, so it stays a quiet context tile rather than a
dramatised number.

### 4.4 Analytics layer

`analytics.py` is a separate, deterministic query layer for the dashboard charts — no ML, no
LLM, fully auditable. It produces the monthly stacked-bar spend history, a damped
linear-trend forecast "ghost bar" for next month (damped toward the recent mean so a single
spike can't blow up the projection), and per-category month-on-month momentum. It is kept
distinct from the detectors: detectors find noteworthy *events*; analytics answers *chart
queries*.

### 4.5 Design principle: cut or cap what the data doesn't support

The detector lineup is the direct expression of Findings 3 and 4. We lead with the two
detectors grounded in the strongest real signals (decline prevention, personal overspend);
we keep subscription radar conservative; and we demote cashflow, cap FX, and guard peer
benchmarking so the system never surfaces a confident number the data cannot back. This is a
deliberate stance, encoded in the ranking priors and the per-detector thresholds.

---

## 5. Component 3 — Financial Agent (the AI advisor)

Implemented in `insight_engine/agent.py`. A Claude tool-calling agent bound to one user's
data, using `claude-opus-4-8` with adaptive thinking.

- **Computed, never hallucinated.** Every figure the agent states comes from a tool backed
  by the deterministic engine — `get_top_insights`, `explain_insight`, `category_spending`,
  `upcoming_charges`. The model plans and explains; it does not invent numbers.
- **Manual tool-use loop with gated actions.** Read-only tools auto-execute. The two
  state-changing tools, `set_budget` and `cancel_subscription`, are gated behind an explicit
  confirmation before they "apply" (simulated here). This is exactly the kind of
  hard-to-reverse action that warrants human-in-the-loop confirmation.
- **Prompt caching.** The stable prefix (system prompt + tool definitions) is cached; only
  the per-turn messages vary, so the cache prefix isn't invalidated between turns.
- **Graceful degradation.** With no API key or SDK installed, `available()` returns false
  and the rest of the pipeline (detectors, dashboard) is unaffected — the agent is the chat
  layer, not a dependency of the analytics.

A representative exchange: the user asks *"why do I keep getting declined?"*; the agent calls
`explain_insight("decline_shield")` and answers with the measured decline history and the
next predicted charge; the user asks what to trim; the agent calls the overspend and
spending tools, identifies the one stretched category, and offers a one-tap, confirmation-
gated budget action.

---

## 6. User interaction

The ranking layer's purpose is that the user receives **one** push, not a stream of alerts:
the ranked hero insight becomes the notification. Tapping it opens the agent with that
context already loaded, where the user can ask *why*, get a plan, and take a gated action in
the same conversation. The dashboard (`dashboard/revolut_insights_dashboard.html`) mirrors
this — a lock-screen push, the Decline Shield hero, an agent chat panel, and the supporting
tiles in ranked order, all populated from real engine output for one user.

---

## 7. Business impact & feasibility

**Impact.** Clean, personal categories make budgets trustworthy, which is the precondition
for any insight or advisor to be believed. The hero insight attacks a concrete Revolut cost
directly: 73.4% of users hit an insufficient-balance decline, and a preventive nudge before
the next charge reduces failed-payment friction and the churn that follows. The framing
shift matters — from "here is where your money went" (retrospective, like everyone else) to
"we saw a problem coming and helped you avoid it" (preventive).

**Feasibility.** Categorisation runs at the merchant level (4,605 merchants, not 1M rows)
and resolves ~99% of merchants with deterministic rules, ISO pins, and local embeddings at
near-zero marginal cost; only the uncertain tail reaches the LLM, batched and cached by
merchant. The insight engine is plain pandas over one categorised table, and the analytics
layer is fully deterministic. The agent's expensive component (the LLM) is invoked only in
conversation, not on every transaction. This is a deployable shape, not a demo-only toy.

---

## 8. Limitations (owned, not hidden)

- **No external ground truth.** "Agreement" is against the MCC-derived silver label, not
  truth — which is exactly why the *disagreements* (the 34 corrections) are the interesting
  output. On synthetic data MCC is the ground truth, so the categoriser's value is
  normalisation (396 → 16), coverage of the no-MCC rows, and surfacing corrections — never a
  claim to "beat" MCC.
- **Synthetic artifacts.** "Record Shops" is a large MCC standing in for streaming, and some
  generic merchant names are reused across unrelated MCCs; the engine leans on structural
  signals (head-words, ISO codes, amount) rather than over-fitting these.
- **Synthetic recurrence is noisy**, so Subscription Radar is intentionally conservative.
- **Per-category monthly spend is volatile**, so Overspend Alert fires only on well-sampled,
  stable categories with a robust threshold to avoid false alarms.
- **FX, fees, and recurring income are effectively absent** in this data — surfaced as
  context or cut, never faked.
- **No live balance data**, so Decline Shield flags the risk window and predicted charge
  rather than inventing a shortfall amount.

---

## 9. Repository map

| Path | What it is |
|---|---|
| `engine_cat/` | the categorisation engine — `clean.py` (load/clean + merchant catalog), `taxonomy.py` (16-category taxonomy + anchors + rules), `embed.py` (MiniLM → nearest anchor), `llm.py` (Claude tail classifier), `categorize.py` (the cascade + correction detection) |
| `run_categorize.py` | end-to-end run + silver-label evaluation report |
| `results_cat/` | generated outputs: `merchant_categories.csv`, `corrections.csv`, `transactions_categorised.parquet`, `report.md` |
| `insight_engine/` | detectors, ranking, analytics, engine orchestration, API, and the agent |
| `insight_engine/agent.py` | the Claude tool-calling advisor |
| `dashboard/revolut_insights_dashboard.html` | the demo UI (push → hero → agent → tiles) |

**Run:**

```bash
# categorise -> merchant catalog + categorised transactions + report
python run_categorize.py            # add ANTHROPIC_API_KEY to enable the LLM tier

# insight engine demo (dashboard JSON for one user)
uv run python -m insight_engine.run_demo --user <owner_id>

# the agent (needs ANTHROPIC_API_KEY + `anthropic`)
uv run python -m insight_engine.agent --user <owner_id>
```
