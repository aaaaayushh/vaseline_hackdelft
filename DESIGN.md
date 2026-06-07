# Spending IQ — Design Document
### Revolut AI Challenge · HackDelft 2026

> **Vision:** Turn raw card transactions into a financial co-pilot. We don't just *show* users where money went — we categorize it accurately and personally, surface the one insight that matters this week, and let them act on it in a conversation.

This document is the master framework. It covers the three required components — **Categorization Logic**, **Insight Engine**, **User Interaction** — plus the evaluation methodology, tech stack, and the deck/exec-summary skeleton.

---

## 1. Executive Summary (the 1-page deliverable)

**Problem.** Revolut's spending analytics are only as good as transaction categorization — and categorization is hard. The dataset ships a `category` column, but it is a 1:1 copy of the raw MCC code: **396 messy, overlapping labels** ("Record Shops", "Misc. Food Stores") that nobody budgets by. Worse, **~26% of card spend** goes to digital merchants scattered across half a dozen wrong labels, and **3.3% of transactions have no MCC at all** (ATM, fees, transfers) so they can't be categorized by lookup. Bad categories → useless budgets → users who don't understand their money.

**Our solution — Spending IQ, three engines on one data contract:**

1. **Categorization Engine.** A clean, well-known cascade — global rules → gradient-boosted classifier → LLM fallback — that maps every transaction (incl. the no-MCC ones) onto a **12-category budgeting taxonomy** (Plaid-standard). It is **personalized**: the model learns each user's history (their typical category, amount, and timing per merchant) and a behavioral persona, so the *same* merchant is categorized correctly *for that user*. We respect temporal integrity — only the past predicts the future.

2. **Insight Engine.** A modular library of insight detectors over the categorized data, with a ranking layer that surfaces only the top, non-spammy insight. **Hero insight: proactive overspend alerts** — "You're 40% over your usual Dining this month, and it's only the 18th."

3. **Financial Agent.** A Claude-powered, tool-calling advisor. It answers questions with **computed (never hallucinated) numbers**, explains insights, runs what-if plans, and takes **simulated actions** (set a budget, cancel a subscription). It remembers your goals across sessions.

**Why it wins.** Accurate + personal categorization is the foundation; the insight + agent layer is the differentiated product. Numbers are always computed deterministically; the LLM only reasons and explains. The architecture is cost-aware (bulk categorization is pennies; the LLM touches only the uncertain ~3%), which makes it genuinely feasible at Revolut scale.

---

## 2. The data reality (grounded, not assumed)

From our exploration + cleaning notebook (`01_explore_and_clean.ipynb`):

- **1,000,000 transactions · ~18,000 users · Jul–Dec 2025** (~55 txns/user — enough to build per-user history).
- **`category` is a deterministic function of `mcc`** → the shipped categories *are* the raw, messy MCC labels (396 of them). This is the problem to solve, not a ground truth to copy.
- **~25.7% of card spend** is digital merchants (`gracht.app`, `tulp.nl/abonnement`, …) spread across "Record Shops", "Digital Goods", etc.
- **3.3% of rows have no MCC** (ATM/FEE/CHARGE/etc.) — uncategorizable by lookup; our model still categorizes them.
- **Data quality:** 28.5% of rows carry ≥1 anomaly flag (largest: a card-present/`Not PRESENT` conflict at 19.2%; FX markup on home-currency GBP; Pareto-tail outliers). We **flag, don't delete**, and added a normalized `amount_signed_gbp` (credits negative, debits positive) for clean cashflow math.

**Implication for the framework:** there is **no objective ground-truth category** (merchants are synthetic; MCC is itself a noisy proxy). So we do **not** chase a leaderboard accuracy number. We treat the data as a realistic sandbox: mine insights, prove the pipeline runs at scale, and validate *capability* honestly (see §6).

---

## 3. Architecture overview — three engines, one contract

```
   cleaned txns          ┌──────────────────┐   enriched txns      ┌──────────────┐   tools + context   ┌──────────────┐
 (synthetic_*_clean) ──▶ │  CATEGORIZATION  │ ───────────────────▶ │   INSIGHT    │ ───────────────────▶│  FINANCIAL   │
                         │      ENGINE      │  + category          │   ENGINE     │  ranked insight     │    AGENT     │
                         └──────────────────┘  + subcategory       └──────────────┘  objects             └──────────────┘
                                  ▲            + confidence               │                                    │
                                  │            + merchant_id              │                                    │
                          user corrections     + is_recurring            └──────── both queryable by agent ────┘
```

**The contract.** The categorizer stamps every transaction with `{category, subcategory, confidence, merchant_id, is_recurring}`. The Insight Engine consumes that enriched table and emits **typed insight objects** `{type, severity, user_id, payload, explanation}`. The Agent has tools to query *both* the enriched transactions and the insight objects. Each engine is independently testable and swappable.

---

## 4. Component 1 — Categorization Logic

### 4.1 Taxonomy (well-known, not invented)
Adopt the **Plaid Personal Finance Categories (PFC)** scheme: ~16 primary categories, hierarchical (primary → detailed). We collapse to a **12-category budgeting taxonomy** for the demo: Groceries, Dining & Takeaway, Transport, Shopping, Digital & Subscriptions, Bills & Utilities, Entertainment, Health, Travel, Cash & ATM, Fees & Charges, Income & Refunds. We maintain a deterministic **MCC → PFC map** as the prior.

### 4.2 The cascade (clean, reliable, well-known)
```
Layer 0  STRUCTURAL RULES  Deterministic, by transaction TYPE — early-exit, bypasses ML:
                             • ATM                         → Cash & ATM
                             • FEE / CHARGE                → Fees & Charges
                             • REFUND / CREDIT / CHARGEBACK → Income & Refunds
                           (covers the 3.3% no-MCC rows + money-in; no merchant to learn from)

Layer 1  ML CLASSIFIER     For every MERCHANT transaction (~97%, CARD_PAYMENT).
                           LightGBM on:
                             • MCC→PFC mapping  ← a FEATURE / PRIOR, not a gate (overridable)
                             • merchant char-TF-IDF
                             • amount, entry_method, channel, country, time
                             • USER-HISTORY FEATURES  (personalization)
                             • BEHAVIORAL-PERSONA prior (personalization)

Layer 2  LLM FALLBACK      Claude Haiku 4.5 + structured outputs, only when
                           Layer 1 confidence < threshold OR cold-start.
                           Results cached by merchant_id.
```
**MCC is a feature, not a gate.** A merchant transaction is *never* categorized by its MCC alone — the ML always runs and can **override** MCC using merchant name, amount, user history, and persona. This is precisely what lets us fix the ~26% misfiled digital spend (e.g. `gracht.app`, MCC 5735 "Record Shops" → **Digital & Subscriptions**); an early-exit-on-MCC gate would silently propagate that error. Only the structural, non-merchant *types* in Layer 0 get a pure deterministic rule and skip the ML.

Cost: Layers 0–1 handle ~97% of volume in microseconds at ~zero marginal cost, so there's no reason to gate merchant transactions on MCC. Only the uncertain tail (~3%) hits the LLM, via the **Batches API (50% off)** with `output_config.format` (strict JSON). Every LLM answer is cached → the long tail is paid for once.

### 4.3 Personalization — "history categorizes the future"
Two well-known mechanisms (the chosen approach):

**(a) User-history features in the ML model.** For each transaction, derive features from that user's *prior* transactions:
- user's modal/most-frequent category for this merchant
- user's category mix (share of spend per category)
- user's typical amount & amount-percentile for this merchant
- user's time-of-day / day-of-week pattern

This resolves the classic ambiguity — the *same* merchant means different things to different users (Albert Heijn at £80 weekly = Groceries; at £3 = a snack run). The model decides *relative to that user*.

**(b) Behavioral-persona priors.** Cluster users into spend personas (GMM — the same family the data was generated from, so personas recover cleanly). For **cold-start** users/merchants with little history, the persona supplies a sensible prior before personal history exists.

**Corrections close the loop.** When a user (or the agent) corrects a category, that correction updates the user's history features and is a high-weight training/relabel signal — so the system *learns each user's meaning* over time. This is also the agent's feedback hook (§7).

### 4.4 Temporal integrity (the technical-credibility point)
Because we use a user's past to predict their future, **history features are computed only from transactions strictly before the target**, and we **evaluate on a time-ordered split** (train Jul–Oct, test Nov–Dec) — no future leakage. Cold-start (unseen user/merchant) is evaluated separately to show the cascade degrades gracefully to the global model + LLM.

---

## 5. Component 2 — Insight Engine

The Insight Engine is implemented as a standalone Python package (`insight_engine/`) that consumes the enriched table and emits **dashboard-ready JSON** for the React Native app (built later). It is structured so each detector is independently testable and the dashboard payload is a stable contract the front-end codes against.

### 5.1 Framework: detectors → ranking → dashboard
Each insight is a pluggable **detector** implementing one interface and emitting a typed `Insight` object:
```
InsightDetector.fit(ctx)                       # optional population-level precompute (e.g. cohort baselines)
InsightDetector.detect(user_df, ctx) -> [ Insight ]

Insight = { type, user_id, title, explanation, severity (0..1), payload, actions }
         + derived: level (info/notice/warning/alert), insight_id (stable, for dedup/cooldown)
```
A **ranking layer** scores candidates by `severity × per-type prior`, applies dedup + cooldown, and orders them so the **top insight is the push-notification hero** and the rest fill the dashboard — anti-spam by construction.

The **engine** (`InsightEngine`) fits population context once (cohort baselines), then assembles a per-user dashboard:
```
dashboard(user_id) -> {
  user_id, generated_at,
  hero,                 # top-ranked insight → drives the push notification
  insights[],           # all cards in priority order
  sections{ type: card } # one card per detector, for fixed dashboard tiles
}
```
A thin **FastAPI stub** (`insight_engine/api.py`: `/dashboard/{user_id}`, `/insights/{user_id}`, `/users`, `/health`) exposes this for the RN app.

### 5.2 The four dashboard detectors
1. **Subscription Radar** — recurring charges, classified `new` / `price_hike` / `forgotten` / `active`, each with a `confidence`. Two acceptance paths handle noisy synthetic timing: a strict **cadence** path (regular interval, low variability) *and* a **merchant** path (categorized *Digital & Subscriptions* or named `abonnement`, with a stable amount). Reports monthly/annual subscription spend and offers cancel actions.
2. **FX & Fee Leakage** — the uniquely-Revolut detector. On this data the txn-vs-bill spread is symmetric noise (nets to ~0), so the robust signal is the explicit `fee_amount_gbp`: total fees, annualized projection, per-month trend, worst offenders, foreign-currency exposure, and **avoidable fees on home-currency (GBP) transactions** flagged explicitly.
3. **Cashflow Forecast** — net flow over a 30-day horizon (no balance available, so we forecast *flow*, honestly): predicted **upcoming recurring charges** with due dates, projected recurring **income**, and a **discretionary burn** run-rate. Fires when a deficit is projected.
4. **Peer Benchmarking** — per-category spend vs a demographic cohort (`age_group × region`, falling back to `age_group` when a cell is small). Reports ratio-to-median and percentile, surfacing the categories where the user most outspends peers. Cohort baselines are computed once in `fit`.

### 5.3 Categorization handoff (placeholder mapper)
The engine consumes the contract `{category, merchant_id, is_recurring, amount_signed_gbp, ...}`. Because the Categorization Engine is built separately, `load_enriched` tolerantly synthesizes any missing contract column from the cleaned substrate — notably a **placeholder MCC→12-category mapper** (`taxonomy.py`). When the real categorizer ships, point `load_enriched(category_col=...)` at its output and the placeholder is bypassed with **no detector changes**.

### 5.4 Breadth (future detectors)
The framework cleanly accommodates more detectors on the same interface: proactive overspend alert (month-to-date run-rate vs personal baseline), unusual-spend anomaly, and decline-before-payday.

---

## 6. Component 3 — Financial Agent (Claude)

### 6.1 Pattern: tool-calling agent with deterministic computation
**Claude API + tool use** (self-hosted loop, full control). The agent **plans and explains**; it never invents numbers — every figure comes from a tool.

- **Model:** `claude-opus-4-8` (agent reasoning/planning), **adaptive thinking**, `effort: "high"`.
- **Computation tools** (text-to-SQL / pandas over the enriched table): `query_transactions`, `category_breakdown`, `compare_periods`, `forecast_cashflow`, `simulate_budget`.
- **Insight tools:** the Insight Engine detectors exposed as tools (`get_top_insights`, `explain_insight`).
- **Action tools (gated):** `set_budget`, `cancel_subscription`, `recategorize` — **human-in-the-loop confirm** before executing (SDK manual loop / `always_ask`). `recategorize` writes back to the personalization layer (§4.3).
- **Structured outputs** (`output_config.format`): the agent returns typed UI cards the front-end renders cleanly.
- **Memory tool:** persists user goals/preferences across sessions ("remembers your £2k-by-December goal").
- **Prompt caching:** system prompt + taxonomy + tool defs cached → ~90% cheaper, faster demo.
- The SDK **tool runner** handles the loop; we drop to the manual loop only for the gated action confirm.

### 6.2 What the agent does
NL Q&A ("how much on takeout in November?") · multi-step analytics ("compare Q3 vs Q4 dining and explain") · explain any insight · what-if & goal planning · actionable recommendations with simulated actions · personalized monthly review narrative.

---

## 7. User Interaction (UX)

- **Push notification** — the ranked hero insight: *"💡 Dining is 40% above your usual — £340 vs £240. Tap to see why."*
- **Tap → agent opens with context** — explains the cause, proposes a fix, surfaces a forgotten subscription, offers a one-tap action.
- **Monthly summary screen** — the clean 12-category money map with MoM deltas; visual contrast of *before* (396 messy labels / "Record Shops") vs *after* (clean, personal).
- **Correction loop** — user can re-categorize anything; it feeds personalization. This is the moat: *the system learns your meaning of a merchant.*

Demo surface: Streamlit (or polished slides) showing push → chat → summary → action.

### Hero demo flow (~3 min, touches all three engines)
Push: *"Dining 40% above usual"* → tap → agent explains (3 new restaurants + one £80 dinner) → user: *"help me cut this"* → agent proposes a budget + flags a forgotten £9.99 subscription → user taps **Cancel** (simulated action) → confirmation.

---

## 8. Using synthetic data honestly — evaluation methodology

There is **no ground-truth category** (synthetic merchants; MCC is a noisy proxy). So we don't claim to "beat MCC accuracy." Instead:

- **Capability demonstration (real number):** treat MCC→PFC as a weak label; train with a **time-ordered split** and **evaluate on held-out merchants/months**. Reported as: *"the model recovers the right category from behavior + merchant structure even when MCC is absent/unreliable"* — which matters because 3.3% have no MCC and production MCC is noisy.
- **Label-free metrics (full 1M rows):** coverage (we categorize the no-MCC rows lookup can't), consistency (same merchant → same category), personalization lift (does adding user-history features change ambiguous-merchant assignments coherently per user).
- **Qualitative spot-check:** a small human/LLM-judged sample for credibility (with inter-annotator note), explicitly flagged as illustrative.

**Q&A answer:** *"On synthetic data with fictional merchants, MCC is effectively the ground truth — so we don't claim to beat it. We prove our categorizer generalizes to unseen merchants and covers transactions MCC can't, and that personalization changes assignments coherently per user."*

---

## 9. Tech stack
Python end-to-end · pandas + DuckDB (query layer) · scikit-learn + LightGBM (categorizer) · GMM (personas) · Anthropic SDK — `claude-opus-4-8` (agent), `claude-haiku-4-5` (categorizer fallback), Batches API (bulk), prompt caching, structured outputs, memory tool · Streamlit (UX mock). Everything reads one enriched parquet → clean handoffs, each engine demo-able alone.

---

## 10. Business impact & feasibility (the 30% axis)
- **Impact:** accurate, personal categories → trustworthy budgets → the insight + agent layer drives engagement and retention. FX/fee insight is uniquely Revolut.
- **Feasibility / cost:** deterministic rules + LightGBM handle ~97% of volume at ~zero marginal cost and millisecond latency; the LLM touches only the uncertain ~3%, batched at 50% off and cached. We do **not** run a million LLM calls a day. This is the deployable shape, not a demo toy.

---

## 11. Risks & limitations (own them)
- Synthetic recurrence is noisy → we don't over-promise clean subscription detection; overspend is the robust hero.
- No ground truth → capability + label-free evaluation, stated plainly.
- Card-present/FX anomalies are synthetic artifacts → flagged, not silently trusted.
- Cold-start users → persona priors + LLM fallback cover the gap.

---

## 12. Deck skeleton (10 min, ~10 slides) + build plan

**Slides**
1. **Hook** — "Revolut already has a category column. Here are 116,000 transactions it calls *Record Shops* that are actually app subscriptions." (the problem, in one slide)
2. **Vision** — Spending IQ: categorize → insight → act.
3. **Architecture** — three engines, one data contract (the diagram).
4. **Categorization** — taxonomy + cascade + personalization (history features + persona) + temporal integrity.
5. **How we evaluate without labels** — capability + label-free metrics (pre-empts the killer Q&A).
6. **Insight Engine** — detector framework + ranking; the overspend hero.
7. **Financial Agent** — tool-calling, computed-not-hallucinated numbers, actions.
8. **Live demo** — push → explain → plan → cancel (the 3-min flow).
9. **Feasibility at scale** — cost/latency story; Revolut fit (FX).
10. **Close** — before/after categories; what we'd build next.

**Q&A prep:** "it's synthetic, how do you score?" (§8) · "isn't MCC fine?" (1:1 with category, 26% misfiled, 3.3% missing) · "how is this personalized?" (§4.3) · "does the agent make up numbers?" (no — tools compute, §6.1) · "cost at scale?" (§10).

**PoC build order (after this doc):**
1. Taxonomy + MCC→PFC map.
2. Categorizer: rules → LightGBM with user-history features + persona prior; time-ordered split; capability + label-free eval.
3. Overspend detector + ranking layer.
4. Claude agent: computation tools + insight tools + gated actions + memory + structured-output cards.
5. Streamlit UX: push → chat → summary → action.

---

## Appendix — repo artifacts
- `01_explore_and_clean.ipynb` — exploration + cleaning; writes `synthetic_1000000_clean.parquet` (with `dq_*` flags + `amount_signed_gbp`).
- `synthetic_1000000_clean.parquet` — the cleaned substrate all engines read.
- `DESIGN.md` — this document.
