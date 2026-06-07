# Revolut AI Challenge — Categorisation Engine

A hybrid **rules + embeddings + LLM** cascade that turns raw card transactions
into a clean, consumer-friendly spending taxonomy — the "better than MCC"
categoriser the brief asks for.

## Why this design

The dataset's `category` column is a 1:1 lookup from `mcc` (the verbose
ISO-18245 description). That is the *broken baseline* Revolut describes: 396
granular labels like "Record Shops" or "Non-durable Goods, Not Elsewhere
Classified", occasionally wrong, useless for budgeting. There is no external
ground-truth category to predict — so this is a **framework + PoC** task, not a
supervised accuracy contest. We rebuild categories from the signals a user
actually sees and validate the result with silver-label checks.

## The cascade (cheapest confident tier first)

| Tier | Mechanism | Covers |
|---|---|---|
| 0 · type rule | `FEE/ATM/CARD_CREDIT/...` → category | 3% non-merchant rows (no MCC) |
| 0 · MCC pin | ~30 unambiguous ISO codes pinned deterministically | fixes embedding word-traps (fuel "Service Stations" → Transport) |
| A · name keyword | high-precision Dutch head words (Snackbar, Tankstation…) | fills Unknown-MCC, cross-checks MCC |
| B · MCC embedding | embed MCC description → nearest taxonomy anchor (cosine) | the ~88% backbone |
| C · LLM (Claude `claude-opus-4-8`) | classify the hard tail from the name + structured output | Unknown-MCC / low-confidence merchants |
| C' · name embedding | deterministic fallback when no `ANTHROPIC_API_KEY` | keeps the pipeline runnable offline |

Categorisation runs at the **merchant level** (~4.6k unique merchant codes, the
true entity key) and joins back to the 1M transactions — the way a production
system caches merchant categories.

Where a confident MCC mapping disagrees with a strong name signal, we **don't
silently override** — we keep the category and raise a `correction_flag`. That
is exactly the "supermarket-tagged-as-Shopping" fix the brief calls for.

## Results (latest run, LLM tier off → embedding fallback)

- **Taxonomy compression:** 396 MCC categories → **16** consumer categories.
- **Coverage:** ~98% of transactions get a concrete category.
- **Name↔MCC agreement:** **95.3%** on merchants with an informative name —
  the name-derived category independently agrees with the MCC-embedding
  category this often, which validates the engine without ground truth.
- **Corrections surfaced:** 34 merchants where the MCC contradicts a strong
  name signal (e.g. `*.app` digital merchants miscoded as "Record Shops"/
  "Computer Software Stores" → corrected to Entertainment & Digital).

## Run

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install pandas pyarrow numpy scikit-learn matplotlib sentence-transformers anthropic
python run_categorize.py            # add ANTHROPIC_API_KEY to enable the LLM tier
```

Outputs land in `results/`: `merchant_categories.csv`, `corrections.csv`,
`transactions_categorised.parquet`, `report.md`, and two PNG charts.

## Layout

```
engine/
  taxonomy.py     16-category taxonomy + anchors, ISO pins, name keywords, type rules
  clean.py        load + clean + flag artifacts; build merchant catalog
  embed.py        sentence-transformer → nearest taxonomy anchor (+confidence)
  llm.py          Claude classifier for the tail (structured outputs); graceful fallback
  categorize.py   the cascade + correction detection + join to transactions
run_categorize.py end-to-end run, evaluation report, charts
```

## Honest limitations (for Q&A)

- No external ground truth: "agreement" is vs the MCC-derived silver label, not
  truth. We own this — it's why the corrections (where we *dis*agree with MCC)
  are the interesting output.
- Synthetic artifacts: "Record Shops" is the 2nd-biggest MCC (a streaming proxy)
  and generic merchant names ("Witte Molen") are reused across unrelated MCCs.
  The engine leans on structural signals (descriptive head words, ISO codes,
  amount) rather than over-fitting these.
