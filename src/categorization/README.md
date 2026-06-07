# Categorization Engine (Component 1)

Maps every transaction onto the **12-category budgeting taxonomy** and stamps the data
contract the Insight Engine + Agent consume. Implements DESIGN.md §4 / §8.

## Pipeline

```
type-rules (Layer 0)  ──┐
                        ├──►  category, subcategory, confidence, cat_source,
LightGBM   (Layer 1)  ──┤      merchant_id, is_recurring, amount_signed_gbp, needs_llm
LLM fallback (Layer 2)──┘                          → output/df_enriched.parquet
```

| Module | Role |
|---|---|
| `taxonomy.py`  | 12-category taxonomy + deterministic `mcc_to_pfc()` (ISO-18245 ranges) + Layer-0 `TYPE_RULES` |
| `labels.py`    | label = `mcc_to_pfc(mcc)` for merchant rows, `TYPE_RULES` for structural rows |
| `features.py`  | static + merchant char-hash + **temporal-safe** user-history + GMM persona prior |
| `train.py`     | two honest evals (production time-split / capability held-out-merchants) |
| `categorize.py`| the full cascade → writes the enriched contract for all 1M rows |

## Run

```bash
python -m src.categorization.train         # eval -> output/categorizer_eval.json
python -m src.categorization.categorize    # enrich -> output/df_enriched.parquet
```

## Results (full 1M)

| Eval | Setup | Accuracy | Baseline |
|---|---|---|---|
| **Production** | time-split (train Jul–Oct / test Nov–Dec), MCC prior + history | **1.000** | — |
| **Capability** | held-out merchants, **MCC withheld**, behaviour-only | **0.472** | 0.249 (majority) → **1.9×** |

Label-free (all 1M): **100% coverage** (incl. the 3.3% no-MCC rows via Layer 0),
**73%** merchant→single-category consistency (the 27% multi-MCC tail is the ambiguity
personalization resolves), merchant rows reproduce MCC-truth **100%**.

Reading: on synthetic data MCC *is* the ground truth, so with MCC we reproduce it
perfectly (we never claim to "beat" it). The real result is **capability**: the model
recovers the budgeting category from merchant structure + behaviour at ~2× baseline when
MCC is withheld on unseen merchants — the production case where MCC is missing.

## Two key engineering notes

1. **Encoding split.** The production/enrichment model uses **numeric** feature encoding
   (the near-deterministic `mcc_prior` is isolated cleanly by numeric splits); the
   capability model uses **categorical** encoding with default regularization (genuine
   grouping of `merchant_country`/persona/prev-category generalizes to unseen merchants).
   LightGBM's categorical-split smoothing over-regularizes the deterministic prior when
   many categoricals coexist at 1M scale — numeric sidesteps it.
2. **Temporal integrity.** Every user-history feature is a within-group time-ordered
   shift / expanding aggregate — a row only ever sees that user's past (DESIGN.md §4.4).

## Caveat

Because MCC is deterministic here, Layer-1 confidence is ~1.0 and the Layer-2 LLM
fallback fires on 0 rows in this dataset. In production (MCC missing/stale) the
capability-style model's lower confidence (~0.94) is what routes the uncertain tail to
the batched, cached Haiku fallback (DESIGN.md §4.2).
