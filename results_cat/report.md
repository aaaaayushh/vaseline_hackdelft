# Categorisation Engine — Results

_LLM tier (Claude claude-opus-4-8): **OFF — deterministic embedding fallback used**. Set `ANTHROPIC_API_KEY` to activate it for the hard tail._

## Data cleaning

- Rows in / out: **1,000,000 / 1,000,000**
- Real spend rows (completed purchase/ATM, amount>0): **772,961** (77.3%)
- Zero-amount rows flagged: 16,268 | impossible date-order rows: 0 | duplicate txn ids: 0
- Date range: 2025-07-01 00:00:04 → 2025-12-31 23:59:55.862247

## Categorisation

- **Taxonomy compression:** 396 raw MCC categories → **16** consumer categories.
- **Coverage:** 98.3% of transactions assigned a concrete category (17,212 left as Other).
- **Unique merchants categorised:** 4,605

### Where each merchant's category came from (cascade tiers)

- `mcc_embedding`: 3,577 merchants (78%)
- `mcc_rule`: 552 merchants (12%)
- `name_embedding`: 476 merchants (10%)

### Where each *transaction's* category came from

- `mcc_embedding`: 883,225 txns (88%)
- `mcc_rule`: 70,733 txns (7%)
- `type_rule`: 33,303 txns (3%)
- `name_embedding`: 12,739 txns (1%)

## Evaluation (no external ground truth — silver-label checks)

- **Name↔MCC agreement** on merchants with an informative name (n=552): **95.3%** — the name-derived category agrees with the independent MCC-embedding category this often, validating the engine.
- **Corrections surfaced:** **34** merchants where a strong name signal contradicts the MCC category (the 'supermarket-tagged-as-Shopping' fix). Top examples in `results/corrections.csv`.
- **Unknown-MCC recovery:** 32 of 39 merchants with no usable MCC were still categorised from their name.

### Final spend distribution (completed spend only)

- Groceries: 207,686 (26.9%)
- Restaurants & Takeaway: 132,556 (17.1%)
- Shopping: 131,214 (17.0%)
- Transport: 119,610 (15.5%)
- Entertainment & Digital: 72,045 (9.3%)
- Services: 20,593 (2.7%)
- Travel: 17,069 (2.2%)
- Other: 15,722 (2.0%)
- Cash & ATM: 12,852 (1.7%)
- Gambling: 11,520 (1.5%)
- Personal Care: 9,071 (1.2%)
- Bills & Utilities: 8,863 (1.1%)
- Fees & Charges: 8,637 (1.1%)
- Health & Pharmacy: 5,153 (0.7%)
- Income: 312 (0.0%)
- Refunds & Reversals: 58 (0.0%)

### Sample corrections (MCC vs name)

| merchant | MCC category | →MCC-mapped | name suggests | txns |
|---|---|---|---|---|
| gracht.app | Record Shops | Shopping | Entertainment & Digital | 98059 |
| molen.app | Computer Software Stores | Shopping | Entertainment & Digital | 4139 |
| gracht.app | Computer Software Stores | Shopping | Entertainment & Digital | 3300 |
| brug.app | Computer Software Stores | Shopping | Entertainment & Digital | 2899 |
| molen.app | Computer Software Stores | Shopping | Entertainment & Digital | 2059 |
| stoep.app | Computer Software Stores | Shopping | Entertainment & Digital | 1885 |
| stoep.app | Computer Software Stores | Shopping | Entertainment & Digital | 956 |
| winkel.app | Computer Software Stores | Shopping | Entertainment & Digital | 795 |
| gracht.app | Business Services, Not Elsew | Services | Entertainment & Digital | 632 |
| winkel.app | Miscellaneous Personal Servi | Services | Entertainment & Digital | 191 |
| gracht.app | Business Services, Not Elsew | Services | Entertainment & Digital | 121 |
| stoep.app | Miscellaneous Personal Servi | Services | Entertainment & Digital | 107 |