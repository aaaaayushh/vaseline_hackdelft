"""
LLM tier — Claude as the high-accuracy categoriser for the hard tail.

In the cascade, most transactions are resolved cheaply (type rules + MCC
embedding mapping). Only the residual — merchants with an Unknown/missing MCC,
or a low-confidence embedding mapping — reaches this tier. Claude classifies the
merchant from its name (and light context: typical amount, e-commerce rate)
into the taxonomy, and returns a one-line reason that the insight engine can
later surface to the user ("we categorised X as Y because ...").

Model: claude-opus-4-8 with structured outputs (Pydantic schema), so the result
is always a valid taxonomy label. If no ANTHROPIC_API_KEY is configured the
tier reports itself unavailable and the cascade falls back to a deterministic
embedding-of-the-name mapping, so the whole pipeline still runs end-to-end.
"""
from __future__ import annotations

import os

from pydantic import BaseModel, Field

from .taxonomy import CATEGORIES

MODEL = "claude-opus-4-8"
_BATCH = 40  # merchants per request


class _MerchantLabel(BaseModel):
    name: str = Field(description="The merchant name, echoed back verbatim.")
    category: str = Field(description=f"One of: {', '.join(CATEGORIES)}")
    confidence: float = Field(description="0-1 confidence in the label.")
    reason: str = Field(description="Short (<=12 word) justification.")


class _Batch(BaseModel):
    labels: list[_MerchantLabel]


_SYSTEM = (
    "You categorise card-transaction merchants into a fixed personal-finance "
    "taxonomy for a banking app. The merchant names are Dutch. Use the name "
    "and any context to pick the single best category. You MUST choose from "
    "this exact list:\n" + "\n".join(f"- {c}" for c in CATEGORIES)
)


def available() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def classify_merchants(rows: list[dict]) -> dict[str, dict]:
    """
    rows: list of {"name", "median_amount", "ecommerce_rate"}.
    Returns {name: {"category", "confidence", "reason"}}.
    Empty dict if the tier is unavailable (no API key) — caller must fall back.
    """
    if not available() or not rows:
        return {}

    import anthropic
    client = anthropic.Anthropic()
    valid = set(CATEGORIES)
    out: dict[str, dict] = {}

    for i in range(0, len(rows), _BATCH):
        chunk = rows[i:i + _BATCH]
        listing = "\n".join(
            f"- {r['name']} (typical £{r.get('median_amount', 0):.0f}, "
            f"ecommerce={r.get('ecommerce_rate', 0):.0%})"
            for r in chunk
        )
        try:
            resp = client.messages.parse(
                model=MODEL,
                max_tokens=4000,
                system=_SYSTEM,
                messages=[{
                    "role": "user",
                    "content": "Categorise each merchant. Return one label per "
                               "merchant, echoing the name exactly:\n" + listing,
                }],
                output_format=_Batch,
            )
            parsed = resp.parsed_output
            if parsed is None:
                continue
            for lab in parsed.labels:
                cat = lab.category if lab.category in valid else "Other"
                out[lab.name] = {
                    "category": cat,
                    "confidence": float(lab.confidence),
                    "reason": lab.reason,
                }
        except Exception as e:  # never let the tail break the whole run
            print(f"  [llm] batch {i // _BATCH} failed: {e}")
            continue
    return out
