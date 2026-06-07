"""Ranking layer — turn raw detector output into a non-spammy feed.

Scores candidates by severity (with a small per-type relevance prior), drops
empty/duplicate cards, and returns them in priority order. The top insight is
the "hero" the push notification would carry; the rest populate the dashboard.
"""

from __future__ import annotations

from .contract import Insight

# A light prior so that, at equal severity, the more actionable insight wins.
_TYPE_PRIOR = {
    "subscription_radar": 1.05,
    "fx_fee_leakage": 1.0,
    "cashflow_forecast": 1.1,
    "peer_benchmarking": 0.9,
}


def score(insight: Insight) -> float:
    return insight.severity * _TYPE_PRIOR.get(insight.type, 1.0)


def rank(insights: list[Insight], *, top_n: int | None = None,
         min_severity: float = 0.0) -> list[Insight]:
    """Sort insights by score, drop those below ``min_severity``, dedup by
    ``insight_id``, and optionally truncate to ``top_n``."""
    seen: set[str] = set()
    deduped: list[Insight] = []
    for ins in insights:
        if ins.severity < min_severity:
            continue
        if ins.insight_id in seen:
            continue
        seen.add(ins.insight_id)
        deduped.append(ins)
    deduped.sort(key=score, reverse=True)
    return deduped[:top_n] if top_n else deduped
