"""Confidence of a feature from the reports clustered into it.

Three things move the score, as the PRD asks:

- independent reporters: each device gets one vote, its most recent report in
  the cluster, however many times it reports
- agreement: the share of the total vote weight behind the winning state
- recency: a vote halves in weight every half-life, which is short for things
  that come and go (a parked car) and long for things that are poured in
  concrete

    weight      = source * model confidence * GPS accuracy * 0.5 ** (age / half life)
    support     = sum of weights behind the winning state
    agreement   = support / sum of all weights
    confidence  = agreement * (1 - exp(-support / K))

K is 2, so one perfect vote alone gives 0.39 and can never reach the 0.5
publishing threshold. Two good photo votes that agree give about 0.59. A
sensor vote counts for less, so it takes three passes that agree.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime

from .models import Report

K = 2.0
SOURCE_WEIGHT = {"photo": 1.0, "manual": 0.8, "sensor": 0.65}
DEFAULT_HALF_LIFE_DAYS = 365.0
HALF_LIFE_DAYS = {
    ("curb", "blocked"): 14.0,
    ("curb", "damaged"): 180.0,
    ("surface", "rough"): 240.0,
    ("surface", "bump"): 240.0,
    ("surface", "smooth"): 240.0,
}
# A reviewer's sign-off is recorded on the feature, not as a vote, so it is not in here.


@dataclass(frozen=True)
class Score:
    state: str
    confidence: float
    agreement: float
    reporters: int
    support: float
    by_state: dict[str, float]


def vote_weight(report: Report, now: datetime) -> float:
    age_days = max(0.0, (now - report.created_at).total_seconds() / 86400.0)
    half_life = HALF_LIFE_DAYS.get((report.category, report.state), DEFAULT_HALF_LIFE_DAYS)
    recency = 0.5 ** (age_days / half_life)
    accuracy = max(0.4, min(1.0, 8.0 / max(report.accuracy_m, 0.1)))
    return SOURCE_WEIGHT[report.source] * max(0.05, min(1.0, report.model_confidence)) * accuracy * recency


def score(reports: list[Report], now: datetime) -> Score | None:
    """Scores the accepted reports of one cluster. Returns None if there are none."""
    latest: dict[str, Report] = {}
    for r in reports:
        if r.status != "accepted":
            continue
        if r.device not in latest or r.created_at > latest[r.device].created_at:
            latest[r.device] = r
    if not latest:
        return None
    by_state: dict[str, float] = {}
    for r in latest.values():
        by_state[r.state] = by_state.get(r.state, 0.0) + vote_weight(r, now)
    # Ties go to the alphabetically first state so the result does not depend on dict order.
    state = min(by_state, key=lambda s: (-by_state[s], s))
    support = by_state[state]
    total = sum(by_state.values())
    agreement = support / total if total > 0 else 0.0
    confidence = agreement * (1.0 - math.exp(-support / K))
    return Score(
        state=state,
        confidence=round(confidence, 4),
        agreement=round(agreement, 4),
        reporters=len(latest),
        support=round(support, 4),
        by_state={k: round(v, 4) for k, v in by_state.items()},
    )
