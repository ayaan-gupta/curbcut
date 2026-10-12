"""Demo reports for the pilot area.

EVERYTHING HERE IS SYNTHETIC. No person observed any of it. The positions are
real OpenStreetMap kerb nodes and footways, but the reports were invented so a
fresh install has something on the map, a detour to show, a review queue and
an export draft. Every seeded report and every feature built from one carries
synthetic=true, the UI labels them, and the OSM export leaves them out unless
asked, in which case the file says DO NOT UPLOAD.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta

from .geo import to_xy
from .models import ReportIn
from .service import Service, utcnow

# A walk across the pilot area that the seeded missing curb cut sits on.
DEMO_ORIGIN = (-122.320207, 47.618278)
DEMO_DESTINATION = (-122.319026, 47.614324)
DEMO_MISSING_CURB = (-122.3200002, 47.6153571)

M_PER_DEG_LAT = 111_320.0


def _jitter(rng: random.Random, lon: float, lat: float, metres: float = 1.4) -> tuple[float, float]:
    dx, dy = rng.uniform(-metres, metres), rng.uniform(-metres, metres)
    return lon + dx / (M_PER_DEG_LAT * 0.674), lat + dy / M_PER_DEG_LAT


def _far_from_paths(service: Service) -> tuple[float, float]:
    """A point inside the pilot area at least 35 m from every footway (the middle of a block)."""
    b = service.graph.bbox
    steps = 40
    for i in range(1, steps):
        for j in range(1, steps):
            lon = b["west"] + (b["east"] - b["west"]) * i / steps
            lat = b["south"] + (b["north"] - b["south"]) * j / steps
            if service.graph.nearest_edge(lon, lat, 35.0) is None:
                return lon, lat
    return (b["west"] + b["east"]) / 2, (b["south"] + b["north"]) / 2


def seed_demo(service: Service, now: datetime | None = None) -> dict[str, int]:
    now = now or utcnow()
    rng = random.Random(2026)
    device = 0

    def report(category: str, state: str, where: tuple[float, float], source: str, days_ago: float, conf: float = 0.9, who: str | None = None, **extra: object) -> None:
        nonlocal device
        device += 1
        lon, lat = _jitter(rng, *where)
        service.submit(
            ReportIn(category=category, state=state, lon=lon, lat=lat, accuracy_m=rng.uniform(3, 7), source=source, model_confidence=conf, **extra),  # type: ignore[arg-type]
            raw_device=who or f"seed-device-{device}",
            now=now - timedelta(days=days_ago),
            synthetic=True,
            enforce_rate=False,
        )

    # 1. A missing curb cut on the demo walk: three people photographed it.
    for days in (19, 8, 2):
        report("curb", "missing", DEMO_MISSING_CURB, "photo", days, conf=rng.uniform(0.82, 0.95))
    # 2. A ramp blocked by a parked car, seen twice this week.
    for days in (1.5, 0.4):
        report("curb", "blocked", (-122.3194259, 47.6151649), "photo", days, conf=0.9)
    # 3. A broken ramp.
    for days, source in ((40, "photo"), (12, "photo"), (3, "manual")):
        report("curb", "damaged", (-122.3235133, 47.6163198), source, days, conf=0.86)
    # 4. Curb cuts confirmed present. The last one is at a kerb OpenStreetMap tags as raised.
    for where, votes in (((-122.3196059, 47.615213), 3), ((-122.3233663, 47.617628), 2), ((-122.3234787, 47.6151398), 2), ((-122.3235471, 47.6177903), 3), ((-122.320025, 47.6167118), 2)):
        for i in range(votes):
            report("curb", "present", where, "photo", 5 + 9 * i, conf=rng.uniform(0.85, 0.97))
    # 5. One unsure photo of a missing curb cut. Alone it must not publish.
    report("curb", "missing", (-122.3233744, 47.6151431), "photo", 1, conf=0.55)
    # 6. Rough brick sidewalk felt by three phones, and a second stretch felt by two (not enough yet).
    for days in (30, 11, 4):
        report("surface", "rough", (-122.3209774, 47.615823), "sensor", days, conf=rng.uniform(0.75, 0.92))
    for days in (9, 2):
        report("surface", "rough", (-122.323332, 47.6146276), "sensor", days, conf=0.8)
    # 7. A bump three phones agree on.
    for days in (21, 6, 1):
        report("surface", "bump", (-122.3209875, 47.6165711), "sensor", days, conf=rng.uniform(0.8, 0.95), value=rng.uniform(9, 16))
    # 8. Disagreement: two say there is a curb cut, one says there is not. Goes to review.
    for state, days in (("present", 14), ("missing", 6), ("present", 2)):
        report("curb", state, (-122.3234136, 47.6177903), "photo", days, conf=0.9)
    # 9. A steep block measured three times with the inclinometer.
    for days in (16, 9, 5):
        report("slope", "steep", (-122.3224814, 47.6174957), "manual", days, conf=0.9, value=rng.uniform(10.5, 11.5))
    # 10. A footway with no surface tag in OpenStreetMap, reported as paving stones by three people.
    untagged = sorted((e for e in service.graph.edges.values() if e.surface is None and e.kind in ("sidewalk", "footway") and e.length > 20), key=lambda e: e.id)
    if untagged:
        mid = untagged[0].coords[len(untagged[0].coords) // 2]
        for days in (25, 10, 3):
            report("surface", "rough", (mid[0], mid[1]), "manual", days, conf=0.9, material="paving_stones")
    # 11. Two things for the review queue: a device that jumps 700 m in five seconds, and a point in the middle of a block.
    report("curb", "missing", (-122.3162, 47.6188), "photo", 0.01, who="seed-teleporter")
    device += 1
    lon, lat = _jitter(rng, -122.3245, 47.6140)
    service.submit(
        ReportIn(category="curb", state="missing", lon=lon, lat=lat, accuracy_m=5, source="photo", model_confidence=0.9),
        raw_device="seed-teleporter",
        now=now - timedelta(days=0.01) + timedelta(seconds=5),
        synthetic=True,
        enforce_rate=False,
    )
    far = _far_from_paths(service)
    report("surface", "bump", far, "sensor", 0.5, value=12.0)

    counts = service.store.counts()
    counts["devices"] = device
    return counts


def demo_points() -> dict[str, tuple[float, float]]:
    return {"origin": DEMO_ORIGIN, "destination": DEMO_DESTINATION, "missing_curb": DEMO_MISSING_CURB}


__all__ = ["seed_demo", "demo_points", "to_xy"]
