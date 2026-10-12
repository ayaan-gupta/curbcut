"""Rate limits and outlier checks. Each check returns a flag; the service decides what a flag costs."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timedelta

from .config import Settings
from .geo import haversine_m
from .graph import Graph
from .models import Feature, Report, ReportIn
from .store import Store


class RateLimited(Exception):
    def __init__(self, message: str, retry_after_s: int) -> None:
        super().__init__(message)
        self.retry_after_s = retry_after_s


def device_hash(raw_id: str, salt: str) -> str:
    """Devices are stored as a salted hash, never the id the phone sent."""
    return hashlib.sha256(f"{salt}:{raw_id}".encode()).hexdigest()[:20]


def check_rate(store: Store, device: str, now: datetime, settings: Settings, adding: int = 1) -> None:
    day = store.reports_by_device(device, now - timedelta(days=1))
    recent = [r for r in day if (r.received_at or r.created_at) >= now - timedelta(minutes=10)]
    if len(recent) + adding > settings.rate_limit_10min:
        oldest = min((r.received_at or r.created_at) for r in recent) if recent else now
        wait = max(1, int((oldest + timedelta(minutes=10) - now).total_seconds()))
        raise RateLimited(f"This device has sent {len(recent)} reports in the last ten minutes. The limit is {settings.rate_limit_10min}.", wait)
    if len(day) + adding > settings.rate_limit_day:
        raise RateLimited(f"This device has sent {len(day)} reports in the last day. The limit is {settings.rate_limit_day}.", 3600)


@dataclass(frozen=True)
class Screening:
    status: str  # accepted, review or rejected
    flags: list[str]


def screen(report: ReportIn, device: str, observed: datetime, store: Store, graph: Graph, settings: Settings, nearby: Feature | None) -> Screening:
    """Outlier rejection.

    Rejected outright: outside the pilot area, or a GPS fix too loose to place
    within a 5 m cluster. Held for review: implausible travel since the
    device's last report, a point far from any footway, or a report that
    contradicts a feature several people already agree on.
    """
    flags: list[str] = []
    if not graph.contains(report.lon, report.lat):
        return Screening("rejected", ["outside_area"])
    if report.accuracy_m > settings.max_accuracy_m:
        return Screening("rejected", ["low_gps_accuracy"])

    # Compare with whichever of this device's reports was observed closest in time, before or after.
    others = [r for r in store.reports_by_device(device, observed - timedelta(days=1, hours=1)) if r.status != "rejected"]
    if others:
        last = min(others, key=lambda r: abs((observed - r.created_at).total_seconds()))
        seconds = max(1.0, abs((observed - last.created_at).total_seconds()))
        metres = haversine_m(report.lon, report.lat, last.lon, last.lat)
        # Allow for both fixes being off by their stated accuracy before calling it teleporting.
        slack = report.accuracy_m + last.accuracy_m
        if (metres - slack) / seconds > settings.max_speed_mps:
            flags.append("implausible_speed")

    if graph.nearest_edge(report.lon, report.lat, settings.max_distance_from_path_m) is None:
        flags.append("far_from_any_path")

    if nearby is not None and nearby.published and nearby.reporters >= 3 and nearby.confidence >= 0.7 and nearby.state != report.state:
        if not (report.category == "curb" and report.state == "blocked"):  # a parked car is allowed to appear overnight
            flags.append("contradicts_established_feature")

    return Screening("review" if flags else "accepted", flags)
