"""Where reports and features live.

`Store` is the interface. `MemoryStore` is the offline implementation the
tests and a keyless clone use. `PostgisStore` (store_postgis.py) is the real
one. Both pass the same test suite.
"""

from __future__ import annotations

import threading
from datetime import datetime
from typing import Protocol

from .geo import haversine_m
from .models import Feature, Report


class Store(Protocol):
    kind: str

    def add_report(self, report: Report) -> Report: ...
    def update_report(self, report: Report) -> None: ...
    def get_report(self, report_id: int) -> Report | None: ...
    def reports_for_feature(self, feature_id: int) -> list[Report]: ...
    def reports_by_device(self, device: str, since: datetime) -> list[Report]:
        """Reports from one device received at or after `since`, oldest observation first."""
        ...
    def list_reports(self, status: str | None = None, limit: int = 500) -> list[Report]: ...
    def nearest_feature(self, category: str, lon: float, lat: float, radius_m: float) -> tuple[Feature, float] | None: ...
    def add_feature(self, feature: Feature) -> Feature: ...
    def update_feature(self, feature: Feature) -> None: ...
    def get_feature(self, feature_id: int) -> Feature | None: ...
    def list_features(self, published: bool | None = None) -> list[Feature]: ...
    def set_export_decision(self, candidate_id: str, decision: str, incline_sign: int | None) -> None: ...
    def export_decisions(self) -> dict[str, tuple[str, int | None]]: ...
    def counts(self) -> dict[str, int]: ...
    def clear(self) -> None: ...


class MemoryStore:
    kind = "memory"

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._reports: dict[int, Report] = {}
        self._features: dict[int, Feature] = {}
        self._decisions: dict[str, tuple[str, int | None]] = {}
        self._next_report = 1
        self._next_feature = 1

    def add_report(self, report: Report) -> Report:
        with self._lock:
            saved = report.model_copy(update={"id": self._next_report})
            self._reports[saved.id] = saved
            self._next_report += 1
            return saved

    def update_report(self, report: Report) -> None:
        with self._lock:
            self._reports[report.id] = report.model_copy()

    def get_report(self, report_id: int) -> Report | None:
        found = self._reports.get(report_id)
        return found.model_copy() if found else None

    def reports_for_feature(self, feature_id: int) -> list[Report]:
        with self._lock:
            return [r.model_copy() for r in self._reports.values() if r.feature_id == feature_id]

    def reports_by_device(self, device: str, since: datetime) -> list[Report]:
        with self._lock:
            return sorted((r.model_copy() for r in self._reports.values() if r.device == device and (r.received_at or r.created_at) >= since), key=lambda r: r.created_at)

    def list_reports(self, status: str | None = None, limit: int = 500) -> list[Report]:
        with self._lock:
            rows = [r.model_copy() for r in self._reports.values() if status is None or r.status == status]
            return sorted(rows, key=lambda r: r.id, reverse=True)[:limit]

    def nearest_feature(self, category: str, lon: float, lat: float, radius_m: float) -> tuple[Feature, float] | None:
        with self._lock:
            best: tuple[Feature, float] | None = None
            for f in self._features.values():
                if f.category != category:
                    continue
                d = haversine_m(lon, lat, f.lon, f.lat)
                if d <= radius_m and (best is None or d < best[1]):
                    best = (f.model_copy(), d)
            return best

    def add_feature(self, feature: Feature) -> Feature:
        with self._lock:
            saved = feature.model_copy(update={"id": self._next_feature})
            self._features[saved.id] = saved
            self._next_feature += 1
            return saved

    def update_feature(self, feature: Feature) -> None:
        with self._lock:
            self._features[feature.id] = feature.model_copy()

    def get_feature(self, feature_id: int) -> Feature | None:
        found = self._features.get(feature_id)
        return found.model_copy() if found else None

    def list_features(self, published: bool | None = None) -> list[Feature]:
        with self._lock:
            return sorted((f.model_copy() for f in self._features.values() if published is None or f.published == published), key=lambda f: f.id)

    def set_export_decision(self, candidate_id: str, decision: str, incline_sign: int | None) -> None:
        with self._lock:
            self._decisions[candidate_id] = (decision, incline_sign)

    def export_decisions(self) -> dict[str, tuple[str, int | None]]:
        with self._lock:
            return dict(self._decisions)

    def counts(self) -> dict[str, int]:
        with self._lock:
            return {
                "reports": len(self._reports),
                "features": len(self._features),
                "published": sum(1 for f in self._features.values() if f.published),
                "in_review": sum(1 for r in self._reports.values() if r.status == "review"),
            }

    def clear(self) -> None:
        with self._lock:
            self._reports.clear()
            self._features.clear()
            self._decisions.clear()
            self._next_report = 1
            self._next_feature = 1
