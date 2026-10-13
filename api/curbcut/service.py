"""Report intake: screen, cluster within 5 m, rescore, publish above the threshold."""

from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone

from . import abuse, confidence
from .config import Settings
from .graph import Condition, Graph, build_conditions
from .models import COUNTER_STATES, Feature, Report, ReportIn, SubmitResult
from .store import Store


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Service:
    def __init__(self, store: Store, graph: Graph, settings: Settings) -> None:
        self.store = store
        self.graph = graph
        self.settings = settings
        # Clustering reads then writes. One writer at a time keeps two reports that
        # arrive together from each starting a feature. A multi-process deploy would
        # need the same thing as a Postgres advisory lock.
        self._lock = threading.Lock()
        self._conditions: dict[str, Condition] | None = None

    # Intake

    def submit(self, report_in: ReportIn, raw_device: str, now: datetime | None = None, photo: str | None = None, synthetic: bool = False, enforce_rate: bool = True) -> SubmitResult:
        now = now or utcnow()
        device = abuse.device_hash(raw_device, self.settings.device_salt)
        with self._lock:
            if enforce_rate:
                abuse.check_rate(self.store, device, now, self.settings)
            # A phone may queue reports while offline, so it says when each was observed.
            # That time is trusted only within the last day and never into the future.
            observed = min(now, max(now - timedelta(days=1), report_in.recorded_at)) if report_in.recorded_at else now
            near = self.store.nearest_feature(report_in.category, report_in.lon, report_in.lat, self.settings.cluster_radius_m)
            screening = abuse.screen(report_in, device, observed, self.store, self.graph, self.settings, near[0] if near else None)
            report = self.store.add_report(
                Report(
                    id=0,
                    device=device,
                    category=report_in.category,
                    state=report_in.state,
                    lon=report_in.lon,
                    lat=report_in.lat,
                    accuracy_m=report_in.accuracy_m,
                    source=report_in.source,
                    model_confidence=report_in.model_confidence,
                    value=report_in.value,
                    material=report_in.material,
                    photo=photo,
                    created_at=observed,
                    received_at=now,
                    status=screening.status,  # type: ignore[arg-type]
                    flags=screening.flags,
                    synthetic=synthetic or report_in.simulated,
                )
            )
            if report.status == "rejected":
                return SubmitResult(report=report, message=f"Not used: {', '.join(report.flags).replace('_', ' ')}.")
            if report.status == "review":
                return SubmitResult(report=report, message="Held for a reviewer before it counts.")
            feature, merged = self._attach(report, now)
            if feature is None:
                return SubmitResult(report=report, message="Noted. There is no reported issue here for it to confirm or contradict.")
            self._conditions = None
            if feature.published:
                message = f"Merged with {feature.reporters - 1} other reporter{'s' if feature.reporters != 2 else ''}. This {feature.state.replace('_', ' ')} is on the map." if merged else "On the map."
            elif merged:
                message = f"Merged with a nearby report. Confidence {feature.confidence:.2f}, published at {self.settings.publish_threshold:.2f}."
            else:
                message = f"Recorded. One report is never enough on its own: confidence {feature.confidence:.2f}, published at {self.settings.publish_threshold:.2f}."
            return SubmitResult(report=self.store.get_report(report.id) or report, feature=feature, merged=merged, message=message)

    def _attach(self, report: Report, now: datetime) -> tuple[Feature | None, bool]:
        """Puts an accepted report into the nearest cluster within the radius, or starts one."""
        near = self.store.nearest_feature(report.category, report.lon, report.lat, self.settings.cluster_radius_m)
        if near is None:
            if (report.category, report.state) in COUNTER_STATES:
                return None, False
            feature = self.store.add_feature(
                Feature(id=0, category=report.category, state=report.state, lon=report.lon, lat=report.lat, synthetic=report.synthetic, updated_at=now, created_at=now)
            )
            merged = False
        else:
            feature, merged = near[0], True
        report.feature_id = feature.id
        self.store.update_report(report)
        return self.rescore(feature.id, now), merged

    def rescore(self, feature_id: int, now: datetime | None = None) -> Feature:
        """Recomputes position, state, confidence and publication from the feature's reports."""
        now = now or utcnow()
        feature = self.store.get_feature(feature_id)
        assert feature is not None
        reports = [r for r in self.store.reports_for_feature(feature_id) if r.status == "accepted"]
        result = confidence.score(reports, now)
        if result is None:
            feature = feature.model_copy(update={"confidence": 0.0, "reporters": 0, "report_count": 0, "published": False, "updated_at": now})
            self.store.update_feature(feature)
            return feature
        agreeing = [r for r in reports if r.state == result.state]
        values = sorted(r.value for r in agreeing if r.value is not None)
        materials = [r.material for r in agreeing if r.material]
        published = result.confidence >= self.settings.publish_threshold
        if feature.reviewed == "approved":
            published = True
        elif feature.reviewed == "rejected":
            published = False
        feature = feature.model_copy(
            update={
                "state": result.state,
                "lon": sum(r.lon for r in reports) / len(reports),
                "lat": sum(r.lat for r in reports) / len(reports),
                "confidence": result.confidence,
                "agreement": result.agreement,
                "reporters": result.reporters,
                "report_count": len(reports),
                "published": published,
                "value": values[len(values) // 2] if values else None,
                "material": max(set(materials), key=materials.count) if materials else None,
                "synthetic": any(r.synthetic for r in reports),
                "updated_at": now,
            }
        )
        self.store.update_feature(feature)
        return feature

    # Review queue

    def review_report(self, report_id: int, action: str) -> Report:
        with self._lock:
            report = self.store.get_report(report_id)
            if report is None:
                raise KeyError(report_id)
            if action == "accept":
                report.status = "accepted"
                report.flags = [*report.flags, "reviewer_accepted"]
                self.store.update_report(report)
                self._attach(report, utcnow())
            elif action == "reject":
                previous_feature = report.feature_id
                report.status = "rejected"
                report.flags = [*report.flags, "reviewer_rejected"]
                self.store.update_report(report)
                if previous_feature is not None:
                    self.rescore(previous_feature)
            else:
                raise ValueError("action must be accept or reject")
            self._conditions = None
            return self.store.get_report(report_id) or report

    def review_feature(self, feature_id: int, action: str) -> Feature:
        with self._lock:
            feature = self.store.get_feature(feature_id)
            if feature is None:
                raise KeyError(feature_id)
            reviewed = {"approve": "approved", "reject": "rejected", "clear": None}.get(action, "invalid")
            if reviewed == "invalid":
                raise ValueError("action must be approve, reject or clear")
            self.store.update_feature(feature.model_copy(update={"reviewed": reviewed}))
            self._conditions = None
            return self.rescore(feature_id)

    # Routing input

    def conditions(self) -> dict[str, Condition]:
        if self._conditions is None:
            self._conditions = build_conditions(self.graph, self.store.list_features())
        return self._conditions
