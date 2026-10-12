"""Milestone 4: clustering within 5 m, confidence, and the publishing threshold.

PRD acceptance test: "Two reports within 5 m merge, and a lone low-confidence
report does not publish."
"""

from __future__ import annotations

from datetime import timedelta

import pytest

from curbcut import confidence
from curbcut.geo import haversine_m
from curbcut.models import Report
from curbcut.service import Service

from .conftest import NOW, report


def test_acceptance_two_reports_within_5_m_merge(service: Service) -> None:
    first = service.submit(report(0.0, 0.0), "device-one", now=NOW)
    second = service.submit(report(3.0, 2.5), "device-two", now=NOW)  # 3.9 m away
    assert not first.merged and second.merged
    assert first.feature is not None and second.feature is not None
    assert second.feature.id == first.feature.id
    assert second.feature.reporters == 2 and second.feature.report_count == 2
    assert len(service.store.list_features()) == 1
    # The merged feature sits between the two reports.
    assert haversine_m(second.feature.lon, second.feature.lat, first.report.lon, first.report.lat) == pytest.approx(1.95, abs=0.2)


def test_acceptance_a_lone_low_confidence_report_does_not_publish(service: Service) -> None:
    result = service.submit(report(0.0, 0.0, conf=0.3), "device-one", now=NOW)
    assert result.feature is not None
    assert result.feature.published is False
    assert result.feature.confidence < service.settings.publish_threshold
    assert service.store.list_features(published=True) == []


def test_a_lone_report_never_publishes_however_sure_the_model_is(service: Service) -> None:
    result = service.submit(report(0.0, 0.0, conf=1.0), "device-one", now=NOW)
    assert result.feature is not None and result.feature.published is False
    assert result.feature.confidence == pytest.approx(0.39, abs=0.01)


def test_two_agreeing_photo_reporters_publish(service: Service) -> None:
    service.submit(report(0.0, 0.0), "device-one", now=NOW)
    result = service.submit(report(1.0, 1.0), "device-two", now=NOW)
    assert result.feature is not None and result.feature.published
    assert result.feature.state == "missing" and result.feature.confidence > 0.5


def test_reports_more_than_5_m_apart_stay_separate(service: Service) -> None:
    a = service.submit(report(0.0, 0.0), "device-one", now=NOW)
    b = service.submit(report(8.0, 0.0), "device-two", now=NOW)
    assert not b.merged and a.feature is not None and b.feature is not None and a.feature.id != b.feature.id
    assert not a.feature.published and not b.feature.published


def test_different_categories_do_not_merge_even_at_the_same_spot(service: Service) -> None:
    service.submit(report(0.0, 0.0), "device-one", now=NOW)
    other = service.submit(report(0.5, 0.0, state="bump", category="surface", source="sensor"), "device-two", now=NOW)
    assert not other.merged and len(service.store.list_features()) == 2


def test_one_device_reporting_twice_is_still_one_reporter(service: Service) -> None:
    service.submit(report(0.0, 0.0), "device-one", now=NOW)
    again = service.submit(report(1.0, 0.0), "device-one", now=NOW + timedelta(minutes=5))
    assert again.merged and again.feature is not None
    assert again.feature.reporters == 1 and again.feature.report_count == 2
    assert again.feature.published is False


def test_sensor_reports_need_three_independent_passes(service: Service) -> None:
    for i, who in enumerate(("device-one", "device-two")):
        result = service.submit(report(i * 1.0, 0.0, state="rough", category="surface", source="sensor"), who, now=NOW)
    assert result.feature is not None and not result.feature.published
    result = service.submit(report(1.0, 1.0, state="rough", category="surface", source="sensor"), "device-three", now=NOW)
    assert result.feature is not None and result.feature.published


def test_disagreement_lowers_confidence_and_can_unpublish(service: Service) -> None:
    service.submit(report(0.0, 0.0, state="present"), "device-one", now=NOW)
    agreed = service.submit(report(1.0, 0.0, state="present"), "device-two", now=NOW)
    assert agreed.feature is not None and agreed.feature.published
    split = service.submit(report(0.0, 1.0, state="missing"), "device-three", now=NOW)
    assert split.feature is not None
    assert split.feature.state == "present" and split.feature.agreement == pytest.approx(2 / 3, abs=0.01)
    assert split.feature.confidence < agreed.feature.confidence and not split.feature.published


def test_a_smooth_pass_only_counts_against_an_existing_surface_issue(service: Service) -> None:
    alone = service.submit(report(30.0, 0.0, state="smooth", category="surface", source="sensor"), "device-one", now=NOW)
    assert alone.feature is None and service.store.list_features() == []
    service.submit(report(0.0, 0.0, state="rough", category="surface", source="sensor"), "device-two", now=NOW)
    counter = service.submit(report(1.0, 0.0, state="smooth", category="surface", source="sensor"), "device-three", now=NOW)
    assert counter.merged and counter.feature is not None and counter.feature.agreement == pytest.approx(0.5, abs=0.01)


def test_old_reports_fade(service: Service) -> None:
    def vote(days_old: float, state: str = "blocked") -> Report:
        return Report(id=1, device="d", category="curb", state=state, lon=0, lat=0, accuracy_m=5, source="photo", model_confidence=0.9, created_at=NOW - timedelta(days=days_old))

    fresh, fortnight = confidence.vote_weight(vote(0), NOW), confidence.vote_weight(vote(14), NOW)
    assert fortnight == pytest.approx(fresh / 2, rel=1e-6)  # a blocked ramp has a 14 day half life
    # Something poured in concrete lasts: a missing curb cut loses under 3 percent in two weeks.
    assert confidence.vote_weight(vote(14, "missing"), NOW) > 0.97 * confidence.vote_weight(vote(0, "missing"), NOW)

    service.submit(report(0.0, 0.0, state="blocked"), "device-one", now=NOW - timedelta(days=90))
    service.submit(report(1.0, 0.0, state="blocked"), "device-two", now=NOW - timedelta(days=90))
    feature = service.store.list_features()[0]
    assert feature.published  # scored as of the day it was reported
    assert service.rescore(feature.id, NOW).published is False  # three months on, nobody has confirmed it


def test_a_reviewer_can_publish_or_retract_a_feature(service: Service) -> None:
    result = service.submit(report(0.0, 0.0), "device-one", now=NOW)
    assert result.feature is not None and not result.feature.published
    approved = service.review_feature(result.feature.id, "approve")
    assert approved.published and approved.reviewed == "approved"
    rejected = service.review_feature(result.feature.id, "reject")
    assert not rejected.published
