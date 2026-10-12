"""Milestone 4: device rate limits, outlier rejection and the review queue."""

from __future__ import annotations

from datetime import timedelta

import pytest
from fastapi.testclient import TestClient

from curbcut.abuse import RateLimited, device_hash
from curbcut.service import Service

from .conftest import DEVICE_A, NOW, at, report


def test_a_device_is_rate_limited_and_told_when_to_retry(service: Service) -> None:
    service.settings.rate_limit_10min = 5
    for i in range(5):
        service.submit(report(i * 9.0, 0.0), "device-one", now=NOW + timedelta(seconds=i))
    with pytest.raises(RateLimited) as error:
        service.submit(report(50.0, 0.0), "device-one", now=NOW + timedelta(seconds=10))
    assert 0 < error.value.retry_after_s <= 600
    # Another device is unaffected, and the first one is let back in after the window.
    service.submit(report(50.0, 0.0), "device-two", now=NOW + timedelta(seconds=10))
    service.submit(report(50.0, 5.0), "device-one", now=NOW + timedelta(minutes=11))


def test_the_rate_limit_is_a_429_over_http(client: TestClient, settings) -> None:
    settings.rate_limit_10min = 3
    points = [{"category": "surface", "state": "rough", "source": "sensor", "lon": -122.3200 + i * 0.0003, "lat": 47.6160, "accuracy_m": 5} for i in range(6)]
    batch = client.post("/api/reports/batch", json=points, headers=DEVICE_A).json()
    assert batch["accepted"] == 3 and "limit is 3" in batch["rate_limited"]
    single = client.post("/api/reports", data={"report": '{"category":"curb","state":"missing","source":"photo","lon":-122.32,"lat":47.616}'}, headers=DEVICE_A)
    assert single.status_code == 429 and int(single.headers["Retry-After"]) > 0


def test_reports_outside_the_area_or_with_a_loose_gps_fix_are_rejected(service: Service) -> None:
    lon, lat = at(5000, 0)
    far = service.submit(report(0, 0).model_copy(update={"lon": lon, "lat": lat}), "device-one", now=NOW)
    assert far.report.status == "rejected" and far.report.flags == ["outside_area"] and far.feature is None
    loose = service.submit(report(0, 0).model_copy(update={"accuracy_m": 60}), "device-one", now=NOW)
    assert loose.report.status == "rejected" and loose.report.flags == ["low_gps_accuracy"]
    assert service.store.list_features() == []


def test_a_device_that_teleports_is_held_for_review(service: Service) -> None:
    service.submit(report(0.0, 0.0), "device-one", now=NOW)
    jump = service.submit(report(120.0, 0.0), "device-one", now=NOW + timedelta(seconds=4))  # 30 m/s on foot
    assert jump.report.status == "review" and "implausible_speed" in jump.report.flags and jump.feature is None
    stroll = service.submit(report(20.0, 0.0), "device-two", now=NOW)
    later = service.submit(report(60.0, 0.0), "device-two", now=NOW + timedelta(seconds=40))
    assert stroll.report.status == later.report.status == "accepted"


def test_a_point_far_from_any_footway_is_held_for_review(service: Service) -> None:
    result = service.submit(report(30.0, 60.0, state="bump", category="surface", source="sensor"), "device-one", now=NOW)
    assert result.report.status == "review" and result.report.flags == ["far_from_any_path"]


def test_contradicting_an_established_feature_goes_to_review_instead_of_flipping_it(service: Service) -> None:
    for i, who in enumerate(("device-one", "device-two", "device-three")):
        service.submit(report(i * 0.8, 0.0, state="present"), who, now=NOW)
    feature = service.store.list_features()[0]
    assert feature.published and feature.reporters == 3 and feature.confidence >= 0.7
    odd = service.submit(report(0.5, 0.5, state="missing"), "device-four", now=NOW)
    assert odd.report.status == "review" and "contradicts_established_feature" in odd.report.flags
    assert service.store.get_feature(feature.id).agreement == 1.0  # type: ignore[union-attr]
    # A parked car can turn up overnight, so "blocked" is let through.
    blocked = service.submit(report(0.5, 0.5, state="blocked"), "device-five", now=NOW)
    assert blocked.report.status == "accepted"


def test_the_review_queue_lists_held_reports_and_a_reviewer_resolves_them(service: Service) -> None:
    service.submit(report(0.0, 0.0), "device-one", now=NOW)
    held = service.submit(report(120.0, 0.0), "device-one", now=NOW + timedelta(seconds=4))
    assert [r.id for r in service.store.list_reports(status="review")] == [held.report.id]

    accepted = service.review_report(held.report.id, "accept")
    assert accepted.status == "accepted" and accepted.feature_id is not None and "reviewer_accepted" in accepted.flags
    assert service.store.list_reports(status="review") == []

    rejected = service.review_report(held.report.id, "reject")
    assert rejected.status == "rejected"
    assert service.store.get_feature(accepted.feature_id).report_count == 0  # type: ignore[union-attr]


def test_review_endpoints_require_the_token_when_one_is_set(store, settings) -> None:
    from curbcut.main import create_app

    settings.reviewer_token = "test-reviewer-token"
    with TestClient(create_app(settings, store, clock=lambda: NOW)) as client:
        assert client.get("/api/review").status_code == 401
        assert client.get("/api/export/osmchange").status_code == 401
        assert client.get("/api/review", headers={"Authorization": "Bearer test-reviewer-token"}).status_code == 200
        assert client.get("/api/features").status_code == 200


def test_device_ids_are_stored_hashed() -> None:
    hashed = device_hash("my-phone-1234", "salt-a")
    assert "my-phone" not in hashed and len(hashed) == 20
    assert hashed != device_hash("my-phone-1234", "salt-b") and hashed == device_hash("my-phone-1234", "salt-a")


def test_a_queued_walk_uploaded_at_once_is_judged_by_when_each_point_was_observed(service: Service) -> None:
    """Ten points 6 m apart, recorded 5 s apart, all arriving in the same second: a walk, not a teleport."""
    for i in range(10):
        observed = NOW - timedelta(seconds=5 * (10 - i))
        result = service.submit(report(i * 6.0, 0.0, state="rough", category="surface", source="sensor").model_copy(update={"recorded_at": observed}), "device-one", now=NOW)
        assert result.report.status == "accepted", result.report.flags
        assert result.report.created_at == observed and result.report.received_at == NOW
    # A claimed observation time cannot dodge the rate limit, which counts by time of receipt.
    service.settings.rate_limit_10min = 10
    with pytest.raises(RateLimited):
        service.submit(report(0.0, 5.0).model_copy(update={"recorded_at": NOW - timedelta(hours=5)}), "device-one", now=NOW)
    # Nor can it reach into the future or further back than a day.
    late = service.submit(report(0.0, 10.0).model_copy(update={"recorded_at": NOW + timedelta(days=3)}), "device-two", now=NOW)
    old = service.submit(report(9.0, 10.0).model_copy(update={"recorded_at": NOW - timedelta(days=30)}), "device-three", now=NOW)
    assert late.report.created_at == NOW and old.report.created_at == NOW - timedelta(days=1)
