"""The synthetic demo seed: labelled as synthetic, and behaves the way the demo claims."""

from __future__ import annotations

from curbcut import osm_export, routing
from curbcut.config import Settings
from curbcut.graph import Graph
from curbcut.seed import DEMO_DESTINATION, DEMO_ORIGIN, seed_demo
from curbcut.service import Service
from curbcut.store import Store

from .conftest import NOW


def seeded(store: Store, pilot: Graph, settings: Settings) -> Service:
    service = Service(store, pilot, settings)
    seed_demo(service, NOW)
    return service


def test_everything_seeded_is_flagged_synthetic(store: Store, pilot: Graph, settings: Settings) -> None:
    service = seeded(store, pilot, settings)
    reports = service.store.list_reports()
    assert len(reports) >= 35 and all(r.synthetic for r in reports)
    assert all(f.synthetic for f in service.store.list_features())
    assert osm_export.dataset(service.store, settings, NOW)["contains_synthetic_demo_data"] is True
    assert osm_export.dataset(service.store, settings, NOW, include_synthetic=False)["features"] == []


def test_the_seed_shows_a_published_barrier_a_lone_report_a_conflict_and_a_review_queue(store: Store, pilot: Graph, settings: Settings) -> None:
    service = seeded(store, pilot, settings)
    features = service.store.list_features()
    published = {(f.category, f.state) for f in features if f.published}
    assert {("curb", "missing"), ("curb", "blocked"), ("curb", "damaged"), ("curb", "present"), ("surface", "rough"), ("surface", "bump"), ("slope", "steep")} <= published
    lone = [f for f in features if f.reporters == 1 and f.category == "curb" and f.state == "missing"]
    assert lone and not any(f.published for f in lone)
    assert any(f.reporters >= 3 and f.agreement < 0.7 and not f.published for f in features)
    flags = {flag for r in service.store.list_reports(status="review") for flag in r.flags}
    assert {"implausible_speed", "far_from_any_path"} <= flags


def test_the_demo_walk_detours_around_the_seeded_missing_curb_cut(store: Store, pilot: Graph, settings: Settings) -> None:
    before = routing.route(pilot, {}, DEMO_ORIGIN, DEMO_DESTINATION, "manual_wheelchair")
    service = seeded(store, pilot, settings)
    after = routing.route(pilot, service.conditions(), DEMO_ORIGIN, DEMO_DESTINATION, "manual_wheelchair")
    assert before.status == after.status == "ok" and before.route and after.route and after.shortest
    assert after.route.distance_m > before.route.distance_m + 40
    assert any("no curb cut (reported)" in b.reason for b in after.shortest.barriers)
    assert after.route.barriers == []
    # A stroller is not stopped by the same kerb.
    stroller = routing.route(pilot, service.conditions(), DEMO_ORIGIN, DEMO_DESTINATION, "stroller")
    assert stroller.route and stroller.route.distance_m < after.route.distance_m


def test_seeded_export_candidates_cover_add_change_and_confirm(store: Store, pilot: Graph, settings: Settings) -> None:
    service = seeded(store, pilot, settings)
    cands, _ = osm_export.candidates(service.store, pilot, settings)
    kinds = {c.kind for c in cands}
    keys = {c.key for c in cands}
    assert {"change", "confirm", "add"} <= kinds and {"kerb", "surface", "incline"} <= keys
    assert all(c.synthetic for c in cands)
