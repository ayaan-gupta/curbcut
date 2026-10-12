"""Milestone 4: the OpenStreetMap export draft and the open dataset."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from datetime import timedelta
from pathlib import Path

from fastapi.testclient import TestClient

from curbcut import osm_export
from curbcut.service import Service

from .conftest import NOW, report


def publish_missing_at_a(service: Service) -> None:
    service.submit(report(0.0, 0.5), "device-one", now=NOW)
    service.submit(report(0.6, 0.0), "device-two", now=NOW)
    service.submit(report(0.0, -0.5), "device-three", now=NOW)


def test_published_curb_features_become_tag_proposals_on_the_nearest_kerb_node(service: Service) -> None:
    publish_missing_at_a(service)
    cands, unmatched = osm_export.candidates(service.store, service.graph, service.settings)
    assert unmatched == [] and len(cands) == 1
    c = cands[0]
    assert (c.element, c.osm_id, c.key, c.value, c.current, c.kind) == ("node", 3, "kerb", "raised", "lowered", "change")
    assert c.decision == "pending" and "Disagrees" in c.note and c.distance_m < 1.5


def test_nothing_is_exported_until_a_reviewer_approves_it(service: Service) -> None:
    publish_missing_at_a(service)
    xml, summary = osm_export.osm_change(service.store, service.graph, service.settings)
    assert summary["included"] == 0 and "<node" not in xml

    service.store.set_export_decision("node-3-kerb", "approved", None)
    xml, summary = osm_export.osm_change(service.store, service.graph, service.settings)
    assert summary == {"included": 1, "elements": 1, "held_back_synthetic": 0, "synthetic_included": 0}
    root = ET.fromstring(xml)
    assert root.tag == "osmChange" and root.attrib["version"] == "0.6" and "needs human review" in root.attrib["generator"]
    assert [child.tag for child in root] == ["modify"], "a draft only ever modifies existing elements"
    (node,) = root.find("modify")  # type: ignore[misc]
    assert node.tag == "node" and node.attrib["id"] == "3" and node.attrib["version"] == "2"
    assert {t.attrib["k"]: t.attrib["v"] for t in node} == {"barrier": "kerb", "kerb": "raised"}

    service.store.set_export_decision("node-3-kerb", "skipped", None)
    assert osm_export.osm_change(service.store, service.graph, service.settings)[1]["included"] == 0


def test_a_present_curb_cut_adds_kerb_lowered_where_the_node_has_no_kerb_tag(service: Service) -> None:
    for i, who in enumerate(("device-one", "device-two", "device-three")):
        service.submit(report(60.0 + i * 0.4, 10.0, state="present"), who, now=NOW)  # node 2 has barrier=kerb but no kerb value
    (c,) = osm_export.candidates(service.store, service.graph, service.settings)[0]
    assert (c.osm_id, c.key, c.value, c.current, c.kind) == (2, "kerb", "lowered", None, "add")
    # Confirming what the map already says produces no edit.
    for i, who in enumerate(("device-four", "device-five", "device-six")):
        service.submit(report(60.0 + i * 0.4, 0.0, state="present"), who, now=NOW)
    confirm = next(c for c in osm_export.candidates(service.store, service.graph, service.settings)[0] if c.osm_id == 4)
    assert confirm.kind == "confirm"
    service.store.set_export_decision(confirm.id, "approved", None)
    assert 'id="4"' not in osm_export.osm_change(service.store, service.graph, service.settings)[0]


def test_surface_and_incline_go_on_the_way_and_incline_waits_for_a_direction(service: Service) -> None:
    for i, who in enumerate(("device-one", "device-two", "device-three")):
        service.submit(report(20.0 + i, 0.0, state="rough", category="surface", source="manual", material="paving_stones"), who, now=NOW)
        service.submit(report(40.0 + i, 10.0, state="steep", category="slope", source="manual", value=9.0 + i), who, now=NOW + timedelta(minutes=2))
    cands = {c.id: c for c in osm_export.candidates(service.store, service.graph, service.settings)[0]}
    surface, incline = cands["way-2-surface"], cands["way-1-incline"]
    assert (surface.value, surface.current, surface.kind) == ("paving_stones", None, "add")
    assert incline.value == "10%" and incline.needs_direction

    service.store.set_export_decision("way-2-surface", "approved", None)
    service.store.set_export_decision("way-1-incline", "approved", None)
    xml, summary = osm_export.osm_change(service.store, service.graph, service.settings)
    assert summary["included"] == 1 and "incline" not in xml, "an incline with no direction chosen must be left out"

    service.store.set_export_decision("way-1-incline", "approved", -1)
    root = ET.fromstring(osm_export.osm_change(service.store, service.graph, service.settings)[0])
    ways = {w.attrib["id"]: w for w in root.iter("way")}
    assert {t.attrib["k"]: t.attrib["v"] for t in ways["1"].iter("tag")}["incline"] == "-10%"
    assert [n.attrib["ref"] for n in ways["2"].iter("nd")] == ["3", "4"] and ways["2"].attrib["version"] == "4"
    assert {t.attrib["k"]: t.attrib["v"] for t in ways["2"].iter("tag")}["surface"] == "paving_stones"


def test_synthetic_reports_never_reach_an_export_unless_asked_and_then_it_says_do_not_upload(service: Service) -> None:
    for who, dy in (("seed-one", 0.5), ("seed-two", 0.0), ("seed-three", -0.5)):
        service.submit(report(0.0, dy), who, now=NOW, synthetic=True)
    service.store.set_export_decision("node-3-kerb", "approved", None)
    xml, summary = osm_export.osm_change(service.store, service.graph, service.settings)
    assert summary["included"] == 0 and summary["held_back_synthetic"] == 1 and "<node" not in xml
    xml, summary = osm_export.osm_change(service.store, service.graph, service.settings, allow_synthetic=True)
    assert summary["synthetic_included"] == 1 and "DO NOT UPLOAD" in xml
    assert "DO NOT UPLOAD" in ET.fromstring(xml).attrib["generator"]


def test_transient_and_weak_features_are_not_proposed(service: Service) -> None:
    for i, who in enumerate(("device-one", "device-two", "device-three")):
        service.submit(report(i * 0.5, 0.0, state="blocked"), who, now=NOW)
    service.submit(report(60.0, 0.0, state="missing"), "device-four", now=NOW)  # alone, unpublished
    assert osm_export.candidates(service.store, service.graph, service.settings)[0] == []


def test_the_dataset_states_its_licence_and_holds_only_published_features(client: TestClient) -> None:
    body = {"category": "curb", "state": "missing", "source": "photo", "lon": -122.3200002, "lat": 47.6153571, "accuracy_m": 4}
    import json

    for who in ("device-aaaaaaaa", "device-bbbbbbbb", "device-dddddddd"):
        client.post("/api/reports", data={"report": json.dumps(body)}, headers={"X-Device-Id": who})
    client.post("/api/reports", data={"report": json.dumps({**body, "lon": -122.3190})}, headers={"X-Device-Id": "device-cccccccc"})
    response = client.get("/api/export/dataset.geojson")
    data = response.json()
    assert response.headers["content-type"].startswith("application/geo+json")
    assert data["license"] == "ODbL-1.0" and "opendatacommons.org" in data["license_url"] and "OpenStreetMap" in data["attribution"]
    assert len(data["features"]) == 1 and data["features"][0]["properties"]["reporters"] == 3
    assert data["contains_synthetic_demo_data"] is False

    draft = client.get("/api/export/osmchange")
    assert draft.status_code == 200 and "attachment" in draft.headers["content-disposition"] and draft.headers["content-type"].startswith("application/xml")
    listing = client.get("/api/export/candidates").json()
    assert [c["id"] for c in listing["candidates"]] == ["node-7083429604-kerb"]
    decided = client.post("/api/export/candidates/node-7083429604-kerb", json={"decision": "approved"}).json()
    assert decided["decision"] == "approved"
    assert 'id="7083429604"' in client.get("/api/export/osmchange").text
    assert client.post("/api/export/candidates/node-1-kerb", json={"decision": "approved"}).status_code == 404


def test_no_code_in_this_project_writes_to_openstreetmap() -> None:
    """The PRD says contributions go 'through a reviewed workflow'. There is no upload path to find."""
    root = Path(__file__).resolve().parents[2]
    sources = [*root.glob("api/curbcut/**/*.py"), *root.glob("web/src/**/*.ts"), *root.glob("web/src/**/*.tsx"), *root.glob("scripts/**/*.py"), *root.glob("ml/**/*.py")]
    assert len(sources) > 20
    upload = re.compile(r"api\.openstreetmap\.org|/api/0\.6/(changeset|node|way)|osm_?api|changeset/create", re.IGNORECASE)
    offenders = [str(p.relative_to(root)) for p in sources if upload.search(p.read_text(encoding="utf8"))]
    assert offenders == []
