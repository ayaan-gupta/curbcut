"""Milestone 1: tagged points arrive from a phone and come back as a map layer."""

from __future__ import annotations

import io
import json
from datetime import timedelta

from fastapi.testclient import TestClient

from .conftest import DEVICE_A, DEVICE_B, NOW

# A real kerb node in the pilot area.
KERB = {"lon": -122.3200002, "lat": 47.6153571}


def post(client: TestClient, body: dict, headers: dict, photo: bytes | None = None):
    files = {"photo": ("curb.jpg", io.BytesIO(photo), "image/jpeg")} if photo else None
    return client.post("/api/reports", data={"report": json.dumps(body)}, files=files, headers=headers)


def test_a_tagged_point_shows_up_in_the_map_layer(client: TestClient) -> None:
    body = {"category": "surface", "state": "bump", "source": "sensor", "accuracy_m": 4, "model_confidence": 0.8, "value": 11.2, **KERB}
    response = post(client, body, DEVICE_A)
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["report"]["status"] == "accepted" and result["feature"]["published"] is False

    layer = client.get("/api/features", params={"include": "all"}).json()
    assert layer["type"] == "FeatureCollection" and len(layer["features"]) == 1
    point = layer["features"][0]
    assert point["geometry"]["type"] == "Point"
    assert abs(point["geometry"]["coordinates"][0] - KERB["lon"]) < 1e-6
    assert point["properties"]["kind"] == "surface_bump" and point["properties"]["reporters"] == 1
    # The public layer only carries published features.
    assert client.get("/api/features").json()["features"] == []


def test_reports_need_a_device_id_and_a_valid_state(client: TestClient) -> None:
    body = {"category": "curb", "state": "missing", "source": "photo", **KERB}
    assert client.post("/api/reports", data={"report": json.dumps(body)}).status_code == 400
    assert post(client, {**body, "state": "smooth"}, DEVICE_A).status_code == 422
    assert post(client, {**body, "category": "slope", "state": "steep"}, DEVICE_A).status_code == 422  # no incline value


def test_a_batch_of_trace_points_is_stored_in_one_call(client: TestClient) -> None:
    points = [
        {
            "category": "surface",
            "state": "rough",
            "source": "sensor",
            "lon": KERB["lon"] + i * 0.0002,  # 15 m apart
            "lat": KERB["lat"],
            "accuracy_m": 5,
            "model_confidence": 0.8,
            "recorded_at": (NOW - timedelta(seconds=15 * (5 - i))).isoformat(),
        }
        for i in range(5)
    ]
    response = client.post("/api/reports/batch", json=points, headers=DEVICE_B)
    assert response.status_code == 200 and response.json()["accepted"] == 5
    assert len(client.get("/api/features", params={"include": "all"}).json()["features"]) == 5
    assert client.get("/api/health").json()["reports"] == 5


def test_no_response_ever_carries_a_device_identifier(client: TestClient) -> None:
    post(client, {"category": "curb", "state": "missing", "source": "photo", **KERB}, DEVICE_A)
    feature_id = client.get("/api/features", params={"include": "all"}).json()["features"][0]["id"]
    for path in ("/api/features?include=all", f"/api/features/{feature_id}", "/api/review", "/api/export/dataset.geojson"):
        text = client.get(path).text
        assert "device" not in text.lower(), path


def test_an_uploaded_photo_loses_its_exif_block(client: TestClient, settings) -> None:
    exif = b"Exif\x00\x00" + b"GPS-LATITUDE-47.6153" + b"\x00" * 8
    jpeg = b"\xff\xd8" + b"\xff\xe0" + (16).to_bytes(2, "big") + b"JFIF\x00" + b"\x00" * 9
    jpeg += b"\xff\xe1" + (len(exif) + 2).to_bytes(2, "big") + exif
    jpeg += b"\xff\xda" + (8).to_bytes(2, "big") + b"\x00" * 6 + b"image-data" + b"\xff\xd9"
    response = post(client, {"category": "curb", "state": "missing", "source": "photo", **KERB}, DEVICE_A, photo=jpeg)
    assert response.status_code == 200, response.text
    stored = list(settings.upload_dir.glob("*.jpg"))
    assert len(stored) == 1
    saved = stored[0].read_bytes()
    assert b"GPS-LATITUDE" not in saved and b"Exif" not in saved and b"JFIF" in saved and saved.endswith(b"image-data\xff\xd9")
    assert post(client, {"category": "curb", "state": "missing", "source": "photo", **KERB}, DEVICE_B, photo=b"not a jpeg").status_code == 422


def test_the_pilot_area_serves_a_basemap_and_a_network(client: TestClient) -> None:
    meta = client.get("/api/meta").json()
    assert meta["pilot"]["license"] == "ODbL-1.0" and "OpenStreetMap" in meta["pilot"]["attribution"]
    assert set(meta["profiles"]) == {"manual_wheelchair", "powered_wheelchair", "stroller", "walking"}
    base = client.get("/api/basemap").json()
    assert len(base["roads"]["features"]) > 50 and len(base["areas"]["features"]) > 100
    network = client.get("/api/network", params={"profile": "manual_wheelchair"}).json()
    codes = {f["properties"]["code"] for f in network["features"]}
    assert len(network["features"]) == 991 and {"ok", "steps", "slope", "no_curb_cut"} <= codes
