"""Shared fixtures.

Every store-backed test runs twice: once on the in-memory store, and once on
PostGIS when CURBCUT_TEST_DATABASE_URL points at one (`make test` and CI set
it). Without a database the PostGIS half is skipped and says so.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from curbcut.config import Settings
from curbcut.graph import Graph
from curbcut.main import create_app
from curbcut.models import ReportIn
from curbcut.service import Service
from curbcut.store import MemoryStore, Store

NOW = datetime(2026, 10, 1, 12, 0, tzinfo=timezone.utc)
TEST_DB = os.environ.get("CURBCUT_TEST_DATABASE_URL", "")

# A tiny street, in metres east and north of a corner, turned into lon/lat.
# Two sidewalks 10 m apart joined by two crossings 60 m apart.
LON0, LAT0 = -122.3200, 47.6160
M_LAT = 1 / 111_320.0
M_LON = 1 / (111_320.0 * 0.6741)


def at(x: float, y: float) -> list[float]:
    return [round(LON0 + x * M_LON, 8), round(LAT0 + y * M_LAT, 8)]


def street(kerb_a: str = "lowered", kerb_b: str = "lowered", **overrides: Any) -> Graph:
    """north sidewalk n1..n2, south sidewalk s1..s2, crossing A on the west, crossing B on the east."""
    edges = [
        {"id": "north", "way": 1, "u": 1, "v": 2, "kind": "sidewalk", "len": 60.0, "coords": [at(0, 10), at(60, 10)], "surface": "concrete"},
        {"id": "south", "way": 2, "u": 3, "v": 4, "kind": "sidewalk", "len": 60.0, "coords": [at(0, 0), at(60, 0)], "surface": "concrete"},
        {"id": "crossA", "way": 3, "u": 1, "v": 3, "kind": "crossing", "len": 10.0, "coords": [at(0, 10), at(0, 0)], "kerb": {"u": kerb_a, "v": kerb_a}},
        {"id": "crossB", "way": 4, "u": 2, "v": 4, "kind": "crossing", "len": 10.0, "coords": [at(60, 10), at(60, 0)], "kerb": {"u": kerb_b, "v": kerb_b}},
    ]
    for edge in edges:
        edge.update(overrides.get(edge["id"], {}))
    west, south = at(-80, -80)
    east, north = at(140, 90)
    return Graph(
        {
            "meta": {"name": "test street", "bbox": {"south": south, "west": west, "north": north, "east": east}, "attribution": "test", "license": "test"},
            "edges": edges,
            "osm_nodes": {
                "1": {"version": 3, "lon": at(0, 10)[0], "lat": at(0, 10)[1], "tags": {"barrier": "kerb", "kerb": kerb_a}},
                "3": {"version": 2, "lon": at(0, 0)[0], "lat": at(0, 0)[1], "tags": {"barrier": "kerb", "kerb": kerb_a}},
                "2": {"version": 5, "lon": at(60, 10)[0], "lat": at(60, 10)[1], "tags": {"barrier": "kerb"}},
                "4": {"version": 1, "lon": at(60, 0)[0], "lat": at(60, 0)[1], "tags": {"barrier": "kerb", "kerb": kerb_b}},
            },
            "osm_ways": {
                "1": {"version": 7, "nds": [1, 2], "tags": {"highway": "footway", "footway": "sidewalk", "surface": "concrete"}},
                "2": {"version": 4, "nds": [3, 4], "tags": {"highway": "footway", "footway": "sidewalk"}},
                "3": {"version": 2, "nds": [1, 3], "tags": {"highway": "footway", "footway": "crossing"}},
                "4": {"version": 2, "nds": [2, 4], "tags": {"highway": "footway", "footway": "crossing"}},
            },
        }
    )


def report(x: float, y: float, state: str = "missing", category: str = "curb", source: str = "photo", conf: float = 0.9, **extra: Any) -> ReportIn:
    lon, lat = at(x, y)
    return ReportIn(category=category, state=state, lon=lon, lat=lat, accuracy_m=5, source=source, model_confidence=conf, **extra)  # type: ignore[arg-type]


def _postgis_store() -> Store:
    import psycopg

    from curbcut.store_postgis import PostgisStore

    base, name = TEST_DB.rsplit("/", 1)
    with psycopg.connect(f"{base}/postgres", autocommit=True) as conn:
        if conn.execute("SELECT 1 FROM pg_database WHERE datname = %s", (name,)).fetchone() is None:
            conn.execute(f'CREATE DATABASE "{name}"')
    return PostgisStore(TEST_DB)


@pytest.fixture(params=["memory", "postgis"])
def store(request: pytest.FixtureRequest) -> Iterator[Store]:
    if request.param == "memory":
        yield MemoryStore()
        return
    if not TEST_DB:
        pytest.skip("PostGIS half skipped: set CURBCUT_TEST_DATABASE_URL (make test does)")
    pg = _postgis_store()
    pg.clear()
    yield pg
    pg.clear()
    pg.close()  # type: ignore[attr-defined]


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(database_url="", seed_demo=False, upload_dir=tmp_path / "uploads", reviewer_token="", _env_file=None)  # type: ignore[call-arg]


@pytest.fixture
def service(store: Store, settings: Settings) -> Service:
    return Service(store, street(), settings)


@pytest.fixture(scope="session")
def pilot() -> Graph:
    return Graph.load(Settings(_env_file=None).pilot_file)  # type: ignore[call-arg]


@pytest.fixture
def client(store: Store, settings: Settings) -> Iterator[TestClient]:
    """The real app over the real pilot graph, with an empty store and a fixed clock."""
    app = create_app(settings, store, clock=lambda: NOW)
    with TestClient(app) as c:
        yield c


DEVICE_A = {"X-Device-Id": "device-aaaaaaaa"}
DEVICE_B = {"X-Device-Id": "device-bbbbbbbb"}
DEVICE_C = {"X-Device-Id": "device-cccccccc"}
