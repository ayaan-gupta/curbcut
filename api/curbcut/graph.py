"""The walking graph of the pilot area, built from the committed OpenStreetMap extract."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .geo import project_on_polyline
from .models import Feature


@dataclass(frozen=True)
class Edge:
    id: str
    way: int
    u: int
    v: int
    kind: str
    length: float
    coords: list[list[float]]
    surface: str | None = None
    smoothness: str | None = None
    incline: float | None = None
    kerb_u: str | None = None
    kerb_v: str | None = None
    name: str | None = None
    bbox: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)

    def other(self, node: int) -> int:
        return self.v if node == self.u else self.u


@dataclass
class Condition:
    """What crowd reports say about one edge, on top of what OpenStreetMap says."""

    kerb: str | None = None
    blocked: bool = False
    damaged: bool = False
    rough: bool = False
    bump: bool = False
    incline: float | None = None
    material: str | None = None
    low_score: float = 0.0
    features: list[int] = field(default_factory=list)


@dataclass
class Snap:
    edge: Edge
    distance_m: float
    along_m: float
    point: list[float]


class Graph:
    def __init__(self, data: dict[str, Any]) -> None:
        self.meta: dict[str, Any] = data["meta"]
        self.osm_nodes: dict[str, dict[str, Any]] = data.get("osm_nodes", {})
        self.osm_ways: dict[str, dict[str, Any]] = data.get("osm_ways", {})
        self.roads: list[dict[str, Any]] = data.get("roads", [])
        self.areas: list[dict[str, Any]] = data.get("areas", [])
        self.edges: dict[str, Edge] = {}
        self.adjacent: dict[int, list[str]] = {}
        self.node_pos: dict[int, list[float]] = {}
        for raw in data["edges"]:
            lons = [c[0] for c in raw["coords"]]
            lats = [c[1] for c in raw["coords"]]
            kerb = raw.get("kerb", {})
            edge = Edge(
                id=raw["id"],
                way=raw["way"],
                u=raw["u"],
                v=raw["v"],
                kind=raw["kind"],
                length=float(raw["len"]),
                coords=raw["coords"],
                surface=raw.get("surface"),
                smoothness=raw.get("smoothness"),
                incline=raw.get("incline"),
                kerb_u=kerb.get("u"),
                kerb_v=kerb.get("v"),
                name=raw.get("name"),
                bbox=(min(lons), min(lats), max(lons), max(lats)),
            )
            self.edges[edge.id] = edge
            self.adjacent.setdefault(edge.u, []).append(edge.id)
            self.adjacent.setdefault(edge.v, []).append(edge.id)
            self.node_pos[edge.u] = raw["coords"][0]
            self.node_pos[edge.v] = raw["coords"][-1]

    @classmethod
    def load(cls, path: Path) -> "Graph":
        return cls(json.loads(path.read_text()))

    @property
    def bbox(self) -> dict[str, float]:
        return self.meta["bbox"]

    def contains(self, lon: float, lat: float, margin_deg: float = 0.0006) -> bool:
        b = self.bbox
        return b["west"] - margin_deg <= lon <= b["east"] + margin_deg and b["south"] - margin_deg <= lat <= b["north"] + margin_deg

    def nearest_edge(self, lon: float, lat: float, max_m: float, accept: Callable[[Edge], bool] | None = None) -> Snap | None:
        # A degree of latitude is about 111 km, and longitude is shorter, so this margin is generous.
        margin = max_m / 70_000.0
        best: Snap | None = None
        for edge in self.edges.values():
            w, s, e, n = edge.bbox
            if lon < w - margin or lon > e + margin or lat < s - margin or lat > n + margin:
                continue
            if accept is not None and not accept(edge):
                continue
            dist, along, point = project_on_polyline(lon, lat, edge.coords)
            if dist <= max_m and (best is None or dist < best.distance_m):
                best = Snap(edge=edge, distance_m=dist, along_m=along, point=point)
        return best


CURB_SNAP_M = 12.0
SURFACE_SNAP_M = 10.0
ISSUE_STATES = {("curb", "missing"), ("curb", "damaged"), ("curb", "blocked"), ("surface", "rough"), ("surface", "bump"), ("slope", "steep")}


def snap_feature(graph: Graph, feature: Feature) -> Snap | None:
    """Which edge a feature is about. Curb features belong to the nearest crossing if one is close."""
    if feature.category == "curb":
        crossing = graph.nearest_edge(feature.lon, feature.lat, CURB_SNAP_M, accept=lambda e: e.kind == "crossing")
        if crossing is not None:
            return crossing
    return graph.nearest_edge(feature.lon, feature.lat, SURFACE_SNAP_M)


def build_conditions(graph: Graph, features: list[Feature]) -> dict[str, Condition]:
    """Folds features into per-edge conditions.

    Published features become facts the router filters on. Unpublished issue
    features only add to `low_score`, which makes an edge less attractive
    without closing it.
    """
    out: dict[str, Condition] = {}
    for f in features:
        if f.reviewed == "rejected":
            continue
        snap = snap_feature(graph, f)
        if snap is None:
            continue
        cond = out.setdefault(snap.edge.id, Condition())
        if not f.published:
            if (f.category, f.state) in ISSUE_STATES:
                cond.low_score += f.confidence
                cond.features.append(f.id)
            continue
        cond.features.append(f.id)
        if f.category == "curb":
            if f.state == "missing":
                cond.kerb = "raised"
            elif f.state == "present" and cond.kerb is None:
                cond.kerb = "lowered"
            elif f.state == "blocked":
                cond.blocked = True
            elif f.state == "damaged":
                cond.damaged = True
        elif f.category == "surface":
            cond.rough = cond.rough or f.state == "rough"
            cond.bump = cond.bump or f.state == "bump"
            cond.material = f.material or cond.material
        elif f.category == "slope" and f.state == "steep" and f.value is not None:
            cond.incline = max(cond.incline or 0.0, abs(f.value))
    return out
