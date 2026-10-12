"""Draft OpenStreetMap edits for a human to review, and the open dataset export.

Nothing in this project talks to the OpenStreetMap editing API. The output is
an OsmChange file a mapper opens in JOSM, checks against imagery or the
street, and uploads under their own account if they agree with it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from typing import Any
from xml.sax.saxutils import quoteattr

from .config import Settings
from .geo import haversine_m
from .graph import Graph
from .models import Feature
from .store import Store

NODE_MATCH_M = 8.0
WAY_MATCH_M = 8.0
DATASET_LICENSE = "ODbL-1.0"
ALLOWED_KEYS = {"kerb", "surface", "incline"}


@dataclass
class Candidate:
    id: str
    feature_id: int
    element: str  # node or way
    osm_id: int
    key: str
    value: str
    current: str | None
    kind: str  # add, change or confirm
    distance_m: float
    confidence: float
    reporters: int
    synthetic: bool
    reviewed: bool
    lon: float
    lat: float
    needs_direction: bool = False
    decision: str = "pending"
    note: str = ""


def _nearest_kerb_node(graph: Graph, feature: Feature) -> tuple[str, dict[str, Any], float] | None:
    best: tuple[str, dict[str, Any], float] | None = None
    for nid, node in graph.osm_nodes.items():
        tags = node["tags"]
        if "kerb" not in tags and tags.get("barrier") != "kerb":
            continue
        d = haversine_m(feature.lon, feature.lat, node["lon"], node["lat"])
        if d <= NODE_MATCH_M and (best is None or d < best[2]):
            best = (nid, node, d)
    return best


def candidates(store: Store, graph: Graph, settings: Settings) -> tuple[list[Candidate], list[dict[str, Any]]]:
    """Features strong enough to propose to OpenStreetMap, matched to existing elements.

    Returns the candidates and a list of features that qualified but have no
    OSM element close enough to tag, which a mapper has to place by hand.
    """
    decisions = store.export_decisions()
    out: list[Candidate] = []
    unmatched: list[dict[str, Any]] = []
    for f in store.list_features(published=True):
        if f.confidence < settings.export_threshold and f.reviewed != "approved":
            continue
        proposal: tuple[str, int, str, str, str | None, float, bool, str] | None = None
        if f.category == "curb" and f.state in ("present", "missing"):
            hit = _nearest_kerb_node(graph, f)
            if hit is None:
                unmatched.append({"feature_id": f.id, "lon": f.lon, "lat": f.lat, "wants": f"kerb={'lowered' if f.state == 'present' else 'raised'}", "why": f"no kerb node within {NODE_MATCH_M:.0f} m"})
                continue
            nid, node, dist = hit
            current = node["tags"].get("kerb")
            if f.state == "present":
                # A flush kerb is also a curb cut. Do not downgrade it to lowered.
                value = current if current in ("flush", "lowered") else "lowered"
            else:
                value = "raised"
            note = "Disagrees with the current OpenStreetMap tag. Check on the ground." if current not in (None, value) else ""
            proposal = ("node", int(nid), "kerb", value, current, dist, False, note)
        elif f.category == "surface" and f.material:
            snap = graph.nearest_edge(f.lon, f.lat, WAY_MATCH_M)
            if snap is None:
                continue
            current = graph.osm_ways[str(snap.edge.way)]["tags"].get("surface")
            proposal = ("way", snap.edge.way, "surface", f.material, current, snap.distance_m, False, "Applies to the whole way. Split the way first if only part of it has this surface.")
        elif f.category == "slope" and f.state == "steep" and f.value is not None:
            snap = graph.nearest_edge(f.lon, f.lat, WAY_MATCH_M)
            if snap is None:
                continue
            current = graph.osm_ways[str(snap.edge.way)]["tags"].get("incline")
            proposal = ("way", snap.edge.way, "incline", f"{abs(f.value):.0f}%", current, snap.distance_m, True, "The sign depends on the direction the way is drawn in. Choose uphill or downhill before approving.")
        if proposal is None:
            continue
        element, osm_id, key, value, current, dist, needs_direction, note = proposal
        cid = f"{element}-{osm_id}-{key}"
        decision, sign = decisions.get(cid, ("pending", None))
        if needs_direction and sign is not None:
            value = f"{'-' if sign < 0 else ''}{abs(f.value or 0):.0f}%"
        kind = "confirm" if current == value else "change" if current is not None else "add"
        out.append(
            Candidate(
                id=cid,
                feature_id=f.id,
                element=element,
                osm_id=osm_id,
                key=key,
                value=value,
                current=current,
                kind=kind,
                distance_m=round(dist, 1),
                confidence=f.confidence,
                reporters=f.reporters,
                synthetic=f.synthetic,
                reviewed=f.reviewed == "approved",
                lon=f.lon,
                lat=f.lat,
                needs_direction=needs_direction and sign is None,
                decision=decision,
                note=note,
            )
        )
    # One proposal per element and key: keep the most confident.
    best: dict[str, Candidate] = {}
    for c in out:
        if c.id not in best or c.confidence > best[c.id].confidence:
            best[c.id] = c
    return sorted(best.values(), key=lambda c: c.id), unmatched


def osm_change(store: Store, graph: Graph, settings: Settings, allow_synthetic: bool = False) -> tuple[str, dict[str, int]]:
    """An OsmChange 0.6 document holding only the edits a reviewer approved."""
    cands, _ = candidates(store, graph, settings)
    approved = [c for c in cands if c.decision == "approved" and c.kind != "confirm" and not c.needs_direction and c.key in ALLOWED_KEYS]
    held_back = [c for c in approved if c.synthetic and not allow_synthetic]
    included = [c for c in approved if not c.synthetic or allow_synthetic]
    any_synthetic = any(c.synthetic for c in included)

    by_element: dict[tuple[str, int], dict[str, str]] = {}
    for c in included:
        by_element.setdefault((c.element, c.osm_id), {})[c.key] = c.value

    lines = ['<?xml version="1.0" encoding="UTF-8"?>']
    lines.append("<!-- Curbcut draft. Not uploaded anywhere. Open in JOSM, verify every element, then upload under your own account if you agree. -->")
    if any_synthetic:
        lines.append("<!-- SYNTHETIC DEMO DATA. These edits come from seeded demo reports, not from real observations. DO NOT UPLOAD. -->")
    if held_back:
        lines.append(f"<!-- {len(held_back)} approved edit(s) left out because they rest on synthetic demo reports. -->")
    generator = "Curbcut draft (SYNTHETIC DEMO, DO NOT UPLOAD)" if any_synthetic else "Curbcut draft (needs human review)"
    lines.append(f"<osmChange version=\"0.6\" generator={quoteattr(generator)}>")
    lines.append("  <modify>")
    for (element, osm_id), changes in sorted(by_element.items()):
        if element == "node":
            node = graph.osm_nodes[str(osm_id)]
            tags = {**node["tags"], **changes}
            if "kerb" in changes:
                tags.setdefault("barrier", "kerb")
            lines.append(f'    <node id="{osm_id}" version="{node["version"]}" lat="{node["lat"]:.7f}" lon="{node["lon"]:.7f}">')
            lines.extend(f"      <tag k={quoteattr(k)} v={quoteattr(v)}/>" for k, v in sorted(tags.items()))
            lines.append("    </node>")
        else:
            way = graph.osm_ways[str(osm_id)]
            tags = {**way["tags"], **changes}
            lines.append(f'    <way id="{osm_id}" version="{way["version"]}">')
            lines.extend(f'      <nd ref="{ref}"/>' for ref in way["nds"])
            lines.extend(f"      <tag k={quoteattr(k)} v={quoteattr(v)}/>" for k, v in sorted(tags.items()))
            lines.append("    </way>")
    lines.append("  </modify>")
    lines.append("</osmChange>")
    return "\n".join(lines) + "\n", {"included": len(included), "elements": len(by_element), "held_back_synthetic": len(held_back), "synthetic_included": sum(1 for c in included if c.synthetic)}


def dataset(store: Store, settings: Settings, now: datetime, include_synthetic: bool = True) -> dict[str, Any]:
    """Published features as GeoJSON under an open licence. No device ids, no photos."""
    features = [f for f in store.list_features(published=True) if include_synthetic or not f.synthetic]
    return {
        "type": "FeatureCollection",
        "name": "curbcut-accessibility-features",
        "license": DATASET_LICENSE,
        "license_url": "https://opendatacommons.org/licenses/odbl/1-0/",
        "attribution": "Curbcut contributors. Positions were matched against OpenStreetMap data, (c) OpenStreetMap contributors.",
        "generated_at": now.isoformat(),
        "publish_threshold": settings.publish_threshold,
        "contains_synthetic_demo_data": any(f.synthetic for f in features),
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [round(f.lon, 7), round(f.lat, 7)]},
                "properties": {
                    "id": f.id,
                    "category": f.category,
                    "state": f.state,
                    "confidence": f.confidence,
                    "agreement": f.agreement,
                    "reporters": f.reporters,
                    "reports": f.report_count,
                    "value": f.value,
                    "material": f.material,
                    "verified_by_reviewer": f.reviewed == "approved",
                    "synthetic": f.synthetic,
                    "updated_at": f.updated_at.isoformat(),
                },
            }
            for f in features
        ],
    }


def candidate_dicts(cands: list[Candidate]) -> list[dict[str, Any]]:
    return [asdict(c) for c in cands]
