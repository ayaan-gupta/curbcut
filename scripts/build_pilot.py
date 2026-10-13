"""Build the committed pilot area from real OpenStreetMap data.

Pulls the pedestrian network for a small bounding box from the Overpass API,
looks up ground elevation for graph nodes from the USGS 3DEP point query
service, and writes one compact JSON file that the API and the web app read.
A clean clone never needs the network: the output is committed.

    .venv/bin/python scripts/build_pilot.py            # reuse data/raw if present
    .venv/bin/python scripts/build_pilot.py --refresh  # fetch again

Raw downloads go to data/raw (git-ignored). The output keeps OSM element ids
and versions so the export can draft OsmChange edits, and drops contributor
names, uids, changesets and timestamps.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import requests

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "pilot" / "capitol-hill.json"

# south, west, north, east
BBOX = (47.6135, -122.3250, 47.6195, -122.3160)
NAME = "Capitol Hill, Seattle"
OVERPASS = "https://overpass-api.de/api/interpreter"
EPQS = "https://epqs.nationalmap.gov/v1/json"
UA = {"User-Agent": "curbcut-pilot-build/0.1 (https://github.com/ayaan-gupta/curbcut)"}

WALK_HIGHWAYS = {"footway", "path", "pedestrian", "steps", "living_street", "cycleway"}
ROAD_HIGHWAYS = {"residential", "service", "unclassified", "tertiary", "secondary", "primary", "living_street", "pedestrian"}
ALL_HIGHWAYS = sorted(WALK_HIGHWAYS | ROAD_HIGHWAYS)
KEEP_NODE_TAGS = {"kerb", "barrier", "highway", "crossing", "crossing:markings", "tactile_paving", "crossing:island", "wheelchair"}
KEEP_WAY_TAGS = {
    "highway", "footway", "surface", "smoothness", "incline", "width", "name", "kerb", "crossing", "crossing:markings",
    "wheelchair", "access", "foot", "step_count", "handrail", "ramp", "ramp:wheelchair", "lit", "tactile_paving", "bridge", "tunnel", "layer",
}


def haversine(a: tuple[float, float], b: tuple[float, float]) -> float:
    """Metres between two (lon, lat) points."""
    lon1, lat1, lon2, lat2 = map(math.radians, (a[0], a[1], b[0], b[1]))
    h = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371008.8 * math.asin(math.sqrt(h))


def overpass(query: str) -> dict[str, Any]:
    for attempt in range(4):
        r = requests.post(OVERPASS, data={"data": query}, headers=UA, timeout=180)
        if r.status_code == 200 and r.headers.get("content-type", "").startswith("application/json"):
            return r.json()
        time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"Overpass failed: {r.status_code} {r.text[:200]}")


def fetch_raw(refresh: bool) -> tuple[dict[str, Any], dict[str, Any]]:
    RAW.mkdir(parents=True, exist_ok=True)
    walk_path, ctx_path = RAW / "overpass-walk.json", RAW / "overpass-context.json"
    bbox = ",".join(str(v) for v in BBOX)
    if refresh or not walk_path.exists():
        hw = "|".join(ALL_HIGHWAYS)
        walk_path.write_text(json.dumps(overpass(f'[out:json][timeout:90];(way[highway~"^({hw})$"]({bbox}););out meta;>;out meta;')))
        time.sleep(3)
    if refresh or not ctx_path.exists():
        ctx_path.write_text(
            json.dumps(overpass(f'[out:json][timeout:90];(way[building]({bbox});way[leisure~"^(park|pitch|playground)$"]({bbox});way[natural=water]({bbox}););out tags geom;'))
        )
    return json.loads(walk_path.read_text()), json.loads(ctx_path.read_text())


def is_walkable(tags: dict[str, str]) -> bool:
    if tags.get("highway") not in WALK_HIGHWAYS:
        return False
    if tags.get("access") in {"private", "no"} or tags.get("foot") in {"no", "private"}:
        return False
    if tags.get("highway") == "cycleway" and tags.get("foot") not in {"yes", "designated", "permissive"}:
        return False
    return True


def edge_kind(tags: dict[str, str]) -> str:
    hw = tags.get("highway", "")
    if hw == "steps":
        return "steps"
    if hw == "footway":
        fw = tags.get("footway", "")
        if fw in {"sidewalk", "crossing"}:
            return fw
        return "footway"
    return hw


def fetch_elevations(points: dict[str, tuple[float, float]], refresh: bool) -> dict[str, float]:
    """Ground elevation in metres from USGS 3DEP (public domain), cached on disk."""
    cache_path = RAW / "elevation.json"
    cache: dict[str, float] = {} if refresh or not cache_path.exists() else json.loads(cache_path.read_text())
    todo = [k for k in points if k not in cache]

    def one(key: str) -> tuple[str, float | None]:
        lon, lat = points[key]
        for attempt in range(4):
            try:
                r = requests.get(EPQS, params={"x": lon, "y": lat, "units": "Meters", "wkid": 4326}, headers=UA, timeout=40)
                value = float(r.json()["value"])
                if value > -1000:
                    return key, value
            except Exception:  # noqa: BLE001 - the service is flaky, retry then give up on this point
                time.sleep(1.5 * (attempt + 1))
        return key, None

    if todo:
        print(f"elevation: {len(todo)} points to fetch from USGS 3DEP")
        with ThreadPoolExecutor(max_workers=6) as pool:
            for i, (key, value) in enumerate(pool.map(one, todo)):
                if value is not None:
                    cache[key] = round(value, 2)
                if i % 100 == 99:
                    cache_path.write_text(json.dumps(cache))
                    print(f"  {i + 1}/{len(todo)}")
        cache_path.write_text(json.dumps(cache))
    return cache


def simplify_ring(coords: list[list[float]], tolerance_m: float = 1.0) -> list[list[float]]:
    """Radial-distance simplification, good enough for building outlines on a basemap."""
    if len(coords) <= 5:
        return coords
    out = [coords[0]]
    for c in coords[1:-1]:
        if haversine((out[-1][0], out[-1][1]), (c[0], c[1])) >= tolerance_m:
            out.append(c)
    out.append(coords[-1])
    return out


def build(refresh: bool) -> dict[str, Any]:
    walk, ctx = fetch_raw(refresh)
    nodes = {e["id"]: e for e in walk["elements"] if e["type"] == "node"}
    ways = [e for e in walk["elements"] if e["type"] == "way"]
    s, w, n, e = BBOX

    def coord(nid: int) -> tuple[float, float]:
        node = nodes[nid]
        return (round(node["lon"], 7), round(node["lat"], 7))

    walk_ways = [wy for wy in ways if is_walkable(wy.get("tags", {}))]
    use_count: Counter[int] = Counter()
    for wy in walk_ways:
        for i, nid in enumerate(wy["nodes"]):
            use_count[nid] += 2 if i in (0, len(wy["nodes"]) - 1) else 1

    def is_split(nid: int) -> bool:
        tags = nodes[nid].get("tags", {})
        return use_count[nid] > 1 or "kerb" in tags or tags.get("barrier") == "kerb"

    # Split every walkable way at shared nodes and at kerb nodes.
    raw_edges: list[dict[str, Any]] = []
    for wy in walk_ways:
        tags = wy["tags"]
        start = 0
        ids = wy["nodes"]
        for i in range(1, len(ids)):
            if i == len(ids) - 1 or is_split(ids[i]):
                seg = ids[start : i + 1]
                if seg[0] != seg[-1] or len(seg) > 2:
                    raw_edges.append({"way": wy["id"], "nds": seg, "tags": tags})
                start = i

    # Keep the largest connected component so every pair of points can be asked about.
    adj: dict[int, list[int]] = defaultdict(list)
    for idx, ed in enumerate(raw_edges):
        adj[ed["nds"][0]].append(idx)
        adj[ed["nds"][-1]].append(idx)
    seen_edge: dict[int, int] = {}
    comp_sizes: Counter[int] = Counter()
    for idx in range(len(raw_edges)):
        if idx in seen_edge:
            continue
        stack = [idx]
        seen_edge[idx] = idx
        while stack:
            cur = stack.pop()
            comp_sizes[idx] += 1
            for end in (raw_edges[cur]["nds"][0], raw_edges[cur]["nds"][-1]):
                for nxt in adj[end]:
                    if nxt not in seen_edge:
                        seen_edge[nxt] = idx
                        stack.append(nxt)
    main = comp_sizes.most_common(1)[0][0]
    kept = [ed for idx, ed in enumerate(raw_edges) if seen_edge[idx] == main]
    print(f"walkable ways {len(walk_ways)}, edges {len(raw_edges)}, largest component {len(kept)} ({len(kept) / len(raw_edges):.1%})")

    # Elevation: every edge endpoint, plus interior vertices roughly every 20 m on long edges.
    need: dict[str, tuple[float, float]] = {}
    samples: list[list[int]] = []
    for ed in kept:
        ids = ed["nds"]
        chosen = [ids[0]]
        run = 0.0
        for a, b in zip(ids, ids[1:]):
            run += haversine(coord(a), coord(b))
            if b == ids[-1] or run >= 20.0:
                chosen.append(b)
                run = 0.0
        samples.append(chosen)
        for nid in chosen:
            need[str(nid)] = coord(nid)
    elevation = fetch_elevations(need, refresh)

    edges: list[dict[str, Any]] = []
    used_nodes: set[int] = set()
    for ed, chosen in zip(kept, samples):
        ids = ed["nds"]
        tags = ed["tags"]
        coords = [list(coord(nid)) for nid in ids]
        length = sum(haversine(coord(a), coord(b)) for a, b in zip(ids, ids[1:]))
        # Steepest stretch of at least 12 m; shorter spans are inside the noise of a 1 m DEM.
        incline: float | None = None
        if all(str(nid) in elevation for nid in chosen):
            best = 0.0
            measured = False
            for i in range(len(chosen)):
                dist = 0.0
                for j in range(i + 1, len(chosen)):
                    a_idx, b_idx = ids.index(chosen[j - 1]), ids.index(chosen[j])
                    dist += sum(haversine(coord(p), coord(q)) for p, q in zip(ids[a_idx:b_idx], ids[a_idx + 1 : b_idx + 1]))
                    if dist >= 12.0:
                        rise = abs(elevation[str(chosen[j])] - elevation[str(chosen[i])])
                        best = max(best, 100.0 * rise / dist)
                        measured = True
                        break
            incline = round(best, 1) if measured else None
        kerbs = {}
        for end, nid in (("u", ids[0]), ("v", ids[-1])):
            ntags = nodes[nid].get("tags", {})
            if "kerb" in ntags:
                kerbs[end] = ntags["kerb"]
        out: dict[str, Any] = {
            "id": f"w{ed['way']}-{ids[0]}-{ids[-1]}",
            "way": ed["way"],
            "u": ids[0],
            "v": ids[-1],
            "kind": edge_kind(tags),
            "len": round(length, 1),
            "coords": coords,
        }
        if tags.get("surface"):
            out["surface"] = tags["surface"]
        if tags.get("smoothness"):
            out["smoothness"] = tags["smoothness"]
        if incline is not None:
            out["incline"] = incline
        if kerbs:
            out["kerb"] = kerbs
        if tags.get("name"):
            out["name"] = tags["name"]
        edges.append(out)
        used_nodes.update(ids)

    kept_way_ids = {ed["way"] for ed in kept}
    osm_ways = {
        str(wy["id"]): {"version": wy["version"], "nds": wy["nodes"], "tags": wy["tags"]}
        for wy in walk_ways
        if wy["id"] in kept_way_ids
    }
    # Nodes an export might retag: anything already tagged, and both ends of every
    # crossing, where a kerb tag would go even if there is none yet.
    crossing_ends = {nid for ed in kept if edge_kind(ed["tags"]) == "crossing" for nid in (ed["nds"][0], ed["nds"][-1])}
    osm_nodes = {}
    for nid in used_nodes:
        node = nodes[nid]
        tags = node.get("tags", {})
        if any(k in KEEP_NODE_TAGS for k in tags) or nid in crossing_ends:
            osm_nodes[str(nid)] = {"version": node["version"], "lon": round(node["lon"], 7), "lat": round(node["lat"], 7), "tags": tags}

    # Context for the offline basemap: roads, buildings, parks, water.
    roads = []
    for wy in ways:
        tags = wy.get("tags", {})
        if tags.get("highway") in ROAD_HIGHWAYS and tags.get("highway") not in {"pedestrian"}:
            roads.append({"cls": tags["highway"], "name": tags.get("name", ""), "coords": [[round(nodes[i]["lon"], 6), round(nodes[i]["lat"], 6)] for i in wy["nodes"]]})
    areas = []
    for el in ctx["elements"]:
        tags = el.get("tags", {})
        cls = "building" if "building" in tags else tags.get("leisure") or tags.get("natural") or "other"
        ring = [[round(p["lon"], 6), round(p["lat"], 6)] for p in el.get("geometry", [])]
        if len(ring) >= 4:
            entry: dict[str, Any] = {"cls": cls, "coords": simplify_ring(ring, 1.5 if cls == "building" else 3.0)}
            if cls == "park" and tags.get("name"):
                entry["name"] = tags["name"]
            areas.append(entry)

    kinds = Counter(ed["kind"] for ed in edges)
    kerb_counts = Counter(v["tags"].get("kerb", "none") for v in osm_nodes.values() if "kerb" in v["tags"])
    with_incline = [ed["incline"] for ed in edges if "incline" in ed]
    print("edge kinds", dict(kinds))
    print("kerb nodes", dict(kerb_counts))
    print(f"incline known on {len(with_incline)}/{len(edges)} edges, max {max(with_incline):.1f}%, over 8%: {sum(1 for v in with_incline if v > 8)}")

    return {
        "meta": {
            "name": NAME,
            "bbox": {"south": s, "west": w, "north": n, "east": e},
            "source": "OpenStreetMap via the Overpass API; elevation from USGS 3DEP (public domain)",
            "license": "ODbL-1.0",
            "attribution": "(c) OpenStreetMap contributors",
            "osm_base": walk.get("osm3s", {}).get("timestamp_osm_base", ""),
            "note": "Real map data. Contributor names, uids, changesets and timestamps were removed. Incline is the steepest stretch of at least 12 m along each edge, from a 1 m lidar DEM.",
        },
        "edges": edges,
        "osm_nodes": osm_nodes,
        "osm_ways": osm_ways,
        "roads": roads,
        "areas": areas,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", action="store_true", help="fetch from the network even if data/raw exists")
    args = parser.parse_args()
    data = build(args.refresh)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, separators=(",", ":"), ensure_ascii=False))
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
