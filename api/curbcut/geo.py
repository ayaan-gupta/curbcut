"""Small-area geometry in metres. Everything here is pure and has no dependencies."""

from __future__ import annotations

import math

EARTH_RADIUS_M = 6371008.8


def haversine_m(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    h = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(h))


def to_xy(lon: float, lat: float, lon0: float, lat0: float) -> tuple[float, float]:
    """Metres east and north of a reference point. Accurate to centimetres over a neighbourhood."""
    x = math.radians(lon - lon0) * EARTH_RADIUS_M * math.cos(math.radians(lat0))
    y = math.radians(lat - lat0) * EARTH_RADIUS_M
    return x, y


def project_on_polyline(lon: float, lat: float, coords: list[list[float]]) -> tuple[float, float, list[float]]:
    """Closest point on a polyline.

    Returns (distance to the line in metres, distance along the line to that
    point in metres, the point as [lon, lat]).
    """
    best = (float("inf"), 0.0, list(coords[0]))
    along = 0.0
    px, py = 0.0, 0.0
    for a, b in zip(coords, coords[1:]):
        ax, ay = to_xy(a[0], a[1], lon, lat)
        bx, by = to_xy(b[0], b[1], lon, lat)
        dx, dy = bx - ax, by - ay
        seg2 = dx * dx + dy * dy
        t = 0.0 if seg2 == 0 else max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / seg2))
        cx, cy = ax + t * dx, ay + t * dy
        dist = math.hypot(cx - px, cy - py)
        seg = math.sqrt(seg2)
        if dist < best[0]:
            best = (dist, along + t * seg, [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])])
        along += seg
    return best


def cut_polyline(coords: list[list[float]], start_m: float, end_m: float) -> list[list[float]]:
    """The part of a polyline between two distances along it. If start is past end the result runs backwards."""
    if start_m > end_m:
        return list(reversed(cut_polyline(coords, end_m, start_m)))
    out: list[list[float]] = []
    along = 0.0
    for a, b in zip(coords, coords[1:]):
        seg = haversine_m(a[0], a[1], b[0], b[1])
        lo, hi = along, along + seg
        if hi >= start_m and lo <= end_m and seg > 0:
            t0 = max(0.0, (start_m - lo) / seg)
            t1 = min(1.0, (end_m - lo) / seg)
            p0 = [a[0] + t0 * (b[0] - a[0]), a[1] + t0 * (b[1] - a[1])]
            p1 = [a[0] + t1 * (b[0] - a[0]), a[1] + t1 * (b[1] - a[1])]
            if not out or out[-1] != p0:
                out.append(p0)
            if p1 != p0:
                out.append(p1)
        along = hi
    if not out:
        out = [list(coords[0])]
    return out
