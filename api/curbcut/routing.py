"""Accessible routing: profile limits applied as hard filters on the walking graph."""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field

from .geo import cut_polyline
from .graph import Condition, Edge, Graph, Snap
from .models import Barrier, RouteLeg, RouteResponse

# Typical heights. OpenStreetMap's kerb=lowered means about 3 cm, kerb=raised anything a wheel cannot roll over.
KERB_HEIGHT_CM = {"flush": 0.0, "no": 0.0, "lowered": 3.0, "rolled": 6.0, "raised": 12.0, "yes": 12.0}
UNPAVED = {"gravel", "sand", "dirt", "earth", "ground", "grass", "mud", "unpaved", "pebblestone", "woodchips", "cobblestone", "unhewn_cobblestone", "sett", "rock"}
SNAP_M = 60.0


@dataclass(frozen=True)
class ProfileLimits:
    label: str
    max_incline: float | None
    max_kerb_cm: float | None
    blocked_surfaces: frozenset[str]
    penalised_surfaces: dict[str, float] = field(default_factory=dict)
    steps: bool = False
    rough: float | None = None  # None closes the edge, a number multiplies its cost
    bump: float | None = None
    damaged: float | None = None
    blocked_ramp: float | None = None


PROFILES: dict[str, ProfileLimits] = {
    # 8 percent is close to the 1:12 limit for a ramp someone pushes themselves up.
    "manual_wheelchair": ProfileLimits(
        label="Manual wheelchair",
        max_incline=8.0,
        max_kerb_cm=3.0,
        blocked_surfaces=frozenset(UNPAVED),
        penalised_surfaces={"fine_gravel": 1.6, "compacted": 1.4, "paving_stones": 1.15, "bricks": 1.25, "brick": 1.25, "wood": 1.3, "concrete:plates": 1.1},
    ),
    "powered_wheelchair": ProfileLimits(
        label="Powered wheelchair",
        max_incline=12.0,
        max_kerb_cm=6.0,
        blocked_surfaces=frozenset({"sand", "mud", "grass", "pebblestone", "woodchips", "unhewn_cobblestone", "rock", "earth", "ground"}),
        penalised_surfaces={"gravel": 2.0, "dirt": 1.8, "sett": 1.5, "cobblestone": 1.8, "fine_gravel": 1.2, "unpaved": 1.8},
        rough=2.5,
        bump=2.0,
        damaged=2.0,
    ),
    "stroller": ProfileLimits(
        label="Stroller",
        max_incline=15.0,
        max_kerb_cm=15.0,
        blocked_surfaces=frozenset({"sand", "mud", "rock"}),
        penalised_surfaces={"gravel": 1.6, "grass": 1.8, "dirt": 1.4, "cobblestone": 1.5, "unhewn_cobblestone": 2.0, "sett": 1.3, "pebblestone": 2.0, "woodchips": 1.8},
        rough=1.5,
        bump=1.3,
        damaged=1.5,
    ),
    "walking": ProfileLimits(label="Walking", max_incline=None, max_kerb_cm=None, blocked_surfaces=frozenset(), steps=True, rough=1.0, bump=1.0, damaged=1.0, blocked_ramp=1.2),
}


@dataclass(frozen=True)
class Verdict:
    passable: bool
    multiplier: float = 1.0
    reason: str | None = None
    code: str | None = None
    cautions: tuple[str, ...] = ()


def kerb_height(edge: Edge, cond: Condition | None) -> tuple[float | None, bool]:
    """Tallest kerb on a crossing, and whether a crowd report is what says so."""
    states = [s for s in (edge.kerb_u, edge.kerb_v) if s is not None] if edge.kind == "crossing" else []
    reported = False
    if cond is not None and cond.kerb is not None:
        if cond.kerb == "raised":
            states.append("raised")
            reported = True
        elif not states:
            states.append(cond.kerb)
    if not states:
        return None, False
    return max(KERB_HEIGHT_CM.get(s, 12.0) for s in states), reported


def judge(edge: Edge, cond: Condition | None, profile: str) -> Verdict:
    """Can this profile use this edge, and if so at what cost multiplier."""
    p = PROFILES[profile]
    mult = 1.0
    cautions: list[str] = []
    if edge.kind == "steps" and not p.steps:
        return Verdict(False, reason="steps", code="steps")

    height, reported = kerb_height(edge, cond)
    if height is not None and p.max_kerb_cm is not None and height > p.max_kerb_cm:
        return Verdict(False, reason="no curb cut (reported)" if reported else "no curb cut (raised kerb in OpenStreetMap)", code="no_curb_cut")
    if edge.kind == "crossing" and height is None and p.max_kerb_cm is not None:
        mult *= 1.2
        cautions.append("kerb height unknown at a crossing")

    surface = (cond.material if cond and cond.material else None) or edge.surface
    if surface in p.blocked_surfaces:
        return Verdict(False, reason=f"surface: {surface}", code="surface")
    if surface in p.penalised_surfaces:
        mult *= p.penalised_surfaces[surface]

    incline = edge.incline
    if cond is not None and cond.incline is not None:
        incline = max(incline or 0.0, cond.incline)
    if incline is not None and p.max_incline is not None and incline > p.max_incline:
        return Verdict(False, reason=f"slope {incline:.0f}% is over the {p.max_incline:.0f}% limit", code="slope")
    if incline is not None and p.max_incline is not None and incline > 0.6 * p.max_incline:
        mult *= 1.0 + 0.5 * (incline / p.max_incline)

    if cond is not None:
        for flag, limit, text, code in (
            (cond.blocked, p.blocked_ramp, "blocked ramp (reported)", "blocked"),
            (cond.damaged, p.damaged, "damaged curb ramp (reported)", "damaged"),
            (cond.rough, p.rough, "rough surface (reported)", "rough"),
            (cond.bump, p.bump, "bump (reported)", "bump"),
        ):
            if not flag:
                continue
            if limit is None:
                return Verdict(False, reason=text, code=code)
            mult *= limit
            if limit > 1.0:
                cautions.append(text)
        if cond.low_score > 0 and profile != "walking":
            mult *= 1.0 + 1.5 * min(1.0, cond.low_score)
            cautions.append("unconfirmed report nearby")
    return Verdict(True, multiplier=mult, cautions=tuple(cautions))


@dataclass
class _Path:
    cost: float
    distance: float
    edges: list[tuple[str, int]]  # edge id and the node the edge is entered from (0 for partial end pieces)
    geometry: list[list[float]]
    cautions: list[str]
    max_incline: float | None


def _search(graph: Graph, conditions: dict[str, Condition], profile: str, start: Snap, goal: Snap) -> _Path | None:
    """Dijkstra from a point on one edge to a point on another, using only edges the profile may use."""
    verdicts: dict[str, Verdict] = {}

    def verdict(edge: Edge) -> Verdict:
        if edge.id not in verdicts:
            verdicts[edge.id] = judge(edge, conditions.get(edge.id), profile)
        return verdicts[edge.id]

    vs, vg = verdict(start.edge), verdict(goal.edge)
    if not vs.passable or not vg.passable:
        return None

    best: _Path | None = None
    # Both points on the same edge: walking straight along it is one candidate.
    if start.edge.id == goal.edge.id:
        d = abs(goal.along_m - start.along_m)
        best = _Path(d * vs.multiplier, d, [(start.edge.id, 0)], cut_polyline(start.edge.coords, start.along_m, goal.along_m), list(vs.cautions), start.edge.incline)

    dist: dict[int, float] = {}
    prev: dict[int, tuple[int, str] | None] = {}
    heap: list[tuple[float, int]] = []
    for node, d in ((start.edge.u, start.along_m), (start.edge.v, start.edge.length - start.along_m)):
        cost = d * vs.multiplier
        if cost < dist.get(node, float("inf")):
            dist[node] = cost
            prev[node] = None
            heapq.heappush(heap, (cost, node))
    goal_tail = {goal.edge.u: goal.along_m * vg.multiplier, goal.edge.v: (goal.edge.length - goal.along_m) * vg.multiplier}
    best_end: int | None = None
    best_cost = best.cost if best else float("inf")
    done: set[int] = set()
    while heap:
        cost, node = heapq.heappop(heap)
        if node in done:
            continue
        done.add(node)
        if cost >= best_cost:
            break
        if node in goal_tail and cost + goal_tail[node] < best_cost:
            best_cost = cost + goal_tail[node]
            best_end = node
        for eid in graph.adjacent.get(node, []):
            edge = graph.edges[eid]
            v = verdict(edge)
            if not v.passable:
                continue
            nxt = edge.other(node)
            ncost = cost + edge.length * v.multiplier
            if ncost < dist.get(nxt, float("inf")):
                dist[nxt] = ncost
                prev[nxt] = (node, eid)
                heapq.heappush(heap, (ncost, nxt))
    if best_end is None:
        return best

    # Walk back from the goal end to the start.
    chain: list[tuple[str, int]] = []
    node = best_end
    while prev[node] is not None:
        parent, eid = prev[node]  # type: ignore[misc]
        chain.append((eid, parent))
        node = parent
    chain.reverse()
    first_node = node
    geometry = cut_polyline(start.edge.coords, start.along_m, 0.0 if first_node == start.edge.u else start.edge.length)
    distance = start.along_m if first_node == start.edge.u else start.edge.length - start.along_m
    cautions = list(vs.cautions)
    inclines = [start.edge.incline]
    edges: list[tuple[str, int]] = [(start.edge.id, 0)]
    for eid, entered_from in chain:
        edge = graph.edges[eid]
        coords = edge.coords if entered_from == edge.u else list(reversed(edge.coords))
        geometry.extend(coords[1:] if geometry and geometry[-1] == coords[0] else coords)
        distance += edge.length
        cautions.extend(verdict(edge).cautions)
        inclines.append(edge.incline)
        edges.append((eid, entered_from))
    tail = cut_polyline(goal.edge.coords, 0.0 if best_end == goal.edge.u else goal.edge.length, goal.along_m)
    geometry.extend(tail[1:] if geometry and geometry[-1] == tail[0] else tail)
    distance += goal.along_m if best_end == goal.edge.u else goal.edge.length - goal.along_m
    cautions.extend(vg.cautions)
    inclines.append(goal.edge.incline)
    edges.append((goal.edge.id, 0))
    known = [i for i in inclines if i is not None]
    return _Path(best_cost, distance, edges, geometry, cautions, max(known) if known else None)


def _leg(graph: Graph, conditions: dict[str, Condition], path: _Path, judged_for: str) -> RouteLeg:
    barriers: list[Barrier] = []
    seen: set[str] = set()
    for eid, _ in path.edges:
        if eid in seen:
            continue
        seen.add(eid)
        edge = graph.edges[eid]
        v = judge(edge, conditions.get(eid), judged_for)
        if not v.passable:
            mid = edge.coords[len(edge.coords) // 2]
            barriers.append(Barrier(edge_id=eid, reason=v.reason or "not passable", lon=mid[0], lat=mid[1]))
    return RouteLeg(
        distance_m=round(path.distance, 1),
        geometry={"type": "LineString", "coordinates": [[round(c[0], 7), round(c[1], 7)] for c in path.geometry]},
        barriers=barriers,
        max_incline=path.max_incline,
        cautions=sorted(set(path.cautions)),
        edge_ids=[eid for eid, _ in path.edges],
    )


def route(graph: Graph, conditions: dict[str, Condition], origin: tuple[float, float], destination: tuple[float, float], profile: str) -> RouteResponse:
    """Routes for a profile and, for comparison, the plain shortest walk between the same two points."""

    def usable(edge: Edge) -> bool:
        return judge(edge, conditions.get(edge.id), profile).passable

    start_any = graph.nearest_edge(origin[0], origin[1], SNAP_M)
    goal_any = graph.nearest_edge(destination[0], destination[1], SNAP_M)
    if start_any is None or goal_any is None:
        which = "start" if start_any is None else "destination"
        return RouteResponse(status="unreachable", profile=profile, message=f"The {which} is more than {SNAP_M:.0f} m from any mapped footway in the pilot area.")  # type: ignore[arg-type]

    shortest_path = _search(graph, conditions, "walking", start_any, goal_any)
    shortest = _leg(graph, conditions, shortest_path, profile) if shortest_path else None

    start = graph.nearest_edge(origin[0], origin[1], SNAP_M, accept=usable)
    goal = graph.nearest_edge(destination[0], destination[1], SNAP_M, accept=usable)
    path = _search(graph, conditions, profile, start, goal) if start and goal else None
    label = PROFILES[profile].label.lower()
    if path is None:
        reasons = sorted({b.reason for b in shortest.barriers}) if shortest else []
        detail = f" The shortest walk crosses: {', '.join(reasons)}." if reasons else ""
        return RouteResponse(
            status="no_accessible_route",
            profile=profile,  # type: ignore[arg-type]
            shortest=shortest,
            message=f"No route for a {label} avoids every barrier between these points.{detail}",
        )
    leg = _leg(graph, conditions, path, profile)
    detour = round(leg.distance_m - shortest.distance_m, 1) if shortest else None
    avoided = len(shortest.barriers) if shortest else 0
    if avoided:
        message = f"Route for a {label}: {leg.distance_m:.0f} m, {max(0.0, detour or 0.0):.0f} m longer than the shortest walk, which crosses {avoided} barrier{'s' if avoided != 1 else ''}."
    else:
        message = f"Route for a {label}: {leg.distance_m:.0f} m. The shortest walk is already clear."
    return RouteResponse(status="ok", profile=profile, route=leg, shortest=shortest, detour_m=detour, message=message)  # type: ignore[arg-type]
