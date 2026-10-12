"""Milestone 4: accessible routing with hard filters.

PRD acceptance test: "The wheelchair profile route never crosses a segment
marked 'no curb cut' when an alternative exists." The case with no
alternative is covered too: the router must say so instead of crossing.
"""

from __future__ import annotations

import random

import pytest

from curbcut import routing
from curbcut.graph import Condition, Graph, build_conditions
from curbcut.service import Service

from .conftest import NOW, at, report, street

NORTH_WEST = tuple(at(5, 10))
SOUTH_WEST = tuple(at(5, 0))


def plan(graph: Graph, conditions: dict[str, Condition], profile: str = "manual_wheelchair"):
    return routing.route(graph, conditions, NORTH_WEST, SOUTH_WEST, profile)  # type: ignore[arg-type]


def test_with_no_barriers_the_wheelchair_takes_the_near_crossing() -> None:
    result = plan(street(), {})
    assert result.status == "ok" and result.route is not None
    assert "crossA" in result.route.edge_ids and result.route.distance_m == pytest.approx(20.0, abs=0.5)
    assert result.detour_m == pytest.approx(0.0, abs=0.5)


def test_acceptance_wheelchair_route_avoids_a_no_curb_cut_segment_when_an_alternative_exists(service: Service) -> None:
    """Two people report the curb cut at crossing A missing. The feature publishes and the route goes round by B."""
    service.submit(report(0.0, 0.5), "device-one", now=NOW)
    service.submit(report(1.0, 0.0), "device-two", now=NOW)
    conditions = service.conditions()
    assert conditions["crossA"].kerb == "raised"

    result = plan(service.graph, conditions)
    assert result.status == "ok" and result.route is not None
    assert "crossA" not in result.route.edge_ids and "crossB" in result.route.edge_ids
    assert result.route.barriers == []
    assert result.route.distance_m == pytest.approx(120.0, abs=0.5) and result.detour_m == pytest.approx(100.0, abs=1.0)
    # The shortest walk is reported alongside, with the barrier it would have crossed.
    assert result.shortest is not None and [b.edge_id for b in result.shortest.barriers] == ["crossA"]
    assert "no curb cut" in result.shortest.barriers[0].reason


def test_acceptance_with_no_alternative_the_router_refuses_instead_of_crossing(service: Service) -> None:
    for x, who in ((0.0, "device-one"), (1.0, "device-two"), (60.0, "device-three"), (59.0, "device-four")):
        service.submit(report(x, 0.0), who, now=NOW)
    conditions = service.conditions()
    assert conditions["crossA"].kerb == conditions["crossB"].kerb == "raised"

    result = plan(service.graph, conditions)
    assert result.status == "no_accessible_route"
    assert result.route is None, "a refused route must not come with a line to follow"
    assert result.shortest is not None and {b.edge_id for b in result.shortest.barriers} == {"crossA"}
    assert "No route for a manual wheelchair" in result.message and "no curb cut" in result.message


def test_a_raised_kerb_in_openstreetmap_is_a_barrier_without_any_report() -> None:
    result = plan(street(kerb_a="raised"), {})
    assert result.status == "ok" and result.route is not None and "crossA" not in result.route.edge_ids
    assert plan(street(kerb_a="raised", kerb_b="raised"), {}).status == "no_accessible_route"


def test_an_unpublished_report_does_not_close_a_segment(service: Service) -> None:
    service.submit(report(0.0, 0.0, conf=0.6), "device-one", now=NOW)
    conditions = service.conditions()
    assert conditions["crossA"].kerb is None and conditions["crossA"].low_score > 0
    # It is only less attractive. With a 100 m detour as the alternative the near crossing still wins.
    result = plan(service.graph, conditions)
    assert result.route is not None and "crossA" in result.route.edge_ids
    assert "unconfirmed report nearby" in result.route.cautions


def test_profiles_differ_in_what_stops_them() -> None:
    raised = street(kerb_a="raised")
    assert "crossA" in plan(raised, {}, "stroller").route.edge_ids  # type: ignore[union-attr]
    assert "crossA" in plan(raised, {}, "walking").route.edge_ids  # type: ignore[union-attr]
    assert "crossA" not in plan(raised, {}, "powered_wheelchair").route.edge_ids  # type: ignore[union-attr]

    steep = street(crossA={"incline": 10.0})
    assert "crossA" not in plan(steep, {}, "manual_wheelchair").route.edge_ids  # type: ignore[union-attr]
    assert "crossA" in plan(steep, {}, "powered_wheelchair").route.edge_ids  # type: ignore[union-attr]

    gravel = street(crossA={"surface": "gravel"})
    assert "crossA" not in plan(gravel, {}, "manual_wheelchair").route.edge_ids  # type: ignore[union-attr]
    assert "crossA" in plan(gravel, {}, "stroller").route.edge_ids  # type: ignore[union-attr]

    stairs = street(crossA={"kind": "steps"})
    for profile in ("manual_wheelchair", "powered_wheelchair", "stroller"):
        assert "crossA" not in plan(stairs, {}, profile).route.edge_ids  # type: ignore[union-attr]


def test_reported_slope_and_roughness_are_hard_limits_for_a_manual_chair(service: Service) -> None:
    service.submit(report(0.0, 5.0, state="steep", category="slope", source="manual", value=11.0), "device-one", now=NOW)
    service.submit(report(0.5, 5.0, state="steep", category="slope", source="manual", value=12.0), "device-two", now=NOW)
    conditions = service.conditions()
    assert conditions["crossA"].incline == pytest.approx(12.0)
    verdict = routing.judge(service.graph.edges["crossA"], conditions["crossA"], "manual_wheelchair")
    assert not verdict.passable and verdict.code == "slope"
    rough = Condition(rough=True)
    assert not routing.judge(service.graph.edges["north"], rough, "manual_wheelchair").passable
    assert routing.judge(service.graph.edges["north"], rough, "powered_wheelchair").multiplier == pytest.approx(2.5)


def _reachable(graph: Graph, conditions: dict[str, Condition], profile: str, start_edge: str, goal_edge: str) -> bool:
    """An independent check: flood fill over edges the profile may use. Shares only `judge` with the router."""
    ok = {eid for eid, edge in graph.edges.items() if routing.judge(edge, conditions.get(eid), profile).passable}
    if start_edge not in ok or goal_edge not in ok:
        return False
    frontier, seen = [start_edge], {start_edge}
    while frontier:
        edge = graph.edges[frontier.pop()]
        if edge.id == goal_edge:
            return True
        for node in (edge.u, edge.v):
            for nxt in graph.adjacent[node]:
                if nxt in ok and nxt not in seen:
                    seen.add(nxt)
                    frontier.append(nxt)
    return False


def test_acceptance_holds_for_random_trips_across_the_real_pilot_area(pilot: Graph) -> None:
    """300 random trips over real Capitol Hill footways, with a fifth of the kerbed crossings marked 'no curb cut'."""
    rng = random.Random(11)
    crossings = sorted(e.id for e in pilot.edges.values() if e.kind == "crossing")
    marked = set(rng.sample(crossings, len(crossings) // 5))
    conditions = {eid: Condition(kerb="raised") for eid in marked}
    no_curb_cut = {eid for eid, e in pilot.edges.items() if routing.judge(e, conditions.get(eid), "manual_wheelchair").code == "no_curb_cut"}
    assert marked <= no_curb_cut

    box = pilot.bbox
    outcomes = {"ok": 0, "no_accessible_route": 0}
    forced_detours = 0
    for _ in range(300):
        origin = (rng.uniform(box["west"], box["east"]), rng.uniform(box["south"], box["north"]))
        destination = (rng.uniform(box["west"], box["east"]), rng.uniform(box["south"], box["north"]))
        result = routing.route(pilot, conditions, origin, destination, "manual_wheelchair")
        assert result.status in outcomes
        outcomes[result.status] += 1
        usable = lambda e: routing.judge(e, conditions.get(e.id), "manual_wheelchair").passable  # noqa: E731
        start = pilot.nearest_edge(*origin, routing.SNAP_M, accept=usable)
        goal = pilot.nearest_edge(*destination, routing.SNAP_M, accept=usable)
        alternative_exists = start is not None and goal is not None and _reachable(pilot, conditions, "manual_wheelchair", start.edge.id, goal.edge.id)
        if result.status == "ok":
            assert result.route is not None
            assert not (set(result.route.edge_ids) & no_curb_cut), "route crosses a segment marked no curb cut"
            assert result.route.barriers == []
            if result.shortest and result.shortest.barriers:
                forced_detours += 1
        else:
            assert result.route is None
        assert (result.status == "ok") == alternative_exists
    # The test only means something if both outcomes and real detours occurred.
    assert outcomes["ok"] > 150 and outcomes["no_accessible_route"] > 5 and forced_detours > 30


def test_route_geometry_is_continuous_and_starts_and_ends_near_the_request(pilot: Graph) -> None:
    from curbcut.geo import haversine_m
    from curbcut.seed import DEMO_DESTINATION, DEMO_ORIGIN

    result = routing.route(pilot, {}, DEMO_ORIGIN, DEMO_DESTINATION, "manual_wheelchair")
    assert result.status == "ok" and result.route is not None
    line = result.route.geometry["coordinates"]
    assert haversine_m(*line[0], *DEMO_ORIGIN) < 30 and haversine_m(*line[-1], *DEMO_DESTINATION) < 30
    gaps = [haversine_m(*a, *b) for a, b in zip(line, line[1:])]
    assert max(gaps) < 120 and sum(gaps) == pytest.approx(result.route.distance_m, rel=0.02)


def test_build_conditions_ignores_reviewer_rejected_features(service: Service) -> None:
    service.submit(report(0.0, 0.0), "device-one", now=NOW)
    done = service.submit(report(1.0, 0.0), "device-two", now=NOW)
    assert done.feature is not None and service.conditions()["crossA"].kerb == "raised"
    service.review_feature(done.feature.id, "reject")
    assert build_conditions(service.graph, service.store.list_features()) == {}
