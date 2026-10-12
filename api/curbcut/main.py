"""HTTP API. `create_app` takes its dependencies so tests can pass a store and a clock."""

from __future__ import annotations

import json
import re
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Any, AsyncIterator, Callable

from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, Response, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from pydantic import ValidationError

from . import osm_export, routing
from .abuse import RateLimited
from .config import Settings, get_settings
from .graph import Graph
from .models import ExportDecision, Feature, Report, ReportIn, ReviewDecision, RouteRequest, RouteResponse, SubmitResult
from .photos import BadPhoto, save_photo
from .seed import demo_points, seed_demo
from .service import Service, utcnow
from .store import MemoryStore, Store

DEVICE_ID = re.compile(r"^[A-Za-z0-9_-]{8,64}$")


def make_store(settings: Settings) -> Store:
    if settings.database_url:
        from .store_postgis import connect

        return connect(settings.database_url)
    return MemoryStore()


def feature_geojson(f: Feature) -> dict[str, Any]:
    return {
        "type": "Feature",
        "id": f.id,
        "geometry": {"type": "Point", "coordinates": [f.lon, f.lat]},
        "properties": {
            "id": f.id,
            "category": f.category,
            "state": f.state,
            "kind": f"{f.category}_{f.state}",
            "confidence": f.confidence,
            "agreement": f.agreement,
            "reporters": f.reporters,
            "reports": f.report_count,
            "published": f.published,
            "reviewed": f.reviewed,
            "value": f.value,
            "material": f.material,
            "synthetic": f.synthetic,
            "updated_at": f.updated_at.isoformat(),
        },
    }


def public_report(r: Report) -> dict[str, Any]:
    """A report as shown to anyone but the store: no device hash, no photo path."""
    return {
        "id": r.id,
        "category": r.category,
        "state": r.state,
        "lon": r.lon,
        "lat": r.lat,
        "source": r.source,
        "model_confidence": r.model_confidence,
        "value": r.value,
        "material": r.material,
        "created_at": r.created_at.isoformat(),
        "status": r.status,
        "flags": r.flags,
        "feature_id": r.feature_id,
        "has_photo": r.photo is not None,
        "synthetic": r.synthetic,
    }


def create_app(settings: Settings | None = None, store: Store | None = None, clock: Callable[[], datetime] = utcnow) -> FastAPI:
    settings = settings or get_settings()
    state: dict[str, Any] = {}

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        graph = Graph.load(settings.pilot_file)
        the_store = store or make_store(settings)
        service = Service(the_store, graph, settings)
        if settings.seed_demo and the_store.counts()["reports"] == 0:
            seed_demo(service, clock())
        state.update(graph=graph, store=the_store, service=service)
        yield
        close = getattr(the_store, "close", None)
        if close and store is None:
            close()

    app = FastAPI(title="Curbcut API", version="0.1.0", lifespan=lifespan)
    app.add_middleware(CORSMiddleware, allow_origins=[o.strip() for o in settings.cors_origins.split(",") if o.strip()], allow_methods=["*"], allow_headers=["*"])

    def svc() -> Service:
        return state["service"]

    def graph_dep() -> Graph:
        return state["graph"]

    def device(x_device_id: str | None = Header(default=None)) -> str:
        if x_device_id is None or not DEVICE_ID.match(x_device_id):
            raise HTTPException(400, "Send an X-Device-Id header: 8 to 64 letters, digits, dashes or underscores.")
        return x_device_id

    def reviewer(authorization: str | None = Header(default=None)) -> None:
        if settings.reviewer_token and authorization != f"Bearer {settings.reviewer_token}":
            raise HTTPException(401, "Reviewer token required.")

    @app.exception_handler(RateLimited)
    async def rate_limited(_: Any, error: RateLimited) -> JSONResponse:
        return JSONResponse({"detail": str(error)}, status_code=429, headers={"Retry-After": str(error.retry_after_s)})

    # Meta

    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {"ok": True, "store": state["store"].kind, **state["store"].counts()}

    @app.get("/api/meta")
    def meta(graph: Graph = Depends(graph_dep)) -> dict[str, Any]:
        return {
            "pilot": graph.meta,
            "publish_threshold": settings.publish_threshold,
            "cluster_radius_m": settings.cluster_radius_m,
            "review_needs_token": bool(settings.reviewer_token),
            "profiles": {
                key: {"label": p.label, "max_incline": p.max_incline, "max_kerb_cm": p.max_kerb_cm, "steps": p.steps, "blocked_surfaces": sorted(p.blocked_surfaces)}
                for key, p in routing.PROFILES.items()
            },
            "demo": demo_points() if settings.seed_demo else None,
            "counts": state["store"].counts(),
        }

    @app.get("/api/basemap")
    def basemap(response: Response, graph: Graph = Depends(graph_dep)) -> dict[str, Any]:
        """Roads, buildings and parks of the pilot area as GeoJSON, for the offline map style."""
        response.headers["Cache-Control"] = "public, max-age=86400"

        def collection(items: list[dict[str, Any]], geometry: str) -> dict[str, Any]:
            return {
                "type": "FeatureCollection",
                "features": [
                    {
                        "type": "Feature",
                        "geometry": {"type": geometry, "coordinates": [it["coords"]] if geometry == "Polygon" else it["coords"]},
                        "properties": {"cls": it["cls"], "name": it.get("name", "")},
                    }
                    for it in items
                ],
            }

        return {"roads": collection(graph.roads, "LineString"), "areas": collection(graph.areas, "Polygon"), "attribution": graph.meta["attribution"]}

    @app.get("/api/network")
    def network(profile: str = Query("manual_wheelchair"), graph: Graph = Depends(graph_dep), service: Service = Depends(svc)) -> dict[str, Any]:
        """Every footway edge with whether the profile can use it and why not."""
        if profile not in routing.PROFILES:
            raise HTTPException(422, f"profile must be one of {sorted(routing.PROFILES)}")
        conditions = service.conditions()
        features = []
        for edge in graph.edges.values():
            v = routing.judge(edge, conditions.get(edge.id), profile)
            features.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "LineString", "coordinates": edge.coords},
                    "properties": {
                        "id": edge.id,
                        "kind": edge.kind,
                        "passable": v.passable,
                        "code": v.code or ("caution" if v.cautions or v.multiplier > 1.3 else "ok"),
                        "reason": v.reason or "; ".join(v.cautions),
                        "incline": edge.incline,
                        "surface": edge.surface,
                    },
                }
            )
        return {"type": "FeatureCollection", "features": features}

    # Reports

    @app.post("/api/reports", response_model=SubmitResult)
    async def submit_report(
        report: str = Form(..., description="A ReportIn as JSON."),
        photo: UploadFile | None = File(default=None),
        raw_device: str = Depends(device),
        service: Service = Depends(svc),
    ) -> SubmitResult:
        try:
            report_in = ReportIn.model_validate_json(report)
        except ValidationError as error:
            raise HTTPException(422, json.loads(error.json(include_url=False, include_context=False))) from error
        photo_name = None
        if photo is not None:
            if report_in.source != "photo":
                raise HTTPException(422, "Only photo reports carry a photo.")
            try:
                photo_name = save_photo(await photo.read(), settings.upload_dir, settings.max_photo_bytes)
            except BadPhoto as error:
                raise HTTPException(422, str(error)) from error
        return await run_in_threadpool(service.submit, report_in, raw_device, clock(), photo_name)

    @app.post("/api/reports/batch")
    def submit_batch(reports: list[ReportIn], raw_device: str = Depends(device), service: Service = Depends(svc)) -> dict[str, Any]:
        """Tagged points from one recording session. Stops at the rate limit and says how far it got."""
        if len(reports) > 60:
            raise HTTPException(413, "At most 60 reports per batch.")
        results: list[SubmitResult] = []
        limited: str | None = None
        for item in reports:
            try:
                results.append(service.submit(item, raw_device, now=clock()))
            except RateLimited as error:
                limited = str(error)
                break
        return {"accepted": len(results), "rate_limited": limited, "results": [r.model_dump(mode="json") for r in results]}

    @app.get("/api/features")
    def features(include: str = Query("published", pattern="^(published|all)$"), service: Service = Depends(svc)) -> dict[str, Any]:
        rows = service.store.list_features(published=True if include == "published" else None)
        return {"type": "FeatureCollection", "features": [feature_geojson(f) for f in rows]}

    @app.get("/api/features/{feature_id}")
    def feature_detail(feature_id: int, service: Service = Depends(svc)) -> dict[str, Any]:
        f = service.store.get_feature(feature_id)
        if f is None:
            raise HTTPException(404, "No such feature.")
        return {"feature": feature_geojson(f), "reports": [public_report(r) for r in service.store.reports_for_feature(feature_id)]}

    # Routing

    @app.post("/api/route", response_model=RouteResponse)
    def route(request: RouteRequest, graph: Graph = Depends(graph_dep), service: Service = Depends(svc)) -> RouteResponse:
        return routing.route(graph, service.conditions(), request.origin, request.destination, request.profile)

    # Review queue

    @app.get("/api/review", dependencies=[Depends(reviewer)])
    def review_queue(service: Service = Depends(svc)) -> dict[str, Any]:
        held = [public_report(r) for r in service.store.list_reports(status="review")]
        conflicts = [feature_geojson(f)["properties"] | {"lon": f.lon, "lat": f.lat} for f in service.store.list_features() if f.reporters >= 2 and f.agreement < 0.7 and f.reviewed is None]
        lone = [feature_geojson(f)["properties"] | {"lon": f.lon, "lat": f.lat} for f in service.store.list_features(published=False) if f.reviewed is None and f.agreement >= 0.7 and f.report_count > 0]
        return {"held_reports": held, "conflicts": conflicts, "below_threshold": lone}

    @app.post("/api/review/reports/{report_id}", dependencies=[Depends(reviewer)])
    def review_report(report_id: int, decision: ReviewDecision, service: Service = Depends(svc)) -> dict[str, Any]:
        try:
            return public_report(service.review_report(report_id, decision.action))
        except KeyError:
            raise HTTPException(404, "No such report.") from None
        except ValueError as error:
            raise HTTPException(422, str(error)) from error

    @app.post("/api/review/features/{feature_id}", dependencies=[Depends(reviewer)])
    def review_feature(feature_id: int, decision: ReviewDecision, service: Service = Depends(svc)) -> dict[str, Any]:
        try:
            return feature_geojson(service.review_feature(feature_id, decision.action))
        except KeyError:
            raise HTTPException(404, "No such feature.") from None
        except ValueError as error:
            raise HTTPException(422, str(error)) from error

    @app.get("/api/review/photos/{report_id}", dependencies=[Depends(reviewer)])
    def review_photo(report_id: int, service: Service = Depends(svc)) -> FileResponse:
        report = service.store.get_report(report_id)
        if report is None or report.photo is None or not (settings.upload_dir / report.photo).exists():
            raise HTTPException(404, "No photo for that report.")
        return FileResponse(settings.upload_dir / report.photo, media_type="image/jpeg")

    # Export

    @app.get("/api/export/candidates", dependencies=[Depends(reviewer)])
    def export_candidates(graph: Graph = Depends(graph_dep), service: Service = Depends(svc)) -> dict[str, Any]:
        cands, unmatched = osm_export.candidates(service.store, graph, settings)
        return {"candidates": osm_export.candidate_dicts(cands), "unmatched": unmatched, "export_threshold": settings.export_threshold}

    @app.post("/api/export/candidates/{candidate_id}", dependencies=[Depends(reviewer)])
    def decide_candidate(candidate_id: str, decision: ExportDecision, graph: Graph = Depends(graph_dep), service: Service = Depends(svc)) -> dict[str, Any]:
        cands, _ = osm_export.candidates(service.store, graph, settings)
        if candidate_id not in {c.id for c in cands}:
            raise HTTPException(404, "No such export candidate.")
        service.store.set_export_decision(candidate_id, decision.decision, decision.incline_sign)
        cands, _ = osm_export.candidates(service.store, graph, settings)
        return next(osm_export.candidate_dicts([c])[0] for c in cands if c.id == candidate_id)

    @app.get("/api/export/osmchange", dependencies=[Depends(reviewer)])
    def export_osmchange(allow_synthetic: bool = False, graph: Graph = Depends(graph_dep), service: Service = Depends(svc)) -> PlainTextResponse:
        xml, summary = osm_export.osm_change(service.store, graph, settings, allow_synthetic=allow_synthetic)
        name = "curbcut-draft-SYNTHETIC-do-not-upload.osc" if summary["synthetic_included"] else "curbcut-draft.osc"
        return PlainTextResponse(
            xml,
            media_type="application/xml",
            headers={"Content-Disposition": f'attachment; filename="{name}"', "X-Curbcut-Export": json.dumps(summary)},
        )

    @app.get("/api/export/dataset.geojson")
    def export_dataset(include_synthetic: bool = True, service: Service = Depends(svc)) -> JSONResponse:
        data = osm_export.dataset(service.store, settings, clock(), include_synthetic=include_synthetic)
        return JSONResponse(data, media_type="application/geo+json", headers={"Content-Disposition": 'attachment; filename="curbcut-features.geojson"'})

    return app


app = create_app()
