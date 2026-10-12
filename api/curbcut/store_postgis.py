"""PostGIS store. Points are geography, so ST_DWithin and ST_Distance are in metres."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import psycopg
from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from .models import Feature, Report

SCHEMA = """
CREATE EXTENSION IF NOT EXISTS postgis;
CREATE TABLE IF NOT EXISTS features (
    id BIGSERIAL PRIMARY KEY,
    category TEXT NOT NULL,
    state TEXT NOT NULL,
    geog geography(Point, 4326) NOT NULL,
    confidence DOUBLE PRECISION NOT NULL DEFAULT 0,
    agreement DOUBLE PRECISION NOT NULL DEFAULT 1,
    reporters INTEGER NOT NULL DEFAULT 0,
    report_count INTEGER NOT NULL DEFAULT 0,
    published BOOLEAN NOT NULL DEFAULT FALSE,
    reviewed TEXT,
    value DOUBLE PRECISION,
    material TEXT,
    synthetic BOOLEAN NOT NULL DEFAULT FALSE,
    updated_at TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ NOT NULL
);
CREATE INDEX IF NOT EXISTS features_geog_idx ON features USING GIST (geog);
CREATE TABLE IF NOT EXISTS reports (
    id BIGSERIAL PRIMARY KEY,
    device TEXT NOT NULL,
    category TEXT NOT NULL,
    state TEXT NOT NULL,
    geog geography(Point, 4326) NOT NULL,
    accuracy_m DOUBLE PRECISION NOT NULL,
    source TEXT NOT NULL,
    model_confidence DOUBLE PRECISION NOT NULL,
    value DOUBLE PRECISION,
    material TEXT,
    photo TEXT,
    created_at TIMESTAMPTZ NOT NULL,
    received_at TIMESTAMPTZ NOT NULL,
    status TEXT NOT NULL,
    flags JSONB NOT NULL DEFAULT '[]',
    feature_id BIGINT REFERENCES features(id) ON DELETE SET NULL,
    synthetic BOOLEAN NOT NULL DEFAULT FALSE
);
CREATE INDEX IF NOT EXISTS reports_device_idx ON reports (device, received_at);
CREATE INDEX IF NOT EXISTS reports_feature_idx ON reports (feature_id);
CREATE INDEX IF NOT EXISTS reports_geog_idx ON reports USING GIST (geog);
CREATE TABLE IF NOT EXISTS export_decisions (
    candidate_id TEXT PRIMARY KEY,
    decision TEXT NOT NULL,
    incline_sign INTEGER,
    decided_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
"""

REPORT_COLS = "id, device, category, state, ST_X(geog::geometry) AS lon, ST_Y(geog::geometry) AS lat, accuracy_m, source, model_confidence, value, material, photo, created_at, received_at, status, flags, feature_id, synthetic"
FEATURE_COLS = "id, category, state, ST_X(geog::geometry) AS lon, ST_Y(geog::geometry) AS lat, confidence, agreement, reporters, report_count, published, reviewed, value, material, synthetic, updated_at, created_at"


class PostgisStore:
    kind = "postgis"

    def __init__(self, url: str) -> None:
        self.pool = ConnectionPool(url, min_size=1, max_size=6, kwargs={"row_factory": dict_row, "autocommit": True}, open=True)
        self.pool.wait(timeout=30)
        with self.pool.connection() as conn:
            # One lock so two API workers starting together do not race on CREATE EXTENSION.
            conn.execute("SELECT pg_advisory_lock(7109)")
            try:
                conn.execute(SCHEMA)
            finally:
                conn.execute("SELECT pg_advisory_unlock(7109)")

    def close(self) -> None:
        self.pool.close()

    def _all(self, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        with self.pool.connection() as conn:
            return conn.execute(sql, params).fetchall()  # type: ignore[return-value]

    def _one(self, sql: str, params: tuple[Any, ...] = ()) -> dict[str, Any] | None:
        rows = self._all(sql, params)
        return rows[0] if rows else None

    # Reports

    def add_report(self, report: Report) -> Report:
        row = self._one(
            f"""INSERT INTO reports (device, category, state, geog, accuracy_m, source, model_confidence, value, material, photo, created_at, received_at, status, flags, feature_id, synthetic)
                VALUES (%s, %s, %s, ST_SetSRID(ST_MakePoint(%s, %s), 4326)::geography, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING {REPORT_COLS}""",
            (
                report.device, report.category, report.state, report.lon, report.lat, report.accuracy_m, report.source, report.model_confidence,
                report.value, report.material, report.photo, report.created_at, report.received_at or report.created_at, report.status, json.dumps(report.flags),
                report.feature_id, report.synthetic,
            ),
        )
        assert row is not None
        return Report(**row)

    def update_report(self, report: Report) -> None:
        self._all(
            "UPDATE reports SET status = %s, flags = %s, feature_id = %s WHERE id = %s RETURNING id",
            (report.status, json.dumps(report.flags), report.feature_id, report.id),
        )

    def get_report(self, report_id: int) -> Report | None:
        row = self._one(f"SELECT {REPORT_COLS} FROM reports WHERE id = %s", (report_id,))
        return Report(**row) if row else None

    def reports_for_feature(self, feature_id: int) -> list[Report]:
        return [Report(**r) for r in self._all(f"SELECT {REPORT_COLS} FROM reports WHERE feature_id = %s ORDER BY id", (feature_id,))]

    def reports_by_device(self, device: str, since: datetime) -> list[Report]:
        return [Report(**r) for r in self._all(f"SELECT {REPORT_COLS} FROM reports WHERE device = %s AND received_at >= %s ORDER BY created_at", (device, since))]

    def list_reports(self, status: str | None = None, limit: int = 500) -> list[Report]:
        if status is None:
            rows = self._all(f"SELECT {REPORT_COLS} FROM reports ORDER BY id DESC LIMIT %s", (limit,))
        else:
            rows = self._all(f"SELECT {REPORT_COLS} FROM reports WHERE status = %s ORDER BY id DESC LIMIT %s", (status, limit))
        return [Report(**r) for r in rows]

    # Features

    def nearest_feature(self, category: str, lon: float, lat: float, radius_m: float) -> tuple[Feature, float] | None:
        row = self._one(
            f"""WITH p AS (SELECT ST_SetSRID(ST_MakePoint(%s, %s), 4326)::geography AS g)
                SELECT {FEATURE_COLS}, ST_Distance(geog, p.g) AS dist
                FROM features, p
                WHERE category = %s AND ST_DWithin(geog, p.g, %s)
                ORDER BY dist LIMIT 1""",
            (lon, lat, category, radius_m),
        )
        if row is None:
            return None
        dist = float(row.pop("dist"))
        return Feature(**row), dist

    def add_feature(self, feature: Feature) -> Feature:
        row = self._one(
            f"""INSERT INTO features (category, state, geog, confidence, agreement, reporters, report_count, published, reviewed, value, material, synthetic, updated_at, created_at)
                VALUES (%s, %s, ST_SetSRID(ST_MakePoint(%s, %s), 4326)::geography, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING {FEATURE_COLS}""",
            (
                feature.category, feature.state, feature.lon, feature.lat, feature.confidence, feature.agreement, feature.reporters, feature.report_count,
                feature.published, feature.reviewed, feature.value, feature.material, feature.synthetic, feature.updated_at, feature.created_at,
            ),
        )
        assert row is not None
        return Feature(**row)

    def update_feature(self, feature: Feature) -> None:
        self._all(
            """UPDATE features SET state = %s, geog = ST_SetSRID(ST_MakePoint(%s, %s), 4326)::geography, confidence = %s, agreement = %s, reporters = %s,
                      report_count = %s, published = %s, reviewed = %s, value = %s, material = %s, synthetic = %s, updated_at = %s
               WHERE id = %s RETURNING id""",
            (
                feature.state, feature.lon, feature.lat, feature.confidence, feature.agreement, feature.reporters, feature.report_count,
                feature.published, feature.reviewed, feature.value, feature.material, feature.synthetic, feature.updated_at, feature.id,
            ),
        )

    def get_feature(self, feature_id: int) -> Feature | None:
        row = self._one(f"SELECT {FEATURE_COLS} FROM features WHERE id = %s", (feature_id,))
        return Feature(**row) if row else None

    def list_features(self, published: bool | None = None) -> list[Feature]:
        if published is None:
            rows = self._all(f"SELECT {FEATURE_COLS} FROM features ORDER BY id")
        else:
            rows = self._all(f"SELECT {FEATURE_COLS} FROM features WHERE published = %s ORDER BY id", (published,))
        return [Feature(**r) for r in rows]

    # Export decisions

    def set_export_decision(self, candidate_id: str, decision: str, incline_sign: int | None) -> None:
        self._all(
            """INSERT INTO export_decisions (candidate_id, decision, incline_sign) VALUES (%s, %s, %s)
               ON CONFLICT (candidate_id) DO UPDATE SET decision = EXCLUDED.decision, incline_sign = EXCLUDED.incline_sign, decided_at = now()
               RETURNING candidate_id""",
            (candidate_id, decision, incline_sign),
        )

    def export_decisions(self) -> dict[str, tuple[str, int | None]]:
        return {r["candidate_id"]: (r["decision"], r["incline_sign"]) for r in self._all("SELECT candidate_id, decision, incline_sign FROM export_decisions")}

    def counts(self) -> dict[str, int]:
        row = self._one(
            """SELECT (SELECT count(*) FROM reports) AS reports, (SELECT count(*) FROM features) AS features,
                      (SELECT count(*) FROM features WHERE published) AS published, (SELECT count(*) FROM reports WHERE status = 'review') AS in_review"""
        )
        assert row is not None
        return {k: int(v) for k, v in row.items()}

    def clear(self) -> None:
        with self.pool.connection() as conn:
            conn.execute("TRUNCATE reports, features, export_decisions RESTART IDENTITY CASCADE")


def connect(url: str) -> PostgisStore:
    try:
        return PostgisStore(url)
    except psycopg.Error as error:  # pragma: no cover - only on a misconfigured database
        raise RuntimeError(f"could not open the PostGIS store: {error}") from error
