"""Typed shapes that cross the API boundary or the store boundary."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, model_validator

Category = Literal["curb", "surface", "slope"]
Source = Literal["photo", "sensor", "manual"]
ReportStatus = Literal["accepted", "review", "rejected"]
Profile = Literal["manual_wheelchair", "powered_wheelchair", "stroller", "walking"]

STATES: dict[str, tuple[str, ...]] = {
    "curb": ("present", "missing", "damaged", "blocked"),
    "surface": ("smooth", "rough", "bump"),
    "slope": ("steep", "gentle"),
}
# States that are evidence about a feature but never start one on their own.
COUNTER_STATES = {("surface", "smooth"), ("slope", "gentle")}
MATERIALS = ("asphalt", "concrete", "paving_stones", "sett", "cobblestone", "gravel", "fine_gravel", "compacted", "dirt", "grass", "sand", "wood", "metal")


class ReportIn(BaseModel):
    """One observation from one device."""

    category: Category
    state: str
    lon: float = Field(ge=-180, le=180)
    lat: float = Field(ge=-90, le=90)
    accuracy_m: float = Field(default=10.0, ge=0, le=10_000)
    source: Source
    model_confidence: float = Field(default=0.9, ge=0, le=1)
    value: float | None = Field(default=None, description="Incline in percent for slope reports, peak vertical acceleration for bumps.")
    material: str | None = None
    recorded_at: datetime | None = Field(default=None, description="When the phone observed it. Clamped to the last 24 hours. Rate limits use the time of receipt instead.")
    faces_blurred: int = Field(default=0, ge=0)
    regions_blurred: int = Field(default=0, ge=0)

    @model_validator(mode="after")
    def check_state(self) -> "ReportIn":
        if self.state not in STATES[self.category]:
            raise ValueError(f"state must be one of {STATES[self.category]} for category {self.category}")
        if self.material is not None and self.material not in MATERIALS:
            raise ValueError(f"material must be one of {MATERIALS}")
        if self.category == "slope" and self.state == "steep" and self.value is None:
            raise ValueError("a steep slope report needs the measured incline in percent as value")
        return self


class Report(BaseModel):
    id: int
    device: str
    category: Category
    state: str
    lon: float
    lat: float
    accuracy_m: float
    source: Source
    model_confidence: float
    value: float | None = None
    material: str | None = None
    photo: str | None = None
    created_at: datetime  # when it was observed: drives recency and the speed check
    received_at: datetime | None = None  # when the server got it: drives rate limits
    status: ReportStatus = "accepted"
    flags: list[str] = Field(default_factory=list)
    feature_id: int | None = None
    synthetic: bool = False


class Feature(BaseModel):
    id: int
    category: Category
    state: str
    lon: float
    lat: float
    confidence: float = 0.0
    agreement: float = 1.0
    reporters: int = 0
    report_count: int = 0
    published: bool = False
    reviewed: Literal["approved", "rejected"] | None = None
    value: float | None = None
    material: str | None = None
    synthetic: bool = False
    updated_at: datetime
    created_at: datetime


class SubmitResult(BaseModel):
    report: Report
    feature: Feature | None = None
    merged: bool = False
    message: str


class RouteRequest(BaseModel):
    origin: tuple[float, float] = Field(description="lon, lat")
    destination: tuple[float, float] = Field(description="lon, lat")
    profile: Profile = "manual_wheelchair"


class Barrier(BaseModel):
    edge_id: str
    reason: str
    lon: float
    lat: float


class RouteLeg(BaseModel):
    distance_m: float
    geometry: dict
    barriers: list[Barrier] = Field(default_factory=list)
    max_incline: float | None = None
    cautions: list[str] = Field(default_factory=list)
    edge_ids: list[str] = Field(default_factory=list)


class RouteResponse(BaseModel):
    status: Literal["ok", "no_accessible_route", "unreachable"]
    profile: Profile
    route: RouteLeg | None = None
    shortest: RouteLeg | None = Field(default=None, description="The unconstrained shortest walk, with the barriers it crosses for this profile.")
    detour_m: float | None = None
    message: str


class ReviewDecision(BaseModel):
    action: Literal["accept", "reject", "approve", "clear"]


class ExportDecision(BaseModel):
    decision: Literal["approved", "skipped", "pending"]
    incline_sign: Literal[1, -1] | None = None
