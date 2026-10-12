"""Settings, all overridable from the environment (prefix CURBCUT_)."""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CURBCUT_", env_file=".env", extra="ignore")

    # Empty means the in-memory store. A postgresql:// URL means PostGIS.
    database_url: str = ""
    pilot_file: Path = REPO_ROOT / "data" / "pilot" / "capitol-hill.json"
    upload_dir: Path = REPO_ROOT / "data" / "uploads"
    seed_demo: bool = True

    # Aggregation
    cluster_radius_m: float = 5.0
    publish_threshold: float = 0.5
    export_threshold: float = 0.6

    # Anti-abuse
    rate_limit_10min: int = 40
    rate_limit_day: int = 400
    max_accuracy_m: float = 25.0
    max_speed_mps: float = 12.0
    max_distance_from_path_m: float = 25.0
    device_salt: str = "curbcut-dev-salt"

    # Review. Empty token means the review endpoints are open, which is only sensible on localhost.
    reviewer_token: str = ""
    cors_origins: str = "http://localhost:4109,http://127.0.0.1:4109"
    max_photo_bytes: int = 1_500_000


def get_settings() -> Settings:
    return Settings()
