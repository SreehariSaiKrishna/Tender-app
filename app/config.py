"""Application configuration.

Loads non-secret configuration from environment variables (via .env) and
secret-free query/capability lists from the config/ directory. No password,
OTP, or session token is ever read or stored here.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent
CONFIG_DIR = BASE_DIR / "config"


class SavedQuery(BaseSettings):
    name: str
    enabled: bool = True


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(BASE_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # TenderDetail
    tenderdetail_base_url: str = "https://www.tenderdetail.com"
    tenderdetail_login_url: str = "https://www.tenderdetail.com/registeruser/dashboard"
    tenderdetail_username: str = ""
    # Optional: only used if you've chosen to auto-fill login (see README).
    # Stored in plaintext in your local .env - never logged, never written
    # to any file by the agent itself, never committed (gitignored).
    tenderdetail_password: str = ""
    tenderdetail_profile_dir: str = "./data/browser_profile"
    browser_headless: bool = False

    # Database
    database_url: str = "sqlite:///./data/database/tenders.db"

    # AI
    ai_provider: str = "openai"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    ai_relevance_threshold: int = 60

    # SMTP
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_username: str = ""
    smtp_password: str = ""
    smtp_use_tls: bool = True
    report_email_from: str = ""
    report_email_to: str = ""

    # App behaviour
    log_level: str = "INFO"
    download_dir: str = "./data/raw"
    processed_dir: str = "./data/processed"

    @property
    def base_dir(self) -> Path:
        return BASE_DIR

    def resolved_path(self, relative: str) -> Path:
        p = Path(relative)
        return p if p.is_absolute() else (BASE_DIR / p)


@lru_cache
def get_settings() -> Settings:
    return Settings()


def load_saved_queries() -> list[SavedQuery]:
    """Load the configured saved-query names from config/queries.json."""
    path = CONFIG_DIR / "queries.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return [SavedQuery(**q) for q in data.get("queries", [])]


def load_business_capabilities() -> list[str]:
    """Load the business capability list used for AI screening."""
    path = CONFIG_DIR / "capabilities.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    return list(data.get("capabilities", []))
