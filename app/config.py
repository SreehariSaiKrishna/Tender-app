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
    mongodb_uri: str = ""
    mongodb_db_name: str = "tender_intelligence"

    # AI
    ai_provider: str = "openai"
    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    # The submission checklist (app.intelligence.bid_drafter) has to go
    # through every eligibility condition in the tender and find the proof
    # for each one. gpt-4o-mini skipped most of them, and gpt-4.1-mini was
    # too slow for the API's 30s limit. gpt-4.1 covered them all in ~18s,
    # at about $0.04 per checklist.
    checklist_model: str = "gpt-4.1"
    # The bid pack's drafted letters (app.intelligence.bid_drafter) must fill
    # every field from the company data and tailor it to the tender -
    # gpt-4o-mini wrote thin, generic letters that skipped most of the data.
    # Batches run in parallel, so gpt-4.1 still fits the API's 30s limit.
    draft_model: str = "gpt-4.1"
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
    documents_dir: str = "./data/documents"
    documents_batch_limit: int = 15
    # Bytes of library PDFs merged into one bid pack (see
    # app.reports.bid_generator.plan_rows) - anything past it gets a
    # placeholder page to attach separately. The dashboard downloads packs
    # in 3 MB parts (app.api.main.BID_PART_SIZE), so this isn't bounded by
    # the Lambda response cap.
    enclosure_byte_budget: int = 30 * 1024 * 1024
    # Set on the API Lambda only (see template.yaml): a checklist waiting on
    # a tender's documents asks this function to fetch them right away (see
    # app.api.main._start_document_fetch). Empty when running locally.
    pipeline_function_name: str = ""

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


def load_company_profile() -> dict:
    """Load the verified company/legal profile (config/company_profile.json)
    used by app.reports.bid_generator to draft a bid pack's covering letter
    and declarations - see that file's _comment for what "verified" means
    here."""
    path = CONFIG_DIR / "company_profile.json"
    return json.loads(path.read_text(encoding="utf-8"))
