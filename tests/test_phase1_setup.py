"""Smoke tests for Phase 1: config loading and database wiring.

No network access and no TenderDetail login are required for these tests.
"""
from __future__ import annotations

from sqlalchemy import inspect

from app.config import load_business_capabilities, load_saved_queries
from app.database import get_engine
from app.models import Base


def test_saved_queries_load_and_include_expected_names():
    queries = load_saved_queries()
    names = {q.name for q in queries}
    assert "Digital Marketing" in names
    assert "Software Development" in names
    assert len(queries) == 5


def test_business_capabilities_load():
    capabilities = load_business_capabilities()
    assert "Digital marketing" in capabilities
    assert len(capabilities) >= 1


def test_database_tables_can_be_created(tmp_path, monkeypatch):
    db_file = tmp_path / "test_tenders.db"
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{db_file}")

    # get_settings() is lru_cached; for this smoke test we build the engine
    # directly against a throwaway sqlite file instead of the cached settings.
    from sqlalchemy import create_engine

    engine = create_engine(f"sqlite:///{db_file}")
    Base.metadata.create_all(engine)

    inspector = inspect(engine)
    tables = set(inspector.get_table_names())
    assert {
        "tenders",
        "tender_query_matches",
        "collection_runs",
        "download_records",
        "screening_results",
    }.issubset(tables)
