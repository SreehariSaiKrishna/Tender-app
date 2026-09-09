"""Smoke tests for Phase 1: config loading and database wiring.

Uses mongomock (an in-memory fake implementing the pymongo API) - no real
MongoDB, network access, or TenderDetail login required for these tests.
"""
from __future__ import annotations

import mongomock

from app import database
from app.config import load_business_capabilities, load_saved_queries


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


def test_ensure_indexes_creates_the_expected_indexes(monkeypatch):
    client = mongomock.MongoClient()
    collection = client["test_tender_intelligence"]["tenders"]
    monkeypatch.setattr(database, "get_collection", lambda: collection)

    database.ensure_indexes()

    index_names = set(collection.index_information().keys())
    assert {
        "dedup_key_1",
        "tender_ref_1",
        "closing_date_1",
        "disappeared_1",
        "first_seen_1",
        "deadline_changed_1",
        "query_match_count_1",
        "query_matches.query_name_1",
    }.issubset(index_names)
