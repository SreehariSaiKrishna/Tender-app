"""Tests for the document-download step (app.browser.document_collector).

The browser-driving parts (open_tender_detail/list_document_rows/
fetch_document_bytes) are monkeypatched with fakes rather than exercised
against a real page - Playwright itself is covered by manual verification
against the live site (see app.browser.tender_documents' module docstring),
not by these unit tests. What's tested here is the surrounding logic: which
tenders are picked up, how per-tender outcomes are built, and what gets
written back to Mongo.
"""
from __future__ import annotations

import datetime as dt
from pathlib import Path

import mongomock
import pytest

from app.browser import document_collector
from app.browser.document_collector import (
    DocumentDownloadSummary,
    _download_one_tender,
    _extract_and_store_key_dates,
    _needs_download,
    _pending_tenders,
    _slugify,
    _tenders_missing_key_dates,
    run_document_collection,
    run_key_dates_backfill,
)
from app.browser.tender_documents import NoDocumentsFoundError
from app.config import Settings


@pytest.fixture()
def collection():
    client = mongomock.MongoClient()
    return client["test_tender_intelligence"]["tenders"]


def make_tender(collection, dedup_key="ref:1", **overrides) -> dict:
    now = dt.datetime.now(dt.timezone.utc)
    doc = {
        "dedup_key": dedup_key,
        "tender_ref": dedup_key.split(":")[-1],
        "title": "Tender for Digital Signage",
        "domain_match": True,
        "eligibility_match": True,
        "disappeared": False,
        "source_url": "https://www.tenderdetail.com/registeruser/indiatenders/1/1",
        "first_seen": now,
        "last_seen": now,
    }
    doc.update(overrides)
    result = collection.insert_one(doc)
    doc["_id"] = result.inserted_id
    return doc


# --- _slugify -----------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [
        ("TDR-12345", "TDR_12345"),
        ("  spaces  ", "spaces"),
        ("", "tender"),
        ("a/b\\c:d", "a_b_c_d"),
    ],
)
def test_slugify(value, expected):
    assert _slugify(value) == expected


# --- _needs_download -----------------------------------------------------


def test_needs_download_true_when_never_downloaded():
    assert _needs_download({}) is True


def test_needs_download_false_when_downloaded_and_unchanged():
    now = dt.datetime.now(dt.timezone.utc)
    doc = {"documents_downloaded_at": now, "content_last_changed": now - dt.timedelta(days=1)}
    assert _needs_download(doc) is False


def test_needs_download_true_when_content_changed_since_download():
    now = dt.datetime.now(dt.timezone.utc)
    doc = {"documents_downloaded_at": now - dt.timedelta(days=1), "content_last_changed": now}
    assert _needs_download(doc) is True


# --- _pending_tenders -----------------------------------------------------


def test_pending_tenders_only_returns_eligible_with_source_url(collection):
    make_tender(collection, dedup_key="ref:1")  # live + source_url -> pending
    make_tender(collection, dedup_key="ref:2", disappeared=True)  # no longer live
    make_tender(collection, dedup_key="ref:3", source_url=None)  # no source_url

    pending = _pending_tenders(collection, limit=None)

    assert [t["dedup_key"] for t in pending] == ["ref:1"]


def test_pending_tenders_excludes_already_downloaded_and_unchanged(collection):
    now = dt.datetime.now(dt.timezone.utc)
    make_tender(collection, dedup_key="ref:1", documents_downloaded_at=now, content_last_changed=now - dt.timedelta(days=1))
    make_tender(collection, dedup_key="ref:2")  # never downloaded

    pending = _pending_tenders(collection, limit=None)

    assert [t["dedup_key"] for t in pending] == ["ref:2"]


def test_pending_tenders_respects_limit(collection):
    for i in range(5):
        make_tender(collection, dedup_key=f"ref:{i}")

    pending = _pending_tenders(collection, limit=2)

    assert len(pending) == 2


# --- _download_one_tender -----------------------------------------------------


def test_download_one_tender_success(monkeypatch, collection, tmp_path):
    tender = make_tender(collection)

    monkeypatch.setattr(document_collector, "open_tender_detail", lambda page, url: None)
    monkeypatch.setattr(
        document_collector,
        "list_document_rows",
        lambda page: [
            {"filename": "notice.html", "description": "Tender Documents", "url": "https://x/notice.html"},
            {"filename": "boq.xls", "description": "BOQ", "url": "https://x/boq.xls"},
        ],
    )
    monkeypatch.setattr(
        document_collector,
        "fetch_document_bytes",
        lambda page, url: f"content of {url}".encode(),
    )

    outcome = _download_one_tender(page=None, tender=tender, documents_dir=tmp_path, collection=collection)

    assert outcome.status == "success"
    assert outcome.downloaded == ["notice.html", "boq.xls"]

    stored = collection.find_one({"_id": tender["_id"]})
    assert len(stored["documents"]) == 2
    assert stored["documents_downloaded_at"] is not None
    saved_path = Path(stored["documents"][0]["local_path"])
    assert saved_path.read_bytes() == b"content of https://x/notice.html"


def test_download_one_tender_skipped_when_no_documents_found(monkeypatch, collection, tmp_path):
    tender = make_tender(collection)

    monkeypatch.setattr(document_collector, "open_tender_detail", lambda page, url: None)

    def raise_no_docs(page):
        raise NoDocumentsFoundError("nothing here")

    monkeypatch.setattr(document_collector, "list_document_rows", raise_no_docs)

    outcome = _download_one_tender(page=None, tender=tender, documents_dir=tmp_path, collection=collection)

    assert outcome.status == "skipped"
    stored = collection.find_one({"_id": tender["_id"]})
    assert stored.get("documents_downloaded_at") is None


def test_download_one_tender_failed_when_detail_page_does_not_open(monkeypatch, collection, tmp_path):
    tender = make_tender(collection)

    def raise_error(page, url):
        raise RuntimeError("navigation timeout")

    monkeypatch.setattr(document_collector, "open_tender_detail", raise_error)

    outcome = _download_one_tender(page=None, tender=tender, documents_dir=tmp_path, collection=collection)

    assert outcome.status == "failed"
    assert "navigation timeout" in outcome.error_message


def test_download_one_tender_partial_when_one_file_fails(monkeypatch, collection, tmp_path):
    tender = make_tender(collection)
    calls = {"n": 0}

    monkeypatch.setattr(document_collector, "open_tender_detail", lambda page, url: None)
    monkeypatch.setattr(
        document_collector,
        "list_document_rows",
        lambda page: [
            {"filename": "notice.html", "description": "", "url": "https://x/notice.html"},
            {"filename": "boq.xls", "description": "", "url": "https://x/boq.xls"},
        ],
    )

    def flaky_fetch(page, url):
        calls["n"] += 1
        if calls["n"] == 1:
            return b"notice content"
        raise RuntimeError("download timed out")

    monkeypatch.setattr(document_collector, "fetch_document_bytes", flaky_fetch)

    outcome = _download_one_tender(page=None, tender=tender, documents_dir=tmp_path, collection=collection)

    assert outcome.status == "partial"
    assert outcome.downloaded == ["notice.html"]
    assert "download timed out" in outcome.error_message

    stored = collection.find_one({"_id": tender["_id"]})
    assert len(stored["documents"]) == 1  # the one that succeeded is still recorded


def test_download_one_tender_records_error_for_row_with_no_href(monkeypatch, collection, tmp_path):
    tender = make_tender(collection)

    monkeypatch.setattr(document_collector, "open_tender_detail", lambda page, url: None)
    monkeypatch.setattr(
        document_collector,
        "list_document_rows",
        lambda page: [{"filename": "notice.html", "description": "", "url": None}],
    )

    outcome = _download_one_tender(page=None, tender=tender, documents_dir=tmp_path, collection=collection)

    assert outcome.status == "failed"
    assert "no href" in outcome.error_message


# --- _extract_and_store_key_dates -----------------------------------------------------


def test_extract_and_store_key_dates_sets_opening_date_when_parseable(monkeypatch, collection):
    tender = make_tender(collection)
    monkeypatch.setattr(
        document_collector,
        "extract_key_dates",
        lambda page: {
            "Publish Date": "03-09-2026",
            "Last Date of Bid Submission": "14-09-2026",
            "Tender Opening Date": "15-09-2026",
        },
    )

    _extract_and_store_key_dates(page=None, tender=tender, collection=collection)

    stored = collection.find_one({"_id": tender["_id"]})
    assert stored["page_key_dates"]["Tender Opening Date"] == "15-09-2026"
    assert stored["opening_date"] == dt.datetime(2026, 9, 15)  # mongomock strips tzinfo on read-back, like real MongoDB


def test_extract_and_store_key_dates_skips_when_section_absent(monkeypatch, collection):
    tender = make_tender(collection)
    monkeypatch.setattr(document_collector, "extract_key_dates", lambda page: {})

    _extract_and_store_key_dates(page=None, tender=tender, collection=collection)

    stored = collection.find_one({"_id": tender["_id"]})
    assert "opening_date" not in stored
    assert "page_key_dates" not in stored


def test_extract_and_store_key_dates_skips_opening_date_when_unparseable(monkeypatch, collection):
    tender = make_tender(collection)
    monkeypatch.setattr(
        document_collector,
        "extract_key_dates",
        lambda page: {"Tender Opening Date": "To be announced"},
    )

    _extract_and_store_key_dates(page=None, tender=tender, collection=collection)

    stored = collection.find_one({"_id": tender["_id"]})
    assert stored["page_key_dates"]["Tender Opening Date"] == "To be announced"
    assert "opening_date" not in stored


def test_extract_and_store_key_dates_never_raises(monkeypatch, collection):
    tender = make_tender(collection)

    def raise_error(page):
        raise RuntimeError("page crashed")

    monkeypatch.setattr(document_collector, "extract_key_dates", raise_error)

    _extract_and_store_key_dates(page=None, tender=tender, collection=collection)  # must not raise

    stored = collection.find_one({"_id": tender["_id"]})
    assert "opening_date" not in stored


def test_download_one_tender_populates_opening_date_from_page_key_dates(monkeypatch, collection, tmp_path):
    """Integration-level: the full _download_one_tender flow picks up the
    Key Dates scrape too, not just the document table."""
    tender = make_tender(collection)

    monkeypatch.setattr(document_collector, "open_tender_detail", lambda page, url: None)
    monkeypatch.setattr(
        document_collector, "extract_key_dates", lambda page: {"Tender Opening Date": "15-09-2026"}
    )
    monkeypatch.setattr(
        document_collector,
        "list_document_rows",
        lambda page: [{"filename": "notice.html", "description": "", "url": "https://x/notice.html"}],
    )
    monkeypatch.setattr(document_collector, "fetch_document_bytes", lambda page, url: b"content")

    _download_one_tender(page=None, tender=tender, documents_dir=tmp_path, collection=collection)

    stored = collection.find_one({"_id": tender["_id"]})
    assert stored["opening_date"] == dt.datetime(2026, 9, 15)  # mongomock strips tzinfo on read-back, like real MongoDB


# --- run_document_collection -----------------------------------------------------


def test_run_document_collection_short_circuits_when_nothing_pending(monkeypatch, collection):
    """No browser should ever launch when there's nothing to download -
    this only holds if get_collection() is monkeypatched to an empty
    mongomock collection and the run still returns cleanly.
    """
    monkeypatch.setattr(document_collector, "get_collection", lambda: collection)
    settings = Settings(mongodb_uri="mongodb://localhost/test")

    summary = run_document_collection(settings=settings, limit=5)

    assert isinstance(summary, DocumentDownloadSummary)
    assert summary.checked == 0
    assert summary.outcomes == []


# --- _tenders_missing_key_dates / run_key_dates_backfill -----------------------------------------------------


def test_tenders_missing_key_dates_only_returns_eligible_without_page_key_dates(collection):
    make_tender(collection, dedup_key="ref:1")  # live, no page_key_dates yet -> pending
    make_tender(collection, dedup_key="ref:2", disappeared=True)  # no longer live
    make_tender(collection, dedup_key="ref:3", page_key_dates={"Publish Date": "01-01-2026"})  # already has it

    pending = _tenders_missing_key_dates(collection, limit=None)

    assert [t["dedup_key"] for t in pending] == ["ref:1"]


def test_run_key_dates_backfill_short_circuits_when_nothing_pending(monkeypatch, collection):
    """No browser should ever launch when every eligible tender already
    has page_key_dates."""
    monkeypatch.setattr(document_collector, "get_collection", lambda: collection)
    settings = Settings(mongodb_uri="mongodb://localhost/test")

    summary = run_key_dates_backfill(settings=settings, limit=5)

    assert summary.checked == 0
    assert summary.found == 0
    assert summary.failed == 0
