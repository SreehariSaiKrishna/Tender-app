"""app.processing.cleanup - closed tenders are deleted once 7 days past
their closing date (or past disappearing, when they have none), applied or
not, together with their document text in file storage."""
from __future__ import annotations

import datetime as dt

import mongomock
import pytest

from app import storage
from app.processing.cleanup import cleanup_stale_closed_tenders

NOW = dt.datetime.now(dt.timezone.utc)
OLD = NOW - dt.timedelta(days=8)
RECENT = NOW - dt.timedelta(days=2)


@pytest.fixture()
def collection():
    return mongomock.MongoClient()["test"]["tenders"]


def _tender(collection, ref, **fields):
    doc = {"dedup_key": f"ref:{ref}", "disappeared": True, "closing_date": OLD, **fields}
    collection.insert_one(doc)
    return doc


def _remaining(collection):
    return sorted(d["dedup_key"] for d in collection.find())


def test_closed_tenders_past_the_grace_period_are_deleted_even_if_applied(collection):
    _tender(collection, "old")
    _tender(collection, "old-applied", applied=True)
    _tender(collection, "recent", closing_date=RECENT)
    _tender(collection, "live", disappeared=False)

    summary = cleanup_stale_closed_tenders(collection)

    assert summary.deleted == 2
    assert _remaining(collection) == ["ref:live", "ref:recent"]


def test_a_closed_tender_without_a_closing_date_counts_from_when_it_disappeared(collection):
    _tender(collection, "gone-long-ago", closing_date=None, disappeared_at=OLD)
    _tender(collection, "gone-lately", closing_date=None, disappeared_at=RECENT)
    _tender(collection, "never-dated", closing_date=None)  # closed before disappeared_at existed

    cleanup_stale_closed_tenders(collection)

    assert _remaining(collection) == ["ref:gone-lately", "ref:never-dated"]


def test_the_deleted_tenders_document_text_is_removed_from_storage(collection):
    storage.put_text("tender-text/a.txt", "RFP text")
    storage.put_text("tender-text/b.txt", "kept")
    _tender(collection, "a", document_text_key="tender-text/a.txt")
    _tender(collection, "b", closing_date=RECENT, document_text_key="tender-text/b.txt")

    cleanup_stale_closed_tenders(collection)

    assert storage.get_text("tender-text/a.txt") is None
    assert storage.get_text("tender-text/b.txt") == "kept"
