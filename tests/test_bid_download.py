"""GET /tenders/{id}/bid-document(/link) - bid packs aren't kept: one is
offered for download only for a while after it's generated (the bucket's
bid-packs/ rule deletes it after a day). A temporary folder stands in for
the bucket (see tests/conftest.py), so the link's `url` is null here and
the pack comes from /bid-document itself."""
from __future__ import annotations

import datetime as dt

import mongomock
import pytest
from bson import ObjectId
from fastapi.testclient import TestClient

from app import storage
from app.api import main as api

TENDER_ID = "6ab1e8837754c947b03a5759"
PACK_KEY = f"bid-packs/{TENDER_ID}/abc.pdf"


@pytest.fixture()
def tenders(monkeypatch):
    collection = mongomock.MongoClient().db["tenders"]
    monkeypatch.setattr(api, "get_collection", lambda: collection)
    return collection


def _tender(tenders, generated_hours_ago: float | None, **extra):
    doc = {"_id": ObjectId(TENDER_ID), "tender_ref": "57708084", **extra}
    if generated_hours_ago is not None:
        storage.put_bytes(PACK_KEY, b"%PDF-pack", "application/pdf")
        doc["bid_pack_key"] = PACK_KEY
        doc["bid_generated_at"] = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=generated_hours_ago)
    tenders.insert_one(doc)
    return doc


def test_a_fresh_pack_can_be_downloaded(tenders):
    _tender(tenders, generated_hours_ago=1)
    client = TestClient(api.app)
    link = client.get(f"/tenders/{TENDER_ID}/bid-document/link").json()
    assert link == {"url": None, "filename": "bid-pack-57708084.pdf"}  # no bucket locally
    res = client.get(f"/tenders/{TENDER_ID}/bid-document")
    assert res.content == b"%PDF-pack"
    assert "bid-pack-57708084.pdf" in res.headers["content-disposition"]


def test_an_expired_pack_is_not_offered(tenders):
    _tender(tenders, generated_hours_ago=21)
    client = TestClient(api.app)
    assert client.get(f"/tenders/{TENDER_ID}/bid-document/link").status_code == 404
    assert client.get(f"/tenders/{TENDER_ID}").json()["bid_pack_available"] is False


def test_no_pack_until_one_is_generated(tenders):
    _tender(tenders, generated_hours_ago=None)
    client = TestClient(api.app)
    assert client.get(f"/tenders/{TENDER_ID}/bid-document/link").status_code == 404
    tender = client.get(f"/tenders/{TENDER_ID}").json()
    assert tender["bid_pack_available"] is False and "bid_pack_key" not in tender


def test_the_tender_row_says_when_a_pack_is_available(tenders):
    _tender(tenders, generated_hours_ago=2)
    tender = TestClient(api.app).get(f"/tenders/{TENDER_ID}").json()
    assert tender["bid_pack_available"] is True and "bid_pack_key" not in tender
