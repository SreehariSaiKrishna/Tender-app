"""GET /tenders/{id}/bid-document(/info) - a part-by-part download stays on
the file it started with, even when the pack is regenerated in between.
A small in-memory GridFS bucket stands in for MongoDB."""
from __future__ import annotations

import datetime as dt
import io

import mongomock
import pytest
from bson import ObjectId
from fastapi.testclient import TestClient

from app.api import main as api

TENDER_ID = "6ab1e8837754c947b03a5759"


class _GridOut(io.BytesIO):
    def __init__(self, doc, data):
        super().__init__(data)
        self._id, self.filename, self.length = doc["_id"], doc["filename"], doc["length"]


class FakeBucket:
    """upload_from_stream / delete / open_download_stream over a mongomock
    `files` collection, the file bytes kept alongside."""

    def __init__(self):
        self.files = mongomock.MongoClient().db["generated_bids.files"]
        self.data = {}
        self._clock = dt.datetime(2026, 1, 1)

    def upload_from_stream(self, filename, data, metadata=None):
        self._clock += dt.timedelta(seconds=1)
        _id = ObjectId()
        self.files.insert_one({"_id": _id, "filename": filename, "length": len(data), "metadata": metadata or {},
                               "uploadDate": self._clock})
        self.data[_id] = data
        return _id

    def delete(self, _id):
        self.files.delete_one({"_id": _id})
        self.data.pop(_id, None)

    def open_download_stream(self, _id):
        return _GridOut(self.files.find_one({"_id": _id}), self.data[_id])


@pytest.fixture()
def bucket(monkeypatch):
    bids = FakeBucket()
    monkeypatch.setattr(api, "get_generated_bids_bucket", lambda: bids)
    monkeypatch.setattr(api, "get_generated_bids_files_collection", lambda: bids.files)
    monkeypatch.setattr(api, "BID_PART_SIZE", 4)
    return bids


def _upload(bucket, data: bytes):
    return bucket.upload_from_stream("bid-pack.pdf", data, metadata={"tender_id": TENDER_ID})


def test_parts_come_from_the_file_the_download_started_on(bucket):
    client = TestClient(api.app)
    old = _upload(bucket, b"OLD-PACK-1234")
    info = client.get(f"/tenders/{TENDER_ID}/bid-document/info").json()
    assert info["file_id"] == str(old) and info["parts"] == 4

    first = client.get(f"/tenders/{TENDER_ID}/bid-document", params={"part": 0, "file": info["file_id"]})
    assert first.content == b"OLD-"

    # Regenerated mid-download: the new pack is stored, then the old one deleted.
    new = _upload(bucket, b"NEW-PACK-5678")
    bucket.delete(old)
    gone = client.get(f"/tenders/{TENDER_ID}/bid-document", params={"part": 1, "file": info["file_id"]})
    assert gone.status_code == 409  # never a mix of old and new parts

    restart = client.get(f"/tenders/{TENDER_ID}/bid-document/info").json()
    assert restart["file_id"] == str(new)
    data = b"".join(
        client.get(f"/tenders/{TENDER_ID}/bid-document", params={"part": i, "file": restart["file_id"]}).content
        for i in range(restart["parts"])
    )
    assert data == b"NEW-PACK-5678"


def test_without_a_file_id_the_newest_pack_is_served(bucket):
    client = TestClient(api.app)
    _upload(bucket, b"older")
    _upload(bucket, b"newer")
    assert client.get(f"/tenders/{TENDER_ID}/bid-document").content == b"newer"
