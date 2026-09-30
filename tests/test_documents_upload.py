"""Documents library files up to MAX_UPLOAD_SIZE: uploaded (or replaced) in
parts via PUT /uploads/{id}/parts/N + POST /documents/uploads/{id}/finish,
and downloaded/viewed in parts via /documents/{id}/info + ?part=N. A
mongomock collection holds the records; the bytes go to a temporary folder
standing in for the S3 bucket (see tests/conftest.py)."""
from __future__ import annotations

import io

import mongomock
import pytest
from fastapi.testclient import TestClient

from app import storage
from app.api import main as api

UPLOAD_ID = "b" * 32


@pytest.fixture()
def client(monkeypatch):
    records = mongomock.MongoClient().db["company_documents"]
    monkeypatch.setattr(api, "get_company_documents_collection", lambda: records)
    transfers = mongomock.MongoClient().db["stamp_transfers"]
    monkeypatch.setattr(api, "get_stamp_transfers_collection", lambda: transfers)
    monkeypatch.setattr(api, "TRANSFER_PART_SIZE", 1000)
    monkeypatch.setattr(api, "MAX_DOCUMENT_SIZE", 1500)  # the one-request endpoints' cap
    c = TestClient(api.app)
    c.records, c.transfers = records, transfers
    return c


def _send(client, data: bytes) -> int:
    chunks = [data[i:i + api.TRANSFER_PART_SIZE] for i in range(0, len(data), api.TRANSFER_PART_SIZE)]
    for i, chunk in enumerate(chunks):
        assert client.put(f"/uploads/{UPLOAD_ID}/parts/{i}", content=chunk).status_code == 200
    return len(chunks)


def _fetch(client, document_id: str, kind: str = "download") -> bytes:
    info = client.get(f"/documents/{document_id}/info").json()
    return b"".join(client.get(f"/documents/{document_id}/{kind}", params={"part": i}).content
                    for i in range(info["parts"]))


def test_a_document_bigger_than_one_request_round_trips_in_parts(client):
    data = bytes(range(256)) * 12  # 3072 bytes: over the one-request cap, 4 parts
    assert client.post("/documents", files={"file": ("big.pdf", data, "application/pdf")}).status_code == 413

    parts = _send(client, data)
    res = client.post(f"/documents/uploads/{UPLOAD_ID}/finish",
                      data={"parts": parts, "name": "SIV_Letterhead", "filename": "SIV_Letterhead.pdf",
                            "content_type": "application/pdf"})
    assert res.status_code == 200
    doc = res.json()
    assert (doc["name"], doc["filename"], doc["size"]) == ("SIV_Letterhead", "SIV_Letterhead.pdf", len(data))
    assert client.transfers.count_documents({}) == 0

    info = client.get(f"/documents/{doc['id']}/info").json()
    assert info["parts"] == 4 and info["content_type"] == "application/pdf"
    assert _fetch(client, doc["id"]) == data
    assert _fetch(client, doc["id"], "view") == data
    assert client.get(f"/documents/{doc['id']}/download", params={"part": 4}).status_code == 416


def test_replacing_in_parts_keeps_the_name(client):
    original = client.post("/documents", files={"file": ("old.png", b"old", "image/png")},
                           data={"name": "SIV_Stamp"}).json()
    data = b"n" * 2500
    parts = _send(client, data)
    res = client.post(f"/documents/uploads/{UPLOAD_ID}/finish",
                      data={"parts": parts, "filename": "new.png", "content_type": "image/png", "replace": original["id"]})
    assert res.status_code == 200
    replaced = res.json()
    assert replaced["name"] == "SIV_Stamp" and replaced["filename"] == "new.png"
    # Same id, so checklist rows and Sign & Stamp choices that point at it still work.
    assert replaced["id"] == original["id"]
    assert [d["id"] for d in client.get("/documents").json()["documents"]] == [original["id"]]
    assert _fetch(client, replaced["id"]) == data
    # The old file is gone from storage; only the new one is kept.
    record = client.records.find_one({})
    assert record["s3_key"].endswith("/new.png")
    assert storage.get_bytes(record["s3_key"].replace("new.png", "old.png")) is None


def test_deleting_a_document_removes_its_file(client):
    doc = client.post("/documents", files={"file": ("a.txt", b"hello", "text/plain")}).json()
    key = client.records.find_one({})["s3_key"]
    assert storage.get_bytes(key) == b"hello"
    assert client.delete(f"/documents/{doc['id']}").status_code == 200
    assert storage.get_bytes(key) is None and client.records.count_documents({}) == 0
    assert client.delete(f"/documents/{doc['id']}").status_code == 404


def test_whole_file_download_still_works_for_small_files(client):
    doc = client.post("/documents", files={"file": ("a.txt", b"hello", "text/plain")}).json()
    res = client.get(f"/documents/{doc['id']}/download")
    assert res.content == b"hello" and res.headers["content-type"].startswith("text/plain")


def _png() -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGBA", (40, 20), (0, 0, 200, 255)).save(buf, "PNG")
    return buf.getvalue()


def _mark_labels(client) -> dict[str, list[str]]:
    return {kind: [m["label"] for m in options] for kind, options in client.get("/stamp-document/marks").json().items()}


def test_a_document_added_as_a_mark_is_offered_as_one_whatever_its_name(client):
    parts = _send(client, _png())
    res = client.post(f"/documents/uploads/{UPLOAD_ID}/finish",
                      data={"parts": parts, "name": "Director Ravi", "filename": "ravi.png",
                            "content_type": "image/png", "mark_kind": "signature"})
    assert res.status_code == 200 and res.json()["mark_kind"] == "signature"
    doc_id = res.json()["id"]
    # In the Documents list like any other document, and in the signature dropdown.
    assert [d["name"] for d in client.get("/documents").json()["documents"]] == ["Director Ravi"]
    assert _mark_labels(client)["signature"] == ["Director Ravi"]

    # Replacing its file keeps it a signature; renaming can re-file it.
    parts = _send(client, _png())
    replaced = client.post(f"/documents/uploads/{UPLOAD_ID}/finish",
                           data={"parts": parts, "filename": "ravi2.png", "content_type": "image/png",
                                 "replace": doc_id}).json()
    assert replaced["mark_kind"] == "signature"
    renamed = client.put(f"/documents/{replaced['id']}", json={"name": "Company Seal 2", "mark_kind": "stamp"}).json()
    assert renamed["mark_kind"] == "stamp"
    labels = _mark_labels(client)
    assert "Company Seal 2" in labels["stamp"] and "Company Seal 2" not in labels["signature"]


def test_removing_from_sign_and_stamp_keeps_the_document(client):
    stamp = client.post("/documents", files={"file": ("SIV Stamp.png", _png(), "image/png")},
                        data={"name": "SIV_Stamp"}).json()
    assert _mark_labels(client)["stamp"] == ["SIV_Stamp"]  # offered by its name

    assert client.delete(f"/stamp-document/marks/{stamp['id']}").status_code == 200
    assert _mark_labels(client)["stamp"] == []  # gone from Sign & Stamp, despite "Stamp" in the name...
    assert [d["name"] for d in client.get("/documents").json()["documents"]] == ["SIV_Stamp"]  # ...but kept
    assert client.delete(f"/stamp-document/marks/{'0' * 24}").status_code == 404


def test_an_unknown_mark_kind_is_refused(client):
    parts = _send(client, _png())
    res = client.post(f"/documents/uploads/{UPLOAD_ID}/finish",
                      data={"parts": parts, "filename": "x.png", "mark_kind": "logo"})
    assert res.status_code == 422


def test_uploads_over_10_mb_are_refused(client, monkeypatch):
    monkeypatch.setattr(api, "MAX_UPLOAD_SIZE", 2000)
    parts = _send(client, b"x" * 2000)
    assert client.put(f"/uploads/{UPLOAD_ID}/parts/{parts}", content=b"x").status_code == 413
