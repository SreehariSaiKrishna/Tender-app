"""A hand-added tender's own documents (Other Tenders): files uploaded and
links added from the dashboard, then put on disk for the summarizer by
app.processing.manual_documents - and the checklist asking for them."""
from __future__ import annotations

from email.message import Message

import mongomock
import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app import lambda_handler, storage
from app.api import main as api
from app.config import Settings
from app.processing import eligibility, manual_documents


@pytest.fixture()
def tenders(monkeypatch):
    collection = mongomock.MongoClient().db["tenders"]
    transfers = mongomock.MongoClient().db["transfers"]
    monkeypatch.setattr(api, "get_collection", lambda: collection)
    monkeypatch.setattr(api, "get_stamp_transfers_collection", lambda: transfers)
    monkeypatch.setattr(eligibility, "load_match_keywords", lambda: [])
    return collection


def _add_tender(client, **extra) -> str:
    return client.post("/tenders/manual", json={"title": "Hand-added", **extra}).json()["id"]


def _upload(client, tender_id, content=b"%PDF-1.4 rfp", filename="RFP.pdf"):
    upload_id = "a" * 32
    assert client.put(f"/uploads/{upload_id}/parts/0", content=content).status_code == 200
    return client.post(
        f"/tenders/{tender_id}/manual-documents/uploads/{upload_id}/finish",
        json={"parts": 1, "filename": filename, "content_type": "application/pdf"},
    )


def test_upload_link_and_remove(tenders):
    client = TestClient(api.app)
    tender_id = _add_tender(client)

    res = _upload(client, tender_id)
    assert res.status_code == 200
    [doc] = res.json()["documents"]
    assert (doc["kind"], doc["filename"], doc["size"]) == ("file", "RFP.pdf", 12)
    assert "s3_key" not in doc

    res = client.post(f"/tenders/{tender_id}/manual-documents/links", json={"url": "https://portal.gov.in/boq.xlsx"})
    assert [d["kind"] for d in res.json()["documents"]] == ["file", "link"]
    assert client.post(f"/tenders/{tender_id}/manual-documents/links", json={"url": "ftp://x"}).status_code == 422
    assert client.post(f"/tenders/{tender_id}/manual-documents/links",
                       json={"url": "https://portal.gov.in/boq.xlsx"}).status_code == 409

    assert client.get(f"/tenders/{tender_id}/manual-documents/{doc['id']}/file").content == b"%PDF-1.4 rfp"

    key = tenders.find_one()["manual_documents"][0]["s3_key"]
    res = client.delete(f"/tenders/{tender_id}/manual-documents/{doc['id']}")
    assert [d["kind"] for d in res.json()["documents"]] == ["link"]
    assert storage.get_bytes(key) is None


def test_no_size_or_count_limit(tenders):
    client = TestClient(api.app)
    tender_id = _add_tender(client)
    upload_id = "b" * 32
    part = b"x" * api.TRANSFER_PART_SIZE
    for i in range(10):  # 30 MB
        assert client.put(f"/uploads/{upload_id}/parts/{i}", content=part).status_code == 200
    res = client.post(f"/tenders/{tender_id}/manual-documents/uploads/{upload_id}/finish",
                      json={"parts": 10, "filename": "big.pdf"})
    assert res.status_code == 200 and res.json()["documents"][0]["size"] == 10 * api.TRANSFER_PART_SIZE
    for i in range(40):
        client.post(f"/tenders/{tender_id}/manual-documents/links", json={"url": f"https://a.in/{i}.pdf"})
    assert len(tenders.find_one()["manual_documents"]) == 41


def test_changing_documents_forgets_the_text_read_from_the_old_set(tenders):
    client = TestClient(api.app)
    tender_id = _add_tender(client)
    tenders.update_one({}, {"$set": {"document_text_key": "tender-text/x.txt", "document_text_chars": 10}})
    _upload(client, tender_id)
    stored = tenders.find_one()
    assert "document_text_key" not in stored and "document_text_chars" not in stored


def test_scraped_tenders_take_documents_too(tenders):
    tender_id = tenders.insert_one({"title": "Scraped", "query_matches": [{"query_name": "Digital Marketing"}]}).inserted_id
    res = TestClient(api.app).post(f"/tenders/{tender_id}/manual-documents/links", json={"url": "https://a.in/x.pdf"})
    assert res.status_code == 200


def test_merge_keeps_the_tenderdetail_files_already_downloaded(tenders, tmp_path):
    client = TestClient(api.app)
    tender_id = _add_tender(client)
    _upload(client, tender_id)
    scraped_file = tmp_path / "docs" / "TDR1" / "notice.pdf"
    scraped_file.parent.mkdir(parents=True)
    scraped_file.write_bytes(b"%PDF notice")
    tenders.update_one({}, {"$set": {"documents": [{"filename": "notice.pdf", "local_path": str(scraped_file)}]}})

    settings = Settings(documents_dir=str(tmp_path / "docs"))
    manual_documents.collect_manual_documents([tenders.find_one()["_id"]], settings, tenders, merge=True)
    assert [d["filename"] for d in tenders.find_one()["documents"]] == ["notice.pdf", "RFP.pdf"]


def test_checklist_asks_for_documents_then_reads_them(tenders, monkeypatch):
    client = TestClient(api.app)
    tender_id = _add_tender(client, source_url="https://someportal.gov.in/tender/9")

    with pytest.raises(HTTPException) as exc:
        api._require_document_text(tenders.find_one())
    assert "Add this tender's documents first" in exc.value.detail

    _upload(client, tender_id)
    started = []
    monkeypatch.setattr(api, "_start_document_fetch", lambda tender: started.append(tender["_id"]) or True)
    with pytest.raises(HTTPException) as exc:
        api._require_document_text(tenders.find_one())
    assert exc.value.detail["fetching"] is True
    assert "documents added to this tender" in exc.value.detail["message"]
    assert [str(i) for i in started] == [tender_id]


def test_collector_puts_files_and_links_on_disk(tenders, monkeypatch, tmp_path):
    client = TestClient(api.app)
    tender_id = _add_tender(client)
    _upload(client, tender_id)
    client.post(f"/tenders/{tender_id}/manual-documents/links", json={"url": "https://portal.gov.in/notice"})
    client.post(f"/tenders/{tender_id}/manual-documents/links", json={"url": "https://portal.gov.in/broken.pdf"})

    def fake_download(url, index=1):
        if "broken" in url:
            raise manual_documents.LinkDownloadError("HTTP Error 404: Not Found")
        headers = Message()
        headers["Content-Type"] = "text/html; charset=utf-8"
        return manual_documents._filename_for_link(url, headers, index), b"<html>Scope of work</html>"

    monkeypatch.setattr(manual_documents, "download_link", fake_download)
    settings = Settings(documents_dir=str(tmp_path / "docs"))
    problems = manual_documents.collect_manual_documents([tenders.find_one()["_id"]], settings, tenders)

    stored = tenders.find_one()
    assert [d["filename"] for d in stored["documents"]] == ["RFP.pdf", "notice.html"]
    assert all(open(d["local_path"], "rb").read() for d in stored["documents"])
    assert stored["documents_downloaded_at"]
    assert "404" in stored["documents_download_error"]
    assert list(problems) == [tender_id]


def test_fetch_event_reads_hand_added_documents_without_the_browser(tenders, monkeypatch):
    import app.browser.document_collector as collector
    import app.database as database
    import app.intelligence.document_summarizer as summarizer

    client = TestClient(api.app)
    tender_id = _add_tender(client)
    _upload(client, tender_id)
    seen = {}
    monkeypatch.setattr(database, "get_collection", lambda: tenders)
    monkeypatch.setattr(collector, "run_document_collection", lambda *a, **k: pytest.fail("browser download"))
    monkeypatch.setattr(manual_documents, "collect_manual_documents",
                        lambda ids, settings=None: seen.setdefault("manual", ids) and {})
    monkeypatch.setattr(summarizer, "summarize_pending_documents",
                        lambda settings=None, tender_ids=None: summarizer.SummarizeSummary(summarized=1))

    result = lambda_handler.fetch_tender_documents([tender_id])
    assert [str(i) for i in seen["manual"]] == [tender_id]
    assert result["summarized"] == 1


def test_filename_for_link_adds_an_extension_from_the_content_type():
    headers = Message()
    headers["Content-Type"] = "application/pdf"
    assert manual_documents._filename_for_link("https://a.in/download?id=7", headers, 1) == "download.pdf"
    headers = Message()
    headers["Content-Disposition"] = 'attachment; filename="Tender RFP.docx"'
    assert manual_documents._filename_for_link("https://a.in/get", headers, 1) == "Tender RFP.docx"
