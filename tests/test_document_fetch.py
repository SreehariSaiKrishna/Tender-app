"""On-demand document fetch for a checklist waiting on a tender's documents
- app.api.main._require_document_text and app.lambda_handler's
`fetch_tender_documents` event."""
from __future__ import annotations

import datetime as dt
import json
import sys
import types

import mongomock
import pytest
from fastapi import HTTPException

from app import lambda_handler
from app.api import main as api
from app.config import Settings


@pytest.fixture
def collection(monkeypatch):
    coll = mongomock.MongoClient().db.tenders
    monkeypatch.setattr(api, "get_collection", lambda: coll)
    return coll


@pytest.fixture
def invokes(monkeypatch):
    """Stands in for boto3's Lambda client; records each invoke."""
    calls = []

    class FakeLambda:
        def invoke(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setitem(sys.modules, "boto3", types.SimpleNamespace(client=lambda name: FakeLambda()))
    monkeypatch.setattr(api, "get_settings", lambda: Settings(pipeline_function_name="pipeline-fn"))
    return calls


def make_tender(collection, **fields):
    doc = {"source_url": "https://example.com/t/1", **fields}
    doc["_id"] = collection.insert_one(doc).inserted_id
    return collection.find_one({"_id": doc["_id"]})


def refused(tender) -> HTTPException:
    with pytest.raises(HTTPException) as exc:
        api._require_document_text(tender)
    assert exc.value.status_code == 409
    return exc.value


def test_tender_with_document_text_passes(collection, invokes):
    api._require_document_text(make_tender(collection, document_text="RFP"))
    assert invokes == []


def test_missing_text_starts_a_fetch_and_says_so(collection, invokes):
    tender = make_tender(collection)

    exc = refused(tender)

    assert exc.detail["fetching"] is True
    assert [json.loads(c["Payload"]) for c in invokes] == [{"fetch_tender_documents": [str(tender["_id"])]}]
    assert invokes[0]["FunctionName"] == "pipeline-fn" and invokes[0]["InvocationType"] == "Event"
    stored = collection.find_one({"_id": tender["_id"]})
    assert stored["document_fetch_started_at"] and stored["document_text_requested_at"]


def test_retrying_while_a_fetch_runs_does_not_start_another(collection, invokes):
    tender = make_tender(collection)
    refused(tender)

    exc = refused(collection.find_one({"_id": tender["_id"]}))

    assert exc.detail["fetching"] is True
    assert len(invokes) == 1


def test_a_stale_fetch_is_started_again(collection, invokes):
    long_ago = dt.datetime.now(dt.timezone.utc) - dt.timedelta(hours=1)
    tender = make_tender(collection, document_fetch_started_at=long_ago)

    refused(tender)

    assert len(invokes) == 1


def test_finished_fetch_without_text_says_documents_unreadable(collection, invokes):
    started = dt.datetime(2026, 9, 25, 10, 0)
    tender = make_tender(
        collection, document_fetch_started_at=started, document_fetch_finished_at=started + dt.timedelta(minutes=3)
    )

    exc = refused(tender)

    assert exc.detail["fetching"] is False and "no text could be read" in exc.detail["message"]
    assert invokes == []


def test_without_a_pipeline_function_falls_back_to_the_queue(collection, monkeypatch):
    monkeypatch.setattr(api, "get_settings", lambda: Settings(pipeline_function_name=""))
    tender = make_tender(collection)

    exc = refused(tender)

    assert isinstance(exc.detail, str) and "queued for the next document download" in exc.detail
    assert collection.find_one({"_id": tender["_id"]})["document_text_requested_at"]


def test_handler_fetch_event_downloads_and_reads_just_those_tenders(monkeypatch):
    coll = mongomock.MongoClient().db.tenders
    tender_id = coll.insert_one({"source_url": "https://example.com/t/1"}).inserted_id
    seen = {}

    import app.browser.document_collector as collector
    import app.database as database
    import app.intelligence.document_summarizer as summarizer

    def fake_download(settings=None, tender_ids=None):
        seen["downloaded"] = tender_ids
        return collector.DocumentDownloadSummary(checked=len(tender_ids))

    def fake_summarize(settings=None, tender_ids=None):
        seen["summarized"] = tender_ids
        return summarizer.SummarizeSummary(summarized=1)

    monkeypatch.setattr(collector, "run_document_collection", fake_download)
    monkeypatch.setattr(summarizer, "summarize_pending_documents", fake_summarize)
    monkeypatch.setattr(database, "get_collection", lambda: coll)
    monkeypatch.setattr(lambda_handler, "run_pipeline", lambda *a, **k: pytest.fail("full pipeline run"))

    result = lambda_handler.handler({"fetch_tender_documents": [str(tender_id)]}, None)

    assert seen == {"downloaded": [tender_id], "summarized": [tender_id]}
    assert result["summarized"] == 1
    assert coll.find_one({"_id": tender_id})["document_fetch_finished_at"]
