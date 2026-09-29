"""Lambda entry point for the scheduled pipeline run.

Not reachable via main.py (Click groups aren't directly callable as a Lambda
handler) - this is a separate, minimal entry point that calls the same
app.pipeline.run_pipeline() the CLI's `run` command uses, so neither
duplicates the orchestration.
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import logging
from typing import Any

from app.config import get_settings
from app.pipeline import run_pipeline


def handler(event: dict[str, Any] | None, context: Any) -> dict[str, Any]:
    """Run the full collect -> process -> screen -> (report) pipeline once.

    `event` may carry {"include_report": true} - set by whichever
    EventBridge schedule rule should also trigger the daily report email
    (Step 3 wires this up; only one of the two daily runs sets it).

    Or {"fetch_tender_documents": [tender ids]} - sent by the API when a
    checklist is waiting on those tenders' documents (see
    app.api.main._start_document_fetch): just downloads and reads them,
    the same as the CLI's `fetch-tender-documents`, instead of a full run.

    Exceptions are intentionally NOT caught here - a failed run must
    propagate and mark this invocation as failed, since that's what the
    CloudWatch Alarm (Step 3) watches to alert on a broken run.

    Settings are deliberately read here, inside the handler, and not at
    module import time: `get_settings()` is lru_cached, so reading it at
    import would permanently lock in whatever environment was present
    before this invocation's variables were actually injected.
    """
    settings = get_settings()
    logging.basicConfig(level=settings.log_level, format="%(levelname)s %(message)s")

    fetch_ids = (event or {}).get("fetch_tender_documents")
    if fetch_ids:
        return fetch_tender_documents(fetch_ids, settings)

    include_report = bool((event or {}).get("include_report", False))

    summary = run_pipeline(settings, include_report=include_report, trigger="scheduled")

    return dataclasses.asdict(summary)


def fetch_tender_documents(tender_ids: list[str], settings=None) -> dict[str, Any]:
    """Download and read just these tenders' documents, then mark the fetch
    finished - app.api.main tells the dashboard whether the documents are
    still being read, or were read but had no text in them."""
    from bson import ObjectId

    from app.browser.document_collector import run_document_collection
    from app.database import get_collection
    from app.intelligence.document_summarizer import summarize_pending_documents

    ids = [ObjectId(t) for t in tender_ids]
    downloads = run_document_collection(settings, tender_ids=ids)
    summary = summarize_pending_documents(settings, tender_ids=ids)
    get_collection().update_many(
        {"_id": {"$in": ids}},
        {"$set": {"document_fetch_finished_at": dt.datetime.now(dt.timezone.utc)}},
    )
    return {
        "downloads": [dataclasses.asdict(o) for o in downloads.outcomes],
        "summarized": summary.summarized,
        "failed": summary.failed,
    }
