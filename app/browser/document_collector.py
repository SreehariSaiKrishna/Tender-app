"""Document collector: for each live (not disappeared) tender already
tracked in Mongo, open its detail page and download the files listed under
"View Original Notice/Document" (see app.browser.tender_documents), then
record what was downloaded back onto that tender's own document.

Deliberately covers every live tender, not just domain-relevant ones (see
`domain_match`, app.processing.eligibility) - a tender's EMD/tender
fee/tender value is often only discoverable by reading its documents, and
the dashboard shows those fields for every tender, not just domain-matched
ones.

Mirrors app.browser.collector's shape (one persistent browser context for
the whole batch, one outcome per tender, a single tender's failure never
stops the rest) but drives a different destination: individual tender
detail pages (via each tender's stored `source_url`) rather than a saved
query's Live results list.
"""
from __future__ import annotations

import datetime as dt
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from playwright.sync_api import sync_playwright
from pymongo.collection import Collection

from app.browser.launch import chromium_launch_args
from app.browser.login import ensure_logged_in
from app.browser.tender_documents import (
    DocumentFetchError,
    NoDocumentsFoundError,
    extract_key_dates,
    fetch_document_bytes,
    list_document_rows,
    open_tender_detail,
)
from app.config import Settings, get_settings
from app.database import get_collection
from app.processing.normalizer import parse_indian_date

logger = logging.getLogger(__name__)


@dataclass
class TenderDocumentOutcome:
    tender_id: str
    tender_ref: str | None
    status: str  # "success" | "partial" | "failed" | "skipped"
    downloaded: list[str] = field(default_factory=list)
    error_message: str | None = None


@dataclass
class DocumentDownloadSummary:
    checked: int = 0
    outcomes: list[TenderDocumentOutcome] = field(default_factory=list)

    @property
    def succeeded(self) -> int:
        return sum(1 for o in self.outcomes if o.status == "success")

    @property
    def failed(self) -> int:
        return sum(1 for o in self.outcomes if o.status == "failed")


def _slugify(value: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")
    return slug or "tender"


def _needs_download(doc: dict[str, Any]) -> bool:
    """A tender needs (re-)downloading if it's never been downloaded, or its
    listing content has changed since the last download - mirrors the
    "only new/changed work" idea app.intelligence.scorer uses for
    re-screening.
    """
    downloaded_at = doc.get("documents_downloaded_at")
    if downloaded_at is None:
        return True
    changed_at = doc.get("content_last_changed")
    return changed_at is not None and changed_at > downloaded_at


def _extract_and_store_key_dates(page, tender: dict[str, Any], collection: Collection) -> None:
    """Best-effort: scrape the detail page's "Key Dates" table (see
    app.browser.tender_documents.extract_key_dates) and use it to fill in
    `opening_date` - TenderDetail's own Excel export never includes this
    field at all (see app.processing.normalizer's `opening_date`, always
    None from the listing alone), so the tender's own detail page is the
    only source for it, and it's authoritative site data rather than an AI
    guess from the attached documents.

    Never raises - a missing or unparseable Key Dates section must not
    fail the tender's document download, which is this function's caller.
    """
    try:
        page_key_dates = extract_key_dates(page)
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not read Key Dates for tender %s: %s", tender["_id"], exc)
        return
    if not page_key_dates:
        return

    update: dict[str, Any] = {"page_key_dates": page_key_dates}
    opening_date = parse_indian_date(page_key_dates.get("Tender Opening Date"))
    if opening_date is not None:
        update["opening_date"] = dt.datetime.combine(opening_date, dt.time.min).replace(
            tzinfo=dt.timezone.utc
        )
    collection.update_one({"_id": tender["_id"]}, {"$set": update})


# Tenders summarised before app.intelligence.document_summarizer kept each
# tender's `document_text` have only their summary left - the downloaded
# files were deleted once summarised. The submission checklist needs the
# documents' own text (annexure numbers, prescribed formats), so these are
# downloaded again; the summarizer then re-reads them (documents_downloaded_at
# is newer than the summary) and stores the text.
_MISSING_DOCUMENT_TEXT = {
    "document_summary_generated_at": {"$ne": None},
    "$or": [{"document_text": {"$exists": False}}, {"document_text": {"$in": [None, ""]}}],
}


def _pending_tenders(
    collection: Collection, limit: int | None, tender_ids: list[Any] | None = None
) -> list[dict[str, Any]]:
    """In order: tenders whose documents a checklist is waiting on (see
    app.api.main's `document_text_requested_at` - even if no longer live,
    since the checklist was asked for regardless), new/changed tenders, then
    tenders whose document text was never kept. `tender_ids` restricts the
    pass to just those tenders, downloaded whether or not they'd otherwise
    be due."""
    has_url = {"source_url": {"$nin": [None, ""]}}
    # document_text is large and never needed here - only whether it exists.
    projection = {"document_text": 0}
    if tender_ids is not None:
        pending = list(collection.find({**has_url, "_id": {"$in": tender_ids}}, projection))
        return pending[:limit] if limit else pending

    requested = list(
        collection.find(
            {**has_url, **_MISSING_DOCUMENT_TEXT, "document_text_requested_at": {"$ne": None}}, projection
        ).sort("document_text_requested_at", 1)
    )
    # Every live tender, not just domain-relevant ones - the dashboard shows
    # EMD/tender fee/tender value for every tender, and those are often only
    # discoverable by reading the tender's own documents.
    live = {**has_url, "disappeared": False}
    due = [doc for doc in collection.find(live, projection) if _needs_download(doc)]
    missing_text = list(collection.find({**live, **_MISSING_DOCUMENT_TEXT}, projection))

    pending: list[dict[str, Any]] = []
    seen: set[Any] = set()
    for doc in requested + due + missing_text:
        if doc["_id"] not in seen:
            seen.add(doc["_id"])
            pending.append(doc)
    return pending[:limit] if limit else pending


def _tenders_missing_key_dates(collection: Collection, limit: int | None) -> list[dict[str, Any]]:
    candidates = collection.find(
        {
            "disappeared": False,
            "source_url": {"$nin": [None, ""]},
            "page_key_dates": {"$exists": False},
        }
    )
    result = list(candidates)
    return result[:limit] if limit else result


@dataclass
class KeyDatesBackfillSummary:
    checked: int = 0
    found: int = 0
    failed: int = 0


def run_key_dates_backfill(
    settings: Settings | None = None, limit: int | None = None
) -> KeyDatesBackfillSummary:
    """Visit each eligible tender's detail page just to scrape its "Key
    Dates" table (see app.browser.tender_documents.extract_key_dates) - for
    tenders whose documents were already downloaded before this field
    existed. Unlike run_document_collection, this never re-fetches any
    document file; it's a lighter pass over tenders that already have
    `documents_downloaded_at` set but no `page_key_dates` yet.
    """
    settings = settings or get_settings()
    collection = get_collection()
    limit = settings.documents_batch_limit if limit is None else limit

    pending = _tenders_missing_key_dates(collection, limit)
    summary = KeyDatesBackfillSummary(checked=len(pending))
    if not pending:
        return summary

    profile_dir = settings.resolved_path(settings.tenderdetail_profile_dir)
    profile_dir.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        context = p.chromium.launch_persistent_context(
            str(profile_dir),
            headless=settings.browser_headless,
            viewport={"width": 1400, "height": 900},
            accept_downloads=True,
            args=chromium_launch_args(),
        )
        page = context.new_page()

        try:
            logged_in = ensure_logged_in(
                page,
                settings.tenderdetail_login_url,
                username=settings.tenderdetail_username,
                password=settings.tenderdetail_password,
            )
        except Exception as exc:  # noqa: BLE001 - fatal, stop safely
            logger.error("Login raised an unexpected error: %s", exc)
            logged_in = False

        if not logged_in:
            context.close()
            logger.error("Stopping Key Dates backfill: could not confirm login.")
            return summary

        for tender in pending:
            try:
                open_tender_detail(page, tender.get("source_url"))
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "Could not open detail page for tender %s: %s", tender["_id"], exc
                )
                summary.failed += 1
                continue

            before = collection.find_one({"_id": tender["_id"]}, {"page_key_dates": 1})
            _extract_and_store_key_dates(page, tender, collection)
            after = collection.find_one({"_id": tender["_id"]}, {"page_key_dates": 1})
            if after and after.get("page_key_dates") and not (before or {}).get("page_key_dates"):
                summary.found += 1

        context.close()

    return summary


def run_document_collection(
    settings: Settings | None = None, limit: int | None = None, tender_ids: list[Any] | None = None
) -> DocumentDownloadSummary:
    """Download attached documents for every eligible tender that doesn't
    have them yet (or whose listing changed since the last download, or
    whose document text was never kept) - or just for `tender_ids`.

    Never raises for an individual tender's failure - only for login
    failure or a fatal browser error, which stop the whole run safely (same
    contract as app.browser.collector.run_collection).
    """
    settings = settings or get_settings()
    collection = get_collection()
    limit = settings.documents_batch_limit if limit is None else limit

    pending = _pending_tenders(collection, limit, tender_ids)
    summary = DocumentDownloadSummary(checked=len(pending))
    if not pending:
        return summary

    documents_dir = settings.resolved_path(settings.documents_dir)
    documents_dir.mkdir(parents=True, exist_ok=True)
    profile_dir = settings.resolved_path(settings.tenderdetail_profile_dir)
    profile_dir.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as p:
        context = p.chromium.launch_persistent_context(
            str(profile_dir),
            headless=settings.browser_headless,
            viewport={"width": 1400, "height": 900},
            accept_downloads=True,
            # See app.browser.launch: Lambda-only sandbox workaround args -
            # applying them outside Lambda crashes Chromium on real sites.
            args=chromium_launch_args(),
        )
        page = context.new_page()

        try:
            logged_in = ensure_logged_in(
                page,
                settings.tenderdetail_login_url,
                username=settings.tenderdetail_username,
                password=settings.tenderdetail_password,
            )
        except Exception as exc:  # noqa: BLE001 - fatal, stop safely
            logger.error("Login raised an unexpected error: %s", exc)
            logged_in = False

        if not logged_in:
            context.close()
            logger.error("Stopping document collection: could not confirm login.")
            return summary

        for tender in pending:
            outcome = _download_one_tender(page, tender, documents_dir, collection)
            summary.outcomes.append(outcome)

        context.close()

    return summary


def _download_one_tender(
    page, tender: dict[str, Any], documents_dir: Path, collection: Collection
) -> TenderDocumentOutcome:
    tender_id = str(tender["_id"])
    tender_ref = tender.get("tender_ref")
    source_url = tender.get("source_url")

    try:
        open_tender_detail(page, source_url)
    except Exception as exc:  # noqa: BLE001
        logger.error("Could not open detail page for tender %s: %s", tender_id, exc)
        return TenderDocumentOutcome(
            tender_id=tender_id, tender_ref=tender_ref, status="failed", error_message=str(exc)
        )

    _extract_and_store_key_dates(page, tender, collection)

    try:
        rows = list_document_rows(page)
    except NoDocumentsFoundError as exc:
        return TenderDocumentOutcome(
            tender_id=tender_id, tender_ref=tender_ref, status="skipped", error_message=str(exc)
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Could not list documents for tender %s: %s", tender_id, exc)
        return TenderDocumentOutcome(
            tender_id=tender_id, tender_ref=tender_ref, status="failed", error_message=str(exc)
        )

    dest_dir = documents_dir / _slugify(tender_ref or tender.get("dedup_key") or tender_id)
    dest_dir.mkdir(parents=True, exist_ok=True)

    saved_meta: list[dict[str, Any]] = []
    errors: list[str] = []
    now = dt.datetime.now(dt.timezone.utc)

    for row in rows:
        if not row["url"]:
            errors.append(f"{row['filename']}: Download link had no href")
            continue

        try:
            data = fetch_document_bytes(page, row["url"])
        except DocumentFetchError as exc:
            logger.error(
                "Failed to download '%s' for tender %s: %s", row["filename"], tender_id, exc
            )
            errors.append(f"{row['filename']}: {exc}")
            continue
        except Exception as exc:  # noqa: BLE001 - one file failing must not lose the rest
            logger.error(
                "Failed to download '%s' for tender %s: %s", row["filename"], tender_id, exc
            )
            errors.append(f"{row['filename']}: {exc}")
            continue

        filename = row["filename"] or "document"
        target = dest_dir / filename
        target.write_bytes(data)

        saved_meta.append(
            {
                "filename": filename,
                "description": row["description"],
                "local_path": str(target),
                "downloaded_at": now,
            }
        )

    if saved_meta and not errors:
        status = "success"
    elif saved_meta:
        status = "partial"
    else:
        status = "failed"

    update: dict[str, Any] = {"documents_download_error": "; ".join(errors) or None}
    if saved_meta:
        update["documents"] = saved_meta
        update["documents_downloaded_at"] = now
    collection.update_one({"_id": tender["_id"]}, {"$set": update})

    return TenderDocumentOutcome(
        tender_id=tender_id,
        tender_ref=tender_ref,
        status=status,
        downloaded=[m["filename"] for m in saved_meta],
        error_message="; ".join(errors) if errors else None,
    )
