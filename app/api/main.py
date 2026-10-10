"""Read-mostly API over the `tenders` collection - serves the dashboard
(Step 5). Deliberately narrow: the only writes are the "mark applied",
"decline", "checklist" and "generate bid" endpoints below. Everything except that last
one only needs app.config/app.database - fastapi, mangum, pymongo - to keep
this Lambda's image light. POST /tenders/{id}/generate-bid is the one
exception: it needs reportlab/pypdf (PDF rendering, app.reports.bid_generator)
and, for its AI-drafted documents, the OpenAI SDK via
app.intelligence.bid_drafter/scorer - all three are declared in
app/api/requirements.txt.

Runs locally the same way the CLI does, alongside app.lambda_handler:
    uvicorn app.api.main:app --reload
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import io
import json
import logging
import os
import re
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal
from urllib.parse import quote, urlparse

from bson import ObjectId
from bson.errors import InvalidId
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from starlette.concurrency import run_in_threadpool
from mangum import Mangum
from pydantic import BaseModel, Field, TypeAdapter, ValidationError
from pymongo import DESCENDING

from app import storage
from app.config import CONFIG_DIR, get_settings, load_company_profile, load_saved_queries
from app.convert.converter import ConversionError, convert_to_pdf, is_word_document
from app.convert.word_text import NewParagraph, ParagraphChange, WordTextError, apply_edits, read_paragraphs
from app.text_sections import USE_IN_OPTIONS, default_sections, parse_day, with_text_sections
from app.database import (
    get_automation_runs_collection,
    get_collection,
    get_company_documents_collection,
    get_eligibility_criteria_collection,
    get_stamp_transfers_collection,
    get_text_sections_collection,
)
from app.intelligence.bid_drafter import (
    BidDraftingError,
    draft_checklist_documents,
    draft_presentation,
    plan_submission_checklist,
)
from app.reports.bid_generator import (
    CHECKLIST_VERSION,
    DEFAULT_WHERE,
    STATUS_NOT_APPLICABLE,
    BidGenerationError,
    CompanyDocumentRef,
    ContentLayout,
    MarkPlacement,
    PageMarks,
    build_checklist,
    build_compliance_matrix,
    collapse_unfilled_cvs,
    combine_marked,
    company_background_text,
    document_page_count,
    drop_resolved_missing_information,
    enclosure_documents,
    established_facts,
    generate_bid_package,
    guard_financial_years,
    is_submission_checklist,
    mark_document,
    merge_drafted_open_items,
    prepare_check_list_annexures,
    records_not_drafted,
    refresh_statuses,
    signing_assets,
)
from app.models import UNSCREENED_PRIORITY_RANK, TenderStatus
from app.reports.presentation import is_presentation_row, render_presentation

logger = logging.getLogger(__name__)

app = FastAPI(title="Tender Intelligence API")

# Locally the dashboard is served by this app (same origin, see
# SERVE_DASHBOARD below); any other localhost port is allowed too. There's no login
# locally - keep uvicorn bound to 127.0.0.1. Deployed, API Gateway sets the
# CORS headers itself (template.yaml's CorsConfiguration) and its Cognito
# authorizer rejects any request without a login before it reaches this
# app; FRONTEND_URL (the dashboard's origins, comma-separated - its
# CloudFront address and custom domain, from the FrontendUrl and
# FrontendCustomUrl parameters) are allowed too so this middleware never
# 400s a request from the real dashboard.
app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_origins=[
        origin.strip()
        for origin in os.environ.get("FRONTEND_URL", "").split(",")
        if origin.strip()
    ],
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["content-type", "authorization"],
)

# Local dev (scripts/dev.ps1 sets SERVE_DASHBOARD=1) serves the dashboard
# from this same uvicorn process, so the page can never be up without the
# API behind it. Never set deployed - CloudFront serves the dashboard there.
if os.environ.get("SERVE_DASHBOARD") == "1":
    from fastapi.responses import FileResponse

    _DASHBOARD = Path(__file__).resolve().parents[2] / "frontend" / "index.html"

    @app.get("/", include_in_schema=False)
    def dashboard() -> FileResponse:
        return FileResponse(_DASHBOARD, headers={"Cache-Control": "no-store"})



@app.exception_handler(storage.StorageNotConfiguredError)
def _storage_not_configured(request: Request, exc: storage.StorageNotConfiguredError) -> JSONResponse:
    """A local run with no FILES_BUCKET - say so rather than a bare 500."""
    return JSONResponse(status_code=503, content={"detail": str(exc)})


CLOSING_SOON_DAYS = 7

# Tenders the AI screening scored as at least plausibly relevant to the
# business's capabilities (config/capabilities.json) - "shortlisted" for the
# dashboard's eligibility filter. Unscreened tenders (no latest_priority yet)
# are deliberately excluded rather than assumed relevant.
SHORTLISTED_PRIORITIES = ["High", "Medium"]


def _iso(value: dt.datetime | None) -> str | None:
    if not value:
        return None
    # PyMongo returns naive datetimes (values are stored as UTC but tzinfo
    # is stripped on read), so tag them explicitly - otherwise the browser's
    # `new Date(iso)` parses the string as local time instead of UTC, and the
    # frontend's IST conversion ends up displaying the raw UTC clock reading.
    if value.tzinfo is None:
        value = value.replace(tzinfo=dt.timezone.utc)
    return value.isoformat()


def _serialize(doc: dict[str, Any]) -> dict[str, Any]:
    """Make a Mongo document JSON-safe and drop the noisy verbatim source
    row - the UI has no use for it and it roughly doubles payload size.
    """
    out = dict(doc)
    out["id"] = str(out.pop("_id"))
    out.pop("raw_data", None)
    # The tender documents' full text (kept for the submission checklist -
    # see app.intelligence.document_summarizer) is large and UI-irrelevant.
    out.pop("document_text", None)
    # The editable submission checklist is served on its own (GET
    # /tenders/{id}/checklist) - rows only need checklist_generated_at.
    out.pop("checklist", None)
    out.pop("document_text_key", None)
    out["bid_pack_available"] = _bid_pack_available(doc)
    out["presentation_available"] = bool(doc.get("presentation_key")) and out["bid_pack_available"]
    out.pop("bid_pack_key", None)
    out.pop("presentation_key", None)
    out.pop("edited_fields", None)
    if "manual_documents" in out:
        out["manual_documents"] = [_serialize_manual_document(e) for e in out["manual_documents"]]

    for key in (
        "first_seen",
        "last_seen",
        "published_date",
        "opening_date",
        "closing_date",
        "previous_closing_date",
        "content_last_changed",
        "applied_at",
        "declined_at",
        "documents_downloaded_at",
        "document_summary_generated_at",
        "bid_generated_at",
        "checklist_generated_at",
        "reopened_at",
        "disappeared_at",
        "edited_at",
        "hidden_at",
    ):
        if out.get(key) is not None:
            out[key] = _iso(out[key])

    for event in out.get("status_history", []):
        for key in ("at", "closing_date", "previous_closing_date"):
            if event.get(key) is not None:
                event[key] = _iso(event[key])

    for match in out.get("query_matches", []):
        match["first_seen"] = _iso(match.get("first_seen"))
        match["last_seen"] = _iso(match.get("last_seen"))

    for screening in out.get("screenings", []):
        screening["screened_at"] = _iso(screening.get("screened_at"))

    for document in out.get("documents", []):
        document["downloaded_at"] = _iso(document.get("downloaded_at"))

    return out


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/tenders")
def list_tenders(
    status: str | None = None,
    priority: str | None = None,
    query_name: str | None = None,
    closing_soon: bool = False,
    shortlisted_only: bool = False,
    eligible_only: bool = False,
    reopened_only: bool = False,
    q: str | None = None,
    limit: int = Query(50, ge=1, le=200),
    skip: int = Query(0, ge=0),
) -> dict[str, Any]:
    collection = get_collection()
    # "closed" tenders are exactly the disappeared ones (see
    # app.processing.deduplicator._mark_disappeared) - every other status
    # implies disappeared=False, so only the closed bucket needs to look
    # past the default "still live" filter.
    filt: dict[str, Any] = {"disappeared": status == "closed", "hidden": {"$ne": True}}
    if status:
        filt["status"] = status
    if priority:
        filt["latest_priority"] = priority
    elif shortlisted_only:
        filt["latest_priority"] = {"$in": SHORTLISTED_PRIORITIES}
    if query_name:
        filt["query_matches.query_name"] = query_name
    if closing_soon:
        today = dt.datetime.combine(
            dt.datetime.now(dt.timezone.utc).date(), dt.time.min
        ).replace(tzinfo=dt.timezone.utc)
        cutoff = today + dt.timedelta(days=CLOSING_SOON_DAYS)
        filt["closing_date"] = {"$ne": None, "$gte": today, "$lte": cutoff}
    if eligible_only:
        filt["eligibility_match"] = True
    if reopened_only:
        # Ever reopened (app.processing.deduplicator) - a sticky flag, so
        # this also finds ones that have since closed again (closed bucket).
        filt["reopened"] = True
    if q:
        pattern = re.compile(re.escape(q), re.IGNORECASE)
        filt["$or"] = [{"title": pattern}, {"organisation": pattern}]

    total = collection.count_documents(filt)
    # Eligibility-matched tenders (see app.processing.eligibility) sort
    # first even when not filtered down to them exclusively, so the
    # dashboard surfaces likely-relevant tenders without hiding the rest.
    # Within that, most recently published tenders first (today's on top).
    cursor = (
        collection.find(filt, {"document_text": 0})
        .sort(
            [
                ("eligibility_match", -1),
                ("published_date", DESCENDING),
            ]
        )
        .skip(skip)
        .limit(limit)
    )
    tenders = [_serialize(d) for d in cursor]
    return {"total": total, "count": len(tenders), "tenders": tenders}


@app.get("/tenders/{tender_id}")
def get_tender(tender_id: str) -> dict[str, Any]:
    collection = get_collection()
    try:
        object_id = ObjectId(tender_id)
    except InvalidId:
        raise HTTPException(status_code=400, detail="Invalid tender id")

    doc = collection.find_one({"_id": object_id})
    if doc is None:
        raise HTTPException(status_code=404, detail="Tender not found")
    return _serialize(doc)


@app.post("/tenders/{tender_id}/apply")
def mark_applied(tender_id: str) -> dict[str, Any]:
    """Record that the user applied to this tender - a one-way flag; once
    set, the frontend disables the decline option for this tender and
    there's no "unapply".
    """
    collection = get_collection()
    try:
        object_id = ObjectId(tender_id)
    except InvalidId:
        raise HTTPException(status_code=400, detail="Invalid tender id")

    result = collection.update_one(
        {"_id": object_id},
        {"$set": {"applied": True, "applied_at": dt.datetime.now(dt.timezone.utc)}},
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Tender not found")

    doc = collection.find_one({"_id": object_id})
    return _serialize(doc)


class DeclineRequest(BaseModel):
    reason: str


@app.post("/tenders/{tender_id}/decline")
def mark_declined(tender_id: str, body: DeclineRequest) -> dict[str, Any]:
    """Record that the user decided not to bid on this tender, and why -
    the mirror of POST /tenders/{id}/apply. Also a one-way flag; once set,
    the frontend disables the apply option for this tender, and there's no
    "undecline".
    """
    reason = body.reason.strip()
    if not reason:
        raise HTTPException(status_code=422, detail="A decline reason is required")

    collection = get_collection()
    try:
        object_id = ObjectId(tender_id)
    except InvalidId:
        raise HTTPException(status_code=400, detail="Invalid tender id")

    result = collection.update_one(
        {"_id": object_id},
        {
            "$set": {
                "declined": True,
                "declined_at": dt.datetime.now(dt.timezone.utc),
                "decline_reason": reason,
            }
        },
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Tender not found")

    doc = collection.find_one({"_id": object_id})
    return _serialize(doc)


# --- Manually added tenders ----------------------------------------------------
# Tenders found outside the saved queries (another portal, a newspaper, an
# email) are added by hand under this pseudo-query, so they get the same
# drill-down list, checklist and bid pack as scraped ones. No pipeline pass
# ever processes this name, so app.processing.deduplicator._mark_disappeared
# never closes them.
MANUAL_QUERY_NAME = "Other Tenders"


class ManualTenderRequest(BaseModel):
    title: str
    organisation: str | None = None
    tender_ref: str | None = None
    location: str | None = None
    state: str | None = None
    published_date: dt.date | None = None
    closing_date: dt.date | None = None
    opening_date: dt.date | None = None
    tender_value: float | None = Field(None, ge=0)
    earnest_money: float | None = Field(None, ge=0)
    document_fees: float | None = Field(None, ge=0)
    source_url: str | None = None
    description: str | None = None


def _clean(value: str | None) -> str | None:
    value = (value or "").strip()
    return value or None


def _utc_midnight(value: dt.date | None) -> dt.datetime | None:
    if value is None:
        return None
    return dt.datetime.combine(value, dt.time.min).replace(tzinfo=dt.timezone.utc)


@app.post("/tenders/manual", status_code=201)
def add_manual_tender(body: ManualTenderRequest) -> dict[str, Any]:
    """Adds a tender by hand to the Other Tenders bucket. A tender already
    on record under the same dedup key (same reference number, or same
    title+organisation+closing date) is just added to the bucket rather
    than duplicated."""
    from app.processing.eligibility import (
        compute_eligibility_match,
        emd_in_range,
        load_match_keywords,
        tender_fee_in_range,
        tender_value_in_range,
    )
    from app.processing.normalizer import make_dedup_key

    title = _clean(body.title)
    if not title:
        raise HTTPException(status_code=422, detail="A tender title is required")
    tender_ref = _clean(body.tender_ref)
    organisation = _clean(body.organisation)
    source_url = _clean(body.source_url)
    if source_url and not re.match(r"https?://", source_url, re.IGNORECASE):
        raise HTTPException(status_code=422, detail="The tender link must start with http:// or https://")
    dedup_key, is_synthetic = make_dedup_key(tender_ref, title, organisation, body.closing_date)

    collection = get_collection()
    now = dt.datetime.now(dt.timezone.utc)
    manual_match = {"query_name": MANUAL_QUERY_NAME, "first_seen": now, "last_seen": now, "manual": True}

    existing = collection.find_one({"dedup_key": dedup_key})
    if existing is not None:
        if any(m.get("query_name") == MANUAL_QUERY_NAME for m in existing.get("query_matches", [])):
            raise HTTPException(status_code=409, detail="This tender is already in Other Tenders")
        collection.update_one(
            {"_id": existing["_id"]},
            {"$push": {"query_matches": manual_match}, "$inc": {"query_match_count": 1}},
        )
        return _serialize(collection.find_one({"_id": existing["_id"]}))

    description = _clean(body.description)
    domain_match = compute_eligibility_match(f"{title} {description or ''}", load_match_keywords())
    doc = {
        "dedup_key": dedup_key,
        "tender_ref": tender_ref,
        "ref_is_synthetic": is_synthetic,
        "title": title,
        "organisation": organisation,
        "location": _clean(body.location),
        "state": _clean(body.state),
        "published_date": _utc_midnight(body.published_date or now.date()),
        "opening_date": _utc_midnight(body.opening_date),
        "closing_date": _utc_midnight(body.closing_date),
        "previous_closing_date": None,
        "deadline_changed": False,
        "tender_value": body.tender_value,
        "earnest_money": body.earnest_money,
        "document_fees": body.document_fees,
        "document_url": None,
        "source_url": source_url,
        "description": description,
        "first_seen": now,
        "last_seen": now,
        "content_last_changed": now,
        "times_found": 1,
        "disappeared": False,
        "status": TenderStatus.NEW.value,
        "query_matches": [manual_match],
        "query_match_count": 1,
        "screenings": [],
        "applied": False,
        "applied_at": None,
        "reopened": False,
        "reopened_at": None,
        "reopen_count": 0,
        "status_history": [],
        "latest_priority_rank": UNSCREENED_PRIORITY_RANK,
        "manually_added": True,
        "domain_match": domain_match,
        "eligibility_match": (
            domain_match
            and tender_value_in_range(body.tender_value)
            and emd_in_range(body.earnest_money)
            and tender_fee_in_range(body.document_fees)
        ),
    }
    inserted = collection.insert_one(doc)
    return _serialize(collection.find_one({"_id": inserted.inserted_id}))


@app.delete("/tenders/{tender_id}/manual")
def remove_manual_tender(tender_id: str) -> dict[str, Any]:
    """Takes a tender out of Other Tenders. One that was only ever added by
    hand is deleted outright; one the scrape also found just loses its
    Other Tenders entry - scraped tenders are never deleted."""
    collection = get_collection()
    try:
        object_id = ObjectId(tender_id)
    except InvalidId:
        raise HTTPException(status_code=400, detail="Invalid tender id")
    doc = collection.find_one({"_id": object_id})
    # Already removed (another tab, a double click) - the outcome asked for.
    if doc is None:
        return {"deleted": True}
    if not any(m.get("query_name") == MANUAL_QUERY_NAME for m in doc.get("query_matches", [])):
        return {"deleted": False}

    if doc.get("manually_added") and len(doc.get("query_matches", [])) == 1:
        collection.delete_one({"_id": object_id})
        _delete_files_quietly(_uploaded_keys(doc))
        return {"deleted": True}
    collection.update_one(
        {"_id": object_id},
        {"$pull": {"query_matches": {"query_name": MANUAL_QUERY_NAME}}, "$inc": {"query_match_count": -1}},
    )
    return {"deleted": False}


@app.post("/tenders/{tender_id}/other-tenders")
def add_to_other_tenders(tender_id: str) -> dict[str, Any]:
    """Adds a scraped tender to Other Tenders too (its saved queries keep it)."""
    collection = get_collection()
    try:
        object_id = ObjectId(tender_id)
    except InvalidId:
        raise HTTPException(status_code=400, detail="Invalid tender id")
    now = dt.datetime.now(dt.timezone.utc)
    result = collection.update_one(
        {"_id": object_id, "query_matches.query_name": {"$ne": MANUAL_QUERY_NAME}},
        {
            "$push": {"query_matches": {"query_name": MANUAL_QUERY_NAME, "first_seen": now, "last_seen": now, "manual": True}},
            "$inc": {"query_match_count": 1},
        },
    )
    doc = collection.find_one({"_id": object_id})
    if doc is None:
        raise HTTPException(status_code=404, detail="Tender not found")
    if result.modified_count == 0:
        raise HTTPException(status_code=409, detail="This tender is already in Other Tenders")
    return _serialize(doc)


# Fields an edit can change - the same as the Add Tender form.
_EDITABLE_TEXT = ("title", "organisation", "tender_ref", "location", "state", "source_url", "description")
_EDITABLE_DATES = ("published_date", "closing_date", "opening_date")
_EDITABLE_AMOUNTS = ("tender_value", "earnest_money", "document_fees")


@app.put("/tenders/{tender_id}")
def edit_tender(tender_id: str, body: ManualTenderRequest) -> dict[str, Any]:
    """Edits any tender's details. A scraped tender's edits are kept in
    `edited_fields` too, so the next scrape doesn't overwrite them (see
    app.processing.deduplicator)."""
    from app.processing.eligibility import (
        compute_eligibility_match,
        emd_in_range,
        load_match_keywords,
        tender_fee_in_range,
        tender_value_in_range,
    )
    from app.processing.normalizer import make_dedup_key

    collection = get_collection()
    try:
        object_id = ObjectId(tender_id)
    except InvalidId:
        raise HTTPException(status_code=400, detail="Invalid tender id")
    doc = collection.find_one({"_id": object_id}, {"document_text": 0, "raw_data": 0})
    if doc is None:
        raise HTTPException(status_code=404, detail="Tender not found")

    values: dict[str, Any] = {name: _clean(getattr(body, name)) for name in _EDITABLE_TEXT}
    if not values["title"]:
        raise HTTPException(status_code=422, detail="A tender title is required")
    if values["source_url"] and not re.match(r"https?://", values["source_url"], re.IGNORECASE):
        raise HTTPException(status_code=422, detail="The tender link must start with http:// or https://")
    values.update({name: _utc_midnight(getattr(body, name)) for name in _EDITABLE_DATES})
    values.update({name: getattr(body, name) for name in _EDITABLE_AMOUNTS})

    def _differs(name: str) -> bool:
        old, new = doc.get(name), values[name]
        if isinstance(old, dt.datetime) or isinstance(new, dt.datetime):
            old_day = old.date() if isinstance(old, dt.datetime) else None
            return old_day != (new.date() if new else None)
        return old != new

    changed = {name: values[name] for name in values if _differs(name)}
    if not changed:
        return _serialize(collection.find_one({"_id": object_id}))

    update: dict[str, Any] = dict(changed)
    if doc.get("manually_added"):
        dedup_key, is_synthetic = make_dedup_key(
            values["tender_ref"], values["title"], values["organisation"],
            values["closing_date"].date() if values["closing_date"] else None,
        )
        if dedup_key != doc.get("dedup_key"):
            if collection.find_one({"dedup_key": dedup_key, "_id": {"$ne": object_id}}, {"_id": 1}):
                raise HTTPException(status_code=409, detail="Another tender already has this reference number "
                                                            "(or the same title, organisation and closing date)")
            update.update({"dedup_key": dedup_key, "ref_is_synthetic": is_synthetic})
    else:
        update.update({f"edited_fields.{name}": value for name, value in changed.items()})

    merged = {**doc, **values}
    domain_match = compute_eligibility_match(
        f"{merged.get('title') or ''} {merged.get('description') or ''}", load_match_keywords()
    ) or (not doc.get("manually_added") and bool(doc.get("domain_match")))
    update["domain_match"] = domain_match
    update["eligibility_match"] = (
        domain_match
        and tender_value_in_range(merged.get("tender_value"))
        and emd_in_range(merged.get("earnest_money"))
        and tender_fee_in_range(merged.get("document_fees"))
    )
    update["edited_at"] = dt.datetime.now(dt.timezone.utc)
    collection.update_one({"_id": object_id}, {"$set": update})
    return _serialize(collection.find_one({"_id": object_id}))


@app.delete("/tenders/{tender_id}")
def delete_tender(tender_id: str) -> dict[str, Any]:
    """Deletes a tender from the dashboard. One added by hand is deleted
    outright, with its uploaded documents. A scraped one is hidden for good
    instead - deleting it would only bring it back with the next scrape."""
    collection = get_collection()
    try:
        object_id = ObjectId(tender_id)
    except InvalidId:
        raise HTTPException(status_code=400, detail="Invalid tender id")
    doc = collection.find_one({"_id": object_id}, {"document_text": 0, "raw_data": 0})
    if doc is None:
        # Already gone (another tab, a double click) - the outcome asked for.
        return {"deleted": True}
    # The record goes first - files that can't be deleted (see
    # _delete_files_quietly) never keep a tender on the dashboard.
    scraped = any(m.get("query_name") != MANUAL_QUERY_NAME for m in doc.get("query_matches", []))
    if doc.get("manually_added") and not scraped:
        collection.delete_one({"_id": object_id})
    else:
        collection.update_one(
            {"_id": object_id},
            {"$set": {"hidden": True, "hidden_at": dt.datetime.now(dt.timezone.utc), "manual_documents": []}},
        )
    _delete_files_quietly(_uploaded_keys(doc) + [doc[k] for k in ("bid_pack_key", "presentation_key") if doc.get(k)])
    return {"deleted": True}


def _uploaded_keys(doc: dict[str, Any]) -> list[str]:
    return [e["s3_key"] for e in doc.get("manual_documents", []) if e.get("s3_key")]


def _delete_files_quietly(keys: list[str]) -> None:
    """Best effort: a file that can't be deleted (no permission - e.g. a
    local run on a role without s3:DeleteObject - or a storage blip) is
    logged and left behind, never failing the request that removed it."""
    for key in keys:
        try:
            storage.delete(key)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not delete stored file %s: %s", key, exc)


# --- A tender's own documents added by hand ----------------------------------
# Uploaded files (kept under tender-documents/, see
# app.processing.manual_documents) and links to documents or pages on any
# site - on any tender. The checklist reads them through the same on-demand
# fetch as a TenderDetail tender's documents (_require_document_text) - the
# pipeline function downloads the links, reads every file's text (alongside
# a scraped tender's TenderDetail documents) and summarises it.
#
# No limit on a file's size or on how many a tender has - like Sign & Stamp,
# only the API function's memory bounds one upload (it's joined there before
# being stored).


def _tender_for_documents_or_404(tender_id: str) -> tuple[ObjectId, dict[str, Any]]:
    try:
        object_id = ObjectId(tender_id)
    except InvalidId:
        raise HTTPException(status_code=400, detail="Invalid tender id")
    tender = get_collection().find_one({"_id": object_id}, {"document_text": 0, "raw_data": 0})
    if tender is None:
        raise HTTPException(status_code=404, detail="Tender not found")
    return object_id, tender


def _serialize_manual_document(entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": entry["id"],
        "kind": entry.get("kind"),
        "filename": entry.get("filename"),
        "url": entry.get("url"),
        "size": entry.get("size"),
        "added_at": _iso(entry.get("added_at")),
    }


def _manual_documents_response(tender: dict[str, Any]) -> dict[str, Any]:
    return {
        "documents": [_serialize_manual_document(e) for e in tender.get("manual_documents", [])],
        # Their text has been read since the documents last changed.
        "documents_read": bool(tender.get("document_text_key") or tender.get("document_text")),
        "download_error": tender.get("documents_download_error"),
    }


def _save_manual_document(object_id: ObjectId, tender: dict[str, Any], entry: dict[str, Any]) -> dict[str, Any]:
    """Adds `entry` and forgets the text read from the previous set of
    documents, so the next checklist reads them all again."""
    collection = get_collection()
    collection.update_one(
        {"_id": object_id},
        {"$push": {"manual_documents": entry}, **_DOCUMENTS_CHANGED},
    )
    return _manual_documents_response(collection.find_one({"_id": object_id}, {"document_text": 0}))


# The previously read text no longer matches the documents (the stored file
# itself is overwritten on the next read).
_DOCUMENTS_CHANGED = {
    "$unset": {
        "document_text_key": "",
        "document_text_chars": "",
        "document_text": "",
        "document_fetch_started_at": "",
        "document_fetch_finished_at": "",
        "documents_download_error": "",
    }
}


@app.get("/tenders/{tender_id}/manual-documents")
def list_manual_documents(tender_id: str) -> dict[str, Any]:
    _, tender = _tender_for_documents_or_404(tender_id)
    return _manual_documents_response(tender)


class ManualDocumentUploadFinish(BaseModel):
    parts: int = Field(..., ge=1)
    filename: str
    content_type: str | None = None


@app.post("/tenders/{tender_id}/manual-documents/uploads/{upload_id}/finish")
def finish_manual_document_upload(tender_id: str, upload_id: str, body: ManualDocumentUploadFinish) -> dict[str, Any]:
    """A tender document sent with PUT /uploads/{id}/parts/N - any size."""
    from app.processing.manual_documents import safe_filename, tender_document_key

    object_id, tender = _tender_for_documents_or_404(tender_id)
    content = _take_upload(upload_id, body.parts, max_size=None)
    if not content:
        raise HTTPException(status_code=422, detail="File is empty")
    document_id = uuid.uuid4().hex
    filename = safe_filename(body.filename)
    key = tender_document_key(object_id, document_id, filename)
    storage.put_bytes(key, content, body.content_type or "application/octet-stream")
    entry = {
        "id": document_id,
        "kind": "file",
        "filename": filename,
        "content_type": body.content_type or "application/octet-stream",
        "size": len(content),
        "s3_key": key,
        "added_at": dt.datetime.now(dt.timezone.utc),
    }
    try:
        return _save_manual_document(object_id, tender, entry)
    except HTTPException:
        _delete_files_quietly([key])
        raise


class ManualDocumentLink(BaseModel):
    url: str


@app.post("/tenders/{tender_id}/manual-documents/links")
def add_manual_document_link(tender_id: str, body: ManualDocumentLink) -> dict[str, Any]:
    """A link to one of the tender's documents (or its page) on any site -
    downloaded when the documents are next read."""
    object_id, tender = _tender_for_documents_or_404(tender_id)
    url = body.url.strip()
    if not re.match(r"https?://\S+$", url, re.IGNORECASE):
        raise HTTPException(status_code=422, detail="Enter a full link starting with http:// or https://")
    if any(e.get("url") == url for e in tender.get("manual_documents", [])):
        raise HTTPException(status_code=409, detail="That link is already added")
    entry = {
        "id": uuid.uuid4().hex,
        "kind": "link",
        "url": url,
        "added_at": dt.datetime.now(dt.timezone.utc),
    }
    return _save_manual_document(object_id, tender, entry)


@app.delete("/tenders/{tender_id}/manual-documents/{document_id}")
def remove_manual_document(tender_id: str, document_id: str) -> dict[str, Any]:
    object_id, tender = _tender_for_documents_or_404(tender_id)
    entry = next((e for e in tender.get("manual_documents", []) if e.get("id") == document_id), None)
    if entry is None:
        raise HTTPException(status_code=404, detail="Document not found")
    collection = get_collection()
    collection.update_one(
        {"_id": object_id},
        {"$pull": {"manual_documents": {"id": document_id}}, **_DOCUMENTS_CHANGED},
    )
    _delete_files_quietly(_uploaded_keys({"manual_documents": [entry]}))
    return _manual_documents_response(collection.find_one({"_id": object_id}, {"document_text": 0}))


def _manual_file_or_404(tender_id: str, document_id: str) -> dict[str, Any]:
    _, tender = _tender_for_documents_or_404(tender_id)
    entry = next((e for e in tender.get("manual_documents", []) if e.get("id") == document_id), None)
    if entry is None or entry.get("kind") != "file":
        raise HTTPException(status_code=404, detail="Document not found")
    return entry


@app.get("/tenders/{tender_id}/manual-documents/{document_id}/link")
def manual_document_link(tender_id: str, document_id: str) -> dict[str, Any]:
    """Like /bid-document/link: `url` is null locally, where .../file serves it."""
    entry = _manual_file_or_404(tender_id, document_id)
    return {"url": storage.download_url(entry["s3_key"], entry["filename"]), "filename": entry["filename"]}


@app.get("/tenders/{tender_id}/manual-documents/{document_id}/file")
def manual_document_file(tender_id: str, document_id: str) -> Response:
    entry = _manual_file_or_404(tender_id, document_id)
    return Response(
        content=_document_bytes(entry["s3_key"]),
        media_type=entry.get("content_type") or "application/octet-stream",
        headers={"Content-Disposition": _content_disposition("attachment", entry["filename"])},
    )


def _load_company_documents_for_generation() -> list[CompanyDocumentRef]:
    """The documents library (see the Documents tab section below), reshaped
    for app.reports.bid_generator - `open_bytes` is lazy so generating a bid
    pack only reads the bytes of documents it actually ends up referencing.
    """
    refs = []
    for record in get_company_documents_collection().find():
        metadata = record.get("metadata") or {}
        refs.append(
            CompanyDocumentRef(
                id=str(record["_id"]),
                name=metadata.get("display_name") or record["filename"],
                filename=record["filename"],
                content_type=metadata.get("content_type") or "application/octet-stream",
                open_bytes=lambda key=record["s3_key"]: _document_bytes(key),
                size=record.get("length"),
                mark_kind=metadata.get("mark_kind"),
            )
        )
    return refs


def _find_tender(tender_id: str) -> tuple[ObjectId, dict[str, Any]]:
    """The tender, with its document text read back from file storage (see
    app.storage.load_document_text) for the checklist and bid pack."""
    try:
        object_id = ObjectId(tender_id)
    except InvalidId:
        raise HTTPException(status_code=400, detail="Invalid tender id")
    tender = get_collection().find_one({"_id": object_id})
    if tender is None:
        raise HTTPException(status_code=404, detail="Tender not found")
    return object_id, storage.load_document_text(tender)


def _letterhead_path(company_profile: dict[str, Any]) -> Path | None:
    """company_profile.json's letterhead_image, when that file exists."""
    if not company_profile.get("letterhead_image"):
        return None
    candidate = CONFIG_DIR / company_profile["letterhead_image"]
    return candidate if candidate.is_file() else None


def _logo_path(company_profile: dict[str, Any]) -> Path | None:
    """company_profile.json's logo_image (for presentation slides), when
    that file exists."""
    if not company_profile.get("logo_image"):
        return None
    candidate = CONFIG_DIR / company_profile["logo_image"]
    return candidate if candidate.is_file() else None


def _generation_inputs() -> tuple[list[dict[str, Any]], dict[str, Any], list[CompanyDocumentRef]]:
    """Eligibility criteria, company profile and documents library - what
    both the checklist and the bid pack are built from. The profile carries
    the Documents tab's Text sections too (see app.text_sections)."""
    criteria_doc = get_eligibility_criteria_collection().find_one({"_id": CRITERIA_DOC_ID})
    return (
        (criteria_doc or {}).get("criteria", []),
        with_text_sections(load_company_profile(), _load_text_sections()),
        _load_company_documents_for_generation(),
    )


# --- Master Bid Submission Checklist ------------------------------------------
# Every document this tender asks for, as data on the tender (see
# app.reports.bid_generator.build_checklist), so the team can edit rows,
# pick library documents and tick letterhead/signature/stamp on the
# dashboard - POST /generate-bid then builds the pack row by row from this
# saved version.


# A fetch that hasn't finished by then (the pipeline Lambda's 15-minute
# limit) is taken to have failed, and the next request starts another.
_DOCUMENT_FETCH_TIMEOUT = dt.timedelta(minutes=16)


def _start_document_fetch(tender: dict[str, Any]) -> bool:
    """Asks PipelineFunction to download and read this tender's documents
    right away (see app.lambda_handler.fetch_tender_documents), rather than
    leaving them for the next scheduled run - at most once per
    _DOCUMENT_FETCH_TIMEOUT, since the dashboard keeps asking while it
    waits. Running locally (no pipeline function) the fetch runs in this
    process instead - see _fetch_documents_locally. True while a fetch is
    running; False when it couldn't be started."""
    function_name = get_settings().pipeline_function_name
    run_locally = not function_name and not os.environ.get("AWS_LAMBDA_FUNCTION_NAME")
    if not function_name and not run_locally:
        return False
    now = dt.datetime.now(dt.timezone.utc)
    claimed = get_collection().update_one(
        {
            "_id": tender["_id"],
            "$or": [
                {"document_fetch_started_at": None},
                {"document_fetch_started_at": {"$lt": now - _DOCUMENT_FETCH_TIMEOUT}},
            ],
        },
        {"$set": {"document_fetch_started_at": now}},
    )
    if not claimed.modified_count:
        return True  # already running
    if run_locally:
        threading.Thread(target=_fetch_documents_locally, args=(tender["_id"],), daemon=True).start()
        return True
    try:
        import boto3

        boto3.client("lambda").invoke(
            FunctionName=function_name,
            InvocationType="Event",
            Payload=json.dumps({"fetch_tender_documents": [str(tender["_id"])]}).encode(),
        )
    except Exception as exc:  # noqa: BLE001 - fall back to the scheduled run
        logger.error("Could not start document fetch for tender %s: %s", tender["_id"], exc)
        get_collection().update_one({"_id": tender["_id"]}, {"$unset": {"document_fetch_started_at": ""}})
        return False
    return True


def _fetch_documents_locally(tender_id: ObjectId) -> None:
    """Running the API locally there's no pipeline function to invoke, so
    the fetch runs here in the background - the same as the CLI's
    `fetch-tender-documents`. Marked finished even when it fails, so the
    dashboard stops waiting and says the documents couldn't be read."""
    from app.lambda_handler import fetch_tender_documents

    try:
        fetch_tender_documents([str(tender_id)])
    except Exception:  # noqa: BLE001 - reported through document_fetch_finished_at
        logger.exception("Local document fetch failed for tender %s", tender_id)
        get_collection().update_one(
            {"_id": tender_id}, {"$set": {"document_fetch_finished_at": dt.datetime.now(dt.timezone.utc)}}
        )


def _documents_unreadable(tender: dict[str, Any]) -> bool:
    """The last on-demand fetch finished, but left no document text - the
    files were downloaded and read and none had any (scanned images,
    unsupported formats), or the download itself failed."""
    started, finished = tender.get("document_fetch_started_at"), tender.get("document_fetch_finished_at")
    return started is not None and finished is not None and finished >= started


def _is_tenderdetail_url(url: str | None) -> bool:
    host = (urlparse(url or "").hostname or "").lower()
    return host == "tenderdetail.com" or host.endswith(".tenderdetail.com")


def _require_document_text(tender: dict[str, Any]) -> None:
    """A checklist is only built from the tender's own documents (the files
    under "View Original Notice/Document", read into `document_text` by
    app.intelligence.document_summarizer) - never from its summary alone,
    which drops annexure numbers and prescribed formats. Without that text,
    the tender is flagged for the next document download (see
    app.browser.document_collector._pending_tenders), a fetch is started
    now (_start_document_fetch), and the request is refused (409) with a
    `detail` of {"message", "fetching"} - the dashboard retries while
    `fetching` is true."""
    if (tender.get("document_text") or "").strip():
        return
    # Other Tenders: read from the documents added by hand - or, with none,
    # from a TenderDetail link like any scraped tender. Any other link is
    # just a reference; there's no knowing where its documents are.
    hand_added = bool(tender.get("manual_documents"))
    if tender.get("manually_added") and not hand_added and not _is_tenderdetail_url(tender.get("source_url")):
        raise HTTPException(
            status_code=409,
            detail="Add this tender's documents first - open Documents on the tender and upload the files "
                   "or add links to them. The checklist is built from what they say.",
        )
    if not hand_added and not tender.get("source_url"):
        raise HTTPException(
            status_code=409,
            detail="This tender has no source page to download its documents from, so a checklist "
                   "can't be built from its documents.",
        )
    if _documents_unreadable(tender):
        if hand_added:
            problem = tender.get("documents_download_error")
            raise HTTPException(
                status_code=409,
                detail={
                    "fetching": False,
                    "message": "No text could be read from the documents added to this tender (scanned images "
                               "or an unsupported format, or a link that couldn't be downloaded)"
                               + (f" - {problem}" if problem else "")
                               + ". Upload the files themselves (PDF, Word, Excel or ZIP) under Documents, "
                                 "then generate the checklist again.",
                },
            )
        raise HTTPException(
            status_code=409,
            detail={
                "fetching": False,
                "message": "This tender's documents (View Original Notice/Document) were downloaded, but no "
                           "text could be read from them (scanned images or an unsupported format, or the "
                           "download failed), so the checklist can't be built from them. Check the documents "
                           "on the tender's page, or run "
                           f"`python main.py fetch-tender-documents {tender['_id']}` to see why.",
            },
        )
    get_collection().update_one(
        {"_id": tender["_id"], "document_text_requested_at": None},
        {"$set": {"document_text_requested_at": dt.datetime.now(dt.timezone.utc)}},
    )
    which = "the documents added to this tender" if hand_added else \
        "this tender's documents (View Original Notice/Document)"
    if _start_document_fetch(tender):
        raise HTTPException(
            status_code=409,
            detail={
                "fetching": True,
                "message": f"Downloading and reading {which} now - this takes a few minutes. The checklist "
                           "is built as soon as they're read.",
            },
        )
    if tender.get("manually_added"):
        raise HTTPException(
            status_code=409,
            detail=f"Couldn't start reading {which} - try generating the checklist again in a minute.",
        )
    raise HTTPException(
        status_code=409,
        detail="This tender's documents (View Original Notice/Document) haven't been read yet, so the "
               "checklist can't be built from them. They're queued for the next document download - or "
               f"run `python main.py fetch-tender-documents {tender['_id']}` now - then generate the "
               "checklist again.",
    )


def _build_tender_checklist(
    tender: dict[str, Any],
    eligibility_criteria: list[dict[str, Any]],
    company_profile: dict[str, Any],
    company_documents: list[CompanyDocumentRef],
) -> dict[str, Any]:
    """The AI reads the tender's documents for the rows (see
    app.intelligence.bid_drafter.plan_submission_checklist); without it
    (no OPENAI_API_KEY, a bad response) they come from the document summary.
    Refused (409) until the documents themselves have been read - see
    _require_document_text."""
    _require_document_text(tender)
    library_names = [d.name for d in enclosure_documents(company_documents, company_profile)]
    plan, note = None, None
    try:
        plan = plan_submission_checklist(tender, eligibility_criteria, company_profile, library_names)
    except BidDraftingError as exc:
        logger.warning("AI submission checklist unavailable for tender %s: %s", tender.get("_id"), exc)
        note = (f"AI checklist building was unavailable ({exc}) - these rows come from the tender's extracted "
                "document list; check them against the tender document.")
    return build_checklist(tender, eligibility_criteria, company_profile, company_documents, plan, note)


def _checklist_response(
    tender: dict[str, Any],
    company_profile: dict[str, Any],
    company_documents: list[CompanyDocumentRef],
) -> dict[str, Any]:
    checklist = drop_resolved_missing_information(tender["checklist"], tender)
    checklist["generated_at"] = _iso(checklist.get("generated_at"))
    checklist["updated_at"] = _iso(checklist.get("updated_at"))
    summary = tender.get("document_summary") or {}
    return {
        "tender": {
            "id": str(tender["_id"]),
            "title": tender.get("title"),
            "organisation": tender.get("organisation"),
            "tender_ref": tender.get("tender_ref"),
            "source_url": tender.get("source_url"),
            "published_date": _iso(tender.get("published_date")),
            "closing_date": _iso(tender.get("closing_date")),
            "opening_date": _iso(tender.get("opening_date")),
            "tender_value": tender.get("tender_value"),
            "earnest_money": tender.get("earnest_money"),
            "document_fees": tender.get("document_fees"),
            "document_summary": {
                k: summary.get(k)
                for k in (
                    "estimated_bid_amount", "emd_amount", "tender_fee_amount",
                    "tender_opening_date", "key_dates", "technical_criteria_table",
                )
            },
            "bid_pack_available": _bid_pack_available(tender),
        },
        "checklist": checklist,
        # What a row's "Attach" picker can choose from.
        "library": [
            {"id": d.id, "name": d.name, "filename": d.filename}
            for d in enclosure_documents(company_documents, company_profile)
        ],
    }


@app.post("/tenders/{tender_id}/checklist")
def generate_checklist(tender_id: str) -> dict[str, Any]:
    """(Re)builds this tender's submission checklist from the tender, the
    eligibility profile and the documents library - replacing any edits."""
    object_id, tender = _find_tender(tender_id)
    eligibility_criteria, company_profile, company_documents = _generation_inputs()
    checklist = _build_tender_checklist(tender, eligibility_criteria, company_profile, company_documents)
    get_collection().update_one(
        {"_id": object_id},
        {"$set": {"checklist": checklist, "checklist_generated_at": checklist["generated_at"]}},
    )
    return _checklist_response({**tender, "checklist": checklist}, company_profile, company_documents)


@app.get("/tenders/{tender_id}/checklist")
def get_checklist(tender_id: str) -> dict[str, Any]:
    object_id, tender = _find_tender(tender_id)
    if not tender.get("checklist"):
        raise HTTPException(status_code=404, detail="No checklist has been generated for this tender yet")
    eligibility_criteria, company_profile, company_documents = _generation_inputs()
    if not is_submission_checklist(tender["checklist"]):
        # Saved in the older six-section review format - rebuilt once, as
        # the submission checklist, rather than shown half-understood.
        checklist = _build_tender_checklist(tender, eligibility_criteria, company_profile, company_documents)
        get_collection().update_one({"_id": object_id}, {"$set": {"checklist": checklist}})
        tender = {**tender, "checklist": checklist}
    return _checklist_response(tender, company_profile, company_documents)


class ChecklistRow(BaseModel):
    id: str | None = None
    document: str = ""
    what_to_upload: str = ""
    where: str = DEFAULT_WHERE
    status: str = ""
    source: Literal["upload", "draft"] = "upload"
    document_id: str | None = None
    letterhead: bool = False
    signature: bool = False
    stamp: bool = False
    # Absent on checklists saved before these existed - defaults keep them loading.
    notary: bool = False
    section: str = ""
    reference: str = ""
    format_text: str = ""
    notes: str = ""
    done: bool = False
    origin: Literal["auto", "user"] = "user"


class ChecklistNote(BaseModel):
    id: str | None = None
    section: Literal["open_items", "missing_information"] = "open_items"
    requirement: str = ""
    evidence: str = ""
    done: bool = False
    origin: Literal["auto", "user", "ai_draft"] = "user"


class ChecklistHeader(BaseModel):
    bid_number: str = ""
    bid_end: str = ""
    tender: str = ""
    organisation: str = ""
    bidder: str = ""
    summary_only: bool = False


class ChecklistUpdateRequest(BaseModel):
    items: list[ChecklistRow]
    notes: list[ChecklistNote] | None = None
    header: ChecklistHeader | None = None


def _clean_rows(models: list[BaseModel], required: str) -> list[dict[str, Any]]:
    rows = []
    for model in models:
        row = model.model_dump()
        for key, value in row.items():
            if isinstance(value, str) and key != "id":
                row[key] = value.strip()
        if not row[required]:
            continue  # a blank row added then left empty
        row["id"] = row["id"] or uuid.uuid4().hex
        rows.append(row)
    return rows


@app.put("/tenders/{tender_id}/checklist")
def update_checklist(tender_id: str, body: ChecklistUpdateRequest) -> dict[str, Any]:
    object_id, tender = _find_tender(tender_id)
    if not tender.get("checklist"):
        raise HTTPException(status_code=404, detail="No checklist has been generated for this tender yet")
    checklist = {
        **tender["checklist"],
        "version": CHECKLIST_VERSION,
        "items": _clean_rows(body.items, "document"),
        "updated_at": dt.datetime.now(dt.timezone.utc),
    }
    for row in checklist["items"]:
        row["where"] = row["where"] or DEFAULT_WHERE
    if body.notes is not None:
        checklist["notes"] = _clean_rows(body.notes, "requirement")
    if body.header is not None:
        checklist["header"] = body.header.model_dump()
    get_collection().update_one({"_id": object_id}, {"$set": {"checklist": checklist}})
    _, company_profile, company_documents = _generation_inputs()
    return _checklist_response({**tender, "checklist": checklist}, company_profile, company_documents)


# A bid-pack build that hasn't finished by then (ApiFunction's own Timeout
# in template.yaml, plus a margin) is taken to have failed, and the next
# request starts another.
_BID_GENERATION_TIMEOUT = dt.timedelta(minutes=6)


@app.post("/tenders/{tender_id}/generate-bid")
def generate_bid(tender_id: str, response: Response) -> dict[str, Any]:
    """Builds a bid pack PDF for this tender (see _build_bid_pack) and
    marks the tender applied.

    Deployed, the build (AI drafting plus the PDF merge) regularly outlasts
    TenderHttpApi's hard 30s limit, so this only starts it - ApiFunction
    invokes itself asynchronously with {"generate_bid": tender_id} (see
    handler) - and answers 202 with the same body as GET
    /tenders/{id}/generate-bid, which the dashboard polls until it's
    "done" or "failed". Locally (no Lambda) it builds in this request and
    returns the updated tender, as before.

    This never contacts the tendering authority or any external system -
    see this project's stated scope in README.md. It only drafts a local
    PDF for a human to review, complete and sign before anything is
    actually submitted.
    """
    object_id, tender = _find_tender(tender_id)
    # Without a saved checklist the build makes one first, which needs the
    # tender's documents - refuse now (and start fetching them) rather than
    # have the background build fail on it.
    if not is_submission_checklist(tender.get("checklist")):
        _require_document_text(tender)

    function_name = os.environ.get("AWS_LAMBDA_FUNCTION_NAME")
    if not function_name:
        return _build_bid_pack(object_id, tender)

    collection = get_collection()
    now = dt.datetime.now(dt.timezone.utc)
    run_id = uuid.uuid4().hex
    claimed = collection.update_one(
        {
            "_id": object_id,
            "$or": [
                {"bid_generation_started_at": None},
                {"bid_generation_started_at": {"$lt": now - _BID_GENERATION_TIMEOUT}},
            ],
        },
        {
            "$set": {"bid_generation_started_at": now, "bid_generation_run": run_id},
            "$unset": {"bid_generation_error": ""},
        },
    )
    if claimed.modified_count:  # otherwise one is already running
        try:
            import boto3

            boto3.client("lambda").invoke(
                FunctionName=function_name,
                InvocationType="Event",
                Payload=json.dumps({"generate_bid": tender_id, "run": run_id}).encode(),
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Could not start bid-pack generation for tender %s: %s", tender_id, exc)
            collection.update_one(
                {"_id": object_id}, {"$unset": {"bid_generation_started_at": "", "bid_generation_run": ""}}
            )
            raise HTTPException(status_code=503, detail="Couldn't start generating the bid pack - try again.")

    response.status_code = 202
    return _bid_generation_status(collection.find_one({"_id": object_id}))


@app.get("/tenders/{tender_id}/generate-bid")
def bid_generation_status(tender_id: str) -> dict[str, Any]:
    """Where the background bid-pack build POST /generate-bid started has
    got to - see _bid_generation_status."""
    try:
        object_id = ObjectId(tender_id)
    except InvalidId:
        raise HTTPException(status_code=400, detail="Invalid tender id")
    tender = get_collection().find_one({"_id": object_id})
    if tender is None:
        raise HTTPException(status_code=404, detail="Tender not found")
    return _bid_generation_status(tender)


def _bid_generation_status(tender: dict[str, Any]) -> dict[str, Any]:
    """{"status": "running" | "done" | "failed" | "none", "message"}, plus
    the serialized tender once it's "done"."""
    started = tender.get("bid_generation_started_at")
    if started is not None:
        if started.tzinfo is None:
            started = started.replace(tzinfo=dt.timezone.utc)
        if dt.datetime.now(dt.timezone.utc) - started < _BID_GENERATION_TIMEOUT:
            return {"status": "running", "message": None}
        return {"status": "failed", "message": "Generating the bid pack took too long and was stopped - try again."}
    if tender.get("bid_generation_error"):
        return {"status": "failed", "message": tender["bid_generation_error"]}
    if _bid_pack_available(tender):
        return {"status": "done", "message": None, "tender": _serialize(tender)}
    return {"status": "none", "message": None}


def _run_bid_generation(tender_id: str, run_id: str | None) -> None:
    """The background half of POST /generate-bid (ApiFunction invoked with
    {"generate_bid": tender_id, "run": run_id}). Never raises - a failure is
    recorded on the tender for GET /generate-bid to report.

    Lambda retries an async invocation that timed out or crashed (up to
    twice), which would re-run - and re-bill - the OpenAI drafting. So each
    run first takes its `run` id off the tender: a retry, or a run another
    request has since replaced, finds it gone and does nothing."""
    if not run_id:
        logger.warning("Skipping bid-pack generation for tender %s - no run id", tender_id)
        return
    try:
        picked_up = get_collection().update_one(
            {"_id": ObjectId(tender_id), "bid_generation_run": run_id},
            {"$unset": {"bid_generation_run": ""}},
        )
    except Exception:  # noqa: BLE001 - the status times out on its own
        logger.exception("Could not pick up bid-pack generation for tender %s", tender_id)
        return
    if not picked_up.modified_count:
        logger.warning("Skipping bid-pack generation for tender %s - run %s is a retry or was replaced",
                       tender_id, run_id)
        return
    try:
        object_id, tender = _find_tender(tender_id)
        _build_bid_pack(object_id, tender)
    except Exception as exc:  # noqa: BLE001
        if isinstance(exc, HTTPException):
            detail = exc.detail
            message = detail.get("message") if isinstance(detail, dict) else str(detail)
        else:
            logger.exception("Bid-pack generation failed for tender %s", tender_id)
            message = "Generating the bid pack failed unexpectedly - try again."
        try:
            get_collection().update_one(
                {"_id": ObjectId(tender_id)},
                {"$set": {"bid_generation_error": message}, "$unset": {"bid_generation_started_at": ""}},
            )
        except Exception:  # noqa: BLE001 - the status times out on its own
            logger.exception("Could not record the bid-pack failure for tender %s", tender_id)


def _build_bid_pack(object_id: ObjectId, tender: dict[str, Any]) -> dict[str, Any]:
    """Builds a bid pack PDF for this tender from its saved submission
    checklist (see app.reports.bid_generator.generate_bid_package): every
    row's document in S.No order - library documents attached, and each
    document the bidder must write AI-drafted from the tender (see
    app.intelligence.bid_drafter.draft_checklist_documents) - marks the
    tender applied and returns it.

    The pack isn't kept: it's put in file storage under bid-packs/ only so
    the browser can download it (GET /tenders/{id}/bid-document/link), and
    that prefix expires after a day (template.yaml) - it's too big for one
    API response.
    """
    tender_id = str(object_id)
    collection = get_collection()
    eligibility_criteria, company_profile, company_documents = _generation_inputs()

    checklist = tender.get("checklist")
    if not is_submission_checklist(checklist):
        checklist = _build_tender_checklist(tender, eligibility_criteria, company_profile, company_documents)
    # A saved checklist may still list an issued record (office address
    # proof, an empanelment certificate) as a document to draft.
    checklist = records_not_drafted(checklist)

    # AI-draft every row the bidder writes itself. `facts` tells the
    # drafting prompt which company facts are already established, from the
    # compliance matrix against the eligibility profile. Never a hard
    # failure: rows without a draft get a templated page to complete.
    facts = established_facts(build_compliance_matrix(eligibility_criteria, company_profile, company_documents))
    to_draft = [
        r for r in checklist.get("items", [])
        if r.get("source") == "draft" and r.get("status") != STATUS_NOT_APPLICABLE
    ]
    # A presentation row becomes a slide deck (app.reports.presentation),
    # drafted alongside the other documents rather than as a letter.
    decks = [r for r in to_draft if is_presentation_row(r)]
    to_draft = [r for r in to_draft if not is_presentation_row(r)]
    drafted: dict[str, Any] = {}
    drafted_decks: dict[str, Any] = {}
    errors: list[str] = []
    drafting_note = None
    if to_draft or decks:
        # The brochure and other reference documents are never enclosed in
        # the pack - their text only gives the drafter descriptive background.
        background = company_background_text(company_documents, company_profile)
        with ThreadPoolExecutor(max_workers=len(decks) or 1) as pool:
            deck_runs = {r["id"]: pool.submit(draft_presentation, tender, r, company_profile, facts,
                                              company_background=background) for r in decks}
            if to_draft:
                try:
                    drafted, draft_errors = draft_checklist_documents(
                        tender, to_draft, company_profile, facts, company_background=background,
                        pack_rows=checklist.get("items", []), conditions=checklist.get("conditions") or [],
                    )
                    errors += draft_errors
                except BidDraftingError as exc:
                    errors.append(str(exc))
            for r in decks:
                try:
                    drafted_decks[r["id"]] = deck_runs[r["id"]].result()
                except BidDraftingError as exc:
                    errors.append(f"{r['document']}: {exc}")
        if errors:
            logger.warning("AI bid-document drafting incomplete for tender %s: %s", tender_id, errors)
            drafting_note = (
                "AI drafting was unavailable for some documents (" + "; ".join(errors) + "). Those pages are "
                "templates to complete by hand."
            )

    # Every presentation row gets a deck - its fixed slides come from the
    # verified company data even when its drafted slides are unavailable.
    rendered: dict[str, Any] = {}
    for r in decks:
        try:
            rendered[r["id"]] = render_presentation(
                tender, company_profile, checklist.get("header"), drafted_decks.get(r["id"]),
                _logo_path(company_profile))
        except Exception:  # noqa: BLE001 - the row falls back to a page to complete by hand
            logger.exception("Rendering the presentation failed for tender %s", tender_id)

    letterhead = _letterhead_path(company_profile)
    if letterhead is None and company_profile.get("letterhead_image"):
        logger.warning("Letterhead image %s not found - bid pack will use plain pages",
                       CONFIG_DIR / company_profile["letterhead_image"])

    drafted = collapse_unfilled_cvs(drafted, to_draft, company_profile)
    # A turnover table must give each year its own CA-certified figure.
    drafted = guard_financial_years(drafted, company_profile)
    drafted = prepare_check_list_annexures(drafted, to_draft)
    deck_items = [SimpleNamespace(title=r["document"], open_items=drafted_decks[r["id"]].open_items)
                  for r in decks if r["id"] in drafted_decks and r["id"] in rendered]
    checklist = merge_drafted_open_items(checklist, list(drafted.values()) + deck_items)
    checklist = refresh_statuses(checklist, company_documents, set(drafted) | set(rendered))

    try:
        pdf_bytes = generate_bid_package(
            tender,
            eligibility_criteria,
            company_profile,
            company_documents,
            drafted_documents=drafted,
            drafting_note=drafting_note,
            letterhead_image=letterhead,
            checklist=checklist,
            presentations={row_id: deck.pdf for row_id, deck in rendered.items()},
        )
    except BidGenerationError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    now = dt.datetime.now(dt.timezone.utc)
    pack_key = f"{storage.BID_PACKS_PREFIX}{tender_id}/{uuid.uuid4().hex}.pdf"
    storage.put_bytes(pack_key, pdf_bytes, "application/pdf")
    # The PowerPoint version of the (first) presentation, downloadable
    # beside the pack (GET /tenders/{id}/presentation/link) - kept and
    # expired with it.
    presentation_key = None
    if rendered:
        presentation_key = f"{storage.BID_PACKS_PREFIX}{tender_id}/{uuid.uuid4().hex}.pptx"
        storage.put_bytes(presentation_key, next(iter(rendered.values())).pptx, PPTX_MEDIA_TYPE)
    # The previous pack goes as soon as the new one is stored.
    for old in ("bid_pack_key", "presentation_key"):
        if tender.get(old):
            try:
                storage.delete(tender[old])
            except Exception as exc:  # noqa: BLE001 - it expires on its own anyway
                logger.warning("Could not delete the previous %s for tender %s: %s", old, tender_id, exc)

    update: dict[str, Any] = {
        "bid_pack_key": pack_key,
        "presentation_key": presentation_key,
        "bid_generated_at": now,
        "checklist": checklist,
    }
    if not tender.get("checklist_generated_at"):
        update["checklist_generated_at"] = checklist.get("generated_at") or now
    if not tender.get("applied"):
        update["applied"] = True
        update["applied_at"] = now
    collection.update_one(
        {"_id": object_id},
        {"$set": update, "$unset": {"bid_generation_started_at": "", "bid_generation_error": ""}},
    )

    doc = collection.find_one({"_id": object_id})
    return _serialize(doc)


# How long after generation a bid pack is offered for download. The
# bid-packs/ lifecycle rule (template.yaml) deletes it after a day; S3 runs
# those rules lazily, so the pack may linger a little past that, but it's
# never offered past this window.
BID_PACK_AVAILABLE_FOR = dt.timedelta(hours=20)


def _bid_pack_available(tender: dict[str, Any]) -> bool:
    generated_at = tender.get("bid_generated_at")
    if not tender.get("bid_pack_key") or generated_at is None:
        return False
    if generated_at.tzinfo is None:
        generated_at = generated_at.replace(tzinfo=dt.timezone.utc)
    return dt.datetime.now(dt.timezone.utc) - generated_at < BID_PACK_AVAILABLE_FOR


def _bid_pack_key_or_404(tender_id: str) -> tuple[dict[str, Any], str]:
    try:
        object_id = ObjectId(tender_id)
    except InvalidId:
        raise HTTPException(status_code=400, detail="Invalid tender id")
    tender = get_collection().find_one({"_id": object_id}, {"bid_pack_key": 1, "bid_generated_at": 1, "tender_ref": 1})
    if tender is None:
        raise HTTPException(status_code=404, detail="Tender not found")
    if not _bid_pack_available(tender):
        raise HTTPException(status_code=404, detail="No bid pack to download - generate it again.")
    return tender, tender["bid_pack_key"]


def _bid_pack_filename(tender: dict[str, Any]) -> str:
    return f"bid-pack-{tender.get('tender_ref') or tender['_id']}.pdf"


@app.get("/tenders/{tender_id}/bid-document/link")
def bid_document_link(tender_id: str) -> dict[str, Any]:
    """A short-lived link the browser downloads the bid pack from directly
    (it's too big for one API response). `url` is null when running
    locally without a bucket - GET /tenders/{id}/bid-document serves it then."""
    tender, key = _bid_pack_key_or_404(tender_id)
    return {"url": storage.download_url(key, _bid_pack_filename(tender)), "filename": _bid_pack_filename(tender)}


@app.get("/tenders/{tender_id}/bid-document")
def download_bid_document(tender_id: str) -> Response:
    """The whole pack in one response - for local runs only (see
    bid_document_link); deployed, packs outgrow the Lambda response cap."""
    tender, key = _bid_pack_key_or_404(tender_id)
    content = storage.get_bytes(key)
    if content is None:
        raise HTTPException(status_code=404, detail="No bid pack to download - generate it again.")
    return Response(
        content=content,
        media_type="application/pdf",
        headers={"Content-Disposition": _content_disposition("attachment", _bid_pack_filename(tender))},
    )


PPTX_MEDIA_TYPE = "application/vnd.openxmlformats-officedocument.presentationml.presentation"


def _presentation_key_or_404(tender_id: str) -> tuple[dict[str, Any], str]:
    """The technical presentation generated with the tender's current bid
    pack (see _build_bid_pack) - offered for as long as the pack is."""
    try:
        object_id = ObjectId(tender_id)
    except InvalidId:
        raise HTTPException(status_code=400, detail="Invalid tender id")
    tender = get_collection().find_one(
        {"_id": object_id}, {"bid_pack_key": 1, "presentation_key": 1, "bid_generated_at": 1, "tender_ref": 1})
    if tender is None:
        raise HTTPException(status_code=404, detail="Tender not found")
    if not (tender.get("presentation_key") and _bid_pack_available(tender)):
        raise HTTPException(status_code=404, detail="No presentation to download - generate the bid pack again.")
    return tender, tender["presentation_key"]


def _presentation_filename(tender: dict[str, Any]) -> str:
    return f"presentation-{tender.get('tender_ref') or tender['_id']}.pptx"


@app.get("/tenders/{tender_id}/presentation/link")
def presentation_link(tender_id: str) -> dict[str, Any]:
    """Like /bid-document/link, for the PowerPoint presentation."""
    tender, key = _presentation_key_or_404(tender_id)
    return {"url": storage.download_url(key, _presentation_filename(tender)), "filename": _presentation_filename(tender)}


@app.get("/tenders/{tender_id}/presentation")
def download_presentation(tender_id: str) -> Response:
    """The PowerPoint presentation itself - for local runs (see presentation_link)."""
    tender, key = _presentation_key_or_404(tender_id)
    content = storage.get_bytes(key)
    if content is None:
        raise HTTPException(status_code=404, detail="No presentation to download - generate the bid pack again.")
    return Response(
        content=content,
        media_type=PPTX_MEDIA_TYPE,
        headers={"Content-Disposition": _content_disposition("attachment", _presentation_filename(tender))},
    )


# --- Eligibility criteria ---------------------------------------------------
# Read-only view of the company's bid-eligibility profile (see
# app.processing.eligibility and Eligibility.pdf) - the criteria the
# business must meet and which document types prove each one, kept in
# Mongo. No file storage here; this just surfaces the reference text.

CRITERIA_DOC_ID = "company_eligibility_profile"


@app.get("/eligibility")
def list_eligibility_criteria() -> dict[str, Any]:
    doc = get_eligibility_criteria_collection().find_one({"_id": CRITERIA_DOC_ID})
    criteria = doc.get("criteria", []) if doc else []
    rows = [
        {
            "id": c["id"],
            "criterion": c["criterion"],
            "requirement": c["requirement"],
            "supporting_documents": c["supporting_documents"],
        }
        for c in criteria
    ]
    return {"criteria": rows}


def _slugify(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "criterion"


class CriterionCreateRequest(BaseModel):
    criterion: str
    requirement: str
    supporting_documents: list[str] = []


@app.post("/eligibility")
def create_eligibility_criterion(body: CriterionCreateRequest) -> dict[str, Any]:
    criterion = body.criterion.strip()
    requirement = body.requirement.strip()
    if not criterion or not requirement:
        raise HTTPException(
            status_code=422, detail="Criterion and requirement are both required"
        )
    supporting_documents = [d.strip() for d in body.supporting_documents if d.strip()]

    collection = get_eligibility_criteria_collection()
    existing = collection.find_one({"_id": CRITERIA_DOC_ID})
    existing_ids = {c["id"] for c in (existing or {}).get("criteria", [])}
    base_id = _slugify(criterion)
    new_id = base_id
    suffix = 2
    while new_id in existing_ids:
        new_id = f"{base_id}-{suffix}"
        suffix += 1

    new_criterion = {
        "id": new_id,
        "criterion": criterion,
        "requirement": requirement,
        "supporting_documents": supporting_documents,
        "match_keywords": [],
    }
    collection.update_one(
        {"_id": CRITERIA_DOC_ID},
        {"$push": {"criteria": new_criterion}},
        upsert=True,
    )
    return {
        "id": new_id,
        "criterion": criterion,
        "requirement": requirement,
        "supporting_documents": supporting_documents,
    }


class CriterionUpdateRequest(BaseModel):
    criterion: str
    requirement: str
    supporting_documents: list[str] = []


@app.put("/eligibility/{criterion_id}")
def update_eligibility_criterion(
    criterion_id: str, body: CriterionUpdateRequest
) -> dict[str, Any]:
    criterion = body.criterion.strip()
    requirement = body.requirement.strip()
    if not criterion or not requirement:
        raise HTTPException(
            status_code=422, detail="Criterion and requirement are both required"
        )
    supporting_documents = [d.strip() for d in body.supporting_documents if d.strip()]

    result = get_eligibility_criteria_collection().update_one(
        {"_id": CRITERIA_DOC_ID, "criteria.id": criterion_id},
        {
            "$set": {
                "criteria.$.criterion": criterion,
                "criteria.$.requirement": requirement,
                "criteria.$.supporting_documents": supporting_documents,
            }
        },
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Criterion not found")

    return {
        "id": criterion_id,
        "criterion": criterion,
        "requirement": requirement,
        "supporting_documents": supporting_documents,
    }


@app.delete("/eligibility/{criterion_id}")
def delete_eligibility_criterion(criterion_id: str) -> dict[str, Any]:
    result = get_eligibility_criteria_collection().update_one(
        {"_id": CRITERIA_DOC_ID},
        {"$pull": {"criteria": {"id": criterion_id}}},
    )
    if result.modified_count == 0:
        raise HTTPException(status_code=404, detail="Criterion not found")
    return {"status": "deleted"}


# --- Text sections -----------------------------------------------------------
# The Documents tab's Text sections (see app.text_sections): company text the
# team edits by hand, each with an "as on" date, an optional expiry and the
# kinds of bid document it's used in. Every checklist / bid-pack build reads
# them (_generation_inputs), so an edit applies to the next generated pack.

_TEXT_SECTIONS_SEEDED_ID = "seeded"


def _serialize_text_section(doc: dict[str, Any]) -> dict[str, Any]:
    return {
        "id": str(doc["_id"]),
        "header": doc.get("header", ""),
        "text": doc.get("text", ""),
        "as_on": doc.get("as_on"),
        "valid_until": doc.get("valid_until"),
        "use_in": doc.get("use_in", []),
        "updated_at": _iso(doc.get("updated_at")),
    }


def _load_text_sections() -> list[dict[str, Any]]:
    """Every section, in display order - the defaults (app.text_sections.
    default_sections) are added the first time, and only then, so deleting
    them all leaves the tab empty rather than bringing them back."""
    collection = get_text_sections_collection()
    if collection.find_one({"_id": _TEXT_SECTIONS_SEEDED_ID}) is None:
        now = dt.datetime.now(dt.timezone.utc)
        defaults = [
            {**section, "position": n, "updated_at": now}
            for n, section in enumerate(default_sections(load_company_profile()))
        ]
        if collection.count_documents({"meta": {"$ne": True}}) == 0:
            collection.insert_many(defaults)
        collection.update_one({"_id": _TEXT_SECTIONS_SEEDED_ID}, {"$set": {"meta": True}}, upsert=True)
    docs = collection.find({"meta": {"$ne": True}}).sort([("position", 1), ("_id", 1)])
    return [_serialize_text_section(d) for d in docs]


class TextSectionRequest(BaseModel):
    header: str
    text: str = ""
    as_on: str | None = None
    valid_until: str | None = None
    use_in: list[str] = []


def _clean_text_section(body: TextSectionRequest) -> dict[str, Any]:
    header = body.header.strip()
    if not header:
        raise HTTPException(status_code=422, detail="A section needs a header")
    dates = {}
    for field in ("as_on", "valid_until"):
        value = (getattr(body, field) or "").strip() or None
        try:
            parse_day(value)
        except ValueError:
            raise HTTPException(status_code=422, detail=f"{field} must be a date (YYYY-MM-DD)")
        dates[field] = value
    use_in = [k for k in USE_IN_OPTIONS if k in body.use_in] or ["all"]
    return {
        "header": header,
        "text": body.text.strip(),
        **dates,
        "use_in": use_in,
        "updated_at": dt.datetime.now(dt.timezone.utc),
    }


def _text_section_id(section_id: str) -> ObjectId:
    try:
        return ObjectId(section_id)
    except InvalidId:
        raise HTTPException(status_code=404, detail="Section not found")


@app.get("/text-sections")
def list_text_sections() -> dict[str, Any]:
    return {
        "sections": _load_text_sections(),
        "use_in_options": [{"key": k, "label": v} for k, v in USE_IN_OPTIONS.items()],
    }


@app.post("/text-sections")
def create_text_section(body: TextSectionRequest) -> dict[str, Any]:
    collection = get_text_sections_collection()
    _load_text_sections()  # adds the defaults first, so a new section goes after them
    last = collection.find_one({"meta": {"$ne": True}}, sort=[("position", -1)])
    doc = {**_clean_text_section(body), "position": (last or {}).get("position", -1) + 1}
    doc["_id"] = collection.insert_one(doc).inserted_id
    return _serialize_text_section(doc)


@app.put("/text-sections/{section_id}")
def update_text_section(section_id: str, body: TextSectionRequest) -> dict[str, Any]:
    collection = get_text_sections_collection()
    object_id = _text_section_id(section_id)
    result = collection.update_one({"_id": object_id}, {"$set": _clean_text_section(body)})
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Section not found")
    return _serialize_text_section(collection.find_one({"_id": object_id}))


@app.delete("/text-sections/{section_id}")
def delete_text_section(section_id: str) -> dict[str, Any]:
    result = get_text_sections_collection().delete_one({"_id": _text_section_id(section_id)})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Section not found")
    return {"status": "deleted"}


# --- Documents -----------------------------------------------------------
# A small document library the user manages by hand from the dashboard
# (company certificates, licenses, etc) - upload, rename, delete, view and
# download. Distinct from the `documents` array on a tender, which is
# attachments collected automatically by the pipeline. Each file's details
# are a record in app.database.get_company_documents_collection(); its
# bytes are in file storage (app.storage, under company-documents/).
#
# One request or response can carry at most MAX_DOCUMENT_SIZE: Mangum
# returns the response base64-encoded, which inflates size by ~33%, and a
# synchronous Lambda invoke tops out at 6 MB - 4 MB raw keeps the encoded
# payload safely under that. Files up to MAX_UPLOAD_SIZE therefore travel in
# TRANSFER_PART_SIZE parts instead: uploaded with PUT /uploads/{id}/parts/N
# then a .../finish call (documents library or Sign & Stamp), and downloaded
# with ?part=N. The Sign & Stamp tab alone sets no size limit - only the
# API Lambda's memory bounds what it can mark.
MAX_DOCUMENT_SIZE = 4 * 1024 * 1024
MAX_UPLOAD_SIZE = 10 * 1024 * 1024
TRANSFER_PART_SIZE = 3 * 1024 * 1024
_TRANSFER_KEY_RE = re.compile(r"^[0-9a-f]{32}$")


def _mb(size: int) -> int:
    return size // (1024 * 1024)


def _transfer_key(key: str) -> str:
    if not _TRANSFER_KEY_RE.match(key):
        raise HTTPException(status_code=400, detail="Invalid upload id")
    return key


@app.put("/uploads/{upload_id}/parts/{part}")
async def upload_part(upload_id: str, part: int, request: Request) -> dict[str, Any]:
    """One TRANSFER_PART_SIZE slice of a file for a .../finish call -
    `upload_id` is a random 32-hex id the dashboard picks per file. Kept in
    app.database.get_stamp_transfers_collection() until finished (or for an
    hour at most). Any number of parts is accepted - Sign & Stamp has no
    size limit; the Documents library's .../finish enforces its own (see
    _take_upload)."""
    key = _transfer_key(upload_id)
    if part < 0:
        raise HTTPException(status_code=400, detail="Invalid part number")
    data = await request.body()
    if not data:
        raise HTTPException(status_code=422, detail="Part is empty")
    if len(data) > TRANSFER_PART_SIZE:
        raise HTTPException(status_code=413, detail=f"Parts must be at most {_mb(TRANSFER_PART_SIZE)} MB")
    # In a thread: pymongo blocks, and the dashboard sends several parts at
    # once - on the event loop they'd be stored one after another.
    await run_in_threadpool(
        get_stamp_transfers_collection().replace_one,
        {"kind": "upload", "key": key, "part": part},
        {"kind": "upload", "key": key, "part": part, "data": data, "created_at": dt.datetime.now(dt.timezone.utc)},
        upsert=True,
    )
    return {"part": part, "size": len(data)}


def _take_upload(upload_id: str, parts: int, max_size: int | None = MAX_UPLOAD_SIZE, keep: bool = False) -> bytes:
    """An upload's `parts` joined back into the file - its parts deleted, or
    with `keep` left for reuse (their hour restarted). 409 when any is
    missing (never sent, or expired), 413 over `max_size` (None: no limit)."""
    key = _transfer_key(upload_id)
    transfers = get_stamp_transfers_collection()
    stored = {d["part"]: d["data"] for d in transfers.find({"kind": "upload", "key": key})}
    if sorted(stored) != list(range(parts)):
        raise HTTPException(status_code=409, detail="The upload is incomplete or has expired - upload the file again")
    if keep:
        transfers.update_many({"kind": "upload", "key": key}, {"$set": {"created_at": dt.datetime.now(dt.timezone.utc)}})
    else:
        transfers.delete_many({"kind": "upload", "key": key})
    content = b"".join(stored[i] for i in range(parts))
    if max_size is not None and len(content) > max_size:
        raise HTTPException(status_code=413, detail=f"File exceeds the {_mb(max_size)} MB limit")
    return content


def _serialize_document(record: dict[str, Any]) -> dict[str, Any]:
    metadata = record.get("metadata") or {}
    return {
        "id": str(record["_id"]),
        "name": metadata.get("display_name") or record["filename"],
        "filename": record["filename"],
        "content_type": metadata.get("content_type") or "application/octet-stream",
        "size": record.get("length"),
        "uploaded_at": _iso(record.get("uploadDate")),
        "mark_kind": metadata.get("mark_kind"),
    }


def _document_key(document_id: object, filename: str) -> str:
    # The filename is kept for readability in the bucket; slashes would
    # otherwise add path levels.
    safe = re.sub(r"[\\/]+", "_", filename).strip() or "document"
    return f"{storage.COMPANY_DOCUMENTS_PREFIX}{document_id}/{safe}"


def _document_bytes(key: str) -> bytes:
    content = storage.get_bytes(key)
    if content is None:
        raise HTTPException(status_code=404, detail="Document file not found in storage")
    return content


def _content_disposition(kind: str, filename: str) -> str:
    # RFC 5987: an ASCII fallback for older clients plus a UTF-8 filename*
    # for everything else - filename is user-supplied, so both are also
    # stripped of control characters (incl. CR/LF) to avoid header injection.
    ascii_fallback = re.sub(r"[^\x20-\x7e]", "_", filename).replace('"', "'") or "document"
    return f'{kind}; filename="{ascii_fallback}"; filename*=UTF-8\'\'{quote(filename)}'


def _document_object_id(document_id: str) -> ObjectId:
    try:
        return ObjectId(document_id)
    except InvalidId:
        raise HTTPException(status_code=400, detail="Invalid document id")


def _get_document_or_404(document_id: str) -> dict[str, Any]:
    record = get_company_documents_collection().find_one({"_id": _document_object_id(document_id)})
    if record is None:
        raise HTTPException(status_code=404, detail="Document not found")
    return record


def _check_document_content(content: bytes, max_size: int) -> None:
    if not content:
        raise HTTPException(status_code=422, detail="File is empty")
    if len(content) > max_size:
        raise HTTPException(status_code=413, detail=f"File exceeds the {_mb(max_size)} MB limit")


# What a document is used as on the Sign & Stamp tab when it was added there
# as one (metadata.mark_kind) - see _mark_candidates. NOT_A_MARK: removed
# from Sign & Stamp (DELETE /stamp-document/marks/{id}) - still in the
# library, but never offered as a mark, whatever it's named.
MarkKind = Literal["letterhead", "signature", "stamp"]
NOT_A_MARK = "none"


def _store_document(
    content: bytes, filename: str | None, name: str | None, content_type: str | None, mark_kind: str | None = None
) -> dict[str, Any]:
    display_name = (name or filename or "Untitled").strip() or "Untitled"
    metadata = {"display_name": display_name, "content_type": content_type or "application/octet-stream"}
    if mark_kind:
        metadata["mark_kind"] = mark_kind
    document_id = ObjectId()
    stored_filename = filename or display_name
    key = _document_key(document_id, stored_filename)
    storage.put_bytes(key, content, metadata["content_type"])
    record = {
        "_id": document_id,
        "filename": stored_filename,
        "length": len(content),
        "uploadDate": dt.datetime.now(dt.timezone.utc),
        "s3_key": key,
        "metadata": metadata,
    }
    get_company_documents_collection().insert_one(record)
    return _serialize_document(record)


def _replace_document(
    document_id: str,
    content: bytes,
    filename: str | None,
    name: str | None,
    content_type: str | None,
    mark_kind: str | None = None,
) -> dict[str, Any]:
    """Swap out a document's file while keeping it as the same row in the
    list - same id, so checklist rows and Sign & Stamp choices that point
    at it keep working - as opposed to DELETE+POST, which would also lose
    the display name unless the caller re-types it. The new file is stored
    before the old one is deleted, so a failed upload never leaves the
    document missing its content.
    """
    existing = _get_document_or_404(document_id)
    existing_meta = existing.get("metadata") or {}
    stored_filename = filename or existing["filename"]
    key = _document_key(existing["_id"], stored_filename)
    new_content_type = content_type or "application/octet-stream"
    storage.put_bytes(key, content, new_content_type)
    metadata = {
        **existing_meta,
        "display_name": (name or existing_meta.get("display_name") or stored_filename).strip(),
        "content_type": new_content_type,
    }
    if mark_kind:
        metadata["mark_kind"] = mark_kind
    update = {
        "filename": stored_filename,
        "length": len(content),
        "uploadDate": dt.datetime.now(dt.timezone.utc),
        "s3_key": key,
        "metadata": metadata,
    }
    get_company_documents_collection().update_one({"_id": existing["_id"]}, {"$set": update})
    if existing.get("s3_key") and existing["s3_key"] != key:
        storage.delete(existing["s3_key"])
    return _serialize_document({**existing, **update})


@app.get("/documents")
def list_documents() -> dict[str, Any]:
    cursor = get_company_documents_collection().find(sort=[("uploadDate", DESCENDING)])
    return {"documents": [_serialize_document(d) for d in cursor]}


@app.post("/documents")
async def upload_document(
    file: UploadFile = File(...), name: str | None = Form(None)
) -> dict[str, Any]:
    """One-request upload, up to MAX_DOCUMENT_SIZE - the dashboard uses
    PUT /uploads/{id}/parts/N + POST /documents/uploads/{id}/finish."""
    content = await file.read()
    _check_document_content(content, MAX_DOCUMENT_SIZE)
    return _store_document(content, file.filename, name, file.content_type)


@app.post("/documents/uploads/{upload_id}/finish")
def finish_document_upload(
    upload_id: str,
    parts: int = Form(..., ge=1),
    name: str | None = Form(None),
    filename: str | None = Form(None),
    content_type: str | None = Form(None),
    replace: str | None = Form(None),
    mark_kind: MarkKind | None = Form(None),
) -> dict[str, Any]:
    """Stores an upload sent in parts (up to MAX_UPLOAD_SIZE) as a new
    document - or, with `replace` (a document id), as that document's new
    file (see PUT /documents/{id}/replace). `mark_kind`: added from the Sign
    & Stamp tab as a letterhead/signature/stamp (a replaced file keeps the
    one it had)."""
    content = _take_upload(upload_id, parts)
    _check_document_content(content, MAX_UPLOAD_SIZE)
    if replace:
        return _replace_document(replace, content, filename, name, content_type, mark_kind)
    return _store_document(content, filename, name, content_type, mark_kind)


class DocumentRenameRequest(BaseModel):
    name: str
    mark_kind: MarkKind | None = None  # also files it as a Sign & Stamp letterhead/signature/stamp


@app.put("/documents/{document_id}")
def rename_document(document_id: str, body: DocumentRenameRequest) -> dict[str, Any]:
    name = body.name.strip()
    if not name:
        raise HTTPException(status_code=422, detail="Name is required")
    object_id = _document_object_id(document_id)

    update = {"metadata.display_name": name}
    if body.mark_kind:
        update["metadata.mark_kind"] = body.mark_kind
    documents = get_company_documents_collection()
    result = documents.update_one({"_id": object_id}, {"$set": update})
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Document not found")
    return _serialize_document(documents.find_one({"_id": object_id}))


@app.put("/documents/{document_id}/replace")
async def replace_document(
    document_id: str, file: UploadFile = File(...), name: str | None = Form(None)
) -> dict[str, Any]:
    """One-request replace, up to MAX_DOCUMENT_SIZE - see _replace_document."""
    content = await file.read()
    _check_document_content(content, MAX_DOCUMENT_SIZE)
    return _replace_document(document_id, content, file.filename, name, file.content_type)


@app.delete("/documents/{document_id}")
def delete_document(document_id: str) -> dict[str, Any]:
    record = _get_document_or_404(document_id)
    get_company_documents_collection().delete_one({"_id": record["_id"]})
    storage.delete(record["s3_key"])
    return {"status": "deleted"}


@app.get("/documents/{document_id}/info")
def document_info(document_id: str) -> dict[str, Any]:
    """How many ?part=N requests /download and /view need for this file."""
    record = _get_document_or_404(document_id)
    return {**_serialize_document(record), "part_size": TRANSFER_PART_SIZE,
            "parts": max(1, -(-record["length"] // TRANSFER_PART_SIZE))}


def _document_response(document_id: str, kind: str, part: int | None) -> Response:
    """The whole document (small files only - see MAX_DOCUMENT_SIZE) or, with
    `part`, one TRANSFER_PART_SIZE slice of it."""
    record = _get_document_or_404(document_id)
    content_type = (record.get("metadata") or {}).get("content_type") or "application/octet-stream"
    content = _document_bytes(record["s3_key"])
    if part is not None:
        if part < 0 or part * TRANSFER_PART_SIZE >= max(len(content), 1):
            raise HTTPException(status_code=416, detail="Part out of range")
        content = content[part * TRANSFER_PART_SIZE:(part + 1) * TRANSFER_PART_SIZE]
        content_type = "application/octet-stream"
    return Response(
        content=content,
        media_type=content_type,
        headers={"Content-Disposition": _content_disposition(kind, record["filename"])},
    )


@app.get("/documents/{document_id}/download")
def download_document(document_id: str, part: int | None = None) -> Response:
    return _document_response(document_id, "attachment", part)


@app.get("/documents/{document_id}/view")
def view_document(document_id: str, part: int | None = None) -> Response:
    return _document_response(document_id, "inline", part)


# --- Sign & Stamp ------------------------------------------------------------
# The dashboard's Sign & Stamp tab: upload one or more PDFs/images and get
# them back with the letterhead, signature and/or seal on every page, placed
# the same way a bid pack places them (app.reports.bid_generator.mark_document)
# - several files come back combined into one PDF. Nothing is kept: files
# (of any size) pass through temporary parts only
# because one Lambda request/response can't carry them, and those parts
# expire an hour after they were last used; the marked result sits under bid-packs/
# (deleted after a day, see template.yaml) only for the browser to download.

# Which Documents-library entries are offered as each mark, by display name
# or filename: "OAKS_LetterHead" / "SIV Letter Head.pdf", "Vijaykumari_sign" /
# "Suman_Signature", "SIV_Stamp" / "Company Seal". The signature and seal
# named in company_profile.json's authorized_signatory always count too.
_MARK_NAME_PATTERNS = {
    "letterhead": re.compile(r"letter[\s_-]*head", re.IGNORECASE),
    "signature": re.compile(r"sign", re.IGNORECASE),
    "stamp": re.compile(r"stamp|seal", re.IGNORECASE),
}
# `letterhead=default` etc. (or `true`): the letterhead configured in
# company_profile.json, or its authorized_signatory's signature/seal.
DEFAULT_MARK = "default"


def _is_image_doc(doc: CompanyDocumentRef) -> bool:
    return doc.content_type.startswith("image/") or bool(
        re.search(r"\.(png|jpe?g|gif|bmp|webp|tiff?)$", doc.filename, re.IGNORECASE))


def _is_pdf_doc(doc: CompanyDocumentRef) -> bool:
    return doc.content_type == "application/pdf" or doc.filename.lower().endswith(".pdf")


def _mark_candidates(
    company_profile: dict[str, Any], documents: list[CompanyDocumentRef]
) -> dict[str, list[CompanyDocumentRef]]:
    """The library documents offered for each mark, in the library's order.
    Signatures/stamps must be images (drawn over the page); a letterhead can
    also be a PDF - its first page is used (see _pdf_first_page_as_image)."""
    signatory = company_profile.get("authorized_signatory", {})
    configured = {
        "signature": (signatory.get("signature_document_name") or "").strip().lower(),
        "stamp": (signatory.get("seal_document_name") or "").strip().lower(),
    }
    out: dict[str, list[CompanyDocumentRef]] = {kind: [] for kind in _MARK_NAME_PATTERNS}
    for doc in documents:
        if doc.mark_kind == NOT_A_MARK:
            continue  # removed from Sign & Stamp - kept in the library only
        for kind, pattern in _MARK_NAME_PATTERNS.items():
            if doc.mark_kind:  # added on the Sign & Stamp tab as this kind
                if doc.mark_kind != kind:
                    continue
            else:
                named = bool(pattern.search(doc.name) or pattern.search(doc.filename))
                if not (named or doc.name.strip().lower() == configured.get(kind)):
                    continue
            if _is_image_doc(doc) or (kind == "letterhead" and _is_pdf_doc(doc)):
                out[kind].append(doc)
    return out


def _pdf_first_page_as_image(data: bytes) -> bytes:
    """A PDF letterhead's first page as a 200 dpi JPEG - so it's laid over
    pages and images exactly like an image letterhead."""
    import pypdfium2  # only needed for PDF letterheads

    try:
        pdf = pypdfium2.PdfDocument(data)
        image = pdf[0].render(scale=200 / 72).to_pil().convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=422, detail=f"Couldn't read the letterhead PDF: {exc}") from exc
    out = io.BytesIO()
    image.save(out, format="JPEG", quality=92)
    return out.getvalue()


def _stamp_asset(kind: str, choice: str, company_profile: dict[str, Any]) -> tuple[bytes, str]:
    """(image bytes, content type) of the chosen letterhead, signature or
    stamp - `choice` is a library document id from GET /stamp-document/marks,
    or DEFAULT_MARK. A 422 naming what to upload/configure when it isn't there."""
    if choice == DEFAULT_MARK:
        if kind == "letterhead":
            path = _letterhead_path(company_profile)
            if path is None:
                raise HTTPException(status_code=422, detail="No letterhead image is configured (company_profile.json's letterhead_image)")
            return path.read_bytes(), "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
        signatory = company_profile.get("authorized_signatory", {})
        sig_doc, seal_doc = signing_assets(company_profile, _load_company_documents_for_generation())
        doc, field = (sig_doc, "signature_document_name") if kind == "signature" else (seal_doc, "seal_document_name")
        if doc is None:
            raise HTTPException(status_code=422, detail=(
                f"No {kind} found - upload it on the Documents tab as '{signatory.get(field) or f'the {kind}'}'"))
    else:
        candidates = _mark_candidates(company_profile, _load_company_documents_for_generation())[kind]
        doc = next((d for d in candidates if d.id == choice), None)
        if doc is None:
            raise HTTPException(status_code=422, detail=f"That {kind} is no longer in the Documents library")
    data = doc.open_bytes()
    if _is_pdf_doc(doc):
        return _pdf_first_page_as_image(data), "image/jpeg"
    return data, doc.content_type


@app.get("/stamp-document/marks")
def stamp_marks() -> dict[str, Any]:
    """The Sign & Stamp tab's buttons: for each mark, the library documents
    that can be used, by their Documents-tab names. The configured
    letterhead image is offered only when the library has none."""
    company_profile = load_company_profile()
    candidates = _mark_candidates(company_profile, _load_company_documents_for_generation())
    marks = {kind: [{"id": d.id, "label": d.name} for d in docs] for kind, docs in candidates.items()}
    if not marks["letterhead"] and _letterhead_path(company_profile) is not None:
        marks["letterhead"] = [{"id": DEFAULT_MARK, "label": "Letterhead"}]
    return marks


@app.delete("/stamp-document/marks/{document_id}")
def remove_stamp_mark(document_id: str) -> dict[str, Any]:
    """Takes a document out of the Sign & Stamp tab's letterheads/signatures/
    stamps without deleting it - it stays in the Documents library (DELETE
    /documents/{id} deletes it). Adding it back: rename it from the Sign &
    Stamp popup, or PUT /documents/{id} with a mark_kind."""
    result = get_company_documents_collection().update_one(
        {"_id": _document_object_id(document_id)}, {"$set": {"metadata.mark_kind": NOT_A_MARK}}
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"status": "removed"}


@app.get("/stamp-document/assets/{kind}")
def stamp_asset(kind: Literal["letterhead", "signature", "stamp"], id: str = DEFAULT_MARK) -> Response:
    """A mark's image, for the Sign & Stamp tab's preview (a PDF letterhead
    as an image of its first page)."""
    data, content_type = _stamp_asset(kind, id, load_company_profile())
    return Response(content=data, media_type=content_type, headers={"Cache-Control": "private, max-age=300"})


class MarkPlacementModel(BaseModel):
    page: int = Field(ge=0)
    mark: Literal["signature", "stamp"]
    x: float = Field(ge=-1, le=2)
    y: float = Field(ge=-1, le=2)
    w: float = Field(gt=0, le=2)
    h: float = Field(gt=0, le=2)


_PLACEMENTS = TypeAdapter(list[MarkPlacementModel])


class ContentLayoutModel(BaseModel):
    page: int = Field(ge=0)
    scale: float = Field(gt=0, le=1)
    dx: float = Field(ge=-1, le=1)
    dy: float = Field(ge=-1, le=1)


_LAYOUTS = TypeAdapter(list[ContentLayoutModel])


class PageMarksModel(BaseModel):
    """One page's marks - each a library document id (see GET
    /stamp-document/marks) or DEFAULT_MARK; missing/null means none."""

    page: int = Field(ge=0)
    letterhead: str | None = None
    signature: str | None = None
    stamp: str | None = None


_PAGE_MARKS = TypeAdapter(list[PageMarksModel])


@app.post("/stamp-document")
async def stamp_document(
    file: UploadFile = File(...),
    letterhead: str | None = Form(None),
    signature: str | None = Form(None),
    stamp: str | None = Form(None),
    placements: str | None = Form(None),
    layouts: str | None = Form(None),
    pages: str | None = Form(None),
) -> Response:
    """One-request version for small files (up to MAX_DOCUMENT_SIZE, both
    ways) - the dashboard uses the part-by-part upload below instead.
    `placements` (JSON, optional): where the user dragged each signature/
    stamp in the preview - see app.reports.bid_generator.MarkPlacement.
    Without it every page gets the marks at their usual bottom-right spot.
    `layouts` (JSON, optional): how the preview moved pages whose content
    would run into the letterhead - see ContentLayout."""
    options = _stamp_options(letterhead, signature, stamp, placements, layouts, pages)
    content = await file.read()
    marked, media_type, out_name = _run_stamp(content, file.filename or "document", file.content_type or "",
                                              options, max_size=MAX_DOCUMENT_SIZE)
    return Response(
        content=marked,
        media_type=media_type,
        headers={"Content-Disposition": _content_disposition("attachment", out_name)},
    )


# The dashboard sends each file (no size limit) in parts (PUT
# /uploads/{id}/parts/N - see the Documents section), then POST
# /stamp-document/jobs marks them. Deployed, marking large files can outlast
# TenderHttpApi's 30s limit, so - as with POST /tenders/{id}/generate-bid -
# ApiFunction does it in an async invocation of itself ({"stamp_job": id},
# see handler) while the dashboard polls GET /stamp-document/jobs/{id}. The
# result is downloaded straight from file storage through a short-lived link.
STAMP_MAX_FILES = 50
_STAMP_JOB_TIMEOUT = dt.timedelta(minutes=6)  # ApiFunction's Timeout plus a margin


class StampFileModel(BaseModel):
    """One file of a POST /stamp-document/jobs - an upload sent in parts."""

    upload_id: str
    parts: int = Field(ge=1)
    filename: str = "document"
    content_type: str = ""


_STAMP_FILES = TypeAdapter(list[StampFileModel])


@app.post("/stamp-document/jobs")
def start_stamp_job(
    files: str = Form(...),
    letterhead: str | None = Form(None),
    signature: str | None = Form(None),
    stamp: str | None = Form(None),
    placements: str | None = Form(None),
    layouts: str | None = Form(None),
    pages: str | None = Form(None),
    as_pdf: bool = Form(False),
) -> dict[str, Any]:
    """Marks the uploads in `files` (JSON, see StampFileModel - in order) with
    the same options as POST /stamp-document, page numbers counting through
    all the files in turn. One file comes back as it went in (a PDF for a
    PDF, an image for an image - or, with `as_pdf`, always a PDF); several,
    as one combined PDF. Answers with
    GET /stamp-document/jobs/{id}'s body - "done" already when run locally."""
    form = {"letterhead": letterhead, "signature": signature, "stamp": stamp,
            "placements": placements, "layouts": layouts, "pages": pages}
    _stamp_options(**form)  # refused now rather than in the background
    try:
        uploads = _STAMP_FILES.validate_json(files)
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=f"Invalid files: {exc.errors()[0]['msg']}")
    if not uploads:
        raise HTTPException(status_code=422, detail="Choose at least one file")
    if len(uploads) > STAMP_MAX_FILES:
        raise HTTPException(status_code=422, detail=f"Choose at most {STAMP_MAX_FILES} files at a time")
    for upload in uploads:
        _transfer_key(upload.upload_id)

    job_id = uuid.uuid4().hex
    transfers = get_stamp_transfers_collection()
    transfers.insert_one({
        "kind": "job", "key": job_id, "status": "queued", "files": [u.model_dump() for u in uploads],
        "form": form, "as_pdf": as_pdf, "created_at": dt.datetime.now(dt.timezone.utc),
    })
    function_name = os.environ.get("AWS_LAMBDA_FUNCTION_NAME")
    if not function_name:
        _run_stamp_job(job_id)
    else:
        try:
            import boto3

            boto3.client("lambda").invoke(
                FunctionName=function_name,
                InvocationType="Event",
                Payload=json.dumps({"stamp_job": job_id}).encode(),
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Could not start Sign & Stamp job %s: %s", job_id, exc)
            transfers.delete_one({"kind": "job", "key": job_id})
            raise HTTPException(status_code=503, detail="Couldn't start marking the files - try again.")
    return _stamp_job_status(transfers.find_one({"kind": "job", "key": job_id}))


@app.get("/stamp-document/jobs/{job_id}")
def stamp_job_status(job_id: str) -> dict[str, Any]:
    job = get_stamp_transfers_collection().find_one({"kind": "job", "key": _transfer_key(job_id)})
    if job is None:
        raise HTTPException(status_code=404, detail="This job has expired - mark the files again")
    return _stamp_job_status(job)


@app.get("/stamp-document/jobs/{job_id}/result")
def download_stamp_result(job_id: str) -> Response:
    """The marked file in one response - for local runs only; deployed, the
    status's `url` links straight to file storage (results outgrow the
    Lambda response cap)."""
    job = get_stamp_transfers_collection().find_one({"kind": "job", "key": _transfer_key(job_id)})
    result = (job or {}).get("result")
    content = storage.get_bytes(result["key"]) if result else None
    if content is None:
        raise HTTPException(status_code=404, detail="This result has expired - mark the files again")
    return Response(content=content, media_type=result["media_type"],
                    headers={"Content-Disposition": _content_disposition("attachment", result["filename"])})


def _stamp_job_status(job: dict[str, Any]) -> dict[str, Any]:
    """{"job_id", "status": "queued" | "running" | "done" | "failed", "message"},
    plus the result's filename/media_type/size and a download `url` (null
    locally - GET .../result serves it then) once it's "done"."""
    out = {"job_id": job["key"], "status": job["status"], "message": job.get("message")}
    if job["status"] in ("queued", "running"):
        created = job["created_at"]
        if created.tzinfo is None:
            created = created.replace(tzinfo=dt.timezone.utc)
        if dt.datetime.now(dt.timezone.utc) - created > _STAMP_JOB_TIMEOUT:
            return {**out, "status": "failed", "message": "Marking the files took too long and was stopped - try again."}
    if job["status"] == "done":
        result = job["result"]
        out.update(filename=result["filename"], media_type=result["media_type"], size=result["size"],
                   url=storage.download_url(result["key"], result["filename"]))
    return out


def _run_stamp_job(job_id: str) -> None:
    """The background half of POST /stamp-document/jobs. Never raises - the
    outcome is recorded on the job for GET .../jobs/{id} to report. Only a
    "queued" job is picked up, so a Lambda retry of the invocation does nothing."""
    transfers = get_stamp_transfers_collection()
    job_filter = {"kind": "job", "key": job_id}
    try:
        job = transfers.find_one_and_update({**job_filter, "status": "queued"}, {"$set": {"status": "running"}})
    except Exception:  # noqa: BLE001 - the status times out on its own
        logger.exception("Could not pick up Sign & Stamp job %s", job_id)
        return
    if job is None:
        logger.warning("Skipping Sign & Stamp job %s - already picked up or expired", job_id)
        return
    try:
        options = _stamp_options(**job["form"])
        files = []
        for f in job["files"]:
            # Kept: the dashboard reuses a file's upload when it's marked again
            # (other marks, another download) - no 100 MB re-upload each time.
            content = _take_upload(f["upload_id"], f["parts"], max_size=None, keep=True)
            files.append((content, f["filename"] or "document", f["content_type"] or ""))
        marked, media_type, out_name = _run_stamp_files(files, options, as_pdf=bool(job.get("as_pdf")))
        safe_name = re.sub(r"[\\/]+", "_", out_name)
        key = f"{storage.BID_PACKS_PREFIX}stamped/{job_id}/{safe_name}"
        storage.put_bytes(key, marked, media_type)
        transfers.update_one(job_filter, {"$set": {
            "status": "done",
            "result": {"key": key, "filename": out_name, "media_type": media_type, "size": len(marked)},
        }})
    except Exception as exc:  # noqa: BLE001
        if isinstance(exc, HTTPException):
            message = str(exc.detail)
        else:
            logger.exception("Sign & Stamp job %s failed", job_id)
            message = "Marking the files failed unexpectedly - try again."
        try:
            transfers.update_one(job_filter, {"$set": {"status": "failed", "message": message}})
        except Exception:  # noqa: BLE001 - the status times out on its own
            logger.exception("Could not record the failure of Sign & Stamp job %s", job_id)


# A Word file dropped on Sign & Stamp is first made a PDF here - laid out as
# Word does (app.convert) - and from then on previewed, marked and downloaded
# like any other PDF. Deployed, DocxConvertFunction (its own image, with
# LibreOffice) converts it in an async invocation and writes a status file
# next to the result; locally, soffice runs in the request. The dashboard
# polls GET /stamp-document/convert/{id}, then fetches the PDF with
# GET .../result?part=N (no S3 link: the page fetches it, not the browser's
# download, and the bucket has no CORS for that).
_CONVERT_TIMEOUT = dt.timedelta(minutes=4)  # DocxConvertFunction's Timeout plus a margin


class ConvertFileModel(BaseModel):
    upload_id: str
    parts: int = Field(ge=1)
    filename: str = "document.docx"


def _convert_keys(job_id: str) -> dict[str, str]:
    base = f"{storage.BID_PACKS_PREFIX}converted/{job_id}/"
    return {"source_key": f"{base}source", "result_key": f"{base}result.pdf", "status_key": f"{base}status.json"}


@app.post("/stamp-document/convert")
def start_word_conversion(body: ConvertFileModel) -> dict[str, Any]:
    """Starts converting an upload (sent in parts) of a Word document to PDF -
    answers with GET /stamp-document/convert/{id}'s body."""
    if not is_word_document(body.filename):
        raise HTTPException(status_code=422, detail="Only Word documents (.docx, .doc, .odt, .rtf) can be converted")
    # Kept: "Edit text" reads and edits this same upload (see edit_word_text).
    return _start_conversion(_take_upload(body.upload_id, body.parts, max_size=None, keep=True), body.filename)


def _start_conversion(content: bytes, filename: str) -> dict[str, Any]:
    job_id = uuid.uuid4().hex
    keys = _convert_keys(job_id)
    filename = re.sub(r"[\\/]+", "_", filename)
    transfers = get_stamp_transfers_collection()
    transfers.insert_one({"kind": "convert", "key": job_id, "filename": filename,
                          "created_at": dt.datetime.now(dt.timezone.utc), **keys})
    function_name = os.environ.get("DOCX_CONVERT_FUNCTION_NAME")
    if not function_name:  # local: convert right here
        try:
            pdf = convert_to_pdf(content, filename)
        except ConversionError as exc:
            storage.put_text(keys["status_key"], json.dumps({"status": "failed", "message": str(exc)}))
        else:
            storage.put_bytes(keys["result_key"], pdf, "application/pdf")
            storage.put_text(keys["status_key"], json.dumps({"status": "done", "size": len(pdf)}))
    else:
        storage.put_bytes(keys["source_key"], content)
        try:
            import boto3

            boto3.client("lambda").invoke(
                FunctionName=function_name,
                InvocationType="Event",
                Payload=json.dumps({"bucket": get_settings().files_bucket, "filename": filename, **keys}).encode(),
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("Could not start converting %s: %s", filename, exc)
            transfers.delete_one({"kind": "convert", "key": job_id})
            raise HTTPException(status_code=503, detail="Couldn't start converting the Word file - try again.")
    return _convert_status(transfers.find_one({"kind": "convert", "key": job_id}))


def _convert_job(job_id: str) -> dict[str, Any]:
    job = get_stamp_transfers_collection().find_one({"kind": "convert", "key": _transfer_key(job_id)})
    if job is None:
        raise HTTPException(status_code=404, detail="This conversion has expired - add the Word file again")
    return job


def _convert_status(job: dict[str, Any]) -> dict[str, Any]:
    """{"job_id", "status": "running" | "done" | "failed", "message"}, plus the
    PDF's `filename`, `size` and how many ?part=N requests fetch it once "done"."""
    out = {"job_id": job["key"], "status": "running", "message": None}
    text = storage.get_text(job["status_key"])
    if text is None:
        created = job["created_at"]
        if created.tzinfo is None:
            created = created.replace(tzinfo=dt.timezone.utc)
        if dt.datetime.now(dt.timezone.utc) - created > _CONVERT_TIMEOUT:
            return {**out, "status": "failed", "message": "Converting the Word file took too long - try again."}
        return out
    result = json.loads(text)
    if result.get("status") != "done":
        return {**out, "status": "failed", "message": result.get("message") or "Converting the Word file failed."}
    return {**out, "status": "done", "filename": f"{Path(job['filename']).stem or 'document'}.pdf",
            "size": result["size"], "parts": max(1, -(-result["size"] // TRANSFER_PART_SIZE))}


class ParagraphChangeModel(BaseModel):
    """Changes to one paragraph (see app.convert.word_text.ParagraphChange) -
    a field left out stays as it is."""

    text: str | None = None
    bold: bool | None = None
    italic: bool | None = None
    underline: bool | None = None
    align: Literal["left", "center", "right", "justify"] | None = None
    size: float | None = Field(None, ge=1, le=400)
    delete: bool = False


class NewParagraphModel(ParagraphChangeModel):
    after: str  # a paragraph id, or an earlier new paragraph's key
    key: str = ""


class WordEditModel(ConvertFileModel):
    """POST /stamp-document/word/edit - every change so far, by paragraph id
    (from POST /stamp-document/word/paragraphs), against the original upload."""

    edits: dict[str, ParagraphChangeModel] = Field(default_factory=dict, max_length=5000)
    inserts: list[NewParagraphModel] = Field(default_factory=list, max_length=2000)


def _word_upload(body: ConvertFileModel) -> bytes:
    if Path(body.filename).suffix.lower() != ".docx":
        raise HTTPException(status_code=422, detail="Only .docx files can be edited - save it as .docx in Word first")
    return _take_upload(body.upload_id, body.parts, max_size=None, keep=True)


@app.post("/stamp-document/word/paragraphs")
def word_paragraphs(body: ConvertFileModel) -> dict[str, Any]:
    """The text of a .docx upload, paragraph by paragraph, for "Edit text"."""
    try:
        return {"paragraphs": read_paragraphs(_word_upload(body))}
    except WordTextError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@app.post("/stamp-document/word/edit")
def edit_word_text(body: WordEditModel) -> dict[str, Any]:
    """The .docx upload with `edits` and `inserts` made (everything else in
    the file kept - see app.convert.word_text), converted to PDF again: answers like
    POST /stamp-document/convert."""
    try:
        edited = apply_edits(
            _word_upload(body),
            {key: ParagraphChange(**change.model_dump()) for key, change in body.edits.items()},
            [NewParagraph(**new.model_dump()) for new in body.inserts],
        )
    except WordTextError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    return _start_conversion(edited, body.filename)


@app.get("/stamp-document/convert/{job_id}")
def word_conversion_status(job_id: str) -> dict[str, Any]:
    return _convert_status(_convert_job(job_id))


@app.get("/stamp-document/convert/{job_id}/result")
def word_conversion_result(job_id: str, part: int = Query(0, ge=0)) -> Response:
    """One TRANSFER_PART_SIZE slice of the converted PDF."""
    job = _convert_job(job_id)
    content = storage.get_bytes(job["result_key"])
    if content is None:
        raise HTTPException(status_code=404, detail="This conversion has expired - add the Word file again")
    if part * TRANSFER_PART_SIZE >= len(content):
        raise HTTPException(status_code=416, detail="Part out of range")
    return Response(content=content[part * TRANSFER_PART_SIZE:(part + 1) * TRANSFER_PART_SIZE],
                    media_type="application/octet-stream")


def _options_for_pages(options: dict[str, Any], start: int, count: int) -> dict[str, Any]:
    """`options` (see _stamp_options) for one of several files - the pages
    `start` .. `start + count - 1` of them all, numbered from 0 again."""
    def within(page: int) -> bool:
        return start <= page < start + count

    return {
        **options,
        "pages": None if options["pages"] is None
        else {page - start: marks for page, marks in options["pages"].items() if within(page)},
        "placements": None if options["placements"] is None
        else [dataclasses.replace(p, page=p.page - start) for p in options["placements"] if within(p.page)],
        "layouts": None if options["layouts"] is None
        else [dataclasses.replace(lo, page=lo.page - start) for lo in options["layouts"] if within(lo.page)],
    }


def _run_stamp_files(
    files: list[tuple[bytes, str, str]], options: dict[str, Any], as_pdf: bool = False
) -> tuple[bytes, str, str]:
    """Marks each (content, filename, content type) per `options`, its pages
    numbered on from the previous file's - (bytes, media type, download
    filename). One file comes back as _run_stamp returns it (a marked image
    as a one-page PDF with `as_pdf`); several are combined into one PDF."""
    if len(files) == 1:
        content, filename, content_type = files[0]
        marked, media_type, out_name = _run_stamp(content, filename, content_type, options, max_size=None)
        if as_pdf and media_type != "application/pdf":
            try:
                marked = combine_marked([(marked, media_type)])
            except Exception as exc:  # noqa: BLE001
                raise HTTPException(status_code=422, detail=f"Couldn't make a PDF of '{filename}': {exc}") from exc
            media_type, out_name = "application/pdf", f"{Path(out_name).stem}.pdf"
        return marked, media_type, out_name
    assets: dict[tuple[str, str], bytes] = {}  # shared, so each mark is read once for all the files
    marked, start = [], 0
    for content, filename, content_type in files:
        try:
            count = document_page_count(content, filename, content_type)
        except BidGenerationError as exc:
            raise HTTPException(status_code=422, detail=str(exc))
        data, media_type, _ = _run_stamp(content, filename, content_type, _options_for_pages(options, start, count),
                                         max_size=None, assets=assets)
        marked.append((data, media_type))
        start += count
    try:
        combined = combine_marked(marked)
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=422, detail=f"Couldn't combine the files: {exc}") from exc
    stem = Path(files[0][1]).stem or "documents"
    return combined, "application/pdf", f"{stem}-and-{len(files) - 1}-more-signed.pdf"


def _mark_choice(value: str | None) -> str | None:
    """A letterhead/signature/stamp form field: a library document id,
    DEFAULT_MARK (also "true"), or None when unset/"false"."""
    value = (value or "").strip()
    if value.lower() in ("", "false", "0"):
        return None
    return DEFAULT_MARK if value.lower() in ("true", "1", DEFAULT_MARK) else value


def _stamp_options(
    letterhead: str | None,
    signature: str | None,
    stamp: str | None,
    placements: str | None,
    layouts: str | None,
    pages: str | None = None,
) -> dict[str, Any]:
    """The request's marking options, validated before any file is read -
    each mark's choice (see _mark_choice), `placements`/`layouts` parsed
    into bid_generator's types. `pages` (JSON, see PageMarksModel) gives
    each listed page its own marks instead of the same letterhead/
    signature/stamp on every page; unlisted pages are left untouched."""
    letterhead, signature, stamp = _mark_choice(letterhead), _mark_choice(signature), _mark_choice(stamp)
    per_page = None
    if pages:
        try:
            per_page = {
                p.page: {kind: _mark_choice(getattr(p, kind)) for kind in ("letterhead", "signature", "stamp")}
                for p in _PAGE_MARKS.validate_json(pages)
            }
        except ValidationError as exc:
            raise HTTPException(status_code=422, detail=f"Invalid pages: {exc.errors()[0]['msg']}")
        if not any(any(choices.values()) for choices in per_page.values()):
            raise HTTPException(status_code=422, detail="Select a letterhead, signature or stamp for at least one page")
    elif not (letterhead or signature or stamp):
        raise HTTPException(status_code=422, detail="Select at least one of letterhead, signature or stamp")
    try:
        placed = [MarkPlacement(**p.model_dump()) for p in _PLACEMENTS.validate_json(placements)] if placements else None
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=f"Invalid placements: {exc.errors()[0]['msg']}")
    try:
        laid_out = [ContentLayout(**p.model_dump()) for p in _LAYOUTS.validate_json(layouts)] if layouts else None
    except ValidationError as exc:
        raise HTTPException(status_code=422, detail=f"Invalid layouts: {exc.errors()[0]['msg']}")
    return {"letterhead": letterhead, "signature": signature, "stamp": stamp, "placements": placed,
            "layouts": laid_out, "pages": per_page}


def _run_stamp(
    content: bytes,
    filename: str,
    content_type: str,
    options: dict[str, Any],
    *,
    max_size: int | None,
    max_result_size: int | None = None,
    assets: dict[tuple[str, str], bytes] | None = None,
) -> tuple[bytes, str, str]:
    """Marks `content` per `options` (see _stamp_options) - (bytes, media
    type, download filename). `max_size` caps the upload; `max_result_size`
    (default: the same) the marked file - None: no limit. `assets`: mark images already read,
    by (kind, choice) - added to as more are read."""
    if not content:
        raise HTTPException(status_code=422, detail=f"'{filename}' is empty")
    if max_size is not None and len(content) > max_size:
        raise HTTPException(status_code=413, detail=f"File exceeds the {max_size // (1024 * 1024)} MB limit")

    company_profile = load_company_profile()
    if assets is None:
        assets = {}  # each chosen document read once, however many pages use it

    def asset(kind: str, choice: str | None) -> bytes | None:
        if not choice:
            return None
        if (kind, choice) not in assets:
            assets[kind, choice] = _stamp_asset(kind, choice, company_profile)[0]
        return assets[kind, choice]

    pages = None
    if options["pages"] is not None:
        pages = {
            index: PageMarks(asset("letterhead", c["letterhead"]), asset("signature", c["signature"]),
                             asset("stamp", c["stamp"]))
            for index, c in options["pages"].items()
        }

    try:
        marked, media_type = mark_document(
            content, filename, content_type,
            letterhead_image=asset("letterhead", options["letterhead"]),
            signature=asset("signature", options["signature"]), seal=asset("stamp", options["stamp"]),
            placements=options["placements"], layouts=options["layouts"], pages=pages,
        )
    except BidGenerationError as exc:
        raise HTTPException(status_code=422, detail=str(exc))
    limit = max_result_size or max_size
    if limit is not None and len(marked) > limit:
        raise HTTPException(status_code=413, detail=(
            f"The marked file comes to over {limit // (1024 * 1024)} MB - "
            "split it into smaller files and mark each one"))

    stem = Path(filename).stem or "document"
    extension = {"application/pdf": ".pdf", "image/png": ".png"}.get(media_type, ".jpg")
    return marked, media_type, f"{stem}-signed{extension}"


def _per_query_counts(
    collection,
    today: dt.datetime,
    extra_conditions: list[dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Live-tender `total`/`eligible`/`fresh` counts grouped by saved query
    name (a tender matched to N saved queries is counted once per query, so
    summing across queries can exceed a straight distinct-tender count -
    see stats() and stats_by_query() below, which both build on this so
    their numbers agree with each other).

    `fresh` means "published today and an eligibility-criteria match" -
    `today` is the caller's UTC-midnight cutoff (see stats()) so every
    caller in one request agrees on what "today" means.
    """
    extra_conditions = extra_conditions or []
    tomorrow = today + dt.timedelta(days=1)

    def _bucket_cond(*base: dict[str, Any]) -> dict[str, Any]:
        return {"$and": [*base, *extra_conditions]}

    pipeline = [
        # Deleted from the dashboard (DELETE /tenders/{id}) - never counted.
        {"$match": {"hidden": {"$ne": True}}},
        {"$unwind": "$query_matches"},
        {
            "$group": {
                "_id": "$query_matches.query_name",
                "fresh": {
                    "$sum": {
                        "$cond": [
                            _bucket_cond(
                                {"$eq": ["$disappeared", False]},
                                {"$eq": ["$eligibility_match", True]},
                                {"$gte": ["$published_date", today]},
                                {"$lt": ["$published_date", tomorrow]},
                            ),
                            1,
                            0,
                        ]
                    }
                },
                "total": {
                    "$sum": {
                        "$cond": [_bucket_cond({"$eq": ["$disappeared", False]}), 1, 0]
                    }
                },
                "eligible": {
                    "$sum": {
                        "$cond": [
                            _bucket_cond(
                                {"$eq": ["$eligibility_match", True]},
                                {"$eq": ["$disappeared", False]},
                            ),
                            1,
                            0,
                        ]
                    }
                },
                "last_checked": {"$max": "$query_matches.last_seen"},
            }
        },
    ]
    return {r["_id"]: r for r in collection.aggregate(pipeline)}


@app.get("/stats")
def stats() -> dict[str, Any]:
    collection = get_collection()
    base_filter = {"disappeared": False, "hidden": {"$ne": True}}

    pipeline = [
        {"$match": base_filter},
        {
            "$facet": {
                "by_status": [{"$group": {"_id": "$status", "count": {"$sum": 1}}}],
                "by_priority": [
                    {"$group": {"_id": "$latest_priority", "count": {"$sum": 1}}}
                ],
            }
        },
    ]
    result = next(iter(collection.aggregate(pipeline)), {})
    by_status = {r["_id"]: r["count"] for r in result.get("by_status", []) if r["_id"]}
    by_priority = {
        r["_id"]: r["count"] for r in result.get("by_priority", []) if r["_id"]
    }

    today = dt.datetime.combine(
        dt.datetime.now(dt.timezone.utc).date(), dt.time.min
    ).replace(tzinfo=dt.timezone.utc)
    cutoff = today + dt.timedelta(days=CLOSING_SOON_DAYS)
    closing_soon = collection.count_documents(
        {
            **base_filter,
            "eligibility_match": True,
            "closing_date": {"$ne": None, "$gte": today, "$lte": cutoff},
        }
    )
    cross_query = collection.count_documents(
        {**base_filter, "query_match_count": {"$gt": 1}}
    )
    reopened = collection.count_documents({**base_filter, "reopened": True})

    # "Total tenders"/"Eligible"/"Fresh tenders" are the sum of the
    # per-saved-query breakdown (see /stats/by-query) rather than a
    # distinct-tender count, so the two views always agree - a tender
    # matched to more than one saved query is counted once per query, both
    # here and in that table.
    by_name = _per_query_counts(collection, today)
    enabled_names = [saved.name for saved in load_saved_queries() if saved.enabled] + [MANUAL_QUERY_NAME]
    total =sum(by_name.get(name, {}).get("total", 0) for name in enabled_names)
    eligible = sum(by_name.get(name, {}).get("eligible", 0) for name in enabled_names)
    fresh = sum(by_name.get(name, {}).get("fresh", 0) for name in enabled_names)

    return {
        "total": total,
        "by_status": by_status,
        "by_priority": by_priority,
        "closing_soon": closing_soon,
        "cross_query": cross_query,
        "reopened": reopened,
        "eligible": eligible,
        "fresh": fresh,
    }


@app.get("/stats/by-query")
def stats_by_query(shortlisted_only: bool = False, eligible_only: bool = False) -> dict[str, Any]:
    """One row per saved query (config/queries.json), for the dashboard's
    grouped landing view - `total` (currently-live tenders matched to this
    query) and `eligible` (of those, an eligibility-criteria match) counts,
    plus `fresh` (published today and an eligibility-criteria match). The
    "Total tenders"/"Eligible"/"Fresh tenders" summary cards (see stats()
    above) are the sum of these same per-query counts, so the two views
    always agree.

    `shortlisted_only` additionally restricts every count except `eligible`
    to tenders the AI screening rated High/Medium priority (see
    SHORTLISTED_PRIORITIES) - `eligible` stays a straight eligibility count
    so it's informative regardless of that filter. `eligible_only` restricts
    every count to tenders matched against the company's eligibility
    criteria (see app.processing.eligibility). Unscreened/unmatched tenders
    are excluded under either filter, same as the /tenders equivalents.
    """
    collection = get_collection()
    extra_conditions: list[dict[str, Any]] = []
    if shortlisted_only:
        extra_conditions.append({"$in": ["$latest_priority", SHORTLISTED_PRIORITIES]})
    if eligible_only:
        extra_conditions.append({"$eq": ["$eligibility_match", True]})

    today = dt.datetime.combine(
        dt.datetime.now(dt.timezone.utc).date(), dt.time.min
    ).replace(tzinfo=dt.timezone.utc)
    by_name = _per_query_counts(collection, today, extra_conditions)

    # Other Tenders (the manually added ones) always comes last, even empty,
    # since it's where new ones are added from.
    names = [saved.name for saved in load_saved_queries() if saved.enabled] + [MANUAL_QUERY_NAME]
    rows = []
    for name in names:
        r = by_name.get(name, {})
        rows.append(
            {
                "query_name": name,
                "manual": name == MANUAL_QUERY_NAME,
                "total": r.get("total", 0),
                "eligible": r.get("eligible", 0),
                "fresh": r.get("fresh", 0),
                "last_checked": _iso(r.get("last_checked")),
            }
        )
    return {"queries": rows}


def _serialize_run(doc: dict[str, Any]) -> dict[str, Any]:
    out = dict(doc)
    out["id"] = str(out.pop("_id"))
    out["started_at"] = _iso(out.get("started_at"))
    out["finished_at"] = _iso(out.get("finished_at"))
    return out


@app.get("/automation/runs")
def list_automation_runs(
    limit: int = Query(20, ge=1, le=100),
    skip: int = Query(0, ge=0),
) -> dict[str, Any]:
    """Run history for app.pipeline.run_pipeline() - when the scrape/screen
    automation last ran and what each step returned, for the dashboard's
    Automation Raw Data tab.
    """
    collection = get_automation_runs_collection()
    total = collection.count_documents({})
    cursor = collection.find({}).sort("started_at", DESCENDING).skip(skip).limit(limit)
    runs = [_serialize_run(d) for d in cursor]
    return {"total": total, "count": len(runs), "runs": runs}


_asgi_handler = Mangum(app)


def handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """Lambda entry point. A CORS preflight (template.yaml's ApiPreflight
    route - no login) is answered here with an empty 204 and never reaches
    the app: CORSMiddleware above only allows localhost and would 400 the
    CloudFront origin. API Gateway adds its own CorsConfiguration headers to
    this response, so the allowed origin stays locked to the dashboard.

    {"generate_bid": tender_id} is this function invoking itself from POST
    /tenders/{id}/generate-bid - the bid-pack build, run outside the API's
    30s limit (see _run_bid_generation). {"stamp_job": id} is the same for
    POST /stamp-document/jobs (see _run_stamp_job)."""
    if "generate_bid" in event:
        _run_bid_generation(str(event["generate_bid"]), event.get("run"))
        return {"ok": True}
    if "stamp_job" in event:
        _run_stamp_job(str(event["stamp_job"]))
        return {"ok": True}
    if event.get("requestContext", {}).get("http", {}).get("method") == "OPTIONS":
        return {"statusCode": 204, "headers": {}, "body": ""}
    return _asgi_handler(event, context)
