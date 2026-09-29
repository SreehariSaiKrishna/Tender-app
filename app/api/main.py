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

import datetime as dt
import io
import json
import logging
import re
import uuid
from pathlib import Path
from typing import Any, Literal
from urllib.parse import quote

import gridfs
from bson import ObjectId
from bson.errors import InvalidId
from fastapi import FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from mangum import Mangum
from pydantic import BaseModel, Field, TypeAdapter, ValidationError
from pymongo import DESCENDING

from app.config import CONFIG_DIR, get_settings, load_company_profile, load_saved_queries
from app.database import (
    get_automation_runs_collection,
    get_collection,
    get_company_documents_bucket,
    get_company_documents_files_collection,
    get_eligibility_criteria_collection,
    get_generated_bids_bucket,
    get_generated_bids_files_collection,
    get_stamp_transfers_collection,
)
from app.intelligence.bid_drafter import BidDraftingError, draft_checklist_documents, plan_submission_checklist
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
    company_background_text,
    drop_resolved_missing_information,
    enclosure_documents,
    established_facts,
    generate_bid_package,
    is_submission_checklist,
    mark_document,
    merge_drafted_open_items,
    refresh_statuses,
    signing_assets,
)

logger = logging.getLogger(__name__)

app = FastAPI(title="Tender Intelligence API")

# Mirrors template.yaml's CorsConfiguration for the deployed API Gateway -
# needed locally too since uvicorn serves no CORS headers on its own and the
# frontend is a different origin (e.g. a local static server or file://).
# No auth, nothing sensitive returned/accepted (the write endpoints below
# only set a decision flag, a free-text reason, or an eligibility criterion
# - by id), so allowing any origin is a reasonable choice here.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["content-type"],
)

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
    ):
        if out.get(key) is not None:
            out[key] = _iso(out[key])

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
    q: str | None = None,
    limit: int = Query(50, ge=1, le=200),
    skip: int = Query(0, ge=0),
) -> dict[str, Any]:
    collection = get_collection()
    # "closed" tenders are exactly the disappeared ones (see
    # app.processing.deduplicator._mark_disappeared) - every other status
    # implies disappeared=False, so only the closed bucket needs to look
    # past the default "still live" filter.
    filt: dict[str, Any] = {"disappeared": status == "closed"}
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
    """Record that the user applied to this tender. This is what protects
    it from app.processing.cleanup's auto-delete of stale closed tenders -
    a one-way flag; once set, the frontend disables the decline option for
    this tender and there's no "unapply".
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
    "undecline". Unlike `applied`, this does NOT protect the tender from
    app.processing.cleanup's auto-delete of stale closed tenders - a
    decline is a decision already made, not one still pending review.
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


def _load_company_documents_for_generation() -> list[CompanyDocumentRef]:
    """The documents library (see the Documents tab section below), reshaped
    for app.reports.bid_generator - `open_bytes` is lazy so generating a bid
    pack only reads the bytes of documents it actually ends up referencing.
    """
    bucket = get_company_documents_bucket()
    refs = []
    for grid_out in bucket.find():
        metadata = grid_out.metadata or {}
        file_id = grid_out._id
        refs.append(
            CompanyDocumentRef(
                id=str(file_id),
                name=metadata.get("display_name") or grid_out.filename,
                filename=grid_out.filename,
                content_type=metadata.get("content_type") or "application/octet-stream",
                open_bytes=lambda fid=file_id: bucket.open_download_stream(fid).read(),
                size=grid_out.length,
                mark_kind=metadata.get("mark_kind"),
            )
        )
    return refs


def _find_tender(tender_id: str) -> tuple[ObjectId, dict[str, Any]]:
    try:
        object_id = ObjectId(tender_id)
    except InvalidId:
        raise HTTPException(status_code=400, detail="Invalid tender id")
    tender = get_collection().find_one({"_id": object_id})
    if tender is None:
        raise HTTPException(status_code=404, detail="Tender not found")
    return object_id, tender


def _letterhead_path(company_profile: dict[str, Any]) -> Path | None:
    """company_profile.json's letterhead_image, when that file exists."""
    if not company_profile.get("letterhead_image"):
        return None
    candidate = CONFIG_DIR / company_profile["letterhead_image"]
    return candidate if candidate.is_file() else None


def _generation_inputs() -> tuple[list[dict[str, Any]], dict[str, Any], list[CompanyDocumentRef]]:
    """Eligibility criteria, company profile and documents library - what
    both the checklist and the bid pack are built from."""
    criteria_doc = get_eligibility_criteria_collection().find_one({"_id": CRITERIA_DOC_ID})
    return (
        (criteria_doc or {}).get("criteria", []),
        load_company_profile(),
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
    waits. True while a fetch is running; False when there's no pipeline
    function to ask (running locally) or it couldn't be invoked."""
    function_name = get_settings().pipeline_function_name
    if not function_name:
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


def _documents_unreadable(tender: dict[str, Any]) -> bool:
    """The last on-demand fetch finished, but left no document text - the
    files were downloaded and read and none had any (scanned images,
    unsupported formats), or the download itself failed."""
    started, finished = tender.get("document_fetch_started_at"), tender.get("document_fetch_finished_at")
    return started is not None and finished is not None and finished >= started


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
    if not tender.get("source_url"):
        raise HTTPException(
            status_code=409,
            detail="This tender has no source page to download its documents from, so a checklist "
                   "can't be built from its documents.",
        )
    if _documents_unreadable(tender):
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
    if _start_document_fetch(tender):
        raise HTTPException(
            status_code=409,
            detail={
                "fetching": True,
                "message": "Downloading and reading this tender's documents (View Original Notice/Document) "
                           "now - this takes a few minutes. The checklist is built as soon as they're read.",
            },
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
            "bid_document_id": tender.get("bid_document_id"),
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


@app.post("/tenders/{tender_id}/generate-bid")
def generate_bid(tender_id: str) -> dict[str, Any]:
    """Builds a bid pack PDF for this tender from its saved submission
    checklist (see app.reports.bid_generator.generate_bid_package): every
    row's document in S.No order - library documents attached, and each
    document the bidder must write AI-drafted from the tender (see
    app.intelligence.bid_drafter.draft_checklist_documents) - then stores
    it in GridFS (get_generated_bids_bucket, one file per tender, replacing
    any previous draft) and marks the tender applied, same one-way
    protection POST /tenders/{id}/apply gives against
    app.processing.cleanup's stale-closed-tender deletion.

    This never contacts the tendering authority or any external system -
    see this project's stated scope in README.md. It only drafts a local
    PDF for a human to review, complete and sign before anything is
    actually submitted.
    """
    collection = get_collection()
    object_id, tender = _find_tender(tender_id)
    eligibility_criteria, company_profile, company_documents = _generation_inputs()

    checklist = tender.get("checklist")
    if not is_submission_checklist(checklist):
        checklist = _build_tender_checklist(tender, eligibility_criteria, company_profile, company_documents)

    # AI-draft every row the bidder writes itself. `facts` tells the
    # drafting prompt which company facts are already established, from the
    # compliance matrix against the eligibility profile. Never a hard
    # failure: rows without a draft get a templated page to complete.
    facts = established_facts(build_compliance_matrix(eligibility_criteria, company_profile, company_documents))
    to_draft = [
        r for r in checklist.get("items", [])
        if r.get("source") == "draft" and r.get("status") != STATUS_NOT_APPLICABLE
    ]
    drafted: dict[str, Any] = {}
    drafting_note = None
    if to_draft:
        # The brochure and other reference documents are never enclosed in
        # the pack - their text only gives the drafter descriptive background.
        background = company_background_text(company_documents, company_profile)
        try:
            drafted, errors = draft_checklist_documents(
                tender, to_draft, company_profile, facts, company_background=background
            )
        except BidDraftingError as exc:
            errors = [str(exc)]
        if errors:
            logger.warning("AI bid-document drafting incomplete for tender %s: %s", tender_id, errors)
            drafting_note = (
                "AI drafting was unavailable for some documents (" + "; ".join(errors) + "). Those pages are "
                "templates to complete by hand."
            )

    letterhead = _letterhead_path(company_profile)
    if letterhead is None and company_profile.get("letterhead_image"):
        logger.warning("Letterhead image %s not found - bid pack will use plain pages",
                       CONFIG_DIR / company_profile["letterhead_image"])

    drafted = collapse_unfilled_cvs(drafted, to_draft, company_profile)
    checklist = merge_drafted_open_items(checklist, drafted.values())
    checklist = refresh_statuses(checklist, company_documents, drafted)

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
        )
    except BidGenerationError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    bids_bucket = get_generated_bids_bucket()
    now = dt.datetime.now(dt.timezone.utc)
    file_id = bids_bucket.upload_from_stream(
        f"bid-pack-{tender_id}.pdf",
        pdf_bytes,
        metadata={"tender_id": tender_id, "content_type": "application/pdf", "generated_at": now},
    )
    # One bid pack per tender - the previous one is deleted only once the new
    # one is stored, so there's never a moment with no pack to download.
    for existing in get_generated_bids_files_collection().find(
        {"metadata.tender_id": tender_id, "_id": {"$ne": file_id}}
    ):
        bids_bucket.delete(existing["_id"])

    update: dict[str, Any] = {
        "bid_document_id": str(file_id),
        "bid_generated_at": now,
        "checklist": checklist,
    }
    if not tender.get("checklist_generated_at"):
        update["checklist_generated_at"] = checklist.get("generated_at") or now
    if not tender.get("applied"):
        update["applied"] = True
        update["applied_at"] = now
    collection.update_one({"_id": object_id}, {"$set": update})

    doc = collection.find_one({"_id": object_id})
    return _serialize(doc)


# A bid pack encloses library documents, so it easily outgrows what one
# Lambda response can carry (6 MB after Mangum's base64 inflation - see
# MAX_DOCUMENT_SIZE) and API Gateway answers 500. The dashboard therefore
# reads /bid-document/info, fetches each ?part=N slice (3 MB raw, ~4 MB
# encoded) and joins them in the browser. With no ?part the whole file is
# returned in one response, which only works for small packs.
BID_PART_SIZE = 3 * 1024 * 1024


def _open_bid_pack(tender_id: str, file_id: str | None = None) -> Any:
    """This tender's bid pack - the newest, or the exact file a part-by-part
    download started on (`file_id`, from /bid-document/info), so its parts
    never mix two versions. 409 when that file was replaced mid-download by
    a newer generation - the dashboard then starts the download again."""
    files = get_generated_bids_files_collection()
    if file_id:
        try:
            doc = files.find_one({"_id": ObjectId(file_id), "metadata.tender_id": tender_id})
        except InvalidId:
            raise HTTPException(status_code=400, detail="Invalid bid pack file id")
        if doc is None:
            raise HTTPException(status_code=409, detail="The bid pack was regenerated while downloading - "
                                                        "download it again for the new version.")
    else:
        doc = files.find_one({"metadata.tender_id": tender_id}, sort=[("uploadDate", DESCENDING)])
        if doc is None:
            raise HTTPException(status_code=404, detail="No bid pack has been generated for this tender yet")
    return get_generated_bids_bucket().open_download_stream(doc["_id"])


@app.get("/tenders/{tender_id}/bid-document/info")
def bid_document_info(tender_id: str) -> dict[str, Any]:
    grid_out = _open_bid_pack(tender_id)
    return {
        "file_id": str(grid_out._id),
        "filename": grid_out.filename,
        "size": grid_out.length,
        "part_size": BID_PART_SIZE,
        "parts": max(1, -(-grid_out.length // BID_PART_SIZE)),
    }


@app.get("/tenders/{tender_id}/bid-document")
def download_bid_document(tender_id: str, part: int | None = None, file: str | None = None) -> Response:
    grid_out = _open_bid_pack(tender_id, file)
    if part is None:
        content = grid_out.read()
    else:
        if part < 0 or part * BID_PART_SIZE >= max(grid_out.length, 1):
            raise HTTPException(status_code=416, detail="Part out of range")
        grid_out.seek(part * BID_PART_SIZE)
        content = grid_out.read(BID_PART_SIZE)
    return Response(
        content=content,
        media_type="application/pdf" if part is None else "application/octet-stream",
        headers={"Content-Disposition": _content_disposition("attachment", grid_out.filename)},
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


# --- Documents -----------------------------------------------------------
# A small document library the user manages by hand from the dashboard
# (company certificates, licenses, etc) - upload, rename, delete, view and
# download. Distinct from the `documents` array on a tender, which is
# attachments collected automatically by the pipeline. Stored in GridFS
# (see app.database.get_company_documents_bucket) since there's no
# general-purpose S3 bucket in this stack and Mongo is already provisioned.
#
# One request or response can carry at most MAX_DOCUMENT_SIZE: Mangum
# returns the response base64-encoded, which inflates size by ~33%, and a
# synchronous Lambda invoke tops out at 6 MB - 4 MB raw keeps the encoded
# payload safely under that. Files up to MAX_UPLOAD_SIZE therefore travel in
# TRANSFER_PART_SIZE parts instead: uploaded with PUT /uploads/{id}/parts/N
# then a .../finish call (documents library or Sign & Stamp), and downloaded
# with ?part=N (as the bid pack is - see BID_PART_SIZE).
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
    hour at most)."""
    key = _transfer_key(upload_id)
    if not 0 <= part < -(-MAX_UPLOAD_SIZE // TRANSFER_PART_SIZE):
        raise HTTPException(status_code=413, detail=f"File exceeds the {_mb(MAX_UPLOAD_SIZE)} MB limit")
    data = await request.body()
    if not data:
        raise HTTPException(status_code=422, detail="Part is empty")
    if len(data) > TRANSFER_PART_SIZE:
        raise HTTPException(status_code=413, detail=f"Parts must be at most {_mb(TRANSFER_PART_SIZE)} MB")
    get_stamp_transfers_collection().replace_one(
        {"kind": "upload", "key": key, "part": part},
        {"kind": "upload", "key": key, "part": part, "data": data, "created_at": dt.datetime.now(dt.timezone.utc)},
        upsert=True,
    )
    return {"part": part, "size": len(data)}


def _take_upload(upload_id: str, parts: int) -> bytes:
    """An upload's `parts` joined back into the file - its parts deleted.
    409 when any is missing (never sent, or expired)."""
    key = _transfer_key(upload_id)
    transfers = get_stamp_transfers_collection()
    stored = {d["part"]: d["data"] for d in transfers.find({"kind": "upload", "key": key})}
    if sorted(stored) != list(range(parts)):
        raise HTTPException(status_code=409, detail="The upload is incomplete or has expired - upload the file again")
    transfers.delete_many({"kind": "upload", "key": key})
    content = b"".join(stored[i] for i in range(parts))
    if len(content) > MAX_UPLOAD_SIZE:
        raise HTTPException(status_code=413, detail=f"File exceeds the {_mb(MAX_UPLOAD_SIZE)} MB limit")
    return content


def _serialize_document(grid_out: Any) -> dict[str, Any]:
    metadata = grid_out.metadata or {}
    return {
        "id": str(grid_out._id),
        "name": metadata.get("display_name") or grid_out.filename,
        "filename": grid_out.filename,
        "content_type": metadata.get("content_type") or "application/octet-stream",
        "size": grid_out.length,
        "uploaded_at": _iso(grid_out.upload_date),
        "mark_kind": metadata.get("mark_kind"),
    }


def _content_disposition(kind: str, filename: str) -> str:
    # RFC 5987: an ASCII fallback for older clients plus a UTF-8 filename*
    # for everything else - filename is user-supplied, so both are also
    # stripped of control characters (incl. CR/LF) to avoid header injection.
    ascii_fallback = re.sub(r"[^\x20-\x7e]", "_", filename).replace('"', "'") or "document"
    return f'{kind}; filename="{ascii_fallback}"; filename*=UTF-8\'\'{quote(filename)}'


def _get_document_or_404(document_id: str) -> Any:
    try:
        object_id = ObjectId(document_id)
    except InvalidId:
        raise HTTPException(status_code=400, detail="Invalid document id")
    try:
        return get_company_documents_bucket().open_download_stream(object_id)
    except gridfs.errors.NoFile:
        raise HTTPException(status_code=404, detail="Document not found")


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
    bucket = get_company_documents_bucket()
    file_id = bucket.upload_from_stream(filename or display_name, content, metadata=metadata)
    return _serialize_document(bucket.open_download_stream(file_id))


def _replace_document(
    document_id: str,
    content: bytes,
    filename: str | None,
    name: str | None,
    content_type: str | None,
    mark_kind: str | None = None,
) -> dict[str, Any]:
    """Swap out a document's file while keeping it as the same row in the
    list - as opposed to DELETE+POST, which would also work but loses the
    display name unless the caller re-types it. Uploads the replacement
    before deleting the old file (rather than the other way round) so a
    failed upload never leaves the document missing its content; the row's
    id does change, but the frontend always reloads the full list after a
    write, so that's invisible to the user.
    """
    try:
        object_id = ObjectId(document_id)
    except InvalidId:
        raise HTTPException(status_code=400, detail="Invalid document id")
    existing = get_company_documents_files_collection().find_one({"_id": object_id})
    if existing is None:
        raise HTTPException(status_code=404, detail="Document not found")
    existing_meta = existing.get("metadata") or {}
    stored = _store_document(content, filename, name or existing_meta.get("display_name"), content_type,
                             mark_kind or existing_meta.get("mark_kind"))
    get_company_documents_bucket().delete(object_id)
    return stored


@app.get("/documents")
def list_documents() -> dict[str, Any]:
    cursor = get_company_documents_bucket().find(sort=[("uploadDate", DESCENDING)])
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
    try:
        object_id = ObjectId(document_id)
    except InvalidId:
        raise HTTPException(status_code=400, detail="Invalid document id")

    update = {"metadata.display_name": name}
    if body.mark_kind:
        update["metadata.mark_kind"] = body.mark_kind
    result = get_company_documents_files_collection().update_one({"_id": object_id}, {"$set": update})
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Document not found")

    grid_out = get_company_documents_bucket().open_download_stream(object_id)
    return _serialize_document(grid_out)


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
    try:
        object_id = ObjectId(document_id)
    except InvalidId:
        raise HTTPException(status_code=400, detail="Invalid document id")
    try:
        get_company_documents_bucket().delete(object_id)
    except gridfs.errors.NoFile:
        raise HTTPException(status_code=404, detail="Document not found")
    return {"status": "deleted"}


@app.get("/documents/{document_id}/info")
def document_info(document_id: str) -> dict[str, Any]:
    """How many ?part=N requests /download and /view need for this file."""
    grid_out = _get_document_or_404(document_id)
    return {**_serialize_document(grid_out), "part_size": TRANSFER_PART_SIZE,
            "parts": max(1, -(-grid_out.length // TRANSFER_PART_SIZE))}


def _document_response(document_id: str, kind: str, part: int | None) -> Response:
    """The whole document (small files only - see MAX_DOCUMENT_SIZE) or, with
    `part`, one TRANSFER_PART_SIZE slice of it."""
    grid_out = _get_document_or_404(document_id)
    content_type = (grid_out.metadata or {}).get("content_type") or "application/octet-stream"
    if part is None:
        content = grid_out.read()
    else:
        if part < 0 or part * TRANSFER_PART_SIZE >= max(grid_out.length, 1):
            raise HTTPException(status_code=416, detail="Part out of range")
        grid_out.seek(part * TRANSFER_PART_SIZE)
        content = grid_out.read(TRANSFER_PART_SIZE)
        content_type = "application/octet-stream"
    return Response(
        content=content,
        media_type=content_type,
        headers={"Content-Disposition": _content_disposition(kind, grid_out.filename)},
    )


@app.get("/documents/{document_id}/download")
def download_document(document_id: str, part: int | None = None) -> Response:
    return _document_response(document_id, "attachment", part)


@app.get("/documents/{document_id}/view")
def view_document(document_id: str, part: int | None = None) -> Response:
    return _document_response(document_id, "inline", part)


# --- Sign & Stamp ------------------------------------------------------------
# The dashboard's Sign & Stamp tab: upload any PDF or image and get it back
# with the letterhead, signature and/or seal on every page, placed the same
# way a bid pack places them (app.reports.bid_generator.mark_document).
# Nothing is kept - files up to MAX_UPLOAD_SIZE pass through temporary
# parts (see below) only because one Lambda request/response can't carry
# them, and those parts are deleted or expire within the hour.

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
    try:
        object_id = ObjectId(document_id)
    except InvalidId:
        raise HTTPException(status_code=400, detail="Invalid document id")
    result = get_company_documents_files_collection().update_one(
        {"_id": object_id}, {"$set": {"metadata.mark_kind": NOT_A_MARK}}
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


# The dashboard sends files up to MAX_UPLOAD_SIZE in parts (PUT
# /uploads/{id}/parts/N - see the Documents section), then /finish marks the
# joined file and stores the result in parts too, fetched with one GET per
# part (as the bid pack download does).
STAMP_MAX_RESULT_SIZE = 25 * 1024 * 1024  # a letterhead/marks can make the result bigger than the upload


@app.post("/stamp-document/uploads/{upload_id}/finish")
def finish_stamp_upload(
    upload_id: str,
    parts: int = Form(..., ge=1),
    filename: str = Form("document"),
    content_type: str = Form(""),
    letterhead: str | None = Form(None),
    signature: str | None = Form(None),
    stamp: str | None = Form(None),
    placements: str | None = Form(None),
    layouts: str | None = Form(None),
    pages: str | None = Form(None),
) -> dict[str, Any]:
    """Joins an upload's `parts`, marks it (same options as POST
    /stamp-document) and stores the result for GET .../results/{id}/parts/N."""
    options = _stamp_options(letterhead, signature, stamp, placements, layouts, pages)
    content = _take_upload(upload_id, parts)
    marked, media_type, out_name = _run_stamp(content, filename, content_type, options,
                                              max_size=MAX_UPLOAD_SIZE, max_result_size=STAMP_MAX_RESULT_SIZE)
    transfers = get_stamp_transfers_collection()
    result_id = uuid.uuid4().hex
    now = dt.datetime.now(dt.timezone.utc)
    chunks = [marked[i:i + TRANSFER_PART_SIZE] for i in range(0, len(marked), TRANSFER_PART_SIZE)] or [b""]
    transfers.insert_many([
        {"kind": "result", "key": result_id, "part": i, "data": chunk, "media_type": media_type, "created_at": now}
        for i, chunk in enumerate(chunks)
    ])
    return {"result_id": result_id, "parts": len(chunks), "filename": out_name, "media_type": media_type,
            "size": len(marked)}


@app.get("/stamp-document/results/{result_id}/parts/{part}")
def download_stamp_part(result_id: str, part: int) -> Response:
    doc = get_stamp_transfers_collection().find_one({"kind": "result", "key": _transfer_key(result_id), "part": part})
    if doc is None:
        raise HTTPException(status_code=404, detail="This result has expired - mark the file again")
    return Response(content=bytes(doc["data"]), media_type="application/octet-stream")


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
    max_size: int,
    max_result_size: int | None = None,
) -> tuple[bytes, str, str]:
    """Marks `content` per `options` (see _stamp_options) - (bytes, media
    type, download filename). `max_size` caps the upload; `max_result_size`
    (default: the same) the marked file."""
    if not content:
        raise HTTPException(status_code=422, detail="File is empty")
    if len(content) > max_size:
        raise HTTPException(status_code=413, detail=f"File exceeds the {max_size // (1024 * 1024)} MB limit")

    company_profile = load_company_profile()
    assets: dict[tuple[str, str], bytes] = {}  # each chosen document read once, however many pages use it

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
    if len(marked) > limit:
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
    base_filter = {"disappeared": False}

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

    # "Total tenders"/"Eligible"/"Fresh tenders" are the sum of the
    # per-saved-query breakdown (see /stats/by-query) rather than a
    # distinct-tender count, so the two views always agree - a tender
    # matched to more than one saved query is counted once per query, both
    # here and in that table.
    by_name = _per_query_counts(collection, today)
    enabled_names = [saved.name for saved in load_saved_queries() if saved.enabled]
    total = sum(by_name.get(name, {}).get("total", 0) for name in enabled_names)
    eligible = sum(by_name.get(name, {}).get("eligible", 0) for name in enabled_names)
    fresh = sum(by_name.get(name, {}).get("fresh", 0) for name in enabled_names)

    return {
        "total": total,
        "by_status": by_status,
        "by_priority": by_priority,
        "closing_soon": closing_soon,
        "cross_query": cross_query,
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

    rows = []
    for saved in load_saved_queries():
        if not saved.enabled:
            continue
        r = by_name.get(saved.name, {})
        rows.append(
            {
                "query_name": saved.name,
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


handler = Mangum(app)
