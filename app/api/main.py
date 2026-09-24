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
import logging
import re
import uuid
from typing import Any, Literal
from urllib.parse import quote

import gridfs
from bson import ObjectId
from bson.errors import InvalidId
from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from mangum import Mangum
from pydantic import BaseModel
from pymongo import DESCENDING

from app.config import CONFIG_DIR, load_company_profile, load_saved_queries
from app.database import (
    get_automation_runs_collection,
    get_collection,
    get_company_documents_bucket,
    get_company_documents_files_collection,
    get_eligibility_criteria_collection,
    get_generated_bids_bucket,
    get_generated_bids_files_collection,
)
from app.intelligence.bid_drafter import BidDraftingError, draft_bid_documents
from app.reports.bid_generator import (
    BidGenerationError,
    CompanyDocumentRef,
    build_checklist,
    build_compliance_matrix,
    company_background_text,
    established_facts,
    generate_bid_package,
    merge_drafted_open_items,
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
    # The editable review checklist is served on its own (GET
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
        collection.find(filt)
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


def _generation_inputs() -> tuple[list[dict[str, Any]], dict[str, Any], list[CompanyDocumentRef]]:
    """Eligibility criteria, company profile and documents library - what
    both the checklist and the bid pack are built from."""
    criteria_doc = get_eligibility_criteria_collection().find_one({"_id": CRITERIA_DOC_ID})
    return (
        (criteria_doc or {}).get("criteria", []),
        load_company_profile(),
        _load_company_documents_for_generation(),
    )


# --- Editable review checklist ------------------------------------------------
# The bid pack's internal review checklist as data on the tender (see
# app.reports.bid_generator.build_checklist), so the team can edit, tick
# and add rows on the dashboard - POST /generate-bid then prints this saved
# version into the pack instead of rebuilding it.


def _checklist_response(tender: dict[str, Any]) -> dict[str, Any]:
    checklist = dict(tender["checklist"])
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
    }


@app.post("/tenders/{tender_id}/checklist")
def generate_checklist(tender_id: str) -> dict[str, Any]:
    """(Re)builds this tender's checklist from its document_summary, the
    eligibility profile and the documents library - replacing any edits."""
    object_id, tender = _find_tender(tender_id)
    checklist = build_checklist(tender, *_generation_inputs())
    get_collection().update_one(
        {"_id": object_id},
        {"$set": {"checklist": checklist, "checklist_generated_at": checklist["generated_at"]}},
    )
    return _checklist_response({**tender, "checklist": checklist})


@app.get("/tenders/{tender_id}/checklist")
def get_checklist(tender_id: str) -> dict[str, Any]:
    _, tender = _find_tender(tender_id)
    if not tender.get("checklist"):
        raise HTTPException(status_code=404, detail="No checklist has been generated for this tender yet")
    return _checklist_response(tender)


ChecklistSection = Literal[
    "open_items", "company_eligibility", "tender_requirements",
    "reference_docs", "not_enclosed", "missing_information",
]


class ChecklistItem(BaseModel):
    id: str | None = None
    section: ChecklistSection
    requirement: str = ""
    source: str = ""
    status: str = ""
    evidence: str = ""
    done: bool = False
    origin: Literal["auto", "user", "ai_draft"] = "user"


class ChecklistUpdateRequest(BaseModel):
    items: list[ChecklistItem]


@app.put("/tenders/{tender_id}/checklist")
def update_checklist(tender_id: str, body: ChecklistUpdateRequest) -> dict[str, Any]:
    object_id, tender = _find_tender(tender_id)
    if not tender.get("checklist"):
        raise HTTPException(status_code=404, detail="No checklist has been generated for this tender yet")
    items = []
    for item in body.items:
        row = item.model_dump()
        for key in ("requirement", "source", "status", "evidence"):
            row[key] = row[key].strip()
        if not row["requirement"]:
            continue  # a blank row added then left empty
        row["id"] = row["id"] or uuid.uuid4().hex
        items.append(row)
    checklist = {**tender["checklist"], "items": items, "updated_at": dt.datetime.now(dt.timezone.utc)}
    get_collection().update_one({"_id": object_id}, {"$set": {"checklist": checklist}})
    return _checklist_response({**tender, "checklist": checklist})


@app.post("/tenders/{tender_id}/generate-bid")
def generate_bid(tender_id: str) -> dict[str, Any]:
    """Drafts a bid pack PDF for this tender (see app.reports.bid_generator)
    from the tender's own AI-extracted document_summary, the company's
    eligibility profile, config/company_profile.json, and whatever's in the
    documents library - then stores it in GridFS (get_generated_bids_bucket,
    one file per tender, replacing any previous draft) and marks the tender
    applied, same one-way protection POST /tenders/{id}/apply gives against
    app.processing.cleanup's stale-closed-tender deletion.

    This never contacts the tendering authority or any external system -
    see this project's stated scope in README.md. It only drafts a local
    PDF for a human to review, complete and sign before anything is
    actually submitted.
    """
    collection = get_collection()
    object_id, tender = _find_tender(tender_id)
    eligibility_criteria, company_profile, company_documents = _generation_inputs()

    # AI-draft the individual bid documents this tender's own paperwork
    # calls for (see app.intelligence.bid_drafter) - `facts` tells the
    # drafting prompt which company facts are already established, from the
    # exact same compliance matrix the pack itself renders further down, so
    # the two never disagree with each other. Never a hard failure: if
    # drafting is unavailable (no OPENAI_API_KEY, a bad response, a slow
    # provider) the pack still generates with a plain templated covering
    # letter (see generate_bid_package's `drafted_documents=None` fallback).
    company_rows = build_compliance_matrix(eligibility_criteria, company_profile, company_documents)
    facts = established_facts(company_rows)
    drafted_documents = None
    drafting_note = None
    # The brochure and other reference documents are never enclosed in the
    # pack - their text only gives the drafter descriptive background.
    background = company_background_text(company_documents, company_profile)
    try:
        drafted_documents = draft_bid_documents(
            tender, eligibility_criteria, company_profile, facts, company_background=background
        )
    except BidDraftingError as exc:
        logger.warning("AI bid-document drafting unavailable for tender %s: %s", tender_id, exc)
        drafting_note = (
            f"AI-assisted document drafting was unavailable for this run ({exc}). A basic covering letter "
            "has been included below; prepare any other required declarations manually using the "
            "compliance matrix further down as your checklist."
        )

    letterhead = None
    if company_profile.get("letterhead_image"):
        candidate = CONFIG_DIR / company_profile["letterhead_image"]
        letterhead = candidate if candidate.is_file() else None
        if letterhead is None:
            logger.warning("Letterhead image %s not found - bid pack will use plain pages", candidate)

    # The pack's checklist pages print the tender's saved (possibly
    # hand-edited) checklist - built fresh only if none was generated yet -
    # with this run's drafted open items swapped in.
    checklist = tender.get("checklist") or build_checklist(
        tender, eligibility_criteria, company_profile, company_documents
    )
    checklist = merge_drafted_open_items(checklist, drafted_documents)

    try:
        pdf_bytes = generate_bid_package(
            tender,
            eligibility_criteria,
            company_profile,
            company_documents,
            drafted_documents=drafted_documents,
            drafting_note=drafting_note,
            letterhead_image=letterhead,
            checklist=checklist,
        )
    except BidGenerationError as exc:
        raise HTTPException(status_code=422, detail=str(exc))

    bids_bucket = get_generated_bids_bucket()
    # One bid pack per tender - delete any previous draft before uploading
    # the new one (rather than accumulating a version per click).
    for existing in get_generated_bids_files_collection().find({"metadata.tender_id": tender_id}):
        bids_bucket.delete(existing["_id"])

    now = dt.datetime.now(dt.timezone.utc)
    file_id = bids_bucket.upload_from_stream(
        f"bid-pack-{tender_id}.pdf",
        pdf_bytes,
        metadata={"tender_id": tender_id, "content_type": "application/pdf", "generated_at": now},
    )

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


@app.get("/tenders/{tender_id}/bid-document")
def download_bid_document(tender_id: str) -> Response:
    doc = get_generated_bids_files_collection().find_one({"metadata.tender_id": tender_id})
    if doc is None:
        raise HTTPException(status_code=404, detail="No bid pack has been generated for this tender yet")
    grid_out = get_generated_bids_bucket().open_download_stream(doc["_id"])
    return Response(
        content=grid_out.read(),
        media_type="application/pdf",
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
# Capped well under API Gateway's payload limit: Mangum returns the response
# base64-encoded, which inflates size by ~33%, and a synchronous Lambda
# invoke response tops out at 6 MB - 4 MB raw keeps the encoded response
# safely under that even for a download of the largest allowed file.
MAX_DOCUMENT_SIZE = 4 * 1024 * 1024


def _serialize_document(grid_out: Any) -> dict[str, Any]:
    metadata = grid_out.metadata or {}
    return {
        "id": str(grid_out._id),
        "name": metadata.get("display_name") or grid_out.filename,
        "filename": grid_out.filename,
        "content_type": metadata.get("content_type") or "application/octet-stream",
        "size": grid_out.length,
        "uploaded_at": _iso(grid_out.upload_date),
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


@app.get("/documents")
def list_documents() -> dict[str, Any]:
    cursor = get_company_documents_bucket().find(sort=[("uploadDate", DESCENDING)])
    return {"documents": [_serialize_document(d) for d in cursor]}


@app.post("/documents")
async def upload_document(
    file: UploadFile = File(...), name: str | None = Form(None)
) -> dict[str, Any]:
    content = await file.read()
    if not content:
        raise HTTPException(status_code=422, detail="File is empty")
    if len(content) > MAX_DOCUMENT_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds the {MAX_DOCUMENT_SIZE // (1024 * 1024)} MB limit",
        )
    display_name = (name or file.filename or "Untitled").strip() or "Untitled"
    filename = file.filename or display_name

    file_id = get_company_documents_bucket().upload_from_stream(
        filename,
        content,
        metadata={
            "display_name": display_name,
            "content_type": file.content_type or "application/octet-stream",
        },
    )
    grid_out = get_company_documents_bucket().open_download_stream(file_id)
    return _serialize_document(grid_out)


class DocumentRenameRequest(BaseModel):
    name: str


@app.put("/documents/{document_id}")
def rename_document(document_id: str, body: DocumentRenameRequest) -> dict[str, Any]:
    name = body.name.strip()
    if not name:
        raise HTTPException(status_code=422, detail="Name is required")
    try:
        object_id = ObjectId(document_id)
    except InvalidId:
        raise HTTPException(status_code=400, detail="Invalid document id")

    result = get_company_documents_files_collection().update_one(
        {"_id": object_id}, {"$set": {"metadata.display_name": name}}
    )
    if result.matched_count == 0:
        raise HTTPException(status_code=404, detail="Document not found")

    grid_out = get_company_documents_bucket().open_download_stream(object_id)
    return _serialize_document(grid_out)


@app.put("/documents/{document_id}/replace")
async def replace_document(
    document_id: str, file: UploadFile = File(...), name: str | None = Form(None)
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

    content = await file.read()
    if not content:
        raise HTTPException(status_code=422, detail="File is empty")
    if len(content) > MAX_DOCUMENT_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds the {MAX_DOCUMENT_SIZE // (1024 * 1024)} MB limit",
        )

    existing_name = (existing.get("metadata") or {}).get("display_name")
    display_name = (name or existing_name or file.filename or "Untitled").strip() or "Untitled"
    filename = file.filename or display_name

    bucket = get_company_documents_bucket()
    new_file_id = bucket.upload_from_stream(
        filename,
        content,
        metadata={
            "display_name": display_name,
            "content_type": file.content_type or "application/octet-stream",
        },
    )
    bucket.delete(object_id)
    grid_out = bucket.open_download_stream(new_file_id)
    return _serialize_document(grid_out)


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


@app.get("/documents/{document_id}/download")
def download_document(document_id: str) -> Response:
    grid_out = _get_document_or_404(document_id)
    content_type = (grid_out.metadata or {}).get("content_type") or "application/octet-stream"
    return Response(
        content=grid_out.read(),
        media_type=content_type,
        headers={"Content-Disposition": _content_disposition("attachment", grid_out.filename)},
    )


@app.get("/documents/{document_id}/view")
def view_document(document_id: str) -> Response:
    grid_out = _get_document_or_404(document_id)
    content_type = (grid_out.metadata or {}).get("content_type") or "application/octet-stream"
    return Response(
        content=grid_out.read(),
        media_type=content_type,
        headers={"Content-Disposition": _content_disposition("inline", grid_out.filename)},
    )


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
