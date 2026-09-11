"""Read-mostly API over the `tenders` collection - serves the dashboard
(Step 5). Deliberately narrow: the only writes are the "mark applied" and
"decline" endpoints below, and it only imports app.config/app.database (not
app.processing/app.intelligence/app.browser), so this Lambda's image never
needs Playwright, pandas, or the OpenAI SDK - just fastapi, mangum, and
pymongo.

Runs locally the same way the CLI does, alongside app.lambda_handler:
    uvicorn app.api.main:app --reload
"""
from __future__ import annotations

import datetime as dt
import re
from typing import Any

from bson import ObjectId
from bson.errors import InvalidId
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from mangum import Mangum
from pydantic import BaseModel
from pymongo import DESCENDING

from app.config import load_saved_queries
from app.database import (
    get_automation_runs_collection,
    get_collection,
    get_eligibility_criteria_collection,
)

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

    for key in (
        "first_seen",
        "last_seen",
        "published_date",
        "closing_date",
        "previous_closing_date",
        "content_last_changed",
        "applied_at",
        "declined_at",
    ):
        if out.get(key) is not None:
            out[key] = _iso(out[key])

    for match in out.get("query_matches", []):
        match["first_seen"] = _iso(match.get("first_seen"))
        match["last_seen"] = _iso(match.get("last_seen"))

    for screening in out.get("screenings", []):
        screening["screened_at"] = _iso(screening.get("screened_at"))

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
    # Within that, High priority before Medium/Low/Not Relevant/unscreened
    # (see app.models.priority_rank), then soonest-closing first.
    cursor = (
        collection.find(filt)
        .sort(
            [
                ("eligibility_match", -1),
                ("latest_priority_rank", 1),
                ("closing_date", 1),
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


@app.get("/stats")
def stats() -> dict[str, Any]:
    collection = get_collection()
    base_filter = {"disappeared": False}

    pipeline = [
        {"$match": base_filter},
        {
            "$facet": {
                "total": [{"$count": "count"}],
                "by_status": [{"$group": {"_id": "$status", "count": {"$sum": 1}}}],
                "by_priority": [
                    {"$group": {"_id": "$latest_priority", "count": {"$sum": 1}}}
                ],
            }
        },
    ]
    result = next(iter(collection.aggregate(pipeline)), {})
    total = result.get("total", [{}])[0].get("count", 0) if result.get("total") else 0
    by_status = {r["_id"]: r["count"] for r in result.get("by_status", []) if r["_id"]}
    by_priority = {
        r["_id"]: r["count"] for r in result.get("by_priority", []) if r["_id"]
    }

    today = dt.datetime.combine(
        dt.datetime.now(dt.timezone.utc).date(), dt.time.min
    ).replace(tzinfo=dt.timezone.utc)
    cutoff = today + dt.timedelta(days=CLOSING_SOON_DAYS)
    closing_soon = collection.count_documents(
        {**base_filter, "closing_date": {"$ne": None, "$gte": today, "$lte": cutoff}}
    )
    cross_query = collection.count_documents(
        {**base_filter, "query_match_count": {"$gt": 1}}
    )

    return {
        "total": total,
        "by_status": by_status,
        "by_priority": by_priority,
        "closing_soon": closing_soon,
        "cross_query": cross_query,
    }


@app.get("/stats/by-query")
def stats_by_query(shortlisted_only: bool = False, eligible_only: bool = False) -> dict[str, Any]:
    """One row per saved query (config/queries.json), for the dashboard's
    grouped landing view - `total` (live + closed) and `eligible` (of
    those, an eligibility-criteria match) counts, plus currently-live
    tenders split into `fresh` (status=new) vs. the rest (`live`).

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
                                {"$eq": ["$status", "new"]},
                            ),
                            1,
                            0,
                        ]
                    }
                },
                "live": {
                    "$sum": {
                        "$cond": [
                            _bucket_cond({"$eq": ["$disappeared", False]}),
                            1,
                            0,
                        ]
                    }
                },
                # Every tender matched to this query under the current
                # filters, live or closed - what "Total" shows.
                "total": {"$sum": {"$cond": [_bucket_cond(), 1, 0]}},
                # How many of those are eligibility matches (see
                # app.processing.eligibility) - always shown regardless of
                # the eligible_only toggle, so it stays informative even
                # when that filter is off.
                "eligible": {
                    "$sum": {
                        "$cond": [
                            _bucket_cond({"$eq": ["$eligibility_match", True]}),
                            1,
                            0,
                        ]
                    }
                },
                "last_checked": {"$max": "$query_matches.last_seen"},
            }
        },
    ]
    by_name = {r["_id"]: r for r in collection.aggregate(pipeline)}

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
                "live": r.get("live", 0),
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
