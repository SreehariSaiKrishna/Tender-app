"""Read-only API over the `tenders` collection - serves the dashboard
(Step 5). Deliberately narrow: no writes happen here at all, and it only
imports app.config/app.database (not app.processing/app.intelligence/
app.browser), so this Lambda's image never needs Playwright, pandas, or
the OpenAI SDK - just fastapi, mangum, and pymongo.

Runs locally the same way the CLI does, alongside app.lambda_handler:
    uvicorn app.api.main:app --reload
"""
from __future__ import annotations

import datetime as dt
from typing import Any

from bson import ObjectId
from bson.errors import InvalidId
from fastapi import FastAPI, HTTPException, Query
from mangum import Mangum

from app.database import get_collection

app = FastAPI(title="Tender Intelligence API")

CLOSING_SOON_DAYS = 7


def _iso(value: dt.datetime | None) -> str | None:
    return value.isoformat() if value else None


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
    limit: int = Query(50, ge=1, le=200),
    skip: int = Query(0, ge=0),
) -> dict[str, Any]:
    collection = get_collection()
    filt: dict[str, Any] = {"disappeared": False}
    if status:
        filt["status"] = status
    if priority:
        filt["latest_priority"] = priority
    if query_name:
        filt["query_matches.query_name"] = query_name
    if closing_soon:
        today = dt.datetime.combine(
            dt.datetime.now(dt.timezone.utc).date(), dt.time.min
        ).replace(tzinfo=dt.timezone.utc)
        cutoff = today + dt.timedelta(days=CLOSING_SOON_DAYS)
        filt["closing_date"] = {"$ne": None, "$gte": today, "$lte": cutoff}

    total = collection.count_documents(filt)
    cursor = collection.find(filt).sort("last_seen", -1).skip(skip).limit(limit)
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


handler = Mangum(app)
