"""Deduplication and history tracking (Phase 4).

Ingests one "pass" of normalized tenders - the latest full export per saved
query - and merges it into the `tenders` collection:

  - Matches tenders by dedup_key (the TDR reference number when available,
    else a normalized title+organisation+closing_date key). Two tenders are
    only ever merged if they share the same dedup_key; similar-looking
    titles are never treated as duplicates of each other.
  - Tracks first_seen, last_seen, times_found.
  - Records every saved query that has ever matched a tender (embedded in
    `query_matches`), so cross-query duplicates are detected instead of
    silently discarded.
  - Detects a changed closing date (previous vs current).
  - Detects tenders that disappeared: only for queries that were actually
    part of this pass - a query that wasn't downloaded this run (skipped/
    failed) never causes its previously-tracked tenders to be marked gone,
    since we simply have no fresh data for it.
  - Never deletes a document. A tender that disappears is marked, not removed.

MongoDB has no pure "date" type (only datetime), so `closing_date` /
`published_date` / `previous_closing_date` are stored as UTC-midnight
datetimes and converted back to `date` where day-level arithmetic is needed.
"""
from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass, field

from pymongo import ReplaceOne, UpdateOne
from pymongo.collection import Collection

from app.database import get_collection
from app.models import TenderStatus
from app.processing.normalizer import NormalizedTender

logger = logging.getLogger(__name__)

CLOSING_SOON_DAYS = 7


def _utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _date_to_dt(d: dt.date | None) -> dt.datetime | None:
    if d is None:
        return None
    return dt.datetime.combine(d, dt.time.min).replace(tzinfo=dt.timezone.utc)


def _dt_to_date(value: dt.datetime | None) -> dt.date | None:
    if value is None:
        return None
    return value.date()


def _same_instant(a: dt.datetime | None, b: dt.datetime | None) -> bool:
    """Compare two datetimes ignoring aware-vs-naive mismatches.

    MongoDB (and mongomock, faithfully) returns datetimes read back from the
    database as naive UTC, while a freshly-constructed value in this process
    may still carry tzinfo - comparing the two directly with `!=` would
    otherwise report a change that never happened.
    """
    if a is None or b is None:
        return a is b

    def _naive(d: dt.datetime) -> dt.datetime:
        return d.astimezone(dt.timezone.utc).replace(tzinfo=None) if d.tzinfo else d

    return _naive(a) == _naive(b)


@dataclass
class IngestSummary:
    new: int = 0
    updated: int = 0  # deadline changed on an existing tender
    unchanged: int = 0
    disappeared: int = 0
    closing_soon: int = 0
    cross_query_matches: int = 0
    queries_processed: list[str] = field(default_factory=list)


def ingest_batch(
    query_results: dict[str, list[NormalizedTender]],
    collection: Collection | None = None,
) -> IngestSummary:
    """Merge one pass's worth of per-query exports into the `tenders`
    collection.

    `query_results` must map query_name -> the FULL latest export for that
    query (what "Download Excel" produces). Omit a query entirely if it
    couldn't be downloaded this pass - its history is left untouched.

    Reads and writes are batched (one `find` for every existing document
    this pass touches, one `bulk_write` for every upsert) rather than one
    round-trip per tender - with exports running into the thousands of
    rows, per-document round-trips alone can take minutes and blow past a
    Lambda timeout that the equivalent bulk operations clear in seconds.
    """
    collection = collection if collection is not None else get_collection()
    summary = IngestSummary(queries_processed=list(query_results.keys()))
    now = _utcnow()
    today = now.date()

    keys_this_pass: dict[str, set[str]] = {}
    latest_by_key: dict[str, NormalizedTender] = {}
    for query_name, tenders in query_results.items():
        for t in tenders:
            keys_this_pass.setdefault(t.dedup_key, set()).add(query_name)
            latest_by_key[t.dedup_key] = t

    existing_by_key = {
        doc["dedup_key"]: doc
        for doc in collection.find({"dedup_key": {"$in": list(latest_by_key.keys())}})
    }

    tenders_seen_keys: set[str] = set()
    operations: list[ReplaceOne] = []

    for dedup_key, normalized in latest_by_key.items():
        matched_queries = keys_this_pass[dedup_key]
        existing = existing_by_key.get(dedup_key)
        deadline_changed = False
        is_new = existing is None
        new_closing_date = _date_to_dt(normalized.closing_date)

        if existing is None:
            doc = {
                "dedup_key": dedup_key,
                "tender_ref": normalized.tender_ref,
                "ref_is_synthetic": normalized.ref_is_synthetic,
                "title": normalized.title,
                "organisation": normalized.organisation,
                "location": normalized.location,
                "state": normalized.state,
                "published_date": _date_to_dt(normalized.published_date),
                "closing_date": new_closing_date,
                "previous_closing_date": None,
                "deadline_changed": False,
                "tender_value": normalized.tender_value,
                "earnest_money": normalized.earnest_money,
                "document_url": normalized.document_url,
                "source_url": normalized.source_url,
                "description": normalized.description,
                "raw_data": normalized.raw_data,
                "first_seen": now,
                "last_seen": now,
                # Drives re-screening (see app.intelligence.scorer._needs_screening) -
                # deliberately separate from last_seen, which just means
                # "still live," not "content changed."
                "content_last_changed": now,
                "times_found": 1,
                "disappeared": False,
                "status": TenderStatus.NEW.value,
                "query_matches": [],
                "query_match_count": 0,
                "screenings": [],
            }
        else:
            doc = existing
            existing_closing = doc.get("closing_date")
            if (
                existing_closing is not None
                and new_closing_date is not None
                and not _same_instant(existing_closing, new_closing_date)
            ):
                doc["previous_closing_date"] = existing_closing
                doc["deadline_changed"] = True
                deadline_changed = True
                doc["content_last_changed"] = now

            if new_closing_date is not None:
                doc["closing_date"] = new_closing_date
            if normalized.tender_ref and not doc.get("tender_ref"):
                doc["tender_ref"] = normalized.tender_ref
                doc["ref_is_synthetic"] = False

            doc["title"] = normalized.title or doc.get("title")
            doc["organisation"] = normalized.organisation or doc.get("organisation")
            doc["location"] = normalized.location or doc.get("location")
            doc["state"] = normalized.state or doc.get("state")
            if normalized.tender_value is not None:
                doc["tender_value"] = normalized.tender_value
            if normalized.earnest_money is not None:
                doc["earnest_money"] = normalized.earnest_money
            doc["source_url"] = normalized.source_url or doc.get("source_url")
            doc["raw_data"] = normalized.raw_data
            doc["last_seen"] = now
            doc["times_found"] = doc.get("times_found", 0) + 1
            doc["disappeared"] = False

        tenders_seen_keys.add(dedup_key)
        if len(matched_queries) > 1:
            summary.cross_query_matches += 1

        query_matches = doc.get("query_matches", [])
        matches_by_name = {m["query_name"]: m for m in query_matches}
        for query_name in matched_queries:
            match = matches_by_name.get(query_name)
            if match is None:
                new_match = {"query_name": query_name, "first_seen": now, "last_seen": now}
                query_matches.append(new_match)
                matches_by_name[query_name] = new_match
            else:
                match["last_seen"] = now
        doc["query_matches"] = query_matches
        doc["query_match_count"] = len(query_matches)

        closing_date_for_check = _dt_to_date(doc.get("closing_date"))
        is_closing_soon = (
            closing_date_for_check is not None
            and 0 <= (closing_date_for_check - today).days <= CLOSING_SOON_DAYS
        )
        # Report counters are independent dimensions (a tender can be both
        # "new" and "closing soon" at once) - only the single `status` field
        # needs a priority order, since a tender can only have one status.
        if is_new:
            summary.new += 1
        elif deadline_changed:
            summary.updated += 1
        else:
            summary.unchanged += 1

        if is_closing_soon:
            summary.closing_soon += 1

        if is_closing_soon:
            doc["status"] = TenderStatus.CLOSING_SOON.value
        elif deadline_changed:
            doc["status"] = TenderStatus.UPDATED.value
        elif not is_new:
            doc["status"] = TenderStatus.SEEN.value

        operations.append(ReplaceOne({"dedup_key": dedup_key}, doc, upsert=True))

    if operations:
        collection.bulk_write(operations, ordered=False)

    _mark_disappeared(collection, query_results.keys(), tenders_seen_keys, summary)
    return summary


def _mark_disappeared(
    collection: Collection,
    processed_queries,
    tenders_seen_keys: set[str],
    summary: IngestSummary,
) -> None:
    processed_queries = set(processed_queries)
    if not processed_queries:
        return

    candidates = collection.find(
        {
            "query_matches.query_name": {"$in": list(processed_queries)},
            "dedup_key": {"$nin": list(tenders_seen_keys)},
            "disappeared": False,
        }
    )

    operations: list[UpdateOne] = []
    for doc in candidates:
        all_match_queries = {m["query_name"] for m in doc.get("query_matches", [])}
        # Only conclude "gone" if every query that ever matched it was
        # actually refreshed this pass - otherwise we just lack fresh data
        # for some of its queries, not evidence it disappeared.
        if all_match_queries and all_match_queries.issubset(processed_queries):
            operations.append(
                UpdateOne(
                    {"_id": doc["_id"]},
                    {"$set": {"disappeared": True, "status": TenderStatus.CLOSED.value}},
                )
            )
            summary.disappeared += 1

    if operations:
        collection.bulk_write(operations, ordered=False)


# --- Query helpers -----------------------------------------------------
# These answer the standing questions from the project spec and are reused
# by the Phase 6 report generator.


def get_new_since(since: dt.datetime, collection: Collection | None = None) -> list[dict]:
    """Tenders first seen at or after `since`."""
    collection = collection if collection is not None else get_collection()
    return list(collection.find({"first_seen": {"$gte": since}}))


def get_deadline_changed(collection: Collection | None = None) -> list[dict]:
    """Tenders whose closing date has ever changed since first seen."""
    collection = collection if collection is not None else get_collection()
    return list(collection.find({"deadline_changed": True}))


def get_cross_query_tenders(collection: Collection | None = None) -> list[dict]:
    """Tenders found by more than one saved query."""
    collection = collection if collection is not None else get_collection()
    return list(collection.find({"query_match_count": {"$gt": 1}}))


def get_closing_soon(
    days: int = CLOSING_SOON_DAYS, collection: Collection | None = None
) -> list[dict]:
    """Tenders closing within `days` days from today, not yet closed."""
    collection = collection if collection is not None else get_collection()
    today_midnight = dt.datetime.combine(
        dt.datetime.now(dt.timezone.utc).date(), dt.time.min
    ).replace(tzinfo=dt.timezone.utc)
    cutoff = today_midnight + dt.timedelta(days=days)
    return list(
        collection.find(
            {
                "closing_date": {"$ne": None, "$gte": today_midnight, "$lte": cutoff},
                "disappeared": False,
            }
        )
    )


def get_disappeared(collection: Collection | None = None) -> list[dict]:
    """Tenders no longer found in any of their previously-matching queries."""
    collection = collection if collection is not None else get_collection()
    return list(collection.find({"disappeared": True}))
