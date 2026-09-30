"""Deletes tenders that have sat closed for too long - with everything kept
about them, including their document text in file storage (app.storage).

"Closed" here means scraper-confirmed gone (`disappeared: true`, set by
app.processing.deduplicator._mark_disappeared) - not merely past its
closing_date, since a tender TenderDetail still shows live shouldn't be
deleted just because its nominal deadline passed. Nothing protects a closed
tender past the grace period, applied or not: once it's closed its data is
no longer needed. A closed tender with no closing_date is counted from when
it disappeared (`disappeared_at`) instead.

This is a permanent, hard delete - there is no soft-delete/archive step.
"""
from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass

from pymongo.collection import Collection

from app import storage
from app.database import get_collection

logger = logging.getLogger(__name__)

CLEANUP_GRACE_DAYS = 7


@dataclass
class CleanupSummary:
    deleted: int = 0


def cleanup_stale_closed_tenders(
    collection: Collection | None = None, grace_days: int = CLEANUP_GRACE_DAYS
) -> CleanupSummary:
    collection = collection if collection is not None else get_collection()
    cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=grace_days)
    stale = list(
        collection.find(
            {
                "disappeared": True,
                "$or": [
                    {"closing_date": {"$ne": None, "$lte": cutoff}},
                    {"closing_date": None, "disappeared_at": {"$ne": None, "$lte": cutoff}},
                ],
            },
            {"_id": 1, "document_text_key": 1},
        )
    )
    if not stale:
        return CleanupSummary()

    for doc in stale:
        if doc.get("document_text_key"):
            try:
                storage.delete(doc["document_text_key"])
            except Exception as exc:  # noqa: BLE001 - an orphaned text file mustn't keep the tender
                logger.warning("Could not delete document text for tender %s: %s", doc["_id"], exc)

    result = collection.delete_many({"_id": {"$in": [doc["_id"] for doc in stale]}})
    return CleanupSummary(deleted=result.deleted_count)
