"""Deletes tenders that have sat closed and unapplied for too long.

"Closed" here means scraper-confirmed gone (`disappeared: true`, set by
app.processing.deduplicator._mark_disappeared) - not merely past its
closing_date, since a tender TenderDetail still shows live shouldn't be
deleted just because its nominal deadline passed. Marking a tender
`applied` (see the POST /tenders/{id}/apply endpoint in app.api.main) is
what protects it from this cleanup indefinitely.

This is a permanent, hard delete - there is no soft-delete/archive step.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass

from pymongo.collection import Collection

from app.database import get_collection

CLEANUP_GRACE_DAYS = 7


@dataclass
class CleanupSummary:
    deleted: int = 0


def cleanup_stale_closed_tenders(
    collection: Collection | None = None, grace_days: int = CLEANUP_GRACE_DAYS
) -> CleanupSummary:
    collection = collection if collection is not None else get_collection()
    cutoff = dt.datetime.now(dt.timezone.utc) - dt.timedelta(days=grace_days)
    result = collection.delete_many(
        {
            "disappeared": True,
            "closing_date": {"$ne": None, "$lte": cutoff},
            "applied": {"$ne": True},
        }
    )
    return CleanupSummary(deleted=result.deleted_count)
