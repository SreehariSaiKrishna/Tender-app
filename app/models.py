"""Shared value types for tender documents stored in MongoDB.

Schema overview (see app/database.py for the collection/index setup)
---------------------------------------------------------------------
A single `tenders` collection holds one document per deduplicated tender
(keyed by `dedup_key` - the normalized tender_ref, falling back to a
normalized title+organisation+closing_date key when no reference number is
present in the source export). Two arrays are embedded directly in each
document rather than living in separate collections, since every existing
read of them is already scoped to a single tender:

  - `query_matches`: which saved queries have found this tender, and when.
    A tender can be matched by more than one saved query - this is how
    cross-query duplicates are detected instead of discarded.
  - `screenings`: the AI screening verdict history for this tender, appended
    (never overwritten) so re-screening never silently loses prior
    reasoning. The last element is always the most recent screening.

Per-run collector history (what used to be CollectionRun/DownloadRecord
rows) is not persisted to MongoDB - nothing in the app ever reads it back
programmatically, so it's written as structured log lines instead (see
app/browser/collector.py).
"""
from __future__ import annotations

import enum


class TenderStatus(str, enum.Enum):
    NEW = "new"
    SEEN = "seen"
    UPDATED = "updated"
    CLOSING_SOON = "closing_soon"
    CLOSED = "closed"


class Priority(str, enum.Enum):
    HIGH = "High"
    MEDIUM = "Medium"
    LOW = "Low"
    NOT_RELEVANT = "Not Relevant"


# Denormalized onto tender docs as `latest_priority_rank` (see
# app.intelligence.scorer.screen_tenders and
# app.processing.deduplicator.ingest_batch) so MongoDB can sort by priority
# directly - the raw string sorts alphabetically ("High" < "Low" < "Medium"),
# not by actual priority.
PRIORITY_RANK: dict[str, int] = {
    Priority.HIGH.value: 0,
    Priority.MEDIUM.value: 1,
    Priority.LOW.value: 2,
    Priority.NOT_RELEVANT.value: 3,
}
UNSCREENED_PRIORITY_RANK = 4


def priority_rank(priority: str | None) -> int:
    if priority is None:
        return UNSCREENED_PRIORITY_RANK
    return PRIORITY_RANK.get(priority, UNSCREENED_PRIORITY_RANK)
