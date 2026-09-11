"""One-off backfill: set `latest_priority_rank` on every existing tender
from its current `latest_priority` (see app.models.priority_rank), for
tenders screened before that field existed. New/re-screened tenders get it
automatically via app.intelligence.scorer.screen_tenders - this script only
needs to run once against already-screened data.

Required, not optional: without it, an already-screened tender would sort
as if unscreened (see GET /tenders' default sort in app.api.main) until it
happens to be re-screened.

Usage:
    .\\venv\\Scripts\\python.exe scripts\\backfill_priority_rank.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pymongo import UpdateOne  # noqa: E402

from app.database import get_collection  # noqa: E402
from app.models import priority_rank  # noqa: E402


def main() -> None:
    collection = get_collection()

    operations: list[UpdateOne] = []
    for doc in collection.find({}, {"latest_priority": 1}):
        rank = priority_rank(doc.get("latest_priority"))
        operations.append(
            UpdateOne({"_id": doc["_id"]}, {"$set": {"latest_priority_rank": rank}})
        )

    if operations:
        result = collection.bulk_write(operations, ordered=False)
        print(f"Updated {result.modified_count}/{len(operations)} tenders.")
    else:
        print("No tenders found.")


if __name__ == "__main__":
    main()
