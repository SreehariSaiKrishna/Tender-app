"""One-off backfill: tag every existing tender document with
`eligibility_match` (see app.processing.eligibility), for tenders stored
before that field existed. New/re-ingested tenders get it automatically via
app.processing.deduplicator.ingest_batch - this script only needs to run
once against already-collected data.

Usage:
    .\\venv\\Scripts\\python.exe scripts\\backfill_eligibility.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pymongo import UpdateOne  # noqa: E402

from app.database import get_collection  # noqa: E402
from app.processing.eligibility import (  # noqa: E402
    compute_eligibility_match,
    load_match_keywords,
)


def main() -> None:
    collection = get_collection()
    keywords = load_match_keywords()

    operations: list[UpdateOne] = []
    matched = 0
    for doc in collection.find({}, {"title": 1, "description": 1}):
        text = f"{doc.get('title') or ''} {doc.get('description') or ''}"
        is_match = compute_eligibility_match(text, keywords)
        matched += is_match
        operations.append(
            UpdateOne({"_id": doc["_id"]}, {"$set": {"eligibility_match": is_match}})
        )

    if operations:
        result = collection.bulk_write(operations, ordered=False)
        print(
            f"Updated {result.modified_count}/{len(operations)} tenders "
            f"({matched} matched the eligibility criteria)."
        )
    else:
        print("No tenders found.")


if __name__ == "__main__":
    main()
