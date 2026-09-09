"""Export the current tender database into the JSON shape the published
"Tender Ledger" dashboard artifact expects, ready to be pushed into its
shared db via Claude's Artifact write_db action.

This script only WRITES a local JSON file - it does not talk to the
artifact itself (only Claude's Artifact tool can do that). Run it, then
have Claude read the output file and push it via write_db.

Usage:
    .\\venv\\Scripts\\python.exe scripts\\export_dashboard.py
"""
from __future__ import annotations

import datetime as dt
import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.database import get_collection  # noqa: E402

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "dashboard_export.json"


def _sanitize_doc_id(dedup_key: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-.~:@+]", "_", dedup_key)[:200]


def _iso_date(value: dt.datetime | dt.date | None) -> str | None:
    """closing_date/published_date are stored as UTC-midnight datetimes."""
    if value is None:
        return None
    if isinstance(value, dt.datetime):
        return value.date().isoformat()
    return value.isoformat()


def _iso_dt(value: dt.datetime | None) -> str | None:
    return value.isoformat() if value else None


def _screening_summary(screenings: list[dict[str, Any]]) -> dict[str, Any] | None:
    """The most recent screening verdict (screenings is append-only, so the
    last element is always the latest), or None if never screened.
    """
    if not screenings:
        return None
    latest = screenings[-1]
    return {
        "priority": latest.get("priority"),
        "relevance_score": latest.get("relevance_score"),
        "category": latest.get("category"),
        "reason": latest.get("reason"),
        "recommended_action": latest.get("recommended_action"),
    }


def build_export() -> dict:
    collection = get_collection()
    tenders = list(collection.find({"disappeared": False}))

    docs = []
    counts: dict[str, int] = {}
    cross_query_matches = 0
    all_queries: set[str] = set()

    for t in tenders:
        query_names = sorted({m["query_name"] for m in t.get("query_matches", [])})
        all_queries.update(query_names)
        if len(query_names) > 1:
            cross_query_matches += 1

        status_value = t.get("status")
        counts[status_value] = counts.get(status_value, 0) + 1

        data = {
            "tender_ref": t.get("tender_ref"),
            "dedup_key": t["dedup_key"],
            "title": t.get("title"),
            "organisation": t.get("organisation"),
            "location": t.get("location"),
            "state": t.get("state"),
            "closing_date": _iso_date(t.get("closing_date")),
            "published_date": _iso_date(t.get("published_date")),
            "tender_value": t.get("tender_value"),
            "earnest_money": t.get("earnest_money"),
            "source_url": t.get("source_url"),
            "status": status_value,
            "times_found": t.get("times_found"),
            "first_seen": _iso_dt(t.get("first_seen")),
            "last_seen": _iso_dt(t.get("last_seen")),
            "disappeared": t.get("disappeared"),
            "deadline_changed": t.get("deadline_changed"),
            "query_names": query_names,
            "screening": _screening_summary(t.get("screenings", [])),
        }
        docs.append({"doc_id": _sanitize_doc_id(t["dedup_key"]), "data": data})

    meta = {
        "last_synced": dt.datetime.now(dt.timezone.utc).isoformat(),
        "total": len(tenders),
        "counts": counts,
        "cross_query_matches": cross_query_matches,
        "queries": sorted(all_queries),
    }

    return {"meta": meta, "tenders": docs}


def main() -> None:
    export = build_export()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(export, indent=2), encoding="utf-8")
    print(f"Wrote {len(export['tenders'])} tender documents + meta to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
