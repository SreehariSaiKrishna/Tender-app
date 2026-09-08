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

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import get_settings  # noqa: E402
from app.database import session_scope  # noqa: E402
from app.models import Tender  # noqa: E402

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "dashboard_export.json"


def _sanitize_doc_id(dedup_key: str) -> str:
    return re.sub(r"[^A-Za-z0-9_\-.~:@+]", "_", dedup_key)[:200]


def _iso_date(d: dt.date | None) -> str | None:
    return d.isoformat() if d else None


def _iso_dt(d: dt.datetime | None) -> str | None:
    return d.isoformat() if d else None


def build_export() -> dict:
    with session_scope() as session:
        tenders = session.query(Tender).filter(Tender.disappeared.is_(False)).all()

        docs = []
        counts: dict[str, int] = {}
        cross_query_matches = 0
        all_queries: set[str] = set()

        for t in tenders:
            query_names = sorted({m.query_name for m in t.query_matches})
            all_queries.update(query_names)
            if len(query_names) > 1:
                cross_query_matches += 1

            status_value = t.status.value if hasattr(t.status, "value") else str(t.status)
            counts[status_value] = counts.get(status_value, 0) + 1

            data = {
                "tender_ref": t.tender_ref,
                "dedup_key": t.dedup_key,
                "title": t.title,
                "organisation": t.organisation,
                "location": t.location,
                "state": t.state,
                "closing_date": _iso_date(t.closing_date),
                "published_date": _iso_date(t.published_date),
                "tender_value": t.tender_value,
                "earnest_money": t.earnest_money,
                "source_url": t.source_url,
                "status": status_value,
                "times_found": t.times_found,
                "first_seen": _iso_dt(t.first_seen),
                "last_seen": _iso_dt(t.last_seen),
                "disappeared": t.disappeared,
                "deadline_changed": t.deadline_changed,
                "query_names": query_names,
                # Filled in once Phase 5 (AI screening) is built.
                "screening": None,
            }
            docs.append({"doc_id": _sanitize_doc_id(t.dedup_key), "data": data})

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
