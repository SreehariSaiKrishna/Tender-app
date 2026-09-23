"""One-off backfill: tag every existing tender document with
`domain_match` and `eligibility_match` (see app.processing.eligibility), for
tenders stored before those fields existed, or before the value-range
criteria/tender-fee field were added. New/re-ingested tenders get both
automatically via app.processing.deduplicator.ingest_batch, and
eligibility_match is re-decided again once a tender's documents are
summarized (see app.intelligence.document_summarizer) - this script only
needs to run once against already-collected data, or after a criteria
change like this one.

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
    emd_in_range,
    load_match_keywords,
    tender_fee_in_range,
    tender_value_in_range,
)
from app.processing.normalizer import parse_amount_from_text  # noqa: E402


def main() -> None:
    collection = get_collection()
    keywords = load_match_keywords()

    operations: list[UpdateOne] = []
    matched = 0
    fields = {
        "title": 1,
        "description": 1,
        "tender_value": 1,
        "earnest_money": 1,
        "document_fees": 1,
        "document_summary": 1,
    }
    for doc in collection.find({}, fields):
        text = f"{doc.get('title') or ''} {doc.get('description') or ''}"
        domain_match = compute_eligibility_match(text, keywords)

        # A value the listing itself doesn't disclose may still be known
        # from that tender's own documents, once summarized - see
        # app.intelligence.document_summarizer.summarize_pending_documents,
        # which applies this same merge going forward.
        summary = doc.get("document_summary") or {}

        # Same OR-in-document-text widening as summarize_pending_documents:
        # the listing's title/description alone can miss a domain match
        # that the tender's own RFP spells out clearly (e.g. a vague listing
        # title whose "Eligibility Criteria" section names EdTech/software
        # development explicitly).
        technical_criteria_table = summary.get("technical_criteria_table") or []
        document_domain_text = " ".join(
            [
                *(summary.get("eligibility_requirements") or []),
                *(summary.get("eligibility_technical_criteria") or []),
                *(row.get("criterion") or "" for row in technical_criteria_table),
                *(row.get("expected_evidence") or "" for row in technical_criteria_table),
            ]
        )
        domain_match = domain_match or compute_eligibility_match(document_domain_text, keywords)

        tender_value = doc.get("tender_value")
        if tender_value is None:
            tender_value = parse_amount_from_text(summary.get("estimated_bid_amount"))
        earnest_money = doc.get("earnest_money")
        if earnest_money is None:
            earnest_money = parse_amount_from_text(summary.get("emd_amount"))
        document_fees = doc.get("document_fees")
        if document_fees is None:
            document_fees = parse_amount_from_text(summary.get("tender_fee_amount"))

        is_match = (
            domain_match
            and tender_value_in_range(tender_value)
            and emd_in_range(earnest_money)
            and tender_fee_in_range(document_fees)
        )
        matched += is_match
        operations.append(
            UpdateOne(
                {"_id": doc["_id"]},
                {"$set": {"domain_match": domain_match, "eligibility_match": is_match}},
            )
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
