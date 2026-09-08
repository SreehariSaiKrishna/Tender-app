"""Deduplication and history tracking (Phase 4).

Ingests one "pass" of normalized tenders - the latest full export per saved
query - and merges it into the Tender table:

  - Matches tenders by dedup_key (the TDR reference number when available,
    else a normalized title+organisation+closing_date fallback). Two
    tenders are only ever merged if they share the same dedup_key; similar
    -looking titles are never treated as duplicates of each other.
  - Tracks first_seen, last_seen, times_found.
  - Records every saved query that has ever matched a tender
    (TenderQueryMatch), so cross-query duplicates are detected instead of
    silently discarded.
  - Detects a changed closing date (previous vs current).
  - Detects tenders that disappeared: only for queries that were actually
    part of this pass - a query that wasn't downloaded this run (skipped/
    failed) never causes its previously-tracked tenders to be marked gone,
    since we simply have no fresh data for it.
  - Never deletes a row. A tender that disappears is marked, not removed.
"""
from __future__ import annotations

import datetime as dt
import logging
from dataclasses import dataclass, field

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models import Tender, TenderQueryMatch, TenderStatus
from app.processing.normalizer import NormalizedTender

logger = logging.getLogger(__name__)

CLOSING_SOON_DAYS = 7


def _utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


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
    session: Session, query_results: dict[str, list[NormalizedTender]]
) -> IngestSummary:
    """Merge one pass's worth of per-query exports into the Tender table.

    `query_results` must map query_name -> the FULL latest export for that
    query (what "Download Excel" produces). Omit a query entirely if it
    couldn't be downloaded this pass - its history is left untouched.
    """
    summary = IngestSummary(queries_processed=list(query_results.keys()))
    now = _utcnow()
    today = now.date()

    keys_this_pass: dict[str, set[str]] = {}
    latest_by_key: dict[str, NormalizedTender] = {}
    for query_name, tenders in query_results.items():
        for t in tenders:
            keys_this_pass.setdefault(t.dedup_key, set()).add(query_name)
            latest_by_key[t.dedup_key] = t

    tenders_seen_ids: set[int] = set()

    for dedup_key, normalized in latest_by_key.items():
        matched_queries = keys_this_pass[dedup_key]
        existing = session.query(Tender).filter_by(dedup_key=dedup_key).one_or_none()
        deadline_changed = False
        is_new = existing is None

        if existing is None:
            tender = Tender(
                dedup_key=dedup_key,
                tender_ref=normalized.tender_ref,
                ref_is_synthetic=normalized.ref_is_synthetic,
                title=normalized.title,
                organisation=normalized.organisation,
                location=normalized.location,
                state=normalized.state,
                published_date=normalized.published_date,
                closing_date=normalized.closing_date,
                tender_value=normalized.tender_value,
                earnest_money=normalized.earnest_money,
                document_url=normalized.document_url,
                source_url=normalized.source_url,
                description=normalized.description,
                raw_data=normalized.raw_data,
                first_seen=now,
                last_seen=now,
                times_found=1,
                disappeared=False,
                status=TenderStatus.NEW,
            )
            session.add(tender)
            session.flush()  # obtain tender.id for TenderQueryMatch rows below
        else:
            tender = existing
            if (
                tender.closing_date is not None
                and normalized.closing_date is not None
                and tender.closing_date != normalized.closing_date
            ):
                tender.previous_closing_date = tender.closing_date
                tender.deadline_changed = True
                deadline_changed = True

            if normalized.closing_date is not None:
                tender.closing_date = normalized.closing_date
            if normalized.tender_ref and not tender.tender_ref:
                tender.tender_ref = normalized.tender_ref
                tender.ref_is_synthetic = False

            tender.title = normalized.title or tender.title
            tender.organisation = normalized.organisation or tender.organisation
            tender.location = normalized.location or tender.location
            tender.state = normalized.state or tender.state
            if normalized.tender_value is not None:
                tender.tender_value = normalized.tender_value
            if normalized.earnest_money is not None:
                tender.earnest_money = normalized.earnest_money
            tender.source_url = normalized.source_url or tender.source_url
            tender.raw_data = normalized.raw_data
            tender.last_seen = now
            tender.times_found += 1
            tender.disappeared = False

        tenders_seen_ids.add(tender.id)
        if len(matched_queries) > 1:
            summary.cross_query_matches += 1

        for query_name in matched_queries:
            match = (
                session.query(TenderQueryMatch)
                .filter_by(tender_id=tender.id, query_name=query_name)
                .one_or_none()
            )
            if match is None:
                session.add(
                    TenderQueryMatch(
                        tender_id=tender.id,
                        query_name=query_name,
                        first_seen=now,
                        last_seen=now,
                    )
                )
            else:
                match.last_seen = now

        is_closing_soon = (
            tender.closing_date is not None
            and 0 <= (tender.closing_date - today).days <= CLOSING_SOON_DAYS
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
            tender.status = TenderStatus.CLOSING_SOON
        elif deadline_changed:
            tender.status = TenderStatus.UPDATED
        elif not is_new:
            tender.status = TenderStatus.SEEN

    _mark_disappeared(session, query_results.keys(), tenders_seen_ids, summary)
    return summary


def _mark_disappeared(
    session: Session,
    processed_queries,
    tenders_seen_ids: set[int],
    summary: IngestSummary,
) -> None:
    processed_queries = set(processed_queries)
    if not processed_queries:
        return

    stale_matches = (
        session.query(TenderQueryMatch)
        .filter(TenderQueryMatch.query_name.in_(processed_queries))
        .all()
    )
    candidate_tender_ids = {m.tender_id for m in stale_matches} - tenders_seen_ids

    for tender_id in candidate_tender_ids:
        tender = session.get(Tender, tender_id)
        if tender is None or tender.disappeared:
            continue
        all_match_queries = {
            m.query_name
            for m in session.query(TenderQueryMatch).filter_by(tender_id=tender_id)
        }
        # Only conclude "gone" if every query that ever matched it was
        # actually refreshed this pass - otherwise we just lack fresh data
        # for some of its queries, not evidence it disappeared.
        if all_match_queries and all_match_queries.issubset(processed_queries):
            tender.disappeared = True
            tender.status = TenderStatus.CLOSED
            summary.disappeared += 1


# --- Query helpers -----------------------------------------------------
# These answer the standing questions from the project spec and are reused
# by the Phase 6 report generator.


def get_new_since(session: Session, since: dt.datetime) -> list[Tender]:
    """Tenders first seen at or after `since`."""
    return session.query(Tender).filter(Tender.first_seen >= since).all()


def get_deadline_changed(session: Session) -> list[Tender]:
    """Tenders whose closing date has ever changed since first seen."""
    return session.query(Tender).filter(Tender.deadline_changed.is_(True)).all()


def get_cross_query_tenders(session: Session) -> list[Tender]:
    """Tenders found by more than one saved query."""
    subq = (
        select(TenderQueryMatch.tender_id)
        .group_by(TenderQueryMatch.tender_id)
        .having(func.count(TenderQueryMatch.query_name) > 1)
    )
    return session.query(Tender).filter(Tender.id.in_(subq)).all()


def get_closing_soon(session: Session, days: int = CLOSING_SOON_DAYS) -> list[Tender]:
    """Tenders closing within `days` days from today, not yet closed."""
    today = dt.datetime.now(dt.timezone.utc).date()
    cutoff = today + dt.timedelta(days=days)
    return (
        session.query(Tender)
        .filter(
            Tender.closing_date.is_not(None),
            Tender.closing_date >= today,
            Tender.closing_date <= cutoff,
            Tender.disappeared.is_(False),
        )
        .all()
    )


def get_disappeared(session: Session) -> list[Tender]:
    """Tenders no longer found in any of their previously-matching queries."""
    return session.query(Tender).filter(Tender.disappeared.is_(True)).all()
