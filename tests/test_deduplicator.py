"""Tests for Phase 4: deduplication and history tracking.

Uses an in-memory SQLite database and hand-built NormalizedTender records -
no live TenderDetail login or downloaded files required.
"""
from __future__ import annotations

import datetime as dt

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

from app.models import Base, Tender, TenderStatus
from app.processing.deduplicator import (
    get_closing_soon,
    get_cross_query_tenders,
    get_deadline_changed,
    get_disappeared,
    get_new_since,
    ingest_batch,
)
from app.processing.normalizer import NormalizedTender


@pytest.fixture()
def session():
    engine = create_engine(
        "sqlite:///:memory:",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine)
    s = factory()
    yield s
    s.close()


def make_tender(
    dedup_key: str,
    title: str = "Title",
    org: str = "Org",
    closing_date: dt.date | None = None,
    tender_ref: str | None = None,
    query_name: str = "Q1",
) -> NormalizedTender:
    return NormalizedTender(
        tender_ref=tender_ref,
        ref_is_synthetic=tender_ref is None,
        dedup_key=dedup_key,
        query_name=query_name,
        title=title,
        organisation=org,
        closing_date=closing_date,
        raw_data={"TDR": tender_ref or "", "Title": title},
    )


FAR_FUTURE = dt.date.today() + dt.timedelta(days=60)
SOON = dt.date.today() + dt.timedelta(days=3)


# --- basic new / re-seen -------------------------------------------------


def test_ingest_new_tenders(session):
    batch = {
        "Q1": [
            make_tender("ref:1", title="T1", closing_date=FAR_FUTURE),
            make_tender("ref:2", title="T2", closing_date=FAR_FUTURE),
        ]
    }
    summary = ingest_batch(session, batch)
    session.commit()

    assert summary.new == 2
    assert summary.unchanged == 0
    tenders = session.query(Tender).all()
    assert len(tenders) == 2
    assert {t.status for t in tenders} == {TenderStatus.NEW}
    assert all(t.times_found == 1 for t in tenders)


def test_reingesting_same_data_marks_seen_and_increments_times_found(session):
    batch = {"Q1": [make_tender("ref:1", title="T1", closing_date=FAR_FUTURE)]}
    ingest_batch(session, batch)
    session.commit()

    summary = ingest_batch(session, batch)
    session.commit()

    assert summary.new == 0
    assert summary.unchanged == 1
    tender = session.query(Tender).filter_by(dedup_key="ref:1").one()
    assert tender.status == TenderStatus.SEEN
    assert tender.times_found == 2


# --- deadline change -------------------------------------------------


def test_deadline_change_is_detected(session):
    first_close = FAR_FUTURE
    second_close = FAR_FUTURE + dt.timedelta(days=10)

    ingest_batch(session, {"Q1": [make_tender("ref:1", closing_date=first_close)]})
    session.commit()

    summary = ingest_batch(session, {"Q1": [make_tender("ref:1", closing_date=second_close)]})
    session.commit()

    assert summary.updated == 1
    tender = session.query(Tender).filter_by(dedup_key="ref:1").one()
    assert tender.status == TenderStatus.UPDATED
    assert tender.deadline_changed is True
    assert tender.previous_closing_date == first_close
    assert tender.closing_date == second_close


def test_get_deadline_changed_helper(session):
    ingest_batch(session, {"Q1": [make_tender("ref:1", closing_date=FAR_FUTURE)]})
    session.commit()
    ingest_batch(
        session,
        {"Q1": [make_tender("ref:1", closing_date=FAR_FUTURE + dt.timedelta(days=5))]},
    )
    session.commit()

    changed = get_deadline_changed(session)
    assert len(changed) == 1
    assert changed[0].dedup_key == "ref:1"


# --- cross-query duplicates -------------------------------------------------


def test_tender_found_in_two_queries_is_merged_not_duplicated(session):
    batch = {
        "Digital Marketing": [make_tender("ref:1", closing_date=FAR_FUTURE)],
        "Awareness Campaign": [make_tender("ref:1", closing_date=FAR_FUTURE)],
    }
    summary = ingest_batch(session, batch)
    session.commit()

    assert summary.new == 1  # one Tender row, not two
    assert summary.cross_query_matches == 1

    tenders = session.query(Tender).all()
    assert len(tenders) == 1
    matched_query_names = {m.query_name for m in tenders[0].query_matches}
    assert matched_query_names == {"Digital Marketing", "Awareness Campaign"}


def test_get_cross_query_tenders_helper(session):
    batch = {
        "Digital Marketing": [make_tender("ref:1", closing_date=FAR_FUTURE)],
        "Awareness Campaign": [make_tender("ref:1", closing_date=FAR_FUTURE)],
        "Software Development": [make_tender("ref:2", closing_date=FAR_FUTURE)],
    }
    ingest_batch(session, batch)
    session.commit()

    cross = get_cross_query_tenders(session)
    assert len(cross) == 1
    assert cross[0].dedup_key == "ref:1"


# --- disappearance -------------------------------------------------


def test_tender_missing_from_reprocessed_query_is_marked_disappeared(session):
    ingest_batch(session, {"Q1": [make_tender("ref:1", closing_date=FAR_FUTURE)]})
    session.commit()

    # Q1 is processed again this pass, but no longer contains ref:1.
    summary = ingest_batch(session, {"Q1": []})
    session.commit()

    tender = session.query(Tender).filter_by(dedup_key="ref:1").one()
    assert tender.disappeared is True
    assert tender.status == TenderStatus.CLOSED
    assert summary.disappeared == 1


def test_tender_not_marked_disappeared_if_its_query_was_skipped_this_pass(session):
    ingest_batch(session, {"Q1": [make_tender("ref:1", closing_date=FAR_FUTURE, query_name="Q1")]})
    session.commit()

    # This pass only refreshes a different query - Q1 wasn't downloaded,
    # so we have no fresh evidence ref:1 is gone.
    summary = ingest_batch(session, {"Q2": [make_tender("ref:2", closing_date=FAR_FUTURE, query_name="Q2")]})
    session.commit()

    tender = session.query(Tender).filter_by(dedup_key="ref:1").one()
    assert tender.disappeared is False
    assert summary.disappeared == 0


def test_get_disappeared_helper(session):
    ingest_batch(session, {"Q1": [make_tender("ref:1", closing_date=FAR_FUTURE)]})
    session.commit()
    ingest_batch(session, {"Q1": []})
    session.commit()

    disappeared = get_disappeared(session)
    assert len(disappeared) == 1
    assert disappeared[0].dedup_key == "ref:1"


# --- closing soon -------------------------------------------------


def test_new_tender_that_is_also_closing_soon_counts_in_both(session):
    """Regression: closing-soon and new/updated/unchanged are independent
    counters - a brand new tender that also closes within 7 days must be
    counted in both `new` and `closing_soon`, not just one of them.
    """
    summary = ingest_batch(session, {"Q1": [make_tender("ref:1", closing_date=SOON)]})
    session.commit()

    assert summary.new == 1
    assert summary.closing_soon == 1
    tender = session.query(Tender).filter_by(dedup_key="ref:1").one()
    assert tender.status == TenderStatus.CLOSING_SOON


def test_closing_soon_status_and_helper(session):
    batch = {
        "Q1": [
            make_tender("ref:1", closing_date=SOON),
            make_tender("ref:2", closing_date=FAR_FUTURE),
        ]
    }
    summary = ingest_batch(session, batch)
    session.commit()

    assert summary.closing_soon == 1
    soon = get_closing_soon(session)
    assert len(soon) == 1
    assert soon[0].dedup_key == "ref:1"
    assert soon[0].status == TenderStatus.CLOSING_SOON


# --- "new since" helper -------------------------------------------------


def test_get_new_since_helper(session):
    before = dt.datetime.now(dt.timezone.utc)
    ingest_batch(session, {"Q1": [make_tender("ref:1", closing_date=FAR_FUTURE)]})
    session.commit()

    new_tenders = get_new_since(session, before)
    assert len(new_tenders) == 1

    future_cutoff = dt.datetime.now(dt.timezone.utc) + dt.timedelta(days=1)
    assert get_new_since(session, future_cutoff) == []


# --- never lose / never merge unrelated tenders -------------------------------------------------


def test_similar_titles_with_different_dedup_keys_are_never_merged(session):
    batch = {
        "Q1": [
            make_tender("ref:1", title="Same Title", org="Org A", closing_date=FAR_FUTURE),
            make_tender("ref:2", title="Same Title", org="Org A", closing_date=FAR_FUTURE),
        ]
    }
    summary = ingest_batch(session, batch)
    session.commit()

    assert summary.new == 2
    assert session.query(Tender).count() == 2
