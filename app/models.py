"""SQLAlchemy ORM models.

Schema overview
---------------
- Tender: one row per deduplicated tender (keyed by normalized tender_ref,
  falling back to a normalized title+organisation+closing_date key when no
  reference number is present in the source export).
- TenderQueryMatch: which saved queries have found a given tender, and what
  the raw row looked like the last time that query surfaced it. A tender can
  be matched by more than one saved query - this is how cross-query
  duplicates are detected instead of discarded.
- CollectionRun / DownloadRecord: one row per collector execution / per file
  download, so failures are recorded without crashing the whole workflow.
- ScreeningResult: the AI screening verdict for a tender, versioned by
  screened_at so re-screening never overwrites prior reasoning silently.
"""
from __future__ import annotations

import datetime as dt
import enum

from sqlalchemy import (
    JSON,
    Boolean,
    Date,
    DateTime,
    Enum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


def _utcnow() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


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


class Tender(Base):
    """Canonical, deduplicated tender record."""

    __tablename__ = "tenders"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)

    # Dedup key: normalized reference number, or normalized
    # title|organisation|closing_date when no reference number exists.
    dedup_key: Mapped[str] = mapped_column(String(512), unique=True, index=True)
    tender_ref: Mapped[str | None] = mapped_column(String(255), index=True)
    ref_is_synthetic: Mapped[bool] = mapped_column(Boolean, default=False)

    title: Mapped[str] = mapped_column(Text)
    organisation: Mapped[str | None] = mapped_column(Text)
    location: Mapped[str | None] = mapped_column(String(255))
    state: Mapped[str | None] = mapped_column(String(128))

    published_date: Mapped[dt.date | None] = mapped_column(Date)
    closing_date: Mapped[dt.date | None] = mapped_column(Date)
    previous_closing_date: Mapped[dt.date | None] = mapped_column(Date)
    deadline_changed: Mapped[bool] = mapped_column(Boolean, default=False)

    tender_value: Mapped[float | None] = mapped_column(Float)
    earnest_money: Mapped[float | None] = mapped_column(Float)

    document_url: Mapped[str | None] = mapped_column(Text)
    source_url: Mapped[str | None] = mapped_column(Text)
    description: Mapped[str | None] = mapped_column(Text)

    # Full original row(s), preserved verbatim for traceability.
    raw_data: Mapped[dict | None] = mapped_column(JSON)

    first_seen: Mapped[dt.datetime] = mapped_column(DateTime, default=_utcnow)
    last_seen: Mapped[dt.datetime] = mapped_column(DateTime, default=_utcnow)
    times_found: Mapped[int] = mapped_column(Integer, default=1)
    disappeared: Mapped[bool] = mapped_column(Boolean, default=False)

    status: Mapped[TenderStatus] = mapped_column(
        Enum(TenderStatus), default=TenderStatus.NEW
    )

    query_matches: Mapped[list["TenderQueryMatch"]] = relationship(
        back_populates="tender", cascade="all, delete-orphan"
    )
    screenings: Mapped[list["ScreeningResult"]] = relationship(
        back_populates="tender", cascade="all, delete-orphan"
    )


class TenderQueryMatch(Base):
    """Association between a Tender and a saved query that surfaced it."""

    __tablename__ = "tender_query_matches"
    __table_args__ = (UniqueConstraint("tender_id", "query_name"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    tender_id: Mapped[int] = mapped_column(ForeignKey("tenders.id"))
    query_name: Mapped[str] = mapped_column(String(255), index=True)
    first_seen: Mapped[dt.datetime] = mapped_column(DateTime, default=_utcnow)
    last_seen: Mapped[dt.datetime] = mapped_column(DateTime, default=_utcnow)

    tender: Mapped[Tender] = relationship(back_populates="query_matches")


class CollectionRun(Base):
    """One execution of the collector (a single `collect` invocation)."""

    __tablename__ = "collection_runs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    started_at: Mapped[dt.datetime] = mapped_column(DateTime, default=_utcnow)
    finished_at: Mapped[dt.datetime | None] = mapped_column(DateTime)
    success: Mapped[bool] = mapped_column(Boolean, default=False)
    notes: Mapped[str | None] = mapped_column(Text)

    downloads: Mapped[list["DownloadRecord"]] = relationship(
        back_populates="run", cascade="all, delete-orphan"
    )


class DownloadRecord(Base):
    """Status of a single saved-query export download within a run."""

    __tablename__ = "download_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("collection_runs.id"))
    query_name: Mapped[str] = mapped_column(String(255))
    file_path: Mapped[str | None] = mapped_column(Text)
    row_count: Mapped[int | None] = mapped_column(Integer)
    status: Mapped[str] = mapped_column(String(32))  # success | failed | skipped
    error_message: Mapped[str | None] = mapped_column(Text)
    downloaded_at: Mapped[dt.datetime] = mapped_column(DateTime, default=_utcnow)

    run: Mapped[CollectionRun] = relationship(back_populates="downloads")


class ScreeningResult(Base):
    """AI screening verdict for a tender (Phase 5)."""

    __tablename__ = "screening_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    tender_id: Mapped[int] = mapped_column(ForeignKey("tenders.id"))

    priority: Mapped[Priority] = mapped_column(Enum(Priority))
    relevance_score: Mapped[int] = mapped_column(Integer)
    category: Mapped[str | None] = mapped_column(String(255))
    reason: Mapped[str | None] = mapped_column(Text)
    eligibility_concerns: Mapped[list | None] = mapped_column(JSON)
    missing_information: Mapped[list | None] = mapped_column(JSON)
    recommended_action: Mapped[str | None] = mapped_column(Text)

    model_used: Mapped[str | None] = mapped_column(String(128))
    screened_at: Mapped[dt.datetime] = mapped_column(DateTime, default=_utcnow)

    tender: Mapped[Tender] = relationship(back_populates="screenings")
