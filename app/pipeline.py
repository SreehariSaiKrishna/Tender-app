"""Pure orchestration functions shared by the CLI and the Lambda handler.

`app/cli.py` wraps these with Click argument parsing and formatted printing;
the Lambda handler (`app/lambda_handler.py`, added in a later step) calls
them directly. Keeping the business logic here means neither caller
duplicates it, and neither has to go through the other.
"""
from __future__ import annotations

import dataclasses
import datetime as dt
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from app.browser.collector import DownloadOutcome, run_collection
from app.browser.document_collector import DocumentDownloadSummary, run_document_collection
from app.config import Settings, load_saved_queries
from app.database import get_automation_runs_collection
from app.intelligence.document_summarizer import SummarizeSummary, summarize_pending_documents
from app.intelligence.scorer import ScreenSummary, ScreeningError, get_provider, screen_tenders
from app.processing.cleanup import CleanupSummary, cleanup_stale_closed_tenders
from app.processing.deduplicator import IngestSummary, ingest_batch
from app.processing.excel_reader import ExcelReadError, read_raw_rows
from app.processing.normalizer import NormalizedTender, normalize_rows

logger = logging.getLogger(__name__)


@dataclass
class QueryProcessResult:
    query_name: str
    status: str  # "ok" | "failed"
    row_count: int = 0
    synthetic_count: int = 0
    processed_path: str | None = None
    error_message: str | None = None


@dataclass
class ProcessResult:
    raw_dir: str = ""
    query_results: dict[str, QueryProcessResult] = field(default_factory=dict)
    ingest_summary: IngestSummary | None = None
    any_failed: bool = False


def run_process(settings: Settings, input_dir: str | None = None) -> ProcessResult:
    """Read the latest downloaded export per saved query, normalize it, and
    merge it into the database (dedup + history tracking).

    Also writes normalized JSON per query to the processed/ directory as a
    plain-text audit trail alongside the database.
    """
    raw_dir = Path(input_dir) if input_dir else settings.resolved_path(settings.download_dir)
    processed_dir = settings.resolved_path(settings.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)

    queries = [q.name for q in load_saved_queries() if q.enabled]

    files_by_query: dict[str, Path] = {}
    for query_name in queries:
        slug = re.sub(r"[^A-Za-z0-9]+", "_", query_name).strip("_")
        candidates = sorted(raw_dir.glob(f"{slug}_*.*"), key=lambda p: p.stat().st_mtime)
        if candidates:
            files_by_query[query_name] = candidates[-1]

    result = ProcessResult(raw_dir=str(raw_dir))
    if not files_by_query:
        return result

    normalized_by_query: dict[str, list[NormalizedTender]] = {}
    for query_name, path in files_by_query.items():
        try:
            raw_rows = read_raw_rows(path)
        except ExcelReadError as exc:
            result.any_failed = True
            result.query_results[query_name] = QueryProcessResult(
                query_name=query_name, status="failed", error_message=str(exc)
            )
            continue

        normalized = normalize_rows(raw_rows, query_name)
        normalized_by_query[query_name] = normalized
        synthetic = sum(1 for n in normalized if n.ref_is_synthetic)

        out_path = processed_dir / f"{path.stem}.json"
        out_path.write_text(
            json.dumps([n.model_dump(mode="json") for n in normalized], indent=2),
            encoding="utf-8",
        )
        result.query_results[query_name] = QueryProcessResult(
            query_name=query_name,
            status="ok",
            row_count=len(normalized),
            synthetic_count=synthetic,
            processed_path=str(out_path),
        )

    if normalized_by_query:
        result.ingest_summary = ingest_batch(normalized_by_query)

    return result


@dataclass
class ScreenOutcome:
    summary: ScreenSummary | None = None
    error: str | None = None


def run_screen(
    settings: Settings, limit: int | None = None, only_unscreened: bool = True
) -> ScreenOutcome:
    """Run AI screening against configured business capabilities."""
    try:
        provider = get_provider(settings)
    except ScreeningError as exc:
        return ScreenOutcome(error=str(exc))

    summary = screen_tenders(provider=provider, limit=limit, only_unscreened=only_unscreened)
    return ScreenOutcome(summary=summary)


@dataclass
class CleanupOutcome:
    summary: CleanupSummary | None = None
    error: str | None = None


def run_cleanup(settings: Settings) -> CleanupOutcome:
    """Delete tenders that are confirmed closed, past their closing_date by
    the grace period, and never marked applied (see
    app.processing.cleanup). Errors are caught here (like run_screen) so a
    cleanup failure never fails the whole pipeline run.
    """
    try:
        return CleanupOutcome(summary=cleanup_stale_closed_tenders())
    except Exception as exc:  # noqa: BLE001
        logger.exception("Cleanup step failed")
        return CleanupOutcome(error=str(exc))


@dataclass
class DocumentDownloadOutcome:
    summary: DocumentDownloadSummary | None = None
    error: str | None = None


def run_document_download(settings: Settings) -> DocumentDownloadOutcome:
    """Download attached documents (Tender Document/BOQ/Notice) for every
    eligible tender that doesn't have them yet. Errors are caught here
    (like run_screen/run_cleanup) so a failure never fails the whole
    pipeline run.
    """
    try:
        return DocumentDownloadOutcome(summary=run_document_collection(settings))
    except Exception as exc:  # noqa: BLE001
        logger.exception("Document download step failed")
        return DocumentDownloadOutcome(error=str(exc))


@dataclass
class DocumentSummarizeOutcome:
    summary: SummarizeSummary | None = None
    error: str | None = None


def run_document_summarize(settings: Settings) -> DocumentSummarizeOutcome:
    """Summarize every tender that has downloaded documents but no
    up-to-date AI summary yet. Errors are caught here so a failure never
    fails the whole pipeline run.
    """
    try:
        return DocumentSummarizeOutcome(summary=summarize_pending_documents(settings))
    except Exception as exc:  # noqa: BLE001
        logger.exception("Document summarize step failed")
        return DocumentSummarizeOutcome(error=str(exc))


@dataclass
class PipelineSummary:
    collect_outcomes: list[DownloadOutcome] = field(default_factory=list)
    process_result: ProcessResult | None = None
    screen_outcome: ScreenOutcome | None = None
    cleanup_outcome: CleanupOutcome | None = None
    document_download_outcome: DocumentDownloadOutcome | None = None
    document_summarize_outcome: DocumentSummarizeOutcome | None = None
    reported: bool = False


def run_pipeline(
    settings: Settings, include_report: bool = False, trigger: str = "manual"
) -> PipelineSummary:
    """Run the full pipeline: collect -> process -> cleanup ->
    download_documents -> summarize_documents -> screen -> (report).

    This is the single function both `python main.py run` and the Lambda
    handler call, so neither duplicates the orchestration. `trigger`
    ("manual" | "scheduled") is recorded alongside the run so the dashboard's
    automation history can tell a CLI run from an EventBridge-triggered one.
    """
    summary = PipelineSummary()
    started_at = dt.datetime.now(dt.timezone.utc)

    summary.collect_outcomes = run_collection(settings)
    summary.process_result = run_process(settings)
    # Runs right after ingestion so a tender newly marked `disappeared` this
    # same pass is already eligible for cleanup.
    summary.cleanup_outcome = run_cleanup(settings)
    # Runs after cleanup so this pass's eligibility_match (set during
    # ingestion above) is already in place before deciding what to download.
    summary.document_download_outcome = run_document_download(settings)
    summary.document_summarize_outcome = run_document_summarize(settings)
    summary.screen_outcome = run_screen(settings)

    if include_report:
        # Phase 6 - app/reports/ is still empty; nothing to call yet.
        logger.info("Report requested but Phase 6 (app/reports/) isn't built yet - skipping.")
        summary.reported = False

    _persist_run(summary, started_at, trigger)

    return summary


def _persist_run(summary: PipelineSummary, started_at: dt.datetime, trigger: str) -> None:
    """Record what this run did/found, for the dashboard's automation
    history tab. Best-effort: a logging failure here must never fail the
    pipeline run itself, since the run already succeeded by this point.
    """
    finished_at = dt.datetime.now(dt.timezone.utc)
    any_screen_error = bool(summary.screen_outcome and summary.screen_outcome.error)
    any_cleanup_error = bool(summary.cleanup_outcome and summary.cleanup_outcome.error)
    any_document_download_error = bool(
        summary.document_download_outcome and summary.document_download_outcome.error
    )
    any_document_summarize_error = bool(
        summary.document_summarize_outcome and summary.document_summarize_outcome.error
    )
    status = (
        "failed"
        if (summary.process_result and summary.process_result.any_failed)
        or any_screen_error
        or any_cleanup_error
        or any_document_download_error
        or any_document_summarize_error
        else "ok"
    )

    doc = {
        "started_at": started_at,
        "finished_at": finished_at,
        "duration_seconds": (finished_at - started_at).total_seconds(),
        "trigger": trigger,
        "status": status,
        "collect_outcomes": [dataclasses.asdict(o) for o in summary.collect_outcomes],
        "process_result": dataclasses.asdict(summary.process_result) if summary.process_result else None,
        "cleanup_outcome": dataclasses.asdict(summary.cleanup_outcome) if summary.cleanup_outcome else None,
        "document_download_outcome": (
            dataclasses.asdict(summary.document_download_outcome)
            if summary.document_download_outcome
            else None
        ),
        "document_summarize_outcome": (
            dataclasses.asdict(summary.document_summarize_outcome)
            if summary.document_summarize_outcome
            else None
        ),
        "screen_outcome": dataclasses.asdict(summary.screen_outcome) if summary.screen_outcome else None,
        "reported": summary.reported,
    }

    try:
        get_automation_runs_collection().insert_one(doc)
    except Exception:  # noqa: BLE001
        logger.exception("Failed to persist automation run history - continuing anyway.")
