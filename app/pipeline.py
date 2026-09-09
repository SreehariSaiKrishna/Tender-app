"""Pure orchestration functions shared by the CLI and the Lambda handler.

`app/cli.py` wraps these with Click argument parsing and formatted printing;
the Lambda handler (`app/lambda_handler.py`, added in a later step) calls
them directly. Keeping the business logic here means neither caller
duplicates it, and neither has to go through the other.
"""
from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from app.browser.collector import DownloadOutcome, run_collection
from app.config import Settings, load_saved_queries
from app.intelligence.scorer import ScreenSummary, ScreeningError, get_provider, screen_tenders
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
class PipelineSummary:
    collect_outcomes: list[DownloadOutcome] = field(default_factory=list)
    process_result: ProcessResult | None = None
    screen_outcome: ScreenOutcome | None = None
    reported: bool = False


def run_pipeline(settings: Settings, include_report: bool = False) -> PipelineSummary:
    """Run the full pipeline: collect -> process -> screen -> (report).

    This is the single function both `python main.py run` and the Lambda
    handler call, so neither duplicates the orchestration.
    """
    summary = PipelineSummary()

    summary.collect_outcomes = run_collection(settings)
    summary.process_result = run_process(settings)
    summary.screen_outcome = run_screen(settings)

    if include_report:
        # Phase 6 - app/reports/ is still empty; nothing to call yet.
        logger.info("Report requested but Phase 6 (app/reports/) isn't built yet - skipping.")
        summary.reported = False

    return summary
