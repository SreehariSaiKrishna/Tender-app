"""Collector orchestration: log in, run each saved query, download its Live
export, save it with a timestamp, and record status per query - without
letting one failure crash the whole run.
"""
from __future__ import annotations

import datetime as dt
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from playwright.sync_api import sync_playwright

from app.browser.dashboard import (
    QueryNotFoundError,
    download_excel_export,
    has_zero_results,
    open_live_results,
)
from app.browser.login import ensure_logged_in
from app.config import Settings, get_settings, load_saved_queries
from app.database import session_scope
from app.models import CollectionRun, DownloadRecord

logger = logging.getLogger(__name__)


@dataclass
class DownloadOutcome:
    query_name: str
    status: str  # "success" | "failed" | "skipped"
    file_path: str | None = None
    error_message: str | None = None


def _slugify(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9]+", "_", name).strip("_")
    return slug or "query"


def _timestamp() -> str:
    return dt.datetime.now().strftime("%Y%m%d_%H%M%S")


def run_collection(settings: Settings | None = None) -> list[DownloadOutcome]:
    """Run the full collector: login, then download each enabled saved
    query's Live export. Returns per-query outcomes. Never raises for an
    individual query's failure - only for login failure or a fatal browser
    error, which stop the whole run safely.
    """
    settings = settings or get_settings()
    queries = [q for q in load_saved_queries() if q.enabled]

    download_dir = settings.resolved_path(settings.download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)
    profile_dir = settings.resolved_path(settings.tenderdetail_profile_dir)
    profile_dir.mkdir(parents=True, exist_ok=True)

    outcomes: list[DownloadOutcome] = []

    with session_scope() as session:
        run = CollectionRun()
        session.add(run)
        session.flush()  # obtain run.id
        run_id = run.id

        with sync_playwright() as p:
            context = p.chromium.launch_persistent_context(
                str(profile_dir),
                headless=settings.browser_headless,
                viewport={"width": 1400, "height": 900},
                accept_downloads=True,
            )
            page = context.new_page()

            try:
                logged_in = ensure_logged_in(
                    page,
                    settings.tenderdetail_login_url,
                    username=settings.tenderdetail_username,
                    password=settings.tenderdetail_password,
                )
            except Exception as exc:  # noqa: BLE001 - fatal, stop safely
                logger.error("Login raised an unexpected error: %s", exc)
                logged_in = False

            if not logged_in:
                run.success = False
                run.notes = "Login failed or was not completed in time."
                context.close()
                logger.error("Stopping: could not confirm login.")
                return outcomes

            for query in queries:
                outcome = _collect_one_query(page, settings, query.name, download_dir)
                outcomes.append(outcome)
                session.add(
                    DownloadRecord(
                        run_id=run_id,
                        query_name=outcome.query_name,
                        file_path=outcome.file_path,
                        status=outcome.status,
                        error_message=outcome.error_message,
                    )
                )
                # Return to the dashboard before the next query.
                page.goto(settings.tenderdetail_login_url)

            context.close()

        run.success = any(o.status == "success" for o in outcomes)
        run.finished_at = dt.datetime.now(dt.timezone.utc)

    return outcomes


def _collect_one_query(
    page, settings: Settings, query_name: str, download_dir: Path
) -> DownloadOutcome:
    try:
        open_live_results(page, query_name)
    except QueryNotFoundError as exc:
        logger.error("Skipping '%s': %s", query_name, exc)
        return DownloadOutcome(query_name=query_name, status="failed", error_message=str(exc))
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed opening results for '%s': %s", query_name, exc)
        return DownloadOutcome(query_name=query_name, status="failed", error_message=str(exc))

    if has_zero_results(page):
        logger.info("'%s' has no live tenders right now - skipping.", query_name)
        return DownloadOutcome(
            query_name=query_name,
            status="skipped",
            error_message="No live tenders currently (0 results).",
        )

    try:
        download = download_excel_export(page)
    except Exception as exc:  # noqa: BLE001
        logger.error("Download failed for '%s': %s", query_name, exc)
        return DownloadOutcome(query_name=query_name, status="failed", error_message=str(exc))

    suggested = download.suggested_filename or f"{_slugify(query_name)}.csv"
    suffix = Path(suggested).suffix or ".csv"
    target = download_dir / f"{_slugify(query_name)}_{_timestamp()}{suffix}"

    try:
        download.save_as(str(target))
    except Exception as exc:  # noqa: BLE001
        logger.error("Could not save download for '%s': %s", query_name, exc)
        return DownloadOutcome(query_name=query_name, status="failed", error_message=str(exc))

    logger.info("Downloaded '%s' -> %s", query_name, target)
    return DownloadOutcome(query_name=query_name, status="success", file_path=str(target))
