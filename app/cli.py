"""Command-line entry points for the Tender Intelligence Agent.

`collect`/`process`/`screen` are fully implemented (Phases 2-5). `report`
remains a stub pending Phase 6 (app/reports/ is still empty); `run` chains
the working stages via app.pipeline.run_pipeline and will pick up reporting
automatically once Phase 6 exists.
"""
from __future__ import annotations

import logging
import re
import sys
from pathlib import Path

import click

from app.config import get_settings, load_business_capabilities, load_saved_queries

logger = logging.getLogger(__name__)

# Windows' default console codepage (cp1252) can't encode characters this
# CLI legitimately prints (₹ in amounts, arrows in a relayed Playwright
# request log) - confirmed live (2026-09-17): a PARTIAL download outcome's
# error message containing "→"/"←" crashed click.secho with
# UnicodeEncodeError mid-run. Reconfiguring to UTF-8 (replacing anything
# even UTF-8 can't represent, rather than raising) is a no-op on platforms
# whose stdout is already UTF-8.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")


def _mask_mongo_uri(uri: str) -> str:
    """Never print a connection string's password to the terminal/logs."""
    return re.sub(r"://([^:/@]+):([^@]+)@", r"://\1:***@", uri)


@click.group()
def cli() -> None:
    """Tender Intelligence Agent - research and recommendation only.

    This tool never submits bids, sends enquiries, or makes commitments.
    """


@cli.command("init-db")
def init_db_command() -> None:
    """Create the MongoDB indexes this app relies on, if they don't exist."""
    from app.database import ensure_indexes

    ensure_indexes()
    settings = get_settings()
    click.echo(f"Indexes ready on database: {settings.mongodb_db_name} (tenders collection)")


@cli.command("status")
def status_command() -> None:
    """Show current configuration: saved queries, capabilities, settings."""
    settings = get_settings()
    queries = load_saved_queries()
    capabilities = load_business_capabilities()

    click.echo("=== Tender Intelligence Agent - status ===")
    click.echo(f"MongoDB:             {_mask_mongo_uri(settings.mongodb_uri) or '(not set)'}")
    click.echo(f"Download dir:        {settings.download_dir}")
    click.echo(f"Browser headless:    {settings.browser_headless}")
    click.echo(f"AI provider:         {settings.ai_provider} ({settings.openai_model})")
    click.echo(f"Relevance threshold: {settings.ai_relevance_threshold}")
    click.echo("")
    click.echo(f"Saved queries ({len(queries)}):")
    for q in queries:
        flag = "enabled" if q.enabled else "disabled"
        click.echo(f"  - {q.name}  [{flag}]")
    click.echo("")
    click.echo(f"Business capabilities ({len(capabilities)}):")
    for c in capabilities:
        click.echo(f"  - {c}")


@cli.command("collect")
def collect_command() -> None:
    """Log in to TenderDetail and download each saved query's Live export."""
    from app.browser.collector import run_collection

    logging.basicConfig(level=get_settings().log_level, format="%(levelname)s %(message)s")

    click.echo("Starting collector. A browser window will open now.")
    click.echo("If this is the first run (or the session expired), log in")
    click.echo("manually in that window, including any OTP.")
    click.echo("")

    outcomes = run_collection()

    if not outcomes:
        click.secho("No downloads were attempted (login likely failed). See logs above.", fg="red")
        raise SystemExit(1)

    click.echo("")
    click.echo("=== Collection summary ===")
    failures = 0
    for outcome in outcomes:
        if outcome.status == "success":
            click.secho(f"  OK      {outcome.query_name} -> {outcome.file_path}", fg="green")
        elif outcome.status == "skipped":
            click.secho(f"  SKIPPED {outcome.query_name}: {outcome.error_message}", fg="yellow")
        else:
            failures += 1
            click.secho(f"  FAIL    {outcome.query_name}: {outcome.error_message}", fg="red")

    if failures:
        click.secho(f"{failures} of {len(outcomes)} queries failed.", fg="yellow")
        raise SystemExit(1)


def _print_process_result(result) -> None:
    click.echo("=== Processing summary ===")
    if not result.query_results:
        click.secho(f"No downloaded files found in {result.raw_dir}. Run 'collect' first.", fg="red")
        return

    for query_name, qr in result.query_results.items():
        if qr.status == "ok":
            processed_name = Path(qr.processed_path).name if qr.processed_path else "?"
            click.secho(
                f"  OK     {query_name}: {qr.row_count} tenders "
                f"({qr.synthetic_count} without a reference number) -> {processed_name}",
                fg="green",
            )
        else:
            click.secho(f"  FAIL   {query_name}: {qr.error_message}", fg="red")

    if result.ingest_summary is not None:
        summary = result.ingest_summary
        click.echo("")
        click.echo("=== Database merge summary (dedup + history) ===")
        click.echo(f"  New tenders:            {summary.new}")
        click.echo(f"  Deadline updated:       {summary.updated}")
        click.echo(f"  Unchanged (re-seen):    {summary.unchanged}")
        click.echo(f"  Closing within 7 days:  {summary.closing_soon}")
        click.echo(f"  Disappeared (closed):   {summary.disappeared}")
        click.echo(f"  Found in >1 query:      {summary.cross_query_matches}")
    else:
        click.secho("Nothing was successfully normalized; skipping database merge.", fg="red")


@cli.command("process")
@click.option(
    "--input-dir", default=None, help="Override the raw downloads directory."
)
def process_command(input_dir: str | None) -> None:
    """Normalize the latest downloaded export per saved query, dedupe it
    against everything seen before, and merge it into MongoDB.

    Also writes normalized JSON per query to the processed/ directory as a
    plain-text audit trail alongside the database.
    """
    from app.database import ensure_indexes
    from app.pipeline import run_process

    logging.basicConfig(level=get_settings().log_level, format="%(levelname)s %(message)s")
    ensure_indexes()
    settings = get_settings()

    result = run_process(settings, input_dir=input_dir)
    _print_process_result(result)

    if not result.query_results or result.ingest_summary is None or result.any_failed:
        raise SystemExit(1)


def _print_screen_outcome(outcome, threshold: int) -> None:
    if outcome.error:
        click.secho(outcome.error, fg="red")
        return

    summary = outcome.summary
    click.echo("=== AI screening summary ===")
    click.echo(f"  Screened:        {summary.screened}")
    click.echo(f"  Failed:          {summary.failed}")
    click.echo(f"  Above threshold ({threshold}): {summary.above_threshold}")
    for priority, count in sorted(summary.by_priority.items()):
        click.echo(f"    {priority:<14} {count}")


@cli.command("screen")
@click.option("--limit", default=None, type=int, help="Screen at most N tenders this run.")
@click.option(
    "--all",
    "rescreen_all",
    is_flag=True,
    default=False,
    help="Re-screen every tracked tender, not just new/changed ones.",
)
def screen_command(limit: int | None, rescreen_all: bool) -> None:
    """Run AI screening against configured business capabilities."""
    from app.database import ensure_indexes
    from app.pipeline import run_screen

    logging.basicConfig(level=get_settings().log_level, format="%(levelname)s %(message)s")
    ensure_indexes()
    settings = get_settings()

    outcome = run_screen(settings, limit=limit, only_unscreened=not rescreen_all)
    _print_screen_outcome(outcome, settings.ai_relevance_threshold)

    if outcome.error or (outcome.summary and outcome.summary.failed):
        raise SystemExit(1)


@cli.command("download-documents")
@click.option("--limit", default=None, type=int, help="Download documents for at most N tenders this run.")
def download_documents_command(limit: int | None) -> None:
    """Download each eligible tender's attached documents (Tender Document,
    BOQ, Notice) from its detail page on TenderDetail.
    """
    from app.browser.document_collector import run_document_collection

    logging.basicConfig(level=get_settings().log_level, format="%(levelname)s %(message)s")

    click.echo("Starting document collector. A browser window will open now.")
    click.echo("If this is the first run (or the session expired), log in")
    click.echo("manually in that window, including any OTP.")
    click.echo("")

    summary = run_document_collection(limit=limit)

    click.echo("=== Document download summary ===")
    click.echo(f"  Checked: {summary.checked}")
    failures = 0
    for outcome in summary.outcomes:
        label = outcome.tender_ref or outcome.tender_id
        if outcome.status == "success":
            click.secho(f"  OK      {label}: {', '.join(outcome.downloaded)}", fg="green")
        elif outcome.status == "partial":
            click.secho(
                f"  PARTIAL {label}: got {', '.join(outcome.downloaded)}; {outcome.error_message}",
                fg="yellow",
            )
        elif outcome.status == "skipped":
            click.secho(f"  SKIPPED {label}: {outcome.error_message}", fg="yellow")
        else:
            failures += 1
            click.secho(f"  FAIL    {label}: {outcome.error_message}", fg="red")

    if failures:
        click.secho(f"{failures} of {len(summary.outcomes)} tenders failed.", fg="yellow")
        raise SystemExit(1)


@cli.command("fetch-tender-documents")
@click.argument("tender_ids", nargs=-1, required=True)
def fetch_tender_documents_command(tender_ids: tuple[str, ...]) -> None:
    """Download and read the given tenders' documents (View Original
    Notice/Document) now, so their submission checklist can be built from
    the documents themselves. TENDER_IDS are the dashboard's tender ids.
    """
    from bson import ObjectId
    from bson.errors import InvalidId

    from app.browser.document_collector import run_document_collection
    from app.intelligence.document_summarizer import summarize_pending_documents

    logging.basicConfig(level=get_settings().log_level, format="%(levelname)s %(message)s")
    try:
        ids = [ObjectId(t) for t in tender_ids]
    except InvalidId as exc:
        raise click.BadParameter(str(exc), param_hint="TENDER_IDS")

    click.echo("Downloading documents. A browser window will open now.")
    click.echo("")
    downloads = run_document_collection(tender_ids=ids)
    for outcome in downloads.outcomes:
        label = outcome.tender_ref or outcome.tender_id
        colour = {"success": "green", "partial": "yellow"}.get(outcome.status, "red")
        detail = ", ".join(outcome.downloaded) or outcome.error_message or ""
        click.secho(f"  {outcome.status.upper():<8}{label}: {detail}", fg=colour)
    if downloads.checked < len(ids):
        click.secho(f"  {len(ids) - downloads.checked} tender id(s) not found or without a source page.", fg="yellow")

    summary = summarize_pending_documents(tender_ids=ids)
    click.echo("")
    click.echo(f"Read and summarized: {summary.summarized}   Failed: {summary.failed}")
    if summary.failed or not summary.summarized:
        raise SystemExit(1)
    click.echo("Now use 'Regenerate from tender' on the checklist page.")


@cli.command("backfill-key-dates")
@click.option("--limit", default=None, type=int, help="Check at most N tenders this run.")
def backfill_key_dates_command(limit: int | None) -> None:
    """Visit each eligible tender's detail page to scrape its "Key Dates"
    table (Publish/Submission/Opening dates) - for tenders downloaded
    before this field existed. Never re-fetches document files.
    """
    from app.browser.document_collector import run_key_dates_backfill

    logging.basicConfig(level=get_settings().log_level, format="%(levelname)s %(message)s")

    click.echo("Starting Key Dates backfill. A browser window will open now.")
    click.echo("")

    summary = run_key_dates_backfill(limit=limit)

    click.echo("=== Key Dates backfill summary ===")
    click.echo(f"  Checked: {summary.checked}")
    click.echo(f"  Found:   {summary.found}")
    click.echo(f"  Failed:  {summary.failed}")

    if summary.failed:
        raise SystemExit(1)


@cli.command("summarize-documents")
@click.option("--limit", default=None, type=int, help="Summarize at most N tenders this run.")
def summarize_documents_command(limit: int | None) -> None:
    """Run AI summarization over each tender's downloaded documents."""
    from app.intelligence.document_summarizer import summarize_pending_documents

    logging.basicConfig(level=get_settings().log_level, format="%(levelname)s %(message)s")

    summary = summarize_pending_documents(limit=limit)

    click.echo("=== Document summarization summary ===")
    click.echo(f"  Summarized: {summary.summarized}")
    click.echo(f"  Failed:     {summary.failed}")

    if summary.failed:
        raise SystemExit(1)


@cli.command("report")
def report_command() -> None:
    """Generate and email the daily report (Phase 6)."""
    click.echo("report: not yet implemented (Phase 6).")


@cli.command("run")
def run_command() -> None:
    """Run the full pipeline: collect -> process -> screen -> report."""
    from app.database import ensure_indexes
    from app.pipeline import run_pipeline

    logging.basicConfig(level=get_settings().log_level, format="%(levelname)s %(message)s")
    ensure_indexes()
    settings = get_settings()

    summary = run_pipeline(settings, include_report=False)

    click.echo("=== Collection summary ===")
    for outcome in summary.collect_outcomes:
        if outcome.status == "success":
            click.secho(f"  OK      {outcome.query_name} -> {outcome.file_path}", fg="green")
        elif outcome.status == "skipped":
            click.secho(f"  SKIPPED {outcome.query_name}: {outcome.error_message}", fg="yellow")
        else:
            click.secho(f"  FAIL    {outcome.query_name}: {outcome.error_message}", fg="red")

    click.echo("")
    if summary.process_result is not None:
        _print_process_result(summary.process_result)

    click.echo("")
    outcome = summary.document_download_outcome
    if outcome is not None:
        click.echo("=== Document download summary ===")
        if outcome.error:
            click.secho(outcome.error, fg="red")
        elif outcome.summary is not None:
            click.echo(f"  Checked:    {outcome.summary.checked}")
            click.echo(f"  Succeeded:  {outcome.summary.succeeded}")
            click.echo(f"  Failed:     {outcome.summary.failed}")

    click.echo("")
    doc_summary_outcome = summary.document_summarize_outcome
    if doc_summary_outcome is not None:
        click.echo("=== Document summarization summary ===")
        if doc_summary_outcome.error:
            click.secho(doc_summary_outcome.error, fg="red")
        elif doc_summary_outcome.summary is not None:
            click.echo(f"  Summarized: {doc_summary_outcome.summary.summarized}")
            click.echo(f"  Failed:     {doc_summary_outcome.summary.failed}")

    click.echo("")
    if summary.screen_outcome is not None:
        _print_screen_outcome(summary.screen_outcome, settings.ai_relevance_threshold)

    click.echo("")
    click.echo("report: not yet implemented (Phase 6) - skipped.")


if __name__ == "__main__":
    cli()
