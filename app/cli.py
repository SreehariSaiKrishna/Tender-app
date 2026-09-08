"""Command-line entry points for the Tender Intelligence Agent.

Phase 1 provides `init-db` and `status` (working) plus placeholders for
`collect`, `process`, `screen`, `report`, and `run` that will be implemented
in later phases. Placeholders exist so the CLI shape is stable, but they do
nothing beyond reporting that the feature is not yet built.
"""
from __future__ import annotations

import logging
import re

import click

from app.config import get_settings, load_business_capabilities, load_saved_queries
from app.database import init_db


@click.group()
def cli() -> None:
    """Tender Intelligence Agent - research and recommendation only.

    This tool never submits bids, sends enquiries, or makes commitments.
    """


@cli.command("init-db")
def init_db_command() -> None:
    """Create the SQLite database and tables if they don't exist."""
    init_db()
    settings = get_settings()
    click.echo(f"Database ready at: {settings.database_url}")


@cli.command("status")
def status_command() -> None:
    """Show current configuration: saved queries, capabilities, settings."""
    settings = get_settings()
    queries = load_saved_queries()
    capabilities = load_business_capabilities()

    click.echo("=== Tender Intelligence Agent - status ===")
    click.echo(f"Database URL:        {settings.database_url}")
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
    init_db()

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


@cli.command("process")
@click.option(
    "--input-dir", default=None, help="Override the raw downloads directory."
)
def process_command(input_dir: str | None) -> None:
    """Read the latest downloaded export per saved query, normalize it, and
    merge it into the database (dedup + history tracking, Phase 3/4).

    Also writes normalized JSON per query to the processed/ directory as a
    plain-text audit trail alongside the database.
    """
    import json
    from pathlib import Path

    from app.processing.deduplicator import ingest_batch
    from app.processing.excel_reader import ExcelReadError, read_raw_rows
    from app.processing.normalizer import NormalizedTender, normalize_rows

    logging.basicConfig(level=get_settings().log_level, format="%(levelname)s %(message)s")
    init_db()
    settings = get_settings()
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

    if not files_by_query:
        click.secho(
            f"No downloaded files found in {raw_dir}. Run 'collect' first.", fg="red"
        )
        raise SystemExit(1)

    click.echo("=== Processing summary ===")
    query_results: dict[str, list[NormalizedTender]] = {}
    any_failed = False
    for query_name, path in files_by_query.items():
        try:
            raw_rows = read_raw_rows(path)
        except ExcelReadError as exc:
            any_failed = True
            click.secho(f"  FAIL   {query_name}: {exc}", fg="red")
            continue

        normalized = normalize_rows(raw_rows, query_name)
        query_results[query_name] = normalized
        synthetic = sum(1 for n in normalized if n.ref_is_synthetic)

        out_path = processed_dir / f"{path.stem}.json"
        out_path.write_text(
            json.dumps([n.model_dump(mode="json") for n in normalized], indent=2),
            encoding="utf-8",
        )
        click.secho(
            f"  OK     {query_name}: {len(normalized)} tenders "
            f"({synthetic} without a reference number) -> {out_path.name}",
            fg="green",
        )

    if not query_results:
        click.secho("Nothing was successfully normalized; skipping database merge.", fg="red")
        raise SystemExit(1)

    from app.database import session_scope

    with session_scope() as session:
        summary = ingest_batch(session, query_results)

    click.echo("")
    click.echo("=== Database merge summary (dedup + history) ===")
    click.echo(f"  New tenders:            {summary.new}")
    click.echo(f"  Deadline updated:       {summary.updated}")
    click.echo(f"  Unchanged (re-seen):    {summary.unchanged}")
    click.echo(f"  Closing within 7 days:  {summary.closing_soon}")
    click.echo(f"  Disappeared (closed):   {summary.disappeared}")
    click.echo(f"  Found in >1 query:      {summary.cross_query_matches}")

    if any_failed:
        raise SystemExit(1)


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
    from app.intelligence.scorer import ScreeningError, get_provider, screen_tenders

    logging.basicConfig(level=get_settings().log_level, format="%(levelname)s %(message)s")
    init_db()
    settings = get_settings()

    try:
        provider = get_provider(settings)
    except ScreeningError as exc:
        click.secho(str(exc), fg="red")
        raise SystemExit(1)

    from app.database import session_scope

    with session_scope() as session:
        summary = screen_tenders(
            session, provider=provider, limit=limit, only_unscreened=not rescreen_all
        )

    click.echo("=== AI screening summary ===")
    click.echo(f"  Screened:        {summary.screened}")
    click.echo(f"  Failed:          {summary.failed}")
    click.echo(f"  Above threshold ({settings.ai_relevance_threshold}): {summary.above_threshold}")
    for priority, count in sorted(summary.by_priority.items()):
        click.echo(f"    {priority:<14} {count}")

    if summary.failed:
        raise SystemExit(1)


@cli.command("report")
def report_command() -> None:
    """Generate and email the daily report (Phase 6)."""
    click.echo("report: not yet implemented (Phase 6).")


@cli.command("run")
def run_command() -> None:
    """Run the full pipeline: collect -> process -> screen -> report."""
    click.echo("run: not yet implemented. Will chain collect/process/screen/report.")


if __name__ == "__main__":
    cli()
