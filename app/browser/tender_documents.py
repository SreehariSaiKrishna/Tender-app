"""Tender detail page: locating and fetching the attached files listed
under "View Original Notice/Document" (Tender Document / BOQ / Notice PDF).

Confirmed live against a real tender detail page (2026-09-16):
  - A tender's stored `source_url` (captured from the "Tender Brief"
    =HYPERLINK(...) formula - see app.processing.normalizer) lands
    directly on the page with this document table - no extra
    "View Notice" click needed for a tender already tracked in Mongo.
  - Each row's "Download" link is `target="_blank"` pointing at a real
    file URL (`/tenders/DownloadDocument/...`). Clicking it and waiting
    for page.expect_download() only works for file types Chromium has no
    built-in viewer for (.xls) - for types it CAN render (.html, .pdf) it
    opens a new tab to display the content instead, and no download event
    ever fires. Fetching the href directly via the page's own authenticated
    request context (page.context.request.get) sidesteps this distinction
    entirely and returned correct bytes for all three types tested
    (.xls, .html, .pdf) - so that's the only strategy used below, for
    every file type.
"""
from __future__ import annotations

import re
from urllib.parse import urljoin

from playwright.sync_api import Page, TimeoutError as PlaywrightTimeoutError

DOCUMENTS_HEADING = "View Original Notice/Document"
KEY_DATES_HEADING = "Key Dates"


class NoDocumentsFoundError(RuntimeError):
    """Raised when the document table's heading is present but no rows
    were found under it - a page structure that hasn't been seen before."""


class DocumentFetchError(RuntimeError):
    """Raised when a document's URL could not be fetched (non-2xx
    response, or the row had no href at all)."""


def click_view_notice(page: Page, tender_title: str, timeout_ms: int = 15_000) -> None:
    """Fallback path only, for a search-results row you don't already have
    a stored `source_url` for. Prefer open_tender_detail(page, source_url)
    when the tender is already tracked in Mongo.
    """
    row = page.locator("tr, div.tender-card", has_text=tender_title)
    row.first.wait_for(state="visible", timeout=timeout_ms)
    row.first.get_by_role("button", name=re.compile("View Notice", re.I)).click()


def open_tender_detail(page: Page, source_url: str, timeout_ms: int = 20_000) -> None:
    """Navigate straight to a tender's detail page via its stored
    `source_url` and wait for the document table's heading to appear.
    """
    page.goto(source_url, timeout=timeout_ms)
    page.get_by_text(DOCUMENTS_HEADING).wait_for(state="visible", timeout=timeout_ms)


def list_document_rows(page: Page) -> list[dict]:
    """Return one entry per row in the "View Original Notice/Document"
    table: {filename, description, url}, where `url` is the absolute file
    URL resolved from that row's Download link `href` - scoped to the
    table specifically (via the nearest following <table> after the
    heading), not just any "Download"-labelled link anywhere on the page.
    """
    heading = page.get_by_text(DOCUMENTS_HEADING)
    table = heading.locator("xpath=following::table[1]")
    rows = table.locator("tbody tr")

    out: list[dict] = []
    for i in range(rows.count()):
        row = rows.nth(i)
        cells = row.locator("td")
        href = row.get_by_role("link", name=re.compile("Download", re.I)).get_attribute("href")
        out.append(
            {
                "filename": cells.nth(1).inner_text().strip(),
                "description": cells.nth(2).inner_text().strip(),
                "url": urljoin(page.url, href) if href else None,
            }
        )

    if not out:
        raise NoDocumentsFoundError(
            "'View Original Notice/Document' heading found but its table has no rows."
        )
    return out


def extract_key_dates(page: Page, timeout_ms: int = 5_000) -> dict[str, str]:
    """Return the tender detail page's "Key Dates" table as {label: raw
    value} - confirmed live (2026-09-16): a "Key Dates" heading followed by
    the nearest <table>, one row per label/value pair, e.g.
    {"Publish Date": "03-09-2026", "Last Date of Bid Submission": "14-09-2026",
    "Tender Opening Date": "15-09-2026"} (dates in DD-MM-YYYY). This is
    authoritative site data, not an AI guess from the attached documents -
    prefer it over app.intelligence.document_summarizer's
    DocumentSummary.tender_opening_date whenever both are available.

    Returns {} (not an error) if the section isn't present within
    `timeout_ms` - not every tender detail page necessarily has one, and a
    missing "Key Dates" section must never fail the rest of that tender's
    processing.
    """
    heading = page.get_by_text(KEY_DATES_HEADING, exact=True)
    try:
        heading.first.wait_for(state="visible", timeout=timeout_ms)
    except PlaywrightTimeoutError:
        return {}

    table = heading.first.locator("xpath=following::table[1]")
    rows = table.locator("tr")

    result: dict[str, str] = {}
    for i in range(rows.count()):
        cells = rows.nth(i).locator("td, th")
        if cells.count() < 2:
            continue
        label = cells.nth(0).inner_text().strip()
        value = cells.nth(1).inner_text().strip()
        if label:
            result[label] = value
    return result


def fetch_document_bytes(page: Page, url: str, timeout_ms: int = 60_000) -> bytes:
    """Fetch one document's raw bytes via the page's own authenticated
    request context (shares the browser context's session cookies) -
    deliberately not a click + page.expect_download(), which misses any
    file type Chromium renders instead of downloading (see this module's
    docstring).
    """
    response = page.context.request.get(url, timeout=timeout_ms)
    if not response.ok:
        raise DocumentFetchError(f"GET {url} returned HTTP {response.status}")
    return response.body()
