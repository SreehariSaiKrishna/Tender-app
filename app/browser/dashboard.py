"""Dashboard navigation: locating a saved query's row and opening its
Live results view.

Selectors below were captured by manually inspecting the live dashboard
with the Playwright Inspector's "Pick locator" tool (2026-09-07):

  - Saved-query row: a <tr> whose text includes the query name shown on
    the dashboard (e.g. "Digital Marketing").
  - Within a row, the clickable "View" elements are <span> tags with the
    exact text "View" - confirmed locator: locator("span").filter(has_text="View")
    In dashboard column order (Query Name | Due Date | Fresh | Live | Close
    | My Calendar), the first "View" span in a row opens the Live results
    for that query; the second opens Close.
  - The results page's export button is a real accessible button:
    get_by_role("button", name="Download Excel") - confirmed via Pick locator.

Do not change these without re-confirming against the live site - they were
intentionally NOT guessed.
"""
from __future__ import annotations

import logging
import re

from playwright.sync_api import Page, TimeoutError as PlaywrightTimeoutError

logger = logging.getLogger(__name__)

DOWNLOAD_BUTTON_NAME = "Download Excel"

# Confirmed by manual inspection (2026-09-07): a query with no matching live
# tenders renders "Showing 1 of 0 Pages from 0 tenders" and no Download Excel
# button at all - this is a normal empty result, not an error.
ZERO_RESULTS_PATTERN = re.compile(r"from 0 tenders", re.IGNORECASE)


class QueryNotFoundError(RuntimeError):
    """Raised when a configured saved query name isn't found on the dashboard."""


def open_live_results(page: Page, query_name: str, timeout_ms: int = 15_000) -> None:
    """From the dashboard, open the Live results view for one saved query.

    Assumes `page` is already on the dashboard (see login.ensure_logged_in).
    Raises QueryNotFoundError if the query name isn't present in the table -
    this is treated as a per-query failure by the caller, not a crash.
    """
    row = page.locator("tr", has_text=query_name)
    try:
        row.first.wait_for(state="visible", timeout=timeout_ms)
    except PlaywrightTimeoutError as exc:
        raise QueryNotFoundError(
            f"Saved query '{query_name}' was not found on the dashboard. "
            "Confirm the exact name matches config/queries.json against the "
            "live dashboard."
        ) from exc

    view_spans = row.first.locator("span").filter(has_text="View")
    count = view_spans.count()
    if count < 1:
        raise QueryNotFoundError(
            f"No 'View' controls found in the dashboard row for '{query_name}'."
        )

    # First "View" = Live column (confirmed by column order on dashboard).
    live_view = view_spans.first
    live_view.click()


def has_zero_results(page: Page, timeout_ms: int = 8_000) -> bool:
    """Return True if the results page just opened shows a genuine "0
    tenders" empty state (no Download Excel button will ever appear).
    """
    try:
        page.get_by_text(ZERO_RESULTS_PATTERN).first.wait_for(
            state="visible", timeout=timeout_ms
        )
        return True
    except PlaywrightTimeoutError:
        return False


def download_excel_export(page, timeout_ms: int = 60_000):
    """Click "Download Excel" on a results page and return the Download object.

    Caller is responsible for saving it (download.save_as(...)) - despite the
    button's label, the file that TenderDetail serves is CSV, not .xlsx
    (confirmed 2026-09-07). We keep whatever extension TenderDetail actually
    sends via download.suggested_filename rather than assuming .xlsx.
    """
    with page.expect_download(timeout=timeout_ms) as download_info:
        page.get_by_role("button", name=DOWNLOAD_BUTTON_NAME).click()
    return download_info.value
