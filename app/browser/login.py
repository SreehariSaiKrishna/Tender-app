"""Login handling for TenderDetail.

Two modes, chosen per the presence of TENDERDETAIL_PASSWORD in .env:

  - Manual (default, safer): the browser opens the login page and this
    module only WAITS for a human to complete the entire login (username,
    password, any OTP) directly in the visible window. Nothing is read,
    filled, or stored.
  - Auto-fill (opt-in): if a password is configured, the username and
    password fields are filled and submitted automatically. This still
    falls through to the same manual wait afterward, so any additional
    verification step (OTP, a CAPTCHA, a wrong-password retry) is always
    handled by you, in the browser - this module never attempts to solve
    or bypass one.

Confirmed against the live login form (tenderdetail.com/Account/Logon,
2026-09-08): a "Username" / "Mobile OTP" method toggle, a labeled
USERNAME field, a real <input type="password"> PASSWORD field, and a
"Sign In" button. Selectors below rely on that visible text and on the
standard `type="password"` semantics rather than guessed CSS classes.

The password value itself is never logged or included in any exception
message this module raises or logs.
"""
from __future__ import annotations

import logging
import re

from playwright.sync_api import Page, TimeoutError as PlaywrightTimeoutError

logger = logging.getLogger(__name__)

# Confirmed by manual inspection of the live dashboard (2026-09-07):
# the saved-queries table has a "Query Name" column header.
DASHBOARD_MARKER_TEXT = "Query Name"

# How long to wait for a human to complete manual login (including OTP),
# on a run where the persistent session has expired or doesn't exist yet.
MANUAL_LOGIN_TIMEOUT_MS = 10 * 60 * 1000  # 10 minutes


def is_dashboard_loaded(page: Page, timeout_ms: int = 5_000) -> bool:
    """Return True if the saved-queries dashboard is currently visible."""
    try:
        page.get_by_text(DASHBOARD_MARKER_TEXT, exact=True).first.wait_for(
            state="visible", timeout=timeout_ms
        )
        return True
    except PlaywrightTimeoutError:
        return False


def fill_login_form(page: Page, username: str, password: str) -> None:
    """Fill and submit the Username/Password sign-in form.

    Locators below were confirmed with the Playwright Inspector's "Pick
    locator" tool directly against the live login form
    (tenderdetail.com/Account/Logon, 2026-09-08):
      tab:      get_by_role("link", name=" Username")
      username: get_by_role("textbox", name="Enter your username")
      password: get_by_role("textbox", name="Enter your password")
      submit:   get_by_role("button", name=" Sign In")

    Raises on any Playwright error (element not found, timeout, etc.) -
    caller decides whether to fall back to manual login. Never logs the
    password and never includes it in a raised exception.
    """
    # The site offers two methods (Username / Mobile OTP); this tab is the
    # default per inspection, so this click is best-effort only.
    try:
        page.get_by_role("link", name=re.compile("username", re.IGNORECASE)).first.click(
            timeout=3_000
        )
    except Exception:  # noqa: BLE001
        pass

    username_field = page.get_by_role("textbox", name=re.compile("enter your username", re.IGNORECASE))
    password_field = page.get_by_role("textbox", name=re.compile("enter your password", re.IGNORECASE))

    username_field.wait_for(state="visible", timeout=10_000)
    username_field.fill(username)
    password_field.fill(password)

    try:
        page.get_by_role("button", name=re.compile("sign in", re.IGNORECASE)).first.click(
            timeout=5_000
        )
    except Exception:  # noqa: BLE001
        # Universal fallback: Enter in the password field triggers standard
        # HTML form submission regardless of the button's exact markup.
        password_field.press("Enter")


def ensure_logged_in(
    page: Page, dashboard_url: str, username: str = "", password: str = ""
) -> bool:
    """Navigate to the dashboard and make sure we're authenticated.

    If the persistent browser profile already has a valid session, this
    returns almost immediately. Otherwise, if `password` is provided, it
    attempts to auto-fill and submit the login form once; either way, it
    then pauses (a long wait, not a blocking input()) so a human can
    complete anything further - a fresh manual login, an OTP step, or
    fixing a wrong password - directly in the visible window.

    Returns True if the dashboard loaded successfully, False if login
    could not be confirmed within the timeout (caller must stop safely -
    never retry login automatically, never guess credentials).
    """
    page.goto(dashboard_url)

    if is_dashboard_loaded(page, timeout_ms=5_000):
        logger.info("Existing session is valid - already on dashboard.")
        return True

    if username and password:
        logger.info("Attempting auto-fill login for %s", username)
        try:
            fill_login_form(page, username, password)
            page.wait_for_timeout(1_500)
        except Exception:  # noqa: BLE001 - never log exception text here (may echo form state)
            logger.error("Auto-fill login did not complete - falling back to manual login.")
        else:
            if is_dashboard_loaded(page, timeout_ms=8_000):
                logger.info("Auto-fill login succeeded.")
                return True
            logger.info(
                "Login form submitted, but the dashboard isn't showing yet - "
                "complete any additional step (OTP, verification) manually."
            )

    print()
    print("=" * 70)
    print("MANUAL LOGIN REQUIRED")
    print("A browser window is open at the TenderDetail login page.")
    print("Please log in yourself now, including any OTP if prompted.")
    print("This tool will continue automatically once the dashboard loads.")
    print(f"Waiting up to {MANUAL_LOGIN_TIMEOUT_MS // 60_000} minutes...")
    print("=" * 70)
    print()

    try:
        page.get_by_text(DASHBOARD_MARKER_TEXT, exact=True).first.wait_for(
            state="visible", timeout=MANUAL_LOGIN_TIMEOUT_MS
        )
    except PlaywrightTimeoutError:
        logger.error("Login was not completed within the timeout. Stopping safely.")
        return False

    logger.info("Login detected - dashboard is loaded.")
    return True
