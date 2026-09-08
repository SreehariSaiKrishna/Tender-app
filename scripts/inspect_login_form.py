"""Inspect the real login FORM (not the dashboard) with Playwright Inspector.

Unlike scripts/inspect_login.py, this always opens the actual login page
using a throwaway, temporary browser profile (not your saved session), so
you will always see the username/password form itself, never get skipped
straight to the dashboard by an already-valid session.

It does not read, store, or transmit your password - you only use the
Inspector to POINT AT elements and read out their selectors; you don't
need to actually submit the form at all.

Run it with:
    .\\venv\\Scripts\\python.exe scripts\\inspect_login_form.py

What happens:
1. A Chromium window opens on the login page (tenderdetail.com/Account/Logon).
2. A second small "Playwright Inspector" window also opens, paused.
3. In the Inspector, click the target-icon ("Pick locator"), then click:
     - the USERNAME input field
     - the PASSWORD input field
     - the "Sign In" button
   Each click prints that element's real locator in the Inspector panel
   (bottom box, e.g. `get_by_role("button", name="Sign In")` or similar).
4. Copy each of those three locator strings and send them back to me -
   I'll use them exactly as given instead of guessing.
5. When done, click "Resume" in the Inspector (or just close both windows).
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from playwright.sync_api import sync_playwright  # noqa: E402

LOGIN_FORM_URL = "https://www.tenderdetail.com/Account/Logon"


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="tenderdetail_inspect_") as temp_profile:
        print(f"Using a throwaway profile (not your saved session): {temp_profile}")
        print(f"Opening: {LOGIN_FORM_URL}")
        print()
        print("A Chromium window will open on the LOGIN FORM itself.")
        print("A second small window (Playwright Inspector) will also open -")
        print("click 'Pick locator', then click the username field, the")
        print("password field, and the Sign In button, one at a time.")
        print("Copy each resulting locator string back to me.")

        with sync_playwright() as p:
            context = p.chromium.launch_persistent_context(
                temp_profile,
                headless=False,
                viewport={"width": 1400, "height": 900},
            )
            page = context.new_page()
            page.goto(LOGIN_FORM_URL)

            page.pause()  # opens Playwright Inspector; resumes when you click "Resume"

            print(f"Final URL after you resumed: {page.url}")
            context.close()


if __name__ == "__main__":
    main()
