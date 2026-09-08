"""Manual inspection helper for Phase 2.

Opens a REAL, VISIBLE Chromium window pointed at the TenderDetail login
page, using a persistent browser profile so the session is reusable after
you log in. It does not read, store, or transmit your password or OTP -
you type them directly into the browser window yourself.

Run it with:
    .\\venv\\Scripts\\python.exe scripts\\inspect_login.py

What happens:
1. A Chromium window opens on your screen (this IS the output - no need
   to look at the terminal for it).
2. Log in manually in that window, including any OTP.
3. Navigate to your saved queries dashboard once logged in.
4. Come back to the terminal and press Enter in the Playwright Inspector
   / terminal prompt when asked, OR use the Inspector's element picker
   (see below) to grab real selectors.
5. The authenticated session is saved to data/browser_profile/ so you
   won't need to log in again next time (until it expires).

This script calls page.pause(), which opens the Playwright Inspector - a
separate small window with "Record" / element-picker tools. Click the
target-icon ("Pick locator") in the Inspector, then click any element in
the browser window (a saved query link, a Download/Export button, a
tender row) to see its real selector printed in the Inspector. Copy those
real selectors back to me - do not guess them.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from playwright.sync_api import sync_playwright  # noqa: E402

from app.config import get_settings  # noqa: E402


def main() -> None:
    settings = get_settings()
    profile_dir = settings.resolved_path(settings.tenderdetail_profile_dir)
    profile_dir.mkdir(parents=True, exist_ok=True)

    print(f"Using persistent browser profile: {profile_dir}")
    print(f"Opening: {settings.tenderdetail_login_url}")
    print()
    print("A Chromium window will open now. Log in manually there.")
    print("A second small window (Playwright Inspector) will also open -")
    print("use its 'Pick locator' tool to inspect real elements, then")
    print("click 'Resume' (or close it) when you're done exploring.")

    with sync_playwright() as p:
        context = p.chromium.launch_persistent_context(
            str(profile_dir),
            headless=False,
            viewport={"width": 1400, "height": 900},
        )
        page = context.new_page()
        page.goto(settings.tenderdetail_login_url)

        page.pause()  # opens Playwright Inspector; script resumes when you click "Resume"

        print(f"Final URL after you resumed: {page.url}")
        print(f"Page title: {page.title()}")
        context.close()


if __name__ == "__main__":
    main()
