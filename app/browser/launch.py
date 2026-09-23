"""Shared Chromium launch flags for the persistent-context browser
automation (app.browser.collector / app.browser.document_collector).

`--no-sandbox` / `--disable-setuid-sandbox` / `--single-process` /
`--no-zygote` exist only to work around AWS Lambda's restricted syscall
sandbox (the container runs as root with no fork() available for
Chromium's normal zygote/multi-process model).

Confirmed live (2026-09-16): applying `--single-process`/`--no-zygote`
outside Lambda crashes Chromium's renderer - and with it the whole
browser/context - on a real, JS-heavy site (tenderdetail.com). A trivial
static page (example.com) loaded fine with the same flags; navigating the
identical persistent profile to tenderdetail.com closed the context before
the page ever loaded. These flags must only be applied when actually
running inside Lambda, never for local/dev runs.
"""
from __future__ import annotations

import os

LAMBDA_SANDBOX_WORKAROUND_ARGS = [
    "--no-sandbox",
    "--disable-setuid-sandbox",
    "--single-process",
    "--no-zygote",
    "--disable-gpu",
    "--disable-dev-shm-usage",
]


def chromium_launch_args() -> list[str]:
    """Lambda's sandbox-workaround args when actually running inside
    Lambda (detected the standard way, via AWS_LAMBDA_FUNCTION_NAME - set
    by the Lambda runtime itself, never by this app), otherwise none.
    """
    if os.environ.get("AWS_LAMBDA_FUNCTION_NAME"):
        return list(LAMBDA_SANDBOX_WORKAROUND_ARGS)
    return []
