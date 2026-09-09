"""Lambda entry point for the scheduled pipeline run.

Not reachable via main.py (Click groups aren't directly callable as a Lambda
handler) - this is a separate, minimal entry point that calls the same
app.pipeline.run_pipeline() the CLI's `run` command uses, so neither
duplicates the orchestration.
"""
from __future__ import annotations

import dataclasses
import logging
from typing import Any

from app.config import get_settings
from app.pipeline import run_pipeline


def handler(event: dict[str, Any] | None, context: Any) -> dict[str, Any]:
    """Run the full collect -> process -> screen -> (report) pipeline once.

    `event` may carry {"include_report": true} - set by whichever
    EventBridge schedule rule should also trigger the daily report email
    (Step 3 wires this up; only one of the two daily runs sets it).

    Exceptions are intentionally NOT caught here - a failed run must
    propagate and mark this invocation as failed, since that's what the
    CloudWatch Alarm (Step 3) watches to alert on a broken run.

    Settings are deliberately read here, inside the handler, and not at
    module import time: `get_settings()` is lru_cached, so reading it at
    import would permanently lock in whatever environment was present
    before this invocation's variables were actually injected.
    """
    settings = get_settings()
    logging.basicConfig(level=settings.log_level, format="%(levelname)s %(message)s")
    include_report = bool((event or {}).get("include_report", False))

    summary = run_pipeline(settings, include_report=include_report)

    return dataclasses.asdict(summary)
