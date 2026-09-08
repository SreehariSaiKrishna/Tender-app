"""Reads a downloaded tender export (CSV or Excel) into raw row dicts.

TenderDetail's "Download Excel" button actually serves a .csv file, not a
real .xlsx (confirmed 2026-09-07 by inspecting an actual download). This
reader accepts both, in case that ever changes, and always preserves the
original cell text exactly (no automatic type coercion) so the normalizer
is the only place parsing decisions are made.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


class ExcelReadError(RuntimeError):
    """Raised when a raw export file cannot be read at all."""


def read_raw_rows(file_path: str | Path) -> list[dict[str, Any]]:
    """Read a downloaded export file into a list of raw row dicts.

    Column names and cell values are kept exactly as exported (as strings) -
    no NaN conversion, no numeric coercion - so "Ref. Document", empty
    cells, and formula strings all reach the normalizer unchanged.
    """
    path = Path(file_path)
    if not path.exists():
        raise ExcelReadError(f"File not found: {path}")

    suffix = path.suffix.lower()
    try:
        if suffix in (".xlsx", ".xls"):
            df = pd.read_excel(path, dtype=str, keep_default_na=False)
        else:
            # Default to CSV parsing - this is what TenderDetail's "Download
            # Excel" button actually produces, regardless of its label.
            df = pd.read_csv(
                path, dtype=str, keep_default_na=False, encoding="utf-8-sig"
            )
    except Exception as exc:  # noqa: BLE001
        raise ExcelReadError(f"Could not read {path}: {exc}") from exc

    # Drop fully-blank rows (e.g. a trailing empty line) - never drop a row
    # that has any real data in it.
    df = df[~(df == "").all(axis=1)]

    rows = df.to_dict(orient="records")
    logger.info("Read %d rows from %s", len(rows), path.name)
    return rows
