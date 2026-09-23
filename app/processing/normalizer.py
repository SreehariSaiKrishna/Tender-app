"""Normalizes raw exported tender rows into a canonical shape.

Handles, based on manual inspection of a real TenderDetail export
(2026-09-07):
  - Column name variation (via alias matching, case/space-insensitive).
  - Indian DD/MM/YYYY dates (DueDate, PubDate).
  - The literal =HYPERLINK("url","title") formula TenderDetail embeds in
    the "Tender Brief" column instead of separate title/URL columns.
  - The placeholder text "Ref. Document" TenderDetail uses in DocumentFees/
    EMD/Tender Amount to mean "not disclosed in the listing" - treated as
    unknown (None), never invented as zero.
  - Always preserving the original row verbatim in raw_data - a row is
    never dropped just because some fields fail to parse.
"""
from __future__ import annotations

import datetime as dt
import logging
import re
from typing import Any

from pydantic import BaseModel, ConfigDict

logger = logging.getLogger(__name__)

# Canonical field -> possible source column names. TenderDetail's own
# column names (confirmed) are listed first; the rest are defensive
# aliases in case the export format changes or another query type differs.
COLUMN_ALIASES: dict[str, list[str]] = {
    "tender_ref": ["TDR", "TDR No", "TDR No.", "Tender Detail Ref"],
    "organisation": ["Tendering Authority", "Organisation", "Department", "Authority"],
    "brief": ["Tender Brief", "Title", "Tender Title"],
    "document_fees": ["DocumentFees", "Document Fees"],
    "earnest_money": ["EMD", "Earnest Money", "EMD Amount"],
    "tender_value": ["Tender Amount", "Tender Value", "Estimated Value"],
    "location": ["Location", "City"],
    "state": ["State"],
    "closing_date": ["DueDate", "Due Date", "Closing Date"],
    "opening_date": ["Opening Date", "Tender Opening Date"],
    "tender_no": ["TenderNo", "Tender No", "Tender Number"],
    "address": ["Address"],
    "contact_email": ["ContactEmail", "Contact Email"],
    "tender_id": ["TenderId", "Tender Id"],
    "published_date": ["PubDate", "Published Date", "Publish Date"],
    "exemption": ["Exemption"],
    "quantity": ["Quantity"],
    "info_source": ["Information source", "Information Source", "Source"],
}

# Text values TenderDetail uses to mean "not available" rather than zero/blank.
PLACEHOLDER_VALUES = {"ref. document", "n/a", "na", "-", ""}

HYPERLINK_RE = re.compile(
    r'^=HYPERLINK\(\s*"(?P<url>[^"]*)"\s*,\s*"(?P<title>.*)"\s*\)$',
    re.DOTALL,
)

DATE_FORMATS = ("%d/%m/%Y", "%d-%m-%Y", "%d.%m.%Y", "%d %b %Y", "%d %B %Y", "%Y-%m-%d")


def _normalize_key(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def build_column_map(columns: list[str]) -> dict[str, str]:
    """Map canonical field name -> actual column name present in this file."""
    normalized_lookup = {_normalize_key(c): c for c in columns}
    result: dict[str, str] = {}
    for canonical, aliases in COLUMN_ALIASES.items():
        for alias in aliases:
            key = _normalize_key(alias)
            if key in normalized_lookup:
                result[canonical] = normalized_lookup[key]
                break
    return result


def _clean_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in PLACEHOLDER_VALUES:
        return None
    return text


def parse_indian_date(value: Any) -> dt.date | None:
    """Parse a date in DD/MM/YYYY (TenderDetail's format) or a few other
    common variants. Returns None (never raises) if unparseable, and logs
    a warning so parsing failures are visible without dropping the row.
    """
    text = _clean_text(value)
    if text is None:
        return None
    for fmt in DATE_FORMATS:
        try:
            return dt.datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    logger.warning("Could not parse date value: %r", value)
    return None


def parse_amount(value: Any) -> float | None:
    """Parse a currency/numeric field such as EMD or Tender Amount.

    TenderDetail uses the literal text "Ref. Document" to mean the amount
    isn't disclosed in the listing - that is treated as unknown (None),
    never as zero, and never invented.
    """
    text = _clean_text(value)
    if text is None:
        return None
    cleaned = re.sub(r"[,₹\s]", "", text)
    try:
        return float(cleaned)
    except ValueError:
        logger.warning("Could not parse numeric value: %r", value)
        return None


AMOUNT_IN_TEXT_RE = re.compile(r"[\d][\d,]*(?:\.\d+)?")


def parse_amount_from_text(value: Any) -> float | None:
    """Extract the first numeric amount embedded in free text, e.g. an AI
    document summary field like "INR 45,00,000 (estimated)" -> 4500000.0.

    Unlike parse_amount, `value` isn't expected to be pure digits - only
    the first number found is used (commas stripped). Returns None if no
    digit appears at all - never raises.
    """
    if value is None:
        return None
    match = AMOUNT_IN_TEXT_RE.search(str(value))
    if not match:
        return None
    try:
        return float(match.group(0).replace(",", ""))
    except ValueError:
        return None


def parse_tender_brief(value: Any) -> tuple[str, str | None]:
    """Extract (title, source_url) from TenderDetail's "Tender Brief" cell,
    which is exported as a literal =HYPERLINK("url","title") formula string
    rather than plain text. Falls back to the raw text as the title if the
    formula can't be parsed - the field is never invented, and the row is
    never dropped over this alone.
    """
    text = "" if value is None else str(value).strip()
    match = HYPERLINK_RE.match(text)
    if match:
        return match.group("title").strip(), match.group("url").strip()
    if text:
        logger.warning(
            "Tender Brief did not match the expected HYPERLINK format: %r", text[:120]
        )
    return text, None


def make_dedup_key(
    tender_ref: str | None,
    title: str,
    organisation: str | None,
    closing_date: dt.date | None,
) -> tuple[str, bool]:
    """Return (dedup_key, is_synthetic).

    Prefers the TDR reference number (always present and unique in every
    TenderDetail export observed). Falls back to a normalized
    title+organisation+closing_date combination only when no reference
    number is available, per the dedup design in the project spec.
    """
    if tender_ref:
        return f"ref:{tender_ref.strip().lower()}", False

    parts = [
        re.sub(r"\s+", " ", title.strip().lower()) if title else "",
        re.sub(r"\s+", " ", (organisation or "").strip().lower()),
        closing_date.isoformat() if closing_date else "",
    ]
    return "syn:" + "|".join(parts), True


class NormalizedTender(BaseModel):
    model_config = ConfigDict(frozen=True)

    tender_ref: str | None
    ref_is_synthetic: bool
    dedup_key: str
    query_name: str
    title: str
    organisation: str | None = None
    location: str | None = None
    state: str | None = None
    closing_date: dt.date | None = None
    published_date: dt.date | None = None
    # No confirmed column for this in the real TenderDetail export (see
    # COLUMN_ALIASES) - stays None unless a future export type provides it.
    opening_date: dt.date | None = None
    tender_value: float | None = None
    earnest_money: float | None = None
    document_fees: float | None = None
    document_url: str | None = None
    source_url: str | None = None
    description: str | None = None
    raw_data: dict[str, Any]


def normalize_row(
    row: dict[str, Any], query_name: str, column_map: dict[str, str] | None = None
) -> NormalizedTender:
    """Normalize a single raw exported row. Never raises for a malformed
    field - only that field becomes None, and the original row is always
    preserved verbatim in raw_data.
    """
    column_map = column_map if column_map is not None else build_column_map(list(row.keys()))

    def get(field: str) -> Any:
        col = column_map.get(field)
        return row.get(col) if col else None

    tender_ref = _clean_text(get("tender_ref"))
    title, source_url = parse_tender_brief(get("brief"))
    if not title:
        title = f"(untitled tender {tender_ref or 'unknown'})"

    organisation = _clean_text(get("organisation"))
    closing_date = parse_indian_date(get("closing_date"))

    dedup_key, is_synthetic = make_dedup_key(tender_ref, title, organisation, closing_date)

    return NormalizedTender(
        tender_ref=tender_ref,
        ref_is_synthetic=is_synthetic,
        dedup_key=dedup_key,
        query_name=query_name,
        title=title,
        organisation=organisation,
        location=_clean_text(get("location")),
        state=_clean_text(get("state")),
        closing_date=closing_date,
        published_date=parse_indian_date(get("published_date")),
        opening_date=parse_indian_date(get("opening_date")),
        tender_value=parse_amount(get("tender_value")),
        earnest_money=parse_amount(get("earnest_money")),
        document_fees=parse_amount(get("document_fees")),
        # Not present as a separate column in this export type - the
        # attachment links seen on the website aren't included in the
        # Download Excel output. Left as None rather than guessed.
        document_url=None,
        source_url=source_url,
        # No separate description column exists in this export; the title
        # is the fullest text TenderDetail provides here.
        description=None,
        raw_data=row,
    )


def normalize_rows(rows: list[dict[str, Any]], query_name: str) -> list[NormalizedTender]:
    """Normalize every row from one query's export.

    Rows are never dropped: if a row fails to normalize in some unexpected
    way, it's still emitted with a placeholder title and the raw row
    preserved, and the error is logged.
    """
    if not rows:
        return []

    column_map = build_column_map(list(rows[0].keys()))
    normalized: list[NormalizedTender] = []
    for i, row in enumerate(rows):
        try:
            normalized.append(normalize_row(row, query_name, column_map))
        except Exception as exc:  # noqa: BLE001 - a row must never be lost
            logger.error("Row %d failed to normalize (%s); keeping raw only.", i, exc)
            normalized.append(
                NormalizedTender(
                    tender_ref=None,
                    ref_is_synthetic=True,
                    dedup_key=f"syn:error:{query_name}:{i}",
                    query_name=query_name,
                    title=f"(failed to parse row {i})",
                    raw_data=row,
                )
            )
    return normalized
