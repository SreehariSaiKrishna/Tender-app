"""Summarizes each eligible tender's downloaded documents (see
app.browser.document_collector) into a structured, per-tender summary:
estimated bid amount, EMD, what has to be submitted, key dates, etc.

Same design goals as app.intelligence.scorer: never invents facts (the
prompt instructs the model to say what's missing rather than guess), never
crashes a batch over one bad tender, and stays provider-agnostic via the
same AIProvider protocol / get_provider() factory scorer.py already
defines - a document summary is just a different prompt/response shape
over the same underlying provider.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError
from pymongo.collection import Collection

from app.config import Settings, get_settings
from app.database import get_collection
from app.intelligence.prompts import build_document_system_prompt, build_document_user_prompt
from app.intelligence.scorer import AIProvider, ScreeningError, get_provider
from app.processing.eligibility import (
    compute_eligibility_match,
    emd_in_range,
    load_match_keywords,
    tender_fee_in_range,
    tender_value_in_range,
)
from app.processing.normalizer import parse_amount_from_text

logger = logging.getLogger(__name__)

# Keeps the prompt within a reasonable token budget per file - see
# app.intelligence.prompts.build_document_user_prompt.
MAX_CHARS_PER_FILE = 20_000


class DocumentSummarizationError(RuntimeError):
    """Raised when the provider call or response validation fails. Callers
    treat this as a per-tender failure, same contract as
    app.intelligence.scorer.ScreeningError."""


class TechnicalCriterionRow(BaseModel):
    """One row of a marks-based technical scoring table (e.g. a tender's
    "Detailed Technical Criteria" section: Ref / Criterion / Expected
    Evidence / Marks columns) - see prompts.py rule 4b. Only some tenders'
    documents contain a table like this at all."""

    ref: str | None = None
    criterion: str
    expected_evidence: str | None = None
    # str|int|float since the provider is asked for a string (e.g. "4") but
    # may still return a bare number for a numeric marks column.
    marks: str | int | float | None = None


class DocumentSummary(BaseModel):
    estimated_bid_amount: str | None = None
    emd_amount: str | None = None
    tender_fee_amount: str | None = None
    documents_to_submit: list[str] = Field(default_factory=list)
    key_dates: list[str] = Field(default_factory=list)
    # DD/MM/YYYY per the prompt (app.intelligence.prompts.DOCUMENT_SYSTEM_PROMPT
    # rule 6) - TenderDetail's own Excel export never includes this column
    # (see app.processing.normalizer's `opening_date`, always None from the
    # listing alone), so this is the only source for it the dashboard has.
    tender_opening_date: str | None = None
    eligibility_requirements: list[str] = Field(default_factory=list)
    # The technical-evaluation subset of eligibility_requirements, only
    # populated when the source text itself distinguishes a "Technical
    # Criteria"/"Technical Eligibility" section - see prompts.py rule 4a.
    eligibility_technical_criteria: list[str] = Field(default_factory=list)
    # A marks-based technical scoring table (Ref/Criterion/Expected
    # Evidence/Marks), only populated for tenders whose documents actually
    # contain one - see prompts.py rule 4b / TechnicalCriterionRow above.
    technical_criteria_table: list[TechnicalCriterionRow] = Field(default_factory=list)
    summary_text: str
    missing_information: list[str] = Field(default_factory=list)


# A page with less than this many extracted characters is treated as
# having no real text layer (scanned/image-only) - confirmed live
# (2026-09-16): two real tender-notice PDFs had exactly 0 extractable
# characters and a page-level image XObject, i.e. a scanned page with no
# text layer at all, not just a sparse one.
MIN_TEXT_CHARS_BEFORE_OCR = 20

# winget's UB-Mannheim.TesseractOCR package (the standard Windows Tesseract
# distribution) doesn't add itself to PATH - confirmed on this machine, a
# fresh shell still couldn't resolve `tesseract` after install - so PATH
# alone can't be trusted the way it can on Linux/Lambda (where the engine
# would be apt-installed and put on PATH by the package itself).
_WINDOWS_DEFAULT_TESSERACT_PATHS = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
)


def _tesseract_cmd() -> str | None:
    """Locate the Tesseract OCR binary, or None if it can't be found -
    OCR is then skipped (not a fatal error) for scanned pages."""
    found = os.environ.get("TESSERACT_CMD") or shutil.which("tesseract")
    if found:
        return found
    return next((p for p in _WINDOWS_DEFAULT_TESSERACT_PATHS if os.path.exists(p)), None)


def _ocr_pdf_page(pdf_path: Path, page_number: int, dpi: int = 200) -> str:
    """OCR one page of a PDF that has no extractable text layer - rasterize
    it with pymupdf (no external binary needed) and read it with
    pytesseract (a thin wrapper around the Tesseract OCR engine, which
    DOES need to be installed separately - see _tesseract_cmd).
    """
    import pymupdf
    import pytesseract

    cmd = _tesseract_cmd()
    if not cmd:
        logger.warning("Tesseract OCR engine not found; skipping OCR for %s", pdf_path)
        return ""
    pytesseract.pytesseract.tesseract_cmd = cmd

    doc = pymupdf.open(str(pdf_path))
    try:
        pixmap = doc[page_number].get_pixmap(dpi=dpi)
        image = pixmap.pil_image()
        return pytesseract.image_to_string(image)
    finally:
        doc.close()


def _extract_pdf_text(path: Path) -> str:
    from pypdf import PdfReader  # imported lazily so tests never need the SDK configured

    reader = PdfReader(str(path))
    texts = [(page.extract_text() or "") for page in reader.pages]

    for i, text in enumerate(texts):
        if len(text.strip()) >= MIN_TEXT_CHARS_BEFORE_OCR:
            continue
        # Likely a scanned/image-only page - fall back to OCR rather than
        # treating the whole file as unreadable.
        try:
            ocr_text = _ocr_pdf_page(path, i)
        except Exception as exc:  # noqa: BLE001 - OCR failing must not lose the rest of the file
            logger.warning("OCR failed for %s page %d: %s", path, i, exc)
            continue
        if ocr_text.strip():
            texts[i] = ocr_text

    return "\n".join(texts)


def _extract_html_text(path: Path) -> str:
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "html.parser")
    return soup.get_text(separator="\n", strip=True)


def _extract_excel_text(path: Path) -> str:
    import pandas as pd

    sheets = pd.read_excel(path, sheet_name=None, header=None)
    parts = []
    for name, df in sheets.items():
        parts.append(f"[Sheet: {name}]")
        parts.append(df.to_string(index=False, header=False))
    return "\n".join(parts)


def _extract_docx_text(path: Path) -> str:
    import docx

    document = docx.Document(str(path))
    return "\n".join(p.text for p in document.paragraphs)


# winget's WinRAR install (the one found on this machine) doesn't add
# UnRAR.exe to PATH either - same situation as Tesseract, see
# _tesseract_cmd. On Linux/Lambda, the `unrar` apt package (added to the
# Dockerfile) does put a `unrar` binary on PATH, so PATH/env override is
# checked first there.
_WINDOWS_DEFAULT_UNRAR_PATHS = (
    r"C:\Program Files\WinRAR\UnRAR.exe",
    r"C:\Program Files (x86)\WinRAR\UnRAR.exe",
)


def _unrar_tool() -> str | None:
    found = os.environ.get("UNRAR_TOOL") or shutil.which("unrar") or shutil.which("UnRAR")
    if found:
        return found
    return next((p for p in _WINDOWS_DEFAULT_UNRAR_PATHS if os.path.exists(p)), None)


def _extract_archive_text(path: Path, opener) -> str:
    """Shared logic for .zip/.rar: extract to a scratch directory, then
    recurse extract_text() over each member (so a PDF/DOCX/etc. inside an
    archive is read the same way it would be if downloaded directly - real
    tender archives seen live (2026-09-16) bundle the RFP, SLA, corrigendum
    etc. as separate files inside one .rar).
    """
    import tempfile

    parts = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        with opener(path) as archive:
            archive.extractall(tmp_dir)
        for member_path in sorted(Path(tmp_dir).rglob("*")):
            if not member_path.is_file():
                continue
            text = extract_text(member_path)
            if text:
                parts.append(f"[Archived file: {member_path.name}]\n{text}")
    return "\n\n".join(parts)


def _extract_zip_text(path: Path) -> str:
    import zipfile

    return _extract_archive_text(path, zipfile.ZipFile)


def _extract_rar_text(path: Path) -> str:
    import rarfile

    tool = _unrar_tool()
    if not tool:
        logger.warning("UnRAR tool not found; skipping archive %s", path)
        return ""
    rarfile.UNRAR_TOOL = tool
    return _extract_archive_text(path, rarfile.RarFile)


# Confirmed live (2026-09-16): real tender attachments include archives
# (.rar, .zip) and Office docs (.doc, .docx) alongside the expected
# pdf/xls/html - decoding those as UTF-8 text (the old fallback for any
# unrecognized extension) produced binary garbage rather than an error, so
# it silently fed the AI provider gibberish instead of being treated as
# "not extractable" like a genuinely unsupported file should be. Only
# extensions actually known to be plain text get the generic fallback now.
KNOWN_PLAIN_TEXT_EXTENSIONS = {".txt", ".csv", ".json", ".xml"}


def extract_text(path: Path) -> str:
    """Extract plain text from one downloaded tender document.

    Never raises - returns "" for an unsupported or corrupt file, since one
    bad file must not lose the rest of a tender's documents (same
    philosophy as app.processing.normalizer never dropping a whole row over
    one bad field). "Unsupported" includes binary formats this module has
    no parser for (e.g. legacy .doc) - those are skipped rather than
    decoded as garbled text.
    """
    suffix = path.suffix.lower()
    try:
        if suffix == ".pdf":
            return _extract_pdf_text(path)
        if suffix in (".html", ".htm"):
            return _extract_html_text(path)
        if suffix in (".xls", ".xlsx"):
            return _extract_excel_text(path)
        if suffix == ".docx":
            return _extract_docx_text(path)
        if suffix == ".zip":
            return _extract_zip_text(path)
        if suffix == ".rar":
            return _extract_rar_text(path)
        if suffix in KNOWN_PLAIN_TEXT_EXTENSIONS:
            return path.read_text(encoding="utf-8", errors="ignore")
        logger.warning("No text extractor for %s; skipping it in the AI summary.", path)
        return ""
    except Exception as exc:  # noqa: BLE001
        logger.error("Could not extract text from %s: %s", path, exc)
        return ""


# The downloaded files are deleted once summarised (see
# _delete_local_documents), so their text is kept on the tender as
# `document_text` - the submission checklist and bid drafting (see
# app.intelligence.bid_drafter) read annexure numbers and prescribed
# formats from it, detail the summary itself doesn't carry. Capped to
# keep the Mongo document well under its 16 MB limit.
MAX_DOCUMENT_TEXT_CHARS = 150_000


def joined_document_text(file_texts: dict[str, str]) -> str:
    parts = [f"--- Document: {name} ---\n{text.strip()}" for name, text in file_texts.items() if text.strip()]
    return "\n\n".join(parts)[:MAX_DOCUMENT_TEXT_CHARS]


def summarize_tender_documents(
    tender: dict[str, Any], file_texts: dict[str, str], provider: AIProvider
) -> DocumentSummary:
    """Summarize one tender's already-extracted document texts. Raises
    DocumentSummarizationError on any failure - caller decides how to
    handle a single tender's failure without aborting others.
    """
    system_prompt = build_document_system_prompt()
    user_prompt = build_document_user_prompt(tender, file_texts, max_chars=MAX_CHARS_PER_FILE)

    try:
        raw = provider.generate(system_prompt, user_prompt)
    except ScreeningError as exc:
        raise DocumentSummarizationError(str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise DocumentSummarizationError(f"Provider call failed: {exc}") from exc

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise DocumentSummarizationError(f"Provider response was not valid JSON: {exc}") from exc

    try:
        return DocumentSummary.model_validate(parsed)
    except ValidationError as exc:
        raise DocumentSummarizationError(
            f"Provider response did not match the expected shape: {exc}"
        ) from exc


def _delete_local_documents(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Delete each document's local file now that its content has already
    been folded into the AI summary - the summary in Mongo is the durable
    record from here on, so the raw file has no further use and shouldn't
    accumulate on disk indefinitely (this is enforced for real in Lambda
    only by /tmp being ephemeral - see app.browser.document_collector's
    storage note - but local/dev runs need it done explicitly).

    Returns a copy of `documents` with each entry's `local_path` cleared to
    None, so the record of what was downloaded (filename, description,
    when) is kept without a dangling path to a deleted file.
    """
    cleared: list[dict[str, Any]] = []
    parent_dirs: set[Path] = set()

    for entry in documents:
        local_path = entry.get("local_path")
        if local_path:
            path = Path(local_path)
            parent_dirs.add(path.parent)
            try:
                path.unlink(missing_ok=True)
            except OSError as exc:
                logger.warning("Could not delete downloaded file %s: %s", path, exc)
        cleared.append({**entry, "local_path": None})

    for directory in parent_dirs:
        try:
            if directory.is_dir() and not any(directory.iterdir()):
                directory.rmdir()
        except OSError:
            pass  # not empty, or already gone - either way, nothing to do

    return cleared


@dataclass
class SummarizeSummary:
    summarized: int = 0
    failed: int = 0


def _needs_summary(doc: dict[str, Any]) -> bool:
    if not doc.get("documents"):
        return False
    generated_at = doc.get("document_summary_generated_at")
    if generated_at is None:
        return True
    downloaded_at = doc.get("documents_downloaded_at")
    return downloaded_at is not None and downloaded_at > generated_at


def summarize_pending_documents(
    settings: Settings | None = None,
    provider: AIProvider | None = None,
    limit: int | None = None,
    collection: Collection | None = None,
) -> SummarizeSummary:
    """Summarize every tender that has downloaded documents but no
    up-to-date AI summary yet (never summarized, or re-downloaded since the
    last summary).
    """
    settings = settings or get_settings()
    provider = provider or get_provider(settings)
    collection = collection if collection is not None else get_collection()

    candidates = [
        doc
        for doc in collection.find({"documents": {"$exists": True, "$ne": []}})
        if _needs_summary(doc)
    ]
    if limit is not None:
        candidates = candidates[:limit]

    # Loaded once per pass (not per tender) - see app.processing.eligibility.
    eligibility_keywords = load_match_keywords()

    summary = SummarizeSummary()
    for tender in candidates:
        file_texts: dict[str, str] = {}
        for entry in tender.get("documents", []):
            local_path = entry.get("local_path")
            if not local_path:
                continue
            path = Path(local_path)
            if not path.exists():
                continue
            text = extract_text(path)
            if text:
                file_texts[entry.get("filename", path.name)] = text

        if not file_texts:
            logger.error(
                "No extractable text for tender %s (%s); skipping summary.",
                tender["_id"],
                tender.get("tender_ref"),
            )
            summary.failed += 1
            continue

        try:
            result = summarize_tender_documents(tender, file_texts, provider)
        except DocumentSummarizationError as exc:
            logger.error(
                "Document summary failed for tender %s (%s): %s",
                tender["_id"],
                tender.get("tender_ref"),
                exc,
            )
            summary.failed += 1
            continue

        cleared_documents = _delete_local_documents(tender.get("documents", []))
        # Re-decide eligibility now that this tender's documents may reveal a
        # value the listing itself never disclosed - merge listing values
        # (authoritative when present) with amounts parsed out of the AI
        # summary, the same way scripts/backfill_eligibility.py does for
        # tenders that already had a summary before these fields existed.
        effective_tender_value = tender.get("tender_value")
        if effective_tender_value is None:
            effective_tender_value = parse_amount_from_text(result.estimated_bid_amount)
        effective_emd = tender.get("earnest_money")
        if effective_emd is None:
            effective_emd = parse_amount_from_text(result.emd_amount)
        effective_fee = tender.get("document_fees")
        if effective_fee is None:
            effective_fee = parse_amount_from_text(result.tender_fee_amount)

        # domain_match started as a listing-only (title/description) keyword
        # check (see app.processing.deduplicator) - a tender's RFP can spell
        # out the actual scope of work in far more detail than its listing
        # does, so OR in a second pass over the AI-extracted eligibility/
        # technical-criteria text before deciding the final match. Never
        # turns a real match back off - only adds signal the listing alone
        # didn't have.
        document_domain_text = " ".join(
            [
                *result.eligibility_requirements,
                *result.eligibility_technical_criteria,
                *(row.criterion for row in result.technical_criteria_table),
                *(
                    row.expected_evidence
                    for row in result.technical_criteria_table
                    if row.expected_evidence
                ),
            ]
        )
        domain_match = tender.get("domain_match", False) or compute_eligibility_match(
            document_domain_text, eligibility_keywords
        )

        eligibility_match = (
            domain_match
            and tender_value_in_range(effective_tender_value)
            and emd_in_range(effective_emd)
            and tender_fee_in_range(effective_fee)
        )

        collection.update_one(
            {"_id": tender["_id"]},
            {
                "$set": {
                    "document_summary": result.model_dump(),
                    "document_text": joined_document_text(file_texts),
                    "document_summary_generated_at": dt.datetime.now(dt.timezone.utc),
                    "documents": cleared_documents,
                    "domain_match": domain_match,
                    "eligibility_match": eligibility_match,
                }
            },
        )
        summary.summarized += 1

    return summary
