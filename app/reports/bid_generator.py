"""Builds a tender's Master Bid Submission Checklist and, from it, a single
downloadable "bid pack" PDF.

The checklist (see build_checklist) lists every document the tender asks
for - S.No / Document / What to Upload / Where / Status, like the team's
own submission checklists - with, per row, whether it's an existing
record to attach from the documents library (app.api.main's /documents -
GridFS, get_company_documents_bucket) or a document the bidder must write,
and whether it goes on the letterhead and carries the signature and stamp.
The rows normally come from app.intelligence.bid_drafter's AI reading of
the tender; when that's unavailable they're derived from the tender's
AI-extracted document_summary and the library itself.

The bid pack (see generate_bid_package) follows the saved, possibly
hand-edited, checklist: the checklist page first, then every row's
document in S.No order - library files merged in (placed on the letterhead
and signed/stamped when the row says so), drafted documents rendered on
their own pages, and a clearly marked placeholder page for anything still
missing - then internal review notes on plain pages, to be removed before
submission. Reference material like the company brochure is never
enclosed - it only feeds the drafter background text (see
company_background_text).

Same non-negotiable rule as app.intelligence.document_summarizer and
app.processing.eligibility: never invent evidence. Every fact about the
tender comes from the `tenders` collection and its AI-extracted
`document_summary`/`document_text`; every fact about the company comes from
config/company_profile.json (see app.config.load_company_profile) or from a
document the user has actually uploaded. Anything missing is listed as
missing, never guessed.

This module itself does no AI calls and has no dependency on them - it
only uses whatever checklist plan/drafted documents its caller
(app.api.main) hands it, falling back to deterministic rows and templated
pages when those are None. That keeps this module's own tests fast/offline
and keeps a checklist and bid pack always produceable even without an
OPENAI_API_KEY.

This module only drafts a document. Per this project's stated scope (see
README.md), it never submits anything on the user's behalf - the generated
PDF is a starting point for a human to review, fill remaining gaps in, and
sign before it goes anywhere near an actual tender portal.
"""
from __future__ import annotations

import datetime as dt
import html
import io
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

from pypdf import PageObject, PdfReader, PdfWriter, Transformation
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas as pdf_canvas
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    KeepTogether,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

from app.processing.normalizer import parse_amount_from_text

if TYPE_CHECKING:
    from app.intelligence.bid_drafter import DraftedDocument, SubmissionChecklistPlan

DISCLAIMER = (
    "This is an internally generated draft, compiled automatically from the tender's "
    "own documents and the company's verified profile/document library. It is NOT a "
    "submitted bid and must not be treated as one. Every “TO BE FILLED FROM COMPANY "
    "RECORDS” placeholder and every placeholder page is a genuine gap - resolve it, and "
    "have the authorized signatory review and sign the final package, before anything "
    "is submitted to the tendering authority."
)


def _safe(text: Any) -> str:
    """Escapes text for use inside a reportlab Paragraph (which parses its
    input as a small XML subset) and swaps the rupee sign for "Rs." -
    Helvetica/the other base-14 PDF fonts this module uses have no glyph
    for it, which would otherwise render as a black box. Every string that
    ends up in a Paragraph and didn't originate as a literal in this module
    (tender titles, AI-extracted requirement text, an organisation name...)
    must go through this first - unescaped "&"/"<"/">" would otherwise be a
    malformed-XML crash waiting to happen on real tender text."""
    return html.escape(str(text), quote=False).replace("₹", "Rs. ")


class BidGenerationError(RuntimeError):
    """Raised when a bid pack can't be produced (e.g. a required company
    document is corrupt) - callers should surface this rather than return a
    partial/misleading PDF."""


@dataclass
class ComplianceRow:
    requirement: str
    source: str
    status: str
    evidence: str = ""


@dataclass
class CompanyDocumentRef:
    """One row from get_company_documents_bucket() - just enough to match
    against eligibility criteria and, for PDFs, merge into the pack."""

    id: str
    name: str
    filename: str
    content_type: str
    open_bytes: Any  # callable[[], bytes] - lazy, so unrelated docs are never read
    size: int | None = None  # bytes, from GridFS `length`; None -> measured via open_bytes when needed


# Keyword hints for the DEFAULT_CRITERIA ids in app.processing.eligibility -
# used only to make matching sharper for the common case; any criterion
# (including custom ones added via POST /eligibility) still falls back to
# the generic word-overlap match in `_match_documents` below.
_CRITERION_KEYWORD_HINTS: dict[str, list[str]] = {
    "legal-status": ["incorporation", "moa", "aoa", "cin"],
    "dpiit-startup-recognition": ["dpiit", "startup", "start-up"],
    "turnover": ["turnover", "ca certificate", "financial statement"],
    "pan-gst": ["pan", "gst", "gstr", "returns"],
    "non-blacklisting": ["blacklist", "undertaking", "declaration"],
    "industry-experience": ["profile", "brochure", "incorporation"],
    "relevant-technical-experience": ["work order", "purchase order", "completion", "client"],
    "education-digital-content-experience": ["work order", "client", "completion"],
    "social-media-digital-marketing-experience": ["work order", "agreement", "client"],
    "organizational-capability": ["profile", "brochure", "credential", "27001", "cmmi"],
    "msme-registration": ["udyam", "msme"],
    "financial-strength": ["net worth", "ca certificate", "financial statement"],
}

# Maps a criterion id to a company_profile.json field that already answers
# it without needing an uploaded scan - see config/company_profile.json.
_PROFILE_BACKED_CRITERIA: dict[str, str] = {
    "legal-status": "cin",
    "pan-gst": "gstin",
    "msme-registration": "udyam",
    "dpiit-startup-recognition": "dpiit",
}


def _profile_identifier_note(criterion_id: str, profile: dict[str, Any]) -> str | None:
    key = _PROFILE_BACKED_CRITERIA.get(criterion_id)
    if key == "cin":
        return f"CIN {profile.get('cin', '-')} (see company profile)"
    if key == "gstin":
        return f"GSTIN {profile.get('gstin', '-')} / PAN {profile.get('pan', '-')} (see company profile)"
    if key == "udyam":
        for reg in profile.get("registrations", []):
            if "UDYAM" in reg.get("name", "").upper():
                return f"UDYAM {reg.get('number', '-')} (see company profile)"
    if key == "dpiit":
        for reg in profile.get("registrations", []):
            if "DPIIT" in reg.get("name", "").upper():
                return f"DPIIT certificate {reg.get('certificate_no', '-')} (see company profile)"
    return None


# Words too generic to mean anything on their own - nearly every criterion's
# supporting_documents mentions "certificate" or "document", so requiring a
# match on one of these alone would point a criterion like "Turnover" at an
# unrelated "Certificate of Incorporation" upload. Excluded from both sides
# of the overlap in _match_documents so a match always hinges on a word
# that's actually distinctive (incorporation, turnover, udyam, brochure...).
_GENERIC_MATCH_WORDS = {
    "certificate", "certificates", "document", "documents", "copy", "copies",
    "form", "letter", "declaration", "statement", "record", "records", "credential", "credentials",
}


def _match_documents(text: str, hints: list[str], documents: list[CompanyDocumentRef]) -> list[CompanyDocumentRef]:
    """Documents whose display name shares a significant, non-generic word
    (>=4 chars) with `text`/`hints` - a deliberately simple heuristic (this
    is a small, human-curated document library, not a search index) that
    only decides whether to point at a document, never whether the
    tender's requirement is actually satisfied."""
    haystack_words = {
        w.lower() for phrase in ([text] + hints) for w in phrase.replace("/", " ").split() if len(w) >= 4
    } - _GENERIC_MATCH_WORDS
    matches = []
    for doc in documents:
        name_words = {w.lower().strip(".,()") for w in doc.name.replace("_", " ").split()} - _GENERIC_MATCH_WORDS
        if haystack_words & name_words:
            matches.append(doc)
    return matches


def build_compliance_matrix(
    eligibility_criteria: list[dict[str, Any]],
    company_profile: dict[str, Any],
    company_documents: list[CompanyDocumentRef],
) -> list[ComplianceRow]:
    rows: list[ComplianceRow] = []
    for c in eligibility_criteria:
        hints = _CRITERION_KEYWORD_HINTS.get(c["id"], []) + list(c.get("supporting_documents", []))
        matched = _match_documents(c["criterion"] + " " + c["requirement"], hints, company_documents)
        profile_note = _profile_identifier_note(c["id"], company_profile)

        if matched:
            status = "Evidence available"
            evidence = "; ".join(d.name for d in matched)
        elif profile_note:
            status = "Identifier verified - scan not yet uploaded"
            evidence = profile_note
        else:
            status = "TO BE FILLED FROM COMPANY RECORDS"
            evidence = ", ".join(c.get("supporting_documents", [])) or "-"

        rows.append(
            ComplianceRow(
                requirement=f"{c['criterion']}: {c['requirement']}",
                source="Company eligibility profile",
                status=status,
                evidence=evidence,
            )
        )
    return rows


def established_facts(rows: list[ComplianceRow]) -> list[str]:
    """Flattens the "already true" subset of a compliance matrix (evidence
    on file, or an identifier verified straight from company_profile.json)
    into plain statements - fed to app.intelligence.bid_drafter's drafting
    prompt so it never has to re-derive, or guess, which facts are actually
    established for this bidder."""
    return [
        f"{r.requirement.split(':', 1)[0]}: evidence available ({r.evidence})"
        if r.status == "Evidence available"
        else f"{r.requirement.split(':', 1)[0]}: {r.evidence}"
        for r in rows
        if r.status in ("Evidence available", "Identifier verified - scan not yet uploaded")
    ]


def _fmt_amount(value: Any) -> str:
    if value in (None, "", "null"):
        return "Not disclosed"
    if isinstance(value, (int, float)):
        return f"INR {value:,.0f}"
    return str(value)


def _fmt_date(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, (dt.date, dt.datetime)):
        return value.strftime("%d-%b-%Y")
    return str(value)


def _open_image_flowable(doc: CompanyDocumentRef | None, max_width_cm: float, max_height_cm: float):
    if doc is None:
        return None
    try:
        img = Image(io.BytesIO(doc.open_bytes()))
    except Exception:
        return None
    aspect = img.imageHeight / img.imageWidth if img.imageWidth else 1
    width = max_width_cm * cm
    height = width * aspect
    if height > max_height_cm * cm:
        height = max_height_cm * cm
        width = height / aspect if aspect else width
    img.drawWidth = width
    img.drawHeight = height
    return img


def _find_document(documents: list[CompanyDocumentRef], name: str | None) -> CompanyDocumentRef | None:
    if not name:
        return None
    return next((d for d in documents if d.name.strip().lower() == name.strip().lower()), None)


# --- Which library documents are enclosures ---------------------------------

def _is_pdf(doc: CompanyDocumentRef) -> bool:
    return doc.content_type == "application/pdf" or doc.filename.lower().endswith(".pdf")


def is_reference_document(doc: CompanyDocumentRef, company_profile: dict[str, Any]) -> bool:
    """Background material (the company brochure, a contact sheet) - read
    for drafting context (see company_background_text) but never merged
    into or listed as an enclosure of the bid pack itself. Configured by
    company_profile.json's `reference_documents`; anything named
    "...brochure..." counts too."""
    names = {n.strip().lower() for n in company_profile.get("reference_documents", [])}
    return doc.name.strip().lower() in names or "brochure" in doc.name.lower()


def _is_signing_asset(doc: CompanyDocumentRef, company_profile: dict[str, Any]) -> bool:
    """The signature/seal images - stamped onto every signed page, not
    enclosures in their own right."""
    signatory = company_profile.get("authorized_signatory", {})
    names = {
        (signatory.get("signature_document_name") or "").strip().lower(),
        (signatory.get("seal_document_name") or "").strip().lower(),
    } - {""}
    return doc.name.strip().lower() in names


def enclosure_documents(
    company_documents: list[CompanyDocumentRef], company_profile: dict[str, Any]
) -> list[CompanyDocumentRef]:
    return [
        d for d in company_documents
        if not is_reference_document(d, company_profile) and not _is_signing_asset(d, company_profile)
    ]


# The whole pack is served back through a single Lambda/API Gateway
# response (GET /tenders/{id}/bid-document, 6 MB cap), so merged PDF
# enclosures get a byte budget that leaves room for the generated pages.
ENCLOSURE_BYTE_BUDGET = 5 * 1024 * 1024

# Tender wording rarely names our documents directly ("3 years of similar
# experience", "average annual turnover") - when a trigger word appears in
# the tender's document_summary, these extra phrases are matched too.
_TENDER_TOPIC_HINTS: list[tuple[tuple[str, ...], list[str]]] = [
    (("experience", "similar", "projects", "assignments", "completion"), ["work order", "completion"]),
    (("turnover", "financial", "audited", "balance", "net worth", "itr"), ["turnover", "financial statement"]),
    (("gst", "returns", "gstr"), ["gstr", "returns"]),
    (("iso", "27001", "quality", "security"), ["27001"]),
    (("cmmi",), ["cmmi"]),
    (("incorporation", "registration", "cin", "master data"), ["master data", "incorporation"]),
]


@dataclass
class EnclosureSelection:
    selected: list[CompanyDocumentRef]
    over_budget: list[CompanyDocumentRef]  # relevant, but merging would push the pack past the size cap
    not_relevant: list[CompanyDocumentRef]  # didn't match this tender's requirements


def _doc_size(doc: CompanyDocumentRef) -> int:
    if doc.size is None:
        doc.size = len(doc.open_bytes())
    return doc.size


def select_enclosures(
    tender: dict[str, Any],
    company_documents: list[CompanyDocumentRef],
    company_profile: dict[str, Any],
    byte_budget: int = ENCLOSURE_BYTE_BUDGET,
) -> EnclosureSelection:
    """Which enclosure_documents() actually go into this tender's pack:
    the `standard_enclosures` named in company_profile.json always, then
    any document matching the tender's AI-extracted document_summary
    (documents_to_submit / eligibility requirements). A tender with no
    summary yet gets just the standard set. PDFs are added in that order
    until `byte_budget` is spent; the rest are reported, not merged."""
    candidates = enclosure_documents(company_documents, company_profile)
    standard_names = [n.strip().lower() for n in company_profile.get("standard_enclosures", [])]
    standard = sorted(
        (d for d in candidates if d.name.strip().lower() in standard_names),
        key=lambda d: standard_names.index(d.name.strip().lower()),
    )

    summary = tender.get("document_summary") or {}
    requirement_texts = (
        list(summary.get("documents_to_submit", []))
        + list(summary.get("eligibility_requirements", []))
        + list(summary.get("eligibility_technical_criteria", []))
    )
    matched: list[CompanyDocumentRef] = []
    if requirement_texts:
        combined = " ".join(requirement_texts).lower()
        hints = [h for triggers, extra in _TENDER_TOPIC_HINTS if any(t in combined for t in triggers) for h in extra]
        rest = [d for d in candidates if d not in standard]
        for text in requirement_texts:
            for d in _match_documents(text, [], rest):
                if d not in matched:
                    matched.append(d)
        for d in _match_documents("", hints, rest):
            if d not in matched:
                matched.append(d)
        matched.sort(key=rest.index)

    selected: list[CompanyDocumentRef] = []
    over_budget: list[CompanyDocumentRef] = []
    used = 0
    for d in standard + matched:
        if not _is_pdf(d):
            selected.append(d)  # never merged, so costs nothing - listed for separate attachment
            continue
        size = _doc_size(d)
        if used + size > byte_budget:
            over_budget.append(d)
            continue
        used += size
        selected.append(d)

    not_relevant = [d for d in candidates if d not in standard and d not in matched]
    return EnclosureSelection(selected=selected, over_budget=over_budget, not_relevant=not_relevant)


# --- Master Bid Submission Checklist -----------------------------------------
# Stored on the tender (`checklist`) so the team can edit it on the
# dashboard before a bid pack is generated - the pack then follows that
# saved version row by row.
#
#   {"version": 2, "generated_at", "updated_at",
#    "header": {"bid_number", "bid_end", "tender", "organisation", "bidder", "summary_only"},
#    "items": [submission rows - see submission_item], "notes": [review notes - see checklist_item],
#    "builder_note": str | None}

CHECKLIST_VERSION = 2

STATUS_ENCLOSED = "Enclosed"
STATUS_TO_PREPARE = "To be prepared"
STATUS_MISSING = "Missing"
STATUS_NOT_APPLICABLE = "Not applicable"
SUBMISSION_STATUSES = (STATUS_ENCLOSED, STATUS_TO_PREPARE, STATUS_MISSING, STATUS_NOT_APPLICABLE)

# Review notes kept alongside the rows: things to verify before signing,
# and information the tender's documents don't state.
NOTE_SECTIONS = ("open_items", "missing_information")

DEFAULT_WHERE = "Technical Upload"
COVERING_LETTER = "Covering Letter"


def is_submission_checklist(checklist: dict[str, Any] | None) -> bool:
    """False for a checklist saved in the older six-section format (or none
    at all) - those are rebuilt rather than shown half-understood."""
    return bool(checklist) and checklist.get("version") == CHECKLIST_VERSION


def submission_item(
    document: str,
    what_to_upload: str = "",
    where: str = DEFAULT_WHERE,
    source: str = "upload",
    document_id: str | None = None,
    letterhead: bool = False,
    signature: bool = False,
    stamp: bool = False,
    status: str = STATUS_ENCLOSED,
    format_text: str = "",
    notes: str = "",
    origin: str = "auto",
) -> dict[str, Any]:
    return {
        "id": uuid.uuid4().hex,
        "document": document,
        "what_to_upload": what_to_upload,
        "where": where or DEFAULT_WHERE,
        "status": status,
        "source": source,  # upload (attach a library document) | draft (the bidder writes it)
        "document_id": document_id,
        "letterhead": letterhead,
        "signature": signature,
        "stamp": stamp,
        "format_text": format_text,  # the tender's prescribed format / drafting instructions
        "notes": notes,
        "done": False,
        "origin": origin,  # auto (build_checklist) | user (added on the dashboard)
    }


def checklist_item(
    section: str,
    requirement: str,
    source: str = "",
    status: str = "",
    evidence: str = "",
    origin: str = "auto",
    done: bool = False,
) -> dict[str, Any]:
    """One review note (see NOTE_SECTIONS)."""
    return {
        "id": uuid.uuid4().hex,
        "section": section,
        "requirement": requirement,
        "source": source,
        "status": status,
        "evidence": evidence,
        "done": done,
        "origin": origin,  # auto (build_checklist) | user (added on the dashboard) | ai_draft
    }


# Documents the bidder writes itself vs. records it already holds - only
# used when the AI checklist is unavailable (the AI decides this itself
# otherwise, from the tender's own wording).
_BIDDER_WRITTEN_RE = re.compile(
    r"undertaking|declaration|affidavit|annex|appendix|format|\bform\b|letter|proposal|methodology|"
    r"presentation|price bid|financial bid|commercial bid|\bboq\b|power of attorney|authori[sz]ation|"
    r"self[- ]certif|compliance|particulars|no[- ]deviation|acceptance|signed tender",
    re.IGNORECASE,
)
# Signed by someone other than the bidder (a CA, a bank) - attached as-is,
# never self-attested.
_THIRD_PARTY_SIGNED_RE = re.compile(
    r"\bca\b|chartered accountant|audited|balance sheet|profit (and|&) loss|\bitr\b|income tax return|"
    r"bank guarantee|demand draft|\bemd\b|earnest money|bid security",
    re.IGNORECASE,
)
_FINANCIAL_RE = re.compile(r"price|financial bid|commercial bid|\bboq\b|rate quot", re.IGNORECASE)


def default_marks(source: str, text: str) -> tuple[bool, bool, bool]:
    """(letterhead, signature, stamp) a document normally needs: everything
    the bidder writes goes on the letterhead, signed and stamped; copies of
    its own records are self-attested (signed and stamped); third-party
    signed documents are attached untouched."""
    if source == "draft":
        return True, True, True
    if _THIRD_PARTY_SIGNED_RE.search(text):
        return False, False, False
    return False, True, True


def _resolve_library_document(
    name: str | None, candidates: list[CompanyDocumentRef]
) -> CompanyDocumentRef | None:
    """The library document the AI named - by exact name, else the single
    document whose name overlaps it (never a guess between several)."""
    if not name:
        return None
    doc = _find_document(candidates, name)
    if doc is not None:
        return doc
    matches = _match_documents(name, [], candidates)
    return matches[0] if len(matches) == 1 else None


# closing_date comes from the listing as a bare date (stored as midnight
# UTC) - shown as a date; anything with a real time is shown in IST.
IST = dt.timezone(dt.timedelta(hours=5, minutes=30))


def _fmt_bid_end(value: Any) -> str:
    if isinstance(value, dt.datetime):
        if (value.hour, value.minute) == (0, 0):
            return value.strftime("%d-%m-%Y")
        aware = value if value.tzinfo else value.replace(tzinfo=dt.timezone.utc)
        return aware.astimezone(IST).strftime("%d-%m-%Y, %H:%M Hrs")
    if isinstance(value, dt.date):
        return value.strftime("%d-%m-%Y")
    return str(value or "")


def checklist_header(
    tender: dict[str, Any],
    company_profile: dict[str, Any],
    bid_number: str | None = None,
    bid_end: str | None = None,
) -> dict[str, Any]:
    return {
        "bid_number": (bid_number or tender.get("tender_ref") or "").strip(),
        "bid_end": (bid_end or _fmt_bid_end(tender.get("closing_date"))).strip(),
        "tender": tender.get("title") or "",
        "organisation": tender.get("organisation") or "",
        "bidder": company_profile.get("legal_name") or "",
        # Built without the tender's own document text (see
        # app.intelligence.document_summarizer's `document_text`) - only
        # from its summary, so annexure-level detail may be missing.
        "summary_only": not (tender.get("document_text") or "").strip(),
    }


def _row_status(source: str, doc: CompanyDocumentRef | None) -> str:
    if source == "draft":
        return STATUS_TO_PREPARE
    return STATUS_ENCLOSED if doc is not None else STATUS_MISSING


def _library_row(doc: CompanyDocumentRef, notes: str = "") -> dict[str, Any]:
    letterhead, signature, stamp = default_marks("upload", doc.name)
    return submission_item(
        doc.name, f"Copy of {doc.name}", DEFAULT_WHERE, "upload", doc.id,
        letterhead, signature, stamp, STATUS_ENCLOSED, notes=notes,
    )


def _covering_letter_row() -> dict[str, Any]:
    return submission_item(
        COVERING_LETTER, "Covering letter submitting the offer and accepting the tender's terms",
        DEFAULT_WHERE, "draft", None, True, True, True, STATUS_TO_PREPARE,
    )


def _rows_from_plan(
    plan: "SubmissionChecklistPlan", candidates: list[CompanyDocumentRef]
) -> list[dict[str, Any]]:
    rows = []
    for r in plan.rows:
        doc = _resolve_library_document(r.library_document, candidates) if r.source == "upload" else None
        rows.append(submission_item(
            r.document.strip(), r.what_to_upload.strip(), r.where.strip(), r.source,
            doc.id if doc else None, r.letterhead, r.signature, r.stamp,
            _row_status(r.source, doc), r.format_hint.strip(), r.notes.strip(),
        ))
    return rows


def _rows_from_summary(tender: dict[str, Any], selection: EnclosureSelection) -> list[dict[str, Any]]:
    """The deterministic fallback: one row per document the summary says to
    submit (matched to a selected library document where one fits), plus a
    row per remaining selected library document."""
    summary = tender.get("document_summary") or {}
    to_submit = [t for t in summary.get("documents_to_submit", []) if t.strip()]
    rows: list[dict[str, Any]] = []
    if not any(re.search(r"covering letter|bid (submission )?(form|letter)", t, re.IGNORECASE) for t in to_submit):
        rows.append(_covering_letter_row())

    used: set[str] = set()
    for text in to_submit:
        source = "draft" if _BIDDER_WRITTEN_RE.search(text) else "upload"
        doc = None
        if source == "upload":
            doc = next((d for d in _match_documents(text, [], selection.selected) if d.id not in used), None)
            if doc is not None:
                used.add(doc.id)
        letterhead, signature, stamp = default_marks(source, text)
        rows.append(submission_item(
            text.strip()[:80], text.strip(), "Financial Bid" if _FINANCIAL_RE.search(text) else DEFAULT_WHERE,
            source, doc.id if doc else None, letterhead, signature, stamp, _row_status(source, doc),
        ))
    rows += [_library_row(d) for d in selection.selected if d.id not in used]
    return rows


def _checklist_notes(tender: dict[str, Any], company_profile: dict[str, Any]) -> list[dict[str, Any]]:
    summary = tender.get("document_summary") or {}
    open_items = list(company_profile.get("to_be_verified", []))
    if company_profile.get("pan_derivation_note"):
        open_items.append(f"PAN {company_profile.get('pan', '-')}: {company_profile['pan_derivation_note']}")
    notes = [checklist_item("open_items", text) for text in open_items]
    notes += [checklist_item("missing_information", text) for text in summary.get("missing_information", [])]
    return notes


def build_checklist(
    tender: dict[str, Any],
    eligibility_criteria: list[dict[str, Any]],
    company_profile: dict[str, Any],
    company_documents: list[CompanyDocumentRef],
    plan: "SubmissionChecklistPlan | None" = None,
    builder_note: str | None = None,
) -> dict[str, Any]:
    """The Master Bid Submission Checklist for this tender - from `plan`
    (app.intelligence.bid_drafter.plan_submission_checklist) when given,
    else from the tender's document_summary and the library (see
    _rows_from_summary; `builder_note` says why). The company profile's
    `standard_enclosures` are always listed, and so is the covering letter
    when nothing else covers it."""
    candidates = enclosure_documents(company_documents, company_profile)
    selection = select_enclosures(tender, company_documents, company_profile)
    if plan is not None and plan.rows:
        rows = _rows_from_plan(plan, candidates)
    else:
        rows = _rows_from_summary(tender, selection)

    referenced = {r["document_id"] for r in rows if r["document_id"]}
    standard_names = [n.strip().lower() for n in company_profile.get("standard_enclosures", [])]
    for d in candidates:
        if d.name.strip().lower() in standard_names and d.id not in referenced:
            rows.append(_library_row(d, notes="Standard enclosure (company profile)"))

    now = dt.datetime.now(dt.timezone.utc)
    checklist = {
        "version": CHECKLIST_VERSION,
        "generated_at": now,
        "updated_at": now,
        "header": checklist_header(
            tender, company_profile,
            plan.bid_number if plan is not None else None,
            plan.bid_end if plan is not None else None,
        ),
        "items": rows,
        "notes": _checklist_notes(tender, company_profile),
        "builder_note": builder_note,
    }
    return drop_resolved_missing_information(checklist, tender)


# The summarizer's "missing_information" only reflects what the attached
# documents don't state - but the listing itself (tender_value,
# earnest_money, closing_date...) or another field of the same summary can
# still supply it. Each rule: (pattern a missing-info line is about, what
# not to confuse it with, tender fields, summary fields) - the line is
# dropped once any of those fields has a value.
_TENDER_VALUE_PATTERN = (
    r"tender value|estimated (bid |contract |project )?(value|cost|amount)|bid amount|contract value|project value|estimated cost"
)
_RESOLVABLE_MISSING_INFO = (
    (_TENDER_VALUE_PATTERN, None, ("tender_value",), ("estimated_bid_amount",)),
    (r"\bemd\b|earnest money|bid security",
     r"exempt|mode|refund|format|validity|form of|instrument", ("earnest_money",), ("emd_amount",)),
    (r"tender fee|document fee|processing fee|cost of (tender|bid) document",
     r"exempt|mode|refund", ("document_fees",), ("tender_fee_amount",)),
    (r"opening date|bid opening|tender opening|date of opening",
     None, ("opening_date",), ("tender_opening_date",)),
    (r"closing date|submission (deadline|date)|last date|due date|bid end date",
     None, ("closing_date",), ()),
    (r"published date|publish(ing)? date|date of publication",
     None, ("published_date",), ()),
)


def _has_value(value: Any) -> bool:
    return value not in (None, "", [])


def is_missing_info_resolved(text: str, tender: dict[str, Any]) -> bool:
    summary = tender.get("document_summary") or {}
    lowered = text.lower()
    for pattern, exclude, tender_fields, summary_fields in _RESOLVABLE_MISSING_INFO:
        if not re.search(pattern, lowered) or (exclude and re.search(exclude, lowered)):
            continue
        if any(_has_value(tender.get(f)) for f in tender_fields) or any(
            _has_value(summary.get(f)) for f in summary_fields
        ):
            return True
    return False


# EMD is usually set at 2% of the estimated tender value (the GFR norm most
# Indian tenders follow), so when neither the listing nor the documents
# state a value, 50x the EMD is a reasonable ballpark - always labelled as
# an estimate to confirm, never presented as the tender's stated value.
EMD_SHARE_OF_TENDER_VALUE = 0.02
ESTIMATED_TENDER_VALUE_LABEL = "Estimated tender value"


def estimated_tender_value(tender: dict[str, Any]) -> tuple[str, bool] | None:
    """(note for the checklist, whether it's a stated value) - None when
    there is nothing to go on at all."""
    summary = tender.get("document_summary") or {}
    if _has_value(tender.get("tender_value")):
        return f"{_fmt_amount(tender['tender_value'])} (from the tender listing)", True
    if _has_value(summary.get("estimated_bid_amount")):
        return f"{summary['estimated_bid_amount']} (from the tender documents)", True
    emd = tender.get("earnest_money")
    if not _has_value(emd):
        emd = parse_amount_from_text(summary.get("emd_amount"))
    if emd:
        return (
            f"~{_fmt_amount(emd / EMD_SHARE_OF_TENDER_VALUE)} - not stated; estimated from the EMD of "
            f"{_fmt_amount(emd)} assuming the usual {EMD_SHARE_OF_TENDER_VALUE:.0%} EMD rate. Confirm on the portal.",
            False,
        )
    return None


def _estimated_tender_value_item(tender: dict[str, Any]) -> dict[str, Any] | None:
    estimate = estimated_tender_value(tender)
    if estimate is None:
        return None
    note, stated = estimate
    return checklist_item("missing_information", ESTIMATED_TENDER_VALUE_LABEL, evidence=note, done=stated)


def drop_resolved_missing_information(checklist: dict[str, Any], tender: dict[str, Any]) -> dict[str, Any]:
    """Removes auto-generated "missing information" notes the tender record
    already answers, and adds the estimated tender value note - also cleans
    checklists saved before this existed. Notes the user added or edited by
    hand (origin "user") stay."""
    has_estimate = estimated_tender_value(tender) is not None
    notes = [
        i for i in checklist.get("notes", [])
        if not (
            i.get("section") == "missing_information"
            and i.get("origin", "auto") == "auto"
            and i.get("requirement") != ESTIMATED_TENDER_VALUE_LABEL
            and (
                is_missing_info_resolved(i.get("requirement") or "", tender)
                or (has_estimate and re.search(_TENDER_VALUE_PATTERN, (i.get("requirement") or "").lower()))
            )
        )
    ]
    if not any(i.get("requirement") == ESTIMATED_TENDER_VALUE_LABEL for i in notes):
        row = _estimated_tender_value_item(tender)
        if row:
            notes.append(row)
    return {**checklist, "notes": notes}


def merge_drafted_open_items(
    checklist: dict[str, Any], drafted_documents: "Iterable[DraftedDocument] | None"
) -> dict[str, Any]:
    """Swaps in the open items of this run's AI-drafted documents, replacing
    any from a previous run - user-edited/added notes are left untouched,
    and a drafted item already ticked done stays done if it recurs."""
    notes = checklist.get("notes", [])
    previous = [i for i in notes if i.get("origin") == "ai_draft"]
    done_texts = {i.get("requirement") for i in previous if i.get("done")}
    kept = [i for i in notes if i.get("origin") != "ai_draft"]
    drafted = [
        checklist_item("open_items", text, origin="ai_draft", done=text in done_texts)
        for doc in drafted_documents or []
        for text in (f"{doc.title}: {item}" for item in doc.open_items)
    ]
    return {**checklist, "notes": drafted + kept}


# --- What the pack does with each row ------------------------------------------

@dataclass
class RowPlan:
    """How generate_bid_package renders one checklist row."""

    action: str  # "draft" | "attach" | "placeholder" | "skip"
    doc: CompanyDocumentRef | None = None
    reason: str = ""  # why a placeholder stands in


def _is_image(doc: CompanyDocumentRef) -> bool:
    return doc.content_type.startswith("image/") or doc.filename.lower().endswith((".png", ".jpg", ".jpeg"))


def plan_rows(
    checklist: dict[str, Any],
    company_documents: list[CompanyDocumentRef],
    byte_budget: int = ENCLOSURE_BYTE_BUDGET,
) -> dict[str, RowPlan]:
    """Per row id: draft it, attach its library document, or stand in a
    placeholder page (document missing, not a PDF/image, or over the pack's
    size budget - attached in S.No order until the budget runs out)."""
    by_id = {d.id: d for d in company_documents}
    plans: dict[str, RowPlan] = {}
    used = 0
    for row in checklist.get("items", []):
        if row.get("status") == STATUS_NOT_APPLICABLE:
            plans[row["id"]] = RowPlan("skip")
            continue
        if row.get("source") == "draft":
            plans[row["id"]] = RowPlan("draft")
            continue
        doc = by_id.get(row.get("document_id") or "")
        if doc is None:
            plans[row["id"]] = RowPlan("placeholder", reason="Not in the Documents library yet - upload it and "
                                                              "pick it on the checklist, or attach it on the portal.")
        elif not (_is_pdf(doc) or _is_image(doc)):
            plans[row["id"]] = RowPlan("placeholder", doc, f"{doc.filename} is not a PDF or image, so it can't be "
                                                           "merged into this pack - attach it separately.")
        elif used + _doc_size(doc) > byte_budget:
            plans[row["id"]] = RowPlan("placeholder", doc, f"{doc.filename} would push the pack past its size "
                                                           "limit - attach it separately from the Documents library.")
        else:
            used += _doc_size(doc)
            plans[row["id"]] = RowPlan("attach", doc)
    return plans


def refresh_statuses(
    checklist: dict[str, Any],
    company_documents: list[CompanyDocumentRef],
    drafted_ids: Iterable[str] = (),
) -> dict[str, Any]:
    """Each row's Status as the generated pack actually has it: Enclosed once
    drafted/attached, To be prepared for a document still to write, Missing
    for one not attached. Not applicable is the user's call and stays."""
    drafted_ids = set(drafted_ids)
    plans = plan_rows(checklist, company_documents)
    items = []
    for row in checklist.get("items", []):
        plan = plans[row["id"]]
        if plan.action == "draft":
            status = STATUS_ENCLOSED if row["id"] in drafted_ids else STATUS_TO_PREPARE
        elif plan.action == "attach":
            status = STATUS_ENCLOSED
        elif plan.action == "placeholder":
            status = STATUS_MISSING
        else:
            status = row.get("status")
        items.append({**row, "status": status})
    return {**checklist, "items": items}


# Brochures repeat the same text across facing pages - the cap keeps the
# drafting prompt small; it's context, not something to quote wholesale.
_BACKGROUND_MAX_CHARS = 6000


def company_background_text(company_documents: list[CompanyDocumentRef], company_profile: dict[str, Any]) -> str:
    """Plain text of every PDF reference document (see
    is_reference_document), de-duplicated line by line - handed to
    app.intelligence.bid_drafter as descriptive company background. Empty
    string when there's nothing readable; never raises, since a bid pack
    must still generate without it."""
    seen: set[str] = set()
    lines: list[str] = []
    for doc in company_documents:
        if not (_is_pdf(doc) and is_reference_document(doc, company_profile)):
            continue
        try:
            reader = PdfReader(io.BytesIO(doc.open_bytes()))
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception:
            continue
        for line in text.splitlines():
            line = " ".join(line.split())
            if len(line) < 3 or line.lower() in seen:
                continue
            seen.add(line.lower())
            lines.append(line)
    return "\n".join(lines)[:_BACKGROUND_MAX_CHARS]


# --- Page templates -----------------------------------------------------------

# Content-frame insets (left, right, top, bottom) that clear the Oaks
# letterhead's left colour band, header logo/address block and footer
# contact strip - measured off config/letterhead.jpg (a full A4 page).
LETTERHEAD_MARGINS = (2.1 * cm, 2.1 * cm, 3.7 * cm, 3.3 * cm)
PLAIN_MARGINS = (1.8 * cm, 1.8 * cm, 1.6 * cm, 1.6 * cm)


def _build_pdf(story: list[Any], title: str, letterhead_image: str | Path | None) -> bytes:
    """Renders `story` to PDF bytes - on the letterhead (drawn full-page
    behind every page) when one is given and readable, else plain pages."""
    background = None
    if letterhead_image:
        try:
            background = ImageReader(str(letterhead_image))
        except Exception:
            background = None
    left, right, top, bottom = LETTERHEAD_MARGINS if background else PLAIN_MARGINS

    def _draw_background(canvas, _doc):
        if background is not None:
            canvas.drawImage(background, 0, 0, width=A4[0], height=A4[1])

    buf = io.BytesIO()
    doc = BaseDocTemplate(buf, pagesize=A4, leftMargin=left, rightMargin=right, topMargin=top,
                          bottomMargin=bottom, title=title)
    frame = Frame(left, bottom, A4[0] - left - right, A4[1] - top - bottom, id="body",
                  leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
    doc.addPageTemplates([PageTemplate(id="page", frames=[frame], onPage=_draw_background)])
    doc.build(story)
    return buf.getvalue()


# --- Text helpers ----------------------------------------------------------------

_PLACEHOLDER_RE = re.compile(r"(\[TO BE (?:FILLED|VERIFIED)[^\]]*\])")


def _rich(text: Any) -> str:
    """_safe, plus any "[TO BE FILLED FROM COMPANY RECORDS: ...]" placeholder
    highlighted so a reviewer can't miss it on an otherwise finished page."""
    return _PLACEHOLDER_RE.sub(r'<font backColor="#fff3bf" color="#c92a2a">\1</font>', _safe(text))


def _format_paragraph(text: str) -> str:
    """Bolds a letter's "Subject:"/"Ref:" line - the AI drafts these as
    ordinary body paragraphs."""
    stripped = text.strip()
    if stripped.lower().startswith(("subject:", "sub:", "ref:", "reference:")):
        return f"<b>{_rich(stripped)}</b>"
    return _rich(stripped)


def _date_line(body: ParagraphStyle) -> Paragraph:
    return Paragraph(f"Date: {dt.date.today().strftime('%d-%b-%Y')}", ParagraphStyle(
        "dateline", parent=body, alignment=TA_RIGHT))


def _signing_assets(
    company_profile: dict[str, Any], company_documents: list[CompanyDocumentRef]
) -> tuple[CompanyDocumentRef | None, CompanyDocumentRef | None]:
    """The signature and seal images named in company_profile.json's
    authorized_signatory, when they're in the library."""
    signatory = company_profile.get("authorized_signatory", {})
    return (
        _find_document(company_documents, signatory.get("signature_document_name")),
        _find_document(company_documents, signatory.get("seal_document_name")),
    )


def _signature_block(
    company_profile: dict[str, Any],
    company_documents: list[CompanyDocumentRef],
    body: ParagraphStyle,
    signature: bool = True,
    stamp: bool = True,
) -> list[Any]:
    """"For <company>", the signature and/or seal image (each only when the
    row asks for it and the image is in the library) and the printed
    name/designation/place - kept together on one page."""
    signatory = company_profile.get("authorized_signatory", {})
    flow: list[Any] = [
        Spacer(1, 6),
        Paragraph(f"For <b>{_safe(company_profile.get('legal_name', '[Company]'))}</b>", body),
    ]
    sig_doc, seal_doc = _signing_assets(company_profile, company_documents)
    sig_img = _open_image_flowable(sig_doc, 3.5, 1.6) if signature else None
    seal_img = _open_image_flowable(seal_doc, 2.8, 2.8) if stamp else None
    if sig_img or seal_img:
        t = Table([[sig_img or "", seal_img or ""]], colWidths=[6 * cm, 6 * cm], hAlign="LEFT")
        t.setStyle(TableStyle([("VALIGN", (0, 0), (-1, -1), "BOTTOM"), ("LEFTPADDING", (0, 0), (0, 0), 0)]))
        flow.append(t)
    else:
        flow.append(Spacer(1, 36))
    lines = [
        f"<b>{_safe(signatory.get('name', '[TO BE FILLED FROM COMPANY RECORDS: signatory]'))}</b>",
        _safe(signatory.get("designation", "Authorized Signatory")),
    ]
    if signatory.get("place"):
        lines.append(f"Place: {_safe(signatory['place'])}")
    flow.append(Paragraph("<br/>".join(lines), body))
    return [KeepTogether(flow)]


@dataclass
class _Styles:
    base: Any
    body: ParagraphStyle
    h1: ParagraphStyle
    h2: ParagraphStyle
    small: ParagraphStyle
    cell: ParagraphStyle


def _styles() -> _Styles:
    base = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=base["BodyText"], fontSize=10.5, leading=15, alignment=TA_JUSTIFY)
    return _Styles(
        base=base,
        body=body,
        h1=ParagraphStyle("h1", parent=base["Heading1"], fontSize=15, alignment=TA_CENTER, spaceBefore=4,
                          spaceAfter=14, textColor=colors.HexColor("#0b3a5b")),
        h2=ParagraphStyle("h2", parent=base["Heading2"], fontSize=12, spaceBefore=14, spaceAfter=8,
                          textColor=colors.HexColor("#0b3a5b")),
        small=ParagraphStyle("small", parent=body, fontSize=8.5, leading=11, alignment=TA_LEFT),
        cell=ParagraphStyle("cell", parent=body, fontSize=8.5, leading=10.5, alignment=TA_LEFT),
    )


# --- The checklist page (first page of the pack, on the letterhead) -------------

_STATUS_FILL = {
    STATUS_ENCLOSED: "#ebfbee",
    STATUS_TO_PREPARE: "#fff3bf",
    STATUS_MISSING: "#ffe3e3",
    STATUS_NOT_APPLICABLE: "#f1f3f5",
}

_HEADER_STYLE = [
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0b3a5b")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#c9d2db")),
    ("VALIGN", (0, 0), (-1, -1), "TOP"),
]


def _submission_checklist_story(checklist: dict[str, Any], s: _Styles) -> list[Any]:
    header = checklist.get("header") or {}
    heading = ParagraphStyle("cl_heading", parent=s.h1, fontSize=13, leading=16, spaceAfter=2)
    info = ParagraphStyle("cl_info", parent=s.body, fontSize=9.5, leading=13, alignment=TA_LEFT)
    head_cell = ParagraphStyle("cl_head_cell", parent=s.cell, textColor=colors.white, fontName="Helvetica-Bold")
    flow: list[Any] = []
    if header.get("organisation"):
        flow.append(Paragraph(_safe(header["organisation"]).upper(), heading))
    flow.append(Paragraph("MASTER BID SUBMISSION CHECKLIST", ParagraphStyle("cl_title", parent=heading, spaceAfter=10)))
    for label, key in (("GeM Bid No.", "bid_number"), ("Bid End Date/Time", "bid_end"),
                       ("Tender", "tender"), ("Bidder", "bidder")):
        if header.get(key):
            flow.append(Paragraph(f"<b>{label}:</b> {_safe(header[key])}", info))
    flow.append(Paragraph("Submission Checklist", ParagraphStyle("cl_sub", parent=s.h2, spaceBefore=8)))

    rows = [[Paragraph(h, head_cell) for h in ("S.No.", "Document", "What to Upload", "Where", "Status")]]
    style = list(_HEADER_STYLE)
    for n, item in enumerate(checklist.get("items", []), 1):
        status = item.get("status") or "-"
        rows.append([
            str(n),
            Paragraph(_safe(item.get("document") or "-"), s.cell),
            Paragraph(_safe(item.get("what_to_upload") or "-"), s.cell),
            Paragraph(_safe(item.get("where") or "-"), s.cell),
            Paragraph(_safe(status), s.cell),
        ])
        if status in _STATUS_FILL:
            style.append(("BACKGROUND", (4, n), (4, n), colors.HexColor(_STATUS_FILL[status])))
    table = Table(rows, colWidths=[1.35 * cm, 3.3 * cm, 6.45 * cm, 2.9 * cm, 2.6 * cm], repeatRows=1)
    table.setStyle(TableStyle(style + [("FONTSIZE", (0, 1), (0, -1), 8.5), ("ALIGN", (0, 0), (0, -1), "CENTER")]))
    flow.append(table)
    return flow


# --- One page set per checklist row ------------------------------------------------

def _drafted_document_section(
    doc: "DraftedDocument",
    row: dict[str, Any],
    company_profile: dict[str, Any],
    company_documents: list[CompanyDocumentRef],
    s: _Styles,
) -> list[Any]:
    """One AI-drafted document (app.intelligence.bid_drafter) as its own
    page(s): date, title, body paragraphs, then the signature block with
    the signature/seal the row asks for. Its open items go to the internal
    review notes, not onto the page itself."""
    flow: list[Any] = [_date_line(s.body), Paragraph(_safe(doc.title or row.get("document") or ""), s.h1)]
    for para in doc.body_paragraphs:
        if para.strip():
            flow.append(Paragraph(_format_paragraph(para), s.body))
            flow.append(Spacer(1, 6))
    flow += _signature_block(company_profile, company_documents, s.body,
                             bool(row.get("signature")), bool(row.get("stamp")))
    return flow


def _fallback_covering_letter_section(
    tender: dict[str, Any],
    row: dict[str, Any],
    company_profile: dict[str, Any],
    company_documents: list[CompanyDocumentRef],
    s: _Styles,
) -> list[Any]:
    """A plain, templated covering letter - used when the covering letter
    row has no AI draft (no OPENAI_API_KEY, or the drafting call failed)."""
    subject = _safe(tender.get("title") or "the above tender")
    ref_line = f" (Ref: {_safe(tender['tender_ref'])})" if tender.get("tender_ref") else ""
    flow: list[Any] = [
        _date_line(s.body),
        Paragraph(_safe(row.get("document") or COVERING_LETTER), s.h1),
        Paragraph(f"To,<br/>{_safe(tender.get('organisation') or '[Tendering Authority]')}", s.body),
        Spacer(1, 8),
        Paragraph(f"<b>Subject: Techno-Commercial Offer for “{subject}”{ref_line}</b>", s.body),
        Spacer(1, 8),
        Paragraph(
            f"Dear Sir/Madam,<br/><br/>"
            f"We, {_safe(company_profile.get('legal_name', '[Company]'))}, submit our offer for the above tender. "
            "We confirm that we have read and understood the tender document, including all terms, conditions, "
            "annexures and any corrigenda issued up to the bid submission date, and that our offer conforms to "
            "the tender's requirements.<br/><br/>"
            "The documents required by the tender are enclosed with this letter, as listed in the submission "
            "checklist.",
            s.body,
        ),
        Spacer(1, 16),
        Paragraph("Yours faithfully,", s.body),
    ]
    flow += _signature_block(company_profile, company_documents, s.body,
                             bool(row.get("signature")), bool(row.get("stamp")))
    return flow


_PLACEHOLDER_BOX = dict(borderColor=colors.HexColor("#c92a2a"), borderWidth=0.8, borderPadding=10,
                        backColor=colors.HexColor("#fff5f5"))


def _undrafted_document_section(
    row: dict[str, Any],
    company_profile: dict[str, Any],
    company_documents: list[CompanyDocumentRef],
    s: _Styles,
) -> list[Any]:
    """A draft row with no AI draft: its title, what it must contain and the
    signature block, so the page is ready to be completed by hand."""
    flow: list[Any] = [
        _date_line(s.body),
        Paragraph(_safe(row.get("document") or ""), s.h1),
        Paragraph(_rich(
            f"[TO BE FILLED FROM COMPANY RECORDS: {row.get('what_to_upload') or row.get('document')}]"
        ), s.body),
    ]
    if row.get("format_text"):
        flow += [Spacer(1, 6), Paragraph(f"<i>Format: {_safe(row['format_text'])}</i>", s.small)]
    flow += [Spacer(1, 24)]
    flow += _signature_block(company_profile, company_documents, s.body,
                             bool(row.get("signature")), bool(row.get("stamp")))
    return flow


def _placeholder_section(n: int, row: dict[str, Any], reason: str, s: _Styles) -> list[Any]:
    """Stands in, at the right position in the pack, for a document that
    couldn't be attached - so the pack's order still matches the checklist."""
    return [
        Spacer(1, 3 * cm),
        Paragraph(f"{n}. {_safe(row.get('document') or '')}", s.h1),
        Paragraph(
            f"<b>PLACEHOLDER - replace with the actual document before submission.</b><br/><br/>"
            f"To attach: {_safe(row.get('what_to_upload') or row.get('document') or '')}<br/>"
            f"Where: {_safe(row.get('where') or '-')}<br/><br/>{_safe(reason)}",
            ParagraphStyle("placeholder", parent=s.body, alignment=TA_LEFT, **_PLACEHOLDER_BOX),
        ),
    ]


# --- Library documents: letterhead placement and signature/seal overlay -------------

def _letterhead_page(letterhead_image: str | Path | None) -> PageObject | None:
    """A blank A4 page with just the letterhead drawn on it - uploaded pages
    are scaled into its content frame (see _onto_letterhead)."""
    if not letterhead_image:
        return None
    try:
        image = ImageReader(str(letterhead_image))
    except Exception:
        return None
    buf = io.BytesIO()
    c = pdf_canvas.Canvas(buf, pagesize=A4)
    c.drawImage(image, 0, 0, width=A4[0], height=A4[1])
    c.showPage()
    c.save()
    return PdfReader(io.BytesIO(buf.getvalue())).pages[0]


def _onto_letterhead(writer: PdfWriter, page: PageObject, letterhead: PageObject) -> PageObject:
    """Adds a letterhead page to `writer` with `page` scaled into its content
    frame (top-aligned, never enlarged)."""
    left, right, top, bottom = LETTERHEAD_MARGINS
    box_w, box_h = A4[0] - left - right, A4[1] - top - bottom
    llx, lly = float(page.mediabox.left), float(page.mediabox.bottom)
    width, height = float(page.mediabox.width), float(page.mediabox.height)
    scale = min(box_w / width, box_h / height, 1.0)
    base = writer.add_blank_page(width=A4[0], height=A4[1])
    base.merge_page(letterhead)
    base.merge_transformed_page(
        page,
        Transformation().translate(-llx, -lly).scale(scale, scale).translate(
            left + (box_w - width * scale) / 2, bottom + box_h - height * scale
        ),
    )
    return base


def _signing_overlay(
    width: float, height: float, bottom: float, signature: bytes | None, seal: bytes | None
) -> PageObject | None:
    """A transparent page carrying the signature and/or seal at its bottom
    right - merged over each page of a self-attested copy."""
    images = []
    for data, box_w, box_h in ((signature, 3.4 * cm, 1.5 * cm), (seal, 2.4 * cm, 2.4 * cm)):
        if data:
            try:
                images.append((ImageReader(io.BytesIO(data)), box_w, box_h))
            except Exception:
                continue
    if not images:
        return None
    buf = io.BytesIO()
    c = pdf_canvas.Canvas(buf, pagesize=(width, height))
    x = width - 1 * cm
    for image, box_w, box_h in reversed(images):  # seal rightmost, signature to its left
        x -= box_w
        c.drawImage(image, x, bottom, width=box_w, height=box_h, preserveAspectRatio=True, anchor="sw", mask="auto")
        x -= 0.3 * cm
    c.showPage()
    c.save()
    return PdfReader(io.BytesIO(buf.getvalue())).pages[0]


def _image_as_pdf(data: bytes) -> bytes:
    """An image upload (a scanned certificate) as a one-page A4 PDF."""
    image = ImageReader(io.BytesIO(data))
    img_w, img_h = image.getSize()
    left, right, top, bottom = PLAIN_MARGINS
    box_w, box_h = A4[0] - left - right, A4[1] - top - bottom
    scale = min(box_w / img_w, box_h / img_h)
    buf = io.BytesIO()
    c = pdf_canvas.Canvas(buf, pagesize=A4)
    c.drawImage(image, left + (box_w - img_w * scale) / 2, A4[1] - top - img_h * scale,
                width=img_w * scale, height=img_h * scale, mask="auto")
    c.showPage()
    c.save()
    return buf.getvalue()


def _add_attached_pages(
    writer: PdfWriter,
    doc: CompanyDocumentRef,
    row: dict[str, Any],
    letterhead: PageObject | None,
    signature: bytes | None,
    seal: bytes | None,
) -> None:
    """Adds a library document's pages to `writer` - placed on the
    letterhead and/or signed and stamped when the row asks for it."""
    data = doc.open_bytes()
    try:
        pages = list(PdfReader(io.BytesIO(data if _is_pdf(doc) else _image_as_pdf(data))).pages)
    except Exception as exc:
        raise BidGenerationError(f"Could not read '{doc.name}': {exc}") from exc

    on_letterhead = bool(row.get("letterhead")) and letterhead is not None
    for source in pages:
        page = writer.add_page(source)
        page.transfer_rotation_to_content()
        if on_letterhead:
            placed = _onto_letterhead(writer, page, letterhead)
            writer.remove_page(page)
            page = placed
        overlay = _signing_overlay(
            float(page.mediabox.width), float(page.mediabox.height),
            LETTERHEAD_MARGINS[3] if on_letterhead else 1 * cm,
            signature if row.get("signature") else None,
            seal if row.get("stamp") else None,
        )
        if overlay is not None:
            page.merge_page(overlay)


def _read_bytes(doc: CompanyDocumentRef | None) -> bytes | None:
    if doc is None:
        return None
    try:
        return doc.open_bytes()
    except Exception:
        return None


# --- Internal review notes (plain pages, removed before submission) --------------

def _internal_notes_story(
    tender: dict[str, Any],
    checklist: dict[str, Any],
    drafting_note: str | None,
    s: _Styles,
) -> list[Any]:
    document_summary = tender.get("document_summary") or {}
    body = ParagraphStyle("ibody", parent=s.base["BodyText"])
    notes = drop_resolved_missing_information(checklist, tender)["notes"]
    sections: dict[str, list[dict[str, Any]]] = {k: [] for k in NOTE_SECTIONS}
    for item in notes:
        sections.setdefault(item.get("section"), []).append(item)

    def _bullet(item: dict[str, Any]) -> Paragraph:
        text = _rich(item.get("requirement") or "")
        if item.get("evidence"):
            text += f" <i>({_safe(item['evidence'])})</i>"
        return Paragraph(f"{'[x]' if item.get('done') else '•'} {text}", body)

    value_cell = ParagraphStyle("value_cell_int", parent=body, fontSize=9, leading=12)
    small = ParagraphStyle("small_int", parent=body, fontSize=8.5, textColor=colors.HexColor("#495057"))
    story: list[Any] = [
        Paragraph("INTERNAL REVIEW NOTES - remove these pages before submission", ParagraphStyle(
            "banner", parent=s.h1, textColor=colors.HexColor("#c92a2a"))),
        Paragraph(_safe(tender.get("title") or "(untitled tender)"), s.base["Heading3"]),
    ]
    if tender.get("source_url"):
        story.append(Paragraph(f"Source: {_safe(tender['source_url'])}", small))
    story.append(Spacer(1, 8))
    story.append(Paragraph(DISCLAIMER, ParagraphStyle("disclaimer", parent=body, **{
        **_PLACEHOLDER_BOX, "borderWidth": 0.6, "borderPadding": 6})))
    for note in (checklist.get("builder_note"), drafting_note):
        if note:
            story.append(Spacer(1, 10))
            story.append(Paragraph(f"<b>Note:</b> {_safe(note)}", ParagraphStyle(
                "draftnote", parent=body, textColor=colors.HexColor("#a16207"),
                borderColor=colors.HexColor("#facc15"), borderWidth=0.6, borderPadding=6,
                backColor=colors.HexColor("#fefce8"))))

    pending = [(n, i) for n, i in enumerate(checklist.get("items", []), 1)
               if i.get("status") in (STATUS_MISSING, STATUS_TO_PREPARE)]
    story.append(Paragraph("Checklist rows still needing attention", s.h2))
    for n, item in pending:
        story.append(Paragraph(
            f"• {n}. <b>{_safe(item.get('document') or '')}</b> - {_safe(item.get('status'))}"
            + (f" <i>({_safe(item['notes'])})</i>" if item.get("notes") else ""), body))
    if not pending:
        story.append(Paragraph("None - every row is enclosed or marked not applicable.", body))

    story.append(Paragraph("Open items before signing", s.h2))
    for item in sections["open_items"]:
        story.append(_bullet(item))
    if not sections["open_items"]:
        story.append(Paragraph("None recorded.", body))

    story.append(Paragraph("Tender Snapshot", s.h2))
    snapshot_rows = [
        ["Organisation", tender.get("organisation") or "Not disclosed"],
        ["Tender reference", tender.get("tender_ref") or "Not disclosed"],
        ["Published date", _fmt_date(tender.get("published_date"))],
        ["Closing date", _fmt_date(tender.get("closing_date"))],
        ["Opening date", _fmt_date(tender.get("opening_date")) if tender.get("opening_date") else (document_summary.get("tender_opening_date") or "Not disclosed")],
        ["Estimated bid amount", document_summary.get("estimated_bid_amount") or _fmt_amount(tender.get("tender_value"))],
        ["EMD", document_summary.get("emd_amount") or _fmt_amount(tender.get("earnest_money"))],
        ["Tender fee", document_summary.get("tender_fee_amount") or _fmt_amount(tender.get("document_fees"))],
    ]
    snapshot_table = Table([[label, Paragraph(_safe(value), value_cell)] for label, value in snapshot_rows],
                           colWidths=[4.5 * cm, 12.9 * cm])
    snapshot_table.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#c9d2db")),
    ]))
    story.append(snapshot_table)
    if document_summary.get("key_dates"):
        story.append(Paragraph("Key dates", s.h2))
        for d in document_summary["key_dates"]:
            story.append(Paragraph(f"• {_safe(d)}", body))

    if document_summary.get("technical_criteria_table"):
        story.append(Paragraph("Detailed technical criteria (marks-based)", s.h2))
        tc_rows = [["Ref", "Criterion", "Expected Evidence", "Marks"]]
        for row in document_summary["technical_criteria_table"]:
            tc_rows.append([
                _safe(row.get("ref") or "-"),
                Paragraph(_safe(row.get("criterion", "-")), small),
                Paragraph(_safe(row.get("expected_evidence") or "-"), small),
                _safe(row.get("marks") if row.get("marks") is not None else "-"),
            ])
        tc_table = Table(tc_rows, colWidths=[1.3 * cm, 7 * cm, 6 * cm, 2 * cm], repeatRows=1)
        tc_table.setStyle(TableStyle(_HEADER_STYLE + [("FONTSIZE", (0, 0), (-1, -1), 8.5)]))
        story.append(tc_table)

    if sections["missing_information"]:
        story.append(Paragraph("Missing information flagged in the tender documents", s.h2))
        for item in sections["missing_information"]:
            story.append(_bullet(item))
    return story


def generate_bid_package(
    tender: dict[str, Any],
    eligibility_criteria: list[dict[str, Any]],
    company_profile: dict[str, Any],
    company_documents: list[CompanyDocumentRef],
    drafted_documents: "dict[str, DraftedDocument] | None" = None,
    drafting_note: str | None = None,
    letterhead_image: str | Path | None = None,
    checklist: dict[str, Any] | None = None,
) -> bytes:
    """Builds the bid pack PDF from `checklist` (the tender's saved,
    possibly hand-edited submission checklist - built fresh via
    build_checklist's fallback when None) and returns it as bytes:

    1. The Master Bid Submission Checklist page, on the letterhead
       (`letterhead_image`, drawn full-page; plain when None).
    2. Every row's document, in S.No order (rows marked Not applicable are
       left out): a "draft" row's AI draft from `drafted_documents` (by row
       id - a templated covering letter or a to-be-completed page when
       there's none), a library document merged in, or a placeholder page
       when it can't be (see plan_rows). The letterhead and the
       signature/seal go wherever the row's letterhead/signature/stamp say.
    3. Internal review notes on plain pages, to be removed before submitting.
    """
    s = _styles()
    title = f"Bid Pack - {tender.get('title') or tender.get('tender_ref') or 'Tender'}"
    drafted_documents = drafted_documents or {}
    if not is_submission_checklist(checklist):
        checklist = build_checklist(tender, eligibility_criteria, company_profile, company_documents)
    checklist = refresh_statuses(checklist, company_documents, drafted_documents)
    plans = plan_rows(checklist, company_documents)

    letterhead_page = _letterhead_page(letterhead_image)
    sig_doc, seal_doc = _signing_assets(company_profile, company_documents)
    signature, seal = _read_bytes(sig_doc), _read_bytes(seal_doc)

    writer = PdfWriter()

    def _add_rendered(story: list[Any], on_letterhead: bool) -> None:
        data = _build_pdf(story, title, letterhead_image if on_letterhead and letterhead_page is not None else None)
        for page in PdfReader(io.BytesIO(data)).pages:
            writer.add_page(page)

    try:
        _add_rendered(_submission_checklist_story(checklist, s), True)
        for n, row in enumerate(checklist.get("items", []), 1):
            plan = plans[row["id"]]
            if plan.action == "skip":
                continue
            if plan.action == "draft":
                drafted = drafted_documents.get(row["id"])
                if drafted is not None:
                    story = _drafted_document_section(drafted, row, company_profile, company_documents, s)
                elif re.search(r"covering letter", row.get("document") or "", re.IGNORECASE):
                    story = _fallback_covering_letter_section(tender, row, company_profile, company_documents, s)
                else:
                    story = _undrafted_document_section(row, company_profile, company_documents, s)
                _add_rendered(story, bool(row.get("letterhead")))
            elif plan.action == "attach":
                _add_attached_pages(writer, plan.doc, row, letterhead_page, signature, seal)
            else:
                _add_rendered(_placeholder_section(n, row, plan.reason, s), False)
        _add_rendered(_internal_notes_story(tender, checklist, drafting_note, s), False)

        out = io.BytesIO()
        writer.write(out)
        return out.getvalue()
    finally:
        writer.close()
