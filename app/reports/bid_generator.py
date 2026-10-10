"""Builds a tender's Master Bid Submission Checklist and, from it, a single
downloadable "bid pack" PDF.

The checklist (see build_checklist) lists every document the tender asks
for - S.No / Document / What to Upload / Where / Notary / Status (/ Page in
the pack), like the team's own submission checklists, one row per distinct
document, grouped in CHECKLIST_SECTIONS order - with, per row, whether it's an existing
record to attach from the documents library (app.api.main's /documents -
see app.database.get_company_documents_collection) or a document the bidder must write,
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
import functools
import html
import io
import re
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable

from PIL import Image as PILImage
from PIL import ImageChops, ImageOps
from pypdf import PageObject, PdfReader, PdfWriter, Transformation
from reportlab import rl_config
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

# Image streams are written as plain binary rather than ASCII85 text - the
# encoder is pure Python without reportlab's optional C accelerator, and
# was several seconds of a bid pack's build against POST /generate-bid's
# 30-second limit. Binary streams are also smaller.
rl_config.useA85 = 0

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
    malformed-XML crash waiting to happen on real tender text. Text in a
    script the fonts can't draw at all (e.g. a Marathi phrase the drafter
    left untranslated) becomes one highlighted "[TO BE VERIFIED: ...]"
    note instead of a row of black boxes."""
    # U+FFFD: a character the tender's text extraction couldn't decode (a
    # curly quote, usually) - dropped rather than printed as a box.
    text = _NON_LATIN_RUN_RE.sub(UNTRANSLATED, str(text).replace("�", ""))
    return html.escape(text, quote=False).replace("₹", "Rs. ")


# Letters of scripts outside Latin (Greek onwards - Devanagari, Tamil,
# Cyrillic, CJK...) - the base-14 PDF fonts have no glyphs for them. A run
# spans the spaces, digits and punctuation between such words.
_NON_LATIN_LETTER = r"[Ͱ-῿Ⰰ-퟿豈-﫿]"
_NON_LATIN_RUN_RE = re.compile(
    rf"{_NON_LATIN_LETTER}+(?:[\s\d.,;:/()\-–'\"]+{_NON_LATIN_LETTER}+)*")
UNTRANSLATED = "[TO BE VERIFIED: text in a non-English script - translate it into English]"


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
    """One Documents-library file (app.database.get_company_documents_collection)
    - just enough to match against eligibility criteria and, for PDFs, merge
    into the pack."""

    id: str
    name: str
    filename: str
    content_type: str
    open_bytes: Any  # callable[[], bytes] - lazy, so unrelated docs are never read
    size: int | None = None  # bytes, from the record's `length`; None -> measured via open_bytes when needed
    # "letterhead" / "signature" / "stamp" when added as one from the Sign &
    # Stamp tab (app.api.main) - offered there whatever it's named.
    mark_kind: str | None = None


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
        _loose_name(signatory.get("signature_document_name")),
        _loose_name(signatory.get("seal_document_name")),
    } - {""}
    return _loose_name(doc.name) in names or doc.mark_kind in ("signature", "stamp", "letterhead")


def enclosure_documents(
    company_documents: list[CompanyDocumentRef], company_profile: dict[str, Any]
) -> list[CompanyDocumentRef]:
    return [
        d for d in company_documents
        if not is_reference_document(d, company_profile) and not _is_signing_asset(d, company_profile)
    ]


def _default_byte_budget() -> int:
    """Bytes of library PDFs one pack merges (app.config's
    enclosure_byte_budget) - the dashboard downloads packs straight from
    file storage (see app.api.main.bid_document_link), so it isn't bounded
    by one Lambda response."""
    from app.config import get_settings

    return get_settings().enclosure_byte_budget

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
    byte_budget: int | None = None,
) -> EnclosureSelection:
    """Which enclosure_documents() actually go into this tender's pack:
    the `standard_enclosures` named in company_profile.json always, then
    any document matching the tender's AI-extracted document_summary
    (documents_to_submit / eligibility requirements). A tender with no
    summary yet gets just the standard set. PDFs are added in that order
    until `byte_budget` is spent; the rest are reported, not merged."""
    byte_budget = _default_byte_budget() if byte_budget is None else byte_budget
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
    notary: bool = False,
    section: str = "",
    reference: str = "",
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
        # Must be notarised / sworn before submission. Absent on checklists
        # saved before this existed - always read with .get(), as False.
        "notary": notary,
        # One of CHECKLIST_SECTIONS - "" on older checklists (see row_section).
        "section": section,
        # The tender's own annexure / form number, else the clause the row
        # answers - the index's "Ref." column. "" on older checklists.
        "reference": reference,
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

# A document the tender wants notarised / sworn - the fallback when the AI
# plan doesn't say (or isn't available).
_NOTARY_RE = re.compile(r"notar|affidavit|stamp paper|non[- ]judicial|oath commissioner|sworn", re.IGNORECASE)


def needs_notary(*texts: str) -> bool:
    return bool(_NOTARY_RE.search(" ".join(t for t in texts if t)))


# A record someone else issues - the bidder can only attach it, never write
# it: drafting one gives a covering letter that claims a document is
# enclosed when it isn't (see is_issued_record).
_ISSUED_RECORD_RE = re.compile(
    r"empanel|lease|rent agreement|electricity bill|utility bill|address proof|office proof|"
    r"proof of (office|address)|\bproof\b|licen[cs]e\b|registration certificate|completion certificate|"
    r"performance certificate", re.IGNORECASE)


def is_issued_record(document: str) -> bool:
    """A row named for a record issued by someone else (an empanelment
    certificate, office address proof...) rather than a letter, form or
    annexure the bidder writes."""
    return bool(_ISSUED_RECORD_RE.search(document or "")) and not _BIDDER_WRITTEN_RE.search(document or "")


def records_not_drafted(checklist: dict[str, Any]) -> dict[str, Any]:
    """Turns auto-built "draft" rows that are really issued records (see
    is_issued_record) into "upload" rows - Missing until a library document
    is picked - so the pack shows the gap instead of a drafted letter.
    Rows the user added or edited on the dashboard stay as they are."""
    items = []
    for row in checklist.get("items", []):
        if (row.get("source") == "draft" and row.get("origin", "auto") == "auto"
                and is_issued_record(row.get("document") or "")):
            row = {**row, "source": "upload", "letterhead": False}
            if row.get("status") != STATUS_NOT_APPLICABLE:
                row["status"] = STATUS_ENCLOSED if row.get("document_id") else STATUS_MISSING
        items.append(row)
    return {**checklist, "items": items}


# The groups a pack is assembled in, in order, when the tender doesn't
# prescribe its own (see build_checklist). Each row carries one.
SECTION_COVERING = "Covering Letter / Bid Form"
SECTION_ANNEXURES = "Prescribed Annexures"
SECTION_LEGAL = "Legal & Statutory"
SECTION_FINANCIAL = "Financial"
SECTION_EXPERIENCE = "Experience"
SECTION_DECLARATIONS = "Declarations & Undertakings"
SECTION_TECHNICAL = "Technical Proposal"
SECTION_SIGNED_TENDER = "Signed Tender / Acceptance"
SECTION_EMD_FINANCIAL = "EMD / Financial Bid"
CHECKLIST_SECTIONS = (
    SECTION_COVERING, SECTION_ANNEXURES, SECTION_LEGAL, SECTION_FINANCIAL, SECTION_EXPERIENCE,
    SECTION_DECLARATIONS, SECTION_TECHNICAL, SECTION_SIGNED_TENDER, SECTION_EMD_FINANCIAL,
)

# Checked in this order - the first match wins, so the specific ("price
# bid", "covering letter", "non-blacklisting declaration") come before the
# broad ("certificate", "annexure").
_SECTION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (SECTION_EMD_FINANCIAL, re.compile(
        r"\bemd\b|earnest money|bid security|price bid|financial bid|commercial bid|\bboq\b|rate quot|"
        r"tender fee|document fee|processing fee", re.I)),
    (SECTION_COVERING, re.compile(
        r"covering letter|cover letter|bid (submission )?(form|letter)|letter of (bid|offer)", re.I)),
    (SECTION_SIGNED_TENDER, re.compile(
        r"signed tender|tender document|acceptance|terms (and|&) conditions|\bnit\b|corrigend", re.I)),
    (SECTION_DECLARATIONS, re.compile(
        r"declaration|undertaking|affidavit|blacklist|debar|self[- ]certif|power of attorney|authori[sz]ation|"
        r"no[- ]deviation|integrity pact|conflict of interest|non[- ]disclosure", re.I)),
    (SECTION_EXPERIENCE, re.compile(
        r"work order|purchase order|completion|performance certificate|experience|similar (work|project)|"
        r"client (certificate|testimonial)|list of (relevant |completed |similar )?projects|project data|"
        r"project (details|sheet)", re.I)),
    (SECTION_FINANCIAL, re.compile(
        r"turnover|audited|financial statement|balance sheet|profit (and|&) loss|\bitr\b|income tax return|"
        r"net ?worth|solvency|\bca certificate|chartered accountant", re.I)),
    (SECTION_LEGAL, re.compile(
        r"incorporation|\bpan\b|\btan\b|\bgst|udyam|msme|dpiit|start-?up|\bmca\b|master data|registration|"
        r"\biso\b|27001|cmmi|\bepf\b|\besi\b|labour licen|partnership deed|\bllp\b|\bmoa\b|memorandum|"
        r"cancelled cheque|bank (details|proof)", re.I)),
    (SECTION_TECHNICAL, re.compile(
        r"technical proposal|technical bid|methodology|approach|work plan|presentation|\bcv\b|"
        r"curriculum vitae|personnel|manpower|team|company profile|organi[sz]ation(al)? (profile|capability)|"
        r"concept|strategy", re.I)),
    (SECTION_ANNEXURES, re.compile(r"annex|appendix|format|\bform\b|schedule", re.I)),
]


def _known_section(value: str | None) -> str | None:
    """`value` as one of CHECKLIST_SECTIONS (the AI's label, tolerating case
    and punctuation drift), or None."""
    key = re.sub(r"[^a-z]+", "", (value or "").lower())
    if not key:
        return None
    # Exact first - "financial" is inside "emdfinancialbid" too.
    for section in CHECKLIST_SECTIONS:
        if key == re.sub(r"[^a-z]+", "", section.lower()):
            return section
    for section in CHECKLIST_SECTIONS:
        s_key = re.sub(r"[^a-z]+", "", section.lower())
        if key == s_key or key in s_key or s_key in key:
            return section
    return None


def _section_from_text(text: str) -> str | None:
    return next((section for section, pattern in _SECTION_PATTERNS if pattern.search(text or "")), None)


def infer_section(document: str, what_to_upload: str = "", source: str = "upload") -> str:
    """Which CHECKLIST_SECTIONS group a row belongs to, from its own wording
    - its name first ("Annexure 3 - Non-blacklisting Declaration" is a
    declaration, not just an annexure), then with its description."""
    return (_section_from_text(document) or _section_from_text(f"{document} {what_to_upload}")
            or (SECTION_ANNEXURES if source == "draft" else SECTION_LEGAL))


def row_section(row: dict[str, Any]) -> str:
    """A saved row's section - inferred for rows saved without one."""
    return _known_section(row.get("section")) or infer_section(
        row.get("document") or "", row.get("what_to_upload") or "", row.get("source") or "upload"
    )


def _is_pre_attested(name: str | None, company_profile: dict[str, Any] | None) -> bool:
    """A library scan that already carries the seal and signature (see
    company_profile.json's pre_attested_documents)."""
    if not name or not company_profile:
        return False
    names = {n.strip().lower() for n in company_profile.get("pre_attested_documents", [])}
    return name.strip().lower() in names


def default_marks(source: str, text: str, pre_attested: bool = False) -> tuple[bool, bool, bool]:
    """(letterhead, signature, stamp) a document normally needs: everything
    the bidder writes goes on the letterhead, signed and stamped; copies of
    its own records are self-attested (signed and stamped) - unless the
    scan is `pre_attested` already; third-party signed documents are
    attached untouched."""
    if source == "draft":
        return True, True, True
    if pre_attested or _THIRD_PARTY_SIGNED_RE.search(text):
        return False, False, False
    return False, True, True


def _resolve_library_document(
    name: str | None, candidates: list[CompanyDocumentRef], unused: list[CompanyDocumentRef] | None = None
) -> CompanyDocumentRef | None:
    """The library document the AI named - by exact name, else the single
    document among `unused` (default: all) whose name overlaps it, never a
    guess between several - so a loosely named row never falls onto a
    document another row already encloses."""
    if not name:
        return None
    doc = _find_document(candidates, name)
    if doc is not None:
        return doc
    matches = _match_documents(name, [], candidates if unused is None else unused)
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


def _library_row(doc: CompanyDocumentRef, company_profile: dict[str, Any], notes: str = "") -> dict[str, Any]:
    letterhead, signature, stamp = default_marks("upload", doc.name, _is_pre_attested(doc.name, company_profile))
    return submission_item(
        doc.name, f"Copy of {doc.name}", DEFAULT_WHERE, "upload", doc.id,
        letterhead, signature, stamp, STATUS_ENCLOSED, notes=notes,
        section=infer_section(doc.name, source="upload"),
    )


def _covering_letter_row() -> dict[str, Any]:
    return submission_item(
        COVERING_LETTER, "Covering letter submitting the offer and accepting the tender's terms",
        DEFAULT_WHERE, "draft", None, True, True, True, STATUS_TO_PREPARE, section=SECTION_COVERING,
    )


def _name_key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", (text or "").lower()).strip()


def _work_order_description(project: dict[str, Any]) -> str:
    """A work order row's "What to Upload", from the project's own verified
    data (company_profile.json's past_experience) - never the AI's wording."""
    date = project.get("work_order_date") or ""
    m = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", date)
    date = f"{m.group(3)}-{m.group(2)}-{m.group(1)}" if m else date
    ref = " ".join(filter(None, [project.get("work_order_no"), f"dated {date}" if date else ""]))
    by = project.get("issued_by") or project.get("routed_through")
    text = f"Copy of work order {ref}".strip() + (f" issued by {by}" if by else "")
    text += f" - {project.get('project') or project.get('client')}"
    if project.get("value_display"):
        text += f", {project['value_display']}"
    return text + "."


# "Annexure 5", "Annexure-V", "Form 3A", "Appendix B", "Schedule II" - the
# tender's own number for a prescribed format.
_ANNEXURE_LABEL_RE = re.compile(
    r"\b(annexure|annex|appendix|form|schedule|proforma|format)\s*[-–:.]?\s*(?:no\.?\s*)?"
    r"([0-9]{1,3}[a-z]?|[ivxlc]{1,6}|[a-z])\b(?:-[0-9a-z]{1,2}\b|\([0-9a-z]{1,2}\))?",
    re.IGNORECASE)


def annexure_label(*texts: str) -> str:
    """The first tender annexure / form number named in `texts` (e.g.
    "Annexure 5"), or ""."""
    for text in texts:
        for m in _ANNEXURE_LABEL_RE.finditer(text or ""):
            if len(m.group(2)) == 1 and m.group(2).islower():
                continue  # "schedule a meeting" - a lettered annexure is written "Appendix B"
            return " ".join(m.group(0).split()).strip(" -–:.")
    return ""


_LOWER_LETTER_LABEL_RE = re.compile(
    r"\b(Annexure|ANNEXURE|Annex|ANNEX|Appendix|APPENDIX|Form|FORM|Schedule|SCHEDULE|Proforma)(\s*[-–:.]?\s*)"
    r"([a-z])(?=\s*(?:[-–:(;,]|$)|\s+[A-Z])")


def normalise_labels(text: str) -> str:
    """"Appendix a - Covering Letter" (a tender whose text extraction lost
    its capitals) as "Appendix A - Covering Letter"."""
    return _LOWER_LETTER_LABEL_RE.sub(lambda m: f"{m.group(1)}{m.group(2)}{m.group(3).upper()}", text or "")


def row_reference(document: str, format_hint: str = "", basis: str = "") -> str:
    """The index's Ref. for a row - the tender's annexure / form number when
    it prescribes one, else the clause the row answers."""
    return annexure_label(document, format_hint) or " ".join((basis or "").split())[:60]


_FY_RE = re.compile(r"\b(20\d{2})\s*[-–/]\s*(\d{2}|20\d{2})\b")


def _fy_key(start: str, end: str) -> str:
    return f"{start}-{end[-2:]}"


def financial_years_on_record(company_profile: dict[str, Any]) -> set[str]:
    financials = company_profile.get("financials") or {}
    years = [t.get("financial_year") for t in financials.get("annual_turnover", [])]
    years += [b.get("financial_year") for b in financials.get("balance_sheet", [])]
    return {_fy_key(m.group(1), m.group(2)) for y in years if y for m in [_FY_RE.search(str(y))] if m}


def financial_years_not_on_record(text: str, company_profile: dict[str, Any]) -> list[str]:
    """Financial years `text` asks for (e.g. "FY 2022-23") that the
    company's verified financials don't cover - the generic CA certificate on
    file can't prove them."""
    on_record = financial_years_on_record(company_profile)
    if not on_record:
        return []
    asked = []
    for m in _FY_RE.finditer(text or ""):
        key = _fy_key(m.group(1), m.group(2))
        consecutive = (int(key[2:4]) + 1) % 100 == int(key[-2:])  # "2022-23", not a date range
        if consecutive and key not in on_record and key not in asked:
            asked.append(key)
    return asked


_TABLE_AMOUNT_RE = re.compile(
    r"^\s*(?:(?:rs\.?|inr|₹)\s*)?\d{1,3}(?:,\d{2,3})+(?:\.\d+)?\s*(?:/-)?\s*$|^\s*(?:rs\.?|inr|₹)\s*[\d,.]+\s*(?:/-)?\s*$",
    re.IGNORECASE)


def _digits(text: str) -> str:
    return re.sub(r"\D", "", text)


def _fy_in(text: str) -> str | None:
    for m in _FY_RE.finditer(text or ""):
        key = _fy_key(m.group(1), m.group(2))
        if (int(key[2:4]) + 1) % 100 == int(key[-2:]):
            return key
    return None


def guard_financial_years(drafted_documents: dict[str, Any], company_profile: dict[str, Any]) -> dict[str, Any]:
    """In every drafted turnover table, each year's figure must be the
    CA-certified figure for THAT year (company_profile.json's financials).
    A figure put under another year - e.g. a tender asking for FY 2022-23
    to 2024-25 filled with the certificate's 2023-24 to 2025-26 - or a year
    not on record becomes a highlighted placeholder plus an open item, and
    so does an average over a different set of years. Never invents the
    right figure; only stops a false one reaching the pack."""
    financials = company_profile.get("financials") or {}
    certified = {_fy_in(str(t.get("financial_year"))): t.get("amount_display") or ""
                 for t in financials.get("annual_turnover", []) if _fy_in(str(t.get("financial_year")))}
    if not certified:
        return drafted_documents
    average = financials.get("average_turnover") or {}
    # "FY 2023-24 to FY 2025-26" - every year from the first to the last.
    starts = [int(m.group(1)) for m in _FY_RE.finditer(str(average.get("period") or ""))]
    average_years = ({f"{y}-{(y + 1) % 100:02d}" for y in range(min(starts), max(starts) + 1)}
                     if starts else set())

    for doc in drafted_documents.values():
        for table in getattr(doc, "tables", None) or []:
            text = " ".join([table.title or "", *table.columns, *(c for r in table.rows for c in r)]).lower()
            if "turnover" not in text:
                continue
            column_fy = [_fy_in(c) for c in table.columns]
            column_avg = ["average" in c.lower() for c in table.columns]
            table_years: set[str] = {fy for fy in column_fy if fy}
            checks: list[tuple[list[str], int, str | None, bool]] = []
            for row in table.rows:
                labels = [c for c in row if not _TABLE_AMOUNT_RE.match(c)]
                row_fy = next((fy for fy in map(_fy_in, labels) if fy), None)
                row_avg = any("average" in c.lower() for c in labels)
                table_years |= {row_fy} if row_fy else set()
                for i, cell in enumerate(row):
                    if _TABLE_AMOUNT_RE.match(cell) and not _PLACEHOLDER_RE.search(cell):
                        fy = column_fy[i] if i < len(column_fy) and column_fy[i] else row_fy
                        avg = (i < len(column_avg) and column_avg[i]) or row_avg
                        checks.append((row, i, fy, avg))
            for row, i, fy, avg in checks:
                cell = row[i]
                if avg:
                    if not (table_years and average_years and table_years != average_years):
                        continue
                    row[i] = (f"[TO BE VERIFIED: average turnover over FY {', '.join(sorted(table_years))} - "
                              f"the CA-certified average is for {average.get('period')}]")
                elif fy and _digits(cell) != _digits(certified.get(fy, "")):
                    row[i] = (f"[TO BE VERIFIED: FY {fy} turnover - the CA-certified figure is {certified[fy]}]"
                              if fy in certified else
                              f"[TO BE VERIFIED: FY {fy} turnover - not in the CA-certified figures on file]")
                else:
                    continue
                if hasattr(doc, "open_items"):
                    doc.open_items.append(f"{row[i][1:-1]} (was \"{cell.strip()}\")")
    return drafted_documents


# A certificate a Chartered Accountant (or a bank) issues and signs - when the
# tender prescribes its format, the bidder can only prepare it for them.
_CA_CERTIFICATE_RE = re.compile(
    r"\bca\b|chartered accountant|turnover certificate|net ?worth certificate|solvency|auditor", re.IGNORECASE)


def _rows_from_plan(
    plan: "SubmissionChecklistPlan", candidates: list[CompanyDocumentRef], company_profile: dict[str, Any]
) -> list[dict[str, Any]]:
    """One row per distinct document: an AI row that resolves to a library
    document an earlier row already encloses - or repeats an earlier row's
    name - is merged into that row (its tender clause added to the row's
    basis) rather than listed, and attached, a second time."""
    projects = {
        (p.get("library_document") or "").strip().lower(): p
        for p in company_profile.get("past_experience", []) if p.get("library_document")
    }
    rows: list[dict[str, Any]] = []
    bases: dict[str, list[str]] = {}  # row id -> tender clauses it answers
    by_doc: dict[str, dict[str, Any]] = {}
    by_name: dict[tuple[str, str], dict[str, Any]] = {}
    for r in plan.rows:
        r.document, r.basis, r.format_hint = (normalise_labels(r.document), normalise_labels(r.basis),
                                              normalise_labels(r.format_hint))
        if r.document[:1].islower():
            r.document = r.document[0].upper() + r.document[1:]  # "boq" as the tender's lower-cased text gave it
        if r.source == "draft" and is_issued_record(r.document):
            r.source = "upload"  # see records_not_drafted
        doc = None
        if r.source == "upload":
            unused = [c for c in candidates if c.id not in by_doc]
            doc = _resolve_library_document(r.library_document, candidates, unused)
        basis, notes = r.basis.strip(), r.notes.strip()
        notary = r.notary or needs_notary(r.document, r.what_to_upload, r.format_hint, r.notes)
        earlier = (by_doc.get(doc.id) if doc else None) or by_name.get((_name_key(r.document), r.source))
        if earlier is not None:
            if basis and basis not in bases[earlier["id"]]:
                bases[earlier["id"]].append(basis)
            if notes and notes not in earlier["notes"]:
                earlier["notes"] = f"{earlier['notes']} {notes}".strip()
            earlier["notary"] = earlier["notary"] or notary
            continue

        name = r.document.strip()
        project = projects.get((doc.name if doc else "").strip().lower())
        if project and project.get("short_name") and (
            project["short_name"].lower() not in name.lower() or name.lower() == doc.name.lower()
        ):
            # A work order row is named for its project, never generically.
            name = f"Work Order - {project['short_name']}" + (
                f" ({project['scope_label']})" if project.get("scope_label") else "")
        what_to_upload = _work_order_description(project) if project else r.what_to_upload.strip()
        # An attached document's own name places it better than the AI's
        # label (an ISO certificate is a registration, whatever it proves).
        section = (_section_from_text(doc.name) if doc else _section_from_text(r.document)) or _known_section(
            r.section) or infer_section(r.document, r.what_to_upload, r.source)

        # "If applicable" means "if the bidder has it": a document the bidder
        # holds, or an exemption its MSME / start-up registration earns, is
        # never marked Not applicable.
        applicable = r.applicable or doc is not None or (
            _EXEMPTION_RE.search(r.document) is not None and _holds_exemption_registration(company_profile))

        pre_attested = _is_pre_attested(doc.name if doc else r.library_document, company_profile)
        usual = default_marks(r.source, f"{r.document} {r.what_to_upload}", pre_attested)
        # A scan that already carries the seal and signature never gets a
        # second set, whatever the plan says; the dashboard can still tick it.
        given = (None, None, None) if pre_attested else (r.letterhead, r.signature, r.stamp)
        letterhead, signature, stamp = (g if g is not None else d for g, d in zip(given, usual))

        row_text = f"{r.document} {r.what_to_upload} {r.format_hint} {r.notes}"
        ca_format = r.source == "draft" and bool(_CA_CERTIFICATE_RE.search(f"{r.document} {r.what_to_upload}"))
        if ca_format:
            # The tender's own format for a CA's certificate: prepared here,
            # filled from the verified financials, but issued by the CA.
            letterhead, signature, stamp = False, False, False
            notes = (f"{notes} Prepared in the tender's format for the Chartered Accountant to issue on their "
                     "letterhead with UDIN, sign and seal - the bidder does not sign it.").strip()
        if ca_format or section == SECTION_FINANCIAL:
            gaps = financial_years_not_on_record(row_text, company_profile)
            if gaps:
                notes = (f"{notes} Tender asks for FY {', '.join(gaps)}, which the CA-certified figures on file "
                         "do not cover - get a fresh CA certificate for exactly the years asked.").strip()
        row = submission_item(
            name, what_to_upload, r.where.strip(), r.source,
            doc.id if doc else None, letterhead, signature, stamp,
            _row_status(r.source, doc) if applicable else STATUS_NOT_APPLICABLE, r.format_hint.strip(), notes,
            notary=notary and applicable, section=section,
            # An attached record's own name ("GST REG-06 + Annex A") is no
            # tender annexure number - only a document the bidder fills is.
            reference=row_reference(r.document if r.source == "draft" else "", r.format_hint, basis),
        )
        rows.append(row)
        bases[row["id"]] = [basis] if basis else []
        by_name[(_name_key(r.document), r.source)] = row
        if doc is not None:
            by_doc[doc.id] = row

    # The tender clause(s) a row answers lead its notes, so the team can see
    # why it's there - one row may prove several conditions.
    for row in rows:
        if bases[row["id"]]:
            row["notes"] = f"Tender: {'; '.join(bases[row['id']])}. {row['notes']}".strip()
    rows = _collapse_repeated_formats(rows)
    _exempt_payment_rows(rows)
    return rows


_INSTANCE_SUFFIX_RE = re.compile(r"\s*\([^()]*\)\s*$")


def _collapse_repeated_formats(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One prescribed format listed both bare and per instance ("Appendix D
    - Project Data Sheet" plus "... (MP CAEC)", "... (Dumka)") keeps only
    the instances; listed bare and once more with a qualifier ("Appendix H"
    and "Appendix H (Resource List)") it is one row. Notes are merged."""
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        label = annexure_label(row["document"]) if row["source"] == "draft" else ""
        if label:
            groups.setdefault(_loose_name(label), []).append(row)
    dropped: set[str] = set()
    for group in groups.values():
        bare = [r for r in group if not _INSTANCE_SUFFIX_RE.search(r["document"])]
        instances = [r for r in group if _INSTANCE_SUFFIX_RE.search(r["document"])]
        if not bare or not instances:
            continue
        keep, drop = (instances, bare) if len(instances) >= 2 else (bare[:1], bare[1:] + instances)
        for row in drop:
            for note in [n for n in [row["notes"]] if n and n not in keep[0]["notes"]]:
                keep[0]["notes"] = f"{keep[0]['notes']} {note}".strip()
            dropped.add(row["id"])
    return [r for r in rows if r["id"] not in dropped]


_EXEMPTION_RE = re.compile(r"exempt", re.IGNORECASE)
_EMD_RE = re.compile(r"\bemd\b|earnest money|bid security", re.IGNORECASE)
_FEE_RE = re.compile(r"tender fee|document fee|cost of tender|processing fee|bid fee", re.IGNORECASE)
_PAYMENT_PROOF_RE = re.compile(r"proof|payment|receipt|paid|remittance|instrument|\bdd\b|demand draft", re.IGNORECASE)


def _holds_exemption_registration(company_profile: dict[str, Any]) -> bool:
    return any(re.search(r"udyam|msme|dpiit", reg.get("name", ""), re.IGNORECASE)
               for reg in company_profile.get("registrations", []))


def _exempt_payment_rows(rows: list[dict[str, Any]]) -> None:
    """When an EMD (or tender fee) exemption request is in the bid, the
    payment proof for that same charge isn't needed - it's marked Not
    applicable, pointing at the exemption, instead of showing as Missing."""
    for kind in (_EMD_RE, _FEE_RE):
        exemption = next((r["document"] for r in rows if r["status"] != STATUS_NOT_APPLICABLE
                          and _EXEMPTION_RE.search(r["document"]) and kind.search(r["document"])), None)
        if exemption is None:
            continue
        for r in rows:
            if (r["source"] == "upload" and not r["document_id"] and kind.search(r["document"])
                    and _PAYMENT_PROOF_RE.search(f"{r['document']} {r['what_to_upload']}")
                    and not _EXEMPTION_RE.search(r["document"])):
                r["status"] = STATUS_NOT_APPLICABLE
                r["notes"] = f"Exempt as MSME - see \"{exemption}\". {r['notes']}".strip()


def _rows_from_summary(
    tender: dict[str, Any], selection: EnclosureSelection, company_profile: dict[str, Any]
) -> list[dict[str, Any]]:
    """The deterministic fallback: one row per document the summary says to
    submit (matched to a selected library document where one fits), plus a
    row per remaining selected library document."""
    summary = tender.get("document_summary") or {}
    to_submit = [t for t in summary.get("documents_to_submit", []) if t.strip()]
    rows: list[dict[str, Any]] = []
    if not any(re.search(r"covering letter|bid (submission )?(form|letter)", t, re.IGNORECASE) for t in to_submit):
        rows.append(_covering_letter_row())

    used: set[str] = set()
    names: set[str] = set()
    for text in to_submit:
        if _name_key(text) in names:
            continue
        names.add(_name_key(text))
        source = "draft" if _BIDDER_WRITTEN_RE.search(text) else "upload"
        doc = None
        if source == "upload":
            doc = next((d for d in _match_documents(text, [], selection.selected) if d.id not in used), None)
            if doc is not None:
                used.add(doc.id)
        letterhead, signature, stamp = default_marks(
            source, text, _is_pre_attested(doc.name if doc else None, company_profile))
        rows.append(submission_item(
            text.strip()[:80], text.strip(), "Financial Bid" if _FINANCIAL_RE.search(text) else DEFAULT_WHERE,
            source, doc.id if doc else None, letterhead, signature, stamp, _row_status(source, doc),
            notary=needs_notary(text), section=infer_section(text, source=source),
            reference=annexure_label(text),
        ))
    rows += [_library_row(d, company_profile) for d in selection.selected if d.id not in used]
    return rows


def _order_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Rows grouped in CHECKLIST_SECTIONS order (stable - the plan's own
    order within a group). Within a group the forms the bidder fills (an
    experience or turnover annexure, the experience summary) come first and
    the records proving them right after, so an evaluator reads each claim
    followed by its proof."""
    def _key(row: dict[str, Any]) -> tuple[int, int]:
        section = row_section(row)
        text = f"{row.get('document') or ''}".lower()
        if row.get("source") == "draft" or (
                section == SECTION_EXPERIENCE and re.search(r"summary|list of|statement|details of", text)):
            within = 0
        elif section == SECTION_EXPERIENCE and re.search(r"completion|performance|client certificate|testimonial",
                                                          text):
            within = 2
        else:
            within = 1
        return CHECKLIST_SECTIONS.index(section), within

    return sorted(rows, key=_key)


def _insert_in_section(rows: list[dict[str, Any]], row: dict[str, Any]) -> None:
    """Puts `row` after the last row of its own section - or, when there's
    none, before the first row of a later section - keeping a tender's own
    order intact everywhere else."""
    order = CHECKLIST_SECTIONS.index(row_section(row))
    same = [i for i, r in enumerate(rows) if row_section(r) == row_section(row)]
    if same:
        rows.insert(same[-1] + 1, row)
        return
    later = next((i for i, r in enumerate(rows) if CHECKLIST_SECTIONS.index(row_section(r)) > order), len(rows))
    rows.insert(later, row)


_WORK_ORDER_RE = re.compile(r"work order|purchase order|letter of award|\bloa\b", re.IGNORECASE)


def _add_experience_summary(rows: list[dict[str, Any]]) -> None:
    """Evaluators read several work orders through a summary sheet - one is
    added ahead of them when the rows enclose two or more and no row
    already summarises them."""
    work_orders = [i for i, r in enumerate(rows) if r["source"] == "upload" and _WORK_ORDER_RE.search(r["document"])]
    summarised = any(
        r["source"] == "draft" and row_section(r) == SECTION_EXPERIENCE for r in rows
    )
    if len(work_orders) < 2 or summarised:
        return
    rows.insert(work_orders[0], submission_item(
        "Project Experience Summary",
        "Summary sheet of the enclosed work orders - client, work order no. & date, relevant scope, value and status",
        DEFAULT_WHERE, "draft", None, True, True, True, STATUS_TO_PREPARE,
        notes="Summarises the work orders enclosed below.", section=SECTION_EXPERIENCE,
    ))


ELIGIBILITY_COMPLIANCE = "Eligibility Compliance Statement"
_COMPLIANCE_ROW_RE = re.compile(
    r"compliance (statement|sheet|matrix|table)|eligibility (criteria|compliance|statement|details)|"
    r"pre[- ]?qualification (criteria|compliance|details)|qualification criteria", re.IGNORECASE)
# A tender with at least this many stated conditions gets a point-by-point
# compliance statement when it prescribes none.
_MIN_CONDITIONS_FOR_STATEMENT = 3


def _add_compliance_statement(rows: list[dict[str, Any]], conditions: list[dict[str, Any]]) -> None:
    """Evaluators check eligibility condition by condition - a statement
    answering each one (Complied / how / which document, on which page) is
    added after the covering letter when the tender states several
    conditions and no row already answers them."""
    if len(conditions) < _MIN_CONDITIONS_FOR_STATEMENT:
        return
    if any(_COMPLIANCE_ROW_RE.search(f"{r['document']} {r['what_to_upload']}") for r in rows):
        return
    _insert_in_section(rows, submission_item(
        ELIGIBILITY_COMPLIANCE,
        "Point-by-point statement of compliance with every eligibility / qualification condition - "
        "compliance, how the bidder qualifies, and the supporting document with its page number",
        DEFAULT_WHERE, "draft", None, True, True, True, STATUS_TO_PREPARE,
        notes="Answers the tender's eligibility conditions one by one, pointing at the enclosed proof.",
        section=SECTION_COVERING,
    ))


# The tender asks the bidder to present - not "presentation of documents".
_PRESENTATION_ASKED_RE = re.compile(
    r"technical presentation|power ?point presentation|\bppt\b|slide deck|"
    r"presentation (before|to) (the )?([a-z]+ ){0,3}(committee|board|panel|authority|department)|"
    r"(make|give|deliver|conduct)s? (a |the )?(brief |short |technical )?presentation|"
    r"presentation (on|of|covering) (the )?(proposed |technical )?(approach|methodology|work ?plan|concept|"
    r"strategy|proposal|solution|understanding|plan)",
    re.IGNORECASE,
)
_ASKS_FOR_PRESENTATION_RE = re.compile(r"\bpresentation\b|\bppt\b|power ?point|slide deck|\bslides\b", re.IGNORECASE)


def _add_presentation(rows: list[dict[str, Any]], tender: dict[str, Any]) -> None:
    """A tender whose documents ask the bidder to present gets a presentation
    row (drafted into a slide deck, see app.reports.presentation) - when the
    planner left it out and no row already is one."""
    text = tender.get("document_text") or ""
    match = _PRESENTATION_ASKED_RE.search(text)
    if not match:
        return
    if any(_ASKS_FOR_PRESENTATION_RE.search(r.get("document") or "") for r in rows):
        return
    # The sentence that asks for it, for the drafter and the reviewer.
    start = max(text.rfind(".", 0, match.start()), text.rfind("\n", 0, match.start())) + 1
    end_dot = text.find(".", match.end())
    sentence = re.sub(r"\s+", " ", text[start:end_dot + 1 if end_dot != -1 else match.end() + 200]).strip()[:400]
    _insert_in_section(rows, submission_item(
        "Technical Presentation",
        "Presentation on the bidder's understanding, approach, methodology and relevant experience, as the tender asks",
        "Technical Presentation", "draft", None, False, False, False, STATUS_TO_PREPARE,
        format_text=sentence,
        notes="Added because the tender's documents ask for a presentation.",
        section=SECTION_TECHNICAL,
    ))


# Below this much text the tender's own documents weren't really read (a
# link, a title) - the checklist then leans on the standard bid set.
_THIN_TENDER_TEXT_CHARS = 1500


def _thin_tender_text(tender: dict[str, Any]) -> bool:
    return len((tender.get("document_text") or "").strip()) < _THIN_TENDER_TEXT_CHARS


def _add_company_profile(rows: list[dict[str, Any]]) -> None:
    """With no tender text to say otherwise, a bid carries a company profile
    - added when no row is one already."""
    if any(re.search(r"company profile|organi[sz]ation(al)? profile", r["document"], re.I) for r in rows):
        return
    _insert_in_section(rows, submission_item(
        "Company Profile",
        "Company profile - legal identity, registrations, certifications, turnover, services and relevant projects",
        DEFAULT_WHERE, "draft", None, True, True, True, STATUS_TO_PREPARE,
        notes="Standard bid document - the tender's own documents could not be read; check them for more.",
        section=SECTION_TECHNICAL,
    ))


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
    _rows_from_summary; `builder_note` says why). One row per distinct
    document; the company profile's `standard_enclosures` are always listed,
    in the Legal & Statutory group, and so is the covering letter when
    nothing else covers it. Rows are grouped in CHECKLIST_SECTIONS order
    unless the tender prescribes its own (the plan's `tender_order`)."""
    candidates = enclosure_documents(company_documents, company_profile)
    tender_order = False
    conditions: list[dict[str, Any]] = []
    if plan is not None and plan.rows:
        rows = _rows_from_plan(plan, candidates, company_profile)
        tender_order = plan.tender_order
        conditions = [{"ref": c.ref.strip(), "proved_by": c.proved_by.strip()}
                      for c in plan.conditions if c.ref.strip()]
    else:
        selection = select_enclosures(tender, company_documents, company_profile)
        rows = _rows_from_summary(tender, selection, company_profile)

    _add_experience_summary(rows)
    if not _thin_tender_text(tender):
        _add_compliance_statement(rows, conditions)
        _add_presentation(rows, tender)
    if _thin_tender_text(tender):
        _add_company_profile(rows)
    referenced = {r["document_id"] for r in rows if r["document_id"]}
    standard_names = [n.strip().lower() for n in company_profile.get("standard_enclosures", [])]
    standard = sorted(
        (d for d in candidates if d.name.strip().lower() in standard_names and d.id not in referenced),
        key=lambda d: standard_names.index(d.name.strip().lower()),
    )
    for d in standard:
        row = _library_row(d, company_profile, notes="Standard enclosure (company profile)")
        row["section"] = SECTION_LEGAL
        _insert_in_section(rows, row)
    if not tender_order:
        rows = _order_rows(rows)

    if _thin_tender_text(tender):
        warning = ("The tender's own documents could not be read (no usable text was extracted) - this checklist "
                   "is the standard bid set built from the tender's title and the usual eligibility conditions. "
                   "Download the tender documents from the tender page, check them for annexures, formats, EMD "
                   "and eligibility conditions, and rebuild the checklist.")
        builder_note = f"{builder_note} {warning}" if builder_note else warning

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
        "tender_order": tender_order,
        # The tender's eligibility conditions and the rows proving them (the
        # planner's worklist) - what a compliance statement answers.
        "conditions": conditions,
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


_CV_RE = re.compile(r"curriculum vitae|\bcvs?\b|\bresume\b|bio-?data", re.IGNORECASE)
_CV_TITLE_NOISE_RE = re.compile(
    r"curriculum vitae|\bcvs?\b|\bresume\b|bio-?data|\(cv\)|for (the )?proposed( key)?( professional)?( staff)?|"
    r"of (the )?proposed|format|^[\s:–-]+|[\s:–-]+$", re.IGNORECASE)


def collapse_unfilled_cvs(
    drafted_documents: "dict[str, DraftedDocument]",
    rows: list[dict[str, Any]],
    company_profile: dict[str, Any],
) -> "dict[str, DraftedDocument]":
    """A drafted CV that names nobody in company_profile.json's
    key_personnel is a form of blanks (date of birth, nationality, every
    employment...) - it's cut to one placeholder saying which CV to supply
    and what the tender asks of that person, with one open item, so the
    pack shows the gap once instead of twenty times. CVs of listed people
    are left as drafted."""
    names = [k["name"].lower() for k in company_profile.get("key_personnel", []) if k.get("name")]
    by_id = {r["id"]: r for r in rows}
    for row_id, doc in drafted_documents.items():
        row = by_id.get(row_id) or {}
        heading = doc.title or row.get("document") or ""
        if not (_CV_RE.search(heading) or _CV_RE.search(row.get("document") or "")):
            continue
        text = " ".join(doc.body_paragraphs + doc.closing_paragraphs
                        + [c for t in doc.tables for r in t.rows for c in r] + [c for t in doc.tables for c in t.columns])
        if any(n in text.lower() for n in names) or len(_PLACEHOLDER_RE.findall(text)) < 2:
            continue
        role = " ".join(_CV_TITLE_NOISE_RE.sub(" ", heading).split()).strip(" -–:") or "key personnel"
        need = (row.get("what_to_upload") or "").strip().rstrip(".")
        gap = f"CV of the proposed {role} in the tender's prescribed format" + (f" - {need}" if need else "")
        doc.body_paragraphs = [f"[TO BE FILLED FROM COMPANY RECORDS: {gap}]",
                               "Attach the person's signed CV and their degree and experience certificates."]
        doc.tables, doc.closing_paragraphs = [], []
        doc.open_items = [f"[TO BE FILLED FROM COMPANY RECORDS: {gap}]"]
    return drafted_documents


# --- What the pack does with each row ------------------------------------------

@dataclass
class RowPlan:
    """How generate_bid_package renders one checklist row."""

    action: str  # "draft" | "attach" | "placeholder" | "duplicate" | "skip"
    doc: CompanyDocumentRef | None = None
    reason: str = ""  # why a placeholder stands in
    same_as: int | None = None  # "duplicate": the S.No. whose row already encloses this document


def _is_image(doc: CompanyDocumentRef) -> bool:
    return doc.content_type.startswith("image/") or doc.filename.lower().endswith((".png", ".jpg", ".jpeg"))


def plan_rows(
    checklist: dict[str, Any],
    company_documents: list[CompanyDocumentRef],
    byte_budget: int | None = None,
) -> dict[str, RowPlan]:
    """Per row id: draft it, attach its library document, or stand in a
    placeholder page (document missing, not a PDF/image, or over the pack's
    size budget). The budget goes to the smallest documents first, so as
    many as possible are enclosed and only the largest are left to attach
    separately - the pack itself stays in S.No order. A document already
    attached for an earlier row (a hand-edited checklist pointing two rows
    at one file) is a "duplicate": never merged, or charged against the
    budget, a second time."""
    byte_budget = _default_byte_budget() if byte_budget is None else byte_budget
    by_id = {d.id: d for d in company_documents}
    items = checklist.get("items", [])
    mergeable = {
        d.id: d for d in (by_id.get(r.get("document_id") or "") for r in items
                          if r.get("status") != STATUS_NOT_APPLICABLE and r.get("source") != "draft")
        if d is not None and (_is_pdf(d) or _is_image(d))
    }
    fits: set[str] = set()
    used = 0
    for doc in sorted(mergeable.values(), key=_doc_size):
        if used + _doc_size(doc) <= byte_budget:
            used += _doc_size(doc)
            fits.add(doc.id)

    plans: dict[str, RowPlan] = {}
    first_row: dict[str, int] = {}  # document id -> S.No. that encloses it
    for n, row in enumerate(items, 1):
        if row.get("status") == STATUS_NOT_APPLICABLE:
            plans[row["id"]] = RowPlan("skip")
            continue
        if row.get("source") == "draft":
            plans[row["id"]] = RowPlan("draft")
            continue
        doc = by_id.get(row.get("document_id") or "")
        if doc is not None and doc.id in first_row:
            plans[row["id"]] = RowPlan("duplicate", doc, f"Enclosed at S.No. {first_row[doc.id]}",
                                       same_as=first_row[doc.id])
        elif doc is None:
            plans[row["id"]] = RowPlan("placeholder", reason="Not in the Documents library yet - upload it and "
                                                              "pick it on the checklist, or attach it on the portal.")
        elif not (_is_pdf(doc) or _is_image(doc)):
            plans[row["id"]] = RowPlan("placeholder", doc, f"{doc.filename} is not a PDF or image, so it can't be "
                                                           "merged into this pack - attach it separately.")
        elif doc.id not in fits:
            plans[row["id"]] = RowPlan("placeholder", doc, f"{doc.filename} would push the pack past its size "
                                                           "limit - attach it separately from the Documents library.")
        else:
            first_row[doc.id] = n
            plans[row["id"]] = RowPlan("attach", doc)
    return plans


def refresh_statuses(
    checklist: dict[str, Any],
    company_documents: list[CompanyDocumentRef],
    drafted_ids: Iterable[str] = (),
) -> dict[str, Any]:
    """Each row's Status as the generated pack actually has it: Enclosed once
    drafted/attached (or already attached for an earlier row), To be
    prepared for a document still to write, Missing for one not attached.
    Not applicable is the user's call and stays."""
    drafted_ids = set(drafted_ids)
    plans = plan_rows(checklist, company_documents)
    items = []
    for row in checklist.get("items", []):
        plan = plans[row["id"]]
        if plan.action == "draft":
            status = STATUS_ENCLOSED if row["id"] in drafted_ids else STATUS_TO_PREPARE
        elif plan.action in ("attach", "duplicate"):
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


def _is_docx(doc: CompanyDocumentRef) -> bool:
    return doc.filename.lower().endswith(".docx") or "wordprocessingml" in doc.content_type


def _docx_text(data: bytes) -> str:
    """A Word document's paragraphs as plain lines (no python-docx needed:
    the text lives in word/document.xml's <w:t> runs)."""
    import zipfile

    xml = zipfile.ZipFile(io.BytesIO(data)).read("word/document.xml").decode("utf-8")
    paragraphs = re.findall(r"<w:p[ >].*?</w:p>", xml, re.S)
    return "\n".join(html.unescape("".join(re.findall(r"<w:t(?: [^>]*)?>(.*?)</w:t>", p, re.S))) for p in paragraphs)


def company_background_text(company_documents: list[CompanyDocumentRef], company_profile: dict[str, Any]) -> str:
    """Plain text of every PDF or Word (.docx) reference document (see
    is_reference_document), de-duplicated line by line - handed to
    app.intelligence.bid_drafter as descriptive company background. Empty
    string when there's nothing readable; never raises, since a bid pack
    must still generate without it."""
    seen: set[str] = set()
    lines: list[str] = []
    for doc in company_documents:
        if not ((_is_pdf(doc) or _is_docx(doc)) and is_reference_document(doc, company_profile)):
            continue
        try:
            text = _docx_text(doc.open_bytes()) if _is_docx(doc) else "\n".join(
                page.extract_text() or "" for page in PdfReader(io.BytesIO(doc.open_bytes())).pages)
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


def signing_assets(
    company_profile: dict[str, Any], company_documents: list[CompanyDocumentRef]
) -> tuple[CompanyDocumentRef | None, CompanyDocumentRef | None]:
    """The signature and seal images named in company_profile.json's
    authorized_signatory, when they're in the library."""
    signatory = company_profile.get("authorized_signatory", {})
    return (
        _find_signing_asset(company_documents, signatory.get("signature_document_name"), "signature",
                            signatory.get("name") or ""),
        _find_signing_asset(company_documents, signatory.get("seal_document_name"), "stamp",
                            company_profile.get("trade_name") or company_profile.get("legal_name") or ""),
    )


def _loose_name(text: str | None) -> str:
    return re.sub(r"[^a-z0-9]+", "", (text or "").lower())


def _find_signing_asset(
    documents: list[CompanyDocumentRef], name: str | None, kind: str, owner: str
) -> CompanyDocumentRef | None:
    """The configured signature / seal image - by name, ignoring case,
    spaces and underscores ("Oaks Stamp" is "Oaks_stamp"); else the one
    image added as that kind on the Sign & Stamp tab whose name carries the
    owner's first name (the signatory's for a signature, the company's for
    a seal) - so a renamed upload never leaves every page unsigned."""
    if name:
        exact = _find_document(documents, name) or next(
            (d for d in documents if _loose_name(d.name) == _loose_name(name)), None)
        if exact is not None:
            return exact
    first = _loose_name((owner.split() or [""])[0])
    matches = [d for d in documents if d.mark_kind == kind and first and first in _loose_name(d.name)]
    return matches[0] if len(matches) == 1 else None


def _closing_lines(flow: list[Any]) -> list[Any]:
    """Pops the document's last paragraph ("Yours faithfully,") - and the
    one before it when that's short - off `flow`, to go with the
    signature block."""
    tail: list[Any] = []
    paragraphs = 0
    while flow and isinstance(flow[-1], (Paragraph, Spacer)):
        item = flow[-1]
        if isinstance(item, Paragraph):
            if paragraphs == 2 or (paragraphs == 1 and len(item.getPlainText()) > 220):
                break
            paragraphs += 1
        tail.insert(0, flow.pop())
    return tail


def _signature_block(
    company_profile: dict[str, Any],
    company_documents: list[CompanyDocumentRef],
    body: ParagraphStyle,
    signature: bool = True,
    stamp: bool = True,
    lead: list[Any] | None = None,
) -> list[Any]:
    """"For <company>", the signature and/or seal image (each only when the
    row asks for it and the image is in the library) and the printed
    name/designation/place - kept together on one page, with `lead` (the
    document's closing lines, see _closing_lines) so a signature never
    sits alone on a page of its own."""
    signatory = company_profile.get("authorized_signatory", {})
    flow: list[Any] = list(lead or []) + [
        Spacer(1, 6),
        Paragraph(f"For <b>{_safe(company_profile.get('legal_name', '[Company]'))}</b>", body),
    ]
    sig_doc, seal_doc = signing_assets(company_profile, company_documents)
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
    lines.append(f"Date: {dt.date.today().strftime('%d-%m-%Y')}")
    if stamp and seal_img is None:
        lines.append("Company Seal:")  # no seal image on file - affixed by hand
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


PageRange = tuple[int, int]  # first and last page of a row's document in the pack


def fmt_pages(pages: PageRange | None) -> str:
    if not pages:
        return ""
    first, last = pages
    return str(first) if first == last else f"{first}-{last}"


def row_pages(
    n: int, items: list[dict[str, Any]], plans: dict[str, "RowPlan"], page_ranges: dict[str, PageRange] | None
) -> PageRange | None:
    """Where S.No. `n`'s document sits in the pack - a duplicate row points
    at the row that encloses its document."""
    if not page_ranges or not 1 <= n <= len(items):
        return None
    plan = plans.get(items[n - 1]["id"])
    if plan and plan.action == "duplicate" and plan.same_as:
        return page_ranges.get(items[plan.same_as - 1]["id"])
    return page_ranges.get(items[n - 1]["id"])


def _submission_checklist_story(
    checklist: dict[str, Any],
    s: _Styles,
    plans: dict[str, "RowPlan"] | None = None,
    page_ranges: dict[str, PageRange] | None = None,
) -> list[Any]:
    """The checklist page - also the pack's index: each row's annexure /
    clause reference, Notary requirement, Status and, once the pack is laid
    out (`page_ranges`), the pages its document occupies."""
    header = checklist.get("header") or {}
    plans = plans or {}
    heading = ParagraphStyle("cl_heading", parent=s.h1, fontSize=13, leading=16, spaceAfter=2)
    info = ParagraphStyle("cl_info", parent=s.body, fontSize=9.5, leading=13, alignment=TA_LEFT)
    head_cell = ParagraphStyle("cl_head_cell", parent=s.cell, textColor=colors.white, fontName="Helvetica-Bold")
    center_cell = ParagraphStyle("cl_center_cell", parent=s.cell, alignment=TA_CENTER)
    flow: list[Any] = []
    if header.get("organisation"):
        flow.append(Paragraph(_safe(header["organisation"]).upper(), heading))
    flow.append(Paragraph("MASTER BID SUBMISSION CHECKLIST", ParagraphStyle("cl_title", parent=heading, spaceAfter=10)))
    for label, key in (("GeM Bid No.", "bid_number"), ("Bid End Date/Time", "bid_end"),
                       ("Tender", "tender"), ("Bidder", "bidder")):
        if header.get(key):
            flow.append(Paragraph(f"<b>{label}:</b> {_safe(header[key])}", info))
    flow.append(Paragraph("Submission Checklist", ParagraphStyle("cl_sub", parent=s.h2, spaceBefore=8)))

    columns = ("S.No.", "Document", "Annexure / Ref.", "What to Upload", "Where", "Notary", "Status", "Pages")
    rows = [[Paragraph(h, head_cell) for h in columns]]
    style = list(_HEADER_STYLE)
    items = checklist.get("items", [])
    for n, item in enumerate(items, 1):
        status = item.get("status") or "-"
        plan = plans.get(item["id"])
        shown_status = f"{status} (see S.No. {plan.same_as})" if plan and plan.action == "duplicate" else status
        notary = bool(item.get("notary"))
        reference = item.get("reference") or (annexure_label(item.get("document") or "", item.get("format_text") or "")
                                              if item.get("source") == "draft" else "")
        rows.append([
            str(n),
            Paragraph(_safe(item.get("document") or "-"), s.cell),
            Paragraph(_safe(reference or "-"), s.cell),
            Paragraph(_safe(item.get("what_to_upload") or "-"), s.cell),
            Paragraph(_safe(item.get("where") or "-"), s.cell),
            Paragraph("<b>Yes</b>" if notary else "No", center_cell),
            Paragraph(_safe(shown_status), s.cell),
            Paragraph(fmt_pages(row_pages(n, items, plans, page_ranges)) or "-", center_cell),
        ])
        if status in _STATUS_FILL:
            style.append(("BACKGROUND", (6, n), (6, n), colors.HexColor(_STATUS_FILL[status])))
        if notary:
            style.append(("BACKGROUND", (5, n), (5, n), colors.HexColor("#fff3bf")))
    table = Table(rows, colWidths=[1.0 * cm, 3.0 * cm, 2.5 * cm, 3.5 * cm, 1.9 * cm, 1.4 * cm, 1.8 * cm, 1.5 * cm],
                  repeatRows=1)
    table.setStyle(TableStyle(style + [("FONTSIZE", (0, 1), (0, -1), 8.5), ("ALIGN", (0, 0), (0, -1), "CENTER")]))
    flow.append(table)
    notarised = [n for n, item in enumerate(checklist.get("items", []), 1) if item.get("notary")]
    if notarised:
        flow.append(Spacer(1, 6))
        flow.append(Paragraph(
            "Notary = Yes: to be notarised / executed on non-judicial stamp paper before submission (S.No. "
            + ", ".join(str(n) for n in notarised) + ").", s.small))
    return flow


def _cover_page_story(
    checklist: dict[str, Any], tender: dict[str, Any], company_profile: dict[str, Any], s: _Styles
) -> list[Any]:
    """The pack's first page: what is being bid for, by whom."""
    header = checklist.get("header") or {}
    big = ParagraphStyle("cover_big", parent=s.h1, fontSize=20, leading=26, spaceAfter=6)
    mid = ParagraphStyle("cover_mid", parent=s.body, fontSize=12, leading=17, alignment=TA_CENTER)
    label = ParagraphStyle("cover_label", parent=s.body, fontSize=10, leading=14, alignment=TA_LEFT)
    flow: list[Any] = [Spacer(1, 2.2 * cm), Paragraph("TECHNICAL BID SUBMISSION", big)]
    org = header.get("organisation") or tender.get("organisation")
    if org:
        flow.append(Paragraph(f"<b>{_safe(org).upper()}</b>", mid))
    flow.append(Spacer(1, 12))
    flow.append(Paragraph(_safe(header.get("tender") or tender.get("title") or ""), mid))
    flow.append(Spacer(1, 1.2 * cm))
    facts = [
        ("Bid / Tender No.", header.get("bid_number") or tender.get("tender_ref")),
        ("Bid End Date/Time", header.get("bid_end")),
        ("Submitted by", company_profile.get("legal_name")),
        ("Registered Office", company_profile.get("registered_office")),
        ("CIN", company_profile.get("cin")),
        ("PAN / GSTIN", " / ".join(v for v in (company_profile.get("pan"), company_profile.get("gstin")) if v)),
        ("Authorised Signatory", ", ".join(v for v in (
            (company_profile.get("authorized_signatory") or {}).get("name"),
            (company_profile.get("authorized_signatory") or {}).get("designation")) if v)),
        ("Contact", " | ".join(v for v in (company_profile.get("phone"), company_profile.get("email")) if v)),
        ("Date", dt.date.today().strftime("%d-%b-%Y")),
    ]
    table = Table([[Paragraph(f"<b>{k}</b>", label), Paragraph(_safe(v), label)] for k, v in facts if v],
                  colWidths=[4.6 * cm, 11.4 * cm])
    table.setStyle(TableStyle([
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#c9d2db")),
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#eef3f8")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
    ]))
    flow.append(table)
    return flow


def _table_widths(columns: list[str], rows: list[list[str]], frame: float) -> list[float]:
    """Column widths for a drafted table at 8.5pt: every column at least as
    wide as its longest word (so "Ongoing" or "3,02,50,000" never break
    mid-word), the rest of `frame` shared by how much text each holds."""
    char_w, pad = 0.165 * cm, 0.45 * cm
    cells = [[str(columns[i])] + [str(r[i]) if i < len(r) else "" for r in rows] for i in range(len(columns))]
    minimum = [min(max(len(w) for c in col for w in (c.split() or [""])) * char_w + pad, 4 * cm) for col in cells]
    if re.fullmatch(r"s\.?\s*no\.?|sr\.?\s*no\.?|#", str(columns[0]).strip().lower()):
        minimum[0] = 1.3 * cm
    spare = frame - sum(minimum)
    if spare <= 0:
        return [m * frame / sum(minimum) for m in minimum]
    weights = [min(sum(len(c) for c in col[1:]) / max(len(col) - 1, 1), 60) for col in cells]
    weights = [0 if i == 0 and minimum[0] == 1.3 * cm else w for i, w in enumerate(weights)]
    total = sum(weights) or 1
    return [m + spare * w / total for m, w in zip(minimum, weights)]


def _drafted_table(table_data: Any, s: _Styles) -> list[Any]:
    """One `tables` entry of a drafted document, full frame width."""
    columns = [c for c in (table_data.columns or [])]
    rows = [r for r in (table_data.rows or []) if any(str(c).strip() for c in r)]
    if not columns and not rows:
        return []
    width = max([len(columns)] + [len(r) for r in rows])
    columns = (columns + [""] * width)[:width]
    head_cell = ParagraphStyle("dt_head_cell", parent=s.cell, textColor=colors.white, fontName="Helvetica-Bold")
    data = [[Paragraph(_safe(c), head_cell) for c in columns]]
    for r in rows:
        data.append([Paragraph(_rich(c), s.cell) for c in (list(r) + [""] * width)[:width]])
    # splitInRow: a many-column table squeezes long cells (a project
    # description) into rows taller than a page - those continue on the
    # next page rather than failing the whole pack.
    table = Table(data, colWidths=_table_widths(columns, rows, 16.6 * cm), repeatRows=1, splitInRow=1)
    table.setStyle(TableStyle(_HEADER_STYLE + [("FONTSIZE", (0, 0), (-1, -1), 8.5)]))
    flow: list[Any] = []
    if table_data.title:
        flow.append(Paragraph(f"<b>{_safe(table_data.title)}</b>", s.body))
        flow.append(Spacer(1, 4))
    flow += [table, Spacer(1, 10)]
    return flow


# --- The tender's own check list annexure ------------------------------------------
# Many tenders prescribe their own "Check List" annexure (Sr. No. / Details of
# Requirement / Document Type / Yes-No / Page No.) - the same information as
# the master checklist, which the drafter can't know (it has no page
# numbers). Its table is filled here from the checklist once the pack is laid
# out; the drafter only writes the paragraphs around it.

_CHECK_LIST_RE = re.compile(r"check[\s-]?list|index of (enclosed )?documents", re.IGNORECASE)
CHECK_LIST_COLUMNS = ["Sr. No.", "Details of Requirement", "Document Type", "Yes / No", "If Yes Page No."]


def is_check_list_annexure(row: dict[str, Any]) -> bool:
    return row.get("source") == "draft" and bool(_CHECK_LIST_RE.search(row.get("document") or ""))


def _document_type(item: dict[str, Any]) -> str:
    if item.get("source") == "draft" and re.search(r"\bpresentation\b|\bppt\b", item.get("document") or "", re.I):
        return "Presentation slides"
    if item.get("source") == "draft":
        return "Prescribed format / letter on letterhead" if item.get("letterhead") else "Prescribed format"
    return "Self-attested copy" if item.get("signature") or item.get("stamp") else "Copy"


def _check_list_cell(column: str, n: int, item: dict[str, Any], enclosed: bool, pages: str) -> str:
    key = column.lower()
    if re.search(r"\bs\.?\s*no|\bsr\b|\bsr\.|\bsl\.?\s*no|serial|^#$", key):
        return str(n)
    if "page" in key:
        return pages if enclosed and pages else "-"
    if re.search(r"annex|\bref", key) and not re.search(r"requirement|particular|detail|description", key):
        return item.get("reference") or (annexure_label(item.get("document") or "")
                                         if item.get("source") == "draft" else "") or "-"
    if re.search(r"yes|enclosed|submitted|attached|furnished", key):
        return "Yes" if enclosed else "No"
    if "type" in key:
        return _document_type(item)
    if re.search(r"requirement|document|particular|detail|description|name", key):
        return item.get("document") or ""
    return ""


def fill_check_list_annexure(
    doc: "DraftedDocument | None",
    row: dict[str, Any],
    checklist: dict[str, Any],
    plans: dict[str, "RowPlan"],
    page_ranges: dict[str, PageRange] | None,
) -> Any:
    """`doc` (the drafted check list annexure) with its table rebuilt from
    the checklist - one row per document in the pack, Yes/No by whether it
    is enclosed and the pages it occupies - under the tender's own column
    headings when the draft kept them. Pages show "-" until `page_ranges`
    is known."""
    from types import SimpleNamespace

    tables = list(getattr(doc, "tables", None) or [])
    columns = next((list(t.columns) for t in tables if len(t.columns) >= 3), None) or CHECK_LIST_COLUMNS
    title = next((t.title for t in tables if t.title), "")
    items = checklist.get("items", [])
    rows: list[list[str]] = []
    for n, item in enumerate(items, 1):
        plan = plans.get(item["id"])
        if plan is None or plan.action == "skip":
            continue
        enclosed = plan.action in ("draft", "attach", "duplicate")
        pages = fmt_pages(row_pages(n, items, plans, page_ranges))
        rows.append([_check_list_cell(c, len(rows) + 1, item, enclosed, pages) for c in columns])
    table = SimpleNamespace(title=title, columns=columns, rows=rows)
    if doc is None:
        return SimpleNamespace(
            title=row.get("document") or "Check List", body_paragraphs=[], tables=[table],
            closing_paragraphs=["We confirm that the documents listed above are enclosed with our bid, at the "
                                "page numbers indicated."], open_items=[])
    return SimpleNamespace(title=doc.title, body_paragraphs=list(doc.body_paragraphs), tables=[table],
                           closing_paragraphs=list(doc.closing_paragraphs), open_items=[])


# --- Cross-references to other documents in the pack ----------------------------
# The drafter can't know page numbers, so a compliance statement, experience
# summary or covering letter cites an enclosure as "[[PAGE 7]]" (S.No. 7 of
# the checklist - see prompts.BID_DRAFT_SYSTEM_PROMPT); the token becomes
# the enclosure's final pages once the pack is laid out.

PAGE_REF_RE = re.compile(r"\[\[\s*PAGES?\s*(?:S\.?\s*NO\.?\s*)?(\d{1,3})\s*\]\]", re.IGNORECASE)
# A reference the drafter couldn't number - "[[PAGE [TO BE FILLED ...]]]",
# "[[PAGE ?]]" - for a document that isn't in the pack.
BAD_PAGE_REF_RE = re.compile(r"\[\[\s*PAGES?\b(?:[^\[\]]|\[[^\[\]]*\])*\]{1,3}", re.IGNORECASE)
_BAD_PAGE_REF_TEXT = "[TO BE VERIFIED: cites a document that is not in this bid pack]"
# Stands in for a page reference before the layout is known - as wide as a
# real one, so the document's length doesn't change once it's filled.
_PAGE_REF_STAND_IN = "Pages 000-000"


def _document_texts(doc: Any) -> list[str]:
    texts = [getattr(doc, "title", "") or "", *(getattr(doc, "body_paragraphs", None) or []),
             *(getattr(doc, "closing_paragraphs", None) or [])]
    for t in getattr(doc, "tables", None) or []:
        texts += [t.title or "", *t.columns, *(c for r in t.rows for c in r)]
    return [str(t) for t in texts]


_BLANK_FIELD_RE = re.compile(r"^[^:\n]{2,70}:\s*(?:_+|\.{3,}|-)?\s*$")
_NOT_A_FIELD_RE = re.compile(
    r"^(to|ref|subject|sub|dear .*|.*(as under|following|below|mention|include|instructions?|note|"
    r"declare|certify|undertake|annexure|appendix)\b.*)\s*:", re.IGNORECASE)
_SKIP_FILL_CHECK_RE = re.compile(r"check[\s-]?list|index|presentation|\bcvs?\b|curriculum vitae", re.IGNORECASE)


def looks_unfilled(doc: Any, company_profile: dict[str, Any], document: str = "") -> bool:
    """A drafted document that is still the bare format: it has empty fields
    (a "<field>:" line or a table cell) yet never names the bidder (legal
    name, CIN, PAN, GSTIN or signatory) nor raises a placeholder - or it
    leaves three or more "<field>:" lines empty."""
    name = f"{document} {getattr(doc, 'title', '')}"
    if doc is None or _SKIP_FILL_CHECK_RE.search(name) or _FINANCIAL_RE.search(name):
        return False  # a price bid's cells are the bidder's own, left blank on purpose
    texts = _document_texts(doc)[1:]  # the title doesn't count
    joined = " ".join(texts).lower()
    signatory = (company_profile.get("authorized_signatory") or {}).get("name")
    identifiers = [company_profile.get(k) for k in ("legal_name", "trade_name", "cin", "pan", "gstin")] + [signatory]
    names_bidder = any(i and i.lower() in joined for i in identifiers)
    blank_fields = [t for t in texts if _BLANK_FIELD_RE.match(t.strip()) and not _NOT_A_FIELD_RE.match(t.strip())]
    empty_cells = any(not str(c).strip() for t in getattr(doc, "tables", None) or [] for r in t.rows for c in r)
    has_gaps = bool(blank_fields) or empty_cells
    return (has_gaps and not names_bidder and not _PLACEHOLDER_RE.search(" ".join(texts))) or len(blank_fields) >= 3


def has_page_refs(doc: Any) -> bool:
    return doc is not None and any(BAD_PAGE_REF_RE.search(t) for t in _document_texts(doc))


def _page_ref_text(
    n: int, checklist: dict[str, Any], plans: dict[str, "RowPlan"], page_ranges: dict[str, PageRange] | None
) -> str:
    items = checklist.get("items", [])
    if not 1 <= n <= len(items):
        return f"[TO BE VERIFIED: reference to S.No. {n} - the checklist has no such row]"
    plan = plans.get(items[n - 1]["id"])
    if plan is None or plan.action == "skip":
        return "Not applicable"
    if plan.action == "placeholder":
        return f"[TO BE VERIFIED: S.No. {n} ({items[n - 1].get('document') or ''}) is not enclosed yet]"
    if page_ranges is None:
        return _PAGE_REF_STAND_IN
    pages = row_pages(n, items, plans, page_ranges)
    if not pages:
        return f"S.No. {n}"
    return f"{'Page' if pages[0] == pages[1] else 'Pages'} {fmt_pages(pages)}"


def resolve_page_refs(
    doc: Any, checklist: dict[str, Any], plans: dict[str, "RowPlan"], page_ranges: dict[str, PageRange] | None
) -> Any:
    """`doc` with every "[[PAGE n]]" replaced by S.No. n's pages in the pack
    (a stand-in of the same width while `page_ranges` is None)."""
    from types import SimpleNamespace

    def sub(text: Any) -> str:
        text = PAGE_REF_RE.sub(lambda m: _page_ref_text(int(m.group(1)), checklist, plans, page_ranges), str(text))
        return BAD_PAGE_REF_RE.sub(_BAD_PAGE_REF_TEXT, text)

    tables = [SimpleNamespace(title=sub(t.title or ""), columns=[sub(c) for c in t.columns],
                              rows=[[sub(c) for c in r] for r in t.rows]) for t in getattr(doc, "tables", None) or []]
    return SimpleNamespace(
        title=sub(doc.title or ""), body_paragraphs=[sub(p) for p in doc.body_paragraphs], tables=tables,
        closing_paragraphs=[sub(p) for p in getattr(doc, "closing_paragraphs", None) or []],
        open_items=list(getattr(doc, "open_items", None) or []))


def prepare_check_list_annexures(
    drafted_documents: "dict[str, DraftedDocument]", rows: list[dict[str, Any]]
) -> "dict[str, DraftedDocument]":
    """Drops the open items the drafter raised for a check list annexure
    (its rows and page numbers are filled when the pack is built - see
    fill_check_list_annexure), so they don't reach the review notes."""
    for row in rows:
        doc = drafted_documents.get(row["id"])
        if doc is not None and is_check_list_annexure(row):
            doc.open_items = []
    return drafted_documents


# A price schedule's rate / quantity / amount cells are the bidder's own
# figures - left blank to fill by hand, never a placeholder or "As quoted...".
_PRICE_FILLER_RE = re.compile(
    r"^\s*(\[TO BE (FILLED|VERIFIED)[^\]]*\]|as quoted in the (price schedule|financial bid)[^\n]*|-)\s*$",
    re.IGNORECASE)


def _blank_price_cells(table_data: Any) -> Any:
    from types import SimpleNamespace

    rows = [[("" if _PRICE_FILLER_RE.match(str(c)) else c) for c in r] for r in (table_data.rows or [])]
    return SimpleNamespace(title=table_data.title, columns=table_data.columns, rows=rows)


# --- One page set per checklist row ------------------------------------------------

def _drafted_document_section(
    doc: "DraftedDocument",
    row: dict[str, Any],
    company_profile: dict[str, Any],
    company_documents: list[CompanyDocumentRef],
    s: _Styles,
) -> list[Any]:
    """One AI-drafted document (app.intelligence.bid_drafter) as its own
    page(s): date, title, body paragraphs, any tables, closing paragraphs,
    then the signature block with the signature/seal the row asks for. Its
    open items go to the internal review notes, not onto the page itself."""
    flow: list[Any] = [_date_line(s.body),
                       Paragraph(_safe(normalise_labels(doc.title or row.get("document") or "")), s.h1)]
    for para in doc.body_paragraphs:
        if para.strip():
            flow.append(Paragraph(_format_paragraph(para), s.body))
            flow.append(Spacer(1, 6))
    price_bid = row.get("source") == "draft" and bool(_FINANCIAL_RE.search(row.get("document") or ""))
    for table_data in getattr(doc, "tables", None) or []:
        flow += _drafted_table(_blank_price_cells(table_data) if price_bid else table_data, s)
    for para in getattr(doc, "closing_paragraphs", None) or []:
        if para.strip():
            flow.append(Paragraph(_format_paragraph(para), s.body))
            flow.append(Spacer(1, 6))
    flow += _signature_block(company_profile, company_documents, s.body,
                             bool(row.get("signature")), bool(row.get("stamp")), lead=_closing_lines(flow))
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
                             bool(row.get("signature")), bool(row.get("stamp")), lead=_closing_lines(flow))
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
                             bool(row.get("signature")), bool(row.get("stamp")), lead=_closing_lines(flow))
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


@functools.lru_cache(maxsize=16)
def _signing_overlay(
    width: float, height: float, bottom: float, signature: bytes | None, seal: bytes | None
) -> PageObject | None:
    """A transparent page carrying the signature and/or seal at its bottom
    right - merged over each page of a self-attested copy. Cached: a pack's
    pages share a handful of sizes, and merging one overlay page onto many
    pages (pypdf's watermark pattern) embeds its images once, not per page."""
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


_STARTXREF_RE = re.compile(rb"startxref\s+(\d+)\s*")


def _pdf_reader(data: bytes) -> PdfReader:
    """A PdfReader for `data`, repairing what PDF viewers shrug off but
    pypdf doesn't: a file padded after its startxref (pypdf only looks for it
    near the end - e.g. a test file filled out to 100 MB), or a damaged
    cross-reference table (rewritten by pdfium, which rebuilds it)."""
    if b"startxref" not in data[-2048:]:
        # pypdf would take seconds to give up on a padded file - trim it first.
        matches = list(_STARTXREF_RE.finditer(data))
        if matches and matches[-1].end() < len(data) - 2048:
            data = data[:matches[-1].end()] + b"\n%%EOF\n"
    try:
        reader = PdfReader(io.BytesIO(data))
        len(reader.pages)
        return reader
    except Exception as first_error:
        matches = list(_STARTXREF_RE.finditer(data))
        if matches:
            trimmed = data[:matches[-1].end()] + b"\n%%EOF\n"
            try:
                reader = PdfReader(io.BytesIO(trimmed))
                len(reader.pages)
                return reader
            except Exception:  # noqa: BLE001 - try pdfium next
                pass
        try:
            import pypdfium2

            out = io.BytesIO()
            pypdfium2.PdfDocument(data).save(out)
            reader = PdfReader(io.BytesIO(out.getvalue()))
            len(reader.pages)
            return reader
        except Exception:  # noqa: BLE001 - the original error says more
            raise first_error


def _attached_source_pages(doc: CompanyDocumentRef) -> list[PageObject]:
    """A library document's own pages (an image upload as one A4 page)."""
    data = doc.open_bytes()
    try:
        return list(_pdf_reader(data if _is_pdf(doc) else _image_as_pdf(data)).pages)
    except Exception as exc:
        raise BidGenerationError(f"Could not read '{doc.name}': {exc}") from exc


def _add_attached_pages(
    writer: PdfWriter,
    doc: CompanyDocumentRef,
    row: dict[str, Any],
    letterhead: PageObject | None,
    signature: bytes | None,
    seal: bytes | None,
    pages: list[PageObject] | None = None,
) -> None:
    """Adds a library document's pages (`pages`, when already read) to
    `writer` - placed on the letterhead and/or signed and stamped when the
    row asks for it."""
    pages = pages if pages is not None else _attached_source_pages(doc)

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


# --- Marking a standalone upload (the dashboard's Sign & Stamp tab) ------------------

def _fit(image: PILImage.Image, box_w: float, box_h: float) -> PILImage.Image:
    scale = min(box_w / image.width, box_h / image.height)
    return image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))), PILImage.LANCZOS)


MARK_KINDS = ("signature", "stamp")


@dataclass
class MarkPlacement:
    """Where the user dragged one signature/stamp on one page of the Sign &
    Stamp tab's preview - as fractions of that (final, on-letterhead when
    chosen) page's width/height, measured from its top-left corner."""

    page: int
    mark: str  # one of MARK_KINDS
    x: float
    y: float
    w: float
    h: float


@dataclass
class ContentLayout:
    """How one page's own content is moved on the letterhead so it clears
    the header, footer and left band - worked out by the Sign & Stamp tab's
    preview from where the page's ink actually is, and left out for pages
    that already clear them. Content is scaled by `scale` about the page's
    top-left corner, then shifted by `dx`/`dy` (fractions of the page's
    width/height, from its top-left corner)."""

    page: int
    scale: float
    dx: float
    dy: float


def _content_transformation(layout: ContentLayout, box: Any) -> Transformation:
    """`layout` as a PDF transformation - PDF space runs bottom-up from the
    mediabox's lower-left corner, the layout top-down from its upper-left."""
    s, llx, lly = layout.scale, float(box.left), float(box.bottom)
    width, height = float(box.width), float(box.height)
    return Transformation().scale(s, s).translate(
        llx * (1 - s) + layout.dx * width,
        lly * (1 - s) + height * (1 - s) - layout.dy * height,
    )


def _open_mark(data: bytes | None) -> PILImage.Image | None:
    if not data:
        return None
    try:
        return ImageOps.exif_transpose(PILImage.open(io.BytesIO(data))).convert("RGBA")
    except Exception:
        return None


def _image_source(image: str | Path | bytes) -> Any:
    """A letterhead given as a file path or as image bytes, in the form
    PIL / reportlab's ImageReader open."""
    return io.BytesIO(image) if isinstance(image, bytes) else str(image)


def _marks_bottom(page_height: float, on_letterhead: bool) -> float:
    """How far above the page's bottom edge the usual signature/stamp sit -
    clear of the letterhead's footer strip (its height scaled to the page)
    when there is one."""
    return LETTERHEAD_MARGINS[3] * page_height / A4[1] if on_letterhead else 1 * cm


def _mark_image(
    data: bytes,
    letterhead_image: str | Path | bytes | None,
    signature: bytes | None,
    seal: bytes | None,
    placements: list[MarkPlacement] | None = None,
    layout: ContentLayout | None = None,
) -> tuple[bytes, str]:
    """An image upload with the letterhead laid over it at full size
    (multiplied, so the upload shows through its white areas; its content
    moved per `layout` first) and/or signed and stamped at its bottom right
    (or wherever `placements` puts them) - returned as an image of the same
    size and kind (PNG stays PNG, anything else becomes JPEG)."""
    try:
        source = PILImage.open(io.BytesIO(data))
        is_png = source.format == "PNG"
        source = ImageOps.exif_transpose(source).convert("RGBA")
    except Exception as exc:
        raise BidGenerationError(f"Could not read the image: {exc}") from exc

    canvas = PILImage.new("RGBA", source.size, "white")
    if letterhead_image and layout is not None:
        size = (max(1, round(source.width * layout.scale)), max(1, round(source.height * layout.scale)))
        layer = PILImage.new("RGBA", canvas.size)
        layer.paste(source.resize(size, PILImage.LANCZOS), (round(layout.dx * canvas.width), round(layout.dy * canvas.height)))
        source = layer
    canvas.alpha_composite(source)
    # Marks are sized as if the image were an A4 page, as _image_as_pdf treats one.
    pt = min(canvas.width / A4[0], canvas.height / A4[1])  # pixels per PDF point
    if letterhead_image:
        letterhead = PILImage.open(_image_source(letterhead_image)).convert("RGB").resize(canvas.size, PILImage.LANCZOS)
        canvas = ImageChops.multiply(canvas.convert("RGB"), letterhead).convert("RGBA")
    mark_bottom = _marks_bottom(canvas.height, bool(letterhead_image)) * (1 if letterhead_image else pt)

    marks = {"signature": _open_mark(signature), "stamp": _open_mark(seal)}
    if placements is not None:
        for p in placements:
            image = marks.get(p.mark)
            if image is None or p.page != 0:
                continue
            size = (max(1, round(p.w * canvas.width)), max(1, round(p.h * canvas.height)))
            layer = PILImage.new("RGBA", canvas.size)
            layer.paste(image.resize(size, PILImage.LANCZOS), (round(p.x * canvas.width), round(p.y * canvas.height)))
            canvas.alpha_composite(layer)  # a layer, so marks dragged part-way off the page are clipped
    else:
        x = canvas.width - 1 * cm * pt
        for image, box_w, box_h in ((marks["stamp"], 2.4 * cm, 2.4 * cm), (marks["signature"], 3.4 * cm, 1.5 * cm)):
            if image is None:  # stamp rightmost, signature to its left
                continue
            box_w, box_h = box_w * pt, box_h * pt
            x -= box_w
            image = _fit(image, box_w, box_h)
            canvas.alpha_composite(image, (round(x), round(canvas.height - mark_bottom - image.height)))
            x -= 0.3 * cm * pt

    out = io.BytesIO()
    if is_png:
        canvas.save(out, format="PNG", optimize=True)
        return out.getvalue(), "image/png"
    canvas.convert("RGB").save(out, format="JPEG", quality=90)
    return out.getvalue(), "image/jpeg"


@functools.lru_cache(maxsize=16)
def _letterhead_overlay(letterhead_image: str | bytes, width: float, height: float) -> PageObject | None:
    """A page carrying just the letterhead, stretched to `width` x `height`
    and multiplied - merged over each page, the page keeps its size and its
    content shows through the letterhead's white areas (even on a scan with
    an opaque white background). Cached like _signing_overlay, so a
    document's pages share one embedded letterhead image."""
    try:
        image = ImageReader(_image_source(letterhead_image))
    except Exception:
        return None
    buf = io.BytesIO()
    c = pdf_canvas.Canvas(buf, pagesize=(width, height))
    c.setBlendMode("Multiply")
    c.drawImage(image, 0, 0, width=width, height=height)
    c.showPage()
    c.save()
    return PdfReader(io.BytesIO(buf.getvalue())).pages[0]


def _merge_over(page: PageObject, overlay: PageObject) -> None:
    """Merges a page-sized overlay (drawn from 0,0) onto `page`, wherever its
    mediabox starts."""
    box = page.mediabox
    page.merge_transformed_page(overlay, Transformation().translate(float(box.left), float(box.bottom)))


@dataclass
class PageMarks:
    """What one page gets in the Sign & Stamp tab: its own letterhead (a
    file path or image bytes), signature and seal image bytes - any of them
    None. A page with none of them is left exactly as it was."""

    letterhead: str | Path | bytes | None = None
    signature: bytes | None = None
    seal: bytes | None = None

    def marks(self) -> dict[str, bytes | None]:
        return {"signature": self.signature, "stamp": self.seal}


def _place_marks(writer: PdfWriter, placements: list[MarkPlacement], page_marks: Any) -> None:
    """Draws each placement's mark onto its page of `writer` - the mark's
    image being that page's own signature/seal (`page_marks(index)`, a
    PageMarks). One overlay per page; compress_identical_objects later
    folds the repeated images back into one copy each."""
    readers: dict[int, Any] = {}  # id(image bytes) -> ImageReader, so each image is decoded once

    def reader(data: bytes) -> Any:
        if id(data) not in readers:
            try:
                readers[id(data)] = ImageReader(io.BytesIO(data))
            except Exception:
                readers[id(data)] = None
        return readers[id(data)]

    by_page: dict[int, list[tuple[MarkPlacement, Any]]] = {}
    for p in placements:
        if not 0 <= p.page < len(writer.pages):
            continue
        data = page_marks(p.page).marks().get(p.mark)
        image = reader(data) if data else None
        if image is not None:
            by_page.setdefault(p.page, []).append((p, image))
    for index, page_placements in by_page.items():
        page = writer.pages[index]
        width, height = float(page.mediabox.width), float(page.mediabox.height)
        buf = io.BytesIO()
        c = pdf_canvas.Canvas(buf, pagesize=(width, height))
        for p, image in page_placements:
            c.drawImage(image, p.x * width, (1 - p.y - p.h) * height, width=p.w * width,
                        height=p.h * height, mask="auto")
        c.showPage()
        c.save()
        _merge_over(page, PdfReader(io.BytesIO(buf.getvalue())).pages[0])


def mark_document(
    data: bytes,
    filename: str,
    content_type: str,
    *,
    letterhead_image: str | Path | bytes | None = None,
    signature: bytes | None = None,
    seal: bytes | None = None,
    placements: list[MarkPlacement] | None = None,
    layouts: list[ContentLayout] | None = None,
    pages: dict[int, PageMarks] | None = None,
) -> tuple[bytes, str]:
    """An uploaded PDF or image (the dashboard's Sign & Stamp tab) with the
    letterhead laid over every page at the page's own size, and the
    signature and/or seal at the bottom right - or, with `placements`,
    exactly the marks listed there, where they're listed (pages with none
    get none). `pages` (page index -> PageMarks) gives each page its own
    marks instead of `letterhead_image`/`signature`/`seal`; pages not in it
    are left as they are. Unlike a bid pack's library documents, a page's
    content is only moved (per `layouts`, on the letterhead only) when it
    would otherwise run into the letterhead. Returns (bytes, media type):
    a PDF for a PDF, an image for an image."""
    everywhere = PageMarks(letterhead_image, signature, seal)
    page_marks = (lambda index: pages.get(index) or PageMarks()) if pages is not None else (lambda index: everywhere)
    doc = CompanyDocumentRef(id="", name=filename, filename=filename, content_type=content_type,
                             open_bytes=lambda: data)
    by_page = {layout.page: layout for layout in layouts or []}
    if not (_is_pdf(doc) or data.startswith(b"%PDF")):
        first = page_marks(0)
        return _mark_image(data, first.letterhead, first.signature, first.seal, placements, by_page.get(0))

    writer = PdfWriter()
    for index, source in enumerate(_attached_source_pages(doc)):
        page = writer.add_page(source)
        marks = page_marks(index)
        if not (marks.letterhead or marks.signature or marks.seal):
            continue  # an unmarked page is copied exactly as it was
        page.transfer_rotation_to_content()
        width, height = float(page.mediabox.width), float(page.mediabox.height)
        letterhead = marks.letterhead
        if letterhead and index in by_page:
            page.add_transformation(_content_transformation(by_page[index], page.mediabox))
        if letterhead:
            overlay = _letterhead_overlay(letterhead if isinstance(letterhead, bytes) else str(letterhead), width, height)
            if overlay is not None:
                _merge_over(page, overlay)
        if placements is None:
            overlay = _signing_overlay(width, height, _marks_bottom(height, bool(letterhead)), marks.signature, marks.seal)
            if overlay is not None:
                _merge_over(page, overlay)
    if placements is not None:
        _place_marks(writer, placements, page_marks)
    writer.compress_identical_objects()
    out = io.BytesIO()
    writer.write(out)
    return out.getvalue(), "application/pdf"


def _is_pdf_upload(data: bytes, filename: str, content_type: str) -> bool:
    return content_type == "application/pdf" or filename.lower().endswith(".pdf") or data.startswith(b"%PDF")


def document_page_count(data: bytes, filename: str, content_type: str) -> int:
    """How many pages mark_document sees in an upload - an image is one."""
    if not _is_pdf_upload(data, filename, content_type):
        return 1
    try:
        return len(_pdf_reader(data).pages)
    except Exception as exc:
        raise BidGenerationError(f"Could not read '{filename}': {exc}") from exc


def combine_marked(marked: list[tuple[bytes, str]]) -> bytes:
    """Several mark_document results (bytes, media type) as one PDF, in
    order. A marked image becomes one page of its own size, scaled as if it
    were an A4 page (as _mark_image and the Sign & Stamp preview treat it)."""
    writer = PdfWriter()
    for data, media_type in marked:
        if media_type != "application/pdf":
            image = ImageReader(io.BytesIO(data))
            img_w, img_h = image.getSize()
            pt = min(img_w / A4[0], img_h / A4[1])  # pixels per PDF point
            page_w, page_h = img_w / pt, img_h / pt
            buf = io.BytesIO()
            c = pdf_canvas.Canvas(buf, pagesize=(page_w, page_h))
            c.drawImage(image, 0, 0, width=page_w, height=page_h)
            c.showPage()
            c.save()
            data = buf.getvalue()
        writer.append(PdfReader(io.BytesIO(data)))
    writer.compress_identical_objects()
    out = io.BytesIO()
    writer.write(out)
    return out.getvalue()


# --- Continuous page numbers ---------------------------------------------------------

PAGE_NUMBER_BOTTOM = 0.45 * cm  # below the letterhead's footer contact strip


def stamp_page_numbers(writer: PdfWriter, count: int, serials: list[int | None] | None = None) -> None:
    """"Page N of <count>" at the bottom centre of the first `count` pages of
    `writer` - on a small white ground, so it reads on a scan too. With
    `serials` (per page, the checklist S.No. of the document it belongs to,
    None for the cover and index) it reads "S.No. 4 | Page N of <count>",
    matching the checklist. One overlay PDF holds every page's number, each
    page sized like its target."""
    if count <= 0:
        return
    buf = io.BytesIO()
    c = pdf_canvas.Canvas(buf)
    for i in range(count):
        box = writer.pages[i].cropbox  # the visible area - a scan's may be smaller than its mediabox
        width, height = float(box.width), float(box.height)
        c.setPageSize((width, height))
        serial = serials[i] if serials and i < len(serials) else None
        text = f"Page {i + 1} of {count}"
        if serial is not None:
            text = f"S.No. {serial}  |  {text}"
        text_w = c.stringWidth(text, "Helvetica", 8)
        c.setFillColor(colors.white)
        c.rect(width / 2 - text_w / 2 - 4, PAGE_NUMBER_BOTTOM - 3, text_w + 8, 11, stroke=0, fill=1)
        c.setFillColor(colors.HexColor("#343a40"))
        c.setFont("Helvetica", 8)
        c.drawCentredString(width / 2, PAGE_NUMBER_BOTTOM, text)
        c.showPage()
    c.save()
    overlays = PdfReader(io.BytesIO(buf.getvalue())).pages
    for i in range(count):
        box = writer.pages[i].cropbox
        writer.pages[i].merge_transformed_page(
            overlays[i], Transformation().translate(float(box.left), float(box.bottom)))


# --- Pre-submission QA -------------------------------------------------------------

_GEM_BID_RE = re.compile(r"\bGEM/\d{4}/B/\d+\b", re.IGNORECASE)
_AMOUNT_RE = re.compile(r"(?:rs\.?|inr|₹)\s*[\d,]+(?:\.\d+)?(?:\s*(?:lakh|lakhs|crore|crores|cr\.?))?", re.IGNORECASE)
_FINANCIAL_CLAIM_RE = re.compile(r"turnover|net ?worth|solvency", re.IGNORECASE)
# Documents that must name the tender they answer.
_TENDER_SPECIFIC_RE = re.compile(
    r"covering letter|bid (submission )?(form|letter)|declaration|undertaking|affidavit|authori[sz]ation|"
    r"compliance", re.IGNORECASE)
_PERSONNEL_RE = re.compile(r"curriculum vitae|\bcvs?\b|personnel|team composition|key staff|manpower", re.IGNORECASE)


def _squash(text: str) -> str:
    return re.sub(r"[\s,]+", "", text.lower()).replace("₹", "rs").replace("rs.", "rs").replace("inr", "rs")


def qa_findings(
    tender: dict[str, Any],
    checklist: dict[str, Any],
    plans: dict[str, "RowPlan"],
    page_ranges: dict[str, PageRange] | None,
    drafted_documents: dict[str, Any],
    company_profile: dict[str, Any],
    company_documents: list[CompanyDocumentRef],
    presentations: Iterable[str] = (),
) -> list[str]:
    """What an automated read of the finished pack still finds wrong -
    listed in the internal review notes. Never blocks the pack: each line
    is for the team to fix or confirm before upload."""
    import json

    findings: list[str] = []
    header = checklist.get("header") or {}
    items = checklist.get("items", [])
    presentations = set(presentations)
    bid_number = (header.get("bid_number") or "").strip()
    tender_refs = [r for r in {bid_number, (tender.get("tender_ref") or "").strip()} if len(r) >= 4]
    known_amounts = _squash(json.dumps(company_profile.get("financials") or {}, ensure_ascii=False)
                            + " " + (tender.get("document_text") or "")
                            + " " + json.dumps(tender.get("document_summary") or {}, ensure_ascii=False, default=str))

    if header.get("summary_only") or _thin_tender_text(tender):
        findings.append("The tender's own documents were not read in full - check every annexure, format, "
                        "corrigendum and the GeM ATC against the tender before upload.")

    for n, item in enumerate(items, 1):
        plan = plans.get(item["id"])
        if plan is None or plan.action != "draft" or item["id"] in presentations:
            continue
        where = f"S.No. {n} {item.get('document') or ''}"
        pages = fmt_pages(row_pages(n, items, plans, page_ranges))
        where += f" (page {pages})" if pages else ""
        doc = drafted_documents.get(item["id"])
        if doc is None:
            if re.search(r"covering letter", item.get("document") or "", re.IGNORECASE):
                findings.append(f"{where}: a standard templated letter - check it against the tender's own "
                                "covering letter / bid form, if it prescribes one.")
            elif not is_check_list_annexure(item):
                findings.append(f"{where}: not drafted - its page is a template to complete by hand.")
            continue
        if looks_unfilled(doc, company_profile, item.get("document") or ""):
            findings.append(f"{where}: looks unfilled - the format is there but the bidder's details are not "
                            "filled in. Fill it from the company records before upload.")
        texts = _document_texts(resolve_page_refs(doc, checklist, plans, page_ranges))
        placeholders = [p for t in texts for p in _PLACEHOLDER_RE.findall(t)]
        if placeholders:
            findings.append(f"{where}: {len(placeholders)} placeholder(s) still to resolve, e.g. {placeholders[0]}")
        joined = " ".join(texts)
        others = sorted({m.upper() for m in _GEM_BID_RE.findall(joined)} - {bid_number.upper()})
        if others and bid_number:
            findings.append(f"{where}: mentions bid number(s) {', '.join(others)}, not this tender's {bid_number} "
                            "- possible copy-paste from another tender.")
        if (tender_refs and _TENDER_SPECIFIC_RE.search(item.get("document") or "")
                and not any(r.lower() in joined.lower() for r in tender_refs)):
            findings.append(f"{where}: does not quote the tender reference ({' / '.join(tender_refs)}).")
        for text in texts:
            if not _FINANCIAL_CLAIM_RE.search(text):
                continue
            for amount in _AMOUNT_RE.findall(text):
                if _squash(amount) not in known_amounts:
                    findings.append(f"{where}: financial figure \"{amount.strip()}\" matches neither the "
                                    "CA-certified figures on file nor the tender - verify it.")
        if "Chartered Accountant to issue" in (item.get("notes") or ""):
            findings.append(f"{where}: prepared for the CA - get it issued on the CA's letterhead with UDIN, "
                            "signed and sealed, and enclose that instead.")

    # Every eligibility condition must be proved by something actually in the pack.
    for condition in checklist.get("conditions") or []:
        proofs = [_name_key(p) for p in re.split(r"[;,]", condition.get("proved_by") or "") if p.strip()]
        if not proofs:
            continue
        rows = [(n, i) for n, i in enumerate(items, 1)
                if any(p and (p in _name_key(i.get("document") or "") or _name_key(i.get("document") or "") in p)
                       for p in proofs)]
        if not rows:
            findings.append(f"Condition \"{condition.get('ref')}\": its proof ({condition.get('proved_by')}) is not "
                            "on the checklist - confirm the condition is covered.")
        elif not any(plans.get(i["id"]) and plans[i["id"]].action in ("attach", "draft", "duplicate", "skip")
                     for _, i in rows):
            findings.append(f"Condition \"{condition.get('ref')}\": its proof (S.No. "
                            f"{', '.join(str(n) for n, _ in rows)}) is not enclosed - the bid fails this condition "
                            "without it.")

    signature_doc, seal_doc = signing_assets(company_profile, company_documents)
    active = [i for i in items if plans.get(i["id"]) and plans[i["id"]].action not in ("skip", "duplicate")]
    if signature_doc is None and any(i.get("signature") for i in active):
        findings.append("No signature image in the Documents library - every signed page needs a wet signature.")
    if seal_doc is None and any(i.get("stamp") for i in active):
        findings.append("No seal image in the Documents library - every sealed page needs the company seal.")

    today = dt.date.today()
    for cert in company_profile.get("certifications", []):
        try:
            valid_until = dt.date.fromisoformat(str(cert.get("valid_until") or ""))
        except ValueError:
            continue
        if valid_until < today:
            findings.append(f"{cert.get('name')} expired on {valid_until.strftime('%d-%m-%Y')} - renew it or "
                            "leave it out of the bid.")

    if not [k for k in company_profile.get("key_personnel", []) if k.get("name")] and any(
            _PERSONNEL_RE.search(i.get("document") or "") for i in active):
        findings.append("No key personnel on record (company_profile.json key_personnel) - the personnel list and "
                        "CVs cannot be filled; add each proposed person with qualification and experience.")
    return findings


# Checked by a person on the final set - what no automated read can confirm.
MANUAL_QA_CHECKS = (
    "Every annexure compared line by line with the tender's own format - no heading, clause, declaration, "
    "column or row changed, merged or left out.",
    "Every condition of the eligibility criteria, ATC, scope of work, corrigenda and GeM bid document is answered "
    "by an enclosed document.",
    "Tender number, tender name, department and authority names and dates are this tender's on every page.",
    "Turnover, net worth and project values match the CA certificates and audited statements enclosed.",
    "Each proposed person meets the tender's minimum qualification and experience.",
    "Every page needing it is signed and sealed; notarised documents are executed on stamp paper.",
    "Page numbers run continuously and every index / compliance page reference points at the right page.",
    "Portal requirements met: file names, file size limits, upload slots and document sequence.",
)


# --- Internal review notes (plain pages, removed before submission) --------------

def _internal_notes_story(
    tender: dict[str, Any],
    checklist: dict[str, Any],
    drafting_note: str | None,
    s: _Styles,
    qa: list[str] | None = None,
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

    story.append(Paragraph("Pre-submission QA - found automatically", s.h2))
    for finding in qa or []:
        story.append(Paragraph(f"• {_rich(finding)}", body))
    if not qa:
        story.append(Paragraph("Nothing found - still complete the checks below.", body))
    story.append(Paragraph("Pre-submission QA - check by hand before upload", s.h2))
    for check in MANUAL_QA_CHECKS:
        story.append(Paragraph(f"[ ] {_safe(check)}", body))

    pending = [(n, i) for n, i in enumerate(checklist.get("items", []), 1)
               if i.get("status") in (STATUS_MISSING, STATUS_TO_PREPARE)]
    story.append(Paragraph("Checklist rows still needing attention", s.h2))
    for n, item in pending:
        story.append(Paragraph(
            f"• {n}. <b>{_safe(item.get('document') or '')}</b> - {_safe(item.get('status'))}"
            + (f" <i>({_safe(item['notes'])})</i>" if item.get("notes") else ""), body))
    if not pending:
        story.append(Paragraph("None - every row is enclosed or marked not applicable.", body))

    notarised = [(n, i) for n, i in enumerate(checklist.get("items", []), 1)
                 if i.get("notary") and i.get("status") != STATUS_NOT_APPLICABLE]
    story.append(Paragraph("DOCUMENTS TO BE NOTARISED", s.h2))
    for n, item in notarised:
        affidavit = re.search(r"affidavit|stamp paper|non[- ]judicial", " ".join(
            str(item.get(k) or "") for k in ("document", "what_to_upload", "format_text", "notes")), re.I)
        story.append(Paragraph(
            f"• {n}. <b>{_safe(item.get('document') or '')}</b> - get it notarised"
            + (" (execute on non-judicial stamp paper of the value the tender prescribes)" if affidavit else "")
            + " before uploading.", body))
    if not notarised:
        story.append(Paragraph("None - the tender asks for no notarised document.", body))

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
    cover_page: bool = True,
    presentations: dict[str, bytes] | None = None,
) -> bytes:
    """Builds the bid pack PDF from `checklist` (the tender's saved,
    possibly hand-edited submission checklist - built fresh via
    build_checklist's fallback when None) and returns it as bytes:

    1. A cover page (`cover_page`), then the Master Bid Submission Checklist
       - also the table of contents (the page each row's document starts
       on) - on the letterhead (`letterhead_image`, drawn full-page; plain
       when None).
    2. Every row's document, in S.No order (rows marked Not applicable are
       left out): a "draft" row's AI draft from
       `drafted_documents` (by row id - a templated covering letter or a
       to-be-completed page when there's none), a library document merged
       in, or a placeholder page when it can't be (see plan_rows). A
       library document is merged once, however many rows point at it. The
       letterhead and the signature/seal go wherever the row's
       letterhead/signature/stamp say. A presentation row with a rendered
       deck in `presentations` (by row id - PDF bytes, see
       app.reports.presentation) is its slides instead.
    3. Internal review notes on plain pages, to be removed before submitting
       - with the automated pre-submission QA (see qa_findings).

    Every page of 1-2 carries "Page N of T" - a document's pages prefixed
    with its checklist S.No. ("S.No. 4 | Page N of T"); the index, a check list
    annexure and every "[[PAGE n]]" cross-reference in a draft show the
    final page numbers (see resolve_page_refs).
    """
    s = _styles()
    title = f"Bid Pack - {tender.get('title') or tender.get('tender_ref') or 'Tender'}"
    drafted_documents = guard_financial_years(drafted_documents or {}, company_profile)
    presentations = presentations or {}
    if not is_submission_checklist(checklist):
        checklist = build_checklist(tender, eligibility_criteria, company_profile, company_documents)
    checklist = refresh_statuses(checklist, company_documents, set(drafted_documents) | set(presentations))
    plans = plan_rows(checklist, company_documents)

    letterhead_page = _letterhead_page(letterhead_image)
    sig_doc, seal_doc = signing_assets(company_profile, company_documents)
    signature, seal = _read_bytes(sig_doc), _read_bytes(seal_doc)

    def _render(story: list[Any], on_letterhead: bool) -> list[PageObject]:
        data = _build_pdf(story, title, letterhead_image if on_letterhead and letterhead_page is not None else None)
        return list(PdfReader(io.BytesIO(data)).pages)

    # Everything after the checklist is laid out first, so the checklist
    # can say where each row's document starts. A part is ("pages", pages)
    # or ("attach", doc, row, source pages).
    items = checklist.get("items", [])
    shown = [(n, row) for n, row in enumerate(items, 1) if plans[row["id"]].action not in ("skip", "duplicate")]
    parts: list[tuple[Any, ...]] = []
    row_part: dict[str, int] = {}  # row id -> index of its first part
    merged: set[str] = set()  # library document ids already in the pack
    # part index -> its row, for documents holding page numbers (a check list
    # annexure, or a draft citing "[[PAGE n]]") - re-rendered once the layout
    # is known.
    paged: dict[int, dict[str, Any]] = {}

    def _paged_story(row: dict[str, Any], page_ranges: dict[str, PageRange] | None) -> list[Any]:
        doc = drafted_documents.get(row["id"])
        if is_check_list_annexure(row):
            doc = fill_check_list_annexure(doc, row, checklist, plans, page_ranges)
        doc = resolve_page_refs(doc, checklist, plans, page_ranges)
        return _drafted_document_section(doc, row, company_profile, company_documents, s)

    for n, row in shown:
        plan = plans[row["id"]]
        if plan.action == "attach" and plan.doc.id in merged:
            continue
        row_part[row["id"]] = len(parts)
        if plan.action == "draft" and row["id"] in presentations:
            parts.append(("pages", list(PdfReader(io.BytesIO(presentations[row["id"]])).pages)))
        elif plan.action == "draft":
            drafted = drafted_documents.get(row["id"])
            if is_check_list_annexure(row) or has_page_refs(drafted):
                paged[len(parts)] = row
                story = _paged_story(row, None)
            elif drafted is not None:
                story = _drafted_document_section(drafted, row, company_profile, company_documents, s)
            elif re.search(r"covering letter", row.get("document") or "", re.IGNORECASE):
                story = _fallback_covering_letter_section(tender, row, company_profile, company_documents, s)
            else:
                story = _undrafted_document_section(row, company_profile, company_documents, s)
            parts.append(("pages", _render(story, bool(row.get("letterhead")))))
        elif plan.action == "attach":
            merged.add(plan.doc.id)
            parts.append(("attach", plan.doc, row, _attached_source_pages(plan.doc)))
        else:
            parts.append(("pages", _render(_placeholder_section(n, row, plan.reason, s), False)))

    front = _render(_cover_page_story(checklist, tender, company_profile, s), True) if cover_page else []

    def _layout(first: int) -> dict[str, PageRange]:
        spans = []
        for part in parts:
            length = len(part[1]) if part[0] == "pages" else len(part[3])
            spans.append((first, first + max(length, 1) - 1))
            first += length
        return {row_id: spans[i] for row_id, i in row_part.items()}

    # The index and the documents citing page numbers are rendered with the
    # layout, which they can change by a page themselves (a longer page
    # range wrapping) - so it's redone until it holds.
    page_ranges: dict[str, PageRange] | None = None
    index_pages: list[PageObject] = []
    for _ in range(5):
        index_pages = _render(_submission_checklist_story(checklist, s, plans, page_ranges), True)
        layout = _layout(len(front) + len(index_pages) + 1)
        resized = False
        for i, row in paged.items():
            pages = _render(_paged_story(row, layout), bool(row.get("letterhead")))
            resized = resized or len(pages) != len(parts[i][1])
            parts[i] = ("pages", pages)
        if not resized and layout == page_ranges:
            break
        page_ranges = _layout(len(front) + len(index_pages) + 1) if resized else layout

    qa = qa_findings(tender, checklist, plans, page_ranges, drafted_documents, company_profile,
                     company_documents, presentations)
    # Each part's checklist S.No. (a library document several rows point at
    # carries the first one's), printed with every page number.
    part_serial = {row_part[row["id"]]: n for n, row in reversed(shown) if row["id"] in row_part}
    writer = PdfWriter()
    try:
        for page in front + index_pages:
            writer.add_page(page)
        serials: list[int | None] = [None] * len(writer.pages)
        for i, part in enumerate(parts):
            before = len(writer.pages)
            if part[0] == "pages":
                for page in part[1]:
                    writer.add_page(page)
            else:
                _, doc, row, pages = part
                _add_attached_pages(writer, doc, row, letterhead_page, signature, seal, pages)
            serials += [part_serial.get(i)] * (len(writer.pages) - before)
        stamp_page_numbers(writer, len(writer.pages), serials)
        for page in _render(_internal_notes_story(tender, checklist, drafting_note, s, qa), False):
            writer.add_page(page)

        out = io.BytesIO()
        writer.write(out)
        return out.getvalue()
    finally:
        writer.close()
