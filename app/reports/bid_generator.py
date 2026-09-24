"""Generates a single, downloadable "bid pack" PDF for one tender: an
AI-drafted set of individual bid documents (covering letter, non-
blacklisting declaration, etc. - see app.intelligence.bid_drafter) and the
verified company profile, all on the company letterhead; the company's own
uploaded enclosures (see app.api.main's /documents - GridFS,
get_company_documents_bucket) merged in after them; and, last, an internal
review checklist (open items, requirement-to-evidence compliance matrix)
to be removed before submission. Reference material like the company
brochure is never enclosed - it only feeds the drafter background text
(see company_background_text).

Same non-negotiable rule as app.intelligence.document_summarizer and
app.processing.eligibility: never invent evidence. Every fact about the
tender comes from the `tenders` collection (title/org/dates/amounts) and
its AI-extracted `document_summary` (see app.intelligence.document_summarizer);
every fact about the company comes from config/company_profile.json (see
app.config.load_company_profile - transcribed from actual certificates) or
from a document the user has actually uploaded to the documents library.
Anything a tender's eligibility criteria ask for that isn't backed by
either source is listed as an open item, never guessed.

This module itself does no AI drafting and has no dependency on it - it
only renders whatever `drafted_documents` its caller (app.api.main) hands
it, falling back to a plain templated covering letter when that's None (no
AI provider configured, or app.intelligence.bid_drafter's call failed) -
see _fallback_covering_letter_section. That keeps this module's own tests
fast/offline and keeps a bid pack always produceable even without an
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
from typing import TYPE_CHECKING, Any

from pypdf import PdfReader, PdfWriter
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)

if TYPE_CHECKING:
    from app.intelligence.bid_drafter import DraftedDocument

DISCLAIMER = (
    "This is an internally generated draft, compiled automatically from the tender's "
    "own documents and the company's verified profile/document library. It is NOT a "
    "submitted bid and must not be treated as one. Every row marked “TO BE FILLED FROM "
    "COMPANY RECORDS” or “TO BE VERIFIED BEFORE SIGNING” below is a genuine gap - "
    "resolve it, and have the authorized signatory review and sign the final package, "
    "before anything is submitted to the tendering authority."
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


def _build_tender_requirement_rows(
    document_summary: dict[str, Any],
    company_documents: list[CompanyDocumentRef],
) -> list[ComplianceRow]:
    rows: list[ComplianceRow] = []

    for req in document_summary.get("eligibility_requirements", []):
        matched = _match_documents(req, [], company_documents)
        rows.append(
            ComplianceRow(
                requirement=req,
                source="Tender documents (AI-extracted)",
                status="Evidence available" if matched else "TO BE VERIFIED BEFORE SIGNING",
                evidence="; ".join(d.name for d in matched) if matched else "Not yet matched to an uploaded document",
            )
        )

    for tc in document_summary.get("eligibility_technical_criteria", []):
        matched = _match_documents(tc, [], company_documents)
        rows.append(
            ComplianceRow(
                requirement=tc,
                source="Tender technical eligibility (AI-extracted)",
                status="Evidence available" if matched else "TO BE VERIFIED BEFORE SIGNING",
                evidence="; ".join(d.name for d in matched) if matched else "Not yet matched to an uploaded document",
            )
        )

    for doc_name in document_summary.get("documents_to_submit", []):
        matched = _match_documents(doc_name, [], company_documents)
        rows.append(
            ComplianceRow(
                requirement=doc_name,
                source="Documents to submit (AI-extracted)",
                status="Evidence available" if matched else "TO BE FILLED FROM COMPANY RECORDS",
                evidence="; ".join(d.name for d in matched) if matched else "-",
            )
        )

    return rows


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


def _status_color(status: str):
    if status == "Evidence available":
        return colors.HexColor("#2f9e44")
    if status.startswith("Identifier verified"):
        return colors.HexColor("#1c7ed6")
    return colors.HexColor("#c92a2a")


def _compliance_table(items: list[dict[str, Any]], styles) -> Table:
    """One checklist section (see build_checklist) as a compliance matrix -
    the Done column reflects what the team has ticked off on the dashboard's
    checklist page (the base-14 fonts have no check-mark glyph, so "[x]")."""
    cell = ParagraphStyle("cell", parent=styles["BodyText"], fontSize=8.5, leading=11)
    header = ["Done", "Requirement", "Source", "Status", "Evidence / Notes"]
    data = [header]
    for item in items:
        status = item.get("status") or "-"
        data.append(
            [
                "[x]" if item.get("done") else "[ ]",
                Paragraph(_safe(item.get("requirement") or "-"), cell),
                Paragraph(_safe(item.get("source") or "-"), cell),
                Paragraph(_safe(status), ParagraphStyle("status", parent=cell, textColor=_status_color(status))),
                Paragraph(_safe(item.get("evidence") or "-"), cell),
            ]
        )
    table = Table(data, colWidths=[1.1 * cm, 6 * cm, 3 * cm, 3.2 * cm, 4.2 * cm], repeatRows=1)
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1c2430")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTSIZE", (0, 0), (-1, 0), 9),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c9d2db")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f4f6f8")]),
            ]
        )
    )
    return table


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


# --- Editable review checklist ------------------------------------------------
# The internal review checklist as data, stored on the tender (`checklist`)
# so the team can edit/tick/extend it on the dashboard before a bid pack is
# generated - the pack's checklist pages then print that saved version.

CHECKLIST_SECTIONS = (
    "open_items",
    "company_eligibility",
    "tender_requirements",
    "reference_docs",
    "not_enclosed",
    "missing_information",
)


def checklist_item(
    section: str,
    requirement: str,
    source: str = "",
    status: str = "",
    evidence: str = "",
    origin: str = "auto",
    done: bool = False,
) -> dict[str, Any]:
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


def build_checklist(
    tender: dict[str, Any],
    eligibility_criteria: list[dict[str, Any]],
    company_profile: dict[str, Any],
    company_documents: list[CompanyDocumentRef],
    selection: EnclosureSelection | None = None,
) -> dict[str, Any]:
    """Everything the internal review checklist lists, minus the AI-drafted
    documents' open items (those only exist once a bid pack is drafted - see
    merge_drafted_open_items). The tender snapshot/technical criteria table
    aren't included: they're read straight off the tender."""
    if selection is None:
        selection = select_enclosures(tender, company_documents, company_profile)
    summary = tender.get("document_summary") or {}
    items: list[dict[str, Any]] = []

    open_items = list(company_profile.get("to_be_verified", []))
    if company_profile.get("pan_derivation_note"):
        open_items.append(f"PAN {company_profile.get('pan', '-')}: {company_profile['pan_derivation_note']}")
    if company_profile.get("past_experience"):
        open_items.append("Past Experience table lists every project on record - keep only those relevant "
                          "to this tender's scope before submitting.")
    for d in selection.selected:
        if not _is_pdf(d):
            open_items.append(f"Attach separately (not a PDF, so not merged into this pack): {d.name} ({d.filename})")
    for d in selection.over_budget:
        open_items.append(f"Attach manually from Documents library (relevant, but merging it would exceed the "
                          f"bid pack size limit): {d.name} ({d.filename})")
    items += [checklist_item("open_items", text) for text in open_items]

    for r in build_compliance_matrix(eligibility_criteria, company_profile, company_documents):
        items.append(checklist_item("company_eligibility", r.requirement, r.source, r.status, r.evidence))
    for r in _build_tender_requirement_rows(summary, company_documents):
        items.append(checklist_item("tender_requirements", r.requirement, r.source, r.status, r.evidence))

    for d in company_documents:
        if is_reference_document(d, company_profile):
            items.append(checklist_item("reference_docs", f"{d.name} ({d.filename})"))
    for d in selection.not_relevant:
        items.append(checklist_item("not_enclosed", f"{d.name} ({d.filename})"))
    for text in summary.get("missing_information", []):
        items.append(checklist_item("missing_information", text))

    now = dt.datetime.now(dt.timezone.utc)
    return {"generated_at": now, "updated_at": now, "items": items}


def merge_drafted_open_items(
    checklist: dict[str, Any], drafted_documents: "list[DraftedDocument] | None"
) -> dict[str, Any]:
    """Swaps in the open items of this run's AI-drafted documents, replacing
    any from a previous run - user-edited/added items are left untouched,
    and a drafted item already ticked done stays done if it recurs."""
    previous = [i for i in checklist.get("items", []) if i.get("origin") == "ai_draft"]
    done_texts = {i.get("requirement") for i in previous if i.get("done")}
    kept = [i for i in checklist.get("items", []) if i.get("origin") != "ai_draft"]
    drafted = [
        checklist_item("open_items", text, origin="ai_draft", done=text in done_texts)
        for doc in drafted_documents or []
        for text in (f"{doc.title}: {item}" for item in doc.open_items)
    ]
    return {**checklist, "items": drafted + kept}


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


def _signature_block(
    company_profile: dict[str, Any],
    company_documents: list[CompanyDocumentRef],
    body: ParagraphStyle,
) -> list[Any]:
    """"For <company>", the signature/seal image pair (when those documents
    are in the library) and the printed name/designation/place - shared by
    every signed document in the pack, kept together on one page."""
    signatory = company_profile.get("authorized_signatory", {})
    flow: list[Any] = [
        Spacer(1, 6),
        Paragraph(f"For <b>{_safe(company_profile.get('legal_name', '[Company]'))}</b>", body),
    ]
    sig_img = _open_image_flowable(
        _find_document(company_documents, signatory.get("signature_document_name")), 3.5, 1.6
    )
    seal_img = _open_image_flowable(
        _find_document(company_documents, signatory.get("seal_document_name")), 2.8, 2.8
    )
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


# --- Submission sections (on the letterhead) -----------------------------------

def _fallback_covering_letter_section(
    tender: dict[str, Any],
    company_profile: dict[str, Any],
    company_documents: list[CompanyDocumentRef],
    body: ParagraphStyle,
    h1: ParagraphStyle,
) -> list[Any]:
    """A plain, templated covering letter - used only when no AI-drafted
    documents are available (see generate_bid_package's `drafted_documents`
    param), so a bid pack can always be produced even without an
    OPENAI_API_KEY or if app.intelligence.bid_drafter's call failed."""
    subject = _safe(tender.get("title") or "the above tender")
    ref_line = f" (Ref: {_safe(tender['tender_ref'])})" if tender.get("tender_ref") else ""
    flow: list[Any] = [
        _date_line(body),
        Paragraph("Covering Letter", h1),
        Paragraph(f"To,<br/>{_safe(tender.get('organisation') or '[Tendering Authority]')}", body),
        Spacer(1, 8),
        Paragraph(f"<b>Subject: Techno-Commercial Offer for “{subject}”{ref_line}</b>", body),
        Spacer(1, 8),
        Paragraph(
            f"Dear Sir/Madam,<br/><br/>"
            f"We, {_safe(company_profile.get('legal_name', '[Company]'))}, submit our offer for the above tender. "
            "We confirm that we have read and understood the tender document, including all terms, conditions, "
            "annexures and any corrigenda issued up to the bid submission date, and that our offer conforms to "
            "the tender's requirements.<br/><br/>"
            "The documents required by the tender are enclosed with this letter, as listed in the enclosures "
            "that follow.",
            body,
        ),
        Spacer(1, 16),
        Paragraph("Yours faithfully,", body),
    ]
    flow += _signature_block(company_profile, company_documents, body)
    flow.append(PageBreak())
    return flow


def _drafted_document_section(
    doc: "DraftedDocument",
    company_profile: dict[str, Any],
    company_documents: list[CompanyDocumentRef],
    body: ParagraphStyle,
    h1: ParagraphStyle,
) -> list[Any]:
    """Renders one AI-drafted document (app.intelligence.bid_drafter) as its
    own signed letterhead page(s): date, title, body paragraphs, then the
    same signature block every signed document in the pack uses. Its open
    items go to the internal checklist, not onto the page itself."""
    flow: list[Any] = [_date_line(body), Paragraph(_safe(doc.title), h1)]
    for para in doc.body_paragraphs:
        if para.strip():
            flow.append(Paragraph(_format_paragraph(para), body))
            flow.append(Spacer(1, 6))
    flow += _signature_block(company_profile, company_documents, body)
    flow.append(PageBreak())
    return flow


_HEADER_STYLE = [
    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#0b3a5b")),
    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
    ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#c9d2db")),
    ("VALIGN", (0, 0), (-1, -1), "TOP"),
]


def _company_profile_section(
    company_profile: dict[str, Any],
    company_documents: list[CompanyDocumentRef],
    body: ParagraphStyle,
    h1: ParagraphStyle,
    h2: ParagraphStyle,
    small: ParagraphStyle,
) -> list[Any]:
    value_cell = ParagraphStyle("value_cell", parent=body, fontSize=9, leading=12, alignment=TA_LEFT)
    label_cell = ParagraphStyle("label_cell", parent=value_cell, fontName="Helvetica-Bold")
    header_cell = ParagraphStyle("header_cell", parent=label_cell, textColor=colors.white, fontSize=8.5, leading=10)
    flow: list[Any] = [Paragraph("Company Profile", h1)]

    profile_rows = [
        ["Legal name", company_profile.get("legal_name", "-")],
        ["CIN", company_profile.get("cin", "-")],
        ["GSTIN", company_profile.get("gstin", "-")],
        ["PAN", company_profile.get("pan", "-")],
        ["Date of incorporation", company_profile.get("date_of_incorporation", "-")],
        ["Registered office", company_profile.get("registered_office", "-")],
        ["Correspondence address", company_profile.get("correspondence_address", "-")],
        ["Email", company_profile.get("email", "-")],
        ["Phone", company_profile.get("phone", "-")],
        ["Directors", "; ".join(f"{d['name']} ({d['designation']})" for d in company_profile.get("directors", []))],
    ]
    for reg in company_profile.get("registrations", []):
        profile_rows.append([reg.get("name", "Registration"), reg.get("number") or reg.get("certificate_no", "-")])
    profile_table = Table(
        [[Paragraph(_safe(label), label_cell), Paragraph(_safe(value), value_cell)] for label, value in profile_rows],
        colWidths=[4.8 * cm, 12 * cm],
    )
    profile_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#eef5fa")),
        ("GRID", (0, 0), (-1, -1), 0.3, colors.HexColor("#c9d2db")),
        ("VALIGN", (0, 0), (-1, -1), "TOP"),
    ]))
    flow.append(profile_table)

    if company_profile.get("certifications"):
        flow.append(Paragraph("Certifications", h2))
        cert_rows = [["Certification", "Certificate No.", "Valid Until"]]
        for cert in company_profile["certifications"]:
            cert_rows.append([cert.get("name", "-"), cert.get("certificate_no", "-"), cert.get("valid_until", "-")])
        cert_table = Table(cert_rows, colWidths=[7.8 * cm, 4.5 * cm, 4.5 * cm], repeatRows=1)
        cert_table.setStyle(TableStyle(_HEADER_STYLE + [("FONTSIZE", (0, 0), (-1, -1), 9)]))
        flow.append(cert_table)

    if company_profile.get("past_experience"):
        flow.append(Paragraph("Past Experience", h2))
        exp_rows = [[Paragraph(h, header_cell) for h in ("Client", "Financial Year", "Value (INR Lakh)", "Scope of Work")]]
        for exp in company_profile["past_experience"]:
            exp_rows.append([
                Paragraph(_safe(exp.get("client", "-")), small),
                exp.get("financial_year", "-"),
                f"{exp['value_inr_lakh']:.2f}" if isinstance(exp.get("value_inr_lakh"), (int, float)) else "-",
                Paragraph(_safe(exp.get("description", "-")), small),
            ])
        exp_table = Table(exp_rows, colWidths=[4 * cm, 2.2 * cm, 2.3 * cm, 8.3 * cm], repeatRows=1)
        exp_table.setStyle(TableStyle(_HEADER_STYLE + [("FONTSIZE", (0, 0), (-1, -1), 8.5)]))
        flow.append(exp_table)

    flow.append(PageBreak())
    return flow


def _enclosures_section(
    enclosures: list[CompanyDocumentRef],
    tender: dict[str, Any],
    body: ParagraphStyle,
    h1: ParagraphStyle,
) -> list[Any]:
    flow: list[Any] = [Paragraph("List of Enclosures", h1)]
    if tender.get("tender_ref"):
        flow.append(Paragraph(f"Tender Reference: {_safe(tender['tender_ref'])}", body))
        flow.append(Spacer(1, 6))
    rows = [["S. No.", "Document"]] + [[str(i), Paragraph(_safe(d.name), body)] for i, d in enumerate(enclosures, 1)]
    table = Table(rows, colWidths=[2 * cm, 14.8 * cm], repeatRows=1)
    table.setStyle(TableStyle(_HEADER_STYLE + [("ALIGN", (0, 0), (0, -1), "CENTER")]))
    flow.append(table)
    return flow


# --- Internal review checklist (plain pages, removed before submission) --------

def _internal_checklist_story(
    tender: dict[str, Any],
    checklist: dict[str, Any],
    drafting_note: str | None,
    styles,
    body: ParagraphStyle,
    h1: ParagraphStyle,
    h2: ParagraphStyle,
) -> list[Any]:
    document_summary = tender.get("document_summary") or {}
    sections: dict[str, list[dict[str, Any]]] = {s: [] for s in CHECKLIST_SECTIONS}
    for item in checklist.get("items", []):
        sections.setdefault(item.get("section"), []).append(item)

    def _bullet(item: dict[str, Any]) -> Paragraph:
        text = _rich(item.get("requirement") or "")
        if item.get("evidence"):
            text += f" <i>({_safe(item['evidence'])})</i>"
        return Paragraph(f"{'[x]' if item.get('done') else '•'} {text}", body)

    value_cell = ParagraphStyle("value_cell_int", parent=body, fontSize=9, leading=12)
    small = ParagraphStyle("small_int", parent=body, fontSize=8.5, textColor=colors.HexColor("#495057"))
    story: list[Any] = [
        Paragraph("INTERNAL REVIEW CHECKLIST - remove these pages before submission", ParagraphStyle(
            "banner", parent=h1, textColor=colors.HexColor("#c92a2a"))),
        Paragraph(_safe(tender.get("title") or "(untitled tender)"), styles["Heading3"]),
    ]
    if tender.get("source_url"):
        story.append(Paragraph(f"Source: {_safe(tender['source_url'])}", small))
    story.append(Spacer(1, 8))
    story.append(Paragraph(DISCLAIMER, ParagraphStyle(
        "disclaimer", parent=body, borderColor=colors.HexColor("#c92a2a"), borderWidth=0.6,
        borderPadding=6, backColor=colors.HexColor("#fff5f5"))))
    if drafting_note:
        story.append(Spacer(1, 10))
        story.append(Paragraph(f"<b>Note:</b> {_safe(drafting_note)}", ParagraphStyle(
            "draftnote", parent=body, textColor=colors.HexColor("#a16207"),
            borderColor=colors.HexColor("#facc15"), borderWidth=0.6, borderPadding=6,
            backColor=colors.HexColor("#fefce8"))))

    # Everything a human still has to act on, in one place.
    story.append(Paragraph("Open items before signing", h2))
    for item in sections["open_items"]:
        story.append(_bullet(item))
    if not sections["open_items"]:
        story.append(Paragraph("None recorded.", body))

    if sections["reference_docs"]:
        story.append(Paragraph("Used for drafting only (not enclosed)", h2))
        for item in sections["reference_docs"]:
            story.append(_bullet(item))

    if sections["not_enclosed"]:
        story.append(Paragraph("In the Documents library but not enclosed (didn't match this tender's "
                               "requirements - add manually if needed)", h2))
        for item in sections["not_enclosed"]:
            story.append(_bullet(item))

    # Tender snapshot
    story.append(Paragraph("Tender Snapshot", h2))
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
        story.append(Paragraph("Key dates", h2))
        for d in document_summary["key_dates"]:
            story.append(Paragraph(f"• {_safe(d)}", body))

    story.append(PageBreak())

    # Compliance matrix
    story.append(Paragraph("Requirement-to-Evidence Compliance Matrix", h1))
    if sections["company_eligibility"]:
        story.append(Paragraph("Company eligibility profile", h2))
        story.append(_compliance_table(sections["company_eligibility"], styles))
    if sections["tender_requirements"]:
        story.append(Paragraph("Tender-specific requirements (from this tender's own documents)", h2))
        story.append(_compliance_table(sections["tender_requirements"], styles))

    if document_summary.get("technical_criteria_table"):
        story.append(Paragraph("Detailed technical criteria (marks-based)", h2))
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
        story.append(Paragraph("Missing information flagged in the tender documents", h2))
        for item in sections["missing_information"]:
            story.append(_bullet(item))
    return story


def generate_bid_package(
    tender: dict[str, Any],
    eligibility_criteria: list[dict[str, Any]],
    company_profile: dict[str, Any],
    company_documents: list[CompanyDocumentRef],
    drafted_documents: "list[DraftedDocument] | None" = None,
    drafting_note: str | None = None,
    letterhead_image: str | Path | None = None,
    checklist: dict[str, Any] | None = None,
) -> bytes:
    """Builds the full bid pack PDF and returns it as bytes, in three parts:

    1. The submission set, on the company letterhead (`letterhead_image`,
       drawn full-page; plain pages when None): every drafted bid document
       (see app.intelligence.bid_drafter.draft_bid_documents) each signed on
       its own page(s), the company profile, and a list of enclosures. When
       `drafted_documents` is None (no AI provider, or that call failed - see
       `drafting_note`), a templated covering letter stands in so a pack can
       always be produced.
    2. The PDF enclosures themselves, merged in as-is - only the standard
       set plus documents matching this tender (see select_enclosures).
       Reference documents (the brochure etc. - see is_reference_document)
       and the signature/seal images are never enclosures.
    3. An internal review checklist on plain pages - disclaimer, open items,
       tender snapshot, compliance matrix - for the team to work through and
       then remove before submitting. Printed from `checklist` (the tender's
       saved, possibly hand-edited checklist - see build_checklist) when
       given, else built fresh with this run's drafted open items merged in.
    """
    styles = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=styles["BodyText"], fontSize=10.5, leading=15, alignment=TA_JUSTIFY)
    h1 = ParagraphStyle("h1", parent=styles["Heading1"], fontSize=15, alignment=TA_CENTER, spaceBefore=4,
                        spaceAfter=14, textColor=colors.HexColor("#0b3a5b"))
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontSize=12, spaceBefore=14, spaceAfter=8,
                        textColor=colors.HexColor("#0b3a5b"))
    small = ParagraphStyle("small", parent=body, fontSize=8.5, leading=11, alignment=TA_LEFT)
    title = f"Bid Pack - {tender.get('title') or tender.get('tender_ref') or 'Tender'}"

    selection = select_enclosures(tender, company_documents, company_profile)
    enclosures = selection.selected
    pdf_enclosures = [d for d in enclosures if _is_pdf(d)]

    # --- 1. Submission set, on the letterhead -------------------------------
    submission: list[Any] = []
    if drafted_documents:
        for doc in drafted_documents:
            submission += _drafted_document_section(doc, company_profile, company_documents, body, h1)
    else:
        submission += _fallback_covering_letter_section(tender, company_profile, company_documents, body, h1)
    submission += _company_profile_section(company_profile, company_documents, body, h1, h2, small)
    if enclosures:
        submission += _enclosures_section(enclosures, tender, body, h1)
    while submission and isinstance(submission[-1], PageBreak):
        submission.pop()
    submission_bytes = _build_pdf(submission, title, letterhead_image)

    # --- 3. Internal checklist, plain pages ---------------------------------
    if checklist is None:
        checklist = merge_drafted_open_items(
            build_checklist(tender, eligibility_criteria, company_profile, company_documents, selection),
            drafted_documents,
        )
    checklist_story = _internal_checklist_story(
        tender, checklist, drafting_note, styles, ParagraphStyle("ibody", parent=styles["BodyText"]), h1, h2,
    )
    checklist_bytes = _build_pdf(checklist_story, title, None)

    # --- Assemble: submission, 2. merged PDF enclosures, checklist ------------
    writer = PdfWriter()
    try:
        for page in PdfReader(io.BytesIO(submission_bytes)).pages:
            writer.add_page(page)
        for pdf_doc in pdf_enclosures:
            try:
                for page in PdfReader(io.BytesIO(pdf_doc.open_bytes())).pages:
                    writer.add_page(page)
            except Exception as exc:
                raise BidGenerationError(f"Could not merge enclosure '{pdf_doc.name}': {exc}") from exc
        for page in PdfReader(io.BytesIO(checklist_bytes)).pages:
            writer.add_page(page)
        out = io.BytesIO()
        writer.write(out)
        return out.getvalue()
    finally:
        writer.close()
