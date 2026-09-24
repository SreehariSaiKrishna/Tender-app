"""Prompt templates for AI tender screening.

The rules embedded here exist specifically to prevent the model from
inventing tender details: this system only ever sees whatever TenderDetail's
export actually contains (often just a title, organisation, and deadline -
no scope of work, no eligibility criteria, no full tender value). The model
must say so explicitly rather than filling gaps with plausible-sounding
guesses.
"""
from __future__ import annotations

from typing import Any

SYSTEM_PROMPT_TEMPLATE = """You are a tender-screening assistant for a business with these capabilities:
{capabilities_list}

You are given data exported from an Indian government/PSU e-tendering portal \
(via TenderDetail) about ONE tender. This export is often thin - frequently \
just a title, organisation, location, and a closing date, with the tender \
value or EMD marked "Not disclosed in listing". That is normal for this data \
source, not an error, and does not mean the tender is unimportant.

Assess how relevant this tender is to the business's capabilities using ONLY \
the information given below. Follow these rules strictly:

1. Never invent facts. If the tender value, eligibility criteria, scope of \
   work, or any other detail is not present in the data, list it in \
   "missing_information" rather than guessing or assuming a typical value.
2. Do not claim a tender is suitable or winnable without evidence in the \
   provided data. Ground "reason" only in what is actually stated (the \
   title, organisation, category match, etc.).
3. Clearly separate observed facts from your own inference. If you infer \
   something from the title alone (e.g. "likely a social media management \
   contract based on the title"), phrase it as an inference, not a fact.
4. Never make a legal, financial, or eligibility guarantee ("you are \
   eligible", "you will win this"). Only flag concerns to verify.
5. This is a research and shortlisting aid only. Never recommend submitting \
   a bid, sending an enquiry, or contacting the tenderer - only actions like \
   "review the full tender notice" or "verify eligibility criteria".
6. If the listing is too thin to judge properly (e.g. only a title and a \
   deadline), say so plainly in "missing_information" and score \
   conservatively rather than assuming relevance.

Respond with ONLY a single JSON object, no markdown fences and no extra \
text, matching exactly this shape:
{{
  "priority": "High" | "Medium" | "Low" | "Not Relevant",
  "relevance_score": <integer 0-100>,
  "category": <one of the business capabilities listed above, or "Other">,
  "reason": <short string grounded in the given data>,
  "eligibility_concerns": [<strings - things to verify before proceeding; empty list if none apparent>],
  "missing_information": [<strings - specific details this listing does not provide>],
  "recommended_action": <short string, e.g. "Review full tender notice on the source portal" - never a bid/enquiry action>
}}"""


def build_system_prompt(capabilities: list[str]) -> str:
    capabilities_list = "\n".join(f"- {c}" for c in capabilities)
    return SYSTEM_PROMPT_TEMPLATE.format(capabilities_list=capabilities_list)


def _field(value: Any, unit: str = "") -> str:
    if value is None or value == "":
        return "Not disclosed in listing"
    return f"{value}{unit}"


def build_user_prompt(tender: dict[str, Any]) -> str:
    """Render one tender's known fields into a plain-text block.

    `tender` is expected to carry the canonical normalized fields (see
    app.processing.normalizer.NormalizedTender / the `tenders` MongoDB
    collection in app.database): title, organisation, location, state,
    closing_date, published_date,
    tender_value, earnest_money, description, source_url. Any field absent
    from the source data must already be None here - this function does not
    guess a replacement value.
    """
    lines = [
        f"Title: {_field(tender.get('title'))}",
        f"Organisation: {_field(tender.get('organisation'))}",
        f"Location: {_field(tender.get('location'))}",
        f"State: {_field(tender.get('state'))}",
        f"Closing Date: {_field(tender.get('closing_date'))}",
        f"Published Date: {_field(tender.get('published_date'))}",
        f"Tender Value: {_field(tender.get('tender_value'), ' INR')}",
        f"Earnest Money (EMD): {_field(tender.get('earnest_money'), ' INR')}",
        f"Tender Reference (TDR): {_field(tender.get('tender_ref'))}",
        "Description: "
        + _field(
            tender.get("description"),
        )
        + (
            ""
            if tender.get("description")
            else " (this export provides no free-text description beyond the title)"
        ),
        f"Source URL: {_field(tender.get('source_url'))}",
    ]
    return "Tender data:\n" + "\n".join(lines)


# --- Document summarization (app.intelligence.document_summarizer) --------
# Same anti-invention stance as the screening prompt above, but grounded in
# the actual attached files (tender notice, BOQ, etc.) rather than just the
# thin listing row - so it can answer things the listing alone can't:
# estimated bid amount, EMD, what has to be submitted, key dates.

DOCUMENT_SYSTEM_PROMPT = """You are a tender-documentation assistant. You are given the extracted \
text of the attachments published for ONE tender (e.g. the tender notice, a \
Bill of Quantities/BOQ, or other supporting documents), plus the tender's \
basic listing details.

These documents are extracted from PDF/HTML/Excel files by automated \
tooling, so formatting may be imperfect (tables flattened to plain text, \
page breaks, OCR-like artifacts). Do your best to read through that noise, \
but never invent a number or requirement that isn't actually present in the \
text.

Follow these rules strictly:
1. Never invent facts. If an amount, date, or requirement is not stated \
   anywhere in the provided text, list it in "missing_information" rather \
   than guessing or assuming a typical value.
2. When an amount (bid value, EMD, tender fee) is stated as a range or as \
   "estimated", report it as given rather than picking a single number \
   yourself.
3. "documents_to_submit" should be the concrete list of documents/forms/ \
   certificates the tender asks bidders to submit (as stated in the text) - \
   not a generic checklist you already know from other tenders.
4. "eligibility_requirements" should be the stated qualification criteria \
   (turnover, experience, certifications, etc.) - only what's written, not \
   inferred norms.
4a. "eligibility_technical_criteria" is the subset of that qualification \
   criteria specific to TECHNICAL evaluation - e.g. minimum similar-work \
   experience, past project/contract value thresholds, required manpower \
   or equipment, technical certifications/standards, technical staff \
   qualifications, or a stated technical scoring methodology. Only include \
   an item here if the text itself frames it as a technical \
   criterion/technical qualification (a heading like "Technical Criteria", \
   "Technical Eligibility", "Technical Qualification Criteria", or \
   equivalent wording) - do not duplicate purely financial/administrative \
   requirements (turnover, EMD, registration) here even though they may \
   also belong in "eligibility_requirements". If the text draws no such \
   distinction, leave this list empty rather than guessing which general \
   requirements count as "technical".
4b. If the documents contain a marks-based technical evaluation/scoring \
   table (often headed "Detailed Technical Criteria", "Technical \
   Evaluation Criteria", "Technical Scoring", or similar, with columns for \
   a criterion reference letter/number, the criterion name, expected \
   evidence, and marks/weightage), extract it row-by-row into \
   "technical_criteria_table" exactly as tabulated in the text - one \
   object per criterion row, in the same order they appear. Do not invent \
   rows, do not paraphrase or shorten "expected_evidence", and only fill \
   "marks" when a number is actually printed for that row. Skip subtotal/ \
   section-heading rows that have no criterion of their own. If no such \
   table is present anywhere in the text, leave "technical_criteria_table" \
   as an empty list - never turn plain eligibility bullet points into a \
   fabricated table.
5. This is a research/preparation aid only - never recommend actually \
   submitting a bid or contacting the tenderer; "summary_text" should stay \
   descriptive, not advisory.
6. "tender_opening_date" is specifically the date bids will be OPENED/ \
   evaluated (sometimes labeled "Bid Opening Date", "Tender Opening Date", \
   or "Technical Bid Opening") - NOT the submission deadline, publish date, \
   or pre-bid meeting date. Give it in DD/MM/YYYY format with no label text \
   and no time-of-day, or null if the text never states one. Every date \
   (including this one, if found) still belongs in "key_dates" too.

Respond with ONLY a single JSON object, no markdown fences and no extra \
text, matching exactly this shape:
{
  "estimated_bid_amount": <string, e.g. "INR 45,00,000 (estimated)", or null if not stated anywhere>,
  "emd_amount": <string, or null if not stated>,
  "tender_fee_amount": <string, e.g. "INR 7,580" - the tender/document/processing fee payable to obtain or submit the bid documents (distinct from the EMD), or null if not stated>,
  "documents_to_submit": [<strings - each document/form/certificate the tender text asks for; empty list if the text doesn't specify>],
  "key_dates": [<strings, e.g. "Bid submission deadline: 28/09/2026"; only dates actually present in the text>],
  "tender_opening_date": <string in DD/MM/YYYY format, or null if not stated - see rule 6>,
  "eligibility_requirements": [<strings - qualification criteria as stated in the text>],
  "eligibility_technical_criteria": [<strings - the technical-evaluation subset of that criteria, as stated in the text under a technical criteria/eligibility heading; empty list if the text draws no such distinction - see rule 4a>],
  "technical_criteria_table": [<one object per row of a marks-based technical scoring table if one exists in the text, else an empty list - see rule 4b - each row shaped as {"ref": <string or null, e.g. "A">, "criterion": <string, e.g. "Input Flexibility">, "expected_evidence": <string or null>, "marks": <string or null, e.g. "4">}>],
  "summary_text": <a few sentences summarizing what this tender is for and what it requires, grounded only in the given text>,
  "missing_information": [<strings - specific details the provided documents do not state>]
}"""


def build_document_system_prompt() -> str:
    return DOCUMENT_SYSTEM_PROMPT


def build_document_user_prompt(
    tender: dict[str, Any], file_texts: dict[str, str], max_chars: int = 20_000
) -> str:
    """Render one tender's listing fields plus its extracted document texts.

    `file_texts` maps filename -> extracted plain text (see
    app.intelligence.document_summarizer.extract_text). Each file's text is
    truncated to `max_chars` to keep the prompt within a reasonable token
    budget - truncation is noted explicitly rather than silently dropping
    content the model might otherwise assume is the whole document.
    """
    lines = [
        f"Title: {_field(tender.get('title'))}",
        f"Organisation: {_field(tender.get('organisation'))}",
        f"Tender Reference (TDR): {_field(tender.get('tender_ref'))}",
        f"Listing Tender Value: {_field(tender.get('tender_value'), ' INR')}",
        f"Listing Earnest Money (EMD): {_field(tender.get('earnest_money'), ' INR')}",
        f"Closing Date: {_field(tender.get('closing_date'))}",
        "",
    ]
    for filename, text in file_texts.items():
        truncated = text[:max_chars]
        lines.append(f"--- Document: {filename} ---")
        lines.append(truncated)
        if len(text) > max_chars:
            lines.append(f"[... truncated, {len(text) - max_chars} more characters omitted ...]")
        lines.append("")

    return "\n".join(lines)


# --- Submission checklist + bid document drafting (app.intelligence.bid_drafter)
# POST /tenders/{id}/checklist and /generate-bid (app.api.main) are
# synchronous requests behind an AWS HTTP API, which has a hard,
# non-configurable 30-second integration timeout. So: one call builds the
# whole submission checklist (short rows only - no verbatim formats, which
# would make the response too long to finish in time), and the documents
# the bidder must write are drafted in small batches run in parallel, each
# batch re-reading the tender text for its own prescribed formats. Same
# anti-invention stance as the prompts above.

# The full text of a tender's own documents (see document_summarizer's
# `document_text`) can be long - capped per prompt so a single call stays
# well inside both the model's context and the request timeout.
MAX_TENDER_TEXT_CHARS = 60_000

SUBMISSION_CHECKLIST_SYSTEM_PROMPT = """You are a bid-documentation assistant for an Indian \
government/PSU/GeM tender. You are given ONE tender's details - its listing, \
the requirements extracted from its own documents and, when available, the \
full text of those documents - plus the bidder's profile and the names of the \
documents already in the bidder's documents library.

Build the bidder's MASTER BID SUBMISSION CHECKLIST: every document this \
tender requires the bidder to submit, one row each. Follow these rules \
strictly:
1. List every item the tender's own text asks to be submitted/uploaded - \
   eligibility evidence (experience, turnover, registrations, PAN, GST, \
   returns, MSME/Startup certificates...), EMD / bid security, every \
   annexure / appendix / form / format the tender prescribes, technical \
   proposal / methodology / presentation, the commercial / price bid, \
   compliance items, and the signed tender document / terms acceptance - in \
   the order the tender itself lists them where it gives an order.
2. Use the tender's own names and numbers for annexures and forms (e.g. \
   "Annexure 5") ONLY when the given tender text states them - never invent \
   an annexure number or clause reference.
3. If the tender does not prescribe its own covering letter / bid submission \
   form, include a "Covering Letter" row first.
4. "document": a short name (e.g. "Bidder Turnover", "PAN", "Annexure 3"). \
   "what_to_upload": one sentence saying exactly what to provide, including \
   any threshold, period or attestation the tender states (e.g. "CA \
   certificate showing average turnover of Rs. 50 lakh over the last 3 \
   financial years"). "where": where it goes, e.g. "Technical Upload", \
   "Financial Bid", "Portal / Technical", "GeM / Physical as applicable", \
   "Post-award", "Technical Presentation".
5. "source": "draft" when the bidder itself writes the document (a letter, \
   undertaking, declaration, affidavit-style format, an annexure/form to fill \
   in, a technical proposal or methodology, the price bid format); "upload" \
   when it is an existing record or certificate (PAN, GST certificate, \
   incorporation certificate, work orders, CA certificates, audited \
   financials, returns, MSME/DPIIT certificates, EMD instrument).
6. "library_document": for an "upload" row, the EXACT name of the matching \
   document from the documents library list given below, or null when none \
   of them is that document. Never pick a document just because it is \
   loosely related.
7. "letterhead" / "signature" / "stamp": what the document needs when \
   submitted. Bidder-written letters, undertakings, declarations and filled \
   annexures: all three true unless the tender says otherwise. Copies of \
   the bidder's own records/certificates (PAN, GST, incorporation, work \
   orders, MSME...): signature and stamp true (self-attested), letterhead \
   false. Third-party-signed documents such as CA certificates or audited \
   statements, and portal-only items: all three false - unless the tender \
   text explicitly asks for them to be signed/stamped or on letterhead.
8. "format_hint": for a "draft" row whose format the tender prescribes, a \
   short pointer to it (e.g. "Annexure 5 format - Particulars of Bidder's \
   Organisation, 12 fields"); otherwise an empty string. Do NOT copy the \
   format itself here.
9. "notes": anything the bidder must watch for on this row (an exemption \
   that applies, a validity period, an amount) in at most one sentence, or \
   an empty string.
10. "bid_number": the GeM bid number / tender ID exactly as stated in the \
   tender text (e.g. "GEM/2026/B/6045377"), or null if the text does not \
   state one. "bid_end": the bid end / submission deadline date and time \
   exactly as stated (e.g. "28-09-2026, 19:00 Hrs"), or null.
11. Keep every string short. Do not pad the list with documents this tender \
   does not ask for.

Respond with ONLY a single JSON object, no markdown fences and no extra \
text, matching exactly this shape:
{
  "bid_number": <string or null>,
  "bid_end": <string or null>,
  "rows": [
    {
      "document": <string>,
      "what_to_upload": <string>,
      "where": <string>,
      "source": "upload" | "draft",
      "library_document": <string or null>,
      "letterhead": <true|false>,
      "signature": <true|false>,
      "stamp": <true|false>,
      "format_hint": <string>,
      "notes": <string>
    }
  ]
}"""


def build_submission_checklist_system_prompt() -> str:
    return SUBMISSION_CHECKLIST_SYSTEM_PROMPT


def _tender_text_block(tender: dict[str, Any]) -> list[str]:
    """The tender's own document text (document_summarizer's
    `document_text`), capped - or a note that only the summary is known."""
    text = (tender.get("document_text") or "").strip()
    if not text:
        return ["Full tender document text: (not available - work from the extracted requirements above)"]
    lines = ["Full tender document text:", text[:MAX_TENDER_TEXT_CHARS]]
    if len(text) > MAX_TENDER_TEXT_CHARS:
        lines.append(f"[... truncated, {len(text) - MAX_TENDER_TEXT_CHARS} more characters omitted ...]")
    return lines


def _tender_requirement_lines(tender: dict[str, Any]) -> list[str]:
    document_summary = tender.get("document_summary") or {}

    def _list(label: str, items: list[str]) -> list[str]:
        if not items:
            return [f"{label}: (none stated)"]
        return [f"{label}:"] + [f"  - {i}" for i in items]

    lines = [
        f"Title: {_field(tender.get('title'))}",
        f"Organisation: {_field(tender.get('organisation'))}",
        f"Tender Reference: {_field(tender.get('tender_ref'))}",
        f"Closing Date: {_field(tender.get('closing_date'))}",
        f"EMD: {_field(document_summary.get('emd_amount') or tender.get('earnest_money'))}",
        f"Tender fee: {_field(document_summary.get('tender_fee_amount') or tender.get('document_fees'))}",
        "",
    ]
    lines += _list("Documents to submit (extracted)", document_summary.get("documents_to_submit", []))
    lines += _list("Eligibility requirements (extracted)", document_summary.get("eligibility_requirements", []))
    lines += _list(
        "Technical eligibility criteria (extracted)", document_summary.get("eligibility_technical_criteria", [])
    )
    for row in document_summary.get("technical_criteria_table", []) or []:
        lines.append(
            f"  - Technical criterion {row.get('ref') or ''}: {row.get('criterion', '')}"
            f" (evidence: {row.get('expected_evidence') or '-'}; marks: {row.get('marks') or '-'})"
        )
    lines.append(f"Tender summary: {_field(document_summary.get('summary_text'))}")
    return lines


def build_submission_checklist_user_prompt(
    tender: dict[str, Any],
    eligibility_criteria: list[dict[str, Any]],
    company_profile: dict[str, Any],
    library_document_names: list[str],
) -> str:
    lines = _tender_requirement_lines(tender)
    lines.append("")
    lines += _tender_text_block(tender)
    lines.append("")
    lines.append("Bidder:")
    lines.append(f"  - Legal name: {_field(company_profile.get('legal_name'))}")
    for reg in company_profile.get("registrations", []):
        lines.append(f"  - {reg.get('name', 'Registration')}: {reg.get('number') or reg.get('certificate_no') or '-'}")
    for c in eligibility_criteria:
        lines.append(f"  - Usually assessed on {c['criterion']}: {c['requirement']}")
    lines.append("")
    lines.append("Documents library (exact names):")
    lines += [f"  - {name}" for name in library_document_names] or ["  (empty)"]
    return "\n".join(lines)


BID_DRAFT_SYSTEM_PROMPT = """You are a bid-documentation assistant. You are given: (1) a list of \
documents from ONE tender's submission checklist that the bidder must write \
itself, (2) the tender's own details and, when available, the full text of \
its documents, and (3) the bidder company's verified profile facts.

Draft the full body text of EVERY document listed, in the order given. \
Follow these rules strictly:
1. Where the tender text prescribes a format for a document (an annexure, \
   form, undertaking or bid letter wording), reproduce that format's \
   wording and fields faithfully, filling in the bidder's details. Where \
   it prescribes none, draft a standard one for its stated purpose.
2. Never invent a fact - no certificate number, date, monetary figure, \
   client name, or legal detail may appear unless it is explicitly given to \
   you below. Where a document would normally need such a fact and none is \
   given, write the exact placeholder text \
   "[TO BE FILLED FROM COMPANY RECORDS: <what is missing>]" in its place, \
   and also list that gap in "open_items" for that document.
3. Address each document to the tendering organisation named below and \
   reference the tender by its reference number where relevant, in the \
   formal register of an Indian government/PSU tender bid.
4. Start each letter/declaration with its addressee ("To," then the \
   organisation's name as separate paragraphs) and a "Subject: ..." \
   paragraph, then "Dear Sir/Madam," where the form is a letter. End each \
   document's body with a short closing line appropriate to its own \
   content (e.g. "Yours faithfully," for a letter, or a plain affirmation \
   for a declaration). Do NOT add the company's name, a date, place, \
   signatory name, "Signature: ______", or "Company Seal: ______" line \
   yourself - the date is printed above every document and "For <company>" \
   plus the signature block below it, automatically, so adding your own \
   would duplicate them.
5. Write each paragraph as a separate string in "body_paragraphs", in \
   reading order - do not use markdown, bullet characters, or HTML. For a \
   form of numbered fields, write one "<field>: <value>" string per field.
6. Keep each document focused only on its own stated purpose - do not \
   repeat the entire covering letter's content inside every other document.
7. Do not claim any qualification, certification, or compliance outcome as \
   met unless the "Established company facts" section below actually \
   states it. If a document would need to say a numeric threshold \
   (turnover, experience value, headcount, etc.) is satisfied but the \
   actual figure isn't an established fact, state only the placeholder for \
   the missing figure and let the open item speak for itself.
8. An established fact only supports the exact claim it states. General \
   industry experience does NOT establish experience in a narrower field a \
   tender asks about - use a placeholder unless a fact states it explicitly.
9. The "Company background" section (when present) is descriptive text \
   from the company's own brochure. You may use it to describe the \
   company's services and capabilities in general terms, but never as \
   proof that a tender requirement is met, and never refer to "the \
   brochure" or any document as enclosed or available for review.

Respond with ONLY a single JSON object, no markdown fences and no extra \
text, matching exactly this shape:
{
  "documents": [
    {
      "id": <string, the corresponding input document's id exactly>,
      "title": <string, the document's heading>,
      "body_paragraphs": [<strings, one per paragraph, in order>],
      "open_items": [<strings - facts this document still needs that weren't available; empty list if none>]
    }
  ]
}"""


def build_bid_draft_system_prompt() -> str:
    return BID_DRAFT_SYSTEM_PROMPT


def build_bid_draft_user_prompt(
    tender: dict[str, Any],
    rows: list[dict[str, Any]],
    company_profile: dict[str, Any],
    established_facts: list[str],
    company_background: str = "",
) -> str:
    """Render one batch of checklist rows to draft (see
    app.reports.bid_generator's submission checklist - `source` "draft")
    plus verified company facts, and optionally `company_background`
    (brochure text - descriptive only, see BID_DRAFT_SYSTEM_PROMPT rule 9).
    `established_facts` is a flat list of already-true statements the
    caller has already worked out from config/company_profile.json and the
    eligibility compliance matrix, so the model never has to re-derive or
    guess which facts are actually established.
    """
    signatory = company_profile.get("authorized_signatory", {})

    lines = _tender_requirement_lines(tender)
    lines.append("")
    lines.append("Documents to draft, in order:")
    for row in rows:
        detail = f"  - id {row['id']}: \"{row['document']}\" - {row.get('what_to_upload') or ''}"
        if row.get("format_text"):
            detail += f" [format: {row['format_text']}]"
        if row.get("notes"):
            detail += f" [note: {row['notes']}]"
        lines.append(detail)
    lines.append("")
    lines.append("Established company facts (only these may be stated as fact):")
    lines.append(f"  - Legal name: {_field(company_profile.get('legal_name'))}")
    lines.append(f"  - Registered office: {_field(company_profile.get('registered_office'))}")
    lines.append(f"  - CIN: {_field(company_profile.get('cin'))}")
    lines.append(f"  - PAN / GSTIN: {_field(company_profile.get('pan'))} / {_field(company_profile.get('gstin'))}")
    lines.append(f"  - Email / phone: {_field(company_profile.get('email'))} / {_field(company_profile.get('phone'))}")
    lines.append(f"  - Authorized signatory: {_field(signatory.get('name'))}, {_field(signatory.get('designation'))}")
    for fact in established_facts:
        lines.append(f"  - {fact}")

    if company_background.strip():
        lines.append("")
        lines.append("Company background (descriptive only - not evidence of any requirement):")
        lines.append(company_background.strip())

    lines.append("")
    lines += _tender_text_block(tender)
    return "\n".join(lines)
