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


# --- Bid document drafting (app.intelligence.bid_drafter) ------------------
# Two calls, not one per document - POST /tenders/{id}/generate-bid
# (app.api.main) is a synchronous request behind an AWS HTTP API, which has
# a hard, non-configurable 30-second integration timeout; one AI call per
# document (there can be several) risks blowing that budget. Call 1 decides
# which documents this tender's own paperwork calls for; call 2 drafts all
# of them together. Same anti-invention stance as the prompts above.

BID_PLAN_SYSTEM_PROMPT = """You are a bid-documentation assistant. You are given the details of ONE \
tender a business is preparing to bid on - its listing, the requirements \
extracted from its own tender documents, and the bidder's general \
eligibility profile.

Your only job here is to decide which individual documents the bidder needs \
to DRAFT AND SIGN as part of this submission - not which existing \
certificates/records need to be attached as evidence.

Follow these rules strictly:
1. Only propose documents that are letters, declarations, undertakings, or \
   short narrative certificates the bidder itself must write and sign for \
   this specific tender - e.g. a covering letter, a non-blacklisting \
   declaration, a no-deviation certificate, a manpower/deployment \
   declaration, an authorized-signatory declaration.
2. Never propose an item that is really just an existing record to attach \
   as-is - e.g. "PAN Card", "GST Registration Certificate", "Incorporation \
   Certificate", "Audited Financial Statements", "Company Brochure", "CA \
   Certificate". Those are evidence attachments, handled separately, not \
   documents to draft.
3. Always include exactly one "Covering Letter" as the first document.
4. Base every other proposed document on what this tender's own \
   requirements (documents to submit / eligibility requirements / \
   technical eligibility criteria, given below) actually ask for, or on \
   standard practice for a government/PSU tender bid ONLY when the \
   tender's own requirements are too thin to tell either way. Do not \
   invent tender-specific annexure numbers or clause references you have \
   not been given.
5. Propose at most 6 documents in total, ordered logically (covering \
   letter first).
6. If the given information is too thin to identify anything beyond a \
   covering letter, propose just the covering letter - do not pad the list.

Respond with ONLY a single JSON object, no markdown fences and no extra \
text, matching exactly this shape:
{
  "documents": [
    {
      "title": <short string, e.g. "Non-Blacklisting Declaration">,
      "purpose": <one sentence: what this document must establish and why this tender calls for it>,
      "kind": "covering_letter" | "declaration" | "undertaking" | "certificate_narrative" | "technical_note"
    }
  ]
}"""


def build_bid_plan_system_prompt() -> str:
    return BID_PLAN_SYSTEM_PROMPT


def build_bid_plan_user_prompt(tender: dict[str, Any], eligibility_criteria: list[dict[str, Any]]) -> str:
    """Render one tender's own requirements plus the bidder's general
    eligibility profile - enough context to decide which documents to draft,
    without yet drafting any of them (see build_bid_draft_user_prompt for
    that, which additionally needs verified company facts).
    """
    document_summary = tender.get("document_summary") or {}

    def _list(label: str, items: list[str]) -> list[str]:
        if not items:
            return [f"{label}: (none stated)"]
        return [f"{label}:"] + [f"  - {i}" for i in items]

    lines = [
        f"Title: {_field(tender.get('title'))}",
        f"Organisation: {_field(tender.get('organisation'))}",
        f"Tender Reference: {_field(tender.get('tender_ref'))}",
        "",
    ]
    lines += _list("Documents to submit (from the tender's own documents)", document_summary.get("documents_to_submit", []))
    lines.append("")
    lines += _list("Eligibility requirements (from the tender's own documents)", document_summary.get("eligibility_requirements", []))
    lines.append("")
    lines += _list("Technical eligibility criteria (from the tender's own documents)", document_summary.get("eligibility_technical_criteria", []))
    lines.append("")
    lines.append(f"Tender summary: {_field(document_summary.get('summary_text'))}")
    lines.append("")
    lines.append("Bidder's general eligibility profile (criteria this bidder is normally assessed against):")
    for c in eligibility_criteria:
        lines.append(f"  - {c['criterion']}: {c['requirement']}")

    return "\n".join(lines)


BID_DRAFT_SYSTEM_PROMPT = """You are a bid-documentation assistant. You are given: (1) a list of \
documents already identified as needed for ONE tender submission, (2) the \
tender's own known details, and (3) the bidder company's verified profile \
facts and which of its eligibility criteria already have evidence on file.

Draft the full body text of EVERY document listed, in the order given. \
Follow these rules strictly:
1. Never invent a fact - no certificate number, date, monetary figure, \
   client name, or legal detail may appear unless it is explicitly given to \
   you below. Where a document would normally need such a fact and none is \
   given, write the exact placeholder text \
   "[TO BE FILLED FROM COMPANY RECORDS: <what is missing>]" in its place, \
   and also list that gap in "open_items" for that document.
2. Address each document to the tendering organisation named below and \
   reference the tender by its reference number where relevant, in the \
   formal register of an Indian government/PSU tender bid.
3. Start each letter/declaration with its addressee ("To," then the \
   organisation's name as separate paragraphs) and a "Subject: ..." \
   paragraph, then "Dear Sir/Madam," where the form is a letter. End each \
   document's body with a short closing line appropriate to its own \
   content (e.g. "Yours faithfully," for a letter, or a plain affirmation \
   for a declaration). Do NOT add the company's name, a date, place, \
   signatory name, "Signature: ______", or "Company Seal: ______" line \
   yourself - the date is printed above every document and "For <company>" \
   plus the signature block below it, automatically, so adding your own \
   would duplicate them.
4. Write each paragraph as a separate string in "body_paragraphs", in \
   reading order - do not use markdown, bullet characters, or HTML.
5. Keep each document focused only on its own stated purpose - do not \
   repeat the entire covering letter's content inside every other document.
6. Do not claim any qualification, certification, or compliance outcome as \
   met unless the "Established company facts" section below actually \
   states it. This applies even when only the exact figure is missing: if \
   a document would need to say a numeric threshold (turnover, experience \
   value, headcount, etc.) is satisfied but the actual figure isn't an \
   established fact, do NOT write that the requirement "is met" or \
   "meets the minimum" - state only the placeholder for the missing \
   figure (e.g. "Our audited average annual turnover for the relevant \
   years is [TO BE FILLED FROM COMPANY RECORDS: turnover figures], as \
   certified by our Chartered Accountant.") and let the open item speak \
   for itself, rather than asserting the conclusion the missing number \
   would need to support.
7. An established fact only supports the exact claim it states. General \
   industry experience (e.g. "years in EdTech/IT") does NOT establish \
   experience in a narrower field a tender asks about (e.g. social media \
   management, digital marketing) - for those, use a placeholder unless a \
   fact below states that field explicitly.
8. The "Company background" section (when present) is descriptive text \
   from the company's own brochure. You may use it to describe the \
   company's services, capabilities and track record in general terms, \
   but never as proof that a tender requirement is met, and never refer \
   to "the brochure" or any document as enclosed or available for review.

Respond with ONLY a single JSON object, no markdown fences and no extra \
text, matching exactly this shape:
{
  "documents": [
    {
      "title": <string, matching the corresponding input document's title exactly>,
      "body_paragraphs": [<strings, one per paragraph, in order>],
      "open_items": [<strings - facts this document still needs that weren't available; empty list if none>]
    }
  ]
}"""


def build_bid_draft_system_prompt() -> str:
    return BID_DRAFT_SYSTEM_PROMPT


def build_bid_draft_user_prompt(
    tender: dict[str, Any],
    planned_documents: list[dict[str, Any]],
    company_profile: dict[str, Any],
    established_facts: list[str],
    company_background: str = "",
) -> str:
    """Render the plan (from build_bid_plan_user_prompt's response) plus
    verified company facts, and optionally `company_background` (brochure
    text - descriptive only, see BID_DRAFT_SYSTEM_PROMPT rule 8) - `established_facts` is a flat list of
    already-true statements (e.g. "Legal Status: evidence available (CIN
    U85499TS2025PTC199477)") the caller has already worked out from
    config/company_profile.json and the eligibility compliance matrix (see
    app.reports.bid_generator._build_compliance_matrix), so the model never
    has to re-derive or guess which facts are actually established.
    """
    document_summary = tender.get("document_summary") or {}
    signatory = company_profile.get("authorized_signatory", {})

    lines = [
        f"Title: {_field(tender.get('title'))}",
        f"Organisation: {_field(tender.get('organisation'))}",
        f"Tender Reference: {_field(tender.get('tender_ref'))}",
        f"Tender summary: {_field(document_summary.get('summary_text'))}",
        "",
        "Documents to draft, in order:",
    ]
    for i, doc in enumerate(planned_documents, start=1):
        lines.append(f"  {i}. \"{doc['title']}\" ({doc['kind']}) - {doc['purpose']}")
    lines.append("")
    lines.append("Established company facts (only these may be stated as fact):")
    lines.append(f"  - Legal name: {_field(company_profile.get('legal_name'))}")
    lines.append(f"  - Registered office: {_field(company_profile.get('registered_office'))}")
    lines.append(f"  - Authorized signatory: {_field(signatory.get('name'))}, {_field(signatory.get('designation'))}")
    for fact in established_facts:
        lines.append(f"  - {fact}")

    if company_background.strip():
        lines.append("")
        lines.append("Company background (descriptive only - not evidence of any requirement):")
        lines.append(company_background.strip())

    return "\n".join(lines)
