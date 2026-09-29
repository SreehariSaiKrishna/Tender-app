"""Prompt templates for AI tender screening.

The rules embedded here exist specifically to prevent the model from
inventing tender details: this system only ever sees whatever TenderDetail's
export actually contains (often just a title, organisation, and deadline -
no scope of work, no eligibility criteria, no full tender value). The model
must say so explicitly rather than filling gaps with plausible-sounding
guesses.
"""
from __future__ import annotations

import datetime as dt
import re
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
tender requires the bidder to submit, one row each. Work as the bidder \
preparing this bid would - an evaluator rejects a bid for any eligibility \
condition it does not prove, whether or not the tender repeats that proof in \
a "documents to submit" list. Follow these rules strictly:
1. Work in two passes over the full tender text.
   A. Explicit items: every item the tender's own text asks to be \
   submitted/uploaded - EMD / bid security, every annexure / appendix / \
   form / format the tender prescribes, technical proposal / methodology / \
   presentation, the commercial / price bid, compliance items, and the \
   signed tender document / terms acceptance - in the order the tender \
   itself lists them where it gives an order.
   B. Condition by condition: go through EVERY eligibility / qualification \
   / pre-qualification condition, technical evaluation or marking \
   criterion, personnel / resource requirement and bidder obligation in \
   the text, one at a time. For each one that has to be proved, add the \
   document that proves it - or make sure a row from pass A already does. \
   The extracted summary lists may be incomplete and scanned pages may \
   have lost their numbering (e.g. ". Legal Status:"), so read the text \
   itself. Typical proof, used only for conditions this tender actually \
   states: legal status -> Certificate of Incorporation / partnership \
   deed / LLP registration; business activity -> company profile and the \
   relevant work orders; minimum years of experience -> incorporation \
   certificate and the earliest relevant work order; N similar projects -> \
   the work orders (and completion / performance certificates where the \
   tender asks for them), plus a project experience summary sheet - see \
   rule 14 for how work orders become rows; turnover / net worth -> CA certificate and audited \
   financial statements for the stated years; PAN / GST -> copies; \
   statutory compliance -> the registrations it names (e.g. EPF, ESI, \
   labour licence when manpower is supplied); blacklisting / debarment \
   clause -> self-declaration of non-blacklisting; availability, \
   replacement or on-site deployment of resources -> an undertaking for \
   each; capability claims (e.g. vendor coordination) -> a list of \
   relevant projects; personnel with stated qualifications / experience -> \
   a "List of Proposed Personnel" row (role, name, qualification, \
   experience) plus one CV row PER ROLE the tender names (e.g. "CV - \
   Programmer") covering that person's education and experience \
   certificates; acceptance of terms -> signed tender document / \
   acceptance letter; an EMD or fee exemption -> the MSME / DPIIT \
   certificate that grants it.
   A condition is proved only by a document that actually shows it - e.g. \
   turnover by a CA certificate / audited financials, never by an MSME or \
   Startup certificate.
   Never write a catch-all row such as "Relevant certificates and \
   documents", "Scanned copies of certificates", "Eligibility and \
   qualification details" or "Supporting documents" - even when the \
   tender itself uses such a phrase, list each specific document it \
   covers as its own row.
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
7. "letterhead" / "signature" / "stamp": the usual marks are applied \
   automatically - bidder-written documents on letterhead, signed and \
   stamped; copies of the bidder's own records signed and stamped \
   (self-attested); third-party-signed documents (CA certificates, audited \
   statements) and portal items untouched. Include these keys on a row \
   ONLY when the tender text explicitly asks for something different.
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
11. "basis": the tender clause this row answers, as a short reference in \
   the tender's own terms (e.g. "Eligibility 6 - Financial Capacity", \
   "Sec 4 - Resource CVs", "NIT - EMD"). Every row needs one.
12. Keep every string short, and set "format_hint" only on "draft" rows. \
   Add no row that no stated condition or instruction calls for - but every \
   condition that has to be proved MUST be covered.

13. "conditions": write this FIRST - pass B's worklist. One entry per \
   eligibility condition, evaluation criterion, personnel requirement and \
   bidder obligation found in the text, in the tender's order (a tender \
   with 16 numbered eligibility conditions gets at least 16 entries). \
   "ref": at most 6 words (e.g. "Elig 6 Financial Capacity") - never copy \
   the clause text. "proved_by": the "document" name(s) of the row(s) that \
   prove it, or "" only if it needs no document at all (e.g. the \
   department's own right to verify). Every name in "proved_by" must \
   appear as a row.
14. ONE ROW PER DISTINCT DOCUMENT. A document that proves several \
   conditions (e.g. a work order proving business activity, years of \
   experience AND similar projects; the incorporation certificate proving \
   legal status AND age) is ONE row - list it in "proved_by" of every \
   condition it proves, and put all those clauses in its "basis" (e.g. \
   "Elig 2 Legal Status; Elig 5 Years of Experience"). No two rows may \
   name the same "library_document", and no two rows may have the same \
   "document" name. Work orders: one row PER PROJECT, never a generic \
   "Work Orders" row - "document" is "Work Order - <client short name> \
   (<short scope relevant to this tender>)", e.g. "Work Order - MP CAEC \
   (Social Media & Digital Campaign)", and "library_document" is that \
   project's own library document from "Bidder projects" below. Choose \
   projects by how closely their scope matches THIS tender's scope of work \
   (not by value): when the tender asks for N similar works, the N most \
   relevant projects that qualify; otherwise every relevant project. Order \
   them most relevant first. Never add a project the bidder profile does \
   not list, and never use one project's work order for another.
15. "section": the group each row belongs to, exactly one of: "Covering \
   Letter / Bid Form", "Prescribed Annexures", "Legal & Statutory" \
   (incorporation, PAN, TAN, GST, MSME/Udyam, DPIIT, MCA records, other \
   registrations and certifications), "Financial" (CA turnover certificate, \
   audited financial statements, ITR, net worth, bank solvency), \
   "Experience" (project experience summary, then work orders, completion \
   certificates), "Declarations & Undertakings", "Technical Proposal", \
   "Signed Tender / Acceptance", "EMD / Financial Bid". "tender_order": \
   true ONLY when the tender text itself prescribes the order in which its \
   documents must be submitted/uploaded - then list rows in exactly that \
   order. Otherwise false, and list the rows in the section order above \
   (within Experience: the summary first, then work orders by relevance).
16. "notary": true ONLY when the tender text requires THIS document to be \
   notarised, sworn or attested - an affidavit, a declaration on \
   non-judicial stamp paper, attestation by a notary / oath commissioner / \
   magistrate, "notarized copy". Otherwise false (leave it out). Ordinary \
   self-attested copies (PAN, GST, certificates) and letters on letterhead \
   are never notary rows unless the tender says so.
17. When the tender text given does not state its own requirements (only \
   a title, a link or a summary), build the standard Indian government bid \
   set for a tender of this kind, driven by the "Usually assessed on" \
   conditions: Covering Letter; incorporation, PAN, GST, MSME/Udyam, DPIIT \
   and the relevant certifications; CA turnover certificate and audited \
   financial statements; a Project Experience Summary plus the work orders \
   of the relevant projects; a Company Profile; the self-declaration of \
   non-blacklisting; and the signed tender / acceptance letter - noting in \
   the Covering Letter row's "notes" that the tender's own documents were \
   not available and must be checked for anything more.
18. "applicable": false for a row the tender itself makes conditional \
   ("if applicable", "for manufacturers", "for joint ventures", "for \
   foreign bidders") when that condition does not fit this bidder - e.g. a \
   Manufacturer's Authorisation / Authorized Dealer certificate when the \
   bidder is a service provider supplying no manufactured goods. Keep such \
   a row in the list (with "notes" saying why it doesn't apply) so the \
   evaluator sees it was considered; it is marked Not applicable.
19. EMD / tender fee EXEMPTION: when the tender exempts MSE / MSME (or \
   DPIIT start-up) bidders from the EMD or the tender fee and the bidder \
   holds that registration (see "Bidder"), do NOT list a payment proof for \
   it: list a "draft" row "Request for EMD Exemption (MSME)" (or "... \
   Tender Fee Exemption ...") claiming the exemption under the tender's \
   clause, with "notes" naming the Udyam / DPIIT certificate enclosed in \
   Legal & Statutory as its proof (never a second row for that \
   certificate). List a payment proof only when no exemption applies - \
   "where" then names the portal payment.

Respond with ONLY a single JSON object, no markdown fences and no extra \
text, as compact JSON (no indentation or line breaks). Leave out any row \
key whose value would be an empty string, null or false. The shape:
{
  "bid_number": <string or null>,
  "bid_end": <string or null>,
  "tender_order": <true|false>,
  "conditions": [
    {"ref": <string>, "proved_by": <string>}
  ],
  "rows": [
    {
      "document": <string>,
      "what_to_upload": <string>,
      "where": <string>,
      "source": "upload" | "draft",
      "section": <string>,
      "basis": <string>,
      "library_document": <string, optional>,
      "format_hint": <string, optional>,
      "notes": <string, optional>,
      "notary": <true, optional>,
      "applicable": <false, optional>,
      "letterhead": <true|false, optional>,
      "signature": <true|false, optional>,
      "stamp": <true|false, optional>
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
        "The extracted lists below are from an AI summary and may be incomplete - "
        "the full tender document text is authoritative.",
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


def _tender_scope_text(tender: dict[str, Any]) -> str:
    """What the tender is about, lower-cased - its title, summary and
    extracted requirements plus the start of its own document text (where
    the scope of work usually sits)."""
    summary = tender.get("document_summary") or {}
    parts = [tender.get("title") or "", summary.get("summary_text") or ""]
    for key in ("documents_to_submit", "eligibility_requirements", "eligibility_technical_criteria"):
        parts += [str(i) for i in summary.get(key, []) or []]
    parts.append((tender.get("document_text") or "")[:20_000])
    return " ".join(parts).lower()


_WORD_RE = re.compile(r"[a-z][a-z0-9&+-]{3,}")
_STOP_WORDS = {
    "with", "from", "that", "this", "each", "including", "such", "their", "other", "services", "service",
    "digital", "development", "design", "government", "project", "projects", "based", "per", "shall", "work",
}


def rank_past_experience(company_profile: dict[str, Any], tender: dict[str, Any]) -> list[dict[str, Any]]:
    """company_profile.json's projects, most relevant to this tender first -
    by how many of a project's tags (weighted) and scope words appear in the
    tender's own scope text; the value is only a tie-break. Relevance is a
    hint to the model, never a claim: every project is still listed."""
    scope = _tender_scope_text(tender)
    scope_words = set(_WORD_RE.findall(scope)) - _STOP_WORDS

    def _score(project: dict[str, Any]) -> tuple[float, float]:
        tags = [t.lower() for t in project.get("tags", [])]
        tag_hits = sum(1 for t in tags if re.search(rf"\b{re.escape(t)}", scope))
        text = " ".join([project.get("description") or ""] + list(project.get("scope_items", []))).lower()
        word_hits = len((set(_WORD_RE.findall(text)) - _STOP_WORDS) & scope_words)
        return (3 * tag_hits + word_hits, float(project.get("value_inr_lakh") or 0))

    projects = list(company_profile.get("past_experience", []))
    return sorted(projects, key=_score, reverse=True)


def _ordinal(n: int) -> str:
    return f"{n}{'th' if 11 <= n % 100 <= 13 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def _fmt_iso_date(value: Any) -> str:
    """"2026-02-23" -> "23-02-2026" (Indian tender convention); anything
    else unchanged."""
    text = str(value or "")
    m = re.fullmatch(r"(\d{4})-(\d{2})-(\d{2})", text)
    return f"{m.group(3)}-{m.group(2)}-{m.group(1)}" if m else text


def _project_line(n: int, project: dict[str, Any], detail: bool) -> str:
    wo = project.get("work_order_no") or "[TO BE FILLED FROM COMPANY RECORDS: work order no.]"
    date = _fmt_iso_date(project.get("work_order_date")) or "-"
    value = project.get("value_display") or (
        f"Rs. {project['value_inr_lakh']:.2f} lakh" if project.get("value_inr_lakh")
        else "[TO BE FILLED FROM COMPANY RECORDS: work order value]"
    )
    parts = [
        f"  {n}. {project.get('short_name') or project.get('client')}",
        f"client: {project.get('client')}",
    ]
    if project.get("routed_through"):
        parts.append(f"through: {project['routed_through']}")
    parts += [f"work order: {wo} dated {date}", f"value: {value}"]
    if project.get("quantity"):
        parts.append(f"quantity: {project['quantity']}")
    parts.append(f"FY {project.get('financial_year') or '-'}")
    parts.append(f"status: {project.get('status') or '-'}")
    if detail:
        if project.get("contracted_as"):
            parts.append(f"work order addressed to: {project['contracted_as']}")
        if project.get("end_client_ref"):
            parts.append(f"end-client reference: {project['end_client_ref']}")
        if project.get("project"):
            parts.append(f"project: {project['project']}")
        parts.append(f"scope: {'; '.join(project.get('scope_items', [])) or project.get('description') or '-'}")
    else:
        parts.append(f"tags: {', '.join(project.get('tags', [])) or '-'}")
    parts.append(f"library document: {project['library_document']}" if project.get("library_document")
                 else "evidence: none in the library - do not list it as experience")
    return " | ".join(parts)


def _known(value: Any) -> str:
    return "[not on record]" if value in (None, "", []) else str(value)


def company_data_lines(company_profile: dict[str, Any], tender: dict[str, Any], detail: bool = True) -> list[str]:
    """Every verified fact in config/company_profile.json, as prompt lines -
    projects ranked by relevance to `tender` (rank_past_experience).
    `detail` adds each project's full scope and references (the drafter
    needs them; the checklist planner only needs enough to pick rows)."""
    p = company_profile
    signatory = p.get("authorized_signatory", {})
    lines = [
        f"  - Legal name: {_known(p.get('legal_name'))}"
        + (f" (formerly {p['former_name']}, name changed {_fmt_iso_date(p.get('name_change_date'))})"
           if p.get("former_name") else ""),
        f"  - Constitution: {_known(p.get('constitution'))}; incorporated {_fmt_iso_date(p.get('date_of_incorporation'))}"
        + (f" under the {p['incorporated_under']}" if p.get("incorporated_under") else ""),
        f"  - CIN: {_known(p.get('cin'))}",
        f"  - PAN: {_known(p.get('pan'))} | GSTIN: {_known(p.get('gstin'))}"
        + (f" (GST registered from {_fmt_iso_date(p['gst_registration_date'])})" if p.get("gst_registration_date") else ""),
        f"  - TAN: {_known(p.get('tan'))}",
        f"  - Registered office: {_known(p.get('registered_office'))}",
        f"  - Correspondence address: {_known(p.get('correspondence_address'))}",
        f"  - Email / phone / website: {_known(p.get('email'))} / {_known(p.get('phone'))} / {_known(p.get('website'))}",
        "  - Directors: " + (", ".join(f"{d.get('name')} ({d.get('designation')})" for d in p.get("directors", [])) or "-"),
        f"  - Authorized signatory: {_known(signatory.get('name'))}, {_known(signatory.get('designation'))}, "
        f"place {_known(signatory.get('place'))}",
    ]
    for reg in p.get("registrations", []):
        extra = ", ".join(
            f"{k.replace('_', ' ')} {_fmt_iso_date(v)}" for k, v in reg.items()
            if k not in ("name", "number", "certificate_no", "library_document", "nic_codes") and v
        )
        lines.append(f"  - {reg.get('name')}: {reg.get('number') or reg.get('certificate_no') or '-'}"
                     + (f" ({extra})" if extra else ""))
    for cert in p.get("certifications", []):
        lines.append(f"  - Certification {cert.get('name')}: certificate {cert.get('certificate_no') or '-'}, "
                     f"valid until {_fmt_iso_date(cert.get('valid_until'))}, scope: {cert.get('scope') or '-'}")
    financials = p.get("financials") or {}
    if financials.get("annual_turnover"):
        lines.append("  - Annual turnover (CA certified): " + "; ".join(
            f"FY {t.get('financial_year')} {t.get('amount_display')}" for t in financials["annual_turnover"]))
    if financials.get("average_turnover"):
        avg = financials["average_turnover"]
        lines.append(f"  - Average annual turnover {avg.get('period')}: {avg.get('amount_display')}")
    ca = financials.get("ca_certificate") or {}
    if ca:
        lines.append(f"  - Turnover certified by {ca.get('firm')}, FRN {ca.get('frn')}, {ca.get('signatory')}, "
                     f"M.No {ca.get('membership_no')}, UDIN {ca.get('udin')}, dated {_fmt_iso_date(ca.get('date'))}")
    for bs in financials.get("balance_sheet", []):
        lines.append(f"  - Balance sheet FY {bs.get('financial_year')} ({bs.get('entity')}, as at "
                     f"{_fmt_iso_date(bs.get('as_at'))}): net worth {bs.get('net_worth_display')} (capital "
                     f"{bs.get('share_capital_display')} + surplus {bs.get('surplus_display')}); current assets "
                     f"{bs.get('current_assets_display')}, current liabilities {bs.get('current_liabilities_display')}, "
                     f"current ratio (liquidity) {bs.get('current_ratio')}; source: {bs.get('source')}")
    if financials.get("financial_statement_documents"):
        lines.append("  - Financial statements on file: " + ", ".join(financials["financial_statement_documents"]))
    manpower = p.get("manpower") or {}
    if manpower.get("headcount"):
        lines.append(f"  - Employees: {manpower['headcount']} as on {_fmt_iso_date(manpower.get('as_of'))}"
                     + ("" if manpower.get("verified") else " (last documented figure - state it only 'as on' "
                        "that date; its reconfirmation is already tracked, so it is not an open item)"))
    if p.get("services"):
        lines.append("  - Services (with their evidence):")
        lines += [f"      * {s.get('service')} [evidence: {s.get('evidence')}]" for s in p["services"]]
    projects = rank_past_experience(p, tender)
    if projects:
        lines.append("  - Projects / past experience (ranked most relevant to THIS tender first):")
        lines += [_project_line(n, project, detail) for n, project in enumerate(projects, 1)]
    people = [k for k in p.get("key_personnel", []) if k.get("name")]
    if people:
        lines.append("  - Key personnel (proposed for tenders - fill personnel lists / CVs from these only):")
        for k in people:
            lines.append("      * " + " | ".join(f"{label}: {k[key]}" for key, label in (
                ("name", "name"), ("role", "role"), ("qualification", "qualification"),
                ("experience_years", "experience (years)"), ("experience", "experience"),
                ("skills", "skills"), ("languages", "languages")) if k.get(key)))
    else:
        lines.append("  - Key personnel: none on record - every named role in a personnel list / CV stays a "
                     "placeholder (see rule 16)")
    if p.get("to_be_verified"):
        lines.append("  - NOT verified / not on record (use a placeholder if a document needs one of these; they "
                     "are already tracked in the pack's review notes, so list them in open_items only when a "
                     "document leaves one as a placeholder): " + "; ".join(p["to_be_verified"]))
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
    lines += company_data_lines(company_profile, tender, detail=False)
    for c in eligibility_criteria:
        lines.append(f"  - Usually assessed on {c['criterion']}: {c['requirement']}")
    lines.append("")
    lines.append("Bidder projects: listed above, most relevant first - a work order row's library_document is "
                 "its project's \"library document\".")
    lines.append("")
    lines.append("Documents library (exact names):")
    lines += [f"  - {name}" for name in library_document_names] or ["  (empty)"]
    return "\n".join(lines)


BID_DRAFT_SYSTEM_PROMPT = """You are a bid-documentation assistant preparing an Indian \
government/PSU tender bid, working as the bidder's own bid manager. You are \
given: (1) a list of documents from ONE tender's submission checklist that \
the bidder must write itself, (2) the tender's own details and, when \
available, the full text of its documents, and (3) the bidder company's \
verified company data.

Draft the full body text of EVERY document listed, in the order given. \
Follow these rules strictly:
1. Where the tender text prescribes a format for a document (an annexure, \
   form, undertaking or bid letter wording), reproduce that format's \
   wording and fields faithfully, filling in the bidder's details. Where \
   it prescribes none, draft a standard one for its stated purpose.
2. FILL EVERY FIELD from "Company data" and the tender details below - \
   names, CIN, PAN, GSTIN, registrations and their numbers, certifications, \
   directors, turnover figures and the CA certificate's details, headcount, \
   project clients, work order numbers and dates, values, quantities, scope \
   and status. Never leave a field blank or as a placeholder when the fact \
   is given below. Copy every number, value and date EXACTLY as given - \
   never convert, round or re-express it (e.g. never turn "Rs. 265.46 \
   lakh" into rupees). Never invent a fact: no client, work order, value, \
   date, scope, certificate, registration, director, employee count, \
   turnover, bank detail or compliance claim may appear unless it is given \
   below. Only when a document genuinely needs a fact that is NOT given \
   anywhere below - including anything shown as "[TO BE FILLED FROM \
   COMPANY RECORDS: ...]" in the data - write the exact placeholder \
   "[TO BE FILLED FROM COMPANY RECORDS: <the specific missing fact>]" in \
   its place and list that gap in "open_items" for that document.
3. TAILOR EVERY DOCUMENT TO THIS TENDER. Read the tender's scope of work \
   and write for it: select the company's services, certifications and \
   projects that match it (the projects are listed most relevant first), \
   lead with those, and describe each project through the parts of its \
   own "scope" that match this tender (e.g. for a social media tender: its \
   Facebook/Instagram/YouTube/WhatsApp campaigns, multilingual content, \
   videos, creatives, analytics dashboards and reporting). Tailoring means \
   selecting and reframing the REAL facts given - never adding a \
   capability, claim or project the data does not state, and no \
   superlatives ("substantial", "proven track record", "large-scale") the \
   data doesn't bear out. Do not open with generic brochure language that \
   doesn't match the tender (e.g. calling the company "an EdTech company" \
   in a social media tender); mention unrelated work only briefly, if at \
   all.
4. Address each document to the tendering organisation named below and \
   reference the tender by its reference number where relevant, in the \
   formal register of an Indian government/PSU tender bid.
5. Start each letter/declaration with its addressee ("To," then the \
   organisation's name as separate paragraphs) and a "Subject: ..." \
   paragraph, then "Dear Sir/Madam," where the form is a letter. End each \
   document with a short closing line appropriate to its own content \
   (e.g. "Yours faithfully," for a letter, or a plain affirmation for a \
   declaration). Do NOT add the company's name, a date, place, signatory \
   name, "Signature: ______", or "Company Seal: ______" line yourself - the \
   date is printed above every document and "For <company>" plus the \
   signature block below it, automatically, so adding your own would \
   duplicate them.
6. Write each paragraph as a separate string in "body_paragraphs", in \
   reading order - no markdown, bullet characters, or HTML. For a form of \
   numbered fields, write one "<field>: <value>" string per field.
7. TABLES: any list of projects, turnover years or personnel goes in \
   "tables", not in paragraphs. A Project Experience Summary / list of \
   similar works MUST be a table with the columns "S.No.", "Client / End \
   Client", "Work Order No. & Date", "Relevant Scope", "Value (Rs.)", \
   "Status" - one row per relevant project whose work order is in the \
   documents library (a project marked "evidence: none in the library" is \
   left out unless the tender asks for all experience), most relevant \
   first, filled from the project data: the relevant scope in a short \
   phrase, the value and status exactly as given. Paragraphs that must \
   come after a table (the closing line) go in "closing_paragraphs". When \
   a tender prescribes its own columns, use those.
8. A Covering Letter submits the offer for this tender (title and \
   reference) and, in short paragraphs: introduces the bidder (legal name, \
   former name where relevant, CIN, PAN, GSTIN, MSME/Udyam and DPIIT status \
   with their numbers); states the most relevant verified experience for \
   this tender in one or two sentences (projects, clients, values); gives \
   the CA-certified average turnover; confirms the documents are enclosed \
   as per the submission checklist and the tender's terms are accepted; \
   and names the authorised signatory as the contact (email / phone). \
   A Company Profile states the company's legal identity (with its former \
   name where earlier work orders use it), incorporation, registrations \
   (MSME/Udyam, DPIIT), certifications, CA-certified turnover, headcount \
   (as on its date), the services relevant to this tender and its most \
   relevant projects with their values - led by what this tender needs.
9. Keep each document focused only on its own stated purpose - do not \
   repeat the entire covering letter's content inside every other document.
10. Do not claim this tender's eligibility conditions are met unless the \
   data below shows it; state the facts (e.g. the turnover figures) and let \
   the evaluator compare. A project supports only the scope its own data \
   states - general experience does not prove a narrower field. Resources \
   named in a project's scope (a program manager, field coordinators...) \
   were that project's deployment - never present them as the company's \
   staff structure or headcount.
11. The "Company background" section (when present) is descriptive text \
   from the company's own brochure - use it only for general tone, never \
   as proof of anything and never over the verified company data, and \
   never refer to "the brochure" or any document as enclosed.
12. TENDER FACTS COME FROM THE TENDER, not company records: the IFT / bid \
   number, addenda / corrigenda, performance security %, bid validity, \
   delivery or contract period and the purchaser's address are read from \
   the tender text and copied exactly. Addenda: list the ones the tender \
   text mentions, or write "Nil" when it mentions none. When the text \
   truly doesn't state one of these, write "As per the tender document" - \
   never a company-records placeholder.
13. DATES: every date / "Date of submission" / "this __ day of __ 20__" \
   field is the bid date given below ("Bid date") - fill it in, never leave \
   it as a placeholder.
14. PRICES are never written in these documents: wherever a form asks for \
   the bid / tender amount, a rate or a price, write "As quoted in the \
   Price Schedule / Financial Bid" and add "Quote the price in the \
   Financial Bid" to open_items.
15. FIELDS THAT DON'T APPLY to this bidder (it is a service provider: no \
   factory, no manufacturer, no goods manufactured; or no joint venture, \
   no foreign collaboration) are filled "Not applicable - <legal name> is a \
   service provider" (with the registered office where the form asks for \
   premises) - never a placeholder, and never an invented factory.
16. PERSONNEL: a list of proposed personnel / CVs is filled from the "Key \
   personnel" in the company data - each person's name, role, \
   qualification and experience exactly as given; match the tender's roles \
   to the listed people by role. A role no listed person fills stays \
   "[TO BE FILLED FROM COMPANY RECORDS: name, qualification and experience \
   of the proposed <role>]" - ONE placeholder per role, in the Name column \
   only (put "-" in the other cells of that row) - and one open item per \
   such role. Never invent a person. A CV (or CV format) for a role that \
   no listed person fills is NOT drafted field by field: its body is the \
   single paragraph "[TO BE FILLED FROM COMPANY RECORDS: CV of the proposed \
   <role> in the tender's prescribed format - <the tender's minimum \
   qualification and experience for the role>]", with one open item for \
   it. Never propose the directors or the authorised signatory for a role \
   unless the Key personnel list names them for it.
17. PROPOSAL DOCUMENTS (approach & methodology, work plan, activity / work \
   schedule, deployment plan, team composition by task) are the bidder's \
   own proposal, not company records: DRAFT them in full from the tender's \
   scope of work, deliverables and timelines (e.g. a month-by-month \
   activity schedule for the contract period), drawing on how the \
   company's relevant projects were delivered. Only numeric performance \
   commitments the bidder must decide (targets, KPIs, prices) stay as \
   placeholders - one per table column, not one per cell.

Respond with ONLY a single JSON object, no markdown fences and no extra \
text, matching exactly this shape:
{
  "documents": [
    {
      "id": <string, the corresponding input document's id exactly>,
      "title": <string, the document's heading>,
      "body_paragraphs": [<strings, one per paragraph, in order>],
      "tables": [{"title": <string, may be "">, "columns": [<strings>], "rows": [[<strings, one per column>]]}],
      "closing_paragraphs": [<strings after the tables; empty list if none>],
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
    plus the full verified company data (company_data_lines - projects
    ranked for this tender), and optionally `company_background` (brochure
    text - descriptive only, see BID_DRAFT_SYSTEM_PROMPT rule 11).
    `established_facts` is a flat list of already-true statements the
    caller has already worked out from the eligibility compliance matrix.
    """
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
    today = dt.date.today()
    lines.append("")
    lines.append(f"Bid date: {today.strftime('%d-%m-%Y')} (the {_ordinal(today.day)} day of "
                 f"{today.strftime('%B %Y')})")
    lines.append("")
    lines.append("Company data (verified - use it to fill every field; only these may be stated as fact):")
    lines += company_data_lines(company_profile, tender)
    for fact in established_facts:
        lines.append(f"  - {fact}")

    if company_background.strip():
        lines.append("")
        lines.append("Company background (descriptive only - not evidence of any requirement):")
        lines.append(company_background.strip())

    lines.append("")
    lines += _tender_text_block(tender)
    return "\n".join(lines)
