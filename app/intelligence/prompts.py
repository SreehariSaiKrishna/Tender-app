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
