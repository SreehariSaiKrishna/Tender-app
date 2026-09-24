"""AI steps behind a tender's submission checklist and bid pack - used by
POST /tenders/{id}/checklist and /generate-bid (app.api.main /
app.reports.bid_generator):

- plan_submission_checklist: one call that reads the tender (its extracted
  requirements and, when kept, the full text of its documents) and lists
  every document the bid must include - the rows of the Master Bid
  Submission Checklist, each marked as an existing record to "upload" or a
  document the bidder must "draft", plus whether it needs the letterhead,
  signature and stamp.
- draft_checklist_documents: drafts the body text of every "draft" row, in
  small batches run in parallel.

Same design goals as app.intelligence.scorer/document_summarizer: provider-
agnostic (reuses their AIProvider protocol/get_provider() factory), never
invents facts (the prompts instruct the model to emit an explicit
placeholder for anything not given rather than guess - see prompts.py), and
never crashes the request over a bad response - callers treat a
BidDraftingError as "fall back to the deterministic checklist / templated
pages", not a hard failure.
"""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from pydantic import BaseModel, Field, ValidationError, field_validator

from app.intelligence.prompts import (
    build_bid_draft_system_prompt,
    build_bid_draft_user_prompt,
    build_submission_checklist_system_prompt,
    build_submission_checklist_user_prompt,
)
from app.intelligence.scorer import AIProvider, ScreeningError, get_provider

# Rows drafted per AI call - small enough that each call's response
# (a few full documents) finishes well inside the API's 30-second timeout
# (see prompts.py); the batches themselves run in parallel.
DRAFT_BATCH_SIZE = 3
MAX_PARALLEL_DRAFTS = 6


class BidDraftingError(RuntimeError):
    """Raised when a checklist/draft call or its response fails - callers
    should treat this as "AI unavailable for this step", not crash the
    whole request."""


class SubmissionRow(BaseModel):
    document: str
    what_to_upload: str = ""
    where: str = "Technical Upload"
    source: str = "upload"
    library_document: str | None = None
    letterhead: bool = False
    signature: bool = False
    stamp: bool = False
    format_hint: str = ""
    notes: str = ""

    @field_validator("source")
    @classmethod
    def _known_source(cls, value: str) -> str:
        return "draft" if (value or "").strip().lower() == "draft" else "upload"


class SubmissionChecklistPlan(BaseModel):
    bid_number: str | None = None
    bid_end: str | None = None
    rows: list[SubmissionRow] = Field(default_factory=list)


class DraftedDocument(BaseModel):
    title: str
    body_paragraphs: list[str] = Field(default_factory=list)
    open_items: list[str] = Field(default_factory=list)
    id: str = ""  # the checklist row it was drafted for


class DraftedDocumentSet(BaseModel):
    documents: list[DraftedDocument] = Field(default_factory=list)


def _resolve_provider(provider: AIProvider | None) -> AIProvider:
    """get_provider() raises ScreeningError (e.g. no OPENAI_API_KEY
    configured) - translated to BidDraftingError here so every caller of
    this module only ever has one exception type to catch."""
    if provider is not None:
        return provider
    try:
        return get_provider()
    except ScreeningError as exc:
        raise BidDraftingError(f"No AI provider available: {exc}") from exc


def _call_json(provider: AIProvider, system_prompt: str, user_prompt: str, step: str) -> dict[str, Any]:
    try:
        raw = provider.generate(system_prompt, user_prompt)
    except Exception as exc:  # noqa: BLE001 - deliberately catch-all, see BidDraftingError's docstring
        raise BidDraftingError(f"{step}: provider call failed: {exc}") from exc

    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise BidDraftingError(f"{step}: provider response was not valid JSON: {exc}") from exc


def plan_submission_checklist(
    tender: dict[str, Any],
    eligibility_criteria: list[dict[str, Any]],
    company_profile: dict[str, Any],
    library_document_names: list[str],
    provider: AIProvider | None = None,
) -> SubmissionChecklistPlan:
    """Every document this tender's submission needs, in the tender's own
    order - see this module's docstring."""
    provider = _resolve_provider(provider)
    data = _call_json(
        provider,
        build_submission_checklist_system_prompt(),
        build_submission_checklist_user_prompt(tender, eligibility_criteria, company_profile, library_document_names),
        "Submission checklist",
    )
    try:
        plan = SubmissionChecklistPlan.model_validate(data)
    except ValidationError as exc:
        raise BidDraftingError(f"Submission checklist: response did not match the expected shape: {exc}") from exc

    plan.rows = [r for r in plan.rows if r.document.strip()]
    if not plan.rows:
        raise BidDraftingError("Submission checklist: provider returned no rows.")
    return plan


def _draft_batch(
    tender: dict[str, Any],
    rows: list[dict[str, Any]],
    company_profile: dict[str, Any],
    established_facts: list[str],
    company_background: str,
    provider: AIProvider,
) -> list[DraftedDocument]:
    data = _call_json(
        provider,
        build_bid_draft_system_prompt(),
        build_bid_draft_user_prompt(tender, rows, company_profile, established_facts, company_background),
        "Document drafting",
    )
    try:
        drafted = DraftedDocumentSet.model_validate(data)
    except ValidationError as exc:
        raise BidDraftingError(f"Document drafting: response did not match the expected shape: {exc}") from exc

    # Matched back by id; a response that dropped the ids but kept the
    # order still lines up with its rows positionally.
    ids = {r["id"] for r in rows}
    if not all(d.id in ids for d in drafted.documents) and len(drafted.documents) == len(rows):
        for doc, row in zip(drafted.documents, rows):
            doc.id = row["id"]
    return [d for d in drafted.documents if d.id in ids]


def draft_checklist_documents(
    tender: dict[str, Any],
    rows: list[dict[str, Any]],
    company_profile: dict[str, Any],
    established_facts: list[str],
    provider: AIProvider | None = None,
    company_background: str = "",
    batch_size: int = DRAFT_BATCH_SIZE,
) -> tuple[dict[str, DraftedDocument], list[str]]:
    """Drafts every given checklist row (`id`, `document`,
    `what_to_upload`, optional `format_text`/`notes`) - returns the drafts
    by row id plus one error message per batch that failed, so one bad
    batch never costs the others. Raises BidDraftingError only when no
    provider is available at all."""
    provider = _resolve_provider(provider)
    batches = [rows[i:i + batch_size] for i in range(0, len(rows), batch_size)]
    if not batches:
        return {}, []

    def _run(batch: list[dict[str, Any]]) -> list[DraftedDocument] | BidDraftingError:
        try:
            return _draft_batch(tender, batch, company_profile, established_facts, company_background, provider)
        except BidDraftingError as exc:
            return exc

    drafted: dict[str, DraftedDocument] = {}
    errors: list[str] = []
    with ThreadPoolExecutor(max_workers=min(MAX_PARALLEL_DRAFTS, len(batches))) as pool:
        for batch, result in zip(batches, pool.map(_run, batches)):
            if isinstance(result, BidDraftingError):
                errors.append(f"{', '.join(r['document'] for r in batch)}: {result}")
                continue
            for doc in result:
                drafted[doc.id] = doc
    return drafted, errors
