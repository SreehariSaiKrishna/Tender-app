"""AI drafting of the individual bid documents a tender's own paperwork
calls for (covering letter, non-blacklisting declaration, etc.) - used by
POST /tenders/{id}/generate-bid (app.api.main / app.reports.bid_generator).

Same design goals as app.intelligence.scorer/document_summarizer: provider-
agnostic (reuses their AIProvider protocol/get_provider() factory), never
invents facts (the prompts instruct the model to emit an explicit
placeholder for anything not given rather than guess - see prompts.py), and
never crashes the request over a bad response - callers treat a
BidDraftingError as "fall back to the deterministic parts of the bid pack",
not a hard failure.

Two AI calls, not one per document - see prompts.py's module comment for
why (a synchronous 30-second HTTP API timeout this whole request runs
behind).
"""
from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel, Field, ValidationError

from app.intelligence.prompts import (
    build_bid_draft_system_prompt,
    build_bid_draft_user_prompt,
    build_bid_plan_system_prompt,
    build_bid_plan_user_prompt,
)
from app.intelligence.scorer import AIProvider, ScreeningError, get_provider

# Mirrors prompts.py's BID_PLAN_SYSTEM_PROMPT rule 5 - a hard cap enforced
# here too, independent of whether the model actually respects the prompt.
MAX_DOCUMENTS = 6

ALLOWED_KINDS = {"covering_letter", "declaration", "undertaking", "certificate_narrative", "technical_note"}


class BidDraftingError(RuntimeError):
    """Raised when a plan/draft call or its response fails - callers should
    treat this as "AI drafting unavailable for this bid pack", not crash
    the whole generate-bid request."""


class RequiredDocument(BaseModel):
    title: str
    purpose: str
    kind: str = "declaration"


class DocumentPlan(BaseModel):
    documents: list[RequiredDocument] = Field(default_factory=list)


class DraftedDocument(BaseModel):
    title: str
    body_paragraphs: list[str] = Field(default_factory=list)
    open_items: list[str] = Field(default_factory=list)


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


def plan_required_documents(
    tender: dict[str, Any],
    eligibility_criteria: list[dict[str, Any]],
    provider: AIProvider | None = None,
) -> DocumentPlan:
    """Decides which documents this tender's own paperwork calls for - the
    first of the two AI calls (see this module's docstring)."""
    provider = _resolve_provider(provider)
    system_prompt = build_bid_plan_system_prompt()
    user_prompt = build_bid_plan_user_prompt(tender, eligibility_criteria)

    data = _call_json(provider, system_prompt, user_prompt, "Document planning")
    try:
        plan = DocumentPlan.model_validate(data)
    except ValidationError as exc:
        raise BidDraftingError(f"Document planning: response did not match the expected shape: {exc}") from exc

    plan.documents = [d for d in plan.documents if d.kind in ALLOWED_KINDS] or plan.documents
    if not any(d.kind == "covering_letter" for d in plan.documents):
        plan.documents.insert(
            0,
            RequiredDocument(
                title="Covering Letter",
                purpose="Formally submits the offer and confirms the bidder has read and accepted the tender's terms.",
                kind="covering_letter",
            ),
        )
    if len(plan.documents) > MAX_DOCUMENTS:
        plan.documents = plan.documents[:MAX_DOCUMENTS]
    return plan


def draft_documents(
    tender: dict[str, Any],
    plan: DocumentPlan,
    company_profile: dict[str, Any],
    established_facts: list[str],
    provider: AIProvider | None = None,
    company_background: str = "",
) -> DraftedDocumentSet:
    """Drafts the full body text of every planned document in one call -
    the second of the two AI calls (see this module's docstring)."""
    provider = _resolve_provider(provider)
    system_prompt = build_bid_draft_system_prompt()
    user_prompt = build_bid_draft_user_prompt(
        tender, [d.model_dump() for d in plan.documents], company_profile, established_facts, company_background
    )

    data = _call_json(provider, system_prompt, user_prompt, "Document drafting")
    try:
        drafted = DraftedDocumentSet.model_validate(data)
    except ValidationError as exc:
        raise BidDraftingError(f"Document drafting: response did not match the expected shape: {exc}") from exc

    if not drafted.documents:
        raise BidDraftingError("Document drafting: provider returned no documents.")
    return drafted


def draft_bid_documents(
    tender: dict[str, Any],
    eligibility_criteria: list[dict[str, Any]],
    company_profile: dict[str, Any],
    established_facts: list[str],
    provider: AIProvider | None = None,
    company_background: str = "",
) -> list[DraftedDocument]:
    """Runs both steps and returns the final drafted documents, in plan
    order. Raises BidDraftingError if either step fails - the caller (see
    app.reports.bid_generator.generate_bid_package) decides what a bid pack
    looks like without this section rather than this module deciding for it.
    """
    provider = _resolve_provider(provider)
    plan = plan_required_documents(tender, eligibility_criteria, provider)
    drafted = draft_documents(tender, plan, company_profile, established_facts, provider, company_background)

    # Keep plan order even if the model reordered its response - matched by
    # title since that's the only identifier the draft step round-trips.
    by_title = {d.title.strip().lower(): d for d in drafted.documents}
    ordered = [by_title[p.title.strip().lower()] for p in plan.documents if p.title.strip().lower() in by_title]
    return ordered or drafted.documents
