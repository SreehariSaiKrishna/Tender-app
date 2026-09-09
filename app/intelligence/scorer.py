"""AI screening: scores tenders against configured business capabilities.

Design goals:
  - Provider-agnostic: `AIProvider` is a minimal protocol (one method,
    `generate`) so a non-OpenAI provider can be added later without
    touching the scoring/persistence logic below.
  - Never invents data: the prompt (see prompts.py) instructs the model to
    flag missing information rather than guess, and this module validates
    the response shape strictly rather than trusting free-form output.
  - Never crashes a batch over one bad tender or one bad API response -
    failures are recorded per tender, matching the collector's philosophy.
"""
from __future__ import annotations

import datetime as dt
import json
import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

from pydantic import BaseModel, Field, ValidationError, field_validator
from pymongo.collection import Collection

from app.config import Settings, get_settings, load_business_capabilities
from app.database import get_collection
from app.intelligence.prompts import build_system_prompt, build_user_prompt

logger = logging.getLogger(__name__)

ALLOWED_PRIORITIES = {"High", "Medium", "Low", "Not Relevant"}


class ScreeningError(RuntimeError):
    """Raised when a provider response can't be parsed/validated, or the
    provider call itself fails. Callers treat this as a per-tender failure.
    """


class ScreeningVerdict(BaseModel):
    priority: str
    relevance_score: int = Field(ge=0, le=100)
    category: str
    reason: str
    eligibility_concerns: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    recommended_action: str

    @field_validator("priority")
    @classmethod
    def _priority_must_be_known(cls, v: str) -> str:
        if v not in ALLOWED_PRIORITIES:
            raise ValueError(f"priority must be one of {ALLOWED_PRIORITIES}, got {v!r}")
        return v


class AIProvider(Protocol):
    """Minimal interface a screening provider must implement."""

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        """Return the raw text completion (expected to be a JSON object)."""
        ...


class OpenAIProvider:
    """OpenAI chat-completions based provider, using JSON mode."""

    def __init__(self, api_key: str, model: str) -> None:
        from openai import OpenAI  # imported lazily so tests never need the SDK configured

        if not api_key:
            raise ScreeningError(
                "OPENAI_API_KEY is not set. Add it to .env before running AI screening."
            )
        self._client = OpenAI(api_key=api_key)
        self._model = model

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0,
        )
        content = response.choices[0].message.content
        if not content:
            raise ScreeningError("Provider returned an empty response.")
        return content


def get_provider(settings: Settings | None = None) -> AIProvider:
    """Factory: picks a provider implementation based on settings.ai_provider."""
    settings = settings or get_settings()
    if settings.ai_provider == "openai":
        return OpenAIProvider(api_key=settings.openai_api_key, model=settings.openai_model)
    raise ScreeningError(f"Unknown AI provider: {settings.ai_provider!r}")


def screen_tender(
    tender_data: dict[str, Any], capabilities: list[str], provider: AIProvider
) -> ScreeningVerdict:
    """Screen one tender. Raises ScreeningError on any failure - the caller
    decides how to handle a single tender's failure without aborting others.
    """
    system_prompt = build_system_prompt(capabilities)
    user_prompt = build_user_prompt(tender_data)

    try:
        raw = provider.generate(system_prompt, user_prompt)
    except ScreeningError:
        raise
    except Exception as exc:  # noqa: BLE001
        raise ScreeningError(f"Provider call failed: {exc}") from exc

    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ScreeningError(f"Provider response was not valid JSON: {exc}") from exc

    try:
        return ScreeningVerdict.model_validate(parsed)
    except ValidationError as exc:
        raise ScreeningError(f"Provider response did not match the expected shape: {exc}") from exc


@dataclass
class ScreenSummary:
    screened: int = 0
    failed: int = 0
    by_priority: dict[str, int] = field(default_factory=dict)
    above_threshold: int = 0


def _tender_to_dict(t: dict[str, Any]) -> dict[str, Any]:
    closing_date = t.get("closing_date")
    published_date = t.get("published_date")
    return {
        "title": t.get("title"),
        "organisation": t.get("organisation"),
        "location": t.get("location"),
        "state": t.get("state"),
        "closing_date": closing_date.date().isoformat() if closing_date else None,
        "published_date": published_date.date().isoformat() if published_date else None,
        "tender_value": t.get("tender_value"),
        "earnest_money": t.get("earnest_money"),
        "tender_ref": t.get("tender_ref"),
        "description": t.get("description"),
        "source_url": t.get("source_url"),
    }


def _as_naive_utc(d: dt.datetime) -> dt.datetime:
    """Normalize to a naive UTC datetime for comparison.

    pymongo returns datetimes read back from MongoDB as naive UTC values
    (BSON has no separate "aware" concept), while a freshly-constructed
    datetime in this process may still carry tzinfo - comparing the two
    directly would otherwise raise or compare unequal for the same instant.
    """
    if d.tzinfo is not None:
        return d.astimezone(dt.timezone.utc).replace(tzinfo=None)
    return d


def _needs_screening(t: dict[str, Any]) -> bool:
    screenings = t.get("screenings") or []
    if not screenings:
        return True
    latest = screenings[-1]  # append-only: the last element is always the most recent
    # Deliberately compares against content_last_changed, not last_seen:
    # last_seen advances every time a still-live tender reappears in a
    # download, which is nearly every tender, nearly every day - using it
    # here would re-screen almost everything on every run. Documents
    # written before this field existed fall back to first_seen, which is
    # always <= any real screened_at for something already screened once,
    # so old data doesn't get spuriously re-screened by this change.
    content_changed_at = t.get("content_last_changed") or t["first_seen"]
    return _as_naive_utc(latest["screened_at"]) < _as_naive_utc(content_changed_at)


def screen_tenders(
    provider: AIProvider | None = None,
    capabilities: list[str] | None = None,
    threshold: int | None = None,
    limit: int | None = None,
    only_unscreened: bool = True,
    collection: Collection | None = None,
) -> ScreenSummary:
    """Screen candidate tenders and append a screening entry for each.

    By default only screens tenders that have never been screened, or have
    changed (last_seen advanced) since their most recent screening - this
    keeps repeated `screen` runs cheap rather than re-billing every tender
    every time.
    """
    settings = get_settings()
    provider = provider or get_provider(settings)
    capabilities = capabilities if capabilities is not None else load_business_capabilities()
    threshold = threshold if threshold is not None else settings.ai_relevance_threshold
    collection = collection if collection is not None else get_collection()

    candidates = [
        doc
        for doc in collection.find({"disappeared": False})
        if not only_unscreened or _needs_screening(doc)
    ]
    if limit is not None:
        candidates = candidates[:limit]

    summary = ScreenSummary()
    for tender in candidates:
        try:
            verdict = screen_tender(_tender_to_dict(tender), capabilities, provider)
        except ScreeningError as exc:
            logger.error(
                "Screening failed for tender %s (%s): %s",
                tender["_id"],
                tender.get("tender_ref"),
                exc,
            )
            summary.failed += 1
            continue

        screening_entry = {
            "priority": verdict.priority,
            "relevance_score": verdict.relevance_score,
            "category": verdict.category,
            "reason": verdict.reason,
            "eligibility_concerns": verdict.eligibility_concerns,
            "missing_information": verdict.missing_information,
            "recommended_action": verdict.recommended_action,
            "model_used": getattr(provider, "_model", settings.ai_provider),
            "screened_at": dt.datetime.now(dt.timezone.utc),
        }
        collection.update_one(
            {"_id": tender["_id"]},
            {
                "$push": {"screenings": screening_entry},
                # Denormalized onto the document (matching the existing
                # query_match_count pattern) so the read-only API can filter/
                # sort by priority directly instead of unpacking the array.
                "$set": {
                    "latest_priority": verdict.priority,
                    "latest_relevance_score": verdict.relevance_score,
                },
            },
        )

        summary.screened += 1
        summary.by_priority[verdict.priority] = summary.by_priority.get(verdict.priority, 0) + 1
        if verdict.relevance_score >= threshold:
            summary.above_threshold += 1

    return summary
