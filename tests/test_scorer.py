"""Tests for Phase 5: AI screening.

Uses a FakeProvider implementing the same `generate()` interface as
OpenAIProvider, and mongomock (an in-memory fake implementing the pymongo
API) - no OpenAI API key, network access, or real MongoDB required.
"""
from __future__ import annotations

import datetime as dt
import json

import mongomock
import pytest

from app.intelligence.prompts import build_system_prompt, build_user_prompt
from app.intelligence.scorer import (
    OpenAIProvider,
    ScreeningError,
    ScreeningVerdict,
    screen_tender,
    screen_tenders,
)
from app.models import TenderStatus

VALID_VERDICT = {
    "priority": "High",
    "relevance_score": 85,
    "category": "Digital Marketing",
    "reason": "Title mentions social media management, matching a configured capability.",
    "eligibility_concerns": ["Verify minimum turnover requirement once notice is available"],
    "missing_information": ["Full scope of work not in this export"],
    "recommended_action": "Review full tender notice",
}


class FakeProvider:
    def __init__(self, response=None, responses=None, raise_on_call=False):
        self.response = response
        self.responses = responses or []
        self.raise_on_call = raise_on_call
        self.calls = []

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        if self.raise_on_call:
            raise RuntimeError("simulated network failure")
        if self.responses:
            return self.responses[len(self.calls) - 1]
        return json.dumps(self.response if self.response is not None else VALID_VERDICT)


@pytest.fixture()
def collection():
    client = mongomock.MongoClient()
    return client["test_tender_intelligence"]["tenders"]


def make_tender(collection, dedup_key="ref:1", **overrides) -> dict:
    now = dt.datetime.now(dt.timezone.utc)
    doc = {
        "dedup_key": dedup_key,
        "tender_ref": dedup_key.split(":")[-1],
        "title": "Tender for Social Media Management",
        "organisation": "Some Department",
        "status": TenderStatus.NEW.value,
        "disappeared": False,
        "first_seen": now,
        "last_seen": now,
        "screenings": [],
    }
    doc.update(overrides)
    collection.insert_one(doc)
    return doc


# --- prompts -----------------------------------------------------


def test_system_prompt_includes_capabilities():
    prompt = build_system_prompt(["Digital marketing", "Software development"])
    assert "Digital marketing" in prompt
    assert "Software development" in prompt
    assert "JSON" in prompt


def test_user_prompt_marks_missing_fields_explicitly():
    prompt = build_user_prompt({"title": "Some Tender", "tender_value": None, "organisation": None})
    assert "Some Tender" in prompt
    assert "Not disclosed in listing" in prompt


def test_user_prompt_never_invents_a_value():
    prompt = build_user_prompt({"title": "T", "tender_value": 500000})
    assert "500000" in prompt
    # Untouched fields must still show as missing, not fabricated.
    assert prompt.count("Not disclosed in listing") >= 1


# --- screen_tender / verdict validation -----------------------------------------------------


def test_screen_tender_returns_valid_verdict():
    provider = FakeProvider(response=VALID_VERDICT)
    verdict = screen_tender({"title": "T"}, ["Digital Marketing"], provider)
    assert isinstance(verdict, ScreeningVerdict)
    assert verdict.priority == "High"
    assert verdict.relevance_score == 85
    assert len(provider.calls) == 1


def test_screen_tender_rejects_invalid_priority():
    bad = {**VALID_VERDICT, "priority": "Extremely High"}
    provider = FakeProvider(response=bad)
    with pytest.raises(ScreeningError):
        screen_tender({"title": "T"}, ["Digital Marketing"], provider)


def test_screen_tender_rejects_out_of_range_score():
    bad = {**VALID_VERDICT, "relevance_score": 150}
    provider = FakeProvider(response=bad)
    with pytest.raises(ScreeningError):
        screen_tender({"title": "T"}, ["Digital Marketing"], provider)


def test_screen_tender_rejects_malformed_json():
    provider = FakeProvider(responses=["this is not json"])
    with pytest.raises(ScreeningError):
        screen_tender({"title": "T"}, ["Digital Marketing"], provider)


def test_screen_tender_rejects_missing_required_field():
    incomplete = {k: v for k, v in VALID_VERDICT.items() if k != "reason"}
    provider = FakeProvider(response=incomplete)
    with pytest.raises(ScreeningError):
        screen_tender({"title": "T"}, ["Digital Marketing"], provider)


def test_screen_tender_wraps_provider_exceptions():
    provider = FakeProvider(raise_on_call=True)
    with pytest.raises(ScreeningError):
        screen_tender({"title": "T"}, ["Digital Marketing"], provider)


# --- OpenAIProvider construction -----------------------------------------------------


def test_openai_provider_requires_api_key():
    with pytest.raises(ScreeningError):
        OpenAIProvider(api_key="", model="gpt-4o-mini")


# --- screen_tenders (DB persistence + batch behaviour) -----------------------------------------------------


def test_screen_tenders_persists_results(collection):
    make_tender(collection, dedup_key="ref:1")
    make_tender(collection, dedup_key="ref:2")
    provider = FakeProvider(response=VALID_VERDICT)

    summary = screen_tenders(
        provider=provider, capabilities=["Digital Marketing"], threshold=60, collection=collection
    )

    assert summary.screened == 2
    assert summary.failed == 0
    assert summary.above_threshold == 2

    docs = list(collection.find({}))
    assert sum(len(d["screenings"]) for d in docs) == 2
    result = docs[0]["screenings"][0]
    assert result["priority"] == "High"
    assert result["relevance_score"] == 85
    # Denormalized onto the document itself so the read-only API can filter/
    # sort by priority without unpacking the screenings array.
    assert docs[0]["latest_priority"] == "High"
    assert docs[0]["latest_relevance_score"] == 85


def test_screen_tenders_skips_already_screened_unchanged_tenders(collection):
    make_tender(collection, dedup_key="ref:1")
    provider = FakeProvider(response=VALID_VERDICT)

    first = screen_tenders(
        provider=provider, capabilities=["Digital Marketing"], collection=collection
    )
    assert first.screened == 1

    second = screen_tenders(
        provider=provider, capabilities=["Digital Marketing"], collection=collection
    )
    assert second.screened == 0  # nothing changed since the last screening


def test_screen_tenders_rescreens_when_tender_changes(collection):
    make_tender(collection, dedup_key="ref:1")
    provider = FakeProvider(response=VALID_VERDICT)

    screen_tenders(provider=provider, capabilities=["Digital Marketing"], collection=collection)

    collection.update_one(
        {"dedup_key": "ref:1"},
        {"$set": {"content_last_changed": dt.datetime.now(dt.timezone.utc) + dt.timedelta(days=1)}},
    )

    summary = screen_tenders(
        provider=provider, capabilities=["Digital Marketing"], collection=collection
    )
    assert summary.screened == 1


def test_screen_tenders_does_not_rescreen_merely_because_last_seen_advanced(collection):
    """Regression: a tender reappearing in a later download bumps last_seen
    on nearly every tender, nearly every run - that alone must not trigger
    re-screening, or every run re-screens the whole database. Only a real
    content change (content_last_changed) should.
    """
    make_tender(collection, dedup_key="ref:1")
    provider = FakeProvider(response=VALID_VERDICT)

    screen_tenders(provider=provider, capabilities=["Digital Marketing"], collection=collection)

    collection.update_one(
        {"dedup_key": "ref:1"},
        {"$set": {"last_seen": dt.datetime.now(dt.timezone.utc) + dt.timedelta(days=1)}},
    )

    summary = screen_tenders(
        provider=provider, capabilities=["Digital Marketing"], collection=collection
    )
    assert summary.screened == 0


def test_screen_tenders_continues_past_a_single_failure(collection):
    make_tender(collection, dedup_key="ref:1")
    make_tender(collection, dedup_key="ref:2")
    # First call fails, second succeeds - batch must not abort on the first.
    provider = FakeProvider(responses=["not json", json.dumps(VALID_VERDICT)])

    summary = screen_tenders(
        provider=provider, capabilities=["Digital Marketing"], collection=collection
    )

    assert summary.failed == 1
    assert summary.screened == 1


def test_screen_tenders_respects_limit(collection):
    for i in range(5):
        make_tender(collection, dedup_key=f"ref:{i}")
    provider = FakeProvider(response=VALID_VERDICT)

    summary = screen_tenders(
        provider=provider, capabilities=["Digital Marketing"], limit=2, collection=collection
    )
    assert summary.screened == 2


def test_screen_tenders_excludes_disappeared_tenders(collection):
    make_tender(collection, dedup_key="ref:1", disappeared=True, status=TenderStatus.CLOSED.value)
    make_tender(collection, dedup_key="ref:2")
    provider = FakeProvider(response=VALID_VERDICT)

    summary = screen_tenders(
        provider=provider, capabilities=["Digital Marketing"], collection=collection
    )
    assert summary.screened == 1
