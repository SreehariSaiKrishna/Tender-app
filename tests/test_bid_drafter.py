"""Tests for app.intelligence.bid_drafter - a FakeProvider stands in for
the real OpenAI-backed AIProvider (see test_document_summarizer.py/
test_scorer.py for the same pattern), so no API key or network access is
required.
"""
from __future__ import annotations

import json

import pytest

from app.intelligence.bid_drafter import (
    BidDraftingError,
    MAX_DOCUMENTS,
    draft_bid_documents,
    draft_documents,
    plan_required_documents,
)

TENDER = {
    "title": "Tender for Social Media Management",
    "organisation": "Some Department",
    "tender_ref": "12345",
    "document_summary": {
        "documents_to_submit": ["Non-blacklisting undertaking"],
        "eligibility_requirements": ["3 years of relevant experience"],
        "eligibility_technical_criteria": [],
        "summary_text": "A tender for social media management services.",
    },
}

ELIGIBILITY_CRITERIA = [
    {"id": "legal-status", "criterion": "Legal Status", "requirement": "Registered company.", "supporting_documents": []},
]

COMPANY_PROFILE = {
    "legal_name": "TEST COMPANY PRIVATE LIMITED",
    "registered_office": "1 Test Street",
    "authorized_signatory": {"name": "Jane Doe", "designation": "Director"},
}

VALID_PLAN = {
    "documents": [
        {"title": "Covering Letter", "purpose": "Submits the offer.", "kind": "covering_letter"},
        {"title": "Non-Blacklisting Declaration", "purpose": "Confirms no blacklisting.", "kind": "declaration"},
    ]
}

VALID_DRAFT = {
    "documents": [
        {"title": "Covering Letter", "body_paragraphs": ["We submit our offer."], "open_items": []},
        {
            "title": "Non-Blacklisting Declaration",
            "body_paragraphs": ["We are not blacklisted."],
            "open_items": [],
        },
    ]
}


class FakeProvider:
    def __init__(self, responses=None, raise_on_call=False):
        self.responses = responses or []
        self.raise_on_call = raise_on_call
        self.calls = []

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        if self.raise_on_call:
            raise RuntimeError("simulated network failure")
        return self.responses[len(self.calls) - 1]


def test_plan_required_documents_returns_validated_plan():
    provider = FakeProvider(responses=[json.dumps(VALID_PLAN)])
    plan = plan_required_documents(TENDER, ELIGIBILITY_CRITERIA, provider)
    assert [d.title for d in plan.documents] == ["Covering Letter", "Non-Blacklisting Declaration"]
    assert len(provider.calls) == 1


def test_plan_required_documents_forces_a_covering_letter_first():
    plan_without_letter = {"documents": [{"title": "Non-Blacklisting Declaration", "purpose": "x", "kind": "declaration"}]}
    provider = FakeProvider(responses=[json.dumps(plan_without_letter)])
    plan = plan_required_documents(TENDER, ELIGIBILITY_CRITERIA, provider)
    assert plan.documents[0].kind == "covering_letter"
    assert plan.documents[0].title == "Covering Letter"


def test_plan_required_documents_caps_at_max_documents():
    many = {
        "documents": [
            {"title": f"Doc {i}", "purpose": "x", "kind": "declaration"} for i in range(MAX_DOCUMENTS + 5)
        ]
    }
    provider = FakeProvider(responses=[json.dumps(many)])
    plan = plan_required_documents(TENDER, ELIGIBILITY_CRITERIA, provider)
    assert len(plan.documents) <= MAX_DOCUMENTS


def test_plan_required_documents_raises_on_provider_failure():
    provider = FakeProvider(raise_on_call=True)
    with pytest.raises(BidDraftingError):
        plan_required_documents(TENDER, ELIGIBILITY_CRITERIA, provider)


def test_plan_required_documents_raises_on_invalid_json():
    provider = FakeProvider(responses=["not json"])
    with pytest.raises(BidDraftingError):
        plan_required_documents(TENDER, ELIGIBILITY_CRITERIA, provider)


def test_draft_documents_raises_when_empty():
    provider = FakeProvider(responses=[json.dumps({"documents": []})])
    from app.intelligence.bid_drafter import DocumentPlan, RequiredDocument

    plan = DocumentPlan(documents=[RequiredDocument(title="Covering Letter", purpose="x", kind="covering_letter")])
    with pytest.raises(BidDraftingError):
        draft_documents(TENDER, plan, COMPANY_PROFILE, [], provider)


def test_draft_bid_documents_makes_exactly_two_calls_and_preserves_plan_order():
    # Draft response deliberately reordered - draft_bid_documents must still
    # return documents in the plan's order, not the model's response order.
    reordered_draft = {
        "documents": [
            VALID_DRAFT["documents"][1],
            VALID_DRAFT["documents"][0],
        ]
    }
    provider = FakeProvider(responses=[json.dumps(VALID_PLAN), json.dumps(reordered_draft)])

    documents = draft_bid_documents(TENDER, ELIGIBILITY_CRITERIA, COMPANY_PROFILE, [], provider)

    assert len(provider.calls) == 2
    assert [d.title for d in documents] == ["Covering Letter", "Non-Blacklisting Declaration"]


def test_draft_bid_documents_propagates_drafting_failure():
    provider = FakeProvider(responses=[json.dumps(VALID_PLAN), "not json"])
    with pytest.raises(BidDraftingError):
        draft_bid_documents(TENDER, ELIGIBILITY_CRITERIA, COMPANY_PROFILE, [], provider)
