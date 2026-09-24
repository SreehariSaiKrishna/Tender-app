"""Tests for app.intelligence.bid_drafter - a FakeProvider stands in for
the real OpenAI-backed AIProvider (see test_document_summarizer.py/
test_scorer.py for the same pattern), so no API key or network access is
required.
"""
from __future__ import annotations

import json
import threading

import pytest

from app.intelligence.bid_drafter import (
    BidDraftingError,
    draft_checklist_documents,
    plan_submission_checklist,
)

TENDER = {
    "title": "Tender for Social Media Management",
    "organisation": "Some Department",
    "tender_ref": "12345",
    "document_text": "ANNEXURE 3 - Undertaking of non-blacklisting. We hereby declare ...",
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

VALID_CHECKLIST = {
    "bid_number": "GEM/2026/B/6045377",
    "bid_end": "28-09-2026, 19:00 Hrs",
    "rows": [
        {"document": "PAN", "what_to_upload": "PAN card", "where": "Technical Upload", "source": "upload",
         "library_document": "PAN Card", "letterhead": False, "signature": True, "stamp": True,
         "format_hint": "", "notes": ""},
        {"document": "Annexure 3", "what_to_upload": "Non-blacklisting undertaking", "where": "Technical Upload",
         "source": "DRAFT", "library_document": None, "letterhead": True, "signature": True, "stamp": True,
         "format_hint": "Annexure 3 format", "notes": ""},
        {"document": "  ", "source": "upload"},
    ],
}


class FakeProvider:
    def __init__(self, responses=None, raise_on_call=False, respond=None):
        self.responses = responses or []
        self.raise_on_call = raise_on_call
        self.respond = respond
        self.calls = []
        self._lock = threading.Lock()

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        with self._lock:
            self.calls.append((system_prompt, user_prompt))
            n = len(self.calls)
        if self.raise_on_call:
            raise RuntimeError("simulated network failure")
        if self.respond:
            return self.respond(user_prompt)
        return self.responses[n - 1]


def test_plan_submission_checklist_returns_validated_rows():
    provider = FakeProvider(responses=[json.dumps(VALID_CHECKLIST)])
    plan = plan_submission_checklist(TENDER, ELIGIBILITY_CRITERIA, COMPANY_PROFILE, ["PAN Card"], provider)

    assert plan.bid_number == "GEM/2026/B/6045377"
    assert [r.document for r in plan.rows] == ["PAN", "Annexure 3"]  # the blank row is dropped
    assert plan.rows[1].source == "draft"
    # The prompt carries the tender's own text and the library's names.
    user_prompt = provider.calls[0][1]
    assert "ANNEXURE 3 - Undertaking" in user_prompt
    assert "  - PAN Card" in user_prompt


@pytest.mark.parametrize("provider", [
    FakeProvider(raise_on_call=True),
    FakeProvider(responses=["not json"]),
    FakeProvider(responses=[json.dumps({"rows": []})]),
])
def test_plan_submission_checklist_raises_on_bad_responses(provider):
    with pytest.raises(BidDraftingError):
        plan_submission_checklist(TENDER, ELIGIBILITY_CRITERIA, COMPANY_PROFILE, [], provider)


def _rows(n):
    return [{"id": f"r{i}", "document": f"Annexure {i}", "what_to_upload": "x"} for i in range(n)]


def _echo_drafts(user_prompt: str) -> str:
    ids = [line.split("id ", 1)[1].split(":", 1)[0] for line in user_prompt.splitlines() if line.startswith("  - id ")]
    return json.dumps({"documents": [
        {"id": i, "title": f"Doc {i}", "body_paragraphs": ["Body"], "open_items": []} for i in ids
    ]})


def test_draft_checklist_documents_batches_rows_and_keys_drafts_by_row_id():
    provider = FakeProvider(respond=_echo_drafts)
    drafted, errors = draft_checklist_documents(TENDER, _rows(7), COMPANY_PROFILE, [], provider, batch_size=3)

    assert errors == []
    assert len(provider.calls) == 3
    assert sorted(drafted) == [f"r{i}" for i in range(7)]
    assert drafted["r4"].title == "Doc r4"


def test_draft_checklist_documents_keeps_other_batches_when_one_fails():
    def respond(user_prompt):
        if "id r0:" in user_prompt:
            return "not json"
        return _echo_drafts(user_prompt)

    drafted, errors = draft_checklist_documents(
        TENDER, _rows(4), COMPANY_PROFILE, [], FakeProvider(respond=respond), batch_size=2
    )
    assert sorted(drafted) == ["r2", "r3"]
    assert len(errors) == 1 and errors[0].startswith("Annexure 0, Annexure 1:")


def test_draft_checklist_documents_matches_by_position_when_ids_are_dropped():
    response = json.dumps({"documents": [
        {"title": "First", "body_paragraphs": ["a"]}, {"title": "Second", "body_paragraphs": ["b"]},
    ]})
    drafted, _ = draft_checklist_documents(
        TENDER, _rows(2), COMPANY_PROFILE, [], FakeProvider(responses=[response]), batch_size=2
    )
    assert drafted["r0"].title == "First" and drafted["r1"].title == "Second"
