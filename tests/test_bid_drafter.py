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
    assert "may be incomplete" in user_prompt  # the extracted lists aren't treated as complete
    assert plan.rows[0].basis == ""  # optional in the response


def test_plan_submission_checklist_goes_condition_by_condition():
    response = {
        "conditions": [
            {"ref": "Elig 6 Financial Capacity", "proved_by": "CA Turnover Certificate"},
            {"ref": "Elig 15 Verification", "proved_by": ""},
        ],
        "rows": [{"document": "CA Turnover Certificate", "source": "upload",
                  "basis": "Eligibility 6 - Financial Capacity"}],
    }
    provider = FakeProvider(responses=[json.dumps(response)])
    plan = plan_submission_checklist(TENDER, ELIGIBILITY_CRITERIA, COMPANY_PROFILE, [], provider)

    assert [c.ref for c in plan.conditions] == ["Elig 6 Financial Capacity", "Elig 15 Verification"]
    assert plan.rows[0].basis == "Eligibility 6 - Financial Capacity"
    # The system prompt asks for proof of every condition, not just the
    # tender's explicit "documents to submit" list, and bans catch-all rows.
    system_prompt = provider.calls[0][0]
    assert "Condition by condition" in system_prompt
    assert "CV row PER ROLE" in system_prompt
    assert "Never write a catch-all row" in system_prompt


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


# --- Company data, tailoring, notary -----------------------------------------------

RICH_PROFILE = {
    **COMPANY_PROFILE,
    "former_name": "Old Name Private Limited",
    "cin": "U12345TS2025PTC000001",
    "pan": "ABCDE1234F",
    "gstin": "36ABCDE1234F1Z1",
    "registrations": [{"name": "UDYAM (MSME) Registration", "number": "UDYAM-TS-00-0000000"}],
    "certifications": [{"name": "ISO/IEC 27001:2022", "certificate_no": "ISO-1", "valid_until": "2029-06-25"}],
    "services": [{"service": "Social media campaign execution", "evidence": "WO-SM-1"}],
    "financials": {
        "annual_turnover": [{"financial_year": "2024-25", "amount_display": "Rs. 5,23,89,823"}],
        "average_turnover": {"period": "FY 2023-24 to FY 2025-26", "amount_display": "Rs. 3,49,12,908.33"},
        "ca_certificate": {"firm": "Test & Co", "frn": "000001S", "udin": "UDIN-1", "date": "2026-09-23"},
    },
    "manpower": {"headcount": 45, "as_of": "2026-09-16", "verified": False},
    "past_experience": [
        {"client": "Rural Labs Board", "short_name": "Labs", "work_order_no": "WO-LAB-9", "work_order_date": "2025-08-28",
         "value_display": "Rs. 1,56,00,000", "scope_items": ["Interactive online labs"], "tags": ["edtech"],
         "status": "Ongoing", "library_document": "Work Order - Labs"},
        {"client": "Farmer Welfare Dept", "short_name": "Campaign", "work_order_no": "WO-SM-1",
         "work_order_date": "2026-02-23", "value_display": "Rs. 3,02,50,000",
         "scope_items": ["Facebook, Instagram, YouTube and WhatsApp campaign execution"],
         "tags": ["social media", "campaign"], "status": "Ongoing", "library_document": "Work Order - Campaign"},
    ],
}


def test_draft_prompt_carries_the_full_company_data_and_tailoring_rules():
    response = {"documents": [{"id": "r1", "title": "Project Experience Summary", "body_paragraphs": ["x"],
                               "tables": [{"columns": ["S.No.", "Client"], "rows": [[1, "Farmer Welfare Dept"]]}],
                               "open_items": []}]}
    provider = FakeProvider(responses=[json.dumps(response)])
    drafted, errors = draft_checklist_documents(
        TENDER, [{"id": "r1", "document": "Project Experience Summary"}], RICH_PROFILE, [], provider=provider)

    assert errors == []
    table = drafted["r1"].tables[0]
    assert table.rows == [["1", "Farmer Welfare Dept"]]  # cells come back as text
    system_prompt, user_prompt = provider.calls[0]
    for fact in ("WO-SM-1", "23-02-2026", "Rs. 3,02,50,000", "Rs. 5,23,89,823", "Rs. 3,49,12,908.33",
                 "UDIN-1", "ISO/IEC 27001:2022", "Social media campaign execution", "UDYAM-TS-00-0000000",
                 "Old Name Private Limited", "45 as on 16-09-2026"):
        assert fact in user_prompt, fact
    assert "TAILOR EVERY DOCUMENT TO THIS TENDER" in system_prompt
    assert '"Work Order No. & Date"' in system_prompt  # the experience table's columns
    assert "FILL EVERY FIELD" in system_prompt


def test_projects_are_ranked_by_relevance_to_the_tender():
    from app.intelligence.prompts import company_data_lines, rank_past_experience

    ranked = rank_past_experience(RICH_PROFILE, TENDER)  # a social media tender
    assert [p["short_name"] for p in ranked] == ["Campaign", "Labs"]
    lines = "\n".join(company_data_lines(RICH_PROFILE, TENDER))
    assert lines.index("Campaign") < lines.index("Labs")


def test_checklist_plan_parses_notary_section_and_tender_order():
    plan_json = {"bid_number": None, "bid_end": None, "tender_order": True, "rows": [
        {"document": "Affidavit of non-blacklisting", "source": "draft", "section": "Declarations & Undertakings",
         "notary": True},
        {"document": "PAN", "source": "upload", "library_document": "PAN Card"},
    ]}
    provider = FakeProvider(responses=[json.dumps(plan_json)])
    plan = plan_submission_checklist(TENDER, ELIGIBILITY_CRITERIA, RICH_PROFILE, ["PAN Card"], provider)

    assert plan.tender_order is True
    assert (plan.rows[0].notary, plan.rows[0].section) == (True, "Declarations & Undertakings")
    assert (plan.rows[1].notary, plan.rows[1].section) == (False, "")
    system_prompt, user_prompt = provider.calls[0]
    assert "ONE ROW PER DISTINCT DOCUMENT" in system_prompt
    assert '"notary": true ONLY when' in system_prompt
    assert "library document: Work Order - Campaign" in user_prompt  # projects for per-project work order rows


def test_draft_prompt_gives_the_bid_date_personnel_and_tender_fact_rules():
    import datetime as dt

    profile = {**RICH_PROFILE, "key_personnel": [
        {"name": "Asha Rao", "role": "Social Media Manager", "qualification": "MBA", "experience_years": 7},
    ]}
    provider = FakeProvider(responses=[json.dumps({"documents": []})])
    draft_checklist_documents(TENDER, [{"id": "r1", "document": "List of Proposed Personnel"}], profile, [],
                              provider=provider)
    system_prompt, user_prompt = provider.calls[0]
    assert f"Bid date: {dt.date.today().strftime('%d-%m-%Y')}" in user_prompt
    assert "name: Asha Rao | role: Social Media Manager | qualification: MBA | experience (years): 7" in user_prompt
    for rule in ("TENDER FACTS COME FROM THE TENDER", "PRICES are never written",
                 "FIELDS THAT DON'T APPLY", "Never invent a person"):
        assert rule in system_prompt

    provider = FakeProvider(responses=[json.dumps({"documents": []})])
    draft_checklist_documents(TENDER, [{"id": "r1", "document": "CVs"}], RICH_PROFILE, [], provider=provider)
    assert "Key personnel: none on record" in provider.calls[0][1]
