"""The Documents tab's Text sections (app.text_sections) - stored through
/text-sections and handed to the bid drafter as part of the company data."""
from __future__ import annotations

import datetime as dt

import mongomock
import pytest
from fastapi.testclient import TestClient

from app.api import main as api
from app.intelligence.prompts import company_data_lines
from app.text_sections import default_sections, with_text_sections

PROFILE = {
    "legal_name": "EXAMPLE PVT LTD",
    "email": "info@example.com",
    "manpower": {"headcount": 12, "as_of": "2026-01-31"},
    "authorized_signatory": {"name": "A Person", "designation": "Director"},
    "to_be_verified": ["TAN"],
}


@pytest.fixture()
def client(monkeypatch):
    collection = mongomock.MongoClient().db["text_sections"]
    monkeypatch.setattr(api, "get_text_sections_collection", lambda: collection)
    monkeypatch.setattr(api, "load_company_profile", lambda: PROFILE)
    return TestClient(api.app)


def test_defaults_are_added_once_from_the_profile(client):
    sections = client.get("/text-sections").json()["sections"]
    assert [s["header"] for s in sections] == [s["header"] for s in default_sections(PROFILE)]
    manpower = sections[0]
    assert manpower["text"] == "12 employees" and manpower["as_on"] == "2026-01-31"
    assert next(s for s in sections if s["header"] == "Bank Details")["text"] == ""

    for s in sections:
        client.delete(f"/text-sections/{s['id']}")
    assert client.get("/text-sections").json()["sections"] == []  # not re-added


def test_add_edit_and_delete_a_section(client):
    created = client.post("/text-sections", json={
        "header": " Bank Details 2 ", "text": "A/c 123\n", "as_on": "2026-10-01", "use_in": ["financial", "bogus"],
    }).json()
    assert created["header"] == "Bank Details 2" and created["text"] == "A/c 123"
    assert created["use_in"] == ["financial"]
    assert client.get("/text-sections").json()["sections"][-1]["id"] == created["id"]

    updated = client.put(f"/text-sections/{created['id']}", json={
        "header": "Bank Details 2", "text": "A/c 456", "valid_until": "", "use_in": [],
    }).json()
    assert updated["text"] == "A/c 456" and updated["use_in"] == ["all"] and updated["as_on"] is None

    assert client.delete(f"/text-sections/{created['id']}").status_code == 200
    assert client.delete(f"/text-sections/{created['id']}").status_code == 404


def test_bad_input_is_refused(client):
    assert client.post("/text-sections", json={"header": "  "}).status_code == 422
    assert client.post("/text-sections", json={"header": "X", "as_on": "31-01-2026"}).status_code == 422
    assert client.put("/text-sections/not-an-id", json={"header": "X"}).status_code == 404


def test_sections_reach_the_drafting_prompt():
    sections = [
        {"header": "Manpower", "text": "15 employees", "as_on": "2026-09-30", "valid_until": None,
         "use_in": ["company_profile"]},
        {"header": "ISO", "text": "ISO 9001", "as_on": None, "valid_until": "2026-01-01", "use_in": ["all"]},
        {"header": "Empty", "text": "  ", "as_on": None, "valid_until": None, "use_in": ["all"]},
    ]
    profile = with_text_sections(PROFILE, sections, today=dt.date(2026, 10, 6))
    assert [s["header"] for s in profile["text_sections"]] == ["Manpower", "ISO"]
    assert profile["to_be_verified"][0] == "TAN"
    assert "ISO" in profile["to_be_verified"][1] and "expired on 01-01-2026" in profile["to_be_verified"][1]
    assert PROFILE["to_be_verified"] == ["TAN"]  # the loaded profile isn't changed

    text = "\n".join(company_data_lines(profile, {}))
    assert "* Manpower [use in: Company profile; as on 30-09-2026]:\n          15 employees" in text
    assert "valid until 01-01-2026 - EXPIRED" in text
    assert "Empty" not in text
    assert "Manpower [" not in "\n".join(company_data_lines(profile, {}, detail=False))
