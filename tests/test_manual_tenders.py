"""Other Tenders - tenders added by hand (POST /tenders/manual) get their
own row on the dashboard's query table, are never closed by a pipeline pass,
and can be taken back out (DELETE /tenders/{id}/manual)."""
from __future__ import annotations

import mongomock
import pytest
from fastapi.testclient import TestClient

from app.api import main as api
from app.processing import deduplicator, eligibility


@pytest.fixture()
def tenders(monkeypatch):
    collection = mongomock.MongoClient().db["tenders"]
    monkeypatch.setattr(api, "get_collection", lambda: collection)
    monkeypatch.setattr(api, "load_saved_queries", lambda: [])
    monkeypatch.setattr(eligibility, "load_match_keywords", lambda: ["digital marketing"])
    monkeypatch.setattr(deduplicator, "load_match_keywords", lambda: ["digital marketing"])
    return collection


def test_added_tender_is_listed_under_other_tenders(tenders):
    client = TestClient(api.app)
    res = client.post("/tenders/manual", json={
        "title": "Digital marketing for the state tourism board",
        "organisation": "Tourism Dept",
        "tender_ref": "TB/2026/14",
        "closing_date": "2026-11-30",
        "source_url": "https://example.gov.in/tender/14",
    })
    assert res.status_code == 201
    tender = res.json()
    assert tender["manually_added"] is True
    assert tender["domain_match"] is True
    assert tender["closing_date"].startswith("2026-11-30")

    listed = client.get("/tenders", params={"query_name": api.MANUAL_QUERY_NAME}).json()
    assert [t["id"] for t in listed["tenders"]] == [tender["id"]]


def test_other_tenders_row_shows_even_when_empty(tenders):
    rows = TestClient(api.app).get("/stats/by-query").json()["queries"]
    assert [(r["query_name"], r["total"]) for r in rows] == [("Other Tenders", 0)]


def test_title_is_required_and_link_must_be_http(tenders):
    client = TestClient(api.app)
    assert client.post("/tenders/manual", json={"title": "  "}).status_code == 422
    assert client.post("/tenders/manual", json={"title": "X", "source_url": "javascript:alert(1)"}).status_code == 422


def test_already_scraped_tender_is_added_to_the_bucket_not_duplicated(tenders):
    tenders.insert_one({
        "dedup_key": "ref:57708084", "tender_ref": "57708084", "title": "Scraped", "disappeared": False,
        "query_matches": [{"query_name": "Digital Marketing"}], "query_match_count": 1,
    })
    client = TestClient(api.app)
    res = client.post("/tenders/manual", json={"title": "Scraped", "tender_ref": "57708084"})
    assert res.status_code == 201
    assert tenders.count_documents({}) == 1
    assert [m["query_name"] for m in res.json()["query_matches"]] == ["Digital Marketing", "Other Tenders"]
    assert client.post("/tenders/manual", json={"title": "Scraped", "tender_ref": "57708084"}).status_code == 409

    # Removing it only takes it out of Other Tenders - scraped tenders are kept.
    assert client.delete(f"/tenders/{res.json()['id']}/manual").json() == {"deleted": False}
    assert [m["query_name"] for m in tenders.find_one()["query_matches"]] == ["Digital Marketing"]


def test_hand_added_tender_is_deleted_on_removal(tenders):
    client = TestClient(api.app)
    tender_id = client.post("/tenders/manual", json={"title": "One-off"}).json()["id"]
    assert client.delete(f"/tenders/{tender_id}/manual").json() == {"deleted": True}
    assert tenders.count_documents({}) == 0


def test_remove_twice_is_not_an_error(tenders):
    client = TestClient(api.app)
    tender_id = client.post("/tenders/manual", json={"title": "One-off"}).json()["id"]
    assert client.delete(f"/tenders/{tender_id}/manual").status_code == 200
    assert client.delete(f"/tenders/{tender_id}/manual").json() == {"deleted": True}


def test_edit_a_hand_added_tender(tenders):
    client = TestClient(api.app)
    tender_id = client.post("/tenders/manual", json={"title": "Old", "tender_ref": "A1"}).json()["id"]
    res = client.put(f"/tenders/{tender_id}", json={
        "title": "Digital marketing campaign", "tender_ref": "A2", "closing_date": "2026-12-01", "earnest_money": 5000,
    })
    assert res.status_code == 200
    edited = res.json()
    assert (edited["title"], edited["tender_ref"], edited["earnest_money"]) == ("Digital marketing campaign", "A2", 5000)
    assert edited["closing_date"].startswith("2026-12-01")
    assert edited["domain_match"] is True
    stored = tenders.find_one()
    assert stored["dedup_key"] == "ref:a2" and "edited_fields" not in stored


def test_edits_to_a_scraped_tender_survive_the_next_scrape(tenders):
    from app.processing.normalizer import NormalizedTender

    def listing(closing):
        return NormalizedTender(tender_ref="T9", ref_is_synthetic=False, dedup_key="ref:t9", query_name="Digital Marketing",
                                title="Listing title", closing_date=closing, raw_data={})

    import datetime as dt
    deduplicator.ingest_batch({"Digital Marketing": [listing(dt.date(2026, 11, 1))]}, collection=tenders)
    tender_id = str(tenders.find_one()["_id"])
    client = TestClient(api.app)
    assert client.put(f"/tenders/{tender_id}", json={"title": "My title", "closing_date": "2026-11-20"}).status_code == 200

    deduplicator.ingest_batch({"Digital Marketing": [listing(dt.date(2026, 11, 1))]}, collection=tenders)
    stored = tenders.find_one()
    assert stored["title"] == "My title"
    assert stored["closing_date"].date() == dt.date(2026, 11, 20)
    assert stored["deadline_changed"] is False


def test_delete_hides_a_scraped_tender_and_deletes_a_hand_added_one(tenders):
    client = TestClient(api.app)
    scraped_id = str(tenders.insert_one({
        "dedup_key": "ref:1", "title": "Scraped", "disappeared": False,
        "query_matches": [{"query_name": "Digital Marketing"}], "query_match_count": 1,
    }).inserted_id)
    hand_id = client.post("/tenders/manual", json={"title": "One-off"}).json()["id"]

    assert client.delete(f"/tenders/{scraped_id}").json() == {"deleted": True}
    assert client.delete(f"/tenders/{hand_id}").json() == {"deleted": True}
    assert [d["title"] for d in tenders.find()] == ["Scraped"]
    assert tenders.find_one()["hidden"] is True
    assert client.get("/tenders", params={"query_name": "Digital Marketing"}).json()["total"] == 0


def test_add_a_scraped_tender_to_other_tenders(tenders):
    client = TestClient(api.app)
    tender_id = str(tenders.insert_one({
        "dedup_key": "ref:1", "title": "Scraped", "disappeared": False,
        "query_matches": [{"query_name": "Digital Marketing"}], "query_match_count": 1,
    }).inserted_id)
    res = client.post(f"/tenders/{tender_id}/other-tenders")
    assert [m["query_name"] for m in res.json()["query_matches"]] == ["Digital Marketing", "Other Tenders"]
    assert client.post(f"/tenders/{tender_id}/other-tenders").status_code == 409


def test_pipeline_pass_never_closes_a_manual_tender(tenders):
    TestClient(api.app).post("/tenders/manual", json={"title": "One-off"})
    deduplicator.ingest_batch({"Digital Marketing": []}, collection=tenders)
    assert tenders.find_one()["disappeared"] is False
