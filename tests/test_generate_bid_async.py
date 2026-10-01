"""POST /tenders/{id}/generate-bid deployed: the build outlasts the API's
30s limit, so the request only starts it - ApiFunction invokes itself with
{"generate_bid": id} - and the dashboard polls GET /generate-bid. The build
itself (_build_bid_pack) is stubbed; a mongomock collection holds the tender."""
from __future__ import annotations

import datetime as dt
import json
import sys
import types

import mongomock
import pytest
from bson import ObjectId
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.api import main as api


@pytest.fixture()
def env(monkeypatch):
    tenders = mongomock.MongoClient().db["tenders"]
    tender_id = tenders.insert_one({"title": "T", "checklist": {"version": 1}}).inserted_id
    monkeypatch.setattr(api, "get_collection", lambda: tenders)
    monkeypatch.setattr(api.storage, "load_document_text", lambda t: t)
    monkeypatch.setattr(api, "is_submission_checklist", lambda c: True)
    monkeypatch.setenv("AWS_LAMBDA_FUNCTION_NAME", "tender-agent-ApiFunction-x")

    invocations = []
    fake_boto3 = types.SimpleNamespace(
        client=lambda name: types.SimpleNamespace(invoke=lambda **kw: invocations.append(kw))
    )
    monkeypatch.setitem(sys.modules, "boto3", fake_boto3)

    def build(object_id, tender):
        tenders.update_one({"_id": object_id}, {
            "$set": {"bid_pack_key": "bid-packs/x.pdf", "bid_generated_at": dt.datetime.now(dt.timezone.utc)},
            "$unset": {"bid_generation_started_at": "", "bid_generation_error": ""},
        })
        return {}

    monkeypatch.setattr(api, "_build_bid_pack", build)
    return types.SimpleNamespace(client=TestClient(api.app), tenders=tenders, id=str(tender_id),
                                 invocations=invocations, monkeypatch=monkeypatch)


def test_post_starts_build_in_background_and_polling_reports_done(env):
    res = env.client.post(f"/tenders/{env.id}/generate-bid")
    assert res.status_code == 202
    assert res.json()["status"] == "running"
    assert len(env.invocations) == 1
    assert env.invocations[0]["InvocationType"] == "Event"
    event = json.loads(env.invocations[0]["Payload"])
    assert event["generate_bid"] == env.id and event["run"]

    # A second click while it's running doesn't start another.
    assert env.client.post(f"/tenders/{env.id}/generate-bid").status_code == 202
    assert len(env.invocations) == 1

    assert api.handler(event, None) == {"ok": True}
    status = env.client.get(f"/tenders/{env.id}/generate-bid").json()
    assert status["status"] == "done"
    assert status["tender"]["title"] == "T"


def test_lambda_retry_of_a_run_does_nothing(env):
    builds = []
    env.monkeypatch.setattr(api, "_build_bid_pack", lambda o, t: builds.append(o))
    env.client.post(f"/tenders/{env.id}/generate-bid")
    event = json.loads(env.invocations[0]["Payload"])

    api.handler(event, None)
    api.handler(event, None)  # Lambda's automatic retry
    api.handler({"generate_bid": env.id}, None)  # no run id
    assert len(builds) == 1


def test_failed_build_is_reported_not_raised(env):
    def fail(object_id, tender):
        raise HTTPException(status_code=422, detail="No checklist rows to build from.")

    env.monkeypatch.setattr(api, "_build_bid_pack", fail)
    env.client.post(f"/tenders/{env.id}/generate-bid")
    assert api.handler(json.loads(env.invocations[0]["Payload"]), None) == {"ok": True}

    status = env.client.get(f"/tenders/{env.id}/generate-bid").json()
    assert status == {"status": "failed", "message": "No checklist rows to build from."}

    # Generating again clears the old failure.
    env.monkeypatch.setattr(api, "_build_bid_pack", lambda o, t: {})
    assert env.client.post(f"/tenders/{env.id}/generate-bid").json()["status"] == "running"


def test_stale_build_reads_as_failed_and_can_be_restarted(env):
    stale = dt.datetime.now(dt.timezone.utc) - api._BID_GENERATION_TIMEOUT - dt.timedelta(minutes=1)
    env.tenders.update_one({"_id": ObjectId(env.id)}, {"$set": {"bid_generation_started_at": stale}})
    assert env.client.get(f"/tenders/{env.id}/generate-bid").json()["status"] == "failed"

    assert env.client.post(f"/tenders/{env.id}/generate-bid").status_code == 202
    assert len(env.invocations) == 1
