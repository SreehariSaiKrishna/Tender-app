"""MongoDB client/collection setup.

A single module-level `MongoClient` is created lazily and reused for the
life of the process - on a warm Lambda invocation this means the TCP/TLS
connection to Atlas is reused across runs instead of reconnecting every
time, the same benefit the old SQLAlchemy engine singleton gave locally.

There is no `session_scope()` equivalent here on purpose: every write in
this app is a single-document upsert, which MongoDB already makes atomic,
so there's no multi-document transaction to wrap.
"""
from __future__ import annotations

from pymongo import ASCENDING, DESCENDING, MongoClient
from pymongo.collection import Collection

from app.config import get_settings

_client: MongoClient | None = None


def get_client() -> MongoClient:
    global _client
    if _client is None:
        settings = get_settings()
        if not settings.mongodb_uri:
            raise RuntimeError(
                "MONGODB_URI is not set. Add it to .env before running anything "
                "that touches the database."
            )
        _client = MongoClient(settings.mongodb_uri)
    return _client


def get_collection() -> Collection:
    settings = get_settings()
    return get_client()[settings.mongodb_db_name]["tenders"]


def ensure_indexes() -> None:
    """Create the indexes the query helpers rely on, if they don't already
    exist. Safe to call every cold start - `create_index` is idempotent.
    """
    collection = get_collection()
    collection.create_index("dedup_key", unique=True)
    collection.create_index("tender_ref")
    collection.create_index("closing_date")
    collection.create_index("disappeared")
    collection.create_index("first_seen")
    collection.create_index("deadline_changed")
    collection.create_index("query_match_count")
    collection.create_index("query_matches.query_name")
    collection.create_index([("last_seen", DESCENDING)])
    collection.create_index([("status", ASCENDING)])
