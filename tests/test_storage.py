"""app.storage - a local folder stands in for the bucket only when allowed."""
from __future__ import annotations

import pytest

from app import storage
from app.config import get_settings


def test_round_trip_and_missing_keys():
    storage.put_text("tender-text/1.txt", "RFP")
    assert storage.get_text("tender-text/1.txt") == "RFP"
    storage.delete("tender-text/1.txt")
    assert storage.get_bytes("tender-text/1.txt") is None
    storage.delete("tender-text/1.txt")  # already gone - not an error


def test_keys_cannot_escape_the_storage_folder():
    with pytest.raises(ValueError):
        storage.put_bytes("../outside.txt", b"x")


def test_no_bucket_and_no_local_files_is_refused(monkeypatch, tmp_path):
    """A local run shares the production database - files written to this
    machine's disk would leave Mongo pointing at something production can't read."""
    settings = get_settings().model_copy(update={"files_bucket": "", "files_dir": str(tmp_path), "local_files": False})
    monkeypatch.setattr(storage, "get_settings", lambda: settings)
    with pytest.raises(storage.StorageNotConfiguredError, match="FILES_BUCKET"):
        storage.put_text("tender-text/1.txt", "RFP")
    with pytest.raises(storage.StorageNotConfiguredError):
        storage.get_bytes("company-documents/x/a.pdf")
    assert not any(tmp_path.iterdir())
