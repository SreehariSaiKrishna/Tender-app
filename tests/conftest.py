"""Shared fixtures."""
from __future__ import annotations

import pytest

from app import storage
from app.config import get_settings


@pytest.fixture(autouse=True)
def local_storage(tmp_path, monkeypatch):
    """Every test's file storage (app.storage) is a fresh temporary folder -
    never S3, even when .env names a bucket, and never the repo's data/."""
    settings = get_settings().model_copy(update={"files_bucket": "", "files_dir": str(tmp_path / "files")})
    monkeypatch.setattr(storage, "get_settings", lambda: settings)
    return tmp_path / "files"
