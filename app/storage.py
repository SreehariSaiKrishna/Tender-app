"""File storage: company documents, each tender's document text and the
short-lived bid packs. S3 (settings.files_bucket, see template.yaml) when
deployed; a local folder (settings.files_dir) otherwise, so tests and a
local uvicorn/CLI run need no AWS access.

Keys used (see template.yaml for the lifecycle rules on each prefix):
  company-documents/<id>/<filename>  - the Documents library
  tender-text/<tender_id>.txt        - deleted along with its tender
  bid-packs/<tender_id>/<uuid>.pdf   - expire after a day, never kept

Metadata about each file lives in Mongo; this module only moves bytes.
"""
from __future__ import annotations

import logging
from pathlib import Path
from urllib.parse import quote

from app.config import get_settings

logger = logging.getLogger(__name__)

COMPANY_DOCUMENTS_PREFIX = "company-documents/"
TENDER_TEXT_PREFIX = "tender-text/"
BID_PACKS_PREFIX = "bid-packs/"


def tender_text_key(tender_id: object) -> str:
    return f"{TENDER_TEXT_PREFIX}{tender_id}.txt"


_s3_client = None


def _s3():
    global _s3_client
    if _s3_client is None:
        import boto3

        _s3_client = boto3.client("s3")
    return _s3_client


def _bucket() -> str:
    return get_settings().files_bucket


def _local_path(key: str) -> Path:
    root = get_settings().resolved_path(get_settings().files_dir).resolve()
    path = (root / key).resolve()
    if root not in path.parents:
        raise ValueError(f"Invalid storage key: {key!r}")
    return path


def put_bytes(key: str, data: bytes, content_type: str | None = None) -> None:
    if _bucket():
        extra = {"ContentType": content_type} if content_type else {}
        _s3().put_object(Bucket=_bucket(), Key=key, Body=data, **extra)
        return
    path = _local_path(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


def get_bytes(key: str) -> bytes | None:
    """The object's bytes, or None when there's no such key."""
    if _bucket():
        from botocore.exceptions import ClientError

        try:
            return _s3().get_object(Bucket=_bucket(), Key=key)["Body"].read()
        except ClientError as exc:
            # Access denied is re-raised, not read as "missing" - a
            # permissions mistake must not look like a deleted file.
            if exc.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
                return None
            raise
    path = _local_path(key)
    return path.read_bytes() if path.is_file() else None


def delete(key: str) -> None:
    """Deleting a key that isn't there is not an error."""
    if _bucket():
        _s3().delete_object(Bucket=_bucket(), Key=key)
        return
    _local_path(key).unlink(missing_ok=True)


def put_text(key: str, text: str) -> None:
    put_bytes(key, text.encode("utf-8"), "text/plain; charset=utf-8")


def get_text(key: str) -> str | None:
    data = get_bytes(key)
    return None if data is None else data.decode("utf-8", errors="replace")


def load_document_text(tender: dict) -> dict:
    """`tender` with its `document_text` read back from storage (see
    app.intelligence.document_summarizer), for the code that reads it -
    checklist, bid drafting, prompts. Unchanged when it has no stored text."""
    key = tender.get("document_text_key")
    if not key or tender.get("document_text"):
        return tender
    text = get_text(key)
    if text is None:
        logger.warning("Document text %s for tender %s is missing from storage", key, tender.get("_id"))
        return tender
    return {**tender, "document_text": text}


def download_url(key: str, filename: str, expires_in: int = 300) -> str | None:
    """A short-lived link the browser downloads `key` from directly (S3
    presigned GET, saved as `filename`), or None when running locally,
    where there's no bucket to link to."""
    if not _bucket():
        return None
    ascii_name = "".join(c if 32 <= ord(c) < 127 and c != '"' else "_" for c in filename) or "download"
    return _s3().generate_presigned_url(
        "get_object",
        Params={
            "Bucket": _bucket(),
            "Key": key,
            "ResponseContentDisposition": f"attachment; filename=\"{ascii_name}\"; filename*=UTF-8''{quote(filename)}",
        },
        ExpiresIn=expires_in,
    )
