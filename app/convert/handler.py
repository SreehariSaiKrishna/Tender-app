"""DocxConvertFunction's entry point (template.yaml): converts one Word
document in FilesBucket to PDF for POST /stamp-document/convert
(app.api.main). Invoked asynchronously with
{"bucket", "source_key", "filename", "result_key", "status_key"}; the outcome
is written to `status_key` as JSON - {"status": "done", "size": N} with the
PDF at `result_key`, or {"status": "failed", "message": ...} - which
GET /stamp-document/convert/{id} reads back."""
from __future__ import annotations

import json
import logging
from typing import Any

import boto3

from app.convert.converter import ConversionError, convert_to_pdf

logger = logging.getLogger()
logger.setLevel(logging.INFO)

_s3 = boto3.client("s3")


def handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    bucket = event["bucket"]

    def status(body: dict[str, Any]) -> dict[str, Any]:
        _s3.put_object(Bucket=bucket, Key=event["status_key"], Body=json.dumps(body).encode(),
                       ContentType="application/json")
        return body

    try:
        content = _s3.get_object(Bucket=bucket, Key=event["source_key"])["Body"].read()
        pdf = convert_to_pdf(content, event.get("filename") or "document.docx")
    except ConversionError as exc:
        return status({"status": "failed", "message": str(exc)})
    except Exception:  # noqa: BLE001
        logger.exception("Converting %s failed", event.get("source_key"))
        return status({"status": "failed", "message": "Converting the Word file failed unexpectedly - try again."})
    _s3.put_object(Bucket=bucket, Key=event["result_key"], Body=pdf, ContentType="application/pdf")
    return status({"status": "done", "size": len(pdf)})
