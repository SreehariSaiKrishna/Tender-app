"""Documents added by hand to a tender in Other Tenders (see app.api.main's
manual-documents endpoints): uploaded files, kept in file storage under
tender-documents/, and links to documents or pages on any site.

collect_manual_documents() is this tender's equivalent of
app.browser.document_collector's download: it puts every one of them on
local disk and records them in the tender's `documents`, exactly as a
TenderDetail download would - so app.intelligence.document_summarizer then
reads, summarises and stores their text the same way, and the submission
checklist is built from it. Needs no browser.
"""
from __future__ import annotations

import datetime as dt
import logging
import mimetypes
import re
import urllib.request
from email.message import Message
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

from pymongo.collection import Collection

from app import storage
from app.config import Settings, get_settings
from app.database import get_collection

logger = logging.getLogger(__name__)

TENDER_DOCUMENTS_PREFIX = "tender-documents/"
# No size limit on a link, as with uploads - only the pipeline function's
# memory bounds it. The timeout is per network read, not for the whole file.
LINK_TIMEOUT_SECONDS = 60
# Some portals refuse requests without a browser-like user agent.
_USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"

_EXTENSION_BY_TYPE = {
    "application/pdf": ".pdf",
    "text/html": ".html",
    "application/xhtml+xml": ".html",
    "text/plain": ".txt",
    "application/zip": ".zip",
    "application/x-zip-compressed": ".zip",
    "application/msword": ".doc",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
    "application/vnd.ms-excel": ".xls",
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet": ".xlsx",
}


def safe_filename(name: str | None, fallback: str = "document") -> str:
    name = re.sub(r"[\\/:*?\"<>|\x00-\x1f]+", "_", (name or "").strip()).strip(" .")
    return name[:150] or fallback


def tender_document_key(tender_id: object, document_id: str, filename: str) -> str:
    return f"{TENDER_DOCUMENTS_PREFIX}{tender_id}/{document_id}/{safe_filename(filename)}"


class LinkDownloadError(RuntimeError):
    pass


def _filename_for_link(url: str, headers: Message, index: int) -> str:
    """The server's own filename (Content-Disposition), else the URL's last
    path segment - with an extension added from the content type when it
    has none, since app.intelligence.document_summarizer picks its reader
    by extension."""
    name = headers.get_filename() or unquote(Path(urlparse(url).path).name)
    name = safe_filename(name, fallback=f"link-{index}")
    content_type = (headers.get_content_type() or "").lower()
    if not Path(name).suffix:
        name += _EXTENSION_BY_TYPE.get(content_type) or mimetypes.guess_extension(content_type) or ".html"
    return name


def download_link(url: str, index: int = 1) -> tuple[str, bytes]:
    if not re.match(r"https?://", url, re.IGNORECASE):
        raise LinkDownloadError("only http(s) links can be downloaded")
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=LINK_TIMEOUT_SECONDS) as response:  # noqa: S310 - http(s) only, checked above
            data = response.read()
            headers = response.headers
            final_url = response.geturl()
    except Exception as exc:  # noqa: BLE001 - reported per link
        raise LinkDownloadError(str(exc)) from exc
    if not data:
        raise LinkDownloadError("the link returned an empty file")
    return _filename_for_link(final_url, headers, index), data


def _unique(name: str, taken: set[str]) -> str:
    stem, suffix = Path(name).stem, Path(name).suffix
    candidate, n = name, 2
    while candidate.lower() in taken:
        candidate = f"{stem} ({n}){suffix}"
        n += 1
    taken.add(candidate.lower())
    return candidate


def collect_manual_documents(
    tender_ids: list[Any],
    settings: Settings | None = None,
    collection: Collection | None = None,
    merge: bool = False,
) -> dict[str, list[str]]:
    """Put each tender's hand-added documents on disk and record them as its
    `documents`. With `merge` (a scraped tender, its TenderDetail documents
    just downloaded) they're added to the files already on disk rather than
    replacing them. Returns {tender_id: [error, ...]} for the ones that had
    any trouble; a file or link that fails never loses the rest."""
    settings = settings or get_settings()
    collection = collection if collection is not None else get_collection()
    documents_dir = settings.resolved_path(settings.documents_dir)
    problems: dict[str, list[str]] = {}

    for tender in collection.find({"_id": {"$in": tender_ids}, "manual_documents.0": {"$exists": True}},
                                  {"manual_documents": 1, "tender_ref": 1, "documents": 1,
                                   "documents_download_error": 1}):
        dest_dir = documents_dir / f"manual_{tender['_id']}"
        dest_dir.mkdir(parents=True, exist_ok=True)
        saved: list[dict[str, Any]] = []
        errors: list[str] = []
        taken: set[str] = set()
        if merge:
            for entry in tender.get("documents") or []:
                path = Path(entry["local_path"]) if entry.get("local_path") else None
                if path and path.is_file() and path.parent != dest_dir:
                    saved.append(entry)
                    taken.add((entry.get("filename") or path.name).lower())
            if tender.get("documents_download_error"):
                errors.append(tender["documents_download_error"])
        already = len(saved)
        now = dt.datetime.now(dt.timezone.utc)

        for index, entry in enumerate(tender["manual_documents"], start=1):
            label = entry.get("filename") or entry.get("url") or "document"
            try:
                if entry.get("kind") == "link":
                    filename, data = download_link(entry["url"], index)
                else:
                    data = storage.get_bytes(entry["s3_key"])
                    if data is None:
                        raise LinkDownloadError("the uploaded file is missing from storage")
                    filename = entry["filename"]
            except Exception as exc:  # noqa: BLE001 - one document failing must not lose the rest
                logger.error("Could not get '%s' for tender %s: %s", label, tender["_id"], exc)
                errors.append(f"{label}: {exc}")
                continue
            filename = _unique(safe_filename(filename), taken)
            target = dest_dir / filename
            target.write_bytes(data)
            saved.append({
                "filename": filename,
                "description": entry.get("url") if entry.get("kind") == "link" else "Uploaded by hand",
                "local_path": str(target),
                "downloaded_at": now,
            })

        update: dict[str, Any] = {"documents_download_error": "; ".join(errors) or None}
        if len(saved) > already:
            update["documents"] = saved
            update["documents_downloaded_at"] = now
        collection.update_one({"_id": tender["_id"]}, {"$set": update})
        if errors:
            problems[str(tender["_id"])] = errors
    return problems
