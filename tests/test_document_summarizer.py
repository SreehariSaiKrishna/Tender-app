"""Tests for the document-summarization step (app.intelligence.document_summarizer).

Same style as test_scorer.py: a FakeProvider implementing the same
`generate()` interface as OpenAIProvider, and mongomock - no OpenAI API key,
network access, or real MongoDB required.
"""
from __future__ import annotations

import datetime as dt
import json

import mongomock
import pytest

from app.intelligence.document_summarizer import (
    DocumentSummarizationError,
    DocumentSummary,
    extract_text,
    summarize_pending_documents,
    summarize_tender_documents,
)
from app.intelligence.prompts import build_document_system_prompt, build_document_user_prompt

VALID_SUMMARY = {
    "estimated_bid_amount": "INR 45,00,000 (estimated)",
    "emd_amount": "INR 90,000",
    "documents_to_submit": ["GST certificate", "PAN card"],
    "key_dates": ["Bid submission: 28/09/2026"],
    "eligibility_requirements": ["Minimum 3 years experience"],
    "eligibility_technical_criteria": ["Minimum 3 similar completed projects in last 5 years"],
    "technical_criteria_table": [
        {"ref": "A", "criterion": "Input Flexibility", "expected_evidence": "Ability to generate video from text", "marks": "4"},
    ],
    "summary_text": "A tender for digital signage installation and maintenance.",
    "missing_information": [],
}


class FakeProvider:
    def __init__(self, response=None, responses=None, raise_on_call=False):
        self.response = response
        self.responses = responses or []
        self.raise_on_call = raise_on_call
        self.calls = []

    def generate(self, system_prompt: str, user_prompt: str) -> str:
        self.calls.append((system_prompt, user_prompt))
        if self.raise_on_call:
            raise RuntimeError("simulated network failure")
        if self.responses:
            return self.responses[len(self.calls) - 1]
        return json.dumps(self.response if self.response is not None else VALID_SUMMARY)


@pytest.fixture()
def collection():
    client = mongomock.MongoClient()
    return client["test_tender_intelligence"]["tenders"]


def make_tender(collection, dedup_key="ref:1", **overrides) -> dict:
    now = dt.datetime.now(dt.timezone.utc)
    doc = {
        "dedup_key": dedup_key,
        "tender_ref": dedup_key.split(":")[-1],
        "title": "Tender for Digital Signage",
        "organisation": "Some Department",
        "disappeared": False,
        "first_seen": now,
        "last_seen": now,
    }
    doc.update(overrides)
    collection.insert_one(doc)
    return doc


# --- prompts -----------------------------------------------------


def test_document_system_prompt_mentions_json_shape():
    prompt = build_document_system_prompt()
    assert "estimated_bid_amount" in prompt
    assert "JSON" in prompt


def test_document_user_prompt_includes_listing_and_file_text():
    prompt = build_document_user_prompt(
        {"title": "Some Tender", "tender_ref": "TDR-1"}, {"notice.html": "EMD is INR 50,000"}
    )
    assert "Some Tender" in prompt
    assert "notice.html" in prompt
    assert "EMD is INR 50,000" in prompt


def test_document_user_prompt_truncates_long_files():
    long_text = "x" * 100
    prompt = build_document_user_prompt({"title": "T"}, {"big.pdf": long_text}, max_chars=10)
    assert "x" * 10 in prompt
    assert "truncated" in prompt


# --- extract_text -----------------------------------------------------


def test_extract_text_html(tmp_path):
    path = tmp_path / "notice.html"
    path.write_text("<html><body><h1>Tender Notice</h1><p>EMD: INR 50,000</p></body></html>", encoding="utf-8")
    text = extract_text(path)
    assert "Tender Notice" in text
    assert "EMD: INR 50,000" in text


def test_extract_text_excel(tmp_path):
    pd = pytest.importorskip("pandas")
    path = tmp_path / "boq.xlsx"
    pd.DataFrame({"Item": ["Cement", "Steel"], "Qty": [100, 50]}).to_excel(path, index=False, header=False)
    text = extract_text(path)
    assert "Cement" in text
    assert "Steel" in text


def test_extract_text_plain_fallback(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("Plain text notes", encoding="utf-8")
    assert extract_text(path) == "Plain text notes"


def test_extract_text_skips_unsupported_binary_formats(tmp_path):
    """Regression: an unrecognized binary attachment (legacy .doc, .exe,
    etc.) used to fall through to the generic text-decode fallback and
    silently produce garbled binary "text" fed straight to the AI provider
    - it must now be treated as not-extractable instead. (.rar/.zip/.docx
    ARE supported - see the archive/docx tests below.)
    """
    path = tmp_path / "legacy.doc"
    path.write_bytes(b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1binary garbage")
    assert extract_text(path) == ""


def test_extract_text_docx(tmp_path):
    docx = pytest.importorskip("docx")
    path = tmp_path / "notice.docx"
    document = docx.Document()
    document.add_paragraph("Tender eligibility: minimum 3 years experience")
    document.save(str(path))
    assert "Tender eligibility: minimum 3 years experience" in extract_text(path)


def test_extract_text_zip_recurses_into_members(tmp_path):
    import zipfile

    html_path = tmp_path / "notice.html"
    html_path.write_text("<p>EMD is INR 25,000</p>", encoding="utf-8")
    zip_path = tmp_path / "bundle.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        zf.write(html_path, arcname="notice.html")

    text = extract_text(zip_path)
    assert "EMD is INR 25,000" in text
    assert "notice.html" in text  # archived filename is included as context


def test_extract_text_rar_without_unrar_tool_returns_empty(monkeypatch, tmp_path):
    """No UnRAR binary available must degrade to "not extractable", not a
    crash - mirrors _tesseract_cmd's contract for missing OCR engines."""
    import app.intelligence.document_summarizer as doc_sum

    monkeypatch.setattr(doc_sum, "_unrar_tool", lambda: None)
    path = tmp_path / "archive.rar"
    path.write_bytes(b"Rar!\x1a\x07\x01\x00fake")
    assert extract_text(path) == ""


# --- OCR fallback for scanned/image-only PDFs -----------------------------------------------------


def _make_blank_pdf(path):
    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=200, height=200)
    with open(path, "wb") as f:
        writer.write(f)


def test_extract_pdf_text_falls_back_to_ocr_for_pages_with_no_text_layer(monkeypatch, tmp_path):
    """A blank/scanned page (no embedded text at all) must trigger OCR
    rather than being reported as empty - confirmed live against two real
    scanned tender notices (2026-09-16), both recovered this way.
    """
    import app.intelligence.document_summarizer as doc_sum

    path = tmp_path / "scanned.pdf"
    _make_blank_pdf(path)

    monkeypatch.setattr(doc_sum, "_ocr_pdf_page", lambda pdf_path, page_number, dpi=200: "OCR extracted text")

    text = doc_sum._extract_pdf_text(path)
    assert "OCR extracted text" in text


def test_extract_pdf_text_skips_ocr_when_real_text_present(monkeypatch, tmp_path):
    """A page with a normal text layer must not trigger OCR at all -
    OCR is strictly a fallback for pages pypdf couldn't read anything from.
    """
    import app.intelligence.document_summarizer as doc_sum

    path = tmp_path / "scanned.pdf"
    _make_blank_pdf(path)

    class FakePage:
        def extract_text(self):
            return "Plenty of real text here " * 5

    class FakeReader:
        def __init__(self, _path):
            self.pages = [FakePage()]

    calls = []
    monkeypatch.setattr(doc_sum, "_ocr_pdf_page", lambda *a, **k: calls.append(1))
    monkeypatch.setattr("pypdf.PdfReader", FakeReader)

    doc_sum._extract_pdf_text(path)
    assert calls == []


def test_extract_pdf_text_ocr_failure_does_not_crash(monkeypatch, tmp_path):
    """OCR itself failing (engine missing, corrupt page, etc.) must not
    propagate - the page just stays empty, same as any other single-file
    failure in this module."""
    import app.intelligence.document_summarizer as doc_sum

    path = tmp_path / "scanned.pdf"
    _make_blank_pdf(path)

    def raise_ocr_error(pdf_path, page_number, dpi=200):
        raise RuntimeError("tesseract not found")

    monkeypatch.setattr(doc_sum, "_ocr_pdf_page", raise_ocr_error)

    text = doc_sum._extract_pdf_text(path)  # must not raise
    assert text.strip() == ""


def test_tesseract_cmd_prefers_env_override(monkeypatch):
    import app.intelligence.document_summarizer as doc_sum

    monkeypatch.setenv("TESSERACT_CMD", r"C:\custom\tesseract.exe")
    assert doc_sum._tesseract_cmd() == r"C:\custom\tesseract.exe"


def test_tesseract_cmd_falls_back_to_path(monkeypatch):
    import app.intelligence.document_summarizer as doc_sum

    monkeypatch.delenv("TESSERACT_CMD", raising=False)
    monkeypatch.setattr(doc_sum.shutil, "which", lambda name: "/usr/bin/tesseract")
    assert doc_sum._tesseract_cmd() == "/usr/bin/tesseract"


def test_tesseract_cmd_returns_none_when_not_found(monkeypatch):
    import app.intelligence.document_summarizer as doc_sum

    monkeypatch.delenv("TESSERACT_CMD", raising=False)
    monkeypatch.setattr(doc_sum.shutil, "which", lambda name: None)
    monkeypatch.setattr(doc_sum.os.path, "exists", lambda p: False)
    assert doc_sum._tesseract_cmd() is None


def test_extract_text_never_raises_on_corrupt_file(tmp_path):
    path = tmp_path / "broken.pdf"
    path.write_bytes(b"not actually a pdf")
    assert extract_text(path) == ""


# --- summarize_tender_documents / response validation -----------------------------------------------------


def test_summarize_tender_documents_returns_valid_summary():
    provider = FakeProvider(response=VALID_SUMMARY)
    result = summarize_tender_documents({"title": "T"}, {"notice.html": "some text"}, provider)
    assert isinstance(result, DocumentSummary)
    assert result.estimated_bid_amount == "INR 45,00,000 (estimated)"
    assert result.eligibility_technical_criteria == ["Minimum 3 similar completed projects in last 5 years"]
    assert len(result.technical_criteria_table) == 1
    assert result.technical_criteria_table[0].ref == "A"
    assert result.technical_criteria_table[0].marks == "4"
    assert len(provider.calls) == 1


def test_summarize_tender_documents_defaults_technical_criteria_to_empty_list():
    """Older provider responses (or ones where the text draws no technical/
    general distinction, or no scoring table exists) omit these fields
    entirely - they must default to [] rather than fail validation, same
    contract as the other list fields."""
    without_technical_criteria = {
        k: v
        for k, v in VALID_SUMMARY.items()
        if k not in ("eligibility_technical_criteria", "technical_criteria_table")
    }
    provider = FakeProvider(response=without_technical_criteria)
    result = summarize_tender_documents({"title": "T"}, {"notice.html": "some text"}, provider)
    assert result.eligibility_technical_criteria == []
    assert result.technical_criteria_table == []


def test_summarize_tender_documents_accepts_numeric_marks():
    """The prompt asks for "marks" as a string, but providers sometimes
    return a bare number for a numeric column - must not fail validation."""
    with_numeric_marks = {
        **VALID_SUMMARY,
        "technical_criteria_table": [{"ref": "B", "criterion": "Resolution", "expected_evidence": None, "marks": 4}],
    }
    provider = FakeProvider(response=with_numeric_marks)
    result = summarize_tender_documents({"title": "T"}, {"notice.html": "some text"}, provider)
    assert result.technical_criteria_table[0].marks == 4


def test_summarize_tender_documents_rejects_malformed_json():
    provider = FakeProvider(responses=["this is not json"])
    with pytest.raises(DocumentSummarizationError):
        summarize_tender_documents({"title": "T"}, {"notice.html": "x"}, provider)


def test_summarize_tender_documents_rejects_missing_required_field():
    incomplete = {k: v for k, v in VALID_SUMMARY.items() if k != "summary_text"}
    provider = FakeProvider(response=incomplete)
    with pytest.raises(DocumentSummarizationError):
        summarize_tender_documents({"title": "T"}, {"notice.html": "x"}, provider)


def test_summarize_tender_documents_wraps_provider_exceptions():
    provider = FakeProvider(raise_on_call=True)
    with pytest.raises(DocumentSummarizationError):
        summarize_tender_documents({"title": "T"}, {"notice.html": "x"}, provider)


# --- summarize_pending_documents (DB persistence + batch behaviour) -----------------------------------------------------


def test_summarize_pending_documents_persists_result(collection, tmp_path):
    doc_path = tmp_path / "notice.txt"
    doc_path.write_text("Tender notice text", encoding="utf-8")
    make_tender(
        collection,
        documents=[{"filename": "notice.txt", "local_path": str(doc_path)}],
        documents_downloaded_at=dt.datetime.now(dt.timezone.utc),
    )
    provider = FakeProvider(response=VALID_SUMMARY)

    summary = summarize_pending_documents(provider=provider, collection=collection)

    assert summary.summarized == 1
    assert summary.failed == 0
    stored = collection.find_one({"dedup_key": "ref:1"})
    assert stored["document_summary"]["estimated_bid_amount"] == "INR 45,00,000 (estimated)"
    assert stored["document_summary_generated_at"] is not None


def test_summarize_pending_documents_deletes_local_file_after_success(collection, tmp_path):
    """The raw file's only job was feeding the AI summary - once that
    succeeds, it must be deleted (and its now-empty tender directory too),
    with `local_path` cleared in Mongo rather than left dangling."""
    tender_dir = tmp_path / "57392947"
    tender_dir.mkdir()
    doc_path = tender_dir / "notice.txt"
    doc_path.write_text("Tender notice text", encoding="utf-8")
    make_tender(
        collection,
        documents=[{"filename": "notice.txt", "description": "Tender Documents", "local_path": str(doc_path)}],
        documents_downloaded_at=dt.datetime.now(dt.timezone.utc),
    )
    provider = FakeProvider(response=VALID_SUMMARY)

    summary = summarize_pending_documents(provider=provider, collection=collection)

    assert summary.summarized == 1
    assert not doc_path.exists()
    assert not tender_dir.exists()  # emptied directory is removed too

    stored = collection.find_one({"dedup_key": "ref:1"})
    assert stored["documents"][0]["local_path"] is None
    assert stored["documents"][0]["filename"] == "notice.txt"  # metadata is kept


def test_summarize_pending_documents_keeps_local_file_when_summary_fails(collection, tmp_path):
    doc_path = tmp_path / "notice.txt"
    doc_path.write_text("Tender notice text", encoding="utf-8")
    make_tender(
        collection,
        documents=[{"filename": "notice.txt", "local_path": str(doc_path)}],
        documents_downloaded_at=dt.datetime.now(dt.timezone.utc),
    )
    provider = FakeProvider(responses=["not json"])

    summary = summarize_pending_documents(provider=provider, collection=collection)

    assert summary.failed == 1
    assert doc_path.exists()  # left in place so a retry has something to read


def test_summarize_pending_documents_skips_tenders_without_documents(collection):
    make_tender(collection)  # no `documents` field at all
    provider = FakeProvider(response=VALID_SUMMARY)

    summary = summarize_pending_documents(provider=provider, collection=collection)

    assert summary.summarized == 0
    assert summary.failed == 0


def test_summarize_pending_documents_skips_already_summarized_unchanged_tenders(collection, tmp_path):
    doc_path = tmp_path / "notice.txt"
    doc_path.write_text("Tender notice text", encoding="utf-8")
    make_tender(
        collection,
        documents=[{"filename": "notice.txt", "local_path": str(doc_path)}],
        documents_downloaded_at=dt.datetime.now(dt.timezone.utc),
    )
    provider = FakeProvider(response=VALID_SUMMARY)

    first = summarize_pending_documents(provider=provider, collection=collection)
    assert first.summarized == 1

    second = summarize_pending_documents(provider=provider, collection=collection)
    assert second.summarized == 0  # nothing changed since the last summary


def test_summarize_pending_documents_resummarizes_after_redownload(collection, tmp_path):
    """Regression: after a successful summary the local file is deleted
    (see the deletion test above), so re-summarizing the SAME download can
    never happen again - re-summarizing only makes sense once
    app.browser.document_collector has actually re-downloaded fresh files
    (a new local_path) following a real content change. This models that
    real sequence rather than just bumping the timestamp in place.
    """
    # Each version's file lives in its own subdirectory, like
    # document_collector's real per-tender download directory - not
    # directly in tmp_path, whose root pytest itself still owns.
    v1_dir = tmp_path / "v1"
    v1_dir.mkdir()
    doc_path = v1_dir / "notice.txt"
    doc_path.write_text("Tender notice text", encoding="utf-8")
    make_tender(
        collection,
        documents=[{"filename": "notice.txt", "local_path": str(doc_path)}],
        documents_downloaded_at=dt.datetime.now(dt.timezone.utc),
    )
    provider = FakeProvider(response=VALID_SUMMARY)
    summarize_pending_documents(provider=provider, collection=collection)

    v2_dir = tmp_path / "v2"
    v2_dir.mkdir()
    new_doc_path = v2_dir / "notice_v2.txt"
    new_doc_path.write_text("Updated tender notice text", encoding="utf-8")
    collection.update_one(
        {"dedup_key": "ref:1"},
        {
            "$set": {
                "documents": [{"filename": "notice_v2.txt", "local_path": str(new_doc_path)}],
                "documents_downloaded_at": dt.datetime.now(dt.timezone.utc) + dt.timedelta(days=1),
            }
        },
    )

    summary = summarize_pending_documents(provider=provider, collection=collection)
    assert summary.summarized == 1


def test_summarize_pending_documents_skips_tenders_with_no_extractable_text(collection, tmp_path):
    missing_path = tmp_path / "does_not_exist.pdf"
    make_tender(
        collection,
        documents=[{"filename": "does_not_exist.pdf", "local_path": str(missing_path)}],
        documents_downloaded_at=dt.datetime.now(dt.timezone.utc),
    )
    provider = FakeProvider(response=VALID_SUMMARY)

    summary = summarize_pending_documents(provider=provider, collection=collection)

    assert summary.summarized == 0
    assert summary.failed == 1
    assert len(provider.calls) == 0


def test_summarize_pending_documents_continues_past_a_single_failure(collection, tmp_path):
    doc_path_1 = tmp_path / "notice1.txt"
    doc_path_1.write_text("Tender 1 text", encoding="utf-8")
    doc_path_2 = tmp_path / "notice2.txt"
    doc_path_2.write_text("Tender 2 text", encoding="utf-8")
    make_tender(
        collection,
        dedup_key="ref:1",
        documents=[{"filename": "notice1.txt", "local_path": str(doc_path_1)}],
        documents_downloaded_at=dt.datetime.now(dt.timezone.utc),
    )
    make_tender(
        collection,
        dedup_key="ref:2",
        documents=[{"filename": "notice2.txt", "local_path": str(doc_path_2)}],
        documents_downloaded_at=dt.datetime.now(dt.timezone.utc),
    )
    # First call fails, second succeeds - batch must not abort on the first.
    provider = FakeProvider(responses=["not json", json.dumps(VALID_SUMMARY)])

    summary = summarize_pending_documents(provider=provider, collection=collection)

    assert summary.failed == 1
    assert summary.summarized == 1


def test_summarize_pending_documents_widens_domain_match_from_document_text(monkeypatch, collection, tmp_path):
    """A vague listing (no domain keyword in title/description) can still
    become a match once its own RFP spells out the scope of work - see
    app.intelligence.document_summarizer's document_domain_text widening.
    """
    import app.intelligence.document_summarizer as doc_sum

    monkeypatch.setattr(doc_sum, "load_match_keywords", lambda: ["digital marketing"])

    doc_path = tmp_path / "notice.txt"
    doc_path.write_text("Tender notice text", encoding="utf-8")
    make_tender(
        collection,
        title="Empanelment of Vendors",  # no keyword here
        documents=[{"filename": "notice.txt", "local_path": str(doc_path)}],
        documents_downloaded_at=dt.datetime.now(dt.timezone.utc),
        domain_match=False,
        tender_value=1_00_00_000,
        earnest_money=50_000,
        document_fees=5_000,
    )
    response = {
        **VALID_SUMMARY,
        "eligibility_requirements": ["Prior experience in digital marketing campaigns"],
    }
    provider = FakeProvider(response=response)

    summary = summarize_pending_documents(provider=provider, collection=collection)

    assert summary.summarized == 1
    stored = collection.find_one({"dedup_key": "ref:1"})
    assert stored["domain_match"] is True
    assert stored["eligibility_match"] is True


def test_summarize_pending_documents_never_unsets_existing_domain_match(monkeypatch, collection, tmp_path):
    """The document-text widening only ever ORs in a new match - it must
    never flip an already-true domain_match (earned from the listing) back
    to False just because this particular document doesn't repeat it."""
    import app.intelligence.document_summarizer as doc_sum

    monkeypatch.setattr(doc_sum, "load_match_keywords", lambda: ["digital marketing"])

    doc_path = tmp_path / "notice.txt"
    doc_path.write_text("Tender notice text", encoding="utf-8")
    make_tender(
        collection,
        documents=[{"filename": "notice.txt", "local_path": str(doc_path)}],
        documents_downloaded_at=dt.datetime.now(dt.timezone.utc),
        domain_match=True,
        tender_value=1_00_00_000,
        earnest_money=50_000,
        document_fees=5_000,
    )
    response = {**VALID_SUMMARY, "eligibility_requirements": ["Minimum 3 years experience"]}
    provider = FakeProvider(response=response)

    summarize_pending_documents(provider=provider, collection=collection)

    stored = collection.find_one({"dedup_key": "ref:1"})
    assert stored["domain_match"] is True
    assert stored["eligibility_match"] is True


def test_summarize_pending_documents_respects_limit(collection, tmp_path):
    for i in range(3):
        doc_path = tmp_path / f"notice{i}.txt"
        doc_path.write_text(f"Tender {i} text", encoding="utf-8")
        make_tender(
            collection,
            dedup_key=f"ref:{i}",
            documents=[{"filename": f"notice{i}.txt", "local_path": str(doc_path)}],
            documents_downloaded_at=dt.datetime.now(dt.timezone.utc),
        )
    provider = FakeProvider(response=VALID_SUMMARY)

    summary = summarize_pending_documents(provider=provider, limit=2, collection=collection)
    assert summary.summarized == 2
